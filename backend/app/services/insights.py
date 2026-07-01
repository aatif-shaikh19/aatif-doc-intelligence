import json
import logging
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any, TYPE_CHECKING

from app.models.registry import registry
from app.services.retrieval import RetrievedChunk
from app.services.vector_store import vector_store
from app.utils.config import GROQ_API_KEY, GROQ_MODEL, GROQ_TIMEOUT_SECONDS

if TYPE_CHECKING:
    from groq import Groq

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a document intelligence assistant.\n"
    "Rules:\n"
    "- Summarize only from the supplied document chunks.\n"
    "- Never use outside knowledge.\n"
    "- Never hallucinate.\n"
    "- Return JSON only."
)

USER_PROMPT_TEMPLATE = (
    "Document chunks:\n{chunks}\n\n"
    "Return JSON with this exact shape:\n"
    "{{\n"
    '  "executive_summary": "...",\n'
    '  "risks": ["..."],\n'
    '  "opportunities": ["..."],\n'
    '  "missing_information": ["..."],\n'
    '  "next_actions": ["..."]\n'
    "}}\n"
    "Do not return markdown or explanations."
)

_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
_client: Any | None = None
_cache_signature: tuple[tuple[str, str, int | None, int | None], ...] | None = None
_cache_result: "InsightsResult" | None = None


class InsightsError(Exception):
    pass


class InsightsConfigurationError(InsightsError):
    pass


class InsightsInputError(InsightsError):
    pass


class InsightsResponseError(InsightsError):
    pass


class InsightsServiceError(InsightsError):
    pass


@dataclass(frozen=True)
class InsightsResult:
    executive_summary: str
    risks: list[str]
    opportunities: list[str]
    missing_information: list[str]
    next_actions: list[str]


def validate_groq_configuration() -> None:
    if not GROQ_API_KEY:
        raise InsightsConfigurationError("GROQ_API_KEY is missing")


def _get_client() -> Any:
    global _client
    if _client is None:
        validate_groq_configuration()
        try:
            from groq import Groq
        except ModuleNotFoundError as exc:
            raise InsightsConfigurationError("groq package is not installed") from exc
        _client = Groq(api_key=GROQ_API_KEY, timeout=GROQ_TIMEOUT_SECONDS)
    return _client


def _current_signature() -> tuple[tuple[str, str, int | None, int | None], ...]:
    records = registry.list_all()
    return tuple(
        sorted(
            (
                record.doc_id,
                record.filename,
                record.page_count,
                record.chunk_count,
            )
            for record in records
        )
    )


def _sample_chunks(max_chunks: int = 12) -> list[RetrievedChunk]:
    metadata_items = list(vector_store._metadata.values())
    if not metadata_items:
        return []

    grouped: list[list[dict]] = []
    for doc_id in sorted({item["doc_id"] for item in metadata_items}):
        doc_chunks = [
            metadata
            for metadata in sorted(
                (item for item in metadata_items if item["doc_id"] == doc_id),
                key=lambda item: (item["page_number"], item["chunk_index"]),
            )
        ]
        if doc_chunks:
            grouped.append(doc_chunks)

    sampled: list[RetrievedChunk] = []
    while grouped and len(sampled) < max_chunks:
        next_grouped: list[list[dict]] = []
        for doc_chunks in grouped:
            if not doc_chunks:
                continue
            sampled.append(RetrievedChunk.from_metadata(doc_chunks.pop(0), 0.0))
            if doc_chunks and len(sampled) < max_chunks:
                next_grouped.append(doc_chunks)
        grouped = next_grouped

    return sampled


def _serialize_chunks(chunks: list[RetrievedChunk]) -> str:
    payload = [
        {
            "chunk_id": chunk.chunk_id,
            "filename": chunk.filename,
            "page_number": chunk.page_number,
            "text": chunk.text,
        }
        for chunk in chunks
    ]
    return json.dumps(payload, ensure_ascii=True)


def _parse_model_payload(content: str) -> InsightsResult:
    stripped = content.strip()
    if not stripped:
        raise InsightsResponseError("empty response from Groq")

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        logger.warning("Groq insights response was not pure JSON; extracting embedded JSON object")
        match = _JSON_OBJECT_PATTERN.search(stripped)
        if not match:
            raise InsightsResponseError("invalid JSON from Groq")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise InsightsResponseError("invalid JSON from Groq") from exc

    def to_list(value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise InsightsResponseError("expected a list field in Groq response")
        return [str(item) for item in value if str(item).strip()]

    return InsightsResult(
        executive_summary=str(data.get("executive_summary", "")).strip(),
        risks=to_list(data.get("risks")),
        opportunities=to_list(data.get("opportunities")),
        missing_information=to_list(data.get("missing_information")),
        next_actions=to_list(data.get("next_actions")),
    )


def _call_model(chunks: list[RetrievedChunk]) -> InsightsResult:
    client = _get_client()
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(chunks=_serialize_chunks(chunks))},
        ],
    )
    content = (completion.choices[0].message.content or "").strip()
    logger.info("LLM response received")
    return _parse_model_payload(content)


def generate_insights() -> InsightsResult:
    start = perf_counter()
    try:
        signature = _current_signature()
        if not signature:
            logger.warning("No documents uploaded")
            raise InsightsInputError("no documents uploaded")

        global _cache_signature, _cache_result
        if _cache_signature == signature and _cache_result is not None:
            logger.info("Insights cache hit")
            return _cache_result

        logger.info("LLM request started: model=%s docs=%d", GROQ_MODEL, len(signature))
        chunks = _sample_chunks()
        if not chunks:
            logger.warning("No chunks available for insights")
            raise InsightsInputError("no document chunks available")

        result = _call_model(chunks)
        _cache_signature = signature
        _cache_result = result
        logger.info("Insights generated")
        return result
    except InsightsError:
        raise
    except Exception as exc:
        logger.error("Groq API failure", exc_info=True)
        raise InsightsServiceError("insights generation failed") from exc
    finally:
        duration = perf_counter() - start
        logger.info("LLM duration: %.2fs", duration)


def clear_insights_cache() -> None:
    global _cache_signature, _cache_result
    _cache_signature = None
    _cache_result = None