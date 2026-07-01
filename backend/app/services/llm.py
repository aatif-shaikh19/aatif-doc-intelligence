import json
import logging
import re
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any

from app.services.retrieval import RetrievedChunk
from app.utils.config import GROQ_API_KEY, GROQ_MODEL, GROQ_TIMEOUT_SECONDS

if TYPE_CHECKING:
    from groq import Groq

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a document intelligence assistant.\n"
    "Rules:\n"
    "- Answer ONLY from supplied document chunks.\n"
    "- Never use outside knowledge.\n"
    "- Never hallucinate.\n"
    "- If answer is unsupported, return exactly NOT_FOUND_IN_DOCUMENTS.\n"
    "Return JSON only."
)

USER_PROMPT_TEMPLATE = (
    "Question:\n{question}\n\n"
    "Retrieved Chunks:\n{chunks}\n\n"
    "Return JSON with this exact shape:\n"
    "{{\"answer\":\"...\",\"chunk_ids\":[\"chunk_id_1\",\"chunk_id_2\"]}}\n"
    "Do not return markdown."
)

NOT_FOUND_ANSWER = "The requested information was not found in the uploaded documents."

_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

_client: Any | None = None


class LLMError(Exception):
    pass


class LLMConfigurationError(LLMError):
    pass


class LLMResponseError(LLMError):
    pass


class LLMAuthenticationError(LLMError):
    pass


class LLMTimeoutError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    pass


class LLMServiceError(LLMError):
    pass


@dataclass(frozen=True)
class AnswerCitation:
    chunk_id: str
    filename: str
    page_number: int
    excerpt: str


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    citations: list[AnswerCitation]
    confidence: float


def validate_groq_configuration() -> None:
    if not GROQ_API_KEY:
        raise LLMConfigurationError("GROQ_API_KEY is missing")


def _get_client() -> Any:
    global _client
    if _client is None:
        validate_groq_configuration()
        try:
            from groq import Groq
        except ModuleNotFoundError as exc:
            raise LLMConfigurationError(
                "groq package is not installed"
            ) from exc
        _client = Groq(api_key=GROQ_API_KEY, timeout=GROQ_TIMEOUT_SECONDS)
    return _client


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


def _parse_model_payload(content: str) -> tuple[str, list[str]]:
    stripped = content.strip()
    if stripped == "NOT_FOUND_IN_DOCUMENTS":
        return "NOT_FOUND_IN_DOCUMENTS", []

    if not stripped:
        raise LLMResponseError("empty response from Groq")

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        # Reasoning models (e.g. Qwen3) may emit chain-of-thought text
        # (often wrapped in <think>...</think>) before or around the
        # actual JSON answer. Fall back to extracting the first JSON
        # object found anywhere in the response instead of assuming the
        # whole string is pure JSON.
        logger.warning("Groq response was not pure JSON; extracting embedded JSON object")
        match = _JSON_OBJECT_PATTERN.search(stripped)
        if not match:
            raise LLMResponseError("invalid JSON from Groq")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LLMResponseError("invalid JSON from Groq") from exc

    answer = str(data.get("answer", "")).strip()
    chunk_ids_raw = data.get("chunk_ids", [])
    if not isinstance(chunk_ids_raw, list):
        raise LLMResponseError("chunk_ids must be a list")

    chunk_ids = [str(chunk_id) for chunk_id in chunk_ids_raw]
    return answer, chunk_ids


def _build_citations(
    returned_chunk_ids: list[str],
    chunk_map: dict[str, RetrievedChunk],
) -> list[AnswerCitation]:
    citations: list[AnswerCitation] = []
    seen: set[str] = set()

    for chunk_id in returned_chunk_ids:
        if chunk_id in seen:
            continue
        seen.add(chunk_id)

        chunk = chunk_map.get(chunk_id)
        if chunk is None:
            logger.warning("Invalid citation removed: chunk_id=%s", chunk_id)
            continue

        citations.append(
            AnswerCitation(
                chunk_id=chunk.chunk_id,
                filename=chunk.filename,
                page_number=chunk.page_number,
                excerpt=chunk.text,
            )
        )

    return citations


def _compute_confidence(top_score: float | None, citation_count: int) -> float:
    if top_score is None or citation_count == 0:
        return 0.0
    return round(max(0.0, min(1.0, float(top_score))), 4)


def generate_answer(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    top_score: float | None,
) -> AnswerResult:
    start = perf_counter()

    if not retrieved_chunks:
        logger.warning("No retrieved chunks")
        return AnswerResult(
            answer=NOT_FOUND_ANSWER,
            citations=[],
            confidence=0.0,
        )

    try:
        logger.info("LLM request started: model=%s chunks=%d", GROQ_MODEL, len(retrieved_chunks))
        client = _get_client()
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT_TEMPLATE.format(
                        question=question,
                        chunks=_serialize_chunks(retrieved_chunks),
                    ),
                },
            ],
        )

        content = (completion.choices[0].message.content or "").strip()
        logger.info("LLM response received")

        answer, returned_chunk_ids = _parse_model_payload(content)
        if answer == "NOT_FOUND_IN_DOCUMENTS" or not answer:
            result = AnswerResult(answer=NOT_FOUND_ANSWER, citations=[], confidence=0.0)
            logger.info("Answer generated")
            return result

        chunk_map = {chunk.chunk_id: chunk for chunk in retrieved_chunks}
        citations = _build_citations(returned_chunk_ids, chunk_map)
        confidence = _compute_confidence(top_score, len(citations))

        result = AnswerResult(
            answer=answer,
            citations=citations,
            confidence=confidence,
        )
        logger.info("Answer generated")
        return result
    except LLMError:
        raise
    except Exception as exc:
        error_name = exc.__class__.__name__
        logger.error("Groq API failure", exc_info=True)

        if error_name in {"AuthenticationError"}:
            raise LLMAuthenticationError("groq authentication failed") from exc
        if error_name in {"APITimeoutError", "TimeoutError"}:
            raise LLMTimeoutError("groq request timeout") from exc
        if error_name in {"RateLimitError"}:
            raise LLMRateLimitError("groq rate limited") from exc
        raise LLMServiceError("groq request failed") from exc
    finally:
        duration = perf_counter() - start
        logger.info("LLM duration: %.2fs", duration)