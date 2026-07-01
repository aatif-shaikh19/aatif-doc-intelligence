import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.llm import (
    AnswerResult,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMRateLimitError,
    LLMResponseError,
    LLMServiceError,
    LLMTimeoutError,
    generate_answer,
)
from app.services.retrieval import (
    EmptyQuestionError,
    NoDocumentsIndexedError,
    RetrievalFailureError,
    RetrievalInputError,
    RetrievalResult,
    RetrievedChunk,
    retrieve,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class QueryRequest(BaseModel):
    question: str


class QueryChunk(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    score: float


class QueryResponse(BaseModel):
    match_found: bool
    chunks: list[QueryChunk]
    top_score: float | None
    answer: str
    citations: list[dict]
    confidence: float


def _to_query_chunk(chunk: RetrievedChunk) -> QueryChunk:
    return QueryChunk(**chunk.__dict__)


def _to_response(result: RetrievalResult, answer_result: AnswerResult) -> QueryResponse:
    # match_found reflects whether the LLM actually grounded an answer in a
    # retrieved chunk, not just whether retrieval cleared the similarity
    # threshold -- retrieval can pass chunks through that turn out not to
    # support the question, in which case generate_answer correctly falls
    # back to NOT_FOUND_IN_DOCUMENTS with no citations.
    return QueryResponse(
        match_found=bool(answer_result.citations),
        chunks=[_to_query_chunk(chunk) for chunk in result.chunks],
        top_score=result.top_score,
        answer=answer_result.answer,
        citations=[citation.__dict__ for citation in answer_result.citations],
        confidence=answer_result.confidence,
    )


@router.post("/query", response_model=QueryResponse)
def query_documents(payload: QueryRequest) -> QueryResponse:
    try:
        result = retrieve(payload.question)
    except (EmptyQuestionError, NoDocumentsIndexedError, RetrievalInputError) as exc:
        logger.warning("Query rejected: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RetrievalFailureError as exc:
        logger.error("Query failed", exc_info=True)
        raise HTTPException(status_code=500, detail="retrieval failed") from exc

    if not result.chunks:
        return QueryResponse(
            match_found=False,
            chunks=[],
            top_score=None,
            answer="The requested information was not found in the uploaded documents.",
            citations=[],
            confidence=0.0,
        )

    try:
        answer_result = generate_answer(
            question=payload.question,
            retrieved_chunks=result.chunks,
            top_score=result.top_score,
        )
    except (
        LLMConfigurationError,
        LLMAuthenticationError,
        LLMTimeoutError,
        LLMRateLimitError,
        LLMServiceError,
        LLMResponseError,
    ) as exc:
        logger.error("Query failed", exc_info=True)
        raise HTTPException(status_code=500, detail="llm generation failed") from exc

    return _to_response(result, answer_result)