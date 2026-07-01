import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

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


def _to_query_chunk(chunk: RetrievedChunk) -> QueryChunk:
    return QueryChunk(**chunk.__dict__)


def _to_response(result: RetrievalResult) -> QueryResponse:
    return QueryResponse(
        match_found=result.match_found,
        chunks=[_to_query_chunk(chunk) for chunk in result.chunks],
        top_score=result.top_score,
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

    return _to_response(result)