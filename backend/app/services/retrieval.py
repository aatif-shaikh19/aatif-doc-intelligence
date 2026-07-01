import logging
import os
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from app.services.embeddings import get_model
from app.services.vector_store import vector_store

logger = logging.getLogger(__name__)

DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("RETRIEVAL_SIMILARITY_THRESHOLD", "0.35"))


class RetrievalError(Exception):
    pass


class RetrievalInputError(RetrievalError):
    pass


class EmptyQuestionError(RetrievalInputError):
    pass


class NoDocumentsIndexedError(RetrievalInputError):
    pass


class RetrievalFailureError(RetrievalError):
    pass


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    score: float

    @classmethod
    def from_metadata(cls, metadata: dict, score: float) -> "RetrievedChunk":
        return cls(
            chunk_id=str(metadata["chunk_id"]),
            doc_id=str(metadata["doc_id"]),
            filename=str(metadata["filename"]),
            page_number=int(metadata["page_number"]),
            chunk_index=int(metadata["chunk_index"]),
            text=str(metadata["text"]),
            score=float(score),
        )


@dataclass(frozen=True)
class RetrievalResult:
    match_found: bool
    chunks: list[RetrievedChunk]
    top_score: float | None


def _indexed_chunk_count() -> int:
    index = getattr(vector_store, "_index", None)
    return int(getattr(index, "ntotal", 0)) if index is not None else 0


def retrieve(
    question: str,
    top_k: int = 5,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> RetrievalResult:
    start = perf_counter()
    logger.info(
        "Retrieval started: question_length=%d top_k=%d threshold=%.2f",
        len(question) if isinstance(question, str) else -1,
        top_k,
        threshold,
    )

    try:
        if not isinstance(question, str):
            logger.warning("Invalid question type: %s", type(question).__name__)
            raise RetrievalInputError("question must be a string")

        normalized_question = question.strip()
        if not normalized_question:
            logger.warning("Empty question")
            raise EmptyQuestionError("question must not be empty")

        if _indexed_chunk_count() == 0:
            logger.warning("No documents indexed")
            raise NoDocumentsIndexedError("no uploaded documents available")

        model = get_model()
        query_vector = np.asarray(
            model.encode(
                [normalized_question],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0],
            dtype=np.float32,
        )
        logger.info("Question embedded: dimension=%d", int(query_vector.shape[0]))

        search_results = vector_store.search(query_vector, top_k)
        logger.info("Search completed: retrieved_chunks=%d", len(search_results))

        top_score = float(search_results[0][1]) if search_results else None
        logger.info("Highest similarity score: %s", "%.4f" % top_score if top_score is not None else "none")

        scored_chunks = [
            RetrievedChunk.from_metadata(metadata, score)
            for metadata, score in search_results
        ]

        for chunk in scored_chunks:
            logger.info(
                "Retrieved chunk score: chunk_id=%s score=%.4f",
                chunk.chunk_id,
                chunk.score,
            )

        filtered_chunks = sorted(
            (chunk for chunk in scored_chunks if chunk.score >= threshold),
            key=lambda chunk: chunk.score,
            reverse=True,
        )
        logger.info(
            "Threshold filtering completed: retrieved_chunks=%d kept_chunks=%d threshold=%.4f",
            len(scored_chunks),
            len(filtered_chunks),
            threshold,
        )

        if not filtered_chunks:
            logger.warning("No matches above threshold: threshold=%.2f", threshold)
            return RetrievalResult(match_found=False, chunks=[], top_score=None)

        logger.info("Matches found: %d", len(filtered_chunks))
        return RetrievalResult(match_found=True, chunks=filtered_chunks, top_score=top_score)
    except RetrievalError:
        raise
    except Exception as exc:
        logger.error("Retrieval failure", exc_info=True)
        raise RetrievalFailureError("retrieval failed") from exc
    finally:
        duration = perf_counter() - start
        logger.info("Retrieval duration: %.2fs", duration)