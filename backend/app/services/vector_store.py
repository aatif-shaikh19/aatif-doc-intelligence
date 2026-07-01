import json
import logging
from threading import Lock

import faiss
import numpy as np

from app.models.schemas import Chunk
from app.utils.config import BACKEND_DIR

logger = logging.getLogger(__name__)

VECTOR_STORE_DIR = BACKEND_DIR / "vector_store"
INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"
METADATA_PATH = VECTOR_STORE_DIR / "metadata.json"

EMBEDDING_DIMENSION = 384


class VectorStore:
    """FAISS-backed store for chunk vectors and their metadata.

    Owns ONLY vector storage, lookup, and persistence: no embedding
    generation, no query-time threshold/prompt logic, no registry or
    HTTP concerns. Application-scoped singleton, loaded once at FastAPI
    startup and mutated in place by uploads.
    """

    def __init__(self) -> None:
        self._index: faiss.Index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        self._chunk_ids: list[str] = []  # FAISS row index -> chunk_id
        self._metadata: dict[str, dict] = {}  # chunk_id -> chunk metadata
        self._lock = Lock()

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            logger.warning("Empty chunk list: nothing to add to vector store")
            return

        with self._lock:
            self._index.add(np.asarray(vectors, dtype=np.float32))
            for chunk in chunks:
                self._chunk_ids.append(chunk.chunk_id)
                self._metadata[chunk.chunk_id] = {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "filename": chunk.filename,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "character_count": chunk.character_count,
                }
            logger.info("Added %d vector(s) to vector store", len(chunks))
            self._save_locked()

    def remove(self, doc_id: str) -> None:
        """Remove every chunk belonging to `doc_id` and rebuild the index.

        Trade-off: `IndexFlatIP` (like all FAISS Flat indexes) has no
        in-place delete-by-id, so the only way to drop vectors is to
        reconstruct the surviving vectors and rebuild a fresh index from
        them. This is O(n) in total vector count, which is acceptable at
        this project's scale (<=50 documents) but would need a different
        index type (e.g. `IndexIDMap2` with `remove_ids`) or a
        compaction strategy if the corpus grew much larger.
        """
        with self._lock:
            keep_positions = [
                i
                for i, chunk_id in enumerate(self._chunk_ids)
                if self._metadata[chunk_id]["doc_id"] != doc_id
            ]
            if len(keep_positions) == len(self._chunk_ids):
                logger.info("No chunks found for doc_id=%s; nothing to remove", doc_id)
                return

            removed_count = len(self._chunk_ids) - len(keep_positions)
            surviving_ids = [self._chunk_ids[i] for i in keep_positions]

            rebuilt_index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
            if keep_positions:
                all_vectors = self._index.reconstruct_n(0, self._index.ntotal)
                rebuilt_index.add(all_vectors[keep_positions])

            self._index = rebuilt_index
            self._chunk_ids = surviving_ids
            self._metadata = {cid: self._metadata[cid] for cid in surviving_ids}

            logger.info(
                "Removed %d chunk(s) for doc_id=%s; index rebuilt (%d vector(s) remain)",
                removed_count,
                doc_id,
                self._index.ntotal,
            )
            self._save_locked()

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[dict, float]]:
        """Return up to `top_k` (metadata, similarity_score) pairs for a query vector.

        This is NOT retrieval. It is only:
        - FAISS similarity search over already-embedded vectors
        - returns each match's stored metadata
        - returns each match's similarity score

        It must NOT (and does not):
        - embed the query text
        - apply a similarity threshold
        - call an LLM
        - build a prompt

        Those are retrieval-service concerns (Phase 6) that belong in
        `retrieval.py`, layered on top of this method.
        """
        if self._index.ntotal == 0:
            return []

        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query, k)

        results: list[tuple[dict, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk_id = self._chunk_ids[idx]
            results.append((self._metadata[chunk_id], float(score)))
        return results

    def save(self) -> None:
        with self._lock:
            self._save_locked()

    def _save_locked(self) -> None:
        """Persist the index and metadata. Caller must hold `self._lock`."""
        try:
            VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(INDEX_PATH))
            METADATA_PATH.write_text(
                json.dumps({"chunk_ids": self._chunk_ids, "metadata": self._metadata})
            )
            logger.info("Vector store saved: %d vector(s)", self._index.ntotal)
        except Exception:
            logger.error("FAISS persistence failure", exc_info=True)
            raise

    def load(self) -> None:
        if not INDEX_PATH.exists() or not METADATA_PATH.exists():
            logger.warning("Missing vector store; starting with an empty index")
            return

        with self._lock:
            try:
                self._index = faiss.read_index(str(INDEX_PATH))
                payload = json.loads(METADATA_PATH.read_text())
                self._chunk_ids = payload["chunk_ids"]
                self._metadata = payload["metadata"]
                logger.info("Vector store loaded:")
                logger.info("Vectors: %d", self._index.ntotal)
                logger.info("Metadata entries: %d", len(self._metadata))
            except Exception:
                logger.error("FAISS persistence failure", exc_info=True)
                raise


vector_store = VectorStore()
