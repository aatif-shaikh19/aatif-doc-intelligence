import logging
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from app.models.schemas import Chunk

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Return the shared embedding model, loading it once on first use.

    Application-scoped singleton: `app.main` calls this at FastAPI
    startup so the model is loaded exactly once, before any request is
    served, and every later call (from `embed_chunks`) reuses the same
    in-memory instance.
    """
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
        device = "GPU" if "cuda" in str(_model.device).lower() else "CPU"
        logger.info("Embedding model loaded: %s", MODEL_NAME)
        logger.info("Dimension: %d", _model.get_sentence_embedding_dimension())
        logger.info("Device: %s", device)
    return _model


def embed_chunks(chunks: list[Chunk]) -> np.ndarray:
    """Batch-embed chunk text into L2-normalized float32 vectors.

    Pure: takes typed chunks in, returns a vector array out. No FAISS,
    registry, or upload-status concerns.
    """
    if not chunks:
        logger.warning("Empty chunk list: nothing to embed")
        return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

    logger.info("Embedding started: %d chunk(s)", len(chunks))
    start = time.perf_counter()

    try:
        model = get_model()
        vectors = model.encode(
            [chunk.text for chunk in chunks],
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception:
        logger.error("Embedding failure", exc_info=True)
        raise

    vectors = np.asarray(vectors, dtype=np.float32)

    duration = time.perf_counter() - start
    logger.info("Embedding completed: %d vector(s) generated", len(vectors))
    logger.info("Embedding duration: %.2fs", duration)

    return vectors
