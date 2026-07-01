import logging
import re
import time
import uuid

from app.models.schemas import Chunk, ParsedPage

logger = logging.getLogger(__name__)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


class ChunkingError(Exception):
    """Raised when a page's text cannot be split into chunks."""


def _find_boundary(window: str) -> int:
    """Pick the best split point inside `window` (already <= chunk size).

    Priority: paragraph break, then sentence break, then a hard cut at the
    end of the window.
    """
    paragraph_breaks = list(_PARAGRAPH_BREAK.finditer(window))
    if paragraph_breaks:
        return paragraph_breaks[-1].end()

    sentence_breaks = list(_SENTENCE_BREAK.finditer(window))
    if sentence_breaks:
        return sentence_breaks[-1].end()

    return len(window)


def _split_page_text(text: str, size: int, overlap: int) -> list[str]:
    """Recursively split one page's text into <= `size`-character pieces.

    Consecutive pieces overlap by `overlap` characters so context survives
    a chunk boundary. Operates on a single page's text only, so overlap
    never crosses a page boundary.
    """
    if len(text) <= size:
        return [text]

    pieces: list[str] = []
    start = 0
    while start < len(text):
        remaining = text[start:]
        if len(remaining) <= size:
            pieces.append(remaining)
            break

        boundary = _find_boundary(remaining[:size])
        if boundary <= 0:
            boundary = size

        pieces.append(remaining[:boundary])
        next_start = start + boundary - overlap
        start = next_start if next_start > start else start + boundary

    return pieces


def _chunk_page(page: ParsedPage) -> list[Chunk]:
    text = page.text.strip()
    if not text:
        logger.warning("Empty page skipped: page %d of %s", page.page_number, page.filename)
        return []

    try:
        pieces = _split_page_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    except Exception as exc:
        logger.error(
            "Chunking failure on page %d of %s", page.page_number, page.filename, exc_info=True
        )
        raise ChunkingError(
            f"Failed to chunk page {page.page_number} of {page.filename}"
        ) from exc

    chunks: list[Chunk] = []
    for index, piece in enumerate(pieces):
        piece_text = piece.strip()
        if not piece_text:
            continue
        chunks.append(
            Chunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=page.doc_id,
                filename=page.filename,
                page_number=page.page_number,
                chunk_index=index,
                text=piece_text,
                character_count=len(piece_text),
            )
        )
    return chunks


def chunk_document(pages: list[ParsedPage]) -> list[Chunk]:
    """Split parsed pages into overlapping, boundary-aware chunks.

    Pure: takes typed pages in, returns typed chunks out. No registry,
    embedding, FAISS, retrieval, Claude, or upload-status concerns, and no
    knowledge of FastAPI.
    """
    logger.info("Chunking started: %d page(s)", len(pages))
    start_time = time.perf_counter()

    chunks: list[Chunk] = []
    for page in pages:
        chunks.extend(_chunk_page(page))

    duration = time.perf_counter() - start_time
    logger.info("Chunking completed: %d chunk(s) created", len(chunks))
    logger.info("Chunking duration: %.2fs", duration)

    return chunks