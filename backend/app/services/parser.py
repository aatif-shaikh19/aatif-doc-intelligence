import logging
import time

import fitz

from app.models.schemas import ParsedPage

logger = logging.getLogger(__name__)


class PDFParseError(Exception):
    """Raised when a PDF file cannot be parsed into pages."""


def extract_pages(pdf_path: str, doc_id: str, filename: str) -> list[ParsedPage]:
    """Extract per-page text from a PDF.

    Pure: reads only from pdf_path and returns typed pages. No filesystem
    writes, no registry access, no HTTP/upload knowledge, no chunking.
    """
    logger.info("Parsing started: %s", filename)
    start = time.perf_counter()

    try:
        document = fitz.open(pdf_path)
    except (FileNotFoundError, PermissionError, OSError) as exc:
        logger.error("Cannot read file: %s", filename)
        raise PDFParseError(f"Cannot read file: {filename}") from exc
    except Exception as exc:
        logger.error("Corrupt PDF: %s", filename)
        raise PDFParseError(f"Corrupted PDF: {filename}") from exc

    try:
        if document.needs_pass:
            logger.error("Encrypted PDF: %s", filename)
            raise PDFParseError(f"Encrypted PDF (password-protected): {filename}")

        if document.page_count == 0:
            logger.error("Invalid PDF: %s", filename)
            raise PDFParseError(f"Invalid PDF: {filename}")

        pages: list[ParsedPage] = []
        for index in range(document.page_count):
            page_number = index + 1
            try:
                text = document.load_page(index).get_text()
            except Exception as exc:
                logger.error("Corrupt PDF: %s", filename)
                raise PDFParseError(f"Corrupted PDF: {filename}") from exc

            if not text.strip():
                logger.warning("Page %d contains no text: %s", page_number, filename)
                continue

            pages.append(
                ParsedPage(
                    doc_id=doc_id,
                    filename=filename,
                    page_number=page_number,
                    text=text,
                    character_count=len(text),
                )
            )
    finally:
        document.close()

    duration = time.perf_counter() - start
    logger.info("Parsed %d pages: %s", len(pages), filename)
    logger.info("Parsing time for %s: %.2fs", filename, duration)

    return pages
