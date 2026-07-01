import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.registry import registry
from app.models.schemas import UploadResponse, UploadResult
from app.services.chunker import ChunkingError, chunk_document
from app.services.embeddings import embed_chunks
from app.services.parser import PDFParseError, extract_pages
from app.services.vector_store import vector_store
from app.utils.config import (
    ALLOWED_UPLOAD_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    MAX_FILES_PER_UPLOAD,
    UPLOAD_DIR,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _validate_extension(filename: str) -> str | None:
    if Path(filename).suffix.lower() not in ALLOWED_UPLOAD_EXTENSIONS:
        return "unsupported file type"
    return None


def _save_file(doc_id: str, filename: str, content: bytes) -> Path:
    doc_dir = UPLOAD_DIR / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    file_path = doc_dir / Path(filename).name
    file_path.write_bytes(content)
    return file_path


def _delete_saved_file(doc_id: str) -> None:
    shutil.rmtree(UPLOAD_DIR / doc_id, ignore_errors=True)


async def _process_upload(upload: UploadFile) -> UploadResult:
    filename = upload.filename
    if not filename:
        logger.warning("Rejected upload with missing filename")
        return UploadResult(filename="", status="rejected", reason="missing filename")

    extension_error = _validate_extension(filename)
    if extension_error:
        logger.warning("Rejected %s: %s", filename, extension_error)
        return UploadResult(filename=filename, status="rejected", reason=extension_error)

    content = await upload.read()

    if not content:
        logger.warning("Rejected %s: empty file", filename)
        return UploadResult(filename=filename, status="rejected", reason="empty file")

    if len(content) > MAX_FILE_SIZE_BYTES:
        logger.warning("Rejected %s: exceeds size limit", filename)
        return UploadResult(
            filename=filename, status="rejected", reason="file exceeds size limit"
        )

    doc_id = str(uuid.uuid4())
    try:
        file_path = _save_file(doc_id, filename, content)
    except OSError:
        logger.error("Failed to save %s (doc_id=%s)", filename, doc_id, exc_info=True)
        return UploadResult(filename=filename, status="rejected", reason="could not save file")

    try:
        pages = extract_pages(str(file_path), doc_id, filename)
    except PDFParseError as exc:
        _delete_saved_file(doc_id)
        logger.warning("Rejected %s: %s", filename, exc)
        return UploadResult(filename=filename, status="rejected", reason=str(exc))

    try:
        chunks = chunk_document(pages)
    except ChunkingError as exc:
        _delete_saved_file(doc_id)
        logger.warning("Rejected %s: %s", filename, exc)
        return UploadResult(filename=filename, status="rejected", reason=str(exc))

    try:
        vectors = embed_chunks(chunks)
        vector_store.add(chunks, vectors)
    except Exception:
        _delete_saved_file(doc_id)
        logger.error("Embedding failed for %s", filename, exc_info=True)
        return UploadResult(filename=filename, status="rejected", reason="embedding failed")

    page_count = len(pages)
    chunk_count = len(chunks)
    character_count = sum(page.character_count for page in pages)
    registry.add(
        doc_id=doc_id,
        filename=filename,
        page_count=page_count,
        character_count=character_count,
        chunk_count=chunk_count,
    )
    logger.info("File saved: %s -> doc_id=%s", filename, doc_id)
    return UploadResult(
        filename=filename,
        doc_id=doc_id,
        status="success",
        pages=page_count,
        chunks=chunk_count,
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_documents(files: list[UploadFile] = File(...)) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    if len(files) > MAX_FILES_PER_UPLOAD:
        raise HTTPException(
            status_code=400,
            detail=f"A maximum of {MAX_FILES_PER_UPLOAD} files can be uploaded at once",
        )

    logger.info("Upload started: %d file(s)", len(files))
    results: list[UploadResult] = []

    for upload in files:
        try:
            result = await _process_upload(upload)
        except Exception:
            logger.error("Unexpected error processing %s", upload.filename, exc_info=True)
            result = UploadResult(
                filename=upload.filename or "", status="rejected", reason="upload failed"
            )
        finally:
            await upload.close()
        results.append(result)

    succeeded = sum(1 for r in results if r.status == "success")
    logger.info(
        "Upload completed: %d succeeded, %d rejected", succeeded, len(results) - succeeded
    )
    return UploadResponse(results=results)
