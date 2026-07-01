import logging
import shutil

from fastapi import APIRouter, HTTPException

from app.models.registry import registry
from app.models.schemas import DeleteDocumentResponse, DocumentInfo, DocumentsResponse
from app.services.vector_store import vector_store
from app.utils.config import UPLOAD_DIR

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/documents", response_model=DocumentsResponse)
def list_documents() -> DocumentsResponse:
    records = registry.list_all()
    documents = [
        DocumentInfo(
            doc_id=r.doc_id,
            filename=r.filename,
            uploaded_at=r.uploaded_at,
            pages=r.page_count,
            chunks=r.chunk_count,
        )
        for r in records
    ]
    logger.info("Listed %d document(s)", len(documents))
    return DocumentsResponse(documents=documents)


@router.delete("/documents/{doc_id}", response_model=DeleteDocumentResponse)
def delete_document(doc_id: str) -> DeleteDocumentResponse:
    if registry.get(doc_id) is None:
        logger.warning("Delete rejected: doc_id=%s not found", doc_id)
        raise HTTPException(status_code=404, detail="doc_id not found")

    vector_store.remove(doc_id)
    registry.remove(doc_id)
    shutil.rmtree(UPLOAD_DIR / doc_id, ignore_errors=True)

    logger.info("Deleted document doc_id=%s", doc_id)
    return DeleteDocumentResponse(doc_id=doc_id, status="deleted")
