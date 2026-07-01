import logging

from fastapi import APIRouter

from app.models.registry import registry
from app.models.schemas import DocumentInfo, DocumentsResponse

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
