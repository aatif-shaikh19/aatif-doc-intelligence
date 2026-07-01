from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel


class UploadResult(BaseModel):
    filename: str
    doc_id: Optional[str] = None
    status: Literal["success", "rejected"]
    reason: Optional[str] = None
    # Populated starting Phase 3/4 (parsing/chunking); absent for now.
    pages: Optional[int] = None
    chunks: Optional[int] = None


class UploadResponse(BaseModel):
    results: list[UploadResult]


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    uploaded_at: datetime
    # Populated starting Phase 3/4 (parsing/chunking); absent for now.
    pages: Optional[int] = None
    chunks: Optional[int] = None


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]
