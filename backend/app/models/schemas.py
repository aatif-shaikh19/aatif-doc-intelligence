from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel


class UploadResult(BaseModel):
    filename: str
    doc_id: Optional[str] = None
    status: Literal["success", "rejected"]
    reason: Optional[str] = None
    pages: Optional[int] = None  # populated by the parser (Phase 3)
    chunks: Optional[int] = None  # populated by the chunker (Phase 4)


class UploadResponse(BaseModel):
    results: list[UploadResult]


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    uploaded_at: datetime
    pages: Optional[int] = None  # populated by the parser (Phase 3)
    chunks: Optional[int] = None  # populated by the chunker (Phase 4)


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]


class ParsedPage(BaseModel):
    doc_id: str
    filename: str
    page_number: int
    text: str
    character_count: int


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    character_count: int
