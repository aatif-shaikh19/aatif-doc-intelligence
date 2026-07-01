import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str
    filename: str
    uploaded_at: datetime


class DocumentRegistry:
    """In-memory metadata store for uploaded documents, keyed by doc_id.

    Pure storage: callers decide what gets registered and when. This class
    holds no upload-validation or business logic.
    """

    def __init__(self) -> None:
        self._documents: dict[str, DocumentRecord] = {}
        self._lock = Lock()

    def add(self, doc_id: str, filename: str) -> DocumentRecord:
        record = DocumentRecord(
            doc_id=doc_id,
            filename=filename,
            uploaded_at=datetime.now(timezone.utc),
        )
        with self._lock:
            self._documents[doc_id] = record
        logger.info("Registered document doc_id=%s filename=%s", doc_id, filename)
        return record

    def get(self, doc_id: str) -> DocumentRecord | None:
        with self._lock:
            return self._documents.get(doc_id)

    def list_all(self) -> list[DocumentRecord]:
        with self._lock:
            return list(self._documents.values())


registry = DocumentRegistry()
