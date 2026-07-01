import os

import faiss
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("GROQ_MODEL", "test-model")
os.environ.setdefault("GROQ_TIMEOUT_SECONDS", "1")
os.environ.setdefault("RETRIEVAL_SIMILARITY_THRESHOLD", "0.25")

from app.main import app
from app.models.registry import registry
from app.services.insights import clear_insights_cache
from app.services.vector_store import vector_store


@pytest.fixture(autouse=True)
def reset_backend_state(monkeypatch):
    monkeypatch.setattr("app.main.get_model", lambda: None)
    monkeypatch.setattr("app.main.vector_store.load", lambda: None)
    monkeypatch.setattr(vector_store, "_save_locked", lambda: None)

    vector_store._index = faiss.IndexFlatIP(384)
    vector_store._chunk_ids = []
    vector_store._metadata = {}

    with registry._lock:
        registry._documents.clear()

    clear_insights_cache()
    yield


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client