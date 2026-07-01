import numpy as np

from app.services import retrieval as retrieval_module
from app.services.retrieval import retrieve


def _fake_model():
    class FakeModel:
        def encode(self, texts, **kwargs):
            return np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32)

    return FakeModel()


def _metadata(chunk_id: str, score: float, doc_id: str = "doc-1", chunk_index: int = 0):
    return (
        {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "filename": "resume.pdf",
            "page_number": 1,
            "chunk_index": chunk_index,
            "text": f"Chunk text for {chunk_id}",
            "character_count": 24,
        },
        score,
    )


def test_retrieval_returns_relevant_chunks_sorted(monkeypatch):
    monkeypatch.setattr(retrieval_module, "get_model", _fake_model)
    monkeypatch.setattr(retrieval_module, "_indexed_chunk_count", lambda: 3)
    monkeypatch.setattr(
        retrieval_module.vector_store,
        "search",
        lambda query_vector, top_k: [
            _metadata("chunk-1", 0.48, chunk_index=2),
            _metadata("chunk-2", 0.92, chunk_index=0),
            _metadata("chunk-3", 0.71, chunk_index=1),
        ][:top_k],
    )

    result = retrieve("What projects has Aatif built?", top_k=5)

    assert result.match_found is True
    assert result.top_score == 0.48
    assert [chunk.score for chunk in result.chunks] == [0.92, 0.71, 0.48]


def test_retrieval_filters_below_threshold(monkeypatch):
    monkeypatch.setattr(retrieval_module, "get_model", _fake_model)
    monkeypatch.setattr(retrieval_module, "_indexed_chunk_count", lambda: 2)
    monkeypatch.setattr(
        retrieval_module.vector_store,
        "search",
        lambda query_vector, top_k: [_metadata("chunk-1", 0.24), _metadata("chunk-2", 0.18)],
    )

    result = retrieve("Where did Aatif complete internships?", threshold=0.35)

    assert result.match_found is False
    assert result.chunks == []
    assert result.top_score is None


def test_retrieval_respects_top_k(monkeypatch):
    monkeypatch.setattr(retrieval_module, "get_model", _fake_model)
    monkeypatch.setattr(retrieval_module, "_indexed_chunk_count", lambda: 10)

    seen = {}

    def fake_search(query_vector, top_k):
        seen["top_k"] = top_k
        return [_metadata(f"chunk-{index}", 0.9 - (index * 0.1), chunk_index=index) for index in range(5)]

    monkeypatch.setattr(retrieval_module.vector_store, "search", fake_search)

    retrieve("List the projects.", top_k=2)

    assert seen["top_k"] == 2


def test_retrieval_returns_descending_similarity_order(monkeypatch):
    monkeypatch.setattr(retrieval_module, "get_model", _fake_model)
    monkeypatch.setattr(retrieval_module, "_indexed_chunk_count", lambda: 3)
    monkeypatch.setattr(
        retrieval_module.vector_store,
        "search",
        lambda query_vector, top_k: [
            _metadata("chunk-a", 0.52),
            _metadata("chunk-b", 0.89),
            _metadata("chunk-c", 0.67),
        ],
    )

    result = retrieve("Tell me about internships.")

    assert [chunk.chunk_id for chunk in result.chunks] == ["chunk-b", "chunk-c", "chunk-a"]