import numpy as np

from app.services.insights import InsightsResult
from app.services.llm import AnswerCitation, AnswerResult
from app.services.retrieval import RetrievedChunk, RetrievalResult

from tests.helpers import create_corrupt_pdf, create_pdf


def _upload_pdf(client, pdf_path, content_type="application/pdf"):
    with pdf_path.open("rb") as handle:
        return client.post(
            "/upload",
            files=[("files", (pdf_path.name, handle, content_type))],
        )


def test_upload_valid_pdf_and_list_documents(client, tmp_path, monkeypatch):
    pdf_path = create_pdf(tmp_path, "valid.pdf", ["Python, Go, Rust"])
    monkeypatch.setattr(
        "app.api.upload.embed_chunks",
        lambda chunks: np.zeros((len(chunks), 384), dtype=np.float32),
    )

    response = _upload_pdf(client, pdf_path)

    assert response.status_code == 200
    payload = response.json()
    assert payload["results"][0]["status"] == "success"
    assert payload["results"][0]["pages"] == 1
    assert payload["results"][0]["chunks"] >= 1

    documents_response = client.get("/documents")
    assert documents_response.status_code == 200
    documents = documents_response.json()["documents"]
    assert len(documents) == 1
    assert documents[0]["filename"] == "valid.pdf"


def test_upload_rejects_corrupt_pdf(client, tmp_path):
    pdf_path = create_corrupt_pdf(tmp_path, "corrupt.pdf")

    response = _upload_pdf(client, pdf_path)

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["status"] == "rejected"
    assert "parse" in result["reason"].lower() or "corrupt" in result["reason"].lower()


def test_upload_rejects_empty_pdf(client, tmp_path):
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"")

    response = _upload_pdf(client, pdf_path)

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["status"] == "rejected"
    assert result["reason"] == "empty file"


def test_upload_rejects_unsupported_file(client, tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("plain text")

    response = _upload_pdf(client, file_path, content_type="text/plain")

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["status"] == "rejected"
    assert result["reason"] == "unsupported file type"


def test_query_no_documents_returns_400(client):
    response = client.post("/query", json={"question": "Where did Aatif intern?"})

    assert response.status_code == 400
    assert "no uploaded documents" in response.json()["detail"]


def test_query_no_match_returns_fallback(client, monkeypatch):
    monkeypatch.setattr(
        "app.api.query.retrieve",
        lambda question: RetrievalResult(match_found=False, chunks=[], top_score=None),
    )

    response = client.post("/query", json={"question": "Where did Aatif intern?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["match_found"] is False
    assert payload["chunks"] == []
    assert payload["top_score"] is None
    assert payload["answer"] == "The requested information was not found in the uploaded documents."
    assert payload["citations"] == []
    assert payload["confidence"] == 0.0


def test_query_successful_match_returns_answer_and_citations(client, monkeypatch):
    retrieved_chunk = RetrievedChunk(
        chunk_id="chunk-1",
        doc_id="doc-1",
        filename="resume.pdf",
        page_number=4,
        chunk_index=0,
        text="Aatif built several projects.",
        score=0.91,
    )
    monkeypatch.setattr(
        "app.api.query.retrieve",
        lambda question: RetrievalResult(match_found=True, chunks=[retrieved_chunk], top_score=0.91),
    )
    monkeypatch.setattr(
        "app.api.query.generate_answer",
        lambda question, retrieved_chunks, top_score: AnswerResult(
            answer="Aatif built projects in Python and Go.",
            citations=[
                AnswerCitation(
                    chunk_id="chunk-1",
                    filename="resume.pdf",
                    page_number=4,
                    excerpt="Aatif built several projects.",
                )
            ],
            confidence=0.91,
        ),
    )

    response = client.post("/query", json={"question": "What projects has Aatif built?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["match_found"] is True
    assert payload["top_score"] == 0.91
    assert payload["answer"] == "Aatif built projects in Python and Go."
    assert payload["confidence"] == 0.91
    assert payload["citations"][0]["chunk_id"] == "chunk-1"


def test_insights_no_documents_returns_400(client):
    response = client.post("/insights")

    assert response.status_code == 400
    assert "no documents uploaded" in response.json()["detail"]


def test_insights_generation_cache_and_invalidation(client, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.api.upload.embed_chunks",
        lambda chunks: np.zeros((len(chunks), 384), dtype=np.float32),
    )

    call_count = {"value": 0}

    def fake_call_model(chunks):
        call_count["value"] += 1
        return InsightsResult(
            executive_summary="The documents describe projects and internships.",
            risks=["Limited sample size"],
            opportunities=["Add more evidence"],
            missing_information=["Dates"],
            next_actions=["Review the resume"],
        )

    monkeypatch.setattr("app.services.insights._call_model", fake_call_model)

    first_pdf = create_pdf(tmp_path, "first.pdf", ["Aatif built several projects."])
    second_pdf = create_pdf(tmp_path, "second.pdf", ["Aatif completed internships."])

    first_upload = _upload_pdf(client, first_pdf)
    assert first_upload.status_code == 200

    first_response = client.post("/insights")
    assert first_response.status_code == 200
    assert first_response.json()["executive_summary"] == "The documents describe projects and internships."
    assert call_count["value"] == 1

    second_response = client.post("/insights")
    assert second_response.status_code == 200
    assert call_count["value"] == 1

    second_upload = _upload_pdf(client, second_pdf)
    assert second_upload.status_code == 200

    third_response = client.post("/insights")
    assert third_response.status_code == 200
    assert call_count["value"] == 2