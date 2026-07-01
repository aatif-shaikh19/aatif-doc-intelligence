# Document Intelligence System

Upload PDFs, ask natural-language questions about them, and get answers grounded in the documents with exact citations (filename + page number). Optimized for **accuracy** over latency.

## Overview

The system ingests 1–50 PDFs, extracts and chunks their text, embeds the chunks, and stores them in a local vector index. Questions are answered using retrieval-augmented generation: only chunks retrieved above a similarity threshold are ever shown to the LLM, and the LLM is instructed to answer strictly from those chunks or say the answer isn't found.

## Architecture

```
React Frontend (Documents / Ask tabs)
        │  axios (HTTP)
        ▼
FastAPI Backend
        │
 ┌──────┼───────────────┐
 ▼      ▼               ▼
Upload  Query      Documents
(orchestrates:     (retrieval + LLM
 parse→chunk→       answer generation)
 embed→store)
        │
        ▼
 Embedding Engine (sentence-transformers, all-MiniLM-L6-v2)
        │
        ▼
 FAISS Vector Store (IndexFlatIP, persisted to disk)
        │
        ▼
 Groq API (qwen/qwen3-32b) — grounded answer generation
```

- **Ingestion**: `parser.py` (PyMuPDF, pure text extraction) → `chunker.py` (paragraph/sentence-aware recursive splitting with overlap, pure) → `embeddings.py` (MiniLM, normalized vectors) → `vector_store.py` (FAISS `IndexFlatIP` + JSON metadata, persisted after every mutation). All orchestrated by `api/upload.py` — each service itself has no knowledge of HTTP, the registry, or the pipeline stage before/after it.
- **Retrieval**: `retrieval.py` embeds the question, searches the vector store, and filters every returned chunk against a similarity threshold — chunks below threshold are dropped before the LLM ever sees them.
- **Answer generation**: `llm.py` sends the question plus only the retrieved, thresholded chunks to Groq, with a system prompt that forbids outside knowledge and requires a fixed "not found" response when the answer isn't supported. Any chunk_id Groq claims to have used that wasn't actually retrieved is dropped rather than trusted.
- **Frontend**: three-tab layout (Documents, Ask Questions, Insights). Documents and Ask Questions are functional; Insights is currently a disabled placeholder (see "What I'd improve" below).

## Tech Stack

| Layer | Choice |
|---|---|
| Backend | FastAPI, Python 3.13 |
| PDF parsing | PyMuPDF |
| Chunking | Custom recursive splitter (no extra dependency) |
| Embeddings | sentence-transformers, `all-MiniLM-L6-v2` |
| Vector store | FAISS (`IndexFlatIP`), persisted to disk |
| LLM | Groq API, `qwen/qwen3-32b` |
| Frontend | React + Vite, plain CSS, axios |
| Validation | Pydantic |

## Setup

**Backend**
```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your GROQ_API_KEY
uvicorn app.main:app --reload --port 8001
```

**Frontend**
```bash
cd frontend
npm install
npm run dev   # ensure it serves on the port CORS in main.py allows (5175)
```

Required environment variable: `GROQ_API_KEY` (get one at console.groq.com — free tier is sufficient for this project's scale).

## API

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/upload` | Upload 1–50 PDFs |
| POST | `/query` | Ask a question, get an answer + citations |
| GET | `/documents` | List uploaded documents |
| DELETE | `/documents/{id}` | Delete a document and its vectors |
| GET | `/health` | Health check |

## Optimization choice: Accuracy

The assignment requires optimizing for either latency or accuracy. This project chooses **accuracy**, because a document-QA tool is only useful if its answers are trustworthy — a fast but hallucinated or uncited answer is worse than a slightly slower correct one. Concretely:
- Exact FAISS search (`IndexFlatIP`) rather than an approximate index — affordable at this corpus size (≤50 PDFs) and removes a source of retrieval error.
- Every retrieved chunk is filtered against a similarity threshold, not just the top match — weak matches never reach the LLM.
- The LLM is instructed to answer only from provided chunks and to explicitly say when it can't; citations it returns are cross-checked against what was actually retrieved, and unverifiable ones are silently dropped rather than shown.

## Key Design Decisions

- **Groq (`qwen/qwen3-32b`) instead of Claude/Anthropic** — no Anthropic API access was available for this build, so Groq was used for both retrieval-grounded answering and (planned) insights. The prompting strategy (strict grounding, fixed "not found" sentinel, citation cross-validation) is provider-agnostic and would carry over unchanged if swapped to Claude or another model later.
- **PyMuPDF over pypdf** — more reliable text extraction and page-level fidelity across real-world PDFs, which matters directly for citation accuracy.
- **FAISS over a managed vector DB** — at this scale (a handful of PDFs, low thousands of chunks), an in-process exact-search index is simpler to run and more accurate than an approximate index, with no extra service to manage.
- **`all-MiniLM-L6-v2` over a larger embedding model** — fast on CPU, a well-established baseline, and avoids adding another network dependency; noted as an accuracy lever for later if needed.
- **Custom chunker over LangChain's splitter** — avoids pulling in a heavy dependency for one utility; keeps full control over how page/document metadata is attached to each chunk, which is central to citation correctness.
- **Similarity threshold tuned from 0.35 → 0.25** after live testing showed natural-language questions score lower against MiniLM than keyword-style queries did — the initial threshold was rejecting valid matches.

## Tradeoffs

- **Exact FAISS search is O(n) per query** — fine at ≤50 documents, but would need an approximate index (HNSW/IVF) at larger scale.
- **`IndexFlatIP` has no in-place delete** — deleting a document rebuilds the index from surviving vectors. Cheap here, but O(total vectors) per delete, which wouldn't scale to a large, frequently-changing corpus.
- **In-memory document registry** — simple and fast for a single-process dev deployment, but not persisted independently of the vector store and not safe for multiple backend instances.
- **Groq instead of Claude** — trades away the model this project was originally planned around for one that's actually accessible without an Anthropic API key; the grounding/citation logic is designed to be model-agnostic so this can be swapped later.
- **No re-ranking step** — retrieval relies solely on embedding similarity; a cross-encoder re-ranker would likely improve precision at some latency cost, but wasn't implemented in the interest of time.

## What Would Break at Scale (10k+ documents)

- `IndexFlatIP` exact search would become too slow — needs an approximate index (HNSW/IVF) or a managed/sharded vector DB.
- Document deletion (currently a full index rebuild) would become prohibitively expensive — needs an index type supporting `remove_ids`, or a compaction strategy.
- The in-memory/JSON document registry would need to move to a real database (e.g., Postgres) for consistency and concurrent access.
- A single-process FastAPI server can't scale horizontally with a local-disk vector store — the vector store would need to move to shared/networked storage.
- Local file storage for raw PDFs would need to move to object storage (S3-equivalent).
- A single Groq call per query would need rate-limiting/queuing under concurrent load.

## What I'd Improve With More Time

- Cross-document insights (executive summary, risks, opportunities, missing information, next actions) — architecture is designed for this (separate `insights.py` service, same grounding approach) but not yet implemented.
- Automated tests (parser, chunker, retrieval threshold logic, API integration) with mocked LLM calls.
- OCR support for scanned/image-only PDFs.
- A cross-encoder re-ranking step before sending chunks to the LLM.
- Streaming answers to the frontend instead of waiting for the full response.
- An evaluation set (question → expected citation pairs) to regression-test retrieval quality as the pipeline evolves.