from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import documents, query, upload
from app.services.embeddings import get_model
from app.services.llm import validate_groq_configuration
from app.services.vector_store import vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_groq_configuration()
    get_model()
    vector_store.load()
    yield


app = FastAPI(title="Document Intelligence API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5175"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(documents.router)
app.include_router(query.router)


@app.get("/health")
def health():
    return {"status": "ok"}
