import os
from pathlib import Path

from dotenv import load_dotenv

from app.utils.logging import configure_logging

load_dotenv()
configure_logging(os.getenv("LOG_LEVEL", "INFO"))

BACKEND_DIR = Path(__file__).resolve().parents[2]

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(BACKEND_DIR / "uploads")))
ALLOWED_UPLOAD_EXTENSIONS = {".pdf"}
MAX_FILES_PER_UPLOAD = 50
MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_FILE_SIZE_BYTES", 50 * 1024 * 1024))

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")
GROQ_TIMEOUT_SECONDS = float(os.getenv("GROQ_TIMEOUT_SECONDS", "30"))
