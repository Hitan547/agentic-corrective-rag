import os
import warnings
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    warnings.warn("GROQ_API_KEY not set — LLM calls will fail")

_BASE = os.path.dirname(os.path.abspath(__file__))

GROQ_MODEL    = "llama-3.3-70b-versatile"
DOCS_DIR      = os.path.join(_BASE, "docs")

# ── ChromaDB (replaces FAISS) ──────────────────────────
CHROMA_PATH        = os.path.join(_BASE, "chroma_db")
CHROMA_COLLECTION  = "rag_docs"

# ── BM25 (still persisted with pickle) ────────────────
BM25_PATH     = os.path.join(_BASE, "bm25.pkl")

# ── SQLite session memory (replaces in-memory dict) ───
SQLITE_PATH   = os.path.join(_BASE, "sessions.db")

# ── Model names ───────────────────────────────────────
EMBEDDER_NAME  = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Chunking ──────────────────────────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50

# ── Retrieval ─────────────────────────────────────────
TOP_K             = 5
MAX_RETRIES       = 3
MAX_HISTORY_TURNS = 5
