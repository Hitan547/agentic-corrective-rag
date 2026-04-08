# config.py
import os
import warnings
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    warnings.warn("GROQ_API_KEY not set — LLM calls will fail")

# ── Anchor all paths to the directory this file lives in ──
_BASE = os.path.dirname(os.path.abspath(__file__))

GROQ_MODEL        = "llama-3.3-70b-versatile"
DOCS_DIR          = os.path.join(_BASE, "docs")
FAISS_INDEX_PATH  = os.path.join(_BASE, "faiss.index")
BM25_PATH         = os.path.join(_BASE, "bm25.pkl")
CHUNKS_PATH       = os.path.join(_BASE, "chunks.pkl")
SOURCES_PATH      = os.path.join(_BASE, "sources.pkl")
EMBEDDER_NAME  = os.path.join(_BASE, "models/embedder")
RERANKER_MODEL = os.path.join(_BASE, "models/reranker")
CHUNK_SIZE        = 500
CHUNK_OVERLAP     = 50
TOP_K             = 5
MAX_RETRIES       = 3
MAX_HISTORY_TURNS = 5
