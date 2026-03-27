# config.py
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL        = "llama-3.3-70b-versatile"

DOCS_DIR          = "./docs"
FAISS_INDEX_PATH  = "./faiss.index"
BM25_PATH         = "./bm25.pkl"
CHUNKS_PATH       = "./chunks.pkl"
SOURCES_PATH      = "./sources.pkl"
EMBEDDER_PATH     = "./embedder_model"
EMBEDDER_NAME     = "all-MiniLM-L6-v2"

CHUNK_SIZE        = 500
CHUNK_OVERLAP     = 50
TOP_K             = 5
MAX_RETRIES       = 3
MAX_HISTORY_TURNS = 5

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set in .env file")