import os, shutil, sqlite3, json
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from retriever import load_indexes, reload_indexes, hybrid_retrieve, indexes_loaded as _indexes_loaded
from agent import run_rag_agent
from ingestion import run_ingestion
from config import DOCS_DIR, TOP_K, MAX_HISTORY_TURNS, SQLITE_PATH

# ── SQLite session memory ─────────────────────────────

def _init_db():
    con = sqlite3.connect(SQLITE_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            history    TEXT NOT NULL DEFAULT '[]'
        )
    """)
    con.commit()
    con.close()

def _load_history(session_id: str) -> list:
    con = sqlite3.connect(SQLITE_PATH)
    row = con.execute(
        "SELECT history FROM sessions WHERE session_id=?", (session_id,)
    ).fetchone()
    con.close()
    if not row:
        return []
    raw = json.loads(row[0])
    # Reconstruct LangChain message objects
    msgs = []
    for m in raw:
        if m["role"] == "human":
            msgs.append(HumanMessage(content=m["content"]))
        else:
            msgs.append(AIMessage(content=m["content"]))
    return msgs

def _save_history(session_id: str, history: list):
    raw = [
        {"role": "human" if isinstance(m, HumanMessage) else "ai",
         "content": m.content}
        for m in history
    ]
    con = sqlite3.connect(SQLITE_PATH)
    con.execute(
        "INSERT OR REPLACE INTO sessions (session_id, history) VALUES (?,?)",
        (session_id, json.dumps(raw))
    )
    con.commit()
    con.close()

# ── app lifecycle ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_db()
    load_indexes()
    if not _indexes_loaded():
        from pathlib import Path
        docs_path = Path(DOCS_DIR)
        has_docs = any(docs_path.glob("*.txt")) or any(docs_path.glob("*.pdf"))
        if has_docs:
            print("Cold start: ChromaDB empty, re-indexing docs folder...")
            try:
                run_ingestion()
                reload_indexes()
                print("Cold start ingestion complete.")
            except Exception as e:
                print(f"Cold start ingestion failed: {e}")
        else:
            print("WARNING: No indexes and no docs found. Upload documents first.")
    yield
app = FastAPI(title="Corrective RAG API", version="1.0", lifespan=lifespan)

# ── models ────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:   str
    session_id: str = "default"
    top_k:      int = TOP_K

class QueryResponse(BaseModel):
    answer:       str
    sources:      list
    retries_used: int
    validation:   str
    session_id:   str

# ── routes ────────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "RAG API running 🚀"}

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not _indexes_loaded():
        try: load_indexes()
        except: pass
    if not _indexes_loaded():
        raise HTTPException(503, detail="Indexes not ready. Upload documents first.")

    results = hybrid_retrieve(req.question, top_k=req.top_k)
    if not results:
        raise HTTPException(404, detail="No relevant chunks found.")

    history = _load_history(req.session_id)
    try:
        answer, retries, verdict = run_rag_agent(req.question, results, history)
    except Exception as e:
        if "429" in str(e) or "rate_limit" in str(e).lower() or "rate limit" in str(e).lower():
            raise HTTPException(
                status_code=429,
                detail="Rate limit reached. Please wait 30 seconds and try again."
            )
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    history.append(HumanMessage(content=req.question))
    history.append(AIMessage(content=answer))
    _save_history(req.session_id, history[-(MAX_HISTORY_TURNS * 2):])

    return QueryResponse(
        answer=answer,
        sources=[{"chunk": r["chunk"][:300], "source": r["source"]} for r in results],
        retries_used=retries,
        validation=verdict,
        session_id=req.session_id,
    )

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    allowed = {".txt", ".pdf"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(400, detail="Only .txt and .pdf allowed.")
    os.makedirs(DOCS_DIR, exist_ok=True)
    dest = os.path.join(DOCS_DIR, file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    _reindex()
    return {"status": "uploaded", "filename": file.filename}

def _reindex():
    try:
        run_ingestion()
        reload_indexes()
        print(f"Re-indexing complete. Loaded: {_indexes_loaded()}")
    except Exception as e:
        import traceback
        print(f"Re-indexing failed: {e}"); traceback.print_exc()

@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    con = sqlite3.connect(SQLITE_PATH)
    con.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
    con.commit(); con.close()
    return {"status": "cleared", "session_id": session_id}

@app.get("/health")
def health():
    return {"status": "ok", "indexes_loaded": _indexes_loaded()}
@app.get("/eval")
def get_eval():
    if not os.path.exists("eval_results.json"):
        raise HTTPException(status_code=404, detail="Run evaluate.py first to generate scores.")
    with open("eval_results.json", "r") as f:
        return json.load(f)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
