import os
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from retriever import load_indexes, reload_indexes, hybrid_retrieve, indexes_loaded as _indexes_loaded
from agent import run_rag_agent
from ingestion import run_ingestion
from config import DOCS_DIR, TOP_K, MAX_HISTORY_TURNS

sessions: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_indexes()
    except FileNotFoundError:
        print("WARNING: No indexes found. Upload documents first.")
    yield


app = FastAPI(title="Corrective RAG API", version="1.0", lifespan=lifespan)


@app.get("/")
def home():
    return {"message": "RAG API running 🚀"}


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


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not _indexes_loaded():
        try:
            load_indexes()
        except Exception:
            pass
    if not _indexes_loaded():
        raise HTTPException(
            status_code=503,
            detail="Indexes not ready. Upload and index documents first."
        )
    results = hybrid_retrieve(req.question, top_k=req.top_k)
        )

    results = hybrid_retrieve(req.question, top_k=req.top_k)
    if not results:
        raise HTTPException(status_code=404, detail="No relevant chunks found.")

    history = sessions.get(req.session_id, [])
    answer, retries, verdict = run_rag_agent(req.question, results, history)

    history.append(HumanMessage(content=req.question))
    history.append(AIMessage(content=answer))
    sessions[req.session_id] = history[-(MAX_HISTORY_TURNS * 2):]

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
        raise HTTPException(status_code=400, detail="Only .txt and .pdf files allowed.")

    os.makedirs(DOCS_DIR, exist_ok=True)
    dest = os.path.join(DOCS_DIR, file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    _reindex()
    return {"status": "uploaded", "filename": file.filename,
            "message": "Indexing complete."}


def _reindex():
    try:
        run_ingestion()
        reload_indexes()
        print("Re-indexing complete.")
    except Exception as e:
        print(f"Re-indexing failed: {e}")


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


@app.get("/health")
def health():
    return {"status": "ok", "indexes_loaded": _indexes_loaded()}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
