import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from config import (
    FAISS_INDEX_PATH, BM25_PATH, CHUNKS_PATH,
    SOURCES_PATH, EMBEDDER_PATH
)

_faiss_index    = None
_bm25_index     = None
_chunks         = None
_sources        = None
_model          = None
_reranker       = None

RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def indexes_loaded() -> bool:
    return _faiss_index is not None


def load_indexes():
    global _faiss_index, _bm25_index, _chunks, _sources, _model, _reranker
    _faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    with open(BM25_PATH,    "rb") as f: _bm25_index = pickle.load(f)
    with open(CHUNKS_PATH,  "rb") as f: _chunks     = pickle.load(f)
    with open(SOURCES_PATH, "rb") as f: _sources    = pickle.load(f)
    _model    = SentenceTransformer(EMBEDDER_PATH)
    _reranker = CrossEncoder(RERANKER_MODEL)          # ← reranker loads once
    print(f"Indexes loaded: {_faiss_index.ntotal} vectors, {len(_chunks)} chunks")


def reload_indexes():
    global _faiss_index, _bm25_index, _chunks, _sources, _model, _reranker
    _faiss_index = _bm25_index = _chunks = _sources = _model = _reranker = None
    load_indexes()


def _reciprocal_rank_fusion(lists: list, k: int = 60) -> dict:
    """Returns {doc_id: rrf_score} instead of just a sorted list."""
    scores: dict = {}
    for ranked_list in lists:
        for rank, doc_id in enumerate(ranked_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores                                      # ← return the dict now


def hybrid_retrieve(query: str, top_k: int = 5) -> list:
    if not indexes_loaded():
        raise RuntimeError("Indexes not loaded. Call load_indexes() first.")

    # ── Dense retrieval (FAISS) ───────────────────────────────
    q_emb = _model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    _, dense_ids   = _faiss_index.search(q_emb, top_k * 3)
    dense_ranking  = [int(i) for i in dense_ids[0] if i >= 0]

    # ── Sparse retrieval (BM25) ───────────────────────────────
    bm25_scores    = _bm25_index.get_scores(query.lower().split())
    sparse_ranking = np.argsort(bm25_scores)[::-1][: top_k * 3].tolist()

    # ── Fusion (RRF) — now returns score dict ─────────────────
    rrf_scores     = _reciprocal_rank_fusion([dense_ranking, sparse_ranking])
    fused_ids      = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[: top_k * 2]

    # ── Cross-encoder reranking ───────────────────────────────
    # The cross-encoder scores each (query, chunk) pair together
    # much more accurately than embedding similarity alone
    candidates     = [(query, _chunks[i]) for i in fused_ids]
    ce_scores      = _reranker.predict(candidates)     # shape: (len(candidates),)

    # Sort by cross-encoder score, keep top_k
    ranked         = sorted(
        zip(fused_ids, ce_scores),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    return [
        {
            "chunk":     _chunks[i],
            "source":    _sources[i],
            "chunk_id":  i,
            "rrf_score": round(float(rrf_scores[i]), 4),
            "ce_score":  round(float(score), 4),       # ← reranker confidence
        }
        for i, score in ranked
    ]