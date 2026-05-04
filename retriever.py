import os, pickle
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from config import (
    CHROMA_PATH, CHROMA_COLLECTION,
    BM25_PATH, EMBEDDER_NAME, RERANKER_MODEL
)

# ── module-level singletons ───────────────────────────
_collection  = None
_bm25_index  = None
_chunks      = None
_sources     = None
_model       = None
_reranker    = None

def indexes_loaded() -> bool:
    return _collection is not None

def load_indexes():
    global _collection, _bm25_index, _chunks, _sources, _model, _reranker

    if not os.path.exists(BM25_PATH):
        print("WARNING: No BM25 index found. Upload documents first.")
        return

    # ChromaDB — loads from disk automatically
    client      = chromadb.PersistentClient(path=CHROMA_PATH)
    _collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    # BM25 + chunk/source lists (stored together in one pickle)
    with open(BM25_PATH, "rb") as f:
        data = pickle.load(f)
    _bm25_index = data["bm25"]
    _chunks     = data["chunks"]
    _sources    = data["sources"]

    _model    = SentenceTransformer(EMBEDDER_NAME)
    _reranker = CrossEncoder(RERANKER_MODEL)
    print(f"Indexes loaded: {_collection.count()} vectors, {len(_chunks)} chunks")

def reload_indexes():
    global _collection, _bm25_index, _chunks, _sources, _model, _reranker
    _collection = _bm25_index = _chunks = _sources = _model = _reranker = None
    load_indexes()

# ── RRF fusion ────────────────────────────────────────

def _reciprocal_rank_fusion(lists: list, k: int = 60) -> dict:
    scores: dict = {}
    for ranked_list in lists:
        for rank, doc_id in enumerate(ranked_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores

# ── main retrieval ────────────────────────────────────

def hybrid_retrieve(query: str, top_k: int = 5) -> list:
    if not indexes_loaded():
        raise RuntimeError("Indexes not loaded. Call load_indexes() first.")

    # ── Dense retrieval via ChromaDB ──
    q_emb = _model.encode([query]).tolist()
    chroma_results = _collection.query(
        query_embeddings=q_emb,
        n_results=min(top_k * 3, _collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    # Map returned chunk text → index in _chunks for RRF
    chunk_to_idx = {c: i for i, c in enumerate(_chunks)}
    dense_ranking = [
        chunk_to_idx[doc]
        for doc in chroma_results["documents"][0]
        if doc in chunk_to_idx
    ]

    # ── Sparse retrieval via BM25 ──
    bm25_scores    = _bm25_index.get_scores(query.lower().split())
    sparse_ranking = np.argsort(bm25_scores)[::-1][: top_k * 3].tolist()

    # ── RRF fusion ──
    rrf_scores = _reciprocal_rank_fusion([dense_ranking, sparse_ranking])
    fused_ids  = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[: top_k * 2]

    # ── Cross-encoder reranking ──
    candidates = [(query, _chunks[i]) for i in fused_ids]
    ce_scores  = _reranker.predict(candidates)
    ranked = sorted(
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
            "ce_score":  round(float(score), 4),
        }
        for i, score in ranked
    ]
