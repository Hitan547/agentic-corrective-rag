# tests/test_unit.py
import pytest

# ── RRF logic ─────────────────────────────────────────────────────────────────

def test_rrf_prefers_doc_appearing_in_both_lists():
    from retriever import _reciprocal_rank_fusion
    scores = _reciprocal_rank_fusion([[0, 1, 2], [2, 0, 1]])
    # doc 2 is rank-0 in sparse and rank-2 in dense → should beat doc 1
    assert scores[2] > scores[1]

def test_rrf_returns_all_docs():
    from retriever import _reciprocal_rank_fusion
    scores = _reciprocal_rank_fusion([[0, 1], [1, 2]])
    assert set(scores.keys()) == {0, 1, 2}

def test_rrf_scores_are_positive():
    from retriever import _reciprocal_rank_fusion
    scores = _reciprocal_rank_fusion([[0, 1, 2]])
    assert all(v > 0 for v in scores.values())

# ── Config sanity ─────────────────────────────────────────────────────────────

def test_config_values_are_sane():
    from config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, MAX_RETRIES
    assert CHUNK_SIZE > CHUNK_OVERLAP,  "overlap must be smaller than chunk size"
    assert TOP_K > 0,                   "TOP_K must be positive"
    assert MAX_RETRIES >= 1,            "need at least 1 retry"

def test_groq_api_key_present(monkeypatch):
    # patch so we don't need a real key in CI
    monkeypatch.setenv("GROQ_API_KEY", "gsk_fakekeyfortesting1234567890")
    import importlib, config
    importlib.reload(config)             # re-reads env
    assert len(config.GROQ_API_KEY) > 10

# ── Agent routing logic ───────────────────────────────────────────────────────

def test_route_returns_done_on_pass():
    from agent import route_after_validation
    state = {"validation_result": "PASS", "retry_count": 0}
    assert route_after_validation(state) == "done"

def test_route_returns_retry_on_fail_within_limit():
    from agent import route_after_validation
    state = {"validation_result": "FAIL", "retry_count": 0}
    assert route_after_validation(state) == "retry"

def test_route_returns_done_when_retries_exhausted():
    from agent import route_after_validation
    state = {"validation_result": "FAIL", "retry_count": 3}
    assert route_after_validation(state) == "done"

def test_increment_retry_node():
    from agent import increment_retry_node
    result = increment_retry_node({"retry_count": 1})
    assert result["retry_count"] == 2

# ── Retriever output shape (mocked indexes) ───────────────────────────────────

@pytest.fixture
def mock_indexes(monkeypatch):
    """Patches all globals in retriever so no files need to exist."""
    import numpy as np
    import retriever

    # Fake chunks and sources
    fake_chunks  = ["Paris is in France.", "Tower is 330m tall.", "Built in 1889."]
    fake_sources = ["doc1.txt", "doc1.txt", "doc1.txt"]

    # Fake FAISS index that always returns ids [0, 1, 2]
    class FakeFaiss:
        ntotal = 3
        def search(self, vec, k):
            ids = np.array([[0, 1, 2]])
            return None, ids

    # Fake BM25 that returns uniform scores
    class FakeBM25:
        def get_scores(self, tokens):
            return np.array([0.9, 0.5, 0.3])

    # Fake embedder
    class FakeModel:
        def encode(self, texts, convert_to_numpy=True):
            return np.random.rand(len(texts), 384).astype("float32")

    # Fake cross-encoder
    class FakeReranker:
        def predict(self, pairs):
            return np.array([0.9, 0.7, 0.5][: len(pairs)])

    monkeypatch.setattr(retriever, "_faiss_index", FakeFaiss())
    monkeypatch.setattr(retriever, "_bm25_index",  FakeBM25())
    monkeypatch.setattr(retriever, "_chunks",      fake_chunks)
    monkeypatch.setattr(retriever, "_sources",     fake_sources)
    monkeypatch.setattr(retriever, "_model",       FakeModel())
    monkeypatch.setattr(retriever, "_reranker",    FakeReranker())
    return fake_chunks


def test_hybrid_retrieve_returns_top_k(mock_indexes):
    from retriever import hybrid_retrieve
    results = hybrid_retrieve("Where is Paris?", top_k=2)
    assert len(results) == 2

def test_hybrid_retrieve_result_has_required_keys(mock_indexes):
    from retriever import hybrid_retrieve
    result = hybrid_retrieve("Where is Paris?", top_k=1)[0]
    assert "chunk"     in result
    assert "source"    in result
    assert "rrf_score" in result
    assert "ce_score"  in result

def test_hybrid_retrieve_scores_are_floats(mock_indexes):
    from retriever import hybrid_retrieve
    result = hybrid_retrieve("test", top_k=1)[0]
    assert isinstance(result["rrf_score"], float)
    assert isinstance(result["ce_score"],  float)