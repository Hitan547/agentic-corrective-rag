# tests/test_integration.py
# Run with:  pytest tests/test_integration.py -v -m integration
# These call real APIs — don't run in CI automatically.

import pytest

pytestmark = pytest.mark.integration   # tag so CI can skip these


def test_groq_connection_live():
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage
    from config import GROQ_API_KEY, GROQ_MODEL
    llm = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=GROQ_API_KEY)
    r   = llm.invoke([HumanMessage(content="Reply with just the word OK")])
    assert len(r.content) > 0


def test_full_pipeline_live():
    """Ingests a tiny doc, retrieves, runs agent — end to end."""
    import os
    from pathlib import Path

    # Write test doc
    Path("./docs").mkdir(exist_ok=True)
    test_file = Path("./docs/_pytest_temp.txt")
    test_file.write_text(
        "The Eiffel Tower is in Paris, France. "
        "It was built in 1889. It is 330 metres tall."
    )

    try:
        from ingestion import run_ingestion
        from retriever import load_indexes, hybrid_retrieve
        from agent import run_rag_agent

        run_ingestion()
        load_indexes()

        results = hybrid_retrieve("How tall is the Eiffel Tower?", top_k=3)
        assert len(results) > 0
        assert "ce_score" in results[0]          # reranker ran

        answer, retries, verdict = run_rag_agent(
            "How tall is the Eiffel Tower?", results
        )
        assert "330" in answer or "metres" in answer.lower()
        assert verdict in {"PASS", "FAIL"}

    finally:
        test_file.unlink(missing_ok=True)        # always clean up