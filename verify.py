# verify.py  — tests each component individually
import sys

def check(label, fn):
    try:
        fn()
        print(f"  PASS  {label}")
    except Exception as e:
        print(f"  FAIL  {label}: {e}")
        sys.exit(1)

print("\n=== Corrective RAG — environment check ===\n")

# 1. Config / API key
def test_config():
    from config import GROQ_API_KEY
    assert len(GROQ_API_KEY) > 10, "GROQ_API_KEY looks invalid"
check("Config + GROQ key loaded", test_config)

# 2. Groq connection
def test_groq():
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage
    from config import GROQ_API_KEY, GROQ_MODEL
    llm = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=GROQ_API_KEY)
    r = llm.invoke([HumanMessage(content="Say OK")])
    assert "ok" in r.content.lower() or len(r.content) > 0
check("Groq API connection", test_groq)

# 3. Ingestion
def test_ingestion():
    import os
    from pathlib import Path
    Path("./docs").mkdir(exist_ok=True)
    test_file = "./docs/_verify_test.txt"
    Path(test_file).write_text(
        "The Eiffel Tower is in Paris, France. "
        "It was built in 1889 for the World's Fair. "
        "It is 330 metres tall."
    )
    from ingestion import run_ingestion
    run_ingestion()
    os.remove(test_file)
check("Ingestion pipeline", test_ingestion)

# 4. Retriever
def test_retriever():
    from retriever import load_indexes, hybrid_retrieve
    load_indexes()
    results = hybrid_retrieve("Where is the Eiffel Tower?", top_k=3)
    assert len(results) > 0
    assert "chunk" in results[0]
    assert "source" in results[0]
check("Hybrid retrieval (BM25 + FAISS)", test_retriever)

# 5. Agent
def test_agent():
    from retriever import hybrid_retrieve
    from agent import run_rag_agent
    results = hybrid_retrieve("How tall is the Eiffel Tower?", top_k=3)
    answer, retries, verdict = run_rag_agent(
        "How tall is the Eiffel Tower?", results
    )
    assert len(answer) > 10, f"Answer too short: {answer}"
    print(f"\n  Answer:   {answer[:120]}")
    print(f"  Retries:  {retries}")
    print(f"  Verdict:  {verdict}")
check("LangGraph agent (generate + validate)", test_agent)

print("\n=== All checks passed — ready to run ===\n")
