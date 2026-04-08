# Agentic Corrective RAG — Document Q&A

![CI](https://github.com/Hitan547/agentic-corrective-rag/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![LLM](https://img.shields.io/badge/LLM-LLaMA%203.3%2070B-orange)
![Framework](https://img.shields.io/badge/framework-LangGraph-green)

A production-aware document Q&A system that answers questions **only from your uploaded documents** — not from the model's imagination. Built with hybrid retrieval, cross-encoder reranking, and a self-correcting LangGraph agent that automatically retries if the answer isn't grounded in the source material.

---

## What It Does

Upload any PDF or TXT file, ask a question, and get an answer backed by:
- The exact source chunks it used
- A validation verdict (PASS/FAIL)
- How many self-correction retries were needed

---

## Architecture

```
PDF/TXT Upload
      │
      ▼
┌─────────────────────────────────┐
│         Ingestion Pipeline      │
│  PyMuPDF → Chunking → Embeddings│
│  FAISS Index + BM25 Index       │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│       Hybrid Retrieval          │
│  FAISS (dense) + BM25 (sparse)  │
│  → RRF Fusion                   │
│  → Cross-Encoder Reranking      │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│     Corrective RAG Agent        │
│  LangGraph StateGraph           │
│  Generate → Validate → Retry    │
│  (up to 3 automatic retries)    │
└─────────────────────────────────┘
      │
      ▼
  Streamlit UI + FastAPI Backend
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | LLaMA 3.3 70B via Groq API |
| Agent Framework | LangGraph (StateGraph) |
| Dense Retrieval | FAISS + `all-MiniLM-L6-v2` |
| Sparse Retrieval | BM25 (rank-bm25) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| PDF Parsing | PyMuPDF (fitz) |
| Backend | FastAPI |
| Frontend | Streamlit |
| Testing | pytest (unit + integration) |
| CI/CD | GitHub Actions |

---

## Key Features

- **Hybrid Search** — combines FAISS semantic search and BM25 keyword search, fused with Reciprocal Rank Fusion (RRF)
- **Cross-Encoder Reranking** — re-scores top candidates by reading query + chunk together for higher precision
- **Self-Correcting Agent** — LangGraph pipeline automatically detects hallucinations and retries up to 3 times
- **Hallucination Validation** — a second LLM call checks every answer against the source context before returning it
- **Session Memory** — remembers last 5 turns of conversation per session
- **Background Indexing** — document ingestion runs in the background without blocking the API
- **CI/CD** — unit tests run automatically on every push via GitHub Actions

---

## Project Structure

```
agentic-corrective-rag/
├── agent.py          # LangGraph corrective RAG agent
├── retriever.py      # Hybrid retrieval + RRF + reranking
├── ingestion.py      # PDF/TXT ingestion + FAISS/BM25 indexing
├── main.py           # FastAPI backend
├── app.py            # Streamlit frontend
├── config.py         # Configuration and constants
├── requirements.txt
├── tests/
│   ├── test_unit.py        # Unit tests (CI)
│   └── test_integration.py # Integration tests (local only)
└── .github/
    └── workflows/
        └── ci.yml          # GitHub Actions CI pipeline
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/Hitan547/agentic-corrective-rag.git
cd agentic-corrective-rag
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up environment**
```bash
# Create a .env file
echo "GROQ_API_KEY=your_key_here" > .env
```
Get your free API key at [console.groq.com](https://console.groq.com)

**4. Run the backend**
```bash
uvicorn main:app --reload --port 8080
```

**5. Run the frontend** (new terminal)
```bash
streamlit run app.py
```

**6. Open** `http://localhost:8501`, upload a PDF, click **Index documents**, then ask questions.

---

## Running Tests

```bash
# Unit tests (fast, no API needed)
python -m pytest tests/test_unit.py -v

# Integration tests (requires GROQ_API_KEY)
python -m pytest tests/test_integration.py -v -m integration
```

---

## How the Agent Works

1. **Generate** — LLaMA 3.3 70B answers using only the retrieved chunks
2. **Validate** — a second LLM call checks if every claim is supported by the context
3. **Retry** — if validation fails, the agent retries with the failure reason as feedback
4. **Stop** — returns the answer after PASS or after 3 retries

---

## Author

**Hitan K** — Final-year CS undergraduate (AI specialization)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-hitan--k-blue)](https://linkedin.com/in/hitan-k)
[![GitHub](https://img.shields.io/badge/GitHub-Hitan547-black)](https://github.com/Hitan547)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Hitan2004-yellow)](https://huggingface.co/Hitan2004)
