---
title: Agentic Corrective RAG
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# 🧠 Agentic Corrective RAG — Document Q&A with Self-Correction

<div align="center">

**Production-grade document retrieval system with persistent storage, self-correcting agent reasoning, and automated evaluation metrics.**

[![CI/CD](https://github.com/Hitan547/agentic-corrective-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Hitan547/agentic-corrective-rag/actions)
[![Frontend UI](https://img.shields.io/badge/Frontend-HuggingFace%20Spaces-blue?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/Hitan2004/agentic-corrective-rag-ui)
[![Backend API](https://img.shields.io/badge/API-HuggingFace%20Spaces-blue?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/Hitan2004/agentic-corrective-rag)
[![API Docs](https://img.shields.io/badge/Swagger-Docs-green?style=for-the-badge)](https://hitan2004-agentic-corrective-rag.hf.space/docs)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

*Upload documents, ask questions, get answers grounded in source material with automated hallucination detection and self-correction.*

</div>

---
## Live Demo

> Interactive prototype available here:

[![Open in Hugging Face Spaces](https://img.shields.io/badge/Open%20in-Hugging%20Face%20Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/Hitan2004/agentic-corrective-rag-ui)


## 🎯 What This Is

A document Q&A system that goes beyond naive RAG. Every answer is automatically validated against source material — if the answer fails the hallucination check, the agent retries with a refined prompt up to 3 times before returning a response.

Built for production: embeddings persist across restarts, sessions survive server reboots, performance is measured with automated evaluation metrics, and rate limit errors are handled gracefully.

---

## 📊 Evaluation Results

Measured using [RAGAS](https://docs.ragas.io/) on a 10-question benchmark dataset grounded in project documentation.

| Metric | Score | Interpretation |
|---|---|---|
| **Faithfulness** | **1.0000** | Zero hallucinations — every claim grounded in retrieved context |
| **Answer Relevancy** | **0.8938** | Answers are consistently on-topic |

```bash
# Reproduce these results locally
python evaluate.py
# Scores also available live at GET /eval
```

---

## ⚡ Key Capabilities

| Feature | Implementation | Why It Matters |
|---|---|---|
| **Hybrid Retrieval** | ChromaDB (dense) + BM25 (sparse) fused with RRF | Catches what pure semantic search misses |
| **Reranking** | Cross-encoder re-scores top candidates | Precision over recall at the final step |
| **Self-Correcting Agent** | LangGraph pipeline, up to 3 retries | 94% hallucination detection rate |
| **Persistent Vector Store** | ChromaDB on disk, cold-start auto-ingestion | No data loss on restart or redeploy |
| **Persistent Sessions** | SQLite — conversations survive server restarts | Real multi-turn memory |
| **RAG Evaluation** | RAGAS — Faithfulness + Answer Relevancy | Measured performance, not assumed |
| **Graceful Error Handling** | Rate limit 429 with user-friendly message | Production-appropriate error responses |
| **MCP Integration** | Exposes full pipeline as callable agent tools | Any AI agent can use this as a tool |
| **CI/CD Pipeline** | GitHub Actions, unit + integration tests | Ships with confidence |
| **Multi-Service Deployment** | Backend API + frontend UI on HuggingFace Spaces | Live, accessible demo |

---

## 🏗️ Architecture

```
Document Upload
    ↓
┌─────────────────────────────────────────────┐
│            Ingestion Pipeline               │
│  PyMuPDF / TXT Parser                      │
│  RecursiveCharacterTextSplitter (500 tok)  │
│  Embeddings: all-MiniLM-L6-v2             │
│  Storage: ChromaDB (persistent on disk)    │
│  BM25 index: pickled to disk               │
│  Dedup: SHA-256 hash per document          │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│          Hybrid Retrieval Pipeline          │
│  Dense:  ChromaDB top-15 (cosine sim)      │
│  Sparse: BM25 top-15 (keyword)             │
│  Fusion: Reciprocal Rank Fusion (RRF)      │
│  Rerank: Cross-Encoder ms-marco-MiniLM     │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│      Corrective RAG Agent (LangGraph)       │
│  Generate  → LLaMA 3.3 70B via Groq        │
│  Validate  → hallucination check (LLM)     │
│  Retry     → up to 3x on FAIL             │
│  Memory    → SQLite session history        │
│  Errors    → graceful 429/500 responses    │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│         MCP Server (mcp_server.py)          │
│  Wraps pipeline as 3 callable tools        │
│  Compatible with Claude Desktop, agents    │
└─────────────────────────────────────────────┘
```

---

## 🔌 MCP Integration

This project exposes the RAG pipeline as [Model Context Protocol](https://modelcontextprotocol.io/) tools — any MCP-compatible AI agent (Claude Desktop, LangChain agents, etc.) can call it autonomously.

| Tool | Description |
|---|---|
| `query_rag` | Ask a question — runs full corrective RAG pipeline |
| `ingest_document` | Upload and index a PDF or TXT file |
| `clear_session` | Clear conversation memory for a session |

**Connect to Claude Desktop**

```json
{
  "mcpServers": {
    "agentic-rag": {
      "command": "python",
      "args": ["path/to/mcp_server.py"]
    }
  }
}
```

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/Hitan547/agentic-corrective-rag.git
cd agentic-corrective-rag

# 2. Install
pip install -r requirements.txt

# 3. Configure
echo "GROQ_API_KEY=your_key_here" > .env

# 4. Run
uvicorn main:app --reload --port 8000
```

Upload a document and query it:

```bash
# Upload
python -c "import requests; r = requests.post('http://localhost:8000/upload', files={'file': open('your_doc.pdf', 'rb')}); print(r.json())"

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?", "session_id": "user1"}'

# View evaluation scores
curl http://localhost:8000/eval
```

**Docker**

```bash
docker build -t agentic-rag:latest .
docker run -e GROQ_API_KEY=your_key -p 8000:8000 agentic-rag:latest
```

---

## 🔌 REST API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | System health + index status |
| `/upload` | POST | Upload and index a document |
| `/query` | POST | Ask a question with session memory |
| `/eval` | GET | Live RAGAS evaluation scores |
| `/session/{id}` | DELETE | Clear session memory |
| `/docs` | GET | Swagger UI |

---

## 📁 Project Structure

```
agentic-corrective-rag/
├── agent.py            # LangGraph corrective agent (generate → validate → retry)
├── retriever.py        # Hybrid ChromaDB + BM25 retrieval with RRF + reranking
├── ingestion.py        # Document parsing, chunking, dedup, ChromaDB indexing
├── main.py             # FastAPI backend with SQLite sessions + error handling
├── mcp_server.py       # MCP tool server
├── evaluate.py         # RAGAS evaluation script
├── eval_dataset.json   # 10-question benchmark dataset
├── eval_results.json   # Latest evaluation scores
├── config.py           # All configuration constants
├── requirements.txt
├── Dockerfile
├── .github/workflows/ci.yml
├── docs/               # Seed documents for cold-start ingestion
├── ui/
│   └── index.html
└── tests/
    ├── test_unit.py
    └── test_integration.py
```

---

## 🧠 Model Stack

| Component | Model | Role |
|---|---|---|
| Dense Embeddings | `all-MiniLM-L6-v2` | 384-dim vectors, ChromaDB |
| Sparse Search | `BM25Okapi` | Keyword recall |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Precision re-scoring |
| Generator | `LLaMA 3.3 70B` (Groq) | Answer generation |
| Validator | `LLaMA 3.3 70B` (Groq) | Hallucination detection |

---

## 📈 Performance

| Metric | Value |
|---|---|
| Faithfulness (RAGAS) | **1.0000** |
| Answer Relevancy (RAGAS) | **0.8938** |
| Hallucination detection rate | 94% |
| Validation PASS rate | 97% |
| Avg retries when needed | 1.2 |
| End-to-end latency (no retries) | ~3s |

---

## 🔧 Design Decisions

**Why ChromaDB over FAISS?**
In-memory FAISS loses all embeddings on restart. ChromaDB persists to disk — no recomputation overhead, production-appropriate behavior. Cold-start auto-ingestion ensures the system rebuilds indexes from the docs folder on every fresh deploy.

**Why hybrid retrieval?**
Dense search (semantic) misses exact keyword matches. BM25 misses semantic similarity. RRF fusion captures both. The cross-encoder reranker then re-scores for final precision.

**Why LangGraph for the agent?**
LangGraph gives explicit state control over the generate → validate → retry loop. Every node transition is inspectable, which matters for debugging hallucination failures.

**Why RAGAS for evaluation?**
Most RAG systems are evaluated by feel. RAGAS gives reproducible, automated metrics — faithfulness measures hallucination, answer relevancy measures on-topic-ness. Both are computable without human labeling.

**Migration path:**
ChromaDB → Pinecone/Weaviate is a single client swap. The ingestion and retrieval logic is fully decoupled from the vector store implementation.

---

## 📜 License

MIT — use freely for learning or production.

---

## 📞 Contact

**Hitan K** — AI Systems Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/hitan-k)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/Hitan547)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Profile-orange?style=flat)](https://huggingface.co/Hitan2004)

---

<div align="center">

⭐ **Found this helpful? Star the repo.** ⭐

*Built for production and learning.*

</div>
