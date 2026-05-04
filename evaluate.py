"""
evaluate.py — RAGAS evaluation for Agentic Corrective RAG
Run: python evaluate.py
Output: eval_results.json
"""

import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from retriever import load_indexes, hybrid_retrieve
from agent import run_rag_agent
from config import TOP_K, GROQ_API_KEY, GROQ_MODEL

# ── Step 1: Load indexes ──────────────────────────────
print("Loading indexes...")
load_indexes()
print("Indexes ready.\n")

# ── Step 2: Load eval dataset ─────────────────────────
with open("eval_dataset.json", "r") as f:
    eval_data = json.load(f)[:5]

print(f"Loaded {len(eval_data)} questions.\n")

# ── Step 3: Run pipeline on each question ─────────────
results = []

for i, item in enumerate(eval_data):
    question     = item["question"]
    ground_truth = item["ground_truth"]

    print(f"[{i+1}/{len(eval_data)}] {question}")

    chunks = hybrid_retrieve(question, top_k=TOP_K)
    answer, retries, verdict = run_rag_agent(question, chunks)
    contexts = [c["chunk"] for c in chunks]

    print(f"  → verdict: {verdict} | retries: {retries}")
    print(f"  → answer: {answer[:80]}...\n")

    results.append({
        "question":     question,
        "answer":       answer,
        "contexts":     contexts,
        "ground_truth": ground_truth,
    })

# ── Step 4: Convert to HuggingFace Dataset ────────────
dataset = Dataset.from_list(results)

# ── Step 5: Configure RAGAS to use Groq + local embeddings ──
groq_llm = LangchainLLMWrapper(
    ChatGroq(model=GROQ_MODEL, temperature=0, api_key=GROQ_API_KEY)
)

# Local embeddings — no OpenAI needed, same model already in your project
hf_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

faith_metric = Faithfulness(llm=groq_llm)
rel_metric   = AnswerRelevancy(llm=groq_llm, embeddings=hf_embeddings)

print("Running RAGAS evaluation...")
print("(This makes LLM calls — takes ~1-2 minutes)\n")

score = evaluate(dataset, metrics=[faith_metric, rel_metric])

# ── Step 6: Print + save results ──────────────────────
scores_df = score.to_pandas()
faith = float(scores_df["faithfulness"].mean())
rel   = float(scores_df["answer_relevancy"].mean())

print("\n=== RAGAS SCORES ===")
print(f"  Faithfulness:     {faith:.4f}")
print(f"  Answer Relevancy: {rel:.4f}")

output = {
    "faithfulness":     round(faith, 4),
    "answer_relevancy": round(rel, 4),
    "num_questions":    len(eval_data),
}

with open("eval_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nSaved to eval_results.json")
print("\n=== DIAGNOSIS ===")

if faith < 0.80:
    print("  Faithfulness low -> generation problem")
elif faith >= 0.90:
    print("  Faithfulness strong -> hallucination well controlled")
else:
    print("  Faithfulness acceptable -> monitor on larger dataset")

if rel < 0.80:
    print("  Answer relevancy low -> retrieval or prompt problem")
elif rel >= 0.90:
    print("  Answer relevancy strong -> answers are on-topic")
else:
    print("  Answer relevancy acceptable -> room to improve")