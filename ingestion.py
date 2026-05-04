import os, pickle, hashlib
from pathlib import Path
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import (
    DOCS_DIR, CHROMA_PATH, CHROMA_COLLECTION,
    BM25_PATH, EMBEDDER_NAME, CHUNK_SIZE, CHUNK_OVERLAP
)

# ── helpers ───────────────────────────────────────────

def read_pdf_text(fpath):
    import fitz
    doc = fitz.open(fpath)
    return "\n".join(page.get_text() for page in doc).strip()

def clean_text(text):
    return " ".join(text.split())

def doc_hash(text: str) -> str:
    """SHA-256 of the document — used to skip duplicate ingestion."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]

# ── loading ───────────────────────────────────────────

def load_documents():
    docs, filenames = [], []
    path = Path(DOCS_DIR)
    path.mkdir(exist_ok=True)
    for fpath in path.glob("*.txt"):
        try:
            text = clean_text(fpath.read_text(encoding="utf-8"))
            docs.append(text); filenames.append(fpath.name)
            print(f"  Loaded txt: {fpath.name}")
        except Exception as e:
            print(f"  Skipped {fpath.name}: {e}")
    for fpath in path.glob("*.pdf"):
        try:
            text = clean_text(read_pdf_text(fpath))
            if text:
                docs.append(text); filenames.append(fpath.name)
                print(f"  Loaded pdf: {fpath.name}")
        except Exception as e:
            print(f"  Skipped {fpath.name}: {e}")
    if not docs:
        raise FileNotFoundError(
            f"No .txt or .pdf files found in '{DOCS_DIR}'."
        )
    print(f"\nLoaded {len(docs)} document(s)")
    return docs, filenames

# ── chunking ──────────────────────────────────────────

def semantic_chunk(docs, filenames):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    all_chunks, all_sources = [], []
    for doc, fname in zip(docs, filenames):
        chunks = splitter.split_text(doc)
        all_chunks.extend(chunks)
        all_sources.extend([fname] * len(chunks))
    avg = sum(len(c) for c in all_chunks) // len(all_chunks)
    print(f"Created {len(all_chunks)} chunks (avg {avg} chars)")
    return all_chunks, all_sources

# ── indexing ──────────────────────────────────────────

def build_and_save_indexes(chunks, sources, model=None):
    if model is None:
        model = SentenceTransformer(EMBEDDER_NAME)

    print("\nBuilding embeddings...")
    embeddings = model.encode(
        chunks, show_progress_bar=True, batch_size=32
    ).tolist()

    # ── ChromaDB ──
    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    # Skip chunks already indexed (dedup by content hash)
    existing_ids = set(collection.get()["ids"])
    new_chunks, new_embeddings, new_sources, new_ids, new_meta = [], [], [], [], []

    for i, (chunk, emb, src) in enumerate(zip(chunks, embeddings, sources)):
        chunk_id = f"doc_{doc_hash(chunk)}"
        if chunk_id not in existing_ids:
            new_chunks.append(chunk)
            new_embeddings.append(emb)
            new_sources.append(src)
            new_ids.append(chunk_id)
            new_meta.append({"source": src})

    if new_chunks:
        collection.add(
            documents=new_chunks,
            embeddings=new_embeddings,
            ids=new_ids,
            metadatas=new_meta,
        )
        print(f"Added {len(new_chunks)} new chunks to ChromaDB")
    else:
        print("No new chunks — all already indexed")

    # ── BM25 (full rebuild, cheap) ──
    all_chunks_in_db = collection.get()["documents"]
    all_sources_in_db = [m["source"] for m in collection.get()["metadatas"]]
    tokenized   = [c.lower().split() for c in all_chunks_in_db]
    bm25_index  = BM25Okapi(tokenized)

    with open(BM25_PATH, "wb") as f:
        pickle.dump({
            "bm25": bm25_index,
            "chunks": all_chunks_in_db,
            "sources": all_sources_in_db
        }, f)

    print(f"BM25 saved — {len(all_chunks_in_db)} total chunks")

# ── entry point ───────────────────────────────────────

def run_ingestion(model=None):
    print("=== Starting ingestion ===\n")
    docs, filenames = load_documents()
    chunks, sources = semantic_chunk(docs, filenames)
    build_and_save_indexes(chunks, sources, model=model)
    print("\n=== Ingestion complete ===")

if __name__ == "__main__":
    run_ingestion()
