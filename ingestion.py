# ingestion.py
import os, pickle
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import (
    DOCS_DIR, FAISS_INDEX_PATH, BM25_PATH,
    CHUNKS_PATH, SOURCES_PATH,
    EMBEDDER_NAME, CHUNK_SIZE, CHUNK_OVERLAP
)


def read_pdf_text(fpath):
    import fitz  # PyMuPDF
    doc = fitz.open(fpath)
    text = []
    for page in doc:
        text.append(page.get_text())
    return "\n".join(text).strip()


def clean_text(text):
    return " ".join(text.split())


def load_documents():
    docs, filenames = [], []
    path = Path(DOCS_DIR)
    path.mkdir(exist_ok=True)

    for fpath in path.glob("*.txt"):
        try:
            text = clean_text(fpath.read_text(encoding="utf-8"))
            docs.append(text)
            filenames.append(fpath.name)
            print(f"  Loaded text: {fpath.name}")
        except Exception as e:
            print(f"  Skipped {fpath.name}: {e}")

    for fpath in path.glob("*.pdf"):
        try:
            text = clean_text(read_pdf_text(fpath))
            if text:
                docs.append(text)
                filenames.append(fpath.name)
                print(f"  Loaded PDF:  {fpath.name}")
            else:
                print(f"  WARNING: {fpath.name} extracted empty text")
        except Exception as e:
            print(f"  Skipped {fpath.name}: {e}")

    if not docs:
        raise FileNotFoundError(
            f"No .txt or .pdf files found in '{DOCS_DIR}'. "
            "Add at least one document and re-run."
        )

    print(f"\nLoaded {len(docs)} document(s)")
    return docs, filenames


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

    print(f"Created {len(all_chunks)} chunks "
          f"(avg {sum(len(c) for c in all_chunks)//len(all_chunks)} chars each)")
    print("\n--- SAMPLE CHUNK ---")
    print(all_chunks[0][:500])
    print("--------------------\n")

    return all_chunks, all_sources


def build_indexes(chunks):
    print("\nBuilding dense embeddings...")

    model = SentenceTransformer(EMBEDDER_NAME)
    embeddings = model.encode(chunks, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)
    print(f"FAISS index: {faiss_index.ntotal} vectors, dim={dim}")

    tokenized = [c.lower().split() for c in chunks]
    bm25_index = BM25Okapi(tokenized)
    print("BM25 index: built")

    return faiss_index, bm25_index  # model not returned — HuggingFace caches it


def save_indexes(faiss_index, bm25_index, chunks, sources):
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)

    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25_index, f)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    with open(SOURCES_PATH, "wb") as f:
        pickle.dump(sources, f)

    print("\nSaved indexes to disk.")


def run_ingestion():
    print("=== Starting ingestion ===\n")
    docs, filenames = load_documents()
    chunks, sources = semantic_chunk(docs, filenames)
    fi, bm25 = build_indexes(chunks)
    save_indexes(fi, bm25, chunks, sources)
    print("\n=== Ingestion complete ===")


if __name__ == "__main__":
    run_ingestion()
