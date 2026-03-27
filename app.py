# app.py
import uuid
import streamlit as st
import requests

API = "http://localhost:8000"

st.set_page_config(
    page_title="Corrective RAG",
    page_icon="📄",
    layout="wide",
)
st.title("📄 Corrective RAG — Document Q&A")
st.caption("Groq LLaMA 3 · FAISS · BM25 · LangGraph self-correction")

# ── Session state init ────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload documents")
    uploaded_files = st.file_uploader(
        "Choose .txt or .pdf files",
        type=["txt", "pdf"],
        accept_multiple_files=True,
    )
    if st.button("Index documents", type="primary") and uploaded_files:
        for f in uploaded_files:
            try:
                r = requests.post(
                    f"{API}/upload",
                    files={"file": (f.name, f.getvalue())},
                    timeout=30,
                )
                if r.status_code == 200:
                    st.success(f"{f.name} — uploaded, indexing started")
                else:
                    st.error(f"{f.name} — {r.json().get('detail', r.text)}")
            except requests.ConnectionError:
                st.error("Cannot reach backend. Is `uvicorn main:app` running?")

    st.divider()

    # Health check
    try:
        h = requests.get(f"{API}/health", timeout=3).json()
        idx_status = "ready" if h.get("indexes_loaded") else "not loaded"
        st.caption(f"Backend: connected  |  Indexes: {idx_status}")
    except Exception:
        st.caption("Backend: not connected")

    st.divider()
    if st.button("Clear conversation"):
        try:
            requests.delete(f"{API}/session/{st.session_state.session_id}", timeout=5)
        except Exception:
            pass
        st.session_state.messages = []
        st.rerun()

    st.caption(f"Session ID: `{st.session_state.session_id[:8]}...`")

# ── Render chat history ───────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            m = msg["meta"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Retries used", m["retries"])
            c2.metric("Validation",   m["validation"])
            c3.metric("Sources found", m["num_sources"])
            if m.get("sources"):
                with st.expander("View source chunks"):
                    for s in m["sources"]:
                        st.markdown(f"**{s['source']}**")
                        st.text(s["chunk"])
                        st.divider()

# ── Chat input ────────────────────────────────────────────────
if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating (with self-correction)..."):
            answer = ""
            meta   = {"retries": 0, "validation": "N/A",
                      "num_sources": 0, "sources": []}
            try:
                r = requests.post(
                    f"{API}/query",
                    json={
                        "question":   question,
                        "session_id": st.session_state.session_id,
                    },
                    timeout=60,
                )
                if r.status_code == 200:
                    data   = r.json()
                    answer = data["answer"]
                    meta   = {
                        "retries":     data["retries_used"],
                        "validation":  data["validation"],
                        "num_sources": len(data["sources"]),
                        "sources":     data["sources"],
                    }
                else:
                    answer = f"Error {r.status_code}: {r.json().get('detail', r.text)}"

            except requests.ConnectionError:
                answer = "Cannot reach backend. Make sure `uvicorn main:app` is running."
            except requests.Timeout:
                answer = "Request timed out. The model may be slow — try again."
            except Exception as e:
                answer = f"Unexpected error: {e}"

        st.markdown(answer)
        c1, c2, c3 = st.columns(3)
        c1.metric("Retries used",  meta["retries"])
        c2.metric("Validation",    meta["validation"])
        c3.metric("Sources found", meta["num_sources"])
        if meta["sources"]:
            with st.expander("View source chunks"):
                for s in meta["sources"]:
                    st.markdown(f"**{s['source']}**")
                    st.text(s["chunk"])
                    st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "meta": meta,
    })