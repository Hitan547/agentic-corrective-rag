from mcp.server.fastmcp import FastMCP
import requests
import time

HF_URL = "https://hitan2004-agentic-corrective-rag.hf.space"

mcp = FastMCP("Agentic Corrective RAG")

def wake_up_hf():
    for i in range(5):
        try:
            r = requests.get(f"{HF_URL}/health", timeout=30)
            if r.status_code == 200:
                print("HuggingFace space is awake")
                return
        except:
            print(f"Attempt {i+1}/5 - Waiting for HF space...")
            time.sleep(15)
    print("Proceeding anyway...")

@mcp.tool()
def query_rag(question: str, session_id: str = "default") -> dict:
    """Query documents using corrective RAG with hallucination detection."""
    response = requests.post(f"{HF_URL}/query",
                             json={"query": question, "session_id": session_id})
    return response.json()

@mcp.tool()
def ingest_document(file_path: str) -> dict:
    """Upload and index a PDF or TXT document."""
    with open(file_path, "rb") as f:
        response = requests.post(f"{HF_URL}/upload", files={"file": f})
    return response.json()

@mcp.tool()
def clear_session(session_id: str) -> dict:
    """Clear conversation history for a session."""
    response = requests.delete(f"{HF_URL}/session/{session_id}")
    return response.json()

if __name__ == "__main__":
    wake_up_hf()
    mcp.run()
