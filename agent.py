#agent.py
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from config import GROQ_API_KEY, GROQ_MODEL, MAX_RETRIES

llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0,
    api_key=GROQ_API_KEY,
)


class RAGState(TypedDict):
    question:          str
    context_chunks:    list
    answer:            str
    validation_result: str
    fail_reason:       str
    retry_count:       int
    chat_history:      list


def generate_node(state: RAGState) -> dict:
    context_text = "\n\n---\n\n".join(
        f"[Source: {r['source']}]\n{r['chunk']}"
        for r in state["context_chunks"]
    )

    history_lines = []
    for msg in state.get("chat_history", [])[-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history_lines.append(f"{role}: {msg.content}")
    history_text = "\n".join(history_lines) or "None"

    correction = ""
    if state.get("retry_count", 0) > 0:
        correction = (
            f"\n\nIMPORTANT CORRECTION REQUIRED: Your previous answer was "
            f"rejected because: {state.get('fail_reason', 'unverifiable claims')}. "
            f"Re-answer using ONLY the context provided."
        )

    prompt = (
        "You are an AI assistant that answers questions AND generates content based on provided documents.\n"
        "Answer ONLY using information from the CONTEXT below.\n"
        "If the answer cannot be found, say exactly: "
        '"I don\'t have enough information in the provided documents."\n'
        "Do NOT invent facts or use outside knowledge."
        + correction
        + f"\n\nPREVIOUS CONVERSATION:\n{history_text}"
        + f"\n\nCONTEXT:\n{context_text}"
        + f"\n\nQUESTION: {state['question']}\n\nAnswer:"
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"answer": response.content}


def validate_node(state: RAGState) -> dict:
    context_text = "\n\n".join(r["chunk"] for r in state["context_chunks"])

    prompt = (
        "You are a strict hallucination checker for a RAG system.\n\n"
        "Given the CONTEXT and the ANSWER below, check:\n"
        "1. Is every factual claim directly supported by the context?\n"
        "2. Does the answer address the question?\n"
        "3. Are there any invented facts not in the context?\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {state['question']}\n"
        f"Answer: {state['answer']}\n\n"
        "Respond in EXACTLY this format:\n"
        "VERDICT: PASS\n"
        "REASON: <one sentence>\n\n"
        "or\n\n"
        "VERDICT: FAIL\n"
        "REASON: <one sentence explaining what is wrong>"
    )

    result = llm.invoke([HumanMessage(content=prompt)])
    text   = result.content.strip()

    verdict = "PASS" if "VERDICT: PASS" in text.upper() else "FAIL"
    reason  = ""
    for line in text.splitlines():
        if line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
            break

    return {"validation_result": verdict, "fail_reason": reason}


def increment_retry_node(state: RAGState) -> dict:
    return {"retry_count": state.get("retry_count", 0) + 1}


def route_after_validation(state: RAGState) -> str:
    if (
        state["validation_result"] == "FAIL"
        and state.get("retry_count", 0) < MAX_RETRIES
    ):
        return "retry"
    return "done"


def _build_graph():
    g = StateGraph(RAGState)
    g.add_node("generate",        generate_node)
    g.add_node("validate",        validate_node)
    g.add_node("increment_retry", increment_retry_node)
    g.set_entry_point("generate")
    g.add_edge("generate", "validate")
    g.add_conditional_edges(
        "validate",
        route_after_validation,
        {"retry": "increment_retry", "done": END},
    )
    g.add_edge("increment_retry", "generate")
    return g.compile()


_rag_graph = _build_graph()


def run_rag_agent(
    question:       str,
    context_chunks: list,
    chat_history:   list = [],
) -> tuple:
    init_state: RAGState = {
        "question":          question,
        "context_chunks":    context_chunks,
        "answer":            "",
        "validation_result": "",
        "fail_reason":       "",
        "retry_count":       0,
        "chat_history":      chat_history,
    }
    final = _rag_graph.invoke(init_state)
    return final["answer"], final["retry_count"], final["validation_result"]