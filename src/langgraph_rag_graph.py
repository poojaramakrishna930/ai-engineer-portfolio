from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from typing import TypedDict, Literal
import os

load_dotenv()

# ── State ──────────────────────────────────────────────────────────────────────
class FullRAGState(TypedDict):
    question:       str
    context:        list[str]
    context_scores: list[float]
    answer:         str
    answer_grade:   float
    retry_count:    int
    route_log:      list[str]


# ── Build retriever ────────────────────────────────────────────────────────────
def build_retriever():
    docs_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs.txt")
    with open(docs_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    docs = [Document(page_content=line, metadata={"doc_id": i})
            for i, line in enumerate(lines)]

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    persist_dir = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
    vectorstore = Chroma.from_documents(
        docs, embeddings,
        persist_directory=persist_dir,
        collection_name="langgraph_rag"
    )
    bm25     = BM25Retriever.from_documents(docs, k=5)
    dense    = vectorstore.as_retriever(search_kwargs={"k": 5})
    return EnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5])


# ── Nodes ──────────────────────────────────────────────────────────────────────
retriever = build_retriever()

def retrieve_node(state: FullRAGState) -> dict:
    question = state["question"]
    retry    = state.get("retry_count", 0)
    log      = state.get("route_log", [])

    # On retry: reformulate the query slightly
    query = question if retry == 0 else f"explain in detail: {question}"

    docs     = retriever.invoke(query)
    contexts = [d.page_content for d in docs[:3]]
    scores   = [float(1.0 / (i + 1)) for i in range(len(contexts))]  # rank-based scores

    print(f"[retrieve] attempt={retry+1} query='{query[:50]}...'")
    return {
        "context":        contexts,
        "context_scores": scores,
        "route_log":      log + [f"retrieve(attempt={retry+1})"],
    }

def grade_context_node(state: FullRAGState) -> dict:
    """
    Grades whether retrieved context is relevant to the question.
    In production: use CrossEncoder or LLM-as-judge.
    Here: keyword overlap as proxy.
    """
    question = state["question"].lower()
    context  = state["context"]
    log      = state.get("route_log", [])

    q_words  = set(question.split())
    scores   = []
    for chunk in context:
        chunk_words = set(chunk.lower().split())
        overlap     = len(q_words & chunk_words) / max(len(q_words), 1)
        scores.append(overlap)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    print(f"[grade_context] avg relevance score = {avg_score:.2f}")
    return {
        "answer_grade": avg_score,
        "route_log":    log + [f"grade_context={avg_score:.2f}"],
    }

def generate_node(state: FullRAGState) -> dict:
    """
    Generates answer from context.
    Replace with real LLM call in Phase 2 projects.
    """
    question = state["question"]
    context  = state["context"]
    log      = state.get("route_log", [])

    context_text = "\n".join(context[:2])
    # Simulated answer — replace with: chain.invoke({"question": ..., "context": ...})
    answer = (f"Based on retrieved context: {context_text[:120]}... "
              f"The answer to '{question}' involves the above concepts.")

    print(f"[generate] answer generated")
    return {
        "answer":    answer,
        "route_log": log + ["generate"],
    }

def fallback_node(state: FullRAGState) -> dict:
    log = state.get("route_log", [])
    print("[fallback] returning best-effort response")
    return {
        "answer":    (f"I was unable to find sufficient context to answer: "
                      f"'{state['question']}'. Please try rephrasing."),
        "route_log": log + ["fallback"],
    }


# ── Routing functions ──────────────────────────────────────────────────────────
def route_after_context_grade(
    state: FullRAGState
) -> Literal["generate", "retrieve", "fallback"]:
    grade       = state.get("answer_grade", 0.0)
    retry_count = state.get("retry_count", 0)

    if grade >= 0.15:                     # threshold: enough relevant context
        print(f"[router] context grade {grade:.2f} → generate")
        return "generate"
    elif retry_count < 2:
        print(f"[router] context grade {grade:.2f} → retry retrieval")
        state["retry_count"] = retry_count + 1
        return "retrieve"
    else:
        print(f"[router] max retries, grade {grade:.2f} → fallback")
        return "fallback"


# ── Build graph ────────────────────────────────────────────────────────────────
def build_full_rag_graph():
    graph = StateGraph(FullRAGState)

    graph.add_node("retrieve",      retrieve_node)
    graph.add_node("grade_context", grade_context_node)
    graph.add_node("generate",      generate_node)
    graph.add_node("fallback",      fallback_node)

    graph.add_edge(START,           "retrieve")
    graph.add_edge("retrieve",      "grade_context")
    graph.add_edge("generate",      END)
    graph.add_edge("fallback",      END)

    graph.add_conditional_edges(
        "grade_context",
        route_after_context_grade,
        {
            "generate": "generate",
            "retrieve": "retrieve",
            "fallback": "fallback",
        }
    )

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ── Run it ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app   = build_full_rag_graph()
    config = {"configurable": {"thread_id": "rag_session_1"}}

    questions = [
        "What is RAG and how does it work?",
        "How do vector databases store embeddings?",
        "What is LangGraph used for?",
    ]

    for question in questions:
        print("\n" + "=" * 60)
        print(f"Question: {question}")
        print("=" * 60)

        initial_state = {
            "question":       question,
            "context":        [],
            "context_scores": [],
            "answer":         "",
            "answer_grade":   0.0,
            "retry_count":    0,
            "route_log":      [],
        }

        result = app.invoke(initial_state, config=config)

        print(f"\nAnswer     : {result['answer'][:120]}...")
        print(f"Grade      : {result['answer_grade']:.2f}")
        print(f"Retries    : {result['retry_count']}")
        print(f"Route taken: {' → '.join(result['route_log'])}")