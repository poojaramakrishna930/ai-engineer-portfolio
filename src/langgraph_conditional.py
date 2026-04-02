from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
import os
import random

load_dotenv()

# ── State with retry tracking ──────────────────────────────────────────────────
class GradedRAGState(TypedDict):
    question:    str
    context:     list[str]
    answer:      str
    grade:       float        # quality score 0.0 to 1.0
    retry_count: int          # how many times we have retried
    route_taken: list[str]    # track routing decisions for debugging


# ── Nodes ──────────────────────────────────────────────────────────────────────
def retrieve_node(state: GradedRAGState) -> dict:
    retry = state.get("retry_count", 0)
    route = state.get("route_taken", [])
    print(f"[retrieve_node] attempt {retry + 1}")

    # Simulate retrieval — quality improves with retry
    docs = [
        f"Retrieved doc (attempt {retry + 1}): "
        f"{'Highly relevant: ' if retry > 0 else 'Somewhat relevant: '}"
        f"RAG answer for '{state['question']}'",
    ]
    return {
        "context":     docs,
        "route_taken": route + [f"retrieve (attempt {retry + 1})"]
    }

def generate_node(state: GradedRAGState) -> dict:
    route = state.get("route_taken", [])
    answer = f"Answer based on context: {state['context'][0][:80]}"
    print(f"[generate_node] generated answer")
    return {
        "answer":      answer,
        "route_taken": route + ["generate"]
    }

def grade_node(state: GradedRAGState) -> dict:
    """
    Grades the answer quality.
    In production: use an LLM-as-judge or RAGAS faithfulness metric.
    Here: simulate with random score that improves on retry.
    """
    route       = state.get("route_taken", [])
    retry_count = state.get("retry_count", 0)

    # Simulate: first attempt often fails, retry usually succeeds
    # In real projects: grade = llm_judge(state["question"], state["answer"])
    if retry_count == 0:
        grade = random.uniform(0.3, 0.65)    # first try: mediocre
    else:
        grade = random.uniform(0.75, 0.95)   # retry: much better

    print(f"[grade_node] grade = {grade:.2f} (retry #{retry_count})")
    return {
        "grade":       grade,
        "route_taken": route + [f"grade={grade:.2f}"]
    }

def fallback_node(state: GradedRAGState) -> dict:
    """Called when max retries reached and quality still low."""
    route = state.get("route_taken", [])
    print("[fallback_node] max retries reached — returning best effort answer")
    return {
        "answer":      f"I could not find a high-quality answer for: {state['question']}. "
                       f"Please rephrase your question or consult a human expert.",
        "route_taken": route + ["fallback"]
    }


# ── Conditional routing function ──────────────────────────────────────────────
def route_after_grading(state: GradedRAGState) -> Literal["generate", "retrieve", "fallback", "__end__"]:
    """
    This function inspects the current state and returns
    the NAME of the next node to execute.
    Return value must exactly match a node name or END.

    Routing logic:
    - grade >= 0.75 → answer is good enough → END
    - grade < 0.75 and retries < 2 → try again → retrieve
    - grade < 0.75 and retries >= 2 → give up → fallback
    """
    grade       = state.get("grade", 0.0)
    retry_count = state.get("retry_count", 0)

    if grade >= 0.75:
        print(f"[router] grade {grade:.2f} >= 0.75 → DONE")
        return END

    elif retry_count < 2:
        print(f"[router] grade {grade:.2f} < 0.75, retry {retry_count} → RETRY")
        # Increment retry count before routing back
        state["retry_count"] = retry_count + 1
        return "retrieve"

    else:
        print(f"[router] grade {grade:.2f} < 0.75, max retries → FALLBACK")
        return "fallback"


# ── Build graph with cycle ─────────────────────────────────────────────────────
def build_graded_rag_graph():
    graph = StateGraph(GradedRAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("grade",    grade_node)
    graph.add_node("fallback", fallback_node)

    # Normal edges
    graph.add_edge(START,      "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "grade")
    graph.add_edge("fallback", END)

    # Conditional edge — this is where the magic happens
    # After "grade" node runs, call route_after_grading(state)
    # to decide where to go next
    graph.add_conditional_edges(
        "grade",                  # source node
        route_after_grading,      # routing function
        {                         # mapping: return value → node name
            "retrieve":  "retrieve",
            "fallback":  "fallback",
            END:         END,
        }
    )

    return graph.compile()


# ── Run it ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = build_graded_rag_graph()

    initial_state = {
        "question":    "How does LangGraph handle retry loops?",
        "context":     [],
        "answer":      "",
        "grade":       0.0,
        "retry_count": 0,
        "route_taken": [],
    }

    print("=" * 60)
    print("Running LangGraph with conditional routing + retry cycle")
    print("=" * 60)

    final_state = app.invoke(initial_state)

    print("\n" + "=" * 60)
    print("Final State")
    print("=" * 60)
    print(f"Answer       : {final_state['answer'][:100]}...")
    print(f"Final grade  : {final_state['grade']:.2f}")
    print(f"Retries used : {final_state['retry_count']}")
    print(f"Route taken  :")
    for step in final_state["route_taken"]:
        print(f"  → {step}")