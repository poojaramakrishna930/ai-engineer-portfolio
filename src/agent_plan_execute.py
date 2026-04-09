from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Literal
import os
import json

load_dotenv()

# Import tools from agent_tools.py
import sys
sys.path.append(os.path.dirname(__file__))
from agent_tools import (
    search_knowledge_base,
    calculate,
    get_current_date,
    lookup_ai_concept,
)

TOOL_MAP = {
    "search_knowledge_base": search_knowledge_base,
    "calculate":             calculate,
    "get_current_date":      get_current_date,
    "lookup_ai_concept":     lookup_ai_concept,
}

# ═══════════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════════
# Notice how much richer state is vs ReAct
# ReAct only needs: messages + iteration
# Plan-and-Execute needs: full plan + results per step + replanning tracking

class PlanExecuteState(TypedDict):
    question:       str                # original user question — never changes
    plan:           list[dict]         # list of step dicts — created by planner
    current_step:   int                # which step we are executing right now
    step_results:   dict               # {step_index: result_string}
    replan_count:   int                # how many times replanner triggered
    final_answer:   str                # synthesised at end


# ═══════════════════════════════════════════════════════════════
# PLANNER NODE
# ═══════════════════════════════════════════════════════════════

def planner_node(state: PlanExecuteState) -> dict:
    """
    Creates a structured step-by-step plan for the given question.

    In production replace simulated_plan with:
        response = planner_llm.invoke([
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=state["question"])
        ])
        plan = json.loads(response.content)

    PLANNER_SYSTEM_PROMPT should instruct:
    - Return a JSON list of steps
    - Each step has: id, description, tool, tool_input, depends_on
    - Be specific about what each step should accomplish
    - Order steps logically — dependent steps after their dependencies
    """
    question = state["question"]
    print(f"\n[planner] Creating plan for: '{question}'")
    print(f"[planner] Calling planner LLM...")

    # ── Simulated planner output ───────────────────────────────────────────────
    # In production: this entire block is replaced by a real LLM call
    # The LLM returns structured JSON matching this schema
    def simulated_plan(q: str) -> list[dict]:
        q_lower = q.lower()

        if "compare" in q_lower and ("rag" in q_lower or "fine-tun" in q_lower):
            return [
                {
                    "id":          1,
                    "description": "Look up RAG definition and core components",
                    "tool":        "lookup_ai_concept",
                    "tool_input":  "rag",
                    "depends_on":  [],
                    "status":      "pending",
                    "result":      None,
                },
                {
                    "id":          2,
                    "description": "Search for RAG production use cases and limitations",
                    "tool":        "search_knowledge_base",
                    "tool_input":  "RAG production use cases advantages limitations",
                    "depends_on":  [1],
                    "status":      "pending",
                    "result":      None,
                },
                {
                    "id":          3,
                    "description": "Search for fine-tuning vs RAG comparison",
                    "tool":        "search_knowledge_base",
                    "tool_input":  "fine-tuning vs RAG tradeoffs when to use each",
                    "depends_on":  [],
                    "status":      "pending",
                    "result":      None,
                },
                {
                    "id":          4,
                    "description": "Synthesise all findings into structured comparison",
                    "tool":        "search_knowledge_base",
                    "tool_input":  "RAG fine-tuning production decision factors",
                    "depends_on":  [1, 2, 3],
                    "status":      "pending",
                    "result":      None,
                },
            ]
        elif "langgraph" in q_lower or "agent" in q_lower:
            return [
                {
                    "id":          1,
                    "description": "Look up LangGraph definition",
                    "tool":        "lookup_ai_concept",
                    "tool_input":  "langgraph",
                    "depends_on":  [],
                    "status":      "pending",
                    "result":      None,
                },
                {
                    "id":          2,
                    "description": "Search for LangGraph agent patterns",
                    "tool":        "search_knowledge_base",
                    "tool_input":  "LangGraph agentic patterns stateful workflows",
                    "depends_on":  [1],
                    "status":      "pending",
                    "result":      None,
                },
                {
                    "id":          3,
                    "description": "Search for LangGraph vs LangChain differences",
                    "tool":        "search_knowledge_base",
                    "tool_input":  "LangGraph LangChain differences cycles state",
                    "depends_on":  [1],
                    "status":      "pending",
                    "result":      None,
                },
            ]
        else:
            # Generic research plan
            return [
                {
                    "id":          1,
                    "description": f"Search knowledge base for: {q}",
                    "tool":        "search_knowledge_base",
                    "tool_input":  q,
                    "depends_on":  [],
                    "status":      "pending",
                    "result":      None,
                },
                {
                    "id":          2,
                    "description": "Get current date for context",
                    "tool":        "get_current_date",
                    "tool_input":  "",
                    "depends_on":  [],
                    "status":      "pending",
                    "result":      None,
                },
            ]

    plan = simulated_plan(question)

    print(f"[planner] Plan created with {len(plan)} steps:")
    for step in plan:
        deps = f" (depends on steps {step['depends_on']})" if step["depends_on"] else ""
        print(f"  Step {step['id']}: {step['description']}{deps}")
        print(f"           tool={step['tool']} input='{step['tool_input'][:40]}'")

    return {
        "plan":         plan,
        "current_step": 0,
        "step_results": {},
        "replan_count": 0,
    }


# ═══════════════════════════════════════════════════════════════
# EXECUTOR NODE
# ═══════════════════════════════════════════════════════════════

def executor_node(state: PlanExecuteState) -> dict:
    """
    Executes the current pending step in the plan.
    Finds the first step whose dependencies are all completed,
    executes its tool, stores the result, marks step as done.
    """
    plan         = state["plan"]
    step_results = state.get("step_results", {})
    current_step = state.get("current_step", 0)

    # Find next executable step — all dependencies must be completed
    next_step = None
    for step in plan:
        if step["status"] == "pending":
            deps_done = all(
                plan[d-1]["status"] == "completed"
                for d in step["depends_on"]
            )
            if deps_done:
                next_step = step
                break

    if not next_step:
        print(f"[executor] No executable steps found — all done or blocked")
        return {}

    print(f"\n[executor] Executing step {next_step['id']}: {next_step['description']}")
    print(f"[executor] Tool: {next_step['tool']} | Input: '{next_step['tool_input'][:50]}'")

    # Execute the tool
    tool_fn    = TOOL_MAP.get(next_step["tool"])
    tool_input = next_step["tool_input"]

    try:
        if tool_input:
            # Try different argument names based on tool
            if next_step["tool"] == "search_knowledge_base":
                result = tool_fn.invoke({"query": tool_input})
            elif next_step["tool"] == "calculate":
                result = tool_fn.invoke({"expression": tool_input})
            elif next_step["tool"] == "lookup_ai_concept":
                result = tool_fn.invoke({"concept": tool_input})
            else:
                result = tool_fn.invoke({"query": tool_input})
        else:
            result = tool_fn.invoke({})

        result_str = str(result)[:300]
        print(f"[executor] Result: {result_str[:80]}...")

    except Exception as e:
        result_str = f"Tool execution failed: {str(e)}"
        print(f"[executor] Error: {result_str}")

    # Update plan — mark step as completed with result
    updated_plan          = [s.copy() for s in plan]
    step_idx              = next_step["id"] - 1
    updated_plan[step_idx]["status"] = "completed"
    updated_plan[step_idx]["result"] = result_str

    # Store result indexed by step id
    updated_results           = dict(step_results)
    updated_results[next_step["id"]] = result_str

    return {
        "plan":         updated_plan,
        "step_results": updated_results,
        "current_step": current_step + 1,
    }


# ═══════════════════════════════════════════════════════════════
# REPLANNER NODE
# ═══════════════════════════════════════════════════════════════

def replanner_node(state: PlanExecuteState) -> dict:
    """
    Checks whether the plan needs adjustment after each step execution.
    Reviews completed step results and decides whether to:
    - Continue with remaining steps unchanged
    - Add new steps based on discovered information
    - Remove steps that are now redundant
    - Mark plan as complete if enough info gathered

    In production replace with:
        response = replanner_llm.invoke([
            SystemMessage(content=REPLANNER_PROMPT),
            HumanMessage(content=f"Original question: {question}
            Plan: {plan}
            Results so far: {step_results}
            Should we adjust the plan?")
        ])
    """
    plan         = state["plan"]
    step_results = state["step_results"]
    question     = state["question"]
    replan_count = state.get("replan_count", 0)

    completed = [s for s in plan if s["status"] == "completed"]
    pending   = [s for s in plan if s["status"] == "pending"]

    print(f"\n[replanner] Reviewing plan: {len(completed)} done, {len(pending)} pending")

    # ── Simulated replanning logic ─────────────────────────────────────────────
    # In production: LLM reads all results and decides adjustments

    # Check if any completed step result suggests we need extra research
    all_results_text = " ".join(str(v) for v in step_results.values()).lower()
    updated_plan     = [s.copy() for s in plan]

    # Example replanning trigger: if results mention "limitations" — add a step
    if ("limitation" in all_results_text and
        replan_count == 0 and
        len(pending) > 0):

        new_step = {
            "id":          len(plan) + 1,
            "description": "Search for workarounds to identified limitations",
            "tool":        "search_knowledge_base",
            "tool_input":  f"solutions workarounds limitations {question[:30]}",
            "depends_on":  [completed[-1]["id"]] if completed else [],
            "status":      "pending",
            "result":      None,
        }
        updated_plan.append(new_step)
        print(f"[replanner] Added new step {new_step['id']}: {new_step['description']}")
        return {
            "plan":         updated_plan,
            "replan_count": replan_count + 1,
        }

    # No changes needed
    print(f"[replanner] Plan is still valid — continuing execution")
    return {"replan_count": replan_count}


# ═══════════════════════════════════════════════════════════════
# SYNTHESISER NODE
# ═══════════════════════════════════════════════════════════════

def synthesise_node(state: PlanExecuteState) -> dict:
    """
    Combines all step results into a final coherent answer.
    In production: llm.invoke(f"Synthesise these research findings: {step_results}")
    """
    question     = state["question"]
    step_results = state["step_results"]
    plan         = state["plan"]

    print(f"\n[synthesise] Combining {len(step_results)} step results into final answer")

    # Build synthesis from all results
    synthesis_parts = []
    for step in plan:
        if step["status"] == "completed" and step["result"]:
            synthesis_parts.append(
                f"From step {step['id']} ({step['description'][:40]}): "
                f"{step['result'][:100]}"
            )

    final_answer = (
        f"Research complete for: '{question}'\n\n"
        f"Findings from {len(synthesis_parts)} research steps:\n"
        + "\n".join(f"  • {p}" for p in synthesis_parts)
        + f"\n\nConclusion: Based on {len(step_results)} sources of evidence, "
          f"the answer has been synthesised from structured research."
    )

    print(f"[synthesise] Final answer ready ({len(final_answer)} chars)")
    return {"final_answer": final_answer}


# ═══════════════════════════════════════════════════════════════
# ROUTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def route_after_executor(
    state: PlanExecuteState
) -> Literal["replanner", "synthesise"]:
    """
    After each step executes:
    - If more pending steps exist → replanner (check if plan needs adjustment)
    - If all steps completed → synthesise (build final answer)
    """
    plan    = state["plan"]
    pending = [s for s in plan if s["status"] == "pending"]

    if pending:
        print(f"[router] {len(pending)} steps remaining → replanner")
        return "replanner"
    else:
        print(f"[router] All steps complete → synthesise")
        return "synthesise"


def route_after_replanner(
    state: PlanExecuteState
) -> Literal["executor", "synthesise"]:
    """
    After replanning:
    - If pending steps still exist → executor (keep going)
    - If no pending steps → synthesise
    """
    plan    = state["plan"]
    pending = [s for s in plan if s["status"] == "pending"]

    if pending:
        print(f"[router] {len(pending)} steps pending → executor")
        return "executor"
    else:
        print(f"[router] No pending steps → synthesise")
        return "synthesise"


# ═══════════════════════════════════════════════════════════════
# BUILD GRAPH
# ═══════════════════════════════════════════════════════════════

def build_plan_execute_graph():
    graph = StateGraph(PlanExecuteState)

    graph.add_node("planner",    planner_node)
    graph.add_node("executor",   executor_node)
    graph.add_node("replanner",  replanner_node)
    graph.add_node("synthesise", synthesise_node)

    # Fixed edges
    graph.add_edge(START,        "planner")
    graph.add_edge("planner",    "executor")
    graph.add_edge("synthesise", END)

    # Conditional edges — create the execute → replan → execute cycle
    graph.add_conditional_edges(
        "executor",
        route_after_executor,
        {
            "replanner":  "replanner",
            "synthesise": "synthesise",
        }
    )
    graph.add_conditional_edges(
        "replanner",
        route_after_replanner,
        {
            "executor":   "executor",
            "synthesise": "synthesise",
        }
    )

    return graph.compile()


# ═══════════════════════════════════════════════════════════════
# RUN AND COMPARE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = build_plan_execute_graph()

    questions = [
        "Compare RAG vs fine-tuning for production AI applications",
        "Explain LangGraph agent patterns and how they work",
    ]

    for question in questions:
        print("\n" + "=" * 65)
        print(f"Question: {question}")
        print("=" * 65)

        initial_state = {
            "question":     question,
            "plan":         [],
            "current_step": 0,
            "step_results": {},
            "replan_count": 0,
            "final_answer": "",
        }

        result = app.invoke(initial_state)

        print("\n" + "─" * 65)
        print("FINAL RESULT")
        print("─" * 65)
        print(f"Steps executed  : {result['current_step']}")
        print(f"Replan count    : {result['replan_count']}")
        print(f"Steps in plan   : {len(result['plan'])}")
        print(f"\nPlan execution summary:")
        for step in result["plan"]:
            status_icon = "✓" if step["status"] == "completed" else "○"
            print(f"  {status_icon} Step {step['id']}: {step['description'][:60]}")

        print(f"\nFinal answer (first 300 chars):")
        print(f"{result['final_answer'][:300]}...")

    # ── Direct comparison with ReAct ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("ReAct vs Plan-and-Execute — architectural comparison")
    print("=" * 65)
    print("""
ReAct:
  Decision point : after EVERY observation
  Plan visibility: none — next action decided on the fly
  LLM calls      : N (one per thought-action-observation loop)
  Best for       : exploratory tasks, 1-3 tool calls, speed
  Risk           : can lose track of goal on complex multi-step tasks

Plan-and-Execute:
  Decision point : upfront (planner) + after each step (replanner)
  Plan visibility: full — all steps visible before execution starts
  LLM calls      : 1 (planner) + N executors + M replanners
  Best for       : complex research, 5+ steps, parallel execution
  Risk           : initial plan may be wrong if question is ambiguous

Hybrid (best of both):
  Use plan-and-execute for the outer loop (what to research)
  Use ReAct for each individual step execution (how to use tools)
  This is what production agents like AutoGPT and GPT-Researcher do
""")