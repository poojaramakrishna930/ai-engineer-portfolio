from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.llms import HuggingFaceHub
from langchain_community.chat_models import ChatHuggingFace
from typing import TypedDict, Annotated, Literal
import os

load_dotenv()

# Import tools from Step 1
from agent_tools import (
    search_knowledge_base,
    calculate,
    get_current_date,
    lookup_ai_concept,
)

# ── Tool registry ──────────────────────────────────────────────────────────────
TOOLS = [
    search_knowledge_base,
    calculate,
    get_current_date,
    lookup_ai_concept,
]

# Tool lookup by name — used in manual tool execution
TOOL_MAP = {t.name: t for t in TOOLS}

# ── State ──────────────────────────────────────────────────────────────────────
class ReActState(TypedDict):
    messages:      Annotated[list, add_messages]  # full conversation + tool calls
    iteration:     int                            # how many ReAct loops ran
    final_answer:  str                            # extracted final answer


# ── Manual ReAct implementation ────────────────────────────────────────────────
# Implement ReAct manually (without a real LLM API) so you can
# see every step clearly. 

def simulated_llm_decision(question: str, history: list) -> dict:
    """
    Simulates what an LLM would return in a ReAct loop.
    In production replace with:
        response = llm_with_tools.invoke(messages)
    Returns: {"type": "tool_call"|"final_answer", "tool": str, "input": str, "answer": str}
    """
    question_lower = question.lower()
    iteration      = len([m for m in history if "Observation" in str(m)]) + 1

    # Simulate multi-step reasoning
    if iteration == 1:
        # First iteration: decide which tool to use
        if any(w in question_lower for w in ["date", "time", "today", "when"]):
            return {"type": "tool_call", "tool": "get_current_date", "input": ""}
        elif any(w in question_lower for w in ["calculate", "math", "percent", "+", "*", "/"]):
            expr = "100 * 0.15"   # simulated extraction from question
            return {"type": "tool_call", "tool": "calculate", "input": expr}
        elif any(w in question_lower for w in ["what is", "define", "explain"]):
            # Extract concept from question
            for concept in ["rag", "hnsw", "bm25", "react", "langgraph"]:
                if concept in question_lower:
                    return {"type": "tool_call", "tool": "lookup_ai_concept", "input": concept}
            return {"type": "tool_call", "tool": "search_knowledge_base", "input": question}
        else:
            return {"type": "tool_call", "tool": "search_knowledge_base", "input": question}

    else:
        # After first tool call: synthesise final answer from observations
        observations = [m for m in history if "Observation" in str(m)]
        obs_text     = observations[-1] if observations else "No observation"
        return {
            "type":   "final_answer",
            "answer": f"Based on my research: {str(obs_text)[:200]}"
        }


# ── ReAct nodes ───────────────────────────────────────────────────────────────
def reason_node(state: ReActState) -> dict:
    """
    Reasoning node — LLM decides what to do next.
    In production: calls LLM with tools bound to it.
    """
    messages   = state["messages"]
    iteration  = state.get("iteration", 0)
    question   = messages[0].content if messages else ""

    print(f"\n[reason_node] iteration={iteration+1}")
    print(f"  Thinking about: '{question[:60]}...'")

    # Simulated LLM decision — replace with real LLM in projects
    decision = simulated_llm_decision(question, messages)

    if decision["type"] == "tool_call":
        thought = (f"Thought: I should use {decision['tool']} "
                   f"with input '{decision['input']}' to answer this.")
        print(f"  {thought}")
        return {
            "messages":  [AIMessage(content=f"TOOL_CALL:{decision['tool']}:{decision['input']}")],
            "iteration": iteration + 1,
        }
    else:
        print(f"  Thought: I have enough information to answer.")
        return {
            "messages":     [AIMessage(content=f"FINAL:{decision['answer']}")],
            "iteration":    iteration + 1,
            "final_answer": decision["answer"],
        }


def act_node(state: ReActState) -> dict:
    """
    Action node — executes the tool the LLM chose.
    Reads the tool call from the last AI message.
    """
    messages    = state["messages"]
    last_msg    = messages[-1].content if messages else ""

    if not last_msg.startswith("TOOL_CALL:"):
        return {}

    # Parse tool call: "TOOL_CALL:tool_name:tool_input"
    parts      = last_msg.split(":", 2)
    tool_name  = parts[1] if len(parts) > 1 else ""
    tool_input = parts[2] if len(parts) > 2 else ""

    print(f"\n[act_node] executing tool: '{tool_name}' with input: '{tool_input}'")

    # Execute the tool
    tool_fn = TOOL_MAP.get(tool_name)
    if tool_fn:
        try:
            if tool_input:
                # Try common arg names
                try:
                    observation = tool_fn.invoke({"query":      tool_input})
                except Exception:
                    try:
                        observation = tool_fn.invoke({"expression": tool_input})
                    except Exception:
                        observation = tool_fn.invoke({"concept":    tool_input})
            else:
                observation = tool_fn.invoke({})

            print(f"  Observation: {str(observation)[:100]}...")
        except Exception as e:
            observation = f"Tool error: {str(e)}"
    else:
        observation = f"Tool '{tool_name}' not found."

    return {
        "messages": [AIMessage(content=f"Observation: {observation}")]
    }


def route_after_reasoning(state: ReActState) -> Literal["act", "__end__"]:
    """
    After reasoning: if LLM chose a tool → act.
    If LLM produced final answer → end.
    """
    messages  = state["messages"]
    last_msg  = messages[-1].content if messages else ""
    iteration = state.get("iteration", 0)

    if last_msg.startswith("FINAL:") or iteration >= 5:
        print(f"\n[router] final answer reached after {iteration} iterations")
        return END

    print(f"\n[router] tool call detected → executing action")
    return "act"


# ── Build ReAct graph ──────────────────────────────────────────────────────────
def build_react_graph():
    graph = StateGraph(ReActState)

    graph.add_node("reason", reason_node)
    graph.add_node("act",    act_node)

    graph.add_edge(START,   "reason")
    graph.add_edge("act",   "reason")    # act → reason creates the loop

    graph.add_conditional_edges(
        "reason",
        route_after_reasoning,
        {"act": "act", END: END}
    )

    return graph.compile()


# ── Run ReAct agent ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = build_react_graph()

    test_questions = [
        "What is the ReAct pattern in AI agents?",
        "What is today's date?",
        "Calculate 250 * 0.18 + 75",
    ]

    for question in test_questions:
        print("\n" + "=" * 65)
        print(f"Question: {question}")
        print("=" * 65)

        initial_state = {
            "messages":     [HumanMessage(content=question)],
            "iteration":    0,
            "final_answer": "",
        }

        result = app.invoke(initial_state)

        print(f"\nFinal answer  : {result['final_answer'][:150]}")
        print(f"Iterations    : {result['iteration']}")
        print(f"Message count : {len(result['messages'])}")
        print("Message trace:")
        for msg in result["messages"]:
            role = "Human" if isinstance(msg, HumanMessage) else "AI"
            print(f"  [{role}] {msg.content[:80]}")