from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated
import os

load_dotenv()

# ── State with message history using reducer ───────────────────────────────────
# Annotated[list, add_messages] means:
# instead of overwriting messages list, APPEND new messages to it
# This is how LangGraph maintains conversation history
class ConversationState(TypedDict):
    messages:       Annotated[list, add_messages]   # appends each turn
    turn_count:     int                              # overwrites each turn
    last_topic:     str                              # overwrites each turn


# ── Nodes ──────────────────────────────────────────────────────────────────────
def conversation_node(state: ConversationState) -> dict:
    """
    Processes the latest message and generates a response.
    In production: replace with real LLM call.
    """
    messages    = state["messages"]
    turn_count  = state.get("turn_count", 0)

    # Get the latest human message
    latest_msg = messages[-1].content if messages else ""
    print(f"[conversation_node] Turn {turn_count + 1}: '{latest_msg}'")

    # Simulate topic detection
    if "rag" in latest_msg.lower():
        topic    = "RAG"
        response = (f"Turn {turn_count + 1}: RAG combines retrieval with generation. "
                    f"It fetches relevant documents and passes them as context to the LLM.")
    elif "langgraph" in latest_msg.lower():
        topic    = "LangGraph"
        response = (f"Turn {turn_count + 1}: LangGraph is a framework for building "
                    f"stateful agent graphs with cycles, conditional routing, and checkpointing.")
    elif "memory" in latest_msg.lower() or "remember" in latest_msg.lower():
        topic    = "Memory"
        # Reference conversation history — this is the power of checkpointing
        prev_topics = [
            m.content[:30] for m in messages
            if isinstance(m, HumanMessage)
        ]
        response = (f"Turn {turn_count + 1}: I remember our conversation! "
                    f"You previously asked about: {prev_topics}")
    else:
        topic    = "General"
        response = f"Turn {turn_count + 1}: Interesting question about '{latest_msg}'!"

    return {
        "messages":   [AIMessage(content=response)],  # add_messages appends this
        "turn_count": turn_count + 1,
        "last_topic": topic,
    }


# ── Build graph with checkpointer ─────────────────────────────────────────────
def build_conversation_graph():
    graph = StateGraph(ConversationState)
    graph.add_node("conversation", conversation_node)
    graph.add_edge(START, "conversation")
    graph.add_edge("conversation", END)

    # MemorySaver stores state in RAM — persists within same Python session
    # For production use: SqliteSaver or PostgresSaver
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ── Multi-turn conversation demo ───────────────────────────────────────────────
if __name__ == "__main__":
    app = build_conversation_graph()

    # thread_id is the conversation identifier
    # Same thread_id = same conversation history loaded from checkpoint
    # Different thread_id = fresh conversation
    config = {"configurable": {"thread_id": "user_123_session_1"}}

    questions = [
        "What is RAG?",
        "How does LangGraph work?",
        "Can you remember what we talked about?",   # tests memory
    ]

    print("=" * 60)
    print("Multi-turn conversation with checkpointing")
    print(f"Thread ID: {config['configurable']['thread_id']}")
    print("=" * 60)

    for question in questions:
        print(f"\nUser: {question}")

        result = app.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=config    # same config = same thread = memory persists
        )

        # Get latest AI response
        ai_response = result["messages"][-1].content
        print(f"AI  : {ai_response}")
        print(f"     [turn={result['turn_count']}, topic={result['last_topic']}]")

    # Show full conversation history stored in checkpoint
    print("\n" + "=" * 60)
    print("Full conversation history from checkpoint")
    print("=" * 60)
    final_state = app.get_state(config)
    for msg in final_state.values["messages"]:
        role = "User" if isinstance(msg, HumanMessage) else "AI"
        print(f"  [{role}] {msg.content[:80]}")

    # Demonstrate: new thread_id = fresh conversation, no memory
    print("\n" + "=" * 60)
    print("New thread_id = fresh conversation (no memory)")
    print("=" * 60)
    new_config = {"configurable": {"thread_id": "user_123_session_2"}}
    result = app.invoke(
        {"messages": [HumanMessage(content="Can you remember what we talked about?")]},
        config=new_config
    )
    print(f"AI: {result['messages'][-1].content}")
    print("^ No memory of previous session — different thread_id")