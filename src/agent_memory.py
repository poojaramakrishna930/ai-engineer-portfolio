from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from typing import TypedDict, Annotated
import os
import json
from datetime import datetime

load_dotenv()

# ═══════════════════════════════════════════════════════════════
# MEMORY TYPE 1: In-context (short-term) — messages in prompt
# ═══════════════════════════════════════════════════════════════

print("=" * 65)
print("Memory Type 1: In-context (short-term)")
print("=" * 65)

class ShortTermState(TypedDict):
    messages:   Annotated[list, add_messages]   # current session messages
    turn_count: int

def short_term_node(state: ShortTermState) -> dict:
    messages   = state["messages"]
    turn_count = state.get("turn_count", 0)

    # Simulate: LLM reads ALL messages in context window
    context_window = messages[-6:]   # last 6 messages = short-term window
    latest         = messages[-1].content

    print(f"  Turn {turn_count + 1}: '{latest}'")
    print(f"  Context window: {len(context_window)} messages visible to LLM")

    # Simulated response
    response = f"[Turn {turn_count+1}] I see {len(context_window)} messages. Responding to: {latest[:50]}"
    return {
        "messages":   [AIMessage(content=response)],
        "turn_count": turn_count + 1,
    }

g1  = StateGraph(ShortTermState)
g1.add_node("chat", short_term_node)
g1.add_edge(START, "chat")
g1.add_edge("chat", END)
ckpt1 = MemorySaver()
app1  = g1.compile(checkpointer=ckpt1)
cfg1  = {"configurable": {"thread_id": "short_term_demo"}}

for msg in ["What is RAG?", "How does chunking work?", "Can you summarise what we discussed?"]:
    result = app1.invoke({"messages": [HumanMessage(content=msg)]}, config=cfg1)
    print(f"  AI: {result['messages'][-1].content}")


# ═══════════════════════════════════════════════════════════════
# MEMORY TYPE 2: External (long-term) — stored in vector DB
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("Memory Type 2: External (long-term) — vector DB")
print("=" * 65)

class LongTermState(TypedDict):
    messages:       Annotated[list, add_messages]
    retrieved_memory: list[str]    # memories fetched from vector DB

# Simulated long-term memory store
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
persist_dir = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

# Seed with past "memories" from previous sessions
past_memories = [
    Document(
        page_content="User prefers concise answers under 100 words.",
        metadata={"type": "preference", "date": "2024-01-01"}
    ),
    Document(
        page_content="User is learning RAG and LangGraph for AI engineer job interviews.",
        metadata={"type": "goal", "date": "2024-01-02"}
    ),
    Document(
        page_content="User is based in Mysore, Karnataka. Looking for remote or Bangalore jobs.",
        metadata={"type": "context", "date": "2024-01-03"}
    ),
    Document(
        page_content="User has 6-18 months experience with Python, LangChain and RAG.",
        metadata={"type": "profile", "date": "2024-01-04"}
    ),
]

memory_store = Chroma.from_documents(
    past_memories, embeddings,
    persist_directory=persist_dir,
    collection_name="agent_long_term_memory"
)

def retrieve_memory_node(state: LongTermState) -> dict:
    """Retrieves relevant past memories before generating response."""
    question = state["messages"][-1].content
    results  = memory_store.similarity_search(question, k=2)
    memories = [doc.page_content for doc in results]
    print(f"  Retrieved {len(memories)} relevant memories:")
    for m in memories:
        print(f"    → {m}")
    return {"retrieved_memory": memories}

def respond_with_memory_node(state: LongTermState) -> dict:
    """Generates response informed by retrieved long-term memories."""
    question  = state["messages"][-1].content
    memories  = state.get("retrieved_memory", [])
    memory_ctx = "\n".join(memories)
    response   = (f"[Using {len(memories)} memories] "
                  f"Memory context: {memory_ctx[:80]}... "
                  f"Response to: {question[:50]}")
    return {"messages": [AIMessage(content=response)]}

def save_to_memory_node(state: LongTermState) -> dict:
    """Saves important information from this conversation to long-term memory."""
    latest_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )
    # Save to vector DB for future sessions
    new_memory = Document(
        page_content=f"User asked: {latest_human}",
        metadata={"type": "interaction", "date": datetime.now().isoformat()}
    )
    memory_store.add_documents([new_memory])
    print(f"  Saved to long-term memory: '{latest_human[:60]}'")
    return {}

g2 = StateGraph(LongTermState)
g2.add_node("retrieve_memory",     retrieve_memory_node)
g2.add_node("respond_with_memory", respond_with_memory_node)
g2.add_node("save_to_memory",      save_to_memory_node)
g2.add_edge(START,               "retrieve_memory")
g2.add_edge("retrieve_memory",   "respond_with_memory")
g2.add_edge("respond_with_memory","save_to_memory")
g2.add_edge("save_to_memory",    END)
app2  = g2.compile()

result = app2.invoke({
    "messages":        [HumanMessage(content="What job am I preparing for?")],
    "retrieved_memory": [],
})
print(f"\n  AI: {result['messages'][-1].content}")


# ═══════════════════════════════════════════════════════════════
# MEMORY TYPE 3: Episodic — checkpointed state snapshots
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("Memory Type 3: Episodic — checkpointed state snapshots")
print("=" * 65)
print("(Already demonstrated in Day 6 — langgraph_checkpointing.py)")
print("Key points:")
print("  → MemorySaver: in-RAM, lost on restart, great for dev")
print("  → SqliteSaver: persists to .db file, survives restarts")
print("  → PostgresSaver: production-grade, multi-user support")
print("  → Same thread_id = same episode, different = new episode")


# ═══════════════════════════════════════════════════════════════
# MEMORY TYPE 4: Semantic — summarised compressed memory
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("Memory Type 4: Semantic — summarised compressed memory")
print("=" * 65)

class SummaryState(TypedDict):
    messages:         Annotated[list, add_messages]
    running_summary:  str      # compressed summary of past conversation
    token_count:      int      # tracks how full the context window is

TOKEN_LIMIT = 500   # simulate a small context window for demo

def estimate_tokens(messages: list) -> int:
    """Rough token estimate: 1 token ≈ 4 characters."""
    total_chars = sum(len(m.content) for m in messages)
    return total_chars // 4

def summarise_messages(messages: list) -> str:
    """
    Simulates LLM summarisation of past messages.
    In production: llm.invoke(f"Summarise this conversation: {messages}")
    """
    topics = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            topics.append(f"User asked about: {msg.content[:40]}")
    return "Summary of past conversation: " + " | ".join(topics[-3:])

def conversation_with_summary_node(state: SummaryState) -> dict:
    messages      = state["messages"]
    summary       = state.get("running_summary", "")
    token_count   = estimate_tokens(messages)

    print(f"  Token count: ~{token_count} tokens")

    # If approaching token limit — summarise and compress
    if token_count > TOKEN_LIMIT:
        print(f"  Token limit ({TOKEN_LIMIT}) approached — summarising...")
        new_summary = summarise_messages(messages)
        print(f"  New summary: {new_summary[:80]}...")

        # Keep only recent messages + summary
        recent_messages = messages[-2:]
        latest          = messages[-1].content

        return {
            "messages":        [SystemMessage(content=f"[SUMMARY] {new_summary}")] + recent_messages[-1:],
            "running_summary": new_summary,
            "token_count":     estimate_tokens(recent_messages),
        }
    else:
        latest   = messages[-1].content
        response = f"[Tokens: {token_count}] Responding to: {latest[:60]}"
        return {
            "messages":   [AIMessage(content=response)],
            "token_count": token_count,
        }

g4  = StateGraph(SummaryState)
g4.add_node("chat_summary", conversation_with_summary_node)
g4.add_edge(START, "chat_summary")
g4.add_edge("chat_summary", END)
ckpt4 = MemorySaver()
app4  = g4.compile(checkpointer=ckpt4)
cfg4  = {"configurable": {"thread_id": "summary_demo"}}

long_messages = [
    "What is RAG?",
    "How does chunking work in RAG pipelines?",
    "What is the difference between BM25 and dense retrieval?",
    "How does LangGraph differ from LangChain?",
    "What is the ReAct agent pattern?",     # this should trigger summarisation
]

print("\n  Simulating long conversation — watch for summarisation trigger:")
for msg in long_messages:
    result = app4.invoke(
        {"messages": [HumanMessage(content=msg)], "running_summary": "", "token_count": 0},
        config=cfg4
    )
    print(f"  [{result['token_count']} tokens] {result['messages'][-1].content[:80]}")