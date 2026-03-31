from dotenv import load_dotenv
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

load_dotenv()

# ── Sample markdown document ──────────────────────────────────────────────────
# In real projects this comes from a file — for clarity we define it inline here
markdown_text = """
# LangChain Framework Guide

LangChain is a framework for building LLM-powered applications.
It provides tools for chaining prompts, managing memory, and integrating retrievers.

## Core Components

LangChain has several core components that work together to build pipelines.
Understanding each component is essential for building production RAG systems.

### Chains

Chains are sequences of calls to LLMs or other utilities.
They can be simple single-step or complex multi-step pipelines.
LangChain Expression Language (LCEL) is the modern way to define chains.

### Retrievers

Retrievers fetch relevant documents from a knowledge base.
They are the bridge between your vector store and your LLM.
A retriever takes a query string and returns a list of documents.

### Memory

Memory allows chains and agents to remember previous interactions.
Short-term memory stores the current conversation history.
Long-term memory persists information across multiple sessions.

## Agents

Agents use LLMs to decide which actions to take and in what order.
They are more flexible than chains because they can handle dynamic workflows.

### ReAct Agent

ReAct stands for Reasoning and Acting.
The agent alternates between thinking about what to do and actually doing it.
Each step produces an observation that feeds into the next reasoning step.

### Tool Use

Agents can use tools like web search, calculators, and code interpreters.
Tools are functions the agent can call to interact with the outside world.
The agent decides which tool to use based on the current task.
"""

# ── Step 1: Define which headers to split on ─────────────────────────────────
# Each tuple is (header_marker, metadata_key_name)
# The metadata_key_name is what appears in chunk.metadata
headers_to_split_on = [
    ("#",   "h1"),    # H1 → metadata["h1"]
    ("##",  "h2"),    # H2 → metadata["h2"]
    ("###", "h3"),    # H3 → metadata["h3"]
]

md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False    # keep header text inside chunk content
)

header_chunks = md_splitter.split_text(markdown_text)

print(f"Total chunks after header splitting: {len(header_chunks)}\n")
print("=" * 60)

for i, chunk in enumerate(header_chunks):
    print(f"Chunk [{i}]")
    print(f"  Metadata : {chunk.metadata}")
    print(f"  Content  : {chunk.page_content[:120]}...")
    print()

# ── Step 2: Further split large chunks recursively ────────────────────────────
# Header chunks can still be long — split them further while keeping metadata
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)
final_chunks = recursive_splitter.split_documents(header_chunks)

print(f"Total chunks after recursive split: {len(final_chunks)}\n")
print("Notice metadata is preserved on every sub-chunk:")
for chunk in final_chunks[:4]:
    print(f"  {chunk.metadata} → {chunk.page_content[:80]}...")

# ── Step 3: Store in Chroma and filter by section ────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

persist_dir = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
store = Chroma.from_documents(
    final_chunks,
    embeddings,
    persist_directory=persist_dir,
    collection_name="markdown_guide"
)

query = "How do agents decide what to do?"
print(f"\nQuery: '{query}'")

# Search everything
print("\nWithout filter:")
results = store.similarity_search(query, k=2)
for doc in results:
    print(f"  [{doc.metadata}] {doc.page_content[:80]}...")

# Search only inside the Agents section
print("\nWith filter h2='Agents':")
filtered = store.similarity_search(
    query,
    k=2,
    filter={"h2": "Agents"}
)
for doc in filtered:
    print(f"  [{doc.metadata}] {doc.page_content[:80]}...")

# Search only inside a specific subsection
print("\nWith filter h3='ReAct Agent':")
subsection = store.similarity_search(
    query,
    k=2,
    filter={"h3": "ReAct Agent"}
)
for doc in subsection:
    print(f"  [{doc.metadata}] {doc.page_content[:80]}...")