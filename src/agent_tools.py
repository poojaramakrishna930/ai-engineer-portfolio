from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from typing import Optional
import os
import json
import math

load_dotenv()

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the AI knowledge base for information about RAG, Langchain, Langgraph,
    Vector Databases, Embeddings and agentic AI concepts. Use this when the user asks about
    AI engineering topics. Returns the most relevant passages found.
    """
    #Load Knowledge Base
    docs_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs.txt")
    with open(docs_path, "r") as f:        
        lines = [line.strip() for line in f.readlines() if line.strip()]

    docs = [Document(page_content=line, metadata={"doc_id": i})
            for i, line in enumerate(lines)]

    #Build Hybrid Retriever
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    persist_dir = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
    vectorstore  = Chroma.from_documents(
        docs, embeddings,
        persist_directory=persist_dir,
        collection_name="agent_kb"
    )
    bm25    = BM25Retriever.from_documents(docs, k=3)
    dense   = vectorstore.as_retriever(search_kwargs={"k": 3})
    hybrid  = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5])

    results = hybrid.invoke(query)
    if not results:
        return "No relevant information found in the knowledge base."
    
    # Format results for LLM consumption
    formatted = []
    for i, doc in enumerate(results[:3]):
        formatted.append(f"[Result {i+1}]: {doc.page_content}")
    return "\n\n".join(formatted)

@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.
    Use this for any arithmetic, percentages, or numerical calculations.
    Input must be a valid Python math expression as a string.
    Examples: '2 + 2', '100 * 0.15', 'math.sqrt(144)'
    """
    try:
        # Safe eval — only allow math operations
        allowed_names = {k: v for k, v in math.__dict__.items()
                         if not k.startswith("__")}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"
    
@tool
def get_current_date() -> str:
    """
    Get today's date and current time.
    Use this when the user asks about the current date, time, or
    how many days until a deadline.
    """
    from datetime import datetime
    now = datetime.now()
    return f"Current date and time: {now.strftime('%A, %B %d, %Y at %H:%M:%S')}"

@tool
def lookup_ai_concept(concept: str) -> str:
    """
    Look up a specific AI or machine learning concept definition.
    Use this for precise technical definitions of terms like:
    RAG, HNSW, BM25, ReAct, embeddings, fine-tuning, etc.
    Returns a structured definition with use cases.
    """
    concepts = {
        "rag": {
            "definition": "Retrieval-Augmented Generation — combines a retrieval system with an LLM",
            "use_case":   "Reduce hallucinations by grounding LLM in real documents",
            "components": "Document store, embedding model, vector DB, LLM"
        },
        "hnsw": {
            "definition": "Hierarchical Navigable Small World — graph-based ANN index",
            "use_case":   "Fast approximate nearest neighbour search in vector DBs",
            "complexity": "O(log n) search vs O(n) brute force"
        },
        "bm25": {
            "definition": "Best Match 25 — probabilistic keyword-based ranking algorithm",
            "use_case":   "Sparse retrieval for exact keyword and technical term matching",
            "strength":   "Finds exact terms dense retrieval misses"
        },
        "react": {
            "definition": "Reasoning + Acting — interleaved thought-action-observation loop",
            "use_case":   "Standard pattern for LLM agents using tools",
            "paper":      "ReAct: Synergizing Reasoning and Acting in Language Models (2022)"
        },
        "langgraph": {
            "definition": "Framework for stateful cyclical agent workflows built on LangChain",
            "use_case":   "Multi-agent systems, retry loops, human-in-the-loop workflows",
            "key_feature": "StateGraph with conditional edges and checkpointing"
        },
    }
    key    = concept.lower().strip()
    result = concepts.get(key)
    if result:
        return json.dumps(result, indent=2)
    return f"Concept '{concept}' not in local lookup. Use search_knowledge_base instead."


# ── Tool introspection — what the LLM sees ────────────────────────────────────
if __name__ == "__main__":
    all_tools = [
        search_knowledge_base,
        calculate,
        get_current_date,
        lookup_ai_concept,
    ]

    print("=" * 65)
    print("Tools registered — what the LLM sees")
    print("=" * 65)
    for t in all_tools:
        print(f"\nName        : {t.name}")
        print(f"Description : {t.description[:100]}...")
        print(f"Args schema : {t.args}")

    print("\n" + "=" * 65)
    print("Testing tools directly")
    print("=" * 65)

    print("\n[calculate]")
    print(calculate.invoke({"expression": "100 * 0.15 + 50"}))

    print("\n[get_current_date]")
    print(get_current_date.invoke({}))

    print("\n[lookup_ai_concept]")
    print(lookup_ai_concept.invoke({"concept": "react"}))

    print("\n[search_knowledge_base]")
    print(search_knowledge_base.invoke({"query": "How do vector databases work?"}))