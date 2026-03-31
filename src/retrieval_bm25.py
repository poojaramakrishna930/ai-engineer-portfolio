from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document   
import os

load_dotenv()

#------1. Load Documents-----------------
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs.txt")
with open(data_path, "r") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

docs = [Document(page_content=line,metadata={"doc_id": i})
        for i, line in enumerate(lines)]

#------2. Create BM25 Retriever-----------------
bm25_retriever = BM25Retriever.from_documents(docs, k=3)

# ── Test 1: Keyword query — BM25 shines here ─────────────────────────────────
query_keyword = "LangGraph StateGraph agents"
print(f"Query 1 (keyword): '{query_keyword}'")
print("-" * 60)
results = bm25_retriever.invoke(query_keyword)
for doc in results:
    print(f"[{doc.metadata['doc_id']}] {doc.page_content}")

# ── Test 2: Semantic query — BM25 weakness ────────────────────────────────────
query_semantic = "How do machines understand human language?"
print(f"\nQuery 2 (semantic): '{query_semantic}'")
print("-" * 60)
results = bm25_retriever.invoke(query_semantic)
for doc in results:
    print(f"  [{doc.metadata['doc_id']}] {doc.page_content}")

print("\nNotice: BM25 struggles with semantic query because")
print("it only matches exact keywords — not meaning.")

# ── Test 3: Exact term — where BM25 beats dense ───────────────────────────────
query_exact = "HNSW approximate nearest neighbour"
print(f"\nQuery 3 (exact term): '{query_exact}'")
print("-" * 60)
results = bm25_retriever.invoke(query_exact)
for doc in results:
    print(f"  [{doc.metadata['doc_id']}] {doc.page_content}")

print("\nKey insight: BM25 finds exact technical terms that")
print("embedding models sometimes miss by over-generalising.")
