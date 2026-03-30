from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import time

load_dotenv()

data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs.txt")
with open(data_path, "r") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

docs = [Document(page_content=line, metadata={"doc_id": i})
        for i, line in enumerate(lines)]

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

query = "How do agents use tools to complete tasks?"

results_table = []

# ── Chroma ───────────────────────────────────────────────────────────────────
t0 = time.time()
chroma_store = Chroma.from_documents(docs, embeddings, collection_name="compare_test")
chroma_build = time.time() - t0

t0 = time.time()
chroma_results = chroma_store.similarity_search(query, k=2)
chroma_search = time.time() - t0

results_table.append({
    "db": "Chroma",
    "build_ms": round(chroma_build * 1000),
    "search_ms": round(chroma_search * 1000),
    "top_result": chroma_results[0].page_content[:70] + "..."
})

# ── FAISS ────────────────────────────────────────────────────────────────────
t0 = time.time()
faiss_store = FAISS.from_documents(docs, embeddings)
faiss_build = time.time() - t0

t0 = time.time()
faiss_results = faiss_store.similarity_search(query, k=2)
faiss_search = time.time() - t0

results_table.append({
    "db": "FAISS",
    "build_ms": round(faiss_build * 1000),
    "search_ms": round(faiss_search * 1000),
    "top_result": faiss_results[0].page_content[:70] + "..."
})

# ── Print comparison ─────────────────────────────────────────────────────────
print(f"\nQuery: '{query}'\n")
print(f"{'DB':<10} {'Build (ms)':<14} {'Search (ms)':<14} Top Result")
print("-" * 90)
for r in results_table:
    print(f"{r['db']:<10} {r['build_ms']:<14} {r['search_ms']:<14} {r['top_result']}")

print("\nKey takeaway:")
print("  Chroma  → easier metadata filtering, disk persistence, better for RAG dev")
print("  FAISS   → faster raw search, better for large-scale or latency-critical apps")
print("  Pinecone → use in production when you need cloud scale + no infra management")
print("  Weaviate → use when you need hybrid search (keyword + semantic) at scale")