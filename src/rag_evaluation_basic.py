from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
import json
import os
import numpy as np

load_dotenv()

# ── Load eval dataset ──────────────────────────────────────────────────────────
eval_path = os.path.join(os.path.dirname(__file__), "..", "data", "eval_dataset.json")
with open(eval_path, "r") as f:
    eval_data = json.load(f)

# ── Load knowledge base ────────────────────────────────────────────────────────
docs_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs.txt")
with open(docs_path, "r") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

docs = [Document(page_content=line, metadata={"doc_id": i})
        for i, line in enumerate(lines)]

# ── Build hybrid retriever ─────────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
persist_dir = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
vectorstore  = Chroma.from_documents(
    docs, embeddings,
    persist_directory=persist_dir,
    collection_name="eval_basic"
)
bm25_retriever  = BM25Retriever.from_documents(docs, k=5)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.5, 0.5]
)

# ── Metric 1: Hit Rate ─────────────────────────────────────────────────────────
# Did retrieval find at least one relevant chunk?
# Simplest eval metric — no LLM needed
def compute_hit_rate(eval_data: list, retriever, k: int = 5) -> float:
    hits = 0
    for item in eval_data:
        results = retriever.invoke(item["question"])
        retrieved_text = " ".join(doc.page_content.lower() for doc in results[:k])

        # Check if any context keyword appears in retrieved text
        found = any(
            kw.lower() in retrieved_text
            for kw in item["context_keywords"]
        )
        if found:
            hits += 1

    return hits / len(eval_data)

# ── Metric 2: MRR — Mean Reciprocal Rank ──────────────────────────────────────
# At what rank does the first relevant result appear?
# MRR = 1.0 means relevant result always at rank 1
# MRR = 0.5 means relevant result on average at rank 2
def compute_mrr(eval_data: list, retriever, k: int = 5) -> float:
    reciprocal_ranks = []
    for item in eval_data:
        results = retriever.invoke(item["question"])
        rr = 0.0
        for rank, doc in enumerate(results[:k]):
            if any(kw.lower() in doc.page_content.lower()
                   for kw in item["context_keywords"]):
                rr = 1.0 / (rank + 1)   # rank is 0-indexed so +1
                break
        reciprocal_ranks.append(rr)
    return np.mean(reciprocal_ranks)

# ── Metric 3: Context Precision (manual, no LLM) ──────────────────────────────
# What fraction of retrieved chunks contain relevant keywords?
def compute_context_precision(eval_data: list, retriever, k: int = 3) -> float:
    precisions = []
    for item in eval_data:
        results = retriever.invoke(item["question"])[:k]
        relevant = sum(
            1 for doc in results
            if any(kw.lower() in doc.page_content.lower()
                   for kw in item["context_keywords"])
        )
        precisions.append(relevant / len(results) if results else 0)
    return np.mean(precisions)

# ── Run evaluation ─────────────────────────────────────────────────────────────
print("=" * 60)
print("RAG Retrieval Evaluation — Manual Metrics")
print("=" * 60)

retrievers = {
    "BM25 only":    bm25_retriever,
    "Dense only":   dense_retriever,
    "Hybrid (RRF)": hybrid_retriever,
}

results_table = []
for name, retriever in retrievers.items():
    hit_rate  = compute_hit_rate(eval_data, retriever, k=5)
    mrr       = compute_mrr(eval_data, retriever, k=5)
    precision = compute_context_precision(eval_data, retriever, k=3)

    results_table.append({
        "name":      name,
        "hit_rate":  hit_rate,
        "mrr":       mrr,
        "precision": precision,
    })

print(f"\n{'Retriever':<20} {'Hit Rate':<12} {'MRR':<12} {'Precision@3'}")
print("-" * 56)
for r in results_table:
    print(f"{r['name']:<20} {r['hit_rate']:.2f}{'':8} "
          f"{r['mrr']:.2f}{'':8} {r['precision']:.2f}")

print("\nKey:")
print("  Hit Rate   : % of questions where at least 1 relevant chunk retrieved")
print("  MRR        : average rank position of first relevant result (1.0 = always rank 1)")
print("  Precision@3: % of top 3 chunks that are relevant")