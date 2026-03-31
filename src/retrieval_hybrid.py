from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
import os

load_dotenv()

# ── Load documents ─────────────────────────────────────────────────────────────
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs.txt")
with open(data_path, "r") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

docs = [Document(page_content=line, metadata={"doc_id": i})
        for i, line in enumerate(lines)]

# ── Build BM25 retriever ───────────────────────────────────────────────────────
bm25_retriever = BM25Retriever.from_documents(docs, k=5)
bm25_retriever.k = 5

# ── Build dense retriever ──────────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
persist_dir = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
vectorstore = Chroma.from_documents(
    docs,
    embeddings,
    persist_directory=persist_dir,
    collection_name="hybrid_search"
)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ── Build hybrid retriever using EnsembleRetriever ────────────────────────────
# weights must sum to 1.0
# 0.5/0.5 = equal weight to both
# 0.3/0.7 = favour dense (semantic) more
# 0.7/0.3 = favour BM25 (keyword) more
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.5, 0.5]      # equal weight — good default
)

# ── Compare all 3 retrievers on same queries ───────────────────────────────────
queries = [
    ("Keyword query",  "HNSW approximate nearest neighbour search"),
    ("Semantic query", "How do machines understand human language?"),
    ("Mixed query",    "LangChain agents using tools to complete tasks"),
]

for query_type, query in queries:
    print(f"\n{'=' * 60}")
    print(f"{query_type}: '{query}'")
    print("=" * 60)

    # BM25 only
    bm25_results = bm25_retriever.invoke(query)
    print(f"\nBM25 only (top 3):")
    for doc in bm25_results[:3]:
        print(f"  [{doc.metadata['doc_id']}] {doc.page_content[:80]}...")

    # Dense only
    dense_results = dense_retriever.invoke(query)
    print(f"\nDense only (top 3):")
    for doc in dense_results[:3]:
        print(f"  [{doc.metadata['doc_id']}] {doc.page_content[:80]}...")

    # Hybrid (RRF fusion)
    hybrid_results = hybrid_retriever.invoke(query)
    print(f"\nHybrid RRF (top 3):")
    for doc in hybrid_results[:3]:
        print(f"  [{doc.metadata['doc_id']}] {doc.page_content[:80]}...")

# ── How RRF works — explained in code ─────────────────────────────────────────
print("\n" + "=" * 60)
print("How Reciprocal Rank Fusion (RRF) works:")
print("=" * 60)

def reciprocal_rank_fusion(
    bm25_results: list,
    dense_results: list,
    k: int = 60             # RRF constant — 60 is standard default
) -> list:
    """
    Manual RRF implementation to show exactly what EnsembleRetriever does.
    Score = sum of 1/(k + rank) across all retriever lists.
    Higher score = better combined rank.
    """
    scores = {}

    for rank, doc in enumerate(bm25_results):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)

    for rank, doc in enumerate(dense_results):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)

    # Sort by combined RRF score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked

sample_query = "How do agents use tools?"
bm25_r  = bm25_retriever.invoke(sample_query)
dense_r = dense_retriever.invoke(sample_query)
fused   = reciprocal_rank_fusion(bm25_r, dense_r)

print(f"\nQuery: '{sample_query}'")
print("\nRRF scores (higher = more relevant across both retrievers):")
for content, score in fused[:3]:
    print(f"  Score={score:.4f} | {content[:80]}...")

print("\nKey insight: a doc ranked #1 by BM25 AND #1 by dense")
print("gets double the RRF score vs a doc only found by one retriever.")
