from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import os

load_dotenv()

# ── Load documents ─────────────────────────────────────────────────────────────
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs.txt")
with open(data_path, "r") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

docs = [Document(page_content=line, metadata={"doc_id": i})
        for i, line in enumerate(lines)]

# ── Build hybrid retriever (from Step 2) ──────────────────────────────────────
bm25_retriever = BM25Retriever.from_documents(docs, k=10)
bm25_retriever.k = 10

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
persist_dir = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
vectorstore = Chroma.from_documents(
    docs,
    embeddings,
    persist_directory=persist_dir,
    collection_name="reranking"
)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.5, 0.5]
)

# ── Load CrossEncoder model ────────────────────────────────────────────────────
# CrossEncoder reads query + document TOGETHER — much more accurate than
# bi-encoder which encodes them separately
# Downloads ~80MB on first run
print("Loading CrossEncoder model...")
cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    max_length=512
)
print("CrossEncoder loaded\n")


# ── Full reranking pipeline ────────────────────────────────────────────────────
def retrieve_and_rerank(
    query: str,
    retriever,
    reranker: CrossEncoder,
    top_n: int = 3
) -> list[tuple[Document, float]]:
    """
    Full production retrieval pipeline:
    1. Retrieve broad set of candidates (k=10)
    2. Rerank with CrossEncoder for precision
    3. Return top_n after reranking
    """
    # Step 1: Retrieve candidates broadly
    candidates = retriever.invoke(query)
    print(f"  Retrieved {len(candidates)} candidates")

    # Step 2: Score each candidate with CrossEncoder
    # CrossEncoder takes [query, document] pairs and returns relevance scores
    pairs = [[query, doc.page_content] for doc in candidates]
    scores = reranker.predict(pairs)

    # Step 3: Zip docs with scores and sort descending
    scored_docs = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return scored_docs[:top_n]


# ── Test the full pipeline ─────────────────────────────────────────────────────
queries = [
    "How do agents decide which tool to use?",
    "What is the difference between RAG and fine-tuning?",
    "How does HNSW enable fast vector search?",
]

for query in queries:
    print(f"\nQuery: '{query}'")
    print("-" * 60)

    # Without reranking — hybrid only
    hybrid_results = hybrid_retriever.invoke(query)
    print("Before reranking (hybrid top 3):")
    for doc in hybrid_results[:3]:
        print(f"  [{doc.metadata['doc_id']}] {doc.page_content[:80]}...")

    # With reranking — hybrid + CrossEncoder
    print("\nAfter reranking (CrossEncoder top 3):")
    reranked = retrieve_and_rerank(query, hybrid_retriever, cross_encoder, top_n=3)
    for doc, score in reranked:
        print(f"  Score={score:.4f} | [{doc.metadata['doc_id']}] {doc.page_content[:80]}...")

# ── Show why reranking matters ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Why CrossEncoder scores differ from embedding similarity:")
print("=" * 60)
print("""
Bi-encoder (embedding model):
  query → [embed] → query_vector
  doc   → [embed] → doc_vector
  score = cosine_similarity(query_vector, doc_vector)
  Problem: query and doc are encoded independently
           — no cross-attention between them

CrossEncoder:
  [query + doc] → [transformer] → relevance_score
  Query and doc are processed TOGETHER
  — full attention between every query token and doc token
  — much more accurate but too slow for full corpus search
  
Best practice: retrieve broadly (k=20) then rerank narrowly (top 3)
""")