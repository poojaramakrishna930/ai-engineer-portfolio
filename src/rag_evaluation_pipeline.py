from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import json
import os
import numpy as np
from datetime import datetime

load_dotenv()

# ── Load everything ────────────────────────────────────────────────────────────
eval_path = os.path.join(os.path.dirname(__file__), "..", "data", "eval_dataset.json")
with open(eval_path, "r") as f:
    eval_data = json.load(f)

docs_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs.txt")
with open(docs_path, "r") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

docs = [Document(page_content=line, metadata={"doc_id": i})
        for i, line in enumerate(lines)]

# ── Build full pipeline ────────────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
persist_dir = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
vectorstore = Chroma.from_documents(
    docs, embeddings,
    persist_directory=persist_dir,
    collection_name="eval_pipeline"
)
bm25_retriever  = BM25Retriever.from_documents(docs, k=10)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.5, 0.5]
)

print("Loading CrossEncoder for reranking...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── Evaluation functions ───────────────────────────────────────────────────────
def retrieve_with_reranking(
    question: str,
    retriever,
    reranker: CrossEncoder,
    final_k: int = 3
) -> list[Document]:
    candidates = retriever.invoke(question)
    pairs  = [[question, doc.page_content] for doc in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:final_k]]

def compute_keyword_recall(
    retrieved_docs: list[Document],
    ground_truth: str,
    keywords: list[str]
) -> float:
    retrieved_text = " ".join(d.page_content.lower() for d in retrieved_docs)
    found = sum(1 for kw in keywords if kw.lower() in retrieved_text)
    return found / len(keywords) if keywords else 0.0

def compute_answer_coverage(answer: str, ground_truth: str) -> float:
    """
    Simple token overlap between answer and ground truth.
    In production use ROUGE or BERTScore for better accuracy.
    """
    answer_tokens = set(answer.lower().split())
    truth_tokens  = set(ground_truth.lower().split())
    if not truth_tokens:
        return 0.0
    overlap = answer_tokens & truth_tokens
    return len(overlap) / len(truth_tokens)

# ── Run full evaluation ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("End-to-End RAG Pipeline Evaluation")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

eval_results = []

for item in eval_data:
    question        = item["question"]
    ground_truth    = item["ground_truth"]
    keywords        = item["context_keywords"]

    # Retrieve with reranking
    retrieved = retrieve_with_reranking(
        question, hybrid_retriever, reranker, final_k=3
    )

    # Compute metrics
    recall   = compute_keyword_recall(retrieved, ground_truth, keywords)
    coverage = compute_answer_coverage(
        retrieved[0].page_content if retrieved else "",
        ground_truth
    )
    context_text = " ".join(d.page_content for d in retrieved)

    eval_results.append({
        "question":       question,
        "keyword_recall": recall,
        "answer_coverage": coverage,
        "chunks_retrieved": len(retrieved),
        "top_chunk":      retrieved[0].page_content[:80] if retrieved else "none",
    })

# ── Print results dashboard ────────────────────────────────────────────────────
print(f"\n{'Question':<45} {'Recall':<10} {'Coverage':<12} {'Chunks'}")
print("-" * 75)
for r in eval_results:
    q_short = r["question"][:43] + ".." if len(r["question"]) > 43 else r["question"]
    recall_bar  = "█" * int(r["keyword_recall"] * 5) + "░" * (5 - int(r["keyword_recall"] * 5))
    coverage_bar = "█" * int(r["answer_coverage"] * 5) + "░" * (5 - int(r["answer_coverage"] * 5))
    print(f"{q_short:<45} {recall_bar} {r['keyword_recall']:.2f}  "
          f"{coverage_bar} {r['answer_coverage']:.2f}  "
          f"{r['chunks_retrieved']}")

# ── Aggregate scores ───────────────────────────────────────────────────────────
avg_recall   = np.mean([r["keyword_recall"]  for r in eval_results])
avg_coverage = np.mean([r["answer_coverage"] for r in eval_results])

print("\n" + "=" * 70)
print("Aggregate Scores")
print("=" * 70)
print(f"  Avg keyword recall  : {avg_recall:.2f}  "
      f"{'✓ Good' if avg_recall >= 0.7 else '✗ Needs improvement'}")
print(f"  Avg answer coverage : {avg_coverage:.2f}  "
      f"{'✓ Good' if avg_coverage >= 0.3 else '✗ Needs improvement'}")

# ── Actionable improvement suggestions ────────────────────────────────────────
print("\nActionable improvements based on scores:")
if avg_recall < 0.7:
    print("  → Low recall: try hybrid search, increase k, improve chunking")
if avg_coverage < 0.3:
    print("  → Low coverage: chunks too narrow, try larger chunk size or parent-child retrieval")
if avg_recall >= 0.7 and avg_coverage >= 0.3:
    print("  → Retrieval looks good! Next step: evaluate generation with RAGAS + LLM")

# ── Save results to JSON for tracking over time ────────────────────────────────
output_path = os.path.join(
    os.path.dirname(__file__), "..", "data", "eval_results.json"
)
with open(output_path, "w") as f:
    json.dump({
        "timestamp":    datetime.now().isoformat(),
        "avg_recall":   avg_recall,
        "avg_coverage": avg_coverage,
        "per_question": eval_results,
    }, f, indent=2)

print(f"\nResults saved to data/eval_results.json")
print("Track this file over time to measure RAG improvements across iterations.")