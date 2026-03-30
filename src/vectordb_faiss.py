from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os
import time

load_dotenv()

# ── 1. Load docs ─────────────────────────────────────────────────────────────
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs.txt")

with open(data_path, "r") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

docs = [Document(page_content=line, metadata={"doc_id": i})
        for i, line in enumerate(lines)]

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# ── 2. Build FAISS index ─────────────────────────────────────────────────────
print("Building FAISS index...")
start = time.time()
faiss_store = FAISS.from_documents(docs, embeddings)
print(f"FAISS index built in {time.time() - start:.2f}s\n")

# ── 3. Similarity search ─────────────────────────────────────────────────────
query = "What is a vector database?"

start = time.time()
results = faiss_store.similarity_search_with_score(query, k=3)
search_time = time.time() - start

print(f"Query: '{query}'")
print(f"Search time: {search_time*1000:.2f}ms")
print("-" * 50)

for doc, score in results:
    print(f"Score : {score:.4f}  (lower = more similar in FAISS/L2)")
    print(f"Text  : {doc.page_content}\n")

# ── 4. Save and reload FAISS index to disk ───────────────────────────────────
# Unlike Chroma, FAISS needs explicit save/load
faiss_dir = os.path.join(os.path.dirname(__file__), "..", "faiss_index")
os.makedirs(faiss_dir, exist_ok=True)

faiss_store.save_local(faiss_dir)
print(f"FAISS index saved to {faiss_dir}")

# Reload — must pass allow_dangerous_deserialization=True (LangChain safety flag)
loaded_faiss = FAISS.load_local(
    faiss_dir,
    embeddings,
    allow_dangerous_deserialization=True
)
print(f"FAISS index reloaded from disk successfully")

# ── 5. Max marginal relevance search ─────────────────────────────────────────
# MMR balances relevance WITH diversity — avoids returning 3 near-identical chunks
print("\nMMR search (diverse results):")
mmr_results = faiss_store.max_marginal_relevance_search(query, k=3, fetch_k=10)
for doc in mmr_results:
    print(f"  → {doc.page_content}")