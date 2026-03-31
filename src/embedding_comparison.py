from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import os
import time
import numpy as np

load_dotenv()

data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs.txt")
loader = TextLoader(data_path)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# ── Test 2 embedding models on same query ─────────────────────────────────────
models = [
    {
        "name": "all-MiniLM-L6-v2",
        "description": "Fast, small, good quality",
        "dims": 384
    },
    {
        "name": "BAAI/bge-small-en-v1.5",
        "description": "Better retrieval quality, same speed",
        "dims": 384
    }
]

query = "How do agents use tools to complete tasks?"
print(f"Query: '{query}'\n")
print("=" * 70)

for model_info in models:
    model_name = model_info["name"]
    print(f"\nModel: {model_name}")
    print(f"Description: {model_info['description']} | Dimensions: {model_info['dims']}")

    # ── Embed and store ───────────────────────────────────────────────────────
    t0 = time.time()
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}  # important for cosine similarity
    )

    # Use unique collection per model to avoid conflicts
    safe_name = model_name.replace("/", "_").replace("-", "_")
    store = Chroma.from_documents(
        chunks,
        embeddings,
        collection_name=f"compare_{safe_name}"
    )
    build_time = time.time() - t0

    # ── Search ────────────────────────────────────────────────────────────────
    t0 = time.time()
    results = store.similarity_search_with_score(query, k=3)
    search_time = time.time() - t0

    print(f"Build time : {build_time:.2f}s | Search time: {search_time*1000:.1f}ms")
    print(f"Top results:")
    for i, (doc, score) in enumerate(results):
        print(f"  [{i+1}] score={score:.4f} | {doc.page_content[:80]}...")

    # ── Show what a single embedding vector looks like ────────────────────────
    sample_vector = embeddings.embed_query("test")
    print(f"Vector sample: [{sample_vector[0]:.4f}, {sample_vector[1]:.4f}, "
          f"{sample_vector[2]:.4f}, ... {len(sample_vector)} dimensions]")

print("\n" + "=" * 70)
print("Key insight: normalize_embeddings=True ensures cosine similarity works correctly")
print("Without it, dot product is used instead — different results for same query")

