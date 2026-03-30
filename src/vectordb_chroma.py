from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import shutil

load_dotenv()

#---1. Load sample docs-----------------
data_path=os.path.join(os.path.dirname(__file__),"..","data","sample_docs.txt")

with open(data_path, "r") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

#---2. Create Document objects-----------------
# Wrap each line as a LangChain Document with metadata
# Metadata is KEY — lets you filter results by topic, source, date etc.
docs = [] 
for i, line in enumerate(lines):
    # Tag each doc with a topic — simulates real-world metadata
    if any(w in line.lower() for w in ["rag", "retrieval", "chunk", "vector"]):
        topic = "retrieval"
    elif any(w in line.lower() for w in ["agent", "react", "langgraph", "langchain"]):
        topic = "agents"
    elif any(w in line.lower() for w in ["llm", "language model", "gpt", "fine-tun", "prompt"]):
        topic = "llm"
    else:
        topic = "general"

    docs.append(Document(
        page_content=line,
        metadata={"doc_id": i, "topic": topic}
    ))

print(f"Loaded {len(docs)} documents")
print(f"Topics: {set(d.metadata['topic'] for d in docs)}\n")

#---3. Create embeddings + Store in chroma-----------------
# Using HuggingFaceEmbeddings with a small model for demo purposes
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

persist_dir = os.path.join(os.path.dirname(__file__),"..", "chroma_db")
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)
vectorstore = Chroma.from_documents(
    documents=docs, embedding=embeddings, persist_directory=persist_dir, # saves to disk — survives restarts
    collection_name="ai-concepts"
)

print(f"Stored {vectorstore._collection.count()} vectors in Chroma\n")

#--3. Basic similarity search-----------------
query = "What is a vector database?"
print(f"Query: '{query}'")
print("-" * 50)

results = vectorstore.similarity_search_with_score(query, k=3)
for doc, score in results:
    print(f"Score: {score:.4f} (lower = more similar in chroma)")
    print(f"Topic: {doc.metadata['topic']} ") 
    print(f"Content: {doc.page_content}...")

for doc, score in results:
    print(f"Score : {score:.4f}  (lower = more similar in chroma)")
    print(f"Text  : {doc.page_content}\n")

# ── 4. Metadata filtering — powerful, often missed by beginners ──────────────
print("Filtered search — only 'retrieval' topic:")
print("-" * 50)

filtered_results = vectorstore.similarity_search(
    query,
    k=3,
    filter={"topic": "retrieval"}   # only search within this topic
)

for doc in filtered_results:
    print(f"[{doc.metadata['topic']}] {doc.page_content}")

# ── 5. Load existing Chroma DB (no re-embedding needed) ─────────────────────
print("\nLoading existing Chroma DB from disk...")
loaded_store = Chroma(
    persist_directory=persist_dir,
    embedding_function=embeddings,
    collection_name="ai-concepts"
)
print(f"Loaded {loaded_store._collection.count()} vectors from disk — no re-embedding!")