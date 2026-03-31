from dotenv import load_dotenv
from langchain.text_splitter import (
    HTMLHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

load_dotenv()

# ── Sample HTML document ──────────────────────────────────────────────────────
# In real projects you fetch this from a URL using requests or WebBaseLoader
html_text = """
<!DOCTYPE html>
<html>
<body>
    <h1>Python for AI Engineers</h1>
    <p>Python is the dominant language for AI and machine learning development.</p>
    <p>Its rich ecosystem of libraries makes it ideal for building AI pipelines.</p>

    <h2>Core Libraries</h2>
    <p>Every AI engineer must know the essential Python libraries for the field.</p>
    <p>These libraries handle everything from data processing to model deployment.</p>

    <h3>LangChain</h3>
    <p>LangChain is a framework for building LLM-powered applications.</p>
    <p>It provides chains, retrievers, agents, and memory components.</p>
    <p>LangChain integrates with most LLM providers and vector databases.</p>

    <h3>HuggingFace Transformers</h3>
    <p>HuggingFace provides thousands of open-source pre-trained models.</p>
    <p>You can load, fine-tune, and deploy models with just a few lines of code.</p>
    <p>The datasets library pairs with transformers for training and evaluation.</p>

    <h2>Vector Databases</h2>
    <p>Vector databases are essential for building RAG pipelines at scale.</p>
    <p>They store embeddings and enable fast semantic similarity search.</p>

    <h3>Chroma</h3>
    <p>Chroma is an open-source vector database that runs locally.</p>
    <p>It requires no server setup and persists data to disk automatically.</p>
    <p>Chroma is the recommended choice for development and prototyping.</p>

    <h3>Pinecone</h3>
    <p>Pinecone is a managed cloud vector database with a generous free tier.</p>
    <p>It handles scaling, replication, and infrastructure automatically.</p>
    <p>Pinecone is the most common production vector database in job descriptions.</p>
</body>
</html>
"""

# ── Step 1: Define which HTML tags to split on ────────────────────────────────
headers_to_split_on = [
    ("h1", "h1"),    # <h1> tag → metadata["h1"]
    ("h2", "h2"),    # <h2> tag → metadata["h2"]
    ("h3", "h3"),    # <h3> tag → metadata["h3"]
]

html_splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

html_chunks = html_splitter.split_text(html_text)

print(f"Total chunks after HTML header splitting: {len(html_chunks)}\n")
print("=" * 60)

for i, chunk in enumerate(html_chunks):
    print(f"Chunk [{i}]")
    print(f"  Metadata : {chunk.metadata}")
    print(f"  Content  : {chunk.page_content[:120]}...")
    print()

# ── Step 2: Further split recursively ────────────────────────────────────────
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)
final_chunks = recursive_splitter.split_documents(html_chunks)

print(f"Total chunks after recursive split: {len(final_chunks)}")
print("\nMetadata preserved on all sub-chunks:")
for chunk in final_chunks[:4]:
    print(f"  {chunk.metadata} → {chunk.page_content[:80]}...")

# ── Step 3: Store and search with section filtering ───────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

persist_dir = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
store = Chroma.from_documents(
    final_chunks,
    embeddings,
    persist_directory=persist_dir,
    collection_name="html_guide"
)

query = "Which vector database should I use for production?"
print(f"\nQuery: '{query}'")

# Without filter — searches all sections
print("\nWithout filter:")
results = store.similarity_search(query, k=2)
for doc in results:
    print(f"  [{doc.metadata}] {doc.page_content[:80]}...")

# Filter to Vector Databases section only
print("\nWith filter h2='Vector Databases':")
filtered = store.similarity_search(
    query,
    k=2,
    filter={"h2": "Vector Databases"}
)
for doc in filtered:
    print(f"  [{doc.metadata}] {doc.page_content[:80]}...")

# ── Step 4: Fetch from a real URL (bonus — real world usage) ─────────────────
print("\n" + "=" * 60)
print("Real world usage — fetch HTML from a URL:")
print("""
from langchain_community.document_loaders import WebBaseLoader

# Load a real webpage
loader = WebBaseLoader("https://docs.langchain.com/docs/")
web_docs = loader.load()

# Then split with HTMLHeaderTextSplitter
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = html_splitter.split_text(web_docs[0].page_content)
""")