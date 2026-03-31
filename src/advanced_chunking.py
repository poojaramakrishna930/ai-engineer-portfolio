from dotenv import load_dotenv
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

load_dotenv()

# ── Load markdown file ────────────────────────────────────────────────────────
md_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_markdown.md")
with open(md_path, "r") as f:
    markdown_text = f.read()

# ── Step 1: Split by markdown headers first ───────────────────────────────────
# This preserves section context as metadata on every chunk
headers_to_split = [
    ("#",   "heading1"),   # H1 → stored as metadata key "heading1"
    ("##",  "heading2"),   # H2 → stored as metadata key "heading2"
    ("###", "heading3"),   # H3 → stored as metadata key "heading3"
]

header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split,
    strip_headers=False    # keep the header text inside the chunk content too
)
header_chunks = header_splitter.split_text(markdown_text)

print(f"After header splitting: {len(header_chunks)} chunks\n")
for chunk in header_chunks:
    print(f"Metadata : {chunk.metadata}")
    print(f"Content  : {chunk.page_content[:100]}...")
    print()

# ── Step 2: Further split large sections recursively ─────────────────────────
# Header chunks can still be large — recursively split them further
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
final_chunks = recursive_splitter.split_documents(header_chunks)

print(f"After recursive split: {len(final_chunks)} final chunks")
print(f"\nSample chunk with rich metadata:")
sample = final_chunks[2] if len(final_chunks) > 2 else final_chunks[0]
print(f"  Metadata : {sample.metadata}")
print(f"  Content  : {sample.page_content}")

# ── Step 3: Store in Chroma and search with metadata filter ──────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

persist_dir = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
store = Chroma.from_documents(
    final_chunks,
    embeddings,
    persist_directory=persist_dir,
    collection_name="markdown_chunks"
)

# ── Search only within a specific section ────────────────────────────────────
query = "How does the retrieval stage work?"
print(f"\nQuery: '{query}'")

# Without filter — searches everything
all_results = store.similarity_search(query, k=2)
print(f"\nWithout filter ({len(all_results)} results):")
for doc in all_results:
    print(f"  [{doc.metadata.get('heading2','?')}] {doc.page_content[:80]}...")

# With filter — only search within "How RAG Works" section
filtered_results = store.similarity_search(
    query,
    k=2,
    filter={"heading2": "How RAG Works"}
)
print(f"\nWith filter heading2='How RAG Works' ({len(filtered_results)} results):")
for doc in filtered_results:
    print(f"  [{doc.metadata.get('heading3','?')}] {doc.page_content[:80]}...")

print("\nKey takeaway: structure-aware chunking + metadata filtering")
print("= much more precise retrieval than plain text splitting")