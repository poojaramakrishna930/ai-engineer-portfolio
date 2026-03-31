from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import os

load_dotenv()

#------Load Document------------------------------------
data_path = os.path.join(os.path.dirname(__file__),"..", 'data', 'sample_docs.txt')

loader = TextLoader(data_path)
documents = loader.load()
full_text = documents[0].page_content

print(f"Original document: {len(full_text)} characters\n")
print("=" * 60)

#------Strategy 1: Fixed-Size Chunks------------------------------------`
# This strategy splits the text into fixed-size chunks, which is simple 
# but may break sentences or paragraphs.`
fixed_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0,separator="")
fixed_chunks = fixed_splitter.split_text(full_text)

print(f"\nStrategy 1 — Fixed size (chunk_size=200, overlap=0)")
print(f"Total chunks: {len(fixed_chunks)}")
print(f"Last chunk (notice mid-sentence cut):")
print(f"  '{fixed_chunks[-1]}'\n")

#------Strategy 2: Recursive Character Splitter------------------------------------
# This strategy tries to split on natural boundaries (like sentences or paragraphs)
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""],
    length_function=len
)

recursive_chunks = recursive_splitter.split_text(full_text)

print(f"Strategy 2 — Recursive (chunk_size=300, overlap=50)")
print(f"Total chunks: {len(recursive_chunks)}")
print(f"First 2 chunks:")

for i, chunk in enumerate(recursive_chunks[:2]):
    print(f"  [{i}] ({len(chunk)} chars) '{chunk[:100]}...'")

# ── Show overlap effect ───────────────────────────────────────────────────────
print(f"\nOverlap demonstration:")
print(f"  End of chunk 0  : '...{recursive_chunks[0][-50:]}'")
print(f"  Start of chunk 1: '{recursive_chunks[1][:50]}...'")
print(f"  ^ These 50 chars overlap — context preserved across boundary\n")

# ── Strategy 3: Token-based (important for LLM context windows) ───────────────
# Splits by tokens instead of characters — more accurate for LLM limits
token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",
    chunk_size=100,       # 100 tokens not 100 characters
    chunk_overlap=20
)
token_chunks = token_splitter.split_text(full_text)

print(f"Strategy 3 — Token-based (chunk_size=100 tokens)")
print(f"Total chunks: {len(token_chunks)}")
print(f"Why use this: guarantees chunks never exceed LLM token limits\n")

# ── Side-by-side comparison ───────────────────────────────────────────────────
print("=" * 60)
print(f"{'Strategy':<25} {'Chunks':<10} {'Avg size':<12} {'Cuts sentences?'}")
print("-" * 60)

strategies = [
    ("Fixed size",      fixed_chunks),
    ("Recursive",       recursive_chunks),
    ("Token-based",     token_chunks),
]
for name, chunks in strategies:
    avg = sum(len(c) for c in chunks) // len(chunks)
    # check if any chunk ends mid-sentence (no period at end)
    cuts = sum(1 for c in chunks if not c.strip().endswith(('.', '?', '!')))
    print(f"{name:<25} {len(chunks):<10} {avg:<12} {cuts}/{len(chunks)} chunks")


