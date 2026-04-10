# src/transformers_tokenization.py
"""
Day 8: Tokenization deep dive
Understand how text becomes tokens and how that affects your prompts
"""

import tiktoken
from transformers import AutoTokenizer
import os

# ─────────────────────────────────────────
# PART 1: tiktoken (OpenAI-style tokenizer)
# ─────────────────────────────────────────

def explore_tiktoken():
    print("=" * 60)
    print("PART 1: tiktoken tokenizer (GPT-4 style)")
    print("=" * 60)

    enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding

    examples = [
        "Hello world",
        "The capital of France is Paris.",
        "def calculate_rag_score(retrieved, relevant):",
        "LangGraph StateGraph with TypedDict reducers",
        "नमस्ते दुनिया",  # Hindi — tokens are less efficient for non-English
        "🤖🔥💡",          # Emojis
    ]

    for text in examples:
        tokens = enc.encode(text)
        decoded = [enc.decode([t]) for t in tokens]
        print(f"\nText    : {text!r}")
        print(f"Token IDs: {tokens}")
        print(f"Tokens  : {decoded}")
        print(f"Count   : {len(tokens)} tokens")


# ─────────────────────────────────────────
# PART 2: HuggingFace tokenizer
# ─────────────────────────────────────────

def explore_hf_tokenizer():
    print("\n" + "=" * 60)
    print("PART 2: HuggingFace tokenizer (sentence-transformers)")
    print("=" * 60)

    # Using a small, fast model — no GPU needed
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    texts = [
        "RAG pipeline with vector database retrieval",
        "LangGraph enables stateful multi-agent workflows",
    ]

    for text in texts:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.encode(text)
        print(f"\nText  : {text}")
        print(f"Tokens: {tokens}")
        print(f"IDs   : {ids}")
        print(f"Count : {len(tokens)} tokens")
        print(f"Note  : [CLS] and [SEP] added → {len(ids)} total IDs")


# ─────────────────────────────────────────
# PART 3: Token counting for RAG pipelines
# ─────────────────────────────────────────

def token_budget_calculator():
    print("\n" + "=" * 60)
    print("PART 3: Token budget calculator for RAG")
    print("=" * 60)

    enc = tiktoken.get_encoding("cl100k_base")

    # Simulate a RAG prompt
    system_prompt = """You are a helpful AI assistant. Answer questions 
    based ONLY on the provided context. If the answer is not in the 
    context, say 'I don't have enough information to answer that.'"""

    context_chunks = [
        "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        "LangGraph extends LangChain with graph-based workflows and persistent state.",
        "StateGraph in LangGraph allows you to define nodes and edges for agent control flow.",
    ]

    user_question = "What is LangGraph and how does it relate to LangChain?"

    # Build the full prompt
    context_str = "\n\n".join(
        [f"Context {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)]
    )
    full_prompt = f"{system_prompt}\n\n{context_str}\n\nQuestion: {user_question}"

    # Count tokens
    system_tokens = len(enc.encode(system_prompt))
    context_tokens = len(enc.encode(context_str))
    question_tokens = len(enc.encode(user_question))
    total_tokens = len(enc.encode(full_prompt))

    print(f"System prompt : {system_tokens:>6} tokens")
    print(f"Context chunks: {context_tokens:>6} tokens")
    print(f"User question : {question_tokens:>6} tokens")
    print(f"{'─'*30}")
    print(f"Total prompt  : {total_tokens:>6} tokens")
    print(f"\nContext window usage:")
    print(f"  GPT-4o (128k)   → {total_tokens/128000*100:.2f}% used")
    print(f"  MiniLM (512)    → {total_tokens/512*100:.1f}% — would EXCEED limit!")
    print(f"\n⚠️  This is why chunk size matters in RAG!")


if __name__ == "__main__":
    explore_tiktoken()
    explore_hf_tokenizer()
    token_budget_calculator()