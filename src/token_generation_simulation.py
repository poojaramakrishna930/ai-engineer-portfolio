# src/token_generation_simulation.py
"""
Day 8: Simulate how LLMs generate tokens
Understand temperature, top-k, top-p sampling
"""

import numpy as np
from typing import Literal


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Apply temperature scaling then softmax."""
    scaled = logits / temperature
    exp = np.exp(scaled - np.max(scaled))  # numerical stability
    return exp / exp.sum()


def top_k_filter(probs: np.ndarray, k: int) -> np.ndarray:
    """Zero out all but top-k probabilities."""
    if k >= len(probs):
        return probs
    threshold = np.sort(probs)[-k]
    filtered = np.where(probs >= threshold, probs, 0.0)
    return filtered / filtered.sum()  # renormalize


def top_p_filter(probs: np.ndarray, p: float) -> np.ndarray:
    """Nucleus sampling: keep smallest set with cumulative prob >= p."""
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumulative = np.cumsum(sorted_probs)

    # Find cutoff: include tokens until cumulative prob >= p
    cutoff_idx = np.searchsorted(cumulative, p) + 1
    top_indices = sorted_indices[:cutoff_idx]

    filtered = np.zeros_like(probs)
    filtered[top_indices] = probs[top_indices]
    return filtered / filtered.sum()


def simulate_sampling():
    print("=" * 60)
    print("TOKEN SAMPLING STRATEGIES SIMULATION")
    print("=" * 60)

    # Simulate vocabulary probabilities (simplified)
    vocab = ["Paris", "Lyon", "France", "the", "city", "capital", "beautiful", "nice", "grande", "is"]
    # Raw logits (what the model actually outputs before softmax)
    raw_logits = np.array([4.5, 2.1, 1.8, 1.2, 0.9, 0.8, 0.5, 0.3, 0.1, 0.05])

    print(f"\nContext: 'The capital of France is ___'")
    print(f"\nVocabulary options: {vocab}")
    print(f"Raw logits:         {raw_logits}")

    # Temperature comparison
    print("\n--- TEMPERATURE EFFECT ---")
    for temp in [0.1, 0.5, 1.0, 1.5, 2.0]:
        probs = softmax(raw_logits, temperature=temp)
        top_token = vocab[np.argmax(probs)]
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        print(f"T={temp}: top='{top_token}' p={probs[0]:.3f}, entropy={entropy:.3f} {'← deterministic' if temp < 0.3 else '← creative' if temp > 1.3 else ''}")

    # Top-K
    print("\n--- TOP-K SAMPLING (base temp=1.0) ---")
    base_probs = softmax(raw_logits, temperature=1.0)
    for k in [1, 3, 5, 10]:
        filtered = top_k_filter(base_probs.copy(), k=k)
        active = sum(1 for p in filtered if p > 0)
        print(f"K={k:2d}: {active} tokens active, top probs: {sorted(filtered[filtered>0], reverse=True)[:3]}")

    # Top-P (nucleus)
    print("\n--- TOP-P (NUCLEUS) SAMPLING ---")
    for p in [0.5, 0.7, 0.9, 0.95]:
        filtered = top_p_filter(base_probs.copy(), p=p)
        active = sum(1 for prob in filtered if prob > 0)
        print(f"P={p}: {active} tokens in nucleus")

    # Greedy vs sampling comparison
    print("\n--- GREEDY vs SAMPLING (10 runs) ---")
    print("Greedy (always deterministic):")
    greedy_results = [vocab[np.argmax(base_probs)]] * 5
    print(f"  {greedy_results}")

    print("Sampling with T=0.8 (varies each run):")
    np.random.seed(None)
    sample_results = [
        vocab[np.random.choice(len(vocab), p=softmax(raw_logits, 0.8))]
        for _ in range(10)
    ]
    print(f"  {sample_results}")


def explain_context_window():
    print("\n" + "=" * 60)
    print("CONTEXT WINDOW ARITHMETIC")
    print("=" * 60)

    models = {
        "GPT-3.5-turbo": 16_385,
        "GPT-4o": 128_000,
        "Claude 3.5 Sonnet": 200_000,
        "LLaMA 3.1 70B": 128_000,
        "Mistral 7B": 32_768,
        "all-MiniLM-L6-v2 (embedding)": 512,
    }

    print(f"\n{'Model':<30} {'Context':<12} {'~Words':<12} {'RAG chunks (500tok)'}")
    print("-" * 70)
    for model, tokens in models.items():
        words = tokens * 0.75
        chunks = tokens // 500
        print(f"{model:<30} {tokens:<12,} {words:<12,.0f} {chunks}")

    print("\n💡 Key insight: embedding models have tiny context windows!")
    print("   → Why we split documents into chunks before embedding (Day 3)")


if __name__ == "__main__":
    simulate_sampling()
    explain_context_window()