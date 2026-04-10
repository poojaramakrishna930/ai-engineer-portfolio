# src/attention_visualization.py
"""
Day 8: Visualise attention weights conceptually
Shows how tokens attend to each other
"""

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    d_k: int,
    mask: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact implementation of the attention formula:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """
    # Step 1: Compute raw scores (how much each query attends to each key)
    scores = Q @ K.T / np.sqrt(d_k)      # shape: (seq_len, seq_len)

    # Step 2: Apply mask (for causal/decoder attention — hide future tokens)
    if mask is not None:
        scores = scores + mask            # masked positions get -inf → 0 after softmax

    # Step 3: Softmax to get attention weights (probabilities)
    attention_weights = np.array([softmax(row) for row in scores])

    # Step 4: Weighted sum of values
    output = attention_weights @ V        # shape: (seq_len, d_model)

    return output, attention_weights


def demonstrate_attention():
    print("=" * 60)
    print("ATTENTION MECHANISM DEMONSTRATION")
    print("=" * 60)

    # Simulate: "The cat sat because it was tired"
    tokens = ["The", "cat", "sat", "because", "it", "was", "tired"]
    seq_len = len(tokens)
    d_k = 4  # tiny dimension for illustration

    np.random.seed(42)

    # Create Q, K, V matrices (in real models, these are learned)
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_k)

    # Manually boost "it" → "cat" attention (to illustrate co-reference)
    # In a real trained model, this emerges from training data
    K[1] = Q[4] * 2  # "cat" key matches "it" query strongly

    output, weights = scaled_dot_product_attention(Q, K, V, d_k)

    print("\nAttention weights matrix (rows=query token, cols=key token):")
    print(f"{'':>10}", end="")
    for t in tokens:
        print(f"{t:>10}", end="")
    print()

    for i, token in enumerate(tokens):
        print(f"{token:>10}", end="")
        for j in range(seq_len):
            bar = "█" * int(weights[i][j] * 20)
            print(f"{weights[i][j]:>10.3f}", end="")
        print()

    print(f"\n🔍 Token 'it' (row 4) attention weights:")
    for j, t in enumerate(tokens):
        bar = "█" * int(weights[4][j] * 30)
        print(f"  → '{t}': {weights[4][j]:.3f} {bar}")

    print("\n💡 Notice 'it' attends most strongly to 'cat' — co-reference resolved!")


def demonstrate_causal_mask():
    """Show how decoder models hide future tokens during training."""
    print("\n" + "=" * 60)
    print("CAUSAL MASKING (Decoder / GPT-style)")
    print("=" * 60)

    tokens = ["The", "cat", "sat"]
    seq_len = len(tokens)

    # Causal mask: upper triangle = -infinity (hidden), lower = 0 (visible)
    mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)

    print("\nCausal mask matrix:")
    print("(0 = can attend, -inf = cannot attend — future token)")
    print(f"{'':>10}", end="")
    for t in tokens:
        print(f"{t:>12}", end="")
    print()
    for i, token in enumerate(tokens):
        print(f"{token:>10}", end="")
        for j in range(seq_len):
            val = mask[i][j]
            display = "0" if val == 0 else "-∞"
            print(f"{display:>12}", end="")
        print()

    print("\n'The' can only see: ['The']")
    print("'cat' can only see: ['The', 'cat']")
    print("'sat' can only see: ['The', 'cat', 'sat']")
    print("\n→ This is why GPT generates LEFT to RIGHT only!")


if __name__ == "__main__":
    demonstrate_attention()
    demonstrate_causal_mask()