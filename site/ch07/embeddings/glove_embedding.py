"""
GloVe Embedding — Loading and Querying Pretrained Vectors
==========================================================

Demonstrates how to:
    1. Load pretrained GloVe embeddings via torchtext
    2. Retrieve word vectors
    3. Compute cosine similarity between words
    4. Find nearest neighbors
    5. Perform word analogy tasks (king - man + woman ≈ queen)

Requirements:
    pip install torchtext torch

Note:
    First run downloads GloVe vectors (~862 MB for 6B/100d).
"""

import torch
import torchtext.vocab as vocab


def load_glove(name: str = "6B", dim: int = 100) -> vocab.GloVe:
    """Load pretrained GloVe vectors."""
    print(f"Loading GloVe (name={name}, dim={dim})...")
    glove = vocab.GloVe(name=name, dim=dim)
    print(f"  Loaded {len(glove.itos)} word vectors of dimension {dim}.")
    return glove


def find_closest_words(
    word_vector: torch.Tensor, glove: vocab.GloVe, top_k: int = 5
) -> list[tuple[str, float]]:
    """Return top-k closest words by cosine similarity."""
    sims = torch.nn.functional.cosine_similarity(
        word_vector.unsqueeze(0), glove.vectors, dim=1
    )
    values, indices = torch.topk(sims, k=top_k)
    return [(glove.itos[idx], val.item()) for idx, val in zip(indices, values)]


def analogy(
    a: str, b: str, c: str, glove: vocab.GloVe, top_k: int = 5
) -> list[tuple[str, float]]:
    """
    Solve: a is to b as c is to ?
    Vector arithmetic: result ≈ b - a + c

    Example: analogy("man", "king", "woman") ≈ "queen"
    """
    vec = glove[b] - glove[a] + glove[c]
    # Exclude input words from results
    exclude = {a, b, c}
    results = find_closest_words(vec, glove, top_k=top_k + len(exclude))
    return [(w, s) for w, s in results if w not in exclude][:top_k]


def main():
    glove = load_glove(name="6B", dim=100)

    # --- Word vector ---
    print("\n--- Word vector for 'hello' ---")
    hello_vec = glove["hello"]
    print(f"  Shape: {hello_vec.shape}")
    print(f"  First 5 dims: {hello_vec[:5].tolist()}")

    # --- Nearest neighbors ---
    print("\n--- Words most similar to 'hello' ---")
    for word, sim in find_closest_words(glove["hello"], glove, top_k=5):
        print(f"  {word:15s}  cosine={sim:.4f}")

    # --- Pairwise similarity ---
    print("\n--- Pairwise cosine similarities ---")
    pairs = [("king", "queen"), ("king", "apple"), ("cat", "dog"), ("good", "bad")]
    for w1, w2 in pairs:
        sim = torch.nn.functional.cosine_similarity(
            glove[w1].unsqueeze(0), glove[w2].unsqueeze(0)
        ).item()
        print(f"  sim({w1}, {w2}) = {sim:.4f}")

    # --- Analogy ---
    print("\n--- Analogy: man → king, woman → ? ---")
    for word, sim in analogy("man", "king", "woman", glove, top_k=5):
        print(f"  {word:15s}  cosine={sim:.4f}")

    print("\n--- Analogy: paris → france, tokyo → ? ---")
    for word, sim in analogy("paris", "france", "tokyo", glove, top_k=5):
        print(f"  {word:15s}  cosine={sim:.4f}")


if __name__ == "__main__":
    main()
