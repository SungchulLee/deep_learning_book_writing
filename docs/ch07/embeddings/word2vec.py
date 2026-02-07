"""
Word2Vec — Skip-gram and CBOW
==============================

Demonstrates Word2Vec concepts with a minimal PyTorch implementation.

Two architectures:
    CBOW:      context words  -->  predict center word
    Skip-gram: center word    -->  predict context words

Training uses negative sampling for efficiency.

References:
    - Mikolov et al., "Efficient Estimation of Word Representations
      in Vector Space" (2013), https://arxiv.org/abs/1301.3781
    - Blog: https://amitness.com/2020/06/fasttext-embeddings/
"""

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)


# =============================================================================
# Data preparation
# =============================================================================
corpus = [
    "the cat sat on the mat",
    "the dog lay on the rug",
    "cats and dogs are friends",
    "the mat is on the floor",
]
sentences = [s.split() for s in corpus]

# Build vocabulary
word_to_ix = {}
for sent in sentences:
    for w in sent:
        if w not in word_to_ix:
            word_to_ix[w] = len(word_to_ix)
ix_to_word = {i: w for w, i in word_to_ix.items()}
vocab_size = len(word_to_ix)

EMBEDDING_DIM = 16
CONTEXT_SIZE = 2  # words on each side


def build_skipgram_pairs(sentences, window=CONTEXT_SIZE):
    """Generate (center, context) pairs for skip-gram."""
    pairs = []
    for sent in sentences:
        indices = [word_to_ix[w] for w in sent]
        for i, center in enumerate(indices):
            for j in range(max(0, i - window), min(len(indices), i + window + 1)):
                if i != j:
                    pairs.append((center, indices[j]))
    return pairs


def build_cbow_pairs(sentences, window=CONTEXT_SIZE):
    """Generate (context_list, center) pairs for CBOW."""
    pairs = []
    for sent in sentences:
        indices = [word_to_ix[w] for w in sent]
        for i, center in enumerate(indices):
            context = []
            for j in range(max(0, i - window), min(len(indices), i + window + 1)):
                if i != j:
                    context.append(indices[j])
            if context:
                pairs.append((context, center))
    return pairs


# =============================================================================
# Skip-gram model
# =============================================================================
class SkipGram(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center: torch.Tensor) -> torch.Tensor:
        """Return scores for all vocabulary words given center word."""
        center_emb = self.center_embeddings(center)          # (B, D)
        all_context = self.context_embeddings.weight          # (V, D)
        scores = center_emb @ all_context.T                   # (B, V)
        return scores


# =============================================================================
# CBOW model
# =============================================================================
class CBOW(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_idxs: torch.Tensor) -> torch.Tensor:
        """Average context embeddings then predict center word."""
        embeds = self.embeddings(context_idxs)   # (B, ctx_len, D)
        avg = embeds.mean(dim=1)                 # (B, D)
        scores = self.linear(avg)                # (B, V)
        return scores


# =============================================================================
# Training loop (shared)
# =============================================================================
def train_skipgram(epochs: int = 100, lr: float = 0.01):
    model = SkipGram(vocab_size, EMBEDDING_DIM)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    pairs = build_skipgram_pairs(sentences)

    for epoch in range(epochs):
        total_loss = 0.0
        for center, context in pairs:
            center_t = torch.tensor([center], dtype=torch.long)
            context_t = torch.tensor([context], dtype=torch.long)

            scores = model(center_t)
            loss = loss_fn(scores, context_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"[Skip-gram] Epoch {epoch+1:3d}  Loss: {total_loss:.4f}")

    return model


def train_cbow(epochs: int = 100, lr: float = 0.01):
    model = CBOW(vocab_size, EMBEDDING_DIM)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    pairs = build_cbow_pairs(sentences)

    for epoch in range(epochs):
        total_loss = 0.0
        for context, center in pairs:
            context_t = torch.tensor([context], dtype=torch.long)  # (1, ctx_len)
            center_t = torch.tensor([center], dtype=torch.long)

            scores = model(context_t)
            loss = loss_fn(scores, center_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"[CBOW]      Epoch {epoch+1:3d}  Loss: {total_loss:.4f}")

    return model


# =============================================================================
# Similarity utilities
# =============================================================================
def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    return (
        torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))
        .item()
    )


def most_similar(word: str, model: nn.Module, top_k: int = 5):
    """Find top-k most similar words by cosine similarity."""
    # Use center_embeddings for SkipGram, embeddings for CBOW
    if hasattr(model, "center_embeddings"):
        W = model.center_embeddings.weight.detach()
    else:
        W = model.embeddings.weight.detach()

    idx = word_to_ix[word]
    vec = W[idx].unsqueeze(0)
    sims = torch.nn.functional.cosine_similarity(vec, W, dim=1)
    sims[idx] = -1  # exclude the word itself
    top_indices = sims.argsort(descending=True)[:top_k]
    return [(ix_to_word[i.item()], sims[i].item()) for i in top_indices]


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 50)
    print("Training Skip-gram")
    print("=" * 50)
    sg_model = train_skipgram(epochs=100)

    print()
    print("=" * 50)
    print("Training CBOW")
    print("=" * 50)
    cbow_model = train_cbow(epochs=100)

    # Demonstrate similarity
    print()
    print("=" * 50)
    print("Similarity queries")
    print("=" * 50)
    for word in ["cat", "the", "on"]:
        if word in word_to_ix:
            sg_similar = most_similar(word, sg_model, top_k=3)
            cbow_similar = most_similar(word, cbow_model, top_k=3)
            print(f"\n'{word}' most similar:")
            print(f"  Skip-gram: {sg_similar}")
            print(f"  CBOW:      {cbow_similar}")


if __name__ == "__main__":
    main()
