# Dense Retrieval

## Learning Objectives

- Understand bi-encoder architecture for dense retrieval
- Compare key dense retrieval models
- Implement contrastive training for retrieval

## Bi-Encoder Architecture

Dense retrieval encodes queries and documents into a shared embedding space using separate encoders:

$$\text{score}(q, d) = \text{sim}(E_q(q), E_d(d))$$

where $E_q$ and $E_d$ are encoder networks and $\text{sim}$ is typically cosine similarity or dot product.

At retrieval time, document embeddings are pre-computed and indexed for efficient search via approximate nearest neighbors (ANN).

## Key Models

| Model | Embedding Dim | Training Data | Key Feature |
|-------|--------------|---------------|-------------|
| DPR | 768 | NQ, TriviaQA | Dual BERT encoders |
| E5 | 1024 | Diverse web pairs | Instruction-tuned |
| BGE | 1024 | C-MTEB | Multilingual |
| GTE | 1024 | Diverse | Good general-purpose |
| OpenAI text-embedding-3 | 3072 | Proprietary | High quality, API |
| Cohere embed-v3 | 1024 | Proprietary | Search-optimized |

## Contrastive Training

Dense retrievers are trained with contrastive loss (InfoNCE):

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(q, d^+) / \tau)}{\exp(\text{sim}(q, d^+) / \tau) + \sum_{d^- \in \mathcal{N}} \exp(\text{sim}(q, d^-) / \tau)}$$

where $d^+$ is the positive (relevant) document, $\mathcal{N}$ are negative documents, and $\tau$ is temperature.

### Hard Negative Mining

Performance improves significantly with **hard negatives**—documents that are similar to the query but not relevant:

```python
def mine_hard_negatives(query_embedding, document_embeddings, positive_idx, k=10):
    """Find hard negatives: similar but non-relevant documents."""
    similarities = query_embedding @ document_embeddings.T
    # Exclude the positive document
    similarities[positive_idx] = -float('inf')
    # Top-k most similar non-relevant documents
    hard_neg_indices = similarities.argsort(descending=True)[:k]
    return hard_neg_indices
```

## Implementation

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Load a pre-trained dense retriever
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Encode documents (do once, store in index)
documents = [
    "AAPL reported Q3 revenue of $81.8B, beating estimates by $1.2B",
    "The Federal Reserve held rates steady at 5.25-5.50%",
    "NVDA guidance implies 170% YoY data center revenue growth",
]
doc_embeddings = model.encode(documents, normalize_embeddings=True)

# Encode query and retrieve
query = "Which company beat revenue estimates?"
query_embedding = model.encode([query], normalize_embeddings=True)

# Cosine similarity (embeddings are normalized)
scores = query_embedding @ doc_embeddings.T
top_idx = scores[0].argsort()[::-1]
print(f"Most relevant: {documents[top_idx[0]]}")
```

## References

1. Karpukhin, V., et al. (2020). "Dense Passage Retrieval for Open-Domain QA." *EMNLP*.
2. Wang, L., et al. (2022). "Text Embeddings by Weakly-Supervised Contrastive Pre-training." *arXiv*.
