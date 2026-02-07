# Reranking

## Learning Objectives

- Understand the two-stage retrieval pipeline
- Implement cross-encoder reranking
- Compare reranking approaches

## Why Reranking?

Bi-encoder retrieval (dense retrieval) is fast but approximate—it independently encodes query and document without cross-attention. **Cross-encoder reranking** provides more accurate relevance scores by jointly processing query and document:

```
Stage 1 (fast): Bi-encoder retrieves top-100 candidates
Stage 2 (accurate): Cross-encoder reranks to top-10
```

## Cross-Encoder Architecture

Unlike bi-encoders, cross-encoders process the query-document pair together:

$$\text{score}(q, d) = \sigma(W \cdot \text{CLS}([q; \text{SEP}; d]) + b)$$

The joint processing allows full cross-attention between query and document tokens, yielding much better relevance judgments.

## Implementation

```python
from sentence_transformers import CrossEncoder

# Load cross-encoder reranker
reranker = CrossEncoder("BAAI/bge-reranker-large")

def rerank(query: str, documents: list, top_k: int = 5):
    """Rerank documents using cross-encoder."""
    # Create query-document pairs
    pairs = [(query, doc["text"]) for doc in documents]

    # Score all pairs
    scores = reranker.predict(pairs)

    # Sort by score
    ranked = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, score in ranked[:top_k]]
```

## Models Comparison

| Model | Parameters | NDCG@10 (BEIR) | Latency |
|-------|-----------|----------------|---------|
| bge-reranker-base | 110M | 52.1 | Fast |
| bge-reranker-large | 335M | 54.3 | Moderate |
| Cohere rerank-v3 | Unknown | 56.8 | API |
| GPT-4 (listwise) | ~1T | 57.2 | Slow, expensive |

## Two-Stage Pipeline

```python
class TwoStageRetriever:
    def __init__(self, retriever, reranker, initial_k=100, final_k=5):
        self.retriever = retriever
        self.reranker = reranker
        self.initial_k = initial_k
        self.final_k = final_k

    def search(self, query):
        # Stage 1: Fast retrieval
        candidates = self.retriever.search(query, k=self.initial_k)
        # Stage 2: Accurate reranking
        reranked = rerank(query, candidates, top_k=self.final_k)
        return reranked
```

## References

1. Nogueira, R. & Cho, K. (2020). "Passage Re-ranking with BERT." *arXiv*.
2. Sun, W., et al. (2023). "Is ChatGPT Good at Search? LLMs as Re-Ranking Agents." *arXiv*.
