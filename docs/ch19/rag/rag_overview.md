# Retrieval-Augmented Generation Overview

## Learning Objectives

- Understand the RAG architecture and its motivation
- Formalize the retrieval-generation pipeline
- Identify when RAG is preferred over fine-tuning

## Architecture

RAG combines a **retriever** that finds relevant documents with a **generator** (LLM) that produces answers conditioned on retrieved context:

```
Query → Retriever → Top-k Documents → LLM(query + documents) → Response
```

Formally, given query $q$:

$$P(y \mid q) = \sum_{d \in \text{Top-k}} P(d \mid q) \cdot P(y \mid q, d)$$

where $P(d \mid q)$ is the retrieval score and $P(y \mid q, d)$ is the generation probability.

## Why RAG?

LLMs alone suffer from:

1. **Hallucination**: Generating plausible but incorrect facts
2. **Stale knowledge**: Training data cutoff means no awareness of recent events
3. **Domain gaps**: General-purpose training may lack specialized knowledge
4. **Unverifiability**: No way to trace claims to sources

RAG addresses all four by grounding generation in retrieved evidence.

## RAG vs. Fine-Tuning

| Dimension | RAG | Fine-Tuning |
|-----------|-----|-------------|
| Knowledge updates | Instant (update index) | Requires retraining |
| Hallucination control | High (grounded in docs) | Moderate |
| Cost | Retrieval overhead | Training cost |
| Domain adaptation | Add domain documents | Labeled data needed |
| Attributability | Citations possible | No attribution |

## Financial Applications

- **SEC filing Q&A**: Query specific sections of 10-K/10-Q filings
- **Research report synthesis**: Combine insights from multiple analyst reports
- **Regulatory compliance**: Retrieve relevant regulations for compliance queries
- **Market intelligence**: Real-time news retrieval for trading decisions

## References

1. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS*.
2. Gao, Y., et al. (2024). "Retrieval-Augmented Generation for Large Language Models: A Survey." *arXiv*.
