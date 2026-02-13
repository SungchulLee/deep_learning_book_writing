# Coreference Resolution

## Learning Objectives

- Understand mention detection and antecedent linking
- Implement end-to-end neural coreference resolution
- Appreciate the role of coreference in document-level IE

## Task Definition

Coreference resolution groups all **mentions** in a document that refer to the same real-world entity into **clusters**.

### Example

*"**Apple** announced **its** quarterly earnings. **The tech giant** reported revenue of $90B. **CEO Tim Cook** said **the company** exceeded expectations."*

Cluster 1: {Apple, its, The tech giant, the company}
Cluster 2: {CEO Tim Cook}

## Mention Types

| Type | Example | Detection |
|------|---------|-----------|
| Proper noun | "Goldman Sachs" | NER |
| Nominal | "the company", "the deal" | NP chunking |
| Pronominal | "it", "they", "his" | POS tagging |

## Classical Approaches

### Mention-Pair Model

Score each pair of mentions independently:

$$s(m_i, m_j) = \mathbf{w}^T \phi(m_i, m_j)$$

Features include string match, distance, gender/number agreement, and semantic similarity.

### Mention-Ranking Model

For each mention, rank all preceding mentions plus a "new entity" option:

$$P(a_j \mid m_i) = \frac{\exp(s(m_i, m_j))}{\sum_{k \leq i} \exp(s(m_i, m_k)) + \exp(s_{\text{new}}(m_i))}$$

## End-to-End Neural Coreference (Lee et al., 2017)

The dominant modern approach jointly performs mention detection and coreference linking.

### Architecture

1. **Span Enumeration**: Consider all spans up to length $L$
2. **Span Representation**: $\mathbf{g}_i = [\mathbf{h}_{\text{start}}; \mathbf{h}_{\text{end}}; \hat{\mathbf{h}}_i; \phi(i)]$ where $\hat{\mathbf{h}}_i$ is an attention-weighted head word representation and $\phi(i)$ encodes span width
3. **Mention Score**: $s_m(i) = \text{FFNN}_m(\mathbf{g}_i)$
4. **Antecedent Score**: $s_a(i, j) = \text{FFNN}_a([\mathbf{g}_i; \mathbf{g}_j; \mathbf{g}_i \circ \mathbf{g}_j; \phi(i,j)])$
5. **Pairwise Score**: $s(i, j) = s_m(i) + s_m(j) + s_a(i, j)$

### Training Objective

Marginalize over all correct antecedents:

$$\mathcal{L} = -\sum_{i=1}^{N} \log \frac{\sum_{j \in \mathcal{Y}(i)} \exp(s(i, j))}{\sum_{j' \in \mathcal{C}(i)} \exp(s(i, j'))}$$

where $\mathcal{Y}(i)$ is the set of correct antecedents and $\mathcal{C}(i)$ includes all candidates plus a dummy "new entity" antecedent.

```python
import torch
import torch.nn as nn

class CorefScorer(nn.Module):
    """Simplified coreference scoring module."""
    def __init__(self, hidden_dim=768, ffnn_dim=1000):
        super().__init__()
        span_dim = hidden_dim * 3 + 20  # start, end, head-attn, width features
        self.mention_score = nn.Sequential(
            nn.Linear(span_dim, ffnn_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(ffnn_dim, 1),
        )
        pair_dim = span_dim * 3 + 20  # g_i, g_j, g_i*g_j, distance features
        self.antecedent_score = nn.Sequential(
            nn.Linear(pair_dim, ffnn_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(ffnn_dim, 1),
        )

    def forward(self, span_i, span_j, pair_features):
        s_m_i = self.mention_score(span_i)
        s_m_j = self.mention_score(span_j)
        pair_repr = torch.cat([span_i, span_j, span_i * span_j, pair_features], -1)
        s_a = self.antecedent_score(pair_repr)
        return s_m_i + s_m_j + s_a
```

## Evaluation Metrics

| Metric | Focus |
|--------|-------|
| MUC | Pairwise link F1 |
| B-cubed | Entity-level precision/recall |
| CEAF | Optimal cluster alignment |
| CoNLL | Average of MUC, B-cubed, CEAF |

## Financial Applications

Coreference is essential for document-level financial IE: tracking entity mentions across 10-K filings (100+ pages), linking pronoun references in earnings call transcripts to speakers, and resolving company aliases (e.g., "Alphabet" = "Google" = "the search giant").

## References

1. Lee, K., et al. (2017). End-to-End Neural Coreference Resolution. *EMNLP*.
2. Joshi, M., et al. (2020). SpanBERT: Improving Pre-Training by Representing and Predicting Spans. *TACL*.
3. Wu, W., et al. (2020). CorefQA: Coreference Resolution as Query-Based Span Prediction. *ACL*.
