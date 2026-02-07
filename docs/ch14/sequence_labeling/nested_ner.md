# Nested NER

## Overview

Standard NER assumes flat, non-overlapping entities. Nested NER handles cases where entities contain other entities:

```
[Bank of [America]_LOC]_ORG
[New York]_LOC → contained within [New York University]_ORG
```

## Approaches

### Span-Based Methods

Enumerate and classify all possible spans:

$$P(y | s_{i,j}) = \text{softmax}(\mathbf{W} \cdot \text{repr}(s_{i,j}) + \mathbf{b})$$

where $s_{i,j}$ is the span from position $i$ to $j$, and $\text{repr}(s_{i,j})$ combines boundary and content representations.

```python
import torch
import torch.nn as nn

class SpanNER(nn.Module):
    def __init__(self, hidden_dim, num_types, max_span_length=10):
        super().__init__()
        self.max_span_length = max_span_length
        self.span_classifier = nn.Linear(hidden_dim * 2, num_types + 1)  # +1 for non-entity
        self.width_embedding = nn.Embedding(max_span_length, hidden_dim)

    def forward(self, hidden_states):
        batch, seq_len, hidden = hidden_states.shape
        spans = []
        for length in range(1, min(self.max_span_length + 1, seq_len + 1)):
            for start in range(seq_len - length + 1):
                end = start + length
                span_repr = torch.cat([
                    hidden_states[:, start, :],
                    hidden_states[:, end - 1, :]
                ], dim=-1)
                spans.append((start, end, span_repr))
        return spans
```

### Layered/Stacked Models

Process entities from innermost to outermost, using detected inner entities as features for outer detection.

### Sequence-to-Set

Predict all entities as a set using pointer networks or sequence-to-sequence generation.

## Datasets

| Dataset | Nested Rate | Entity Types |
|---------|-------------|-------------|
| ACE 2004 | ~30% | 7 |
| ACE 2005 | ~25% | 7 |
| GENIA | ~17% | 5 (biomedical) |

## Summary

1. Flat NER misses 10-30% of entities in domains with nesting
2. Span enumeration is the most common approach for nested NER
3. Computational cost grows as $O(n^2)$ with span-based methods
4. Biaffine span classifiers provide a good balance of speed and accuracy
