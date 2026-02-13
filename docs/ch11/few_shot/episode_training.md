# Episode-Based Training

## Overview

Episode-based training simulates the few-shot evaluation setting during training. Each episode consists of a support set ($N$ classes $\times$ $K$ examples) and query set ($N$ classes $\times$ $Q$ examples). The model classifies queries using only the support set.

## Episode Construction

```python
class EpisodeSampler:
    def __init__(self, dataset, n_way, k_shot, q_query, n_episodes):
        self.class_indices = {}
        for idx, (_, label) in enumerate(dataset):
            self.class_indices.setdefault(label, []).append(idx)

    def __iter__(self):
        for _ in range(self.n_episodes):
            classes = random.sample(list(self.class_indices.keys()), self.n_way)
            support, query = [], []
            for cls in classes:
                indices = random.sample(self.class_indices[cls],
                                       self.k_shot + self.q_query)
                support.extend(indices[:self.k_shot])
                query.extend(indices[self.k_shot:])
            yield support, query
```

## Design Choices

Alignment between training and testing $N$-way $K$-shot settings improves results. Typically 10K-100K training episodes are used; more generally helps up to a saturation point.

## Advantage Over Standard Training

Episode-based training directly optimizes the metric used for evaluation (few-shot classification accuracy) rather than a proxy objective like standard cross-entropy on all classes.
