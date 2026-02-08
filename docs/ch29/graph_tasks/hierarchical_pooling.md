# 29.5.4 Hierarchical Pooling

## Overview
Hierarchical pooling progressively coarsens graphs, creating multi-resolution representations. Key methods:

**DiffPool**: Learns soft cluster assignments $S \in \mathbb{R}^{n \times k}$ via auxiliary GNN. Coarsened graph: $X' = S^T Z$, $A' = S^T A S$.

**TopKPool**: Selects top-k nodes by learned importance score. Retains subgraph induced by selected nodes.

**SAGPool**: Self-attention-based node selection using GNN scores for importance ranking.

## Trade-offs
DiffPool is most expressive but $O(n^2)$ due to dense assignment. TopK and SAG are more efficient but may lose information from dropped nodes.
