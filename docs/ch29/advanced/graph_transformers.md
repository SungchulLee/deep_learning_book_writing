# 29.4.4 Graph Transformers

## Introduction

**Graph Transformers** apply the transformer's self-attention mechanism to graphs, enabling global information exchange in a single layer and avoiding the local message passing bottleneck.

## Key Architectures

### Graphormer (Ying et al., 2021)
Adapts transformers to graphs with three structural encodings:
- **Centrality encoding**: Degree-based bias added to node features
- **Spatial encoding**: Shortest-path distance as attention bias
- **Edge encoding**: Average of edge features along shortest paths

### Graph Transformer (Dwivedi & Bresson, 2020)
Standard multi-head attention on graphs with:
- Laplacian positional encodings as node features
- Edge features incorporated into attention computation

### GPS (General, Powerful, Scalable)
Combines local message passing (MPNN) with global attention:
$$h_v = \text{MPNN}(h_v, \{h_u\}_{u \in \mathcal{N}(v)}) + \text{Transformer}(h_v, \{h_u\}_{u \in V})$$

## Attention Mechanism on Graphs

Standard: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$

With graph bias: $\text{Attention} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + B\right) V$

where $B_{ij}$ encodes structural relationships (distance, connectivity).

## Advantages over Message Passing

1. **Global receptive field** in a single layer
2. **No over-smoothing** (attention is selective)
3. **Captures long-range dependencies** directly
4. **Parallelizable** computation

## Limitations

- $O(n^2)$ complexity for full attention
- Requires positional/structural encodings to be structure-aware
- May overfit on small graphs

## Summary

Graph transformers represent the frontier of graph learning, combining the expressiveness of transformers with graph-structural inductive biases. They are particularly promising for tasks requiring long-range interactions.
