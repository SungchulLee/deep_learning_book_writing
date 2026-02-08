# 29.3.6 Graph Attention Network (GAT)

## Introduction

**Graph Attention Networks (GAT)** by Veličković et al. (2018) introduce **attention mechanisms** to GNNs, allowing nodes to learn which neighbors are most important. Unlike GCN's fixed normalization weights, GAT computes adaptive, data-dependent attention coefficients.

## GAT Layer

### Attention Coefficients

For a pair of connected nodes $i$ and $j$:

$$e_{ij} = \text{LeakyReLU}\left(\mathbf{a}^T [W\mathbf{h}_i \| W\mathbf{h}_j]\right)$$

$$\alpha_{ij} = \text{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i) \cup \{i\}} \exp(e_{ik})}$$

### Node Update

$$\mathbf{h}_i' = \sigma\left(\sum_{j \in \mathcal{N}(i) \cup \{i\}} \alpha_{ij} W \mathbf{h}_j\right)$$

### Multi-Head Attention

To stabilize learning, $K$ independent attention heads are computed and concatenated (or averaged in the final layer):

$$\mathbf{h}_i' = \big\|_{k=1}^{K} \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^k W^k \mathbf{h}_j\right)$$

## GATv2

Brody et al. (2022) identified that GAT's attention is **static** (ranking of attended neighbors is query-independent). **GATv2** fixes this:

$$e_{ij} = \mathbf{a}^T \text{LeakyReLU}\left(W [h_i \| h_j]\right)$$

Moving LeakyReLU inside the attention computation enables **dynamic attention** where the ranking depends on the query node.

## Key Properties

- **Adaptive weights**: Learns importance of each neighbor
- **Inductive**: Attention mechanism generalizes to new graphs
- **Interpretable**: Attention coefficients reveal learned graph structure
- **Multi-head**: Stabilizes training and captures diverse patterns
- **Masked attention**: Only computes attention over existing edges

## Comparison with GCN

| Aspect | GCN | GAT |
|--------|-----|-----|
| Weights | Fixed ($1/\sqrt{d_i d_j}$) | Learned (attention) |
| Expressiveness | Limited | Higher |
| Interpretability | Low | High (attention coefficients) |
| Computation | $O(\|E\| d)$ | $O(\|E\| d + \|E\| K)$ |

## Summary

GAT introduces learnable, adaptive aggregation weights via attention, offering improved expressiveness and interpretability over fixed-weight GNNs. Multi-head attention and GATv2 further enhance its capabilities.
