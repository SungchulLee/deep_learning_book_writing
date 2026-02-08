# 29.3.5 GraphSAGE

## Introduction

**GraphSAGE** (SAmple and aggreGatE) by Hamilton et al. (2017) introduces an **inductive** graph learning framework that generates node embeddings by sampling and aggregating features from a node's local neighborhood. Unlike GCN, GraphSAGE can generalize to unseen nodes and graphs.

## GraphSAGE Layer

$$\mathbf{h}_{\mathcal{N}(v)}^{(l)} = \text{AGG}^{(l)}\left(\{\mathbf{h}_u^{(l-1)} : u \in \mathcal{N}_s(v)\}\right)$$

$$\mathbf{h}_v^{(l)} = \sigma\left(W^{(l)} \cdot [\mathbf{h}_v^{(l-1)} \| \mathbf{h}_{\mathcal{N}(v)}^{(l)}\right])$$

$$\mathbf{h}_v^{(l)} = \frac{\mathbf{h}_v^{(l)}}{\|\mathbf{h}_v^{(l)}\|_2}$$

Key features:
- $\mathcal{N}_s(v)$: **sampled** subset of neighbors (fixed-size)
- Self-information explicitly separated via concatenation
- Optional L2 normalization of output embeddings

## Aggregation Variants

**Mean Aggregator**: $\text{AGG} = \text{mean}(\{\mathbf{h}_u\})$ — element-wise mean of neighbor features.

**Max-Pool Aggregator**: $\text{AGG} = \max(\{\sigma(W_{pool}\mathbf{h}_u + \mathbf{b})\})$ — applies MLP then element-wise max.

**LSTM Aggregator**: $\text{AGG} = \text{LSTM}(\pi(\{\mathbf{h}_u\}))$ — processes neighbors through LSTM with random permutation $\pi$.

## Neighbor Sampling

GraphSAGE samples a fixed number of neighbors per node per layer, creating a computation tree:
- Layer 1: sample $S_1$ neighbors for each node
- Layer 2: sample $S_2$ neighbors for each of those

Total computation: $O(S_1 \cdot S_2 \cdot \ldots)$ per target node. Typical values: $S_1 = 25$, $S_2 = 10$.

## Training

### Supervised Loss
Standard cross-entropy for node classification.

### Unsupervised Loss
$$J(v) = -\log(\sigma(\mathbf{z}_u^T \mathbf{z}_v)) - Q \cdot \mathbb{E}_{v_n \sim P_n} [\log(\sigma(-\mathbf{z}_{v_n}^T \mathbf{z}_v))]$$

Nearby nodes should have similar embeddings; random nodes should be dissimilar.

## Key Advantages over GCN

1. **Inductive**: Generalizes to new nodes/graphs without retraining
2. **Scalable**: Neighbor sampling avoids full-graph computation
3. **Mini-batch training**: Natural mini-batch support via sampling

## Summary

GraphSAGE is foundational for scalable, inductive graph learning. Its sampling strategy and explicit self/neighbor separation make it particularly suitable for large-scale and dynamic graphs common in financial applications.
