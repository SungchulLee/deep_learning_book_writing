# 29.4.3 Jumping Knowledge Networks

## Introduction

**Jumping Knowledge (JK) Networks** (Xu et al., 2018) address over-smoothing by adaptively combining representations from all layers rather than using only the final layer. This enables each node to select the optimal receptive field depth.

## Architecture

After $L$ message passing layers producing $\mathbf{h}_v^{(1)}, \ldots, \mathbf{h}_v^{(L)}$, the final representation is:

$$\mathbf{h}_v^{final} = \text{JK-AGG}\left(\mathbf{h}_v^{(1)}, \mathbf{h}_v^{(2)}, \ldots, \mathbf{h}_v^{(L)}\right)$$

## Aggregation Strategies

**Concatenation**: $\mathbf{h}_v = [\mathbf{h}_v^{(1)} \| \cdots \| \mathbf{h}_v^{(L)}]$. Most expressive but increases dimensionality by $L$.

**Max-pooling**: $\mathbf{h}_v = \max(\mathbf{h}_v^{(1)}, \ldots, \mathbf{h}_v^{(L)})$. Element-wise max across layers.

**LSTM Attention**: $\mathbf{h}_v = \sum_l \alpha_l^v \mathbf{h}_v^{(l)}$ where attention weights are computed by an LSTM over the layer representations. Node-specific layer selection.

## Why It Works

Different nodes benefit from different receptive field sizes. Nodes in dense regions may prefer early layers (local information), while nodes in sparse regions may need deeper layers for sufficient context. JK allows this adaptivity.

## Summary

JK networks are a simple, effective technique that should be considered whenever building multi-layer GNNs. They are orthogonal to the choice of GNN layer type and can be combined with any architecture.
