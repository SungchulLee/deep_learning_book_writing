# 29.2.1 Message Passing Framework

## Introduction

The **Message Passing Neural Network (MPNN)** framework, introduced by Gilmer et al. (2017), provides a unified view of most graph neural network architectures. In each layer, nodes exchange information with their neighbors through a three-step process: message computation, aggregation, and update. This paradigm is the foundation of modern GNNs.

## The Message Passing Paradigm

At each layer $l$, every node $v$ in the graph:

1. **Computes messages** from each neighbor $u \in \mathcal{N}(v)$
2. **Aggregates** all incoming messages
3. **Updates** its own representation

Formally, for layer $l$:

$$\mathbf{m}_v^{(l)} = \bigoplus_{u \in \mathcal{N}(v)} \text{MSG}^{(l)}\left(\mathbf{h}_u^{(l-1)}, \mathbf{h}_v^{(l-1)}, \mathbf{e}_{uv}\right)$$

$$\mathbf{h}_v^{(l)} = \text{UPD}^{(l)}\left(\mathbf{h}_v^{(l-1)}, \mathbf{m}_v^{(l)}\right)$$

where:
- $\mathbf{h}_v^{(l)}$ is the hidden representation of node $v$ at layer $l$
- $\text{MSG}^{(l)}$ is the message function
- $\bigoplus$ is a permutation-invariant aggregation function
- $\text{UPD}^{(l)}$ is the update function
- $\mathbf{e}_{uv}$ is the edge feature between $u$ and $v$

## Receptive Field and Layer Depth

After $K$ message passing layers, each node's representation captures information from its **$K$-hop neighborhood**. The receptive field grows exponentially with depth:

$$|\text{Receptive field}| \leq \sum_{k=0}^{K} d_{max}^k$$

This has important implications:
- **Too few layers**: Limited receptive field, cannot capture long-range dependencies
- **Too many layers**: Over-smoothing (all nodes converge to similar representations)
- **Optimal depth**: Depends on the graph diameter and task requirements

## Connection to Weisfeiler-Leman Test

The message passing framework is closely related to the **1-dimensional Weisfeiler-Leman (1-WL) graph isomorphism test**. Both iteratively refine node labels using neighborhood information. Standard message passing GNNs are at most as powerful as the 1-WL test in distinguishing non-isomorphic graphs.

## Design Space

The message passing framework defines a design space along three axes:

### 1. Message Function
How to compute messages from neighboring nodes:
- **Linear**: $\text{MSG}(\mathbf{h}_u) = W\mathbf{h}_u$
- **MLP**: $\text{MSG}(\mathbf{h}_u, \mathbf{h}_v) = \text{MLP}([\mathbf{h}_u \| \mathbf{h}_v])$
- **Attention-weighted**: $\text{MSG}(\mathbf{h}_u, \mathbf{h}_v) = \alpha_{uv} W\mathbf{h}_u$

### 2. Aggregation Function
How to combine messages (must be permutation-invariant):
- Sum, Mean, Max
- Attention-weighted sum
- Multi-head aggregation

### 3. Update Function
How to combine aggregated messages with the node's own representation:
- **Replace**: $\mathbf{h}_v^{(l)} = \mathbf{m}_v^{(l)}$
- **Residual**: $\mathbf{h}_v^{(l)} = \mathbf{h}_v^{(l-1)} + \mathbf{m}_v^{(l)}$
- **GRU/LSTM**: $\mathbf{h}_v^{(l)} = \text{GRU}(\mathbf{h}_v^{(l-1)}, \mathbf{m}_v^{(l)})$
- **MLP**: $\mathbf{h}_v^{(l)} = \text{MLP}([\mathbf{h}_v^{(l-1)} \| \mathbf{m}_v^{(l)}])$

## Existing GNNs as Special Cases

| GNN | Message | Aggregation | Update |
|-----|---------|-------------|--------|
| GCN | $\frac{1}{\sqrt{d_u d_v}} W\mathbf{h}_u$ | Sum | Self-loop included |
| GraphSAGE | $W\mathbf{h}_u$ | Mean/Max/LSTM | $W[\mathbf{h}_v \| \mathbf{m}_v]$ |
| GAT | $\alpha_{uv} W\mathbf{h}_u$ | Sum | Self-attention |
| GIN | $W\mathbf{h}_u$ | Sum | $\text{MLP}((1+\epsilon)\mathbf{h}_v + \mathbf{m}_v)$ |

## Quantitative Finance Applications

Message passing naturally models information flow in financial networks:
- **Contagion modeling**: How financial distress propagates through interbank networks
- **Signal propagation**: How market signals spread through correlated asset networks
- **Credit risk**: Aggregating counterparty risk through lending networks
- **Supply chain**: Propagating demand/supply shocks through supplier graphs

## Summary

The message passing framework provides a principled and flexible paradigm for learning on graphs. Understanding this framework is essential for selecting, implementing, and designing GNN architectures for specific tasks.
