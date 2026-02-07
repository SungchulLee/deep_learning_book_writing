# A8 Graph Neural Networks

## Overview

This appendix provides complete PyTorch implementations of graph neural network (GNN) architectures that extend deep learning to non-Euclidean, graph-structured data. GNNs learn node, edge, and graph-level representations by propagating and aggregating information along graph edges. Financial systems are inherently graph-structured — interbank networks, supply chains, portfolio holdings, corporate ownership — making GNNs a natural fit for quantitative finance applications.

## Architectures

| Model | Year | Key Innovation |
|-------|------|----------------|
| [GCN](gcn.py) | 2017 | Spectral graph convolution via normalized adjacency |
| [GAT](gat.py) | 2018 | Learnable attention weights over neighbors |
| [GraphSAGE](graphsage.py) | 2017 | Inductive learning via neighborhood sampling and aggregation |
| [GIN](gin.py) | 2019 | Maximally expressive aggregation matching WL isomorphism test |
| [Message Passing](mpnn.py) | 2017 | Unified message passing neural network framework |

## Key Concepts

### Graph Representation

A graph $G = (V, E)$ consists of nodes $V$ with features $\mathbf{h}_v \in \mathbb{R}^d$ and edges $E$ with optional features. GNNs update node representations by aggregating neighbor information:

$$\mathbf{h}_v^{(k)} = \text{UPDATE}\!\left(\mathbf{h}_v^{(k-1)},\; \text{AGGREGATE}\!\left(\{\mathbf{h}_u^{(k-1)} : u \in \mathcal{N}(v)\}\right)\right)$$

### Aggregation Strategies

| Model | Aggregation | Properties |
|-------|-------------|------------|
| GCN | Normalized mean | Symmetric, spectral grounding |
| GAT | Attention-weighted sum | Adaptive, learns edge importance |
| GraphSAGE | Mean, LSTM, or max pooling | Inductive, scalable to unseen nodes |
| GIN | Sum with MLP | Injective, maximally expressive |
| MPNN | Learned message functions | General framework, customizable |

### Expressiveness and the WL Test

GNNs are bounded by the Weisfeiler–Leman (WL) graph isomorphism test in their ability to distinguish graph structures. GIN achieves the WL upper bound by using injective (sum-based) aggregation, while mean and max aggregation (GCN, GraphSAGE) are strictly less expressive.

### Scalability

- **Mini-batch training**: GraphSAGE samples fixed-size neighborhoods for scalable training
- **Full-batch**: GCN operates on the full adjacency matrix (memory-limited for large graphs)
- **Attention sparsity**: GAT computes attention only over existing edges

## Quantitative Finance Applications

- **Interbank network analysis**: Model systemic risk propagation through lending and counterparty networks
- **Corporate relationship graphs**: Detect contagion effects via supply chain, ownership, and board interlock networks
- **Portfolio optimization**: Represent asset correlation structures as graphs for sector-aware allocation
- **Fraud detection**: Identify suspicious transaction patterns in payment and trading networks
- **Knowledge graphs**: Integrate heterogeneous financial data (companies, filings, executives, events) into unified representations
- **Credit risk**: Propagate default risk through guarantee and lending networks

## Prerequisites

- [Ch3: Neural Network Fundamentals](../../ch03/index.md) — MLP building blocks, `nn.Module`
- [A10: Utility Modules — Attention Mechanisms](../utils/attention.py) — attention computation for GAT
- [A10: Utility Modules — Normalization Layers](../utils/normalization.py) — graph normalization techniques
