# 29.4.7 Hypergraphs

## Introduction

A **hypergraph** $H = (V, \mathcal{E})$ generalizes graphs by allowing **hyperedges** that connect any number of nodes simultaneously. This captures higher-order relationships beyond pairwise interactions.

## Motivation

Standard graphs represent only pairwise relations. Many real-world relationships are naturally multi-way:
- A research paper connects all its co-authors simultaneously
- A financial transaction may involve multiple parties
- A drug combination acts on multiple targets together

## Hypergraph Representation

The **incidence matrix** $\mathbf{B} \in \{0,1\}^{|V| \times |\mathcal{E}|}$: $B_{ve} = 1$ if node $v$ is in hyperedge $e$.

Node degree: $d(v) = \sum_e B_{ve}$. Hyperedge degree: $\delta(e) = \sum_v B_{ve}$.

## Hypergraph Neural Networks

### HyperGCN
Approximate hypergraph convolution using clique expansion or Laplacian:
$$\mathbf{h}_v = \sigma\left(\sum_{e \ni v} \frac{1}{\delta(e)} \sum_{u \in e} \frac{1}{\sqrt{d(v)d(u)}} W\mathbf{h}_u\right)$$

### Two-stage Message Passing
1. **Node → Hyperedge**: Aggregate node features for each hyperedge
2. **Hyperedge → Node**: Aggregate hyperedge features for each node

### AllSet / AllDeepSets
Use Deep Sets or transformers for each aggregation stage, enabling learnable set functions.

## Financial Applications
- **Portfolio groups**: A fund holds multiple stocks (hyperedge = fund)
- **Joint defaults**: Multiple firms defaulting in the same crisis event
- **Regulatory clusters**: Assets governed by the same regulation

## Summary

Hypergraph neural networks extend GNNs to model higher-order interactions. The two-stage message passing (node↔hyperedge) is the standard approach, with increasing sophistication in the aggregation functions.
