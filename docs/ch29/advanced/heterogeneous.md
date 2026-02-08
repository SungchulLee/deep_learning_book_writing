# 29.4.5 Heterogeneous Graphs

## Introduction

**Heterogeneous graphs** contain multiple types of nodes and/or edges, reflecting real-world complexity. A heterogeneous graph $G = (V, E, \tau_V, \tau_E)$ has node type mapping $\tau_V: V \to \mathcal{T}_V$ and edge type mapping $\tau_E: E \to \mathcal{T}_E$.

## Examples

- **Academic graphs**: Author → writes → Paper → published_in → Venue
- **Financial networks**: Company → supplies → Company; Bank → lends_to → Bank; Investor → holds → Stock
- **Knowledge graphs**: Entity → relation → Entity

## Heterogeneous GNN Approaches

### RGCN (Relational GCN)
Separate weight matrices per relation type:
$$\mathbf{h}_v^{(l)} = \sigma\left(\sum_{r \in \mathcal{R}} \sum_{u \in \mathcal{N}_r(v)} \frac{1}{|\mathcal{N}_r(v)|} W_r^{(l)} \mathbf{h}_u^{(l-1)}\right)$$

### HAN (Heterogeneous Attention Network)
Uses hierarchical attention: node-level attention within each meta-path, then meta-path-level attention.

### HGT (Heterogeneous Graph Transformer)
Type-specific attention with:
$$\text{Attention}(s, t) = \text{softmax}\left(\frac{(W_{\tau(s)}^Q h_s)(W_{\tau(t)}^K h_t)^T}{\sqrt{d}}\right)$$

## Meta-Paths
A **meta-path** defines a composite relation: Author → Paper → Author (co-authorship).

## Financial Applications
- Company-Bank-Company lending chains
- Investor-Stock-Market multi-type networks
- Supply chain with different relationship types

## Summary

Heterogeneous GNNs handle the multi-typed nature of real-world networks, providing richer representations than homogeneous models. Type-specific transformations and meta-path-based attention are key techniques.
