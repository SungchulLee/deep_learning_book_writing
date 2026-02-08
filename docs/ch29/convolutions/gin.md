# 29.3.7 Graph Isomorphism Network (GIN)

## Introduction

The **Graph Isomorphism Network (GIN)** by Xu et al. (2019) is designed to be **maximally expressive** among message passing GNNs. It is provably as powerful as the 1-Weisfeiler-Leman (1-WL) graph isomorphism test, making it the most theoretically grounded architecture for graph-level representation learning.

## Theoretical Foundation

### WL Test Connection
The 1-WL test iteratively updates node labels:
$$c^{(l)}(v) = \text{HASH}\left(c^{(l-1)}(v), \{\!\{c^{(l-1)}(u) : u \in \mathcal{N}(v)\}\!\}\right)$$

For a GNN to be as powerful as 1-WL, it must be able to distinguish different multisets of neighbor features. Xu et al. proved that **sum aggregation with injective update** achieves this.

### Why Sum Aggregation?
- **Sum** can distinguish multisets (counts multiplicities)
- **Mean** cannot distinguish $\{1,1,1\}$ from $\{1\}$
- **Max** cannot distinguish $\{1,1,1\}$ from $\{1,2,3\}$ element-wise

## GIN Layer

$$\mathbf{h}_v^{(l)} = \text{MLP}^{(l)}\left((1 + \epsilon^{(l)}) \cdot \mathbf{h}_v^{(l-1)} + \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(l-1)}\right)$$

where:
- $\epsilon$ is a learnable scalar (or fixed at 0)
- MLP provides the injective mapping
- Sum aggregation preserves multiset information

## GIN-0 vs GIN-ε

- **GIN-ε**: Learnable $\epsilon$, empirically slightly better
- **GIN-0**: Fixed $\epsilon = 0$, simpler: $\text{MLP}(\mathbf{h}_v + \sum \mathbf{h}_u)$

## Graph-Level Readout

For graph classification, GIN uses **concatenation of all layers' readouts**:

$$\mathbf{h}_G = \text{CONCAT}\left(\text{READOUT}(\{\mathbf{h}_v^{(l)}\}_{v \in V}) \mid l = 0, 1, \ldots, L\right)$$

This Jumping Knowledge-style readout prevents information loss from over-smoothing.

## Expressiveness Results

| Architecture | Aggregation | Update | WL Equivalent? |
|-------------|-------------|--------|----------------|
| GCN | Normalized sum | Linear | No |
| GraphSAGE (mean) | Mean | Concat+Linear | No |
| GAT | Attention-weighted sum | Attention | No |
| **GIN** | **Sum** | **MLP** | **Yes (1-WL)** |

## Summary

GIN is the go-to architecture when maximal expressiveness within the message passing framework is required. Its theoretical guarantees make it particularly valuable for graph classification benchmarks and applications where distinguishing graph structures is critical.
