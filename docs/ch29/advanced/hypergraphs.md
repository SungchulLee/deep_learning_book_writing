# Hypergraph Neural Networks

Standard graphs represent only pairwise relationships: an edge connects exactly two nodes. Many real-world interactions, however, are inherently **multi-way** -- a research paper connects all its co-authors simultaneously, a financial transaction may involve multiple parties, and a drug combination acts on multiple targets together. A **hypergraph** $H = (V, \mathcal{E})$ generalizes graphs by allowing **hyperedges** that connect any number of nodes, capturing these higher-order relationships directly.

## Hypergraph Representation

The algebraic representation of a hypergraph is its **incidence matrix** $\mathbf{B} \in \{0,1\}^{|V| \times |\mathcal{E}|}$, where $B_{ve} = 1$ if node $v$ participates in hyperedge $e$.

Two derived quantities are central to hypergraph neural network architectures:

- **Node degree**: $d(v) = \sum_{e \in \mathcal{E}} B_{ve}$ (number of hyperedges containing $v$)
- **Hyperedge degree**: $\delta(e) = \sum_{v \in V} B_{ve}$ (number of nodes in $e$)

## Hypergraph Neural Networks

### HyperGCN

The simplest approach extends graph convolution to hypergraphs using the incidence matrix and degree-based normalization:

$$
\mathbf{h}_v^{(\ell+1)} = \sigma\!\left(\sum_{e \ni v} \frac{1}{\delta(e)} \sum_{u \in e} \frac{1}{\sqrt{d(v)\,d(u)}}\, \mathbf{W}^{(\ell)}\,\mathbf{h}_u^{(\ell)}\right)
$$

This aggregates information from all nodes that share a hyperedge with $v$, weighted by node and hyperedge degrees.

### Two-Stage Message Passing

A more flexible architecture decomposes hypergraph convolution into two stages:

1. **Node to hyperedge**: Aggregate node features for each hyperedge $e$, producing a hyperedge representation $\mathbf{m}_e = \text{AGG}(\{\mathbf{h}_u : u \in e\})$
2. **Hyperedge to node**: Aggregate hyperedge representations for each node $v$, producing an updated node representation $\mathbf{h}_v' = \text{AGG}(\{\mathbf{m}_e : v \in e\})$

### AllSet and AllDeepSets

Recent architectures replace the simple aggregation functions with learnable set functions (Deep Sets) or transformers, enabling the model to learn task-specific aggregation strategies at each stage. This increases expressiveness while maintaining the two-stage message-passing structure.

## Financial Applications

Hypergraphs arise naturally in finance:

- **Portfolio groups**: A fund holds multiple stocks; each fund defines a hyperedge over its constituent assets
- **Joint defaults**: Multiple firms defaulting in the same crisis event form a hyperedge linking those firms
- **Regulatory clusters**: Assets governed by the same regulation share a hyperedge

!!! tip "When Hypergraphs Add Value"
    Hypergraph models are most beneficial when the higher-order structure carries information beyond what pairwise edges capture. If the clique expansion (replacing each hyperedge with all pairwise edges) loses no information, a standard GNN may suffice. Test both approaches.

## References

[Feng et al. -- Hypergraph Neural Networks (AAAI 2019)](https://arxiv.org/abs/1809.09401)
