# Heterogeneous Graphs

Most real-world networks involve more than one kind of entity.  Academic
graphs connect authors, papers, and venues through different relationships;
financial networks link companies, banks, investors, and instruments.
Standard GNNs treat all nodes and edges identically, losing this structural
richness.  Heterogeneous graph neural networks restore it by assigning
type-specific transformations to each node and edge type.

## Formal Definition

A **heterogeneous graph** $G = (V, E, \tau_V, \tau_E)$ extends the standard
graph definition with two type mappings:

- $\tau_V : V \to \mathcal{T}_V$ assigns each node a type from a set of
  node types $\mathcal{T}_V$.
- $\tau_E : E \to \mathcal{T}_E$ assigns each edge a type from a set of
  edge types $\mathcal{T}_E$.

When $|\mathcal{T}_V| = 1$ and $|\mathcal{T}_E| = 1$, the graph is
homogeneous and standard GNN methods apply directly.

!!! example "Academic Graph"
    Node types: $\mathcal{T}_V = \{\text{Author}, \text{Paper}, \text{Venue}\}$.
    Edge types: $\mathcal{T}_E = \{\text{writes}, \text{cites}, \text{published\_in}\}$.
    A meta-path Author $\to$ Paper $\to$ Author captures co-authorship.

## RGCN (Relational Graph Convolutional Network)

RGCN assigns a separate weight matrix to each edge (relation) type.  The
update rule for node $v$ at layer $l$ is:

$$
\mathbf{h}_v^{(l)} = \sigma\!\left(W_0^{(l)} \mathbf{h}_v^{(l-1)} + \sum_{r \in \mathcal{R}} \sum_{u \in \mathcal{N}_r(v)} \frac{1}{|\mathcal{N}_r(v)|} W_r^{(l)} \mathbf{h}_u^{(l-1)}\right)
$$

where $\mathcal{R}$ is the set of relation types, $\mathcal{N}_r(v)$ is
the set of neighbors of $v$ under relation $r$, and $W_0^{(l)}$ is a
self-loop weight matrix.

The number of parameters grows linearly with $|\mathcal{R}|$, which can be
problematic for knowledge graphs with hundreds of relation types.  Two
common regularization strategies address this:

- **Basis decomposition:**  $W_r^{(l)} = \sum_{b=1}^{B} a_{rb}^{(l)} V_b^{(l)}$,
  sharing $B$ basis matrices across all relations.
- **Block-diagonal decomposition:**  $W_r^{(l)}$ is block-diagonal,
  reducing parameters while maintaining relation-specific capacity.

## HAN (Heterogeneous Attention Network)

HAN introduces a two-level attention mechanism:

### Node-Level Attention

For each meta-path $\Phi$, compute attention between node $v$ and its
meta-path-based neighbors $\mathcal{N}_v^{\Phi}$:

$$
\alpha_{vu}^{\Phi} = \frac{\exp\!\bigl(\text{LeakyReLU}(\mathbf{a}_{\Phi}^{\top} [\mathbf{h}_v' \| \mathbf{h}_u'])\bigr)}{\sum_{k \in \mathcal{N}_v^{\Phi}} \exp\!\bigl(\text{LeakyReLU}(\mathbf{a}_{\Phi}^{\top} [\mathbf{h}_v' \| \mathbf{h}_k'])\bigr)}
$$

where $\mathbf{h}_v' = W_{\Phi} \mathbf{h}_v$ is the type-projected feature.

### Meta-Path-Level Attention

Aggregate across meta-paths with learned importance weights:

$$
\mathbf{z}_v = \sum_{\Phi \in \mathcal{P}} \beta_{\Phi} \cdot \mathbf{z}_v^{\Phi}
$$

where $\beta_{\Phi}$ is computed via an attention mechanism over the set of
meta-paths $\mathcal{P}$, and $\mathbf{z}_v^{\Phi}$ is the node-level
aggregation for meta-path $\Phi$.

## HGT (Heterogeneous Graph Transformer)

HGT applies the Transformer attention mechanism with type-specific
projections.  For a source node $s$ and target node $t$:

$$
\text{Attention}(s, t) = \text{softmax}\!\left(\frac{(W_{\tau(s)}^Q \mathbf{h}_s)(W_{\tau(t)}^K \mathbf{h}_t)^{\top}}{\sqrt{d}}\right)
$$

The key insight is that the query, key, and value projections depend on the
**node types** of $s$ and $t$, while an additional relation-type weight
$W_{\tau_E(e)}^{\text{ATT}}$ can be inserted between query and key to
capture edge-type-specific interactions.

## Meta-Paths

A **meta-path** $\Phi$ is a sequence of node types connected by edge types:

$$
\Phi : T_1 \xrightarrow{r_1} T_2 \xrightarrow{r_2} \cdots \xrightarrow{r_l} T_{l+1}
$$

Meta-paths capture high-order semantic relationships.  Common examples in
academic graphs:

| Meta-path | Semantics |
|---|---|
| Author $\to$ Paper $\to$ Author | Co-authorship |
| Author $\to$ Paper $\to$ Venue $\to$ Paper $\to$ Author | Co-venue authorship |
| Paper $\to$ Paper (via citation) | Citation relationship |

!!! tip "Choosing Meta-Paths"
    Meta-path selection strongly influences model performance.  Domain
    knowledge guides which paths are semantically meaningful.  Automated
    meta-path discovery is an active research area.

## Comparison

| Method | Type handling | Attention | Scalability |
|---|---|---|---|
| RGCN | Per-relation weights | None | Moderate (basis decomposition helps) |
| HAN | Meta-path grouping | Two-level | Requires meta-path enumeration |
| HGT | Per-type projections | Transformer-style | Scales well with mini-batching |

## Reference

- Schlichtkrull, M. et al. "Modeling Relational Data with Graph Convolutional
  Networks." ESWC 2018.
- Wang, X. et al. "Heterogeneous Graph Attention Network." WWW 2019.
- Hu, Z. et al. "Heterogeneous Graph Transformer." WWW 2020.
