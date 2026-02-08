# 29.1.5 Graph Properties

## Introduction

Understanding graph-level properties is essential for selecting appropriate GNN architectures, preprocessing data, and interpreting model behavior. This section covers key topological, structural, and statistical properties of graphs.

## Connectivity

### Connected Components
A **connected component** is a maximal subgraph in which any two nodes are connected by a path. The number of connected components $k$ equals the multiplicity of eigenvalue 0 of the graph Laplacian.

### Strongly Connected Components (Directed)
In a directed graph, a **strongly connected component** (SCC) is a maximal subgraph where every node is reachable from every other node.

### Algebraic Connectivity
The **algebraic connectivity** $\lambda_2$ (Fiedler value) is the second-smallest eigenvalue of the Laplacian. It measures how well connected the graph is:
- $\lambda_2 = 0$: graph is disconnected
- Large $\lambda_2$: graph is well-connected and hard to partition

## Distance Measures

### Shortest Path and Diameter
- **Shortest path** $d(u, v)$: minimum number of edges between $u$ and $v$
- **Eccentricity** $\epsilon(v) = \max_{u \in V} d(u, v)$
- **Diameter** $\text{diam}(G) = \max_{v \in V} \epsilon(v)$
- **Radius** $\text{rad}(G) = \min_{v \in V} \epsilon(v)$
- **Average path length** $\bar{d} = \frac{1}{n(n-1)} \sum_{u \neq v} d(u, v)$

### Effective Resistance
The **effective resistance** between nodes $u$ and $v$ is:
$$R_{uv} = (e_u - e_v)^T L^+ (e_u - e_v)$$
where $L^+$ is the pseudoinverse of the Laplacian. It provides a distance measure that accounts for all paths between nodes.

## Clustering and Community Structure

### Clustering Coefficient
The **local clustering coefficient** of node $v$:
$$C(v) = \frac{2|\{(u, w) : u, w \in \mathcal{N}(v), (u, w) \in E\}|}{d(v)(d(v) - 1)}$$

The **global clustering coefficient** (transitivity):
$$C = \frac{3 \times \text{number of triangles}}{\text{number of connected triples}}$$

### Modularity
**Modularity** $Q$ measures the quality of a community partition:
$$Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{d_i d_j}{2m} \right] \delta(c_i, c_j)$$
where $c_i$ is the community of node $i$ and $m = |E|$.

## Degree Distribution

### Power-Law Distribution
Many real-world networks follow a **power-law** degree distribution:
$$P(d) \propto d^{-\gamma}$$
where $\gamma$ is typically between 2 and 3. Such networks are called **scale-free**.

### Small-World Property
A graph exhibits the **small-world** property if:
1. Average path length is small: $\bar{d} \sim \log n$
2. Clustering coefficient is high: $C \gg C_{random}$

## Graph Isomorphism and Expressiveness

### Weisfeiler-Leman (WL) Test
The **1-WL test** iteratively refines node labels based on neighborhood aggregation. Two graphs are non-isomorphic if their multisets of refined labels differ. The WL test is directly connected to GNN expressiveness:

- Standard message-passing GNNs are at most as powerful as the 1-WL test
- GIN (Graph Isomorphism Network) achieves 1-WL expressiveness

### Graph Invariants
Properties preserved under isomorphism:
- Number of nodes and edges
- Degree sequence
- Spectrum (eigenvalues)
- Number of triangles, cliques, cycles

## Spectral Gap

The **spectral gap** is $\lambda_1 - \lambda_2$ of the adjacency matrix (for regular graphs). A large spectral gap indicates:
- Good expansion properties
- Rapid mixing of random walks
- Efficient information propagation in GNNs

## Quantitative Finance Properties

Key graph properties for financial networks:
- **Density evolution**: How network connectivity changes during market stress
- **Clustering patterns**: Sector-based clustering in correlation networks
- **Hub structure**: Systemically important financial institutions have high centrality
- **Fragility**: Small spectral gap may indicate vulnerability to contagion
- **Core-periphery structure**: Common in interbank networks

## Summary

Graph properties inform both the selection of GNN architectures and the interpretation of learned representations. Properties like diameter, clustering, and spectral gap directly impact how information propagates through message-passing layers, affecting model depth requirements and performance.
