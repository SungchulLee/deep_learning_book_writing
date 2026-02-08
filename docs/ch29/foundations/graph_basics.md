# 29.1.1 Graph Basics

## Introduction

A **graph** is a mathematical structure used to model pairwise relationships between objects. Formally, a graph $G = (V, E)$ consists of a set of **nodes** (or vertices) $V$ and a set of **edges** $E \subseteq V \times V$ connecting pairs of nodes. Graphs provide a natural representation for relational data across domains—social connections, molecular bonds, financial transactions, and network topologies.

## Formal Definitions

### Undirected Graphs

An **undirected graph** is a graph where edges have no orientation. If $(u, v) \in E$, then $(v, u) \in E$. The edge represents a symmetric relationship between nodes $u$ and $v$.

$$G = (V, E), \quad E \subseteq \{\{u, v\} : u, v \in V\}$$

### Directed Graphs (Digraphs)

A **directed graph** (digraph) has edges with orientation. An edge $(u, v) \in E$ indicates a directed relationship from $u$ to $v$, which does not imply $(v, u) \in E$.

$$G = (V, E), \quad E \subseteq \{(u, v) : u, v \in V\}$$

### Weighted Graphs

A **weighted graph** assigns a real-valued weight to each edge via a weight function $w: E \rightarrow \mathbb{R}$. Weights can represent distances, costs, strengths of connections, or transaction amounts.

### Multigraphs

A **multigraph** allows multiple edges between the same pair of nodes (parallel edges) and may include self-loops (edges from a node to itself).

## Key Terminology

| Term | Definition |
|------|-----------|
| **Degree** | Number of edges incident to a node. For directed graphs: in-degree and out-degree |
| **Neighbor** | Node $v$ is a neighbor of $u$ if $(u, v) \in E$. The neighborhood $\mathcal{N}(u) = \{v \in V : (u, v) \in E\}$ |
| **Path** | Sequence of nodes $(v_1, v_2, \ldots, v_k)$ where each consecutive pair is connected by an edge |
| **Cycle** | Path that starts and ends at the same node |
| **Connected** | An undirected graph is connected if there exists a path between every pair of nodes |
| **Subgraph** | A graph $G' = (V', E')$ where $V' \subseteq V$ and $E' \subseteq E \cap (V' \times V')$ |
| **Clique** | A complete subgraph where every pair of nodes is connected |
| **Bipartite** | A graph whose nodes can be divided into two disjoint sets such that edges only connect nodes from different sets |

## Degree Distribution

The **degree** of node $v$ is:

$$d(v) = |\mathcal{N}(v)|$$

For directed graphs:
- **In-degree**: $d_{in}(v) = |\{u : (u, v) \in E\}|$
- **Out-degree**: $d_{out}(v) = |\{u : (v, u) \in E\}|$

The **degree matrix** $D$ is a diagonal matrix where $D_{ii} = d(v_i)$.

The **Handshaking Lemma** states:

$$\sum_{v \in V} d(v) = 2|E|$$

## Common Graph Types

### Complete Graph $K_n$
Every pair of nodes is connected. Has $\binom{n}{2} = \frac{n(n-1)}{2}$ edges.

### Star Graph $S_n$
One central node connected to $n-1$ peripheral nodes.

### Tree
A connected acyclic graph with exactly $|V| - 1$ edges.

### DAG (Directed Acyclic Graph)
A directed graph with no directed cycles. Used extensively in dependency modeling and causal inference.

## Graph Data Structures

Graphs can be stored using several data structures, each with trade-offs:

1. **Adjacency List**: For each node, store a list of its neighbors. Memory-efficient for sparse graphs: $O(|V| + |E|)$.
2. **Adjacency Matrix**: $|V| \times |V|$ matrix. Simple but memory-intensive: $O(|V|^2)$.
3. **Edge List**: List of all edges as pairs $(u, v)$. Compact: $O(|E|)$.
4. **Incidence Matrix**: $|V| \times |E|$ matrix relating nodes to edges.

## Quantitative Finance Context

Graphs naturally arise in quantitative finance:

- **Correlation Networks**: Nodes are assets; edge weights are pairwise correlations
- **Transaction Networks**: Nodes are accounts; directed edges represent money flows
- **Supply Chain Graphs**: Nodes are companies; edges represent supplier-customer relationships
- **Interbank Networks**: Nodes are banks; edges represent lending relationships
- **Options Market Graphs**: Nodes are options contracts; edges connect contracts on the same underlying

Understanding graph structure enables analysis of systemic risk, fraud detection, portfolio construction, and market microstructure.

## Summary

Graphs provide a flexible and powerful formalism for representing relational data. The choice of graph type (directed, weighted, etc.) and storage representation depends on the application domain and computational requirements. In the following sections, we formalize these representations mathematically and implement them using modern deep learning libraries.
