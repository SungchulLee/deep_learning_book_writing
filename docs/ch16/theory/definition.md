# MST Definition

When designing a network -- whether it connects cities with roads, computers with cables, or circuit components with wires -- a fundamental question arises: what is the cheapest way to connect everything? The answer lies in finding a minimum spanning tree. This page formalizes the concept, starting from the definition of a spanning tree and building up to the MST optimization problem.

## Spanning Tree

A **spanning tree** of an undirected, connected graph $G = (V, E)$ is a subgraph $T = (V, E_T)$ that satisfies three properties simultaneously:

1. **Spans all vertices**: $T$ includes every vertex in $V$.
2. **Is connected**: there exists a path between every pair of vertices in $T$.
3. **Is acyclic**: $T$ contains no cycles.

Any subgraph satisfying these three conditions is a tree on $|V|$ vertices and therefore has exactly $|V| - 1$ edges.

??? note "Why exactly |V| - 1 edges?"
    A connected graph on $n$ vertices requires at least $n - 1$ edges (otherwise some vertex is isolated). A tree achieves this minimum: removing any edge disconnects it, and adding any edge creates a cycle. By induction on $n$, every tree on $n$ vertices has exactly $n - 1$ edges.

A connected graph $G$ with $|V| = n$ vertices and $|E| = m$ edges has at least one spanning tree whenever $G$ is connected. In fact, a connected graph may have exponentially many distinct spanning trees. For instance, the complete graph $K_n$ has exactly $n^{n-2}$ spanning trees, a result known as **Cayley's formula**.

## Minimum Spanning Tree

Given a connected, undirected graph $G = (V, E)$ with a weight function $w : E \to \mathbb{R}$, the **weight of a spanning tree** $T = (V, E_T)$ is the sum of its edge weights:

$$
w(T) = \sum_{e \in E_T} w(e)
$$

A **minimum spanning tree** (MST) of $G$ is a spanning tree $T^*$ whose total weight is minimum among all spanning trees of $G$:

$$
T^* = \arg\min_{T \in \mathcal{T}(G)} \sum_{e \in E_T} w(e)
$$

where $\mathcal{T}(G)$ denotes the set of all spanning trees of $G$.

??? note "Existence and finiteness"
    Since a connected graph with $n$ vertices and $m$ edges has a finite (though possibly exponential) number of spanning trees, the minimum over a finite nonempty set always exists. The MST is therefore guaranteed to exist for any connected, weighted, undirected graph.

## Key Properties

An MST satisfies several important structural properties that underpin the algorithms for computing it:

- **Edge count**: an MST of a graph with $n$ vertices has exactly $n - 1$ edges.
- **Subgraph optimality**: for any partition of the vertices into two nonempty sets, the MST includes a lightest edge crossing the partition (the **cut property**, covered on the next page).
- **No heavy cycles**: if adding an edge $e$ to the MST creates a cycle, then $e$ is the heaviest edge in that cycle (the **cycle property**).
- **Uniqueness**: if all edge weights are distinct, the MST is unique. When ties exist, multiple MSTs may share the same total weight.

## Example

Consider a graph with four vertices and five edges:

| Edge | Weight |
|------|--------|
| (A, B) | 4 |
| (A, C) | 1 |
| (B, C) | 3 |
| (B, D) | 2 |
| (C, D) | 5 |

The graph has several spanning trees. Two of them are:

- $T_1 = \{(A,C), (B,C), (B,D)\}$ with weight $1 + 3 + 2 = 6$
- $T_2 = \{(A,B), (A,C), (B,D)\}$ with weight $4 + 1 + 2 = 7$

The MST is $T_1$ with total weight 6. No spanning tree of this graph achieves a lower total weight.

## Formal Problem Statement

The MST problem can be stated precisely as follows.

**Input**: A connected, undirected graph $G = (V, E)$ with weight function $w : E \to \mathbb{R}$.

**Output**: A spanning tree $T^* = (V, E_{T^*})$ of $G$ such that

$$
w(T^*) \le w(T) \quad \text{for all spanning trees } T \text{ of } G
$$

The three classical algorithms for solving this problem -- Kruskal's, Prim's, and Boruvka's -- all rely on the greedy paradigm and are justified by the cut and cycle properties introduced on the following pages.

## Reference

- [Introduction to Algorithms (CLRS), Chapter 23](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Cayley, A. (1889). A theorem on trees. *Quarterly Journal of Mathematics*, 23, 376--378.
