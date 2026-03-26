# Graphic Matroid

The graphic matroid is the most natural and historically important example of a matroid. Given any undirected graph, the edge sets that form forests (acyclic subgraphs) satisfy the matroid axioms. This matroid structure is exactly what makes greedy algorithms like Kruskal's produce optimal spanning trees. Understanding graphic matroids reveals why certain graph problems yield to greedy strategies while others do not.

## Definition

Given an undirected graph $G = (V, E)$, the **graphic matroid** (also called the **cycle matroid**) is $M(G) = (E, \mathcal{I})$ where:

- The **ground set** is the edge set $E$.
- A subset $F \subseteq E$ is **independent** if and only if $F$ is acyclic (i.e., $F$ forms a forest).

The key components map directly to graph concepts:

| Matroid concept | Graph interpretation |
|-----------------|---------------------|
| Ground set $E$ | All edges of $G$ |
| Independent set | Forest (acyclic edge subset) |
| Circuit | Simple cycle |
| Base | Spanning forest |
| Rank of $A \subseteq E$ | Number of edges in a spanning forest of the subgraph induced by $A$ |

## Verifying the Matroid Axioms

We must check that the acyclic subsets of $E$ satisfy the three matroid axioms.

**Axiom 1 (Non-emptiness).** The empty set $\emptyset$ contains no edges, so it is trivially acyclic. Thus $\emptyset \in \mathcal{I}$.

**Axiom 2 (Hereditary property).** If $F$ is acyclic and $F' \subseteq F$, then $F'$ is also acyclic. Removing edges from a forest cannot create a cycle.

**Axiom 3 (Exchange property).** Let $F_1, F_2 \in \mathcal{I}$ with $|F_1| < |F_2|$. Since both are forests, $F_1$ spans at most $|V| - |F_1|$ connected components, and $F_2$ spans $|V| - |F_2|$ components. Because $|F_1| < |F_2|$, the forest $F_2$ has fewer components, so some edge $e \in F_2 \setminus F_1$ connects two components of $F_1$. Adding $e$ to $F_1$ cannot create a cycle (it bridges two components), so $F_1 \cup \{e\} \in \mathcal{I}$.

!!! note "Why the Exchange Property Works"
    A forest on $n$ vertices with $k$ edges has exactly $n - k$ connected components. If $F_2$ has more edges than $F_1$, it has fewer components. By the pigeonhole principle, at least one edge of $F_2$ connects two different components of $F_1$, and adding that edge preserves acyclicity.

## Rank and Bases

The **rank** of the graphic matroid equals $|V| - c(G)$, where $c(G)$ is the number of connected components of $G$. For a connected graph, the rank is $|V| - 1$.

All bases (maximal independent sets) are **spanning forests**. For a connected graph, every base is a spanning tree with exactly $|V| - 1$ edges. The matroid axioms guarantee that all bases have the same cardinality, which in graph terms means all spanning trees of a connected graph have the same number of edges.

## Circuits

A **circuit** in the graphic matroid is a minimal dependent set, which corresponds to a simple cycle in the graph. The fundamental property of circuits connects to graph theory:

!!! note "Unique Circuit Property"
    If $F$ is a forest and $e \notin F$, then $F \cup \{e\}$ contains at most one cycle. This cycle, if it exists, is the unique circuit containing $e$ relative to $F$.

This property is essential for understanding why adding an edge to a spanning tree creates exactly one cycle, and removing any edge from that cycle yields another spanning tree.

## Connection to Minimum Spanning Trees

The graphic matroid explains why Kruskal's algorithm works. By the matroid greedy theorem, a greedy algorithm that processes elements by increasing weight and adds each element if it preserves independence produces a maximum-weight base. For the graphic matroid with negated weights, this gives the minimum spanning tree.

Specifically, Kruskal's algorithm:

1. Sorts edges by weight.
2. Processes each edge in order.
3. Adds the edge if it does not create a cycle (i.e., remains independent in the graphic matroid).

The resulting set is a minimum-weight base, which is exactly a minimum spanning tree.

## Implementation

```python
"""
Graphic matroid operations.

Demonstrates the graphic matroid by verifying axioms, computing rank,
and finding a minimum-weight base (minimum spanning tree) using
the matroid greedy algorithm.
"""

from itertools import combinations

# === Union-Find for Cycle Detection ===

class UnionFind:
    """Union-Find with union by rank and path compression."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        """Unite sets containing x and y. Returns False if already same set."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


# === Graphic Matroid ===

def is_independent(n: int, edges: list[tuple[int, int]]) -> bool:
    """Check if a set of edges forms a forest (is independent).

    Args:
        n: Number of vertices.
        edges: List of (u, v) edges.

    Returns:
        True if the edge set is acyclic.
    """
    uf = UnionFind(n)
    for u, v in edges:
        if not uf.union(u, v):
            return False
    return True


def matroid_rank(n: int, edges: list[tuple[int, int]]) -> int:
    """Compute the rank of an edge set in the graphic matroid.

    The rank equals the size of the largest acyclic subset.

    Args:
        n: Number of vertices.
        edges: List of (u, v) edges.

    Returns:
        Size of a maximum forest within the given edges.
    """
    uf = UnionFind(n)
    rank = 0
    for u, v in edges:
        if uf.union(u, v):
            rank += 1
    return rank


def minimum_spanning_tree(
    n: int,
    weighted_edges: list[tuple[int, int, float]]
) -> list[tuple[int, int, float]]:
    """Find MST using the matroid greedy algorithm (Kruskal's).

    Args:
        n: Number of vertices.
        weighted_edges: List of (u, v, weight) tuples.

    Returns:
        List of edges in the minimum spanning tree.
    """
    sorted_edges = sorted(weighted_edges, key=lambda e: e[2])
    uf = UnionFind(n)
    mst = []

    for u, v, w in sorted_edges:
        if uf.union(u, v):
            mst.append((u, v, w))
            if len(mst) == n - 1:
                break

    return mst


# === Demonstration ===

if __name__ == "__main__":
    # Graph: 4 vertices, 5 edges
    #   0 --1-- 1
    #   |      /|
    #   4    3  2
    #   |  /    |
    #   2 --5-- 3
    n = 4
    edges = [(0, 1), (1, 3), (1, 2), (0, 2), (2, 3)]

    # Check independence of various subsets
    print("Independence checks:")
    forest = [(0, 1), (1, 2), (2, 3)]
    print(f"  {forest}: {is_independent(n, forest)}")

    cycle = [(0, 1), (1, 2), (0, 2)]
    print(f"  {cycle}: {is_independent(n, cycle)}")

    # Rank
    print(f"\nRank of all edges: {matroid_rank(n, edges)}")
    print(f"Expected (|V| - 1): {n - 1}")

    # Minimum spanning tree
    weighted = [(0, 1, 1), (1, 3, 3), (1, 2, 2), (0, 2, 4), (2, 3, 5)]
    mst = minimum_spanning_tree(n, weighted)
    total = sum(w for _, _, w in mst)
    print(f"\nMST edges: {[(u, v, w) for u, v, w in mst]}")
    print(f"MST weight: {total}")
```

**Output:**

```
Independence checks:
  [(0, 1), (1, 2), (2, 3)]: True
  [(0, 1), (1, 2), (0, 2)]: False

Rank of all edges: 3
Expected (|V| - 1): 3

MST edges: [(0, 1, 1), (1, 2, 2), (1, 3, 3)]
MST weight: 6
```

The forest $\{(0,1), (1,2), (2,3)\}$ is independent (acyclic), while $\{(0,1), (1,2), (0,2)\}$ forms a triangle and is dependent. The rank equals $|V| - 1 = 3$ for this connected graph. Kruskal's algorithm (the matroid greedy algorithm) finds the MST with weight 6.

## Cographic Matroid

The **dual** of a graphic matroid is the **cographic matroid** $M^*(G) = (E, \mathcal{I}^*)$, where $F \subseteq E$ is independent if and only if $G \setminus F$ (the graph with edges $F$ removed) remains connected. The bases of $M^*(G)$ are the complements of spanning trees: if $T$ is a spanning tree, then $E \setminus T$ is a base of the cographic matroid.

## Reference

- Whitney, H. (1935). On the abstract properties of linear dependence. *American Journal of Mathematics*, 57(3), 509--533.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 16: Greedy Algorithms.
- Oxley, J. G. (2011). *Matroid Theory* (2nd ed.). Oxford University Press.
