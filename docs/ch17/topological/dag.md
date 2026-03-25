# Directed Acyclic Graph Property

Many real-world processes have a natural ordering: university courses have prerequisites, software builds depend on libraries, and task schedules must respect deadlines. All these scenarios share a common structure -- a directed graph with no cycles. Understanding when a directed graph is acyclic is the foundation of topological sorting, because a topological order exists if and only if the graph is a DAG.

## Definition

A **directed acyclic graph (DAG)** is a directed graph $G = (V, E)$ that contains no directed cycle. Equivalently, there is no vertex $v \in V$ such that a directed path from $v$ leads back to $v$.

More precisely, $G$ is a DAG if and only if there is no sequence of vertices $v_0, v_1, \ldots, v_k$ with $k \geq 1$ such that $(v_i, v_{i+1}) \in E$ for all $0 \leq i < k$ and $v_k = v_0$.

## Key Theorem

The central result connecting DAGs to topological sorting is:

!!! tip "DAG-Topological Order Equivalence"
    A directed graph $G$ admits a topological ordering if and only if $G$ is a DAG.

**Proof sketch (forward direction).** Suppose $G$ has a topological ordering $v_1, v_2, \ldots, v_n$ where every edge $(v_i, v_j)$ satisfies $i < j$. If $G$ contained a directed cycle $v_{a_1} \to v_{a_2} \to \cdots \to v_{a_k} \to v_{a_1}$, then we would need $a_1 < a_2 < \cdots < a_k < a_1$, which is a contradiction. Therefore $G$ must be acyclic. $\square$

**Proof sketch (reverse direction).** Suppose $G$ is a DAG. Every DAG has at least one vertex with in-degree zero (otherwise, following predecessors indefinitely in a finite graph would produce a cycle). Remove such a vertex, add it to the ordering, and repeat on the remaining graph (which is still a DAG). This process produces a valid topological ordering of all vertices. $\square$

## Properties of DAGs

DAGs have several important structural properties that algorithms exploit.

**Source and sink vertices.** Every non-empty DAG has at least one source (a vertex with in-degree zero) and at least one sink (a vertex with out-degree zero). If a DAG had no source, we could trace predecessors indefinitely, eventually revisiting a vertex and forming a cycle -- contradicting the acyclicity assumption.

**Longest path.** The longest path in a DAG can be computed in $O(V + E)$ time using topological sort followed by dynamic programming. This is in contrast to general directed graphs, where finding the longest path is NP-hard.

**Number of topological orderings.** A DAG may have many valid topological orderings. The number of distinct orderings depends on the graph structure. A path graph $v_1 \to v_2 \to \cdots \to v_n$ has exactly one topological order, while a graph with no edges on $n$ vertices has $n!$ orderings.

## Cycle Detection

Since a directed graph admits a topological ordering if and only if it is a DAG, any topological sort algorithm doubles as a cycle detector. Two standard approaches exist:

1. **DFS-based detection.** During a depth-first search, if we encounter a back edge (an edge to a vertex currently on the recursion stack), the graph contains a cycle. See the [DFS-based topological sort](dfs.md) page for details.

2. **Kahn's algorithm.** If Kahn's algorithm terminates without processing all vertices, the remaining vertices form one or more cycles. See the [Kahn's algorithm](kahn.md) page for details.

```python
"""
Cycle detection in a directed graph using DFS.

Demonstrates how back edges in a DFS traversal reveal directed cycles,
confirming that the graph is not a DAG.
"""


# === Cycle Detection via DFS ===
def has_cycle(graph, n):
    """
    Determine whether a directed graph contains a cycle.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list representation of the directed graph.
    n : int
        Number of vertices (labeled 0 through n-1).

    Returns
    -------
    bool
        True if the graph contains a directed cycle, False otherwise.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n

    def dfs(u):
        color[u] = GRAY
        for v in graph.get(u, []):
            if color[v] == GRAY:
                return True  # back edge found => cycle
            if color[v] == WHITE and dfs(v):
                return True
        color[u] = BLACK
        return False

    return any(color[u] == WHITE and dfs(u) for u in range(n))


# === Main ===
if __name__ == "__main__":
    # A DAG: 0 -> 1 -> 3, 0 -> 2 -> 3 -> 4
    dag = {0: [1, 2], 1: [3], 2: [3], 3: [4], 4: []}
    print(f"DAG has cycle: {has_cycle(dag, 5)}")

    # A graph with a cycle: 0 -> 1 -> 2 -> 0
    cyclic = {0: [1], 1: [2], 2: [0]}
    print(f"Cyclic graph has cycle: {has_cycle(cyclic, 3)}")
```

**Output:**
```
DAG has cycle: False
Cyclic graph has cycle: True
```

The three-coloring scheme above marks each vertex as WHITE (unvisited), GRAY (on the current recursion stack), or BLACK (fully processed). A GRAY-to-GRAY edge is a back edge, which proves the existence of a cycle.

## Relationship to Topological Sorting

The DAG property is the gateway to all topological sorting algorithms. Once we confirm a graph is a DAG, we can produce a linear ordering of its vertices such that for every directed edge $(u, v)$, vertex $u$ appears before $v$. The two main approaches are:

- [**Kahn's algorithm**](kahn.md) -- iteratively removes source vertices using a queue
- [**DFS-based sort**](dfs.md) -- uses finish times from depth-first search

Both run in $O(V + E)$ time and can simultaneously verify the DAG property.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 20.
