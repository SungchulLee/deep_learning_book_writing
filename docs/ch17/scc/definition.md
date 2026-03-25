# Strongly Connected Components

In an undirected graph, connected components capture the natural clusters of reachable vertices. For directed graphs, reachability is asymmetric -- vertex $u$ may reach $v$ without $v$ being able to reach $u$. Strongly connected components (SCCs) identify the maximal groups of vertices with mutual reachability, revealing the deep structure hidden within directed graphs.

## Definition

Two vertices $u$ and $v$ in a directed graph $G = (V, E)$ are **strongly connected** if there exists a directed path from $u$ to $v$ and a directed path from $v$ to $u$.

!!! note "Formal Definition"
    A **strongly connected component** of a directed graph $G = (V, E)$ is a maximal set $C \subseteq V$ such that for every pair of vertices $u, v \in C$, there exists a directed path from $u$ to $v$ and a directed path from $v$ to $u$.

The word **maximal** is critical: an SCC cannot be enlarged by adding another vertex while maintaining the mutual reachability property. Every vertex belongs to exactly one SCC, so the SCCs partition $V$.

## Equivalence Relation

Strong connectivity defines an equivalence relation on the vertex set $V$:

- **Reflexive:** Every vertex $u$ reaches itself via the trivial path.
- **Symmetric:** If $u$ reaches $v$ and $v$ reaches $u$, then $v$ reaches $u$ and $u$ reaches $v$.
- **Transitive:** If $u$ reaches $v$ (and $v$ reaches $u$) and $v$ reaches $w$ (and $w$ reaches $v$), then $u$ reaches $w$ via $v$, and $w$ reaches $u$ via $v$.

Since strong connectivity is an equivalence relation, it partitions $V$ into equivalence classes -- these are exactly the strongly connected components.

## Properties

**Uniqueness of decomposition.** Every directed graph has a unique SCC decomposition. This follows directly from the equivalence relation property.

**Single-vertex SCCs.** A vertex with no path to or from any other vertex forms a singleton SCC by itself. In a DAG, every vertex is its own SCC.

**Relationship to DAGs.** The [condensation graph](condensation.md) -- formed by contracting each SCC into a single super-vertex -- is always a DAG. This connection makes SCCs a fundamental tool for understanding the structure of directed graphs.

**Component graph.** If we define $G^{SCC} = (V^{SCC}, E^{SCC})$ where each SCC is a vertex and there is an edge between two SCC vertices if any edge in $G$ connects them, then $G^{SCC}$ contains no directed cycle. If it did, the vertices in the cycle would form a single larger SCC, contradicting maximality.

## Worked Example

Consider the directed graph with vertices $\{0, 1, 2, 3, 4, 5, 6, 7\}$ and edges:

$$
0 \to 1,\quad 1 \to 2,\quad 2 \to 0,\quad 1 \to 3,\quad 3 \to 4,\quad 4 \to 5,\quad 5 \to 3,\quad 6 \to 5,\quad 6 \to 7
$$

The strongly connected components are:

- $C_1 = \{0, 1, 2\}$: the cycle $0 \to 1 \to 2 \to 0$ makes all three mutually reachable.
- $C_2 = \{3, 4, 5\}$: the cycle $3 \to 4 \to 5 \to 3$ makes all three mutually reachable.
- $C_3 = \{6\}$: vertex 6 can reach other vertices but nothing reaches 6.
- $C_4 = \{7\}$: vertex 7 is reachable from 6 but cannot reach any other vertex.

```python
"""
Visualizing strongly connected components in a directed graph.

Uses a brute-force approach (checking all-pairs reachability) to
identify SCCs, illustrating the definition before introducing
efficient algorithms.
"""

from collections import deque


# === Brute-Force SCC Detection ===
def find_sccs_brute(graph, n):
    """
    Find SCCs by checking all-pairs reachability.

    This O(V * (V + E)) approach is for illustration only.
    See Kosaraju's and Tarjan's algorithms for O(V + E) solutions.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list of a directed graph.
    n : int
        Number of vertices.

    Returns
    -------
    list[set[int]]
        List of strongly connected components.
    """
    def reachable(start):
        visited = set()
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if u in visited:
                continue
            visited.add(u)
            for v in graph.get(u, []):
                if v not in visited:
                    queue.append(v)
        return visited

    assigned = [False] * n
    components = []

    for u in range(n):
        if assigned[u]:
            continue
        reach_from_u = reachable(u)
        component = set()
        for v in reach_from_u:
            if u in reachable(v):
                component.add(v)
        for v in component:
            assigned[v] = True
        components.append(component)

    return components


# === Main ===
if __name__ == "__main__":
    graph = {
        0: [1], 1: [2, 3], 2: [0], 3: [4],
        4: [5], 5: [3], 6: [5, 7], 7: [],
    }
    sccs = find_sccs_brute(graph, 8)
    print("Strongly connected components:")
    for i, scc in enumerate(sccs):
        print(f"  C{i+1} = {sorted(scc)}")
```

**Output:**
```
Strongly connected components:
  C1 = [0, 1, 2]
  C2 = [3, 4, 5]
  C3 = [6]
  C4 = [7]
```

The brute-force approach above has $O(V \cdot (V + E))$ complexity. Efficient algorithms by [Kosaraju](kosaraju.md) and [Tarjan](tarjan.md) compute SCCs in $O(V + E)$ time.

## Applications

Strongly connected components appear in many practical contexts:

- **Web graph analysis:** pages in the same SCC are mutually reachable via hyperlinks.
- **Compiler optimization:** SCCs in a call graph identify mutually recursive functions.
- **Social networks:** SCCs in a follow graph reveal tightly connected communities.
- **2-SAT solving:** the satisfiability of a [2-SAT formula](two_sat.md) can be determined by examining the SCCs of its implication graph.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 20.
