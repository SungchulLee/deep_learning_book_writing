# Hamiltonian Path and Cycle

While an Eulerian circuit visits every **edge** exactly once, a Hamiltonian cycle visits every **vertex** exactly once. This seemingly small change in the problem statement has dramatic consequences: determining whether a graph has a Hamiltonian cycle is NP-complete, meaning no efficient general algorithm is known. Nevertheless, several sufficient conditions guarantee existence, and practical backtracking algorithms solve moderate-sized instances.

## Definitions

**Hamiltonian path.** A simple path in $G = (V, E)$ that visits every vertex exactly once.

**Hamiltonian cycle.** A simple cycle that visits every vertex exactly once and returns to the starting vertex. Equivalently, a Hamiltonian path from $v$ to $w$ where $(w, v) \in E$.

**Hamiltonian graph.** A graph that contains a Hamiltonian cycle.

## Contrast with Euler

| Property | Eulerian | Hamiltonian |
|----------|:--------:|:-----------:|
| Visits every ... | edge | vertex |
| Existence check | $O(V + E)$ via degree parity | NP-complete |
| Efficient characterization | Yes (degree theorem) | No known characterization |

## Sufficient Conditions

No simple necessary-and-sufficient condition for Hamiltonicity is known, but several classical theorems provide sufficient conditions.

!!! note "Dirac's Theorem (1952)"
    If $G$ is a simple graph on $n \ge 3$ vertices and every vertex satisfies $\deg(v) \ge n/2$, then $G$ is Hamiltonian.

!!! note "Ore's Theorem (1960)"
    If $G$ is a simple graph on $n \ge 3$ vertices and for every pair of non-adjacent vertices $u, v$ we have $\deg(u) + \deg(v) \ge n$, then $G$ is Hamiltonian.

Ore's theorem generalizes Dirac's theorem: if every vertex has degree at least $n/2$, then any pair of vertices satisfies $\deg(u) + \deg(v) \ge n$.

## Complexity

The Hamiltonian cycle problem is one of Karp's original 21 NP-complete problems (1972). It remains NP-complete even when restricted to:

- Planar graphs with maximum degree 3.
- Bipartite graphs.
- Grid graphs.

The best known exact algorithms run in $O^*(2^n)$ time using dynamic programming over subsets (the Held-Karp algorithm), improving on the $O(n!)$ naive approach.

## Backtracking Algorithm

A backtracking approach builds a path vertex by vertex, pruning branches that cannot lead to a valid Hamiltonian path or cycle. While worst-case exponential, this is practical for small to moderate graphs.

```python
"""
Hamiltonian cycle detection via backtracking.

Tests whether an undirected graph contains a Hamiltonian cycle by
building a path one vertex at a time and pruning invalid extensions.
"""

# === Backtracking Hamiltonian Cycle ===

def hamiltonian_cycle(n: int, edges: list[tuple[int, int]]) -> list[int] | None:
    """Find a Hamiltonian cycle if one exists.

    Args:
        n: Number of vertices (0-indexed).
        edges: List of undirected edges.

    Returns:
        List of vertices forming the cycle, or None if no cycle exists.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    path = [0]
    visited = {0}

    def backtrack() -> bool:
        if len(path) == n:
            # Check if we can return to start
            return 0 in adj[path[-1]]

        last = path[-1]
        for neighbor in sorted(adj[last]):
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                if backtrack():
                    return True
                path.pop()
                visited.remove(neighbor)
        return False

    if backtrack():
        return path + [path[0]]
    return None


# === Demonstration ===

if __name__ == "__main__":
    # Complete graph K4
    k4_edges = [(i, j) for i in range(4) for j in range(i+1, 4)]
    result = hamiltonian_cycle(4, k4_edges)
    print(f"K4 Hamiltonian cycle: {result}")

    # Path graph: 0-1-2-3 (no Hamiltonian cycle)
    path_edges = [(0,1),(1,2),(2,3)]
    result = hamiltonian_cycle(4, path_edges)
    print(f"Path graph cycle: {result}")

    # Petersen graph (Hamiltonian path exists, but no cycle)
    petersen = [
        (0,1),(1,2),(2,3),(3,4),(4,0),  # outer cycle
        (0,5),(1,6),(2,7),(3,8),(4,9),  # spokes
        (5,7),(7,9),(9,6),(6,8),(8,5),  # inner pentagram
    ]
    result = hamiltonian_cycle(10, petersen)
    print(f"Petersen graph cycle: {result}")
```

**Output:**

```
K4 Hamiltonian cycle: [0, 1, 2, 3, 0]
Path graph cycle: None
Petersen graph cycle: None
```

The complete graph $K_4$ satisfies Dirac's condition ($\deg(v) = 3 \ge 4/2$) and indeed has a Hamiltonian cycle. The path graph on 4 vertices has no cycle at all. The Petersen graph famously has Hamiltonian paths but no Hamiltonian cycle.

## Dynamic Programming Approach

The Held-Karp algorithm uses bitmask DP to find a Hamiltonian path in $O(n^2 \cdot 2^n)$ time and $O(n \cdot 2^n)$ space. Define:

$$
\text{dp}[S][v] = \text{True if there is a path visiting exactly the vertices in } S \text{ and ending at } v
$$

The recurrence is:

$$
\text{dp}[S][v] = \bigvee_{u \in S \setminus \{v\},\; (u,v) \in E} \text{dp}[S \setminus \{v\}][u]
$$

A Hamiltonian cycle exists if $\text{dp}[\{0, 1, \dots, n{-}1\}][v] = \text{True}$ for some $v$ adjacent to the starting vertex.

## Reference

- Karp, R. M. (1972). Reducibility among combinatorial problems. In *Complexity of Computer Computations*, pp. 85--103.
- Ore, O. (1960). Note on Hamilton circuits. *The American Mathematical Monthly*, 67(1), 55.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 34: NP-Completeness.
