# Condensation Graph

Every directed graph, no matter how complex, has a hidden DAG structure waiting to be revealed. The **condensation graph** (also called the **DAG of SCCs**) is obtained by collapsing each [strongly connected component](definition.md) into a single super-vertex. The result is always a directed acyclic graph, which means we can apply all the powerful techniques available for DAGs -- topological sorting, shortest paths, dynamic programming -- to analyze the macro-structure of the original graph.

## Definition

!!! note "Formal Definition"
    Let $G = (V, E)$ be a directed graph with strongly connected components $C_1, C_2, \ldots, C_k$. The **condensation graph** $G^{SCC} = (V^{SCC}, E^{SCC})$ is defined as:

    - $V^{SCC} = \{C_1, C_2, \ldots, C_k\}$ (one vertex per SCC)
    - $(C_i, C_j) \in E^{SCC}$ if and only if there exist vertices $u \in C_i$ and $v \in C_j$ with $(u, v) \in E$ and $i \neq j$

Duplicate edges between the same pair of super-vertices are collapsed into a single edge. Self-loops (which would require $i = j$) are excluded by definition.

## The Condensation Is a DAG

!!! tip "Key Property"
    The condensation graph $G^{SCC}$ is always a directed acyclic graph.

**Proof.** Suppose for contradiction that $G^{SCC}$ contains a directed cycle $C_{a_1} \to C_{a_2} \to \cdots \to C_{a_m} \to C_{a_1}$. Then for any $u \in C_{a_1}$, we can reach some vertex in $C_{a_2}$ (via the inter-component edge), then some vertex in $C_{a_3}$, and so on, eventually returning to a vertex in $C_{a_1}$. Within each SCC, all vertices are mutually reachable, so every vertex in $C_{a_1} \cup C_{a_2} \cup \cdots \cup C_{a_m}$ can reach every other. This means they all belong to a single SCC, contradicting the maximality of the SCC decomposition. $\square$

## Construction

Constructing the condensation graph requires:

1. Find all SCCs using [Tarjan's algorithm](tarjan.md) or [Kosaraju's algorithm](kosaraju.md) -- $O(V + E)$.
2. Assign each vertex its SCC label.
3. For each edge $(u, v)$ in $G$, if $\text{scc}[u] \neq \text{scc}[v]$, add edge $(\text{scc}[u], \text{scc}[v])$ to $G^{SCC}$ (deduplicating).

The total construction time is $O(V + E)$.

```python
"""
Condensation graph construction.

Computes strongly connected components using Tarjan's algorithm,
then builds the DAG of SCCs (condensation graph).
"""


# === Tarjan's SCC (helper) ===
def tarjan_scc(graph, n):
    """Find SCCs and return (list of SCCs, vertex-to-SCC mapping)."""
    disc = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    stack = []
    timer = [0]
    sccs = []
    scc_id = [0] * n

    def dfs(u):
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        stack.append(u)
        on_stack[u] = True

        for v in graph.get(u, []):
            if disc[v] == -1:
                dfs(v)
                low[u] = min(low[u], low[v])
            elif on_stack[v]:
                low[u] = min(low[u], disc[v])

        if low[u] == disc[u]:
            component = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc_id[w] = len(sccs)
                component.append(w)
                if w == u:
                    break
            sccs.append(component)

    for u in range(n):
        if disc[u] == -1:
            dfs(u)

    return sccs, scc_id


# === Build Condensation Graph ===
def build_condensation(graph, n):
    """
    Build the condensation (DAG of SCCs) of a directed graph.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list of a directed graph with vertices 0 to n-1.
    n : int
        Number of vertices.

    Returns
    -------
    tuple
        (sccs, condensation_adj, scc_id) where condensation_adj
        maps each SCC index to a list of neighboring SCC indices.
    """
    sccs, scc_id = tarjan_scc(graph, n)
    k = len(sccs)
    cond_edges = set()
    cond_adj = {i: [] for i in range(k)}

    for u in range(n):
        for v in graph.get(u, []):
            if scc_id[u] != scc_id[v]:
                edge = (scc_id[u], scc_id[v])
                if edge not in cond_edges:
                    cond_edges.add(edge)
                    cond_adj[scc_id[u]].append(scc_id[v])

    return sccs, cond_adj, scc_id


# === Main ===
if __name__ == "__main__":
    graph = {
        0: [1], 1: [2, 3], 2: [0], 3: [4],
        4: [5], 5: [3], 6: [5, 7], 7: [],
    }
    sccs, cond_adj, scc_id = build_condensation(graph, 8)

    print("SCCs:")
    for i, scc in enumerate(sccs):
        print(f"  SCC {i}: {sorted(scc)}")

    print("\nCondensation graph edges:")
    for u in cond_adj:
        for v in cond_adj[u]:
            print(f"  SCC {u} -> SCC {v}")
```

**Output:**
```
SCCs:
  SCC 0: [3, 4, 5]
  SCC 1: [0, 1, 2]
  SCC 2: [7]
  SCC 3: [6]

Condensation graph edges:
  SCC 1 -> SCC 0
  SCC 3 -> SCC 0
  SCC 3 -> SCC 2
```

## Applications of the Condensation

Since the condensation graph is a DAG, we can apply DAG algorithms to solve problems on the original graph:

**Reachability.** Two vertices $u$ and $v$ are mutually reachable in $G$ if and only if they belong to the same SCC. One-directional reachability reduces to reachability in the condensation DAG, which can be answered after a topological sort.

**Minimum vertex set for full reachability.** The source SCCs (those with in-degree zero in the condensation) are the components from which all other vertices can be reached. The minimum number of vertices needed to reach all others equals the number of source SCCs.

**Longest path.** The longest path in $G$ (ignoring cycles within SCCs) can be computed by finding the longest path in the condensation DAG, weighting each super-vertex by the number of vertices in its SCC.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 20.
