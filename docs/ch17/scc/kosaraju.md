# Kosaraju's Algorithm

Finding [strongly connected components](definition.md) efficiently requires a clever insight: if we could process vertices in an order that visits "sink" SCCs first, a single DFS pass on the transposed graph would isolate each component. Kosaraju's algorithm achieves this in two DFS passes -- the first determines the processing order and the second extracts the components. The result is an elegant $O(V + E)$ algorithm that is both easy to implement and easy to prove correct.

## Algorithm Overview

Kosaraju's algorithm performs three steps:

1. **First DFS pass on $G$.** Run DFS on the original graph and record vertices in order of decreasing finish time (i.e., push each vertex onto a stack when its DFS call finishes).

2. **Transpose the graph.** Construct the transpose graph $G^T = (V, E^T)$ where every edge $(u, v) \in E$ is reversed to $(v, u) \in E^T$.

3. **Second DFS pass on $G^T$.** Process vertices in the order from step 1 (popping from the stack). Each DFS tree in this pass forms one strongly connected component.

## Why It Works

The key insight is the relationship between finish times and the SCC structure.

!!! note "Finish-Time Property for SCCs"
    If $C_1$ and $C_2$ are two different SCCs and there is an edge from $C_1$ to $C_2$ in the [condensation graph](condensation.md), then the vertex with the latest finish time in $C_1 \cup C_2$ belongs to $C_1$.

**Proof sketch.** If DFS first enters $C_1$, it explores all of $C_1$ and then reaches $C_2$ via the inter-component edge. All vertices in $C_2$ finish before the DFS returns to $C_1$, so $C_1$'s vertices finish later. If DFS first enters $C_2$, it cannot reach $C_1$ (no edge from $C_2$ to $C_1$, since the condensation is a DAG). After finishing $C_2$, DFS eventually starts on a vertex in $C_1$, which finishes later. $\square$

**Consequence.** In the second pass on $G^T$, the vertex with the latest finish time starts DFS in a "source" SCC of the condensation DAG. In $G^T$, inter-component edges are reversed, so this SCC has no outgoing edges in $G^T$. The DFS therefore stays within this SCC, correctly identifying it. Once processed, we move to the next unvisited vertex with the highest finish time, which is in another source SCC of the remaining condensation -- and the process repeats.

## Complexity

Both DFS passes visit each vertex and edge exactly once. Constructing $G^T$ takes $O(V + E)$. Therefore:

$$
T(V, E) = O(V + E)
$$

Space complexity is $O(V + E)$ for storing $G^T$ and the finish-time stack.

## Implementation

```python
"""
Kosaraju's algorithm for finding strongly connected components.

Performs two DFS passes: one on the original graph to determine finish
order, and one on the transposed graph to extract SCCs.
"""


# === Kosaraju's Algorithm ===
def kosaraju_scc(graph, n):
    """
    Find all strongly connected components using Kosaraju's algorithm.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list of a directed graph with vertices 0 to n-1.
    n : int
        Number of vertices.

    Returns
    -------
    list[list[int]]
        List of SCCs, each represented as a list of vertex labels.
    """
    # Pass 1: DFS on original graph, record finish order
    visited = [False] * n
    finish_stack = []

    def dfs1(u):
        visited[u] = True
        for v in graph.get(u, []):
            if not visited[v]:
                dfs1(v)
        finish_stack.append(u)

    for u in range(n):
        if not visited[u]:
            dfs1(u)

    # Build transpose graph
    transpose = {i: [] for i in range(n)}
    for u in range(n):
        for v in graph.get(u, []):
            transpose[v].append(u)

    # Pass 2: DFS on transpose in reverse finish order
    visited = [False] * n
    sccs = []

    def dfs2(u, component):
        visited[u] = True
        component.append(u)
        for v in transpose.get(u, []):
            if not visited[v]:
                dfs2(v, component)

    while finish_stack:
        u = finish_stack.pop()
        if not visited[u]:
            component = []
            dfs2(u, component)
            sccs.append(component)

    return sccs


# === Main ===
if __name__ == "__main__":
    graph = {
        0: [1], 1: [2, 3], 2: [0], 3: [4],
        4: [5], 5: [3], 6: [5, 7], 7: [],
    }
    sccs = kosaraju_scc(graph, 8)
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

## Step-by-Step Trace

Using the example graph above:

**Pass 1 (DFS on $G$):** Starting from vertex 0, the DFS explores $0 \to 1 \to 2$ (back to 0, already visited), then $1 \to 3 \to 4 \to 5$ (back to 3, already visited). Finish order accumulates as vertices complete their exploration. After processing all vertices starting from 0 and then 6, the finish stack (bottom to top) might be: $[7, 5, 4, 3, 2, 0, 1, 6]$.

**Transpose $G^T$:** Reverse all edges. The edge $0 \to 1$ becomes $1 \to 0$, and so on.

**Pass 2 (DFS on $G^T$):** Pop vertex 6 from the stack. DFS on $G^T$ from 6 finds only 6 (no incoming edges in $G$). Pop vertex 1; DFS explores $\{1, 0, 2\}$ -- these form one SCC. Pop vertex 3; DFS explores $\{3, 5, 4\}$ -- another SCC. Pop vertex 7; it is alone.

## Comparison with Tarjan's Algorithm

| Property | Kosaraju's | [Tarjan's](tarjan.md) |
|---|---|---|
| DFS passes | Two (on $G$ and $G^T$) | One |
| Extra storage | Transpose graph $G^T$ | Low-link values and stack |
| Implementation | Simpler to understand | Slightly more complex |
| Time complexity | $O(V + E)$ | $O(V + E)$ |
| Space complexity | $O(V + E)$ for $G^T$ | $O(V)$ extra |

Kosaraju's algorithm is often preferred for teaching due to its conceptual clarity, while Tarjan's is preferred in practice when memory is constrained.

## Reference

- Sharir, M. (1981). A strong-connectivity algorithm and its applications in data flow analysis. *Computers & Mathematics with Applications*, 7(1), 67-72.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 20.
