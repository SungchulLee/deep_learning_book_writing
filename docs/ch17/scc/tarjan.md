# Tarjan's Algorithm

[Kosaraju's algorithm](kosaraju.md) finds strongly connected components in two DFS passes and requires constructing the transpose graph. Tarjan's algorithm achieves the same result in a single DFS pass and with no transpose construction, making it more space-efficient in practice. The key idea is to track, for each vertex, the earliest ancestor reachable through back edges, and to use this information to identify the "root" of each SCC during the DFS itself.

## Low-Link Values

Tarjan's algorithm assigns each vertex $u$ two values:

- $\text{disc}[u]$: the discovery time of $u$ in the DFS (the order in which $u$ is first visited).
- $\text{low}[u]$: the smallest discovery time reachable from $u$ through the DFS subtree of $u$, including back edges.

The low-link value is defined recursively:

$$
\text{low}[u] = \min\!\Big(\text{disc}[u],\ \min_{(u,v) \in E} \text{low}[v],\ \min_{\substack{(u,v) \in E \\ v \text{ on stack}}} \text{disc}[v]\Big)
$$

A vertex $u$ is the **root** of an SCC if $\text{low}[u] = \text{disc}[u]$. This means $u$ cannot reach any vertex discovered earlier than itself, so $u$ and all vertices above it on the stack form a maximal strongly connected set.

## Algorithm

1. Maintain a global timer, a stack, and arrays for $\text{disc}$, $\text{low}$, and whether each vertex is on the stack.
2. For each unvisited vertex $u$, run DFS:
    - Set $\text{disc}[u] = \text{low}[u] = \text{timer}$; increment timer.
    - Push $u$ onto the stack.
    - For each neighbor $v$ of $u$:
        - If $v$ is unvisited, recurse on $v$ and set $\text{low}[u] = \min(\text{low}[u], \text{low}[v])$.
        - If $v$ is on the stack, set $\text{low}[u] = \min(\text{low}[u], \text{disc}[v])$.
    - After processing all neighbors, if $\text{low}[u] = \text{disc}[u]$, pop vertices from the stack until $u$ is popped. These vertices form one SCC.

## Correctness

!!! note "Why Root Detection Works"
    A vertex $u$ with $\text{low}[u] = \text{disc}[u]$ is the first-discovered vertex in its SCC. All other vertices $v$ in the same SCC have $\text{low}[v] < \text{disc}[v]$ because they can reach $u$ (or an earlier vertex) through back edges. When the DFS backtracks to $u$ and finds $\text{low}[u] = \text{disc}[u]$, every vertex of $u$'s SCC is on the stack above $u$, so popping until $u$ extracts exactly one SCC.

**Key invariant:** At all times, the stack contains vertices whose SCC has not yet been fully identified. A vertex remains on the stack until its SCC root is found.

## Complexity

The single DFS pass visits each vertex and edge exactly once:

$$
T(V, E) = O(V + E)
$$

Space complexity is $O(V)$ for the stack, discovery times, and low-link values -- no transpose graph is needed.

## Implementation

```python
"""
Tarjan's algorithm for finding strongly connected components.

Uses a single DFS pass with low-link values to identify SCC roots
and extract components from a stack.
"""


# === Tarjan's SCC Algorithm ===
def tarjan_scc(graph, n):
    """
    Find all strongly connected components using Tarjan's algorithm.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list of a directed graph with vertices 0 to n-1.
    n : int
        Number of vertices.

    Returns
    -------
    list[list[int]]
        List of SCCs, each as a list of vertex labels.
    """
    disc = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    stack = []
    timer = [0]
    sccs = []

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

        # If u is a root of an SCC
        if low[u] == disc[u]:
            component = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                component.append(w)
                if w == u:
                    break
            sccs.append(component)

    for u in range(n):
        if disc[u] == -1:
            dfs(u)

    return sccs


# === Main ===
if __name__ == "__main__":
    graph = {
        0: [1], 1: [2, 3], 2: [0], 3: [4],
        4: [5], 5: [3], 6: [5, 7], 7: [],
    }
    sccs = tarjan_scc(graph, 8)
    print("Strongly connected components:")
    for i, scc in enumerate(sccs):
        print(f"  C{i+1} = {sorted(scc)}")
```

**Output:**
```
Strongly connected components:
  C1 = [3, 4, 5]
  C2 = [0, 1, 2]
  C3 = [7]
  C4 = [6]
```

Note that Tarjan's algorithm outputs SCCs in reverse topological order of the [condensation graph](condensation.md). The SCC containing vertices $\{3, 4, 5\}$ appears first because it is a sink SCC -- no edges leave it to other SCCs.

## Step-by-Step Trace

Starting DFS from vertex 0:

| Step | Vertex | disc | low | Stack | Action |
|---|---|---|---|---|---|
| 1 | 0 | 0 | 0 | [0] | Visit 0 |
| 2 | 1 | 1 | 1 | [0,1] | Visit 1 |
| 3 | 2 | 2 | 2 | [0,1,2] | Visit 2 |
| 4 | 2→0 | - | low[2]=0 | [0,1,2] | Back edge, update low |
| 5 | 1←2 | - | low[1]=0 | [0,1,2] | Backtrack, low[1]=min(1,0) |
| 6 | 3 | 3 | 3 | [0,1,2,3] | Visit 3 |
| 7 | 4 | 4 | 4 | [0,1,2,3,4] | Visit 4 |
| 8 | 5 | 5 | 5 | [0,1,2,3,4,5] | Visit 5 |
| 9 | 5→3 | - | low[5]=3 | [0,1,2,3,4,5] | Back edge |
| 10 | 4←5 | - | low[4]=3 | [0,1,2,3,4,5] | Backtrack |
| 11 | 3←4 | - | low[3]=3 | [0,1,2,3,4,5] | low[3]==disc[3], pop SCC: {3,4,5} |
| 12 | 1←3 | - | low[1]=0 | [0,1,2] | Backtrack |
| 13 | 0←1 | - | low[0]=0 | [0,1,2] | low[0]==disc[0], pop SCC: {0,1,2} |

## Reference

- Tarjan, R. E. (1972). Depth-first search and linear graph algorithms. *SIAM Journal on Computing*, 1(2), 146-160.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 20.
