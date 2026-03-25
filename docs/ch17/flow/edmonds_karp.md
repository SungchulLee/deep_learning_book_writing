# Edmonds-Karp Algorithm

The Ford-Fulkerson method leaves the choice of augmenting path unspecified, which can lead to poor performance or even non-termination with irrational capacities. The Edmonds-Karp algorithm resolves this by always choosing the **shortest augmenting path** (fewest edges) via BFS. This simple rule guarantees termination in $O(VE)$ augmentations and yields an overall $O(VE^2)$ time complexity.

## From Ford-Fulkerson to Edmonds-Karp

Recall that Ford-Fulkerson repeatedly finds an augmenting path in the residual graph and pushes flow along it. The method is correct for any path selection strategy, but the number of augmentations depends on which paths are chosen.

Edmonds-Karp specializes Ford-Fulkerson by using **BFS** (breadth-first search) from the source $s$ to find augmenting paths. Since BFS explores vertices in order of increasing distance, the path found always has the minimum number of edges among all $s$-$t$ paths in the residual graph.

## Algorithm

```
EDMONDS-KARP(G, s, t):
    Initialize flow f(u,v) = 0 for all (u,v)
    while BFS finds a path P from s to t in G_f:
        c_f(P) = min { c_f(u,v) : (u,v) in P }
        for each edge (u,v) in P:
            f(u,v) = f(u,v) + c_f(P)
            f(v,u) = f(v,u) - c_f(P)
    return f
```

Each BFS takes $O(V + E)$ time. The key question is: how many augmentations can occur?

## Monotonicity of Shortest-Path Distances

The crucial property underlying the $O(VE^2)$ bound is that shortest-path distances in the residual graph never decrease.

**Lemma.** Let $\delta_f(s, v)$ denote the shortest-path distance from $s$ to $v$ in residual graph $G_f$. After augmenting along a shortest path, for every vertex $v \in V$:

$$
\delta_{f'}(s, v) \geq \delta_f(s, v)
$$

where $f'$ is the flow after augmentation.

*Proof sketch.* Suppose for contradiction that $\delta_{f'}(s, v) < \delta_f(s, v)$ for some vertex $v$. Pick $v$ with the smallest $\delta_{f'}(s, v)$ among all such vertices. Let $u$ be the predecessor of $v$ on a shortest path from $s$ to $v$ in $G_{f'}$. Then $\delta_{f'}(s, v) = \delta_{f'}(s, u) + 1$.

By our choice of $v$, we have $\delta_{f'}(s, u) \geq \delta_f(s, u)$. Edge $(u, v)$ must exist in $G_{f'}$. If $(u, v)$ also exists in $G_f$, then $\delta_f(s, v) \leq \delta_f(s, u) + 1 \leq \delta_{f'}(s, u) + 1 = \delta_{f'}(s, v)$, contradicting our assumption.

If $(u, v)$ does not exist in $G_f$, it was created by augmenting along $(v, u)$. Since augmentation uses a shortest path, $(v, u)$ lies on a shortest $s$-$t$ path in $G_f$, so $\delta_f(s, u) = \delta_f(s, v) + 1$. Then $\delta_{f'}(s, v) = \delta_{f'}(s, u) + 1 \geq \delta_f(s, u) + 1 = \delta_f(s, v) + 2$, again contradicting $\delta_{f'}(s, v) < \delta_f(s, v)$. $\square$

## Bound on Augmentations

**Theorem.** The Edmonds-Karp algorithm performs at most $O(VE)$ augmentations.

*Proof sketch.* Call an edge $(u, v)$ in the residual graph **critical** on an augmenting path if it has the minimum residual capacity along that path (i.e., it becomes saturated). Each augmentation saturates at least one edge.

When edge $(u, v)$ is critical, $\delta_f(s, v) = \delta_f(s, u) + 1$. For $(u, v)$ to appear again in the residual graph, flow must be pushed along $(v, u)$ in some later augmentation. At that point, $\delta_{f'}(s, u) = \delta_{f'}(s, v) + 1$. By the monotonicity lemma:

$$
\delta_{f'}(s, u) = \delta_{f'}(s, v) + 1 \geq \delta_f(s, v) + 1 = \delta_f(s, u) + 2
$$

So $\delta_{f'}(s, u)$ increases by at least 2 between consecutive times $(u, v)$ is critical. Since distances are bounded by $|V| - 1$, each edge can be critical at most $O(V)$ times. With $O(E)$ edges, there are at most $O(VE)$ total augmentations. $\square$

## Overall Complexity

**Theorem.** The Edmonds-Karp algorithm runs in $O(VE^2)$ time.

*Proof.* Each augmentation requires a BFS taking $O(E)$ time (since $E \geq V - 1$ in a connected graph). With $O(VE)$ augmentations, the total is $O(VE) \cdot O(E) = O(VE^2)$. $\square$

## Worked Example

Consider a network with vertices $\{s, a, b, t\}$:

| Edge | Capacity |
|------|----------|
| $(s, a)$ | 4 |
| $(s, b)$ | 3 |
| $(a, b)$ | 2 |
| $(a, t)$ | 3 |
| $(b, t)$ | 5 |

**Iteration 1.** BFS from $s$ finds shortest path $s \to a \to t$ (2 edges). Bottleneck capacity: $\min(4, 3) = 3$. Push flow 3.

Residual graph updates: $(s, a)$ has residual 1, $(a, t)$ is saturated, reverse edges $(a, s)$ and $(t, a)$ appear with capacity 3.

**Iteration 2.** BFS finds $s \to b \to t$ (2 edges). Bottleneck: $\min(3, 5) = 3$. Push flow 3.

**Iteration 3.** BFS finds $s \to a \to b \to t$ (3 edges). Bottleneck: $\min(1, 2, 2) = 1$. Push flow 1.

**Iteration 4.** BFS cannot reach $t$ from $s$. Algorithm terminates with maximum flow = 7.

Notice that the shortest-path distance from $s$ to $t$ was 2 in iterations 1 and 2, then increased to 3 in iteration 3, confirming the monotonicity property.

## Python Implementation

```python
"""
Edmonds-Karp algorithm for maximum flow.

Uses BFS to find shortest augmenting paths, guaranteeing
O(VE^2) time complexity.
"""

from collections import deque

# === BFS to find augmenting path ===

def bfs(capacity: list, source: int, sink: int, parent: list) -> int:
    """
    Find shortest augmenting path via BFS.

    Returns the bottleneck capacity of the path, or 0 if no path exists.
    """
    n = len(capacity)
    visited = [False] * n
    visited[source] = True
    queue = deque([(source, float('inf'))])

    while queue:
        u, flow = queue.popleft()
        for v in range(n):
            if not visited[v] and capacity[u][v] > 0:
                visited[v] = True
                parent[v] = u
                new_flow = min(flow, capacity[u][v])
                if v == sink:
                    return new_flow
                queue.append((v, new_flow))
    return 0


# === Main algorithm ===

def edmonds_karp(n: int, edges: list, source: int, sink: int) -> int:
    """
    Compute maximum flow using Edmonds-Karp.

    Parameters
    ----------
    n : int
        Number of vertices (labeled 0 to n-1).
    edges : list of (u, v, cap) tuples
        Directed edges with capacities.
    source : int
        Source vertex.
    sink : int
        Sink vertex.

    Returns
    -------
    int
        Maximum flow value.
    """
    capacity = [[0] * n for _ in range(n)]
    for u, v, cap in edges:
        capacity[u][v] += cap

    parent = [-1] * n
    max_flow = 0

    while True:
        augment = bfs(capacity, source, sink, parent)
        if augment == 0:
            break
        max_flow += augment
        v = sink
        while v != source:
            u = parent[v]
            capacity[u][v] -= augment
            capacity[v][u] += augment
            v = u

    return max_flow


# === Example ===

if __name__ == "__main__":
    # s=0, a=1, b=2, t=3
    edges = [
        (0, 1, 4),  # s -> a
        (0, 2, 3),  # s -> b
        (1, 2, 2),  # a -> b
        (1, 3, 3),  # a -> t
        (2, 3, 5),  # b -> t
    ]
    result = edmonds_karp(4, edges, 0, 3)
    print(f"Maximum flow: {result}")  # Expected: 7
```

!!! tip "When to Use Edmonds-Karp"
    Edmonds-Karp is a good default choice when you need a straightforward max-flow implementation. Its $O(VE^2)$ bound is polynomial and the code is simple (just BFS). For better performance on dense graphs, consider Dinic's algorithm with its $O(V^2E)$ bound.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 26.
- Edmonds, J., & Karp, R. M. (1972). Theoretical improvements in algorithmic efficiency for network flow problems. *Journal of the ACM*, 19(2), 248-264.
