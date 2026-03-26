# Ford-Fulkerson Method

The Ford-Fulkerson method is the foundational approach for computing maximum flow in a network. Rather than a single algorithm, it is a **method** (or framework): find an augmenting path in the residual graph, push flow along it, and repeat until no more augmenting paths exist. The specific strategy for finding augmenting paths is left unspecified. Different choices lead to different algorithms (such as Edmonds-Karp) with different performance guarantees.

## Residual Graph

Before describing the method, we need the concept of a residual graph, which captures the remaining capacity available for routing additional flow.

Given a flow network $G = (V, E)$ with capacity function $c$ and a flow $f$, the **residual capacity** of an edge $(u, v)$ is:

$$
c_f(u, v) = c(u, v) - f(u, v)
$$

The **residual graph** $G_f = (V, E_f)$ contains edges with positive residual capacity:

$$
E_f = \{(u, v) \in V \times V : c_f(u, v) > 0\}
$$

This includes both forward edges (where capacity remains) and backward edges (where flow can be canceled). If edge $(u, v)$ carries flow $f(u, v) > 0$, then the residual graph contains the reverse edge $(v, u)$ with residual capacity $f(u, v)$, representing the option to "undo" previously sent flow.

## Augmenting Paths

An **augmenting path** is a simple path from source $s$ to sink $t$ in the residual graph $G_f$. The **bottleneck capacity** (or residual capacity of the path) is the minimum residual capacity along the path:

$$
c_f(p) = \min_{(u,v) \in p} c_f(u, v)
$$

Augmenting along path $p$ means increasing the flow by $c_f(p)$ along each forward edge of $p$ and decreasing the flow by $c_f(p)$ along each backward edge.

## The Method

```text
FORD-FULKERSON(G, s, t):
    Initialize f(u,v) = 0 for all (u,v) in E
    while there exists an augmenting path p in G_f:
        c_f(p) = min { c_f(u,v) : (u,v) in p }
        for each edge (u,v) in p:
            if (u,v) is a forward edge:
                f(u,v) = f(u,v) + c_f(p)
            else:  // (u,v) is a backward edge
                f(v,u) = f(v,u) - c_f(p)
    return f
```

The method terminates when no augmenting path exists in the residual graph. At that point, the flow is maximum.

## Correctness

The correctness of Ford-Fulkerson follows from the **max-flow min-cut theorem**: the value of a maximum flow equals the capacity of a minimum cut.

**Theorem (Max-Flow Min-Cut).** The following are equivalent:

1. $f$ is a maximum flow in $G$.
2. The residual graph $G_f$ contains no augmenting path.
3. $|f| = c(S, T)$ for some cut $(S, T)$ of $G$.

*Proof sketch.* $(1 \Rightarrow 2)$: If an augmenting path existed, we could increase the flow, contradicting maximality. $(2 \Rightarrow 3)$: Define $S = \{v \in V : v \text{ is reachable from } s \text{ in } G_f\}$ and $T = V \setminus S$. Since $t \notin S$ (no augmenting path), $(S, T)$ is a cut. Every edge from $S$ to $T$ must be saturated, and every edge from $T$ to $S$ must carry zero flow, so $|f| = c(S, T)$. $(3 \Rightarrow 1)$: Since $|f| \leq c(S, T)$ for any cut (the weak duality bound), $|f| = c(S, T)$ implies $f$ is maximum. $\square$

## Complexity

With integer capacities, the Ford-Fulkerson method terminates in at most $|f^*|$ augmentations, where $f^*$ is the maximum flow value, since each augmentation increases the flow by at least 1. Each augmentation requires $O(E)$ time to find the path (e.g., via DFS) and update the flow.

**Total time:** $O(|f^*| \cdot E)$

This bound depends on the flow value, not just the graph size. For large capacities, this can be very slow.

!!! warning "Non-Termination with Irrational Capacities"
    With irrational edge capacities, the Ford-Fulkerson method may not terminate and may even converge to a value less than the maximum flow. This pathological behavior motivates the use of BFS (Edmonds-Karp) or other structured path-selection strategies.

## Worked Example

Consider a network with vertices $\{s, a, b, t\}$:

| Edge | Capacity |
|------|----------|
| $(s, a)$ | 10 |
| $(s, b)$ | 8 |
| $(a, b)$ | 5 |
| $(a, t)$ | 7 |
| $(b, t)$ | 10 |

**Iteration 1.** Find augmenting path $s \to a \to t$. Bottleneck: $\min(10, 7) = 7$. Push flow 7.

After iteration 1: $f(s,a) = 7$, $f(a,t) = 7$. Residual capacities: $(s,a): 3$, $(a,t): 0$ (saturated), $(a,s): 7$, $(t,a): 7$.

**Iteration 2.** Find path $s \to b \to t$. Bottleneck: $\min(8, 10) = 8$. Push flow 8.

After iteration 2: additionally $f(s,b) = 8$, $f(b,t) = 8$. Residual: $(s,b): 0$, $(b,t): 2$.

**Iteration 3.** Find path $s \to a \to b \to t$. Bottleneck: $\min(3, 5, 2) = 2$. Push flow 2.

After iteration 3: $f(s,a) = 9$, $f(a,b) = 2$, $f(b,t) = 10$.

**Iteration 4.** No augmenting path exists. Maximum flow = $7 + 8 + 2 = 17$.

## Python Implementation

```python
"""
Ford-Fulkerson method for maximum flow using DFS.

Repeatedly finds augmenting paths in the residual graph via DFS
and pushes flow along them until no path from source to sink exists.
"""

# === DFS to find augmenting path ===

def dfs(capacity: list, source: int, sink: int, visited: list,
        u: int, bottleneck: int) -> int:
    """
    Find augmenting path via DFS and return bottleneck capacity.

    Returns 0 if no augmenting path from u to sink exists.
    """
    if u == sink:
        return bottleneck
    visited[u] = True
    for v in range(len(capacity)):
        if not visited[v] and capacity[u][v] > 0:
            result = dfs(
                capacity, source, sink, visited,
                v, min(bottleneck, capacity[u][v])
            )
            if result > 0:
                capacity[u][v] -= result
                capacity[v][u] += result
                return result
    return 0


# === Main algorithm ===

def ford_fulkerson(n: int, edges: list, source: int, sink: int) -> int:
    """
    Compute maximum flow using Ford-Fulkerson with DFS.

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

    max_flow = 0
    while True:
        visited = [False] * n
        augment = dfs(capacity, source, sink, visited,
                      source, float('inf'))
        if augment == 0:
            break
        max_flow += augment

    return max_flow


# === Example ===

if __name__ == "__main__":
    # s=0, a=1, b=2, t=3
    edges = [
        (0, 1, 10),  # s -> a
        (0, 2, 8),   # s -> b
        (1, 2, 5),   # a -> b
        (1, 3, 7),   # a -> t
        (2, 3, 10),  # b -> t
    ]
    result = ford_fulkerson(4, edges, 0, 3)
    print(f"Maximum flow: {result}")
```

**Output:**

```
Maximum flow: 17
```

The algorithm finds the maximum flow of 17, matching the hand-traced example above. The $O(V^2)$ adjacency matrix representation used here is convenient for small dense graphs; for sparse graphs, an adjacency list with explicit edge objects is more space-efficient.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 26.
- Ford, L. R., & Fulkerson, D. R. (1956). Maximal flow through a network. *Canadian Journal of Mathematics*, 8, 399-404.
