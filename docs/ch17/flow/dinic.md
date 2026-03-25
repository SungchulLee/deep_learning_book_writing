# Dinic's Algorithm

While the Edmonds-Karp algorithm improves Ford-Fulkerson by always choosing shortest augmenting paths, it still sends flow along only one path per iteration. Dinic's algorithm (also spelled Dinitz) pushes this idea further: it builds a **level graph** capturing all shortest paths from source to sink, then sends as much flow as possible through that structure in a single phase. This yields a worst-case complexity of $O(V^2 E)$, and the algorithm performs especially well on unit-capacity networks where it runs in $O(E\sqrt{V})$ time.

## Level Graph

A **level graph** (or **layered graph**) organizes the residual graph into layers by distance from the source.

Given a flow network $G = (V, E)$ with source $s$ and sink $t$, and the current residual graph $G_f$, define the **level** of each vertex $v$ as $\text{level}(v) = d(s, v)$, the shortest-path distance (in number of edges) from $s$ to $v$ in $G_f$.

The level graph $G_L = (V_L, E_L)$ contains only edges $(u, v)$ from $G_f$ that satisfy $\text{level}(v) = \text{level}(u) + 1$. In other words, $G_L$ keeps only those residual edges that move strictly one level closer to the sink.

**Construction via BFS.** Run BFS from $s$ in $G_f$, recording the level of each reachable vertex. Then filter edges: keep $(u, v) \in G_f$ only if $\text{level}(v) = \text{level}(u) + 1$ and $v$ is reachable. This takes $O(V + E)$ time.

If $t$ is not reachable in the BFS, no augmenting path exists and the algorithm terminates. The current flow is maximum.

## Blocking Flow

A **blocking flow** in the level graph $G_L$ is a flow $f'$ such that every path from $s$ to $t$ in $G_L$ contains at least one saturated edge (an edge whose flow equals its residual capacity). In other words, after adding a blocking flow, no $s$-$t$ path remains in $G_L$.

A blocking flow is not necessarily a maximum flow in $G_L$ — it merely blocks all $s$-$t$ paths. Finding a blocking flow is cheaper than finding a maximum flow, and this is the key insight that makes Dinic's algorithm efficient.

**Finding a blocking flow via DFS.** Use depth-first search from $s$ in $G_L$. When a path to $t$ is found, push the minimum residual capacity along that path (saturating at least one edge). Remove saturated edges. If the DFS from a vertex reaches a dead end (no outgoing edges in $G_L$), delete the vertex and backtrack. Continue until no path from $s$ to $t$ exists.

With careful pointer management (advancing pointers rather than restarting), the total work for finding a blocking flow is $O(VE)$.

## Algorithm

Dinic's algorithm repeats two steps — building a level graph and finding a blocking flow — until no augmenting path exists.

```
DINIC(G, s, t):
    Initialize flow f = 0
    while True:
        Build level graph G_L via BFS from s in G_f
        if t is not reachable:
            return f
        Find a blocking flow f' in G_L
        f = f + f'
        Update residual graph G_f
```

Each iteration of the while loop is called a **phase**. In each phase, the distance from $s$ to $t$ in the residual graph strictly increases by at least 1. Since this distance is bounded by $|V| - 1$, there are at most $|V| - 1$ phases.

## Complexity Analysis

**Lemma (distance increase).** Let $d_f(s, t)$ denote the shortest-path distance from $s$ to $t$ in $G_f$. After adding a blocking flow to the level graph, $d_{f'}(s, t) > d_f(s, t)$.

*Proof sketch.* The blocking flow saturates at least one edge on every shortest path. When the residual graph is updated, any new augmenting path must use at least one edge that goes "backward" (from a higher level to a lower level). Such backward edges force the new shortest path to be strictly longer than the previous one. $\square$

**Theorem.** Dinic's algorithm runs in $O(V^2 E)$ time.

*Proof.* There are at most $O(V)$ phases (since the distance increases by at least 1 each phase and is bounded by $|V| - 1$). Each phase consists of:

- Building the level graph via BFS: $O(E)$
- Finding a blocking flow: $O(VE)$

The total time is $O(V) \cdot O(VE) = O(V^2 E)$. $\square$

## Unit-Capacity Networks

On networks where every edge has capacity 1, Dinic's algorithm achieves $O(E\sqrt{V})$ time.

The key observation is that in a unit-capacity network, each blocking flow phase takes $O(E)$ time (each augmenting path saturates an edge, removing it, so the total work across all paths in one phase is $O(E)$). Furthermore, after $\sqrt{V}$ phases, the maximum remaining augmentable flow is at most $\sqrt{V}$ (by a counting argument on the level structure). The remaining flow can be found in at most $\sqrt{V}$ additional phases, each costing $O(E)$.

Total: $O(\sqrt{V}) \cdot O(E) = O(E\sqrt{V})$.

This makes Dinic's algorithm the method of choice for bipartite matching, where the underlying network is unit-capacity and $O(E\sqrt{V})$ matches the best known bound for maximum bipartite matching.

## Worked Example

Consider a network with vertices $\{s, a, b, c, t\}$ and edges:

| Edge | Capacity |
|------|----------|
| $(s, a)$ | 10 |
| $(s, b)$ | 10 |
| $(a, b)$ | 2 |
| $(a, c)$ | 8 |
| $(b, c)$ | 6 |
| $(c, t)$ | 14 |
| $(a, t)$ | 4 |

**Phase 1.** BFS from $s$ gives levels: $\text{level}(s) = 0$, $\text{level}(a) = 1$, $\text{level}(b) = 1$, $\text{level}(c) = 2$, $\text{level}(t) = 2$. The level graph keeps edges $(s,a)$, $(s,b)$, $(a,c)$, $(a,t)$, $(b,c)$, $(c,t)$ (edge $(a,b)$ is excluded since both are at level 1).

The blocking flow finds paths and pushes flow until all $s$-$t$ paths in $G_L$ are blocked. For instance:

- Path $s \to a \to t$: push 4 (saturates $(a,t)$)
- Path $s \to a \to c \to t$: push 6 (saturates remaining capacity on $(a,c)$ after considering available flow)
- Path $s \to b \to c \to t$: push 4 (limited by remaining capacity on $(c,t)$)

Total flow after Phase 1: 14.

**Phase 2.** Rebuild the level graph on the updated residual graph. The distance from $s$ to $t$ has increased. Continue until BFS cannot reach $t$, at which point the flow is maximum.

## Python Implementation

```python
"""
Dinic's algorithm for maximum flow.

Builds level graphs via BFS and finds blocking flows via DFS
to compute max flow in O(V^2 E) time.
"""

from collections import deque

# === Edge representation ===

class Edge:
    """Directed edge with capacity and flow tracking."""

    __slots__ = ['to', 'cap', 'rev']

    def __init__(self, to: int, cap: int, rev: int):
        self.to = to
        self.cap = cap
        self.rev = rev  # index of reverse edge in graph[to]


# === Graph construction ===

def add_edge(graph: list, u: int, v: int, cap: int) -> None:
    """Add edge u -> v with given capacity, and reverse edge with 0 capacity."""
    graph[u].append(Edge(v, cap, len(graph[v])))
    graph[v].append(Edge(u, 0, len(graph[u]) - 1))


# === BFS for level graph ===

def bfs(graph: list, s: int, t: int, level: list) -> bool:
    """Build level graph via BFS. Return True if t is reachable."""
    for i in range(len(level)):
        level[i] = -1
    level[s] = 0
    queue = deque([s])
    while queue:
        u = queue.popleft()
        for e in graph[u]:
            if e.cap > 0 and level[e.to] < 0:
                level[e.to] = level[u] + 1
                queue.append(e.to)
    return level[t] >= 0


# === DFS for blocking flow ===

def dfs(graph: list, u: int, t: int, f: int,
        level: list, iter_: list) -> int:
    """Find blocking flow using DFS with pointer advancement."""
    if u == t:
        return f
    while iter_[u] < len(graph[u]):
        e = graph[u][iter_[u]]
        if e.cap > 0 and level[e.to] == level[u] + 1:
            d = dfs(graph, e.to, t, min(f, e.cap), level, iter_)
            if d > 0:
                e.cap -= d
                graph[e.to][e.rev].cap += d
                return d
        iter_[u] += 1
    return 0


# === Main algorithm ===

def dinic(n: int, edges: list, s: int, t: int) -> int:
    """
    Compute maximum flow using Dinic's algorithm.

    Parameters
    ----------
    n : int
        Number of vertices (labeled 0 to n-1).
    edges : list of (u, v, cap) tuples
        Directed edges with capacities.
    s : int
        Source vertex.
    t : int
        Sink vertex.

    Returns
    -------
    int
        Maximum flow value from s to t.
    """
    graph = [[] for _ in range(n)]
    for u, v, cap in edges:
        add_edge(graph, u, v, cap)

    level = [0] * n
    flow = 0

    while bfs(graph, s, t, level):
        iter_ = [0] * n
        while True:
            f = dfs(graph, s, t, float('inf'), level, iter_)
            if f == 0:
                break
            flow += f

    return flow


# === Example ===

if __name__ == "__main__":
    # s=0, a=1, b=2, c=3, t=4
    edges = [
        (0, 1, 10),  # s -> a
        (0, 2, 10),  # s -> b
        (1, 2, 2),   # a -> b
        (1, 3, 8),   # a -> c
        (2, 3, 6),   # b -> c
        (3, 4, 14),  # c -> t
        (1, 4, 4),   # a -> t
    ]
    result = dinic(5, edges, 0, 4)
    print(f"Maximum flow: {result}")
```

## Comparison with Edmonds-Karp

| Property | Edmonds-Karp | Dinic's |
|----------|-------------|---------|
| Path strategy | Single shortest path | All shortest paths (blocking flow) |
| Time complexity | $O(VE^2)$ | $O(V^2 E)$ |
| Unit-capacity networks | $O(E \cdot \min(E^{1/2}, V^{2/3}))$ | $O(E\sqrt{V})$ |
| Implementation | Simpler (BFS only) | More involved (BFS + DFS) |

Dinic's algorithm dominates Edmonds-Karp when $V < E$, which is the common case in dense networks. For sparse graphs where $E = O(V)$, both yield $O(V^3)$.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapters 24, 26.
- Dinitz, Y. (1970). Algorithm for solution of a problem of maximum flow in networks with power estimation. *Doklady Akademii Nauk SSSR*, 194(4).
