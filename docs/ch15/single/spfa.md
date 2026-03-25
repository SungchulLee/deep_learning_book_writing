# Shortest Path Faster Algorithm

The standard Bellman-Ford algorithm relaxes *every* edge in each of its
$|V| - 1$ passes, even edges whose source vertex has not changed since the
last pass.  The **Shortest Path Faster Algorithm** (SPFA) avoids this waste by
maintaining a queue of vertices whose distances have recently decreased.  Only
edges leaving these vertices need to be relaxed, which often reduces the
practical running time significantly while retaining the ability to handle
negative-weight edges.

## Key Idea

SPFA is a queue-based optimization of Bellman-Ford.  Instead of iterating over
all edges in each pass, it keeps a FIFO queue of "active" vertices — those
whose distance estimates have just improved.  When a vertex $u$ is dequeued,
its outgoing edges are relaxed.  If relaxing edge $(u, v)$ reduces $d[v]$ and
$v$ is not already in the queue, $v$ is enqueued for future processing.

This approach avoids relaxing edges from vertices whose estimates have not
changed, which is the main source of redundant work in standard Bellman-Ford.

## Algorithm

```
SPFA(G, w, s):
    INITIALIZE-SINGLE-SOURCE(G, s)
    queue = {s}
    in_queue = {s: TRUE, all others: FALSE}
    while queue is not empty:
        u = DEQUEUE(queue)
        in_queue[u] = FALSE
        for each edge (u, v) in Adj[u]:
            if d[u] + w(u,v) < d[v]:
                d[v] = d[u] + w(u,v)
                pred[v] = u
                if not in_queue[v]:
                    ENQUEUE(queue, v)
                    in_queue[v] = TRUE
```

The `in_queue` flag prevents duplicate entries, ensuring each vertex appears in
the queue at most once at any given time.

## Complexity

- **Worst case:** $O(VE)$, the same as Bellman-Ford.  Adversarial graphs can
  force each vertex to be enqueued $O(V)$ times.
- **Average case:** Empirically much faster — often close to $O(E)$ on random
  graphs.  However, no tighter worst-case bound is known.
- **Space:** $O(V)$ for the queue and auxiliary arrays.

!!! warning "Worst-case performance"
    Despite good average-case behavior, SPFA can degrade to $O(VE)$ on
    carefully constructed graphs.  For this reason, Dijkstra's algorithm
    (with non-negative weights) or standard Bellman-Ford (with guaranteed
    $O(VE)$) are often preferred in competitive settings where adversarial
    inputs are possible.

## Negative Cycle Detection

SPFA can detect negative cycles by counting how many times each vertex is
enqueued.  If a vertex is enqueued more than $|V| - 1$ times, it lies on or
is reachable from a negative cycle.

This works because in a graph without negative cycles, each vertex's distance
can decrease at most $|V| - 1$ times (once for each possible shortest-path
length).

## Comparison with Bellman-Ford

| Aspect | Bellman-Ford | SPFA |
|---|---|---|
| Edge processing | All edges, $\lvert V\rvert - 1$ times | Only edges from recently improved vertices |
| Worst-case time | $O(VE)$ | $O(VE)$ |
| Practical performance | Consistent | Often much faster, but variable |
| Implementation | Simpler | Slightly more complex (queue management) |
| Negative cycle detection | Extra pass after $\lvert V\rvert - 1$ | Count enqueue operations |

## Worked Example

Consider vertices $\{s, a, b, c, d\}$ with edges:

| Edge | Weight |
|---|---|
| $(s, a)$ | 1 |
| $(s, b)$ | 4 |
| $(a, b)$ | 2 |
| $(a, c)$ | 6 |
| $(b, c)$ | 3 |
| $(b, d)$ | 1 |
| $(c, d)$ | -2 |

**Step 1:** Initialize $d[s]=0$, all others $\infty$.  Enqueue $s$.

**Step 2:** Dequeue $s$.  Relax $(s,a)$: $d[a]=1$, enqueue $a$.
Relax $(s,b)$: $d[b]=4$, enqueue $b$.  Queue: $[a, b]$.

**Step 3:** Dequeue $a$.  Relax $(a,b)$: $d[b]=\min(4, 1+2)=3$, $b$ already
in queue.  Relax $(a,c)$: $d[c]=7$, enqueue $c$.  Queue: $[b, c]$.

**Step 4:** Dequeue $b$.  Relax $(b,c)$: $d[c]=\min(7, 3+3)=6$, $c$ already
in queue.  Relax $(b,d)$: $d[d]=4$, enqueue $d$.  Queue: $[c, d]$.

**Step 5:** Dequeue $c$.  Relax $(c,d)$: $d[d]=\min(4, 6-2)=4$ (no change).
Queue: $[d]$.

**Step 6:** Dequeue $d$.  No outgoing edges.  Queue empty.

**Final distances:** $d[s]=0, d[a]=1, d[b]=3, d[c]=6, d[d]=4$.

## Implementation

```python
"""
Shortest Path Faster Algorithm (SPFA).

A queue-based optimization of Bellman-Ford that avoids redundant
edge relaxations by processing only recently improved vertices.
"""

from collections import deque
from math import inf


# === SPFA algorithm ==========================================================

def spfa(graph: dict, source) -> tuple[dict, dict, bool]:
    """Run SPFA from the given source vertex.

    Parameters
    ----------
    graph : dict
        Adjacency list mapping vertex -> list of (neighbor, weight).
    source : hashable
        The source vertex.

    Returns
    -------
    dist : dict
        Shortest distances from source.
    pred : dict
        Predecessor pointers for path reconstruction.
    no_negative_cycle : bool
        True if no negative cycle is reachable from source.
    """
    n = len(graph)
    dist = {v: inf for v in graph}
    dist[source] = 0
    pred = {v: None for v in graph}
    in_queue = {v: False for v in graph}
    count = {v: 0 for v in graph}  # enqueue count for cycle detection

    queue = deque([source])
    in_queue[source] = True
    count[source] = 1

    while queue:
        u = queue.popleft()
        in_queue[u] = False

        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                pred[v] = u
                if not in_queue[v]:
                    queue.append(v)
                    in_queue[v] = True
                    count[v] += 1
                    if count[v] >= n:
                        return dist, pred, False  # negative cycle

    return dist, pred, True


# === Path reconstruction =====================================================

def get_path(pred: dict, source, target) -> list:
    """Reconstruct the shortest path from source to target."""
    path = []
    v = target
    while v is not None:
        path.append(v)
        v = pred[v]
    path.reverse()
    return path if path and path[0] == source else []


# === Demo ====================================================================

if __name__ == "__main__":
    graph = {
        "s": [("a", 1), ("b", 4)],
        "a": [("b", 2), ("c", 6)],
        "b": [("c", 3), ("d", 1)],
        "c": [("d", -2)],
        "d": [],
    }

    dist, pred, ok = spfa(graph, "s")
    print(f"No negative cycle: {ok}")
    print(f"Distances: {dist}")
    print(f"Path s->d: {get_path(pred, 's', 'd')}")
    print(f"Path s->c: {get_path(pred, 's', 'c')}")

    # Graph with negative cycle
    print("\n--- Graph with negative cycle ---")
    graph_neg = {
        "s": [("a", 1)],
        "a": [("b", -3)],
        "b": [("c", 1)],
        "c": [("a", -1)],  # cycle: a->b->c->a = -3+1+(-1) = -3
    }
    dist2, pred2, ok2 = spfa(graph_neg, "s")
    print(f"No negative cycle: {ok2}")
```

**Output:**

```
No negative cycle: True
Distances: {'s': 0, 'a': 1, 'b': 3, 'c': 6, 'd': 4}
Path s->d: ['s', 'a', 'b', 'd']
Path s->c: ['s', 'a', 'b', 'c']

--- Graph with negative cycle ---
No negative cycle: False
```

## Reference

- Duan, F. (1994). About the Shortest Path Faster Algorithm. *Journal of
  Southwest Jiaotong University*, 29(2), 207-212.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to
  Algorithms* (4th ed.), Chapter 24: Single-Source Shortest Paths.
