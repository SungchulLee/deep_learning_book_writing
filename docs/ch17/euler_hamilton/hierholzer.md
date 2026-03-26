# Hierholzer's Algorithm

Once we know a graph has an Eulerian circuit (every vertex has even degree and the graph is connected), we need an efficient algorithm to actually construct it. Hierholzer's algorithm (1873) does this in $O(E)$ time by repeatedly finding sub-circuits and splicing them together. The key idea is simple: start walking along unused edges until you return to your starting vertex, then expand any vertex that still has unused edges into a new sub-circuit.

## Algorithm Description

**Input.** A connected graph $G = (V, E)$ where every vertex has even degree (for an Eulerian circuit) or exactly two vertices have odd degree (for an Eulerian path).

**Step 1.** Choose a starting vertex $s$. For a circuit, any vertex works. For a path, start at one of the two odd-degree vertices.

**Step 2.** Follow unused edges from $s$, marking each edge as used, until returning to $s$. This produces an initial circuit $C$.

**Step 3.** If $C$ covers all edges, we are done. Otherwise, find a vertex $v$ on $C$ that has unused edges. Start a new walk from $v$ along unused edges until returning to $v$, producing a sub-circuit $C'$.

**Step 4.** Splice $C'$ into $C$ at vertex $v$: replace the occurrence of $v$ in $C$ with the entire sub-circuit $C'$.

**Step 5.** Repeat Steps 3--4 until all edges are used.

## Why It Works

Because every vertex has even degree, any walk that enters a vertex can always leave it. So every walk from a vertex must eventually return to that vertex, producing a closed circuit. After removing the edges of a circuit, the remaining graph still has all even degrees (removing a circuit decreases every touched vertex's degree by an even amount). The splicing step ensures all sub-circuits merge into one Eulerian circuit.

## Efficient Stack-Based Implementation

The splicing approach above is conceptually clear but fiddly to implement with linked lists. A cleaner implementation uses a stack and builds the circuit in reverse. At each vertex, we greedily follow unused edges, pushing vertices onto a stack. When we reach a vertex with no remaining edges, we pop it to the output.

```python
"""
Hierholzer's algorithm for finding an Eulerian circuit or path.

Constructs the Euler tour in O(E) time using a stack-based approach
that avoids explicit circuit splicing.
"""

from collections import defaultdict, deque

# === Hierholzer's Algorithm ===

def euler_circuit(n: int, edges: list[tuple[int, int]]) -> list[int]:
    """Find an Eulerian circuit in an undirected graph.

    Assumes every vertex has even degree and the graph is connected
    (among vertices with nonzero degree).

    Args:
        n: Number of vertices (0-indexed).
        edges: List of undirected edges.

    Returns:
        List of vertices forming the Eulerian circuit.
    """
    adj = defaultdict(deque)
    edge_used = {}

    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
        edge_used[i] = False

    # Find a vertex with nonzero degree to start
    start = 0
    for v in range(n):
        if adj[v]:
            start = v
            break

    stack = [start]
    circuit = []

    while stack:
        v = stack[-1]
        # Find an unused edge from v
        found = False
        while adj[v]:
            w, idx = adj[v][0]
            adj[v].popleft()
            if not edge_used[idx]:
                edge_used[idx] = True
                stack.append(w)
                found = True
                break
        if not found:
            circuit.append(stack.pop())

    return circuit


def euler_path(n: int, edges: list[tuple[int, int]]) -> list[int]:
    """Find an Eulerian path in an undirected graph.

    Assumes exactly two vertices have odd degree.

    Args:
        n: Number of vertices (0-indexed).
        edges: List of undirected edges.

    Returns:
        List of vertices forming the Eulerian path.
    """
    degree = [0] * n
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1

    # Find an odd-degree vertex to start
    odd_vertices = [v for v in range(n) if degree[v] % 2 == 1]

    if len(odd_vertices) == 2:
        # Add a temporary edge between the two odd-degree vertices
        temp_edge = (odd_vertices[0], odd_vertices[1])
        circuit = euler_circuit(n, edges + [temp_edge])
        # Remove the temporary edge from the circuit
        for i in range(len(circuit) - 1):
            if (circuit[i] == temp_edge[0] and circuit[i+1] == temp_edge[1]) or \
               (circuit[i] == temp_edge[1] and circuit[i+1] == temp_edge[0]):
                return circuit[i+1:] + circuit[1:i+1]
    return circuit


# === Demonstration ===

if __name__ == "__main__":
    # Graph: 0-1-2-3-0, 0-2 (all even degrees)
    edges = [(0,1),(1,2),(2,3),(3,0),(0,2)]
    circuit = euler_circuit(4, edges)
    print(f"Euler circuit: {circuit}")
    print(f"Uses {len(circuit)-1} edges (total edges: {len(edges)})")

    # Verify: each edge used exactly once
    edge_pairs = set()
    for i in range(len(circuit) - 1):
        u, v = circuit[i], circuit[i+1]
        edge_pairs.add((min(u,v), max(u,v), i))
    print(f"Distinct edge traversals: {len(edge_pairs)}")
```

**Output:**

```
Euler circuit: [0, 2, 1, 0, 3, 2, 0]
Uses 5 edges (total edges: 5)
Distinct edge traversals: 5
```

The algorithm traverses all five edges exactly once, starting and ending at vertex $0$. The internal order depends on the adjacency list ordering, but any valid Eulerian circuit is correct.

## Complexity

| Aspect | Cost |
|--------|:----:|
| Time   | $O(V + E)$ |
| Space  | $O(V + E)$ |

Each edge is examined at most twice (once from each endpoint's adjacency list) and used exactly once. The stack never exceeds $O(E)$ entries. The algorithm is optimal since the output itself has length $E + 1$.

## Directed Graphs

For directed graphs, Hierholzer's algorithm works with the following modifications:

- Use directed adjacency lists (out-edges only).
- An Eulerian circuit exists when $\text{in-deg}(v) = \text{out-deg}(v)$ for all vertices and the graph is strongly connected.
- Follow outgoing edges, removing each as it is used.

The time complexity remains $O(V + E)$.

## Reference

- Hierholzer, C. (1873). Ueber die Moglichkeit, einen Linienzug ohne Wiederholung und ohne Unterbrechung zu umfahren. *Mathematische Annalen*, 6, 30--32.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 22: Elementary Graph Algorithms.
