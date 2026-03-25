# Directed and Undirected Graphs

Whether edges carry direction is the most fundamental structural choice when modeling a problem as a graph. Undirected graphs capture symmetric relationships like friendship or physical adjacency, while directed graphs (digraphs) model asymmetric ones like web hyperlinks, prerequisite chains, or one-way streets. This distinction affects every aspect of graph algorithms -- from how we define paths and connectivity to the data structures we choose for representation.

## Undirected Graphs

An **undirected graph** $G = (V, E)$ consists of a finite set $V$ of vertices and a set $E$ of edges, where each edge is an unordered pair $\{u, v\}$ with $u, v \in V$ and $u \neq v$ (for simple graphs). The edge $\{u, v\}$ connects $u$ and $v$ symmetrically: if $u$ is adjacent to $v$, then $v$ is adjacent to $u$.

$$
E \subseteq \binom{V}{2} = \{\{u, v\} : u, v \in V, \; u \neq v\}
$$

A simple undirected graph on $n$ vertices has at most $\binom{n}{2} = n(n-1)/2$ edges.

!!! example "Undirected Graph"
    A social network where "friendship" is mutual: if Alice is friends with Bob, then Bob is friends with Alice. The edge $\{Alice, Bob\}$ has no direction.

## Directed Graphs

A **directed graph** (or **digraph**) $G = (V, E)$ has edges that are ordered pairs $(u, v)$, representing a directed connection from $u$ to $v$. The vertex $u$ is the **tail** (or source) and $v$ is the **head** (or target) of the edge.

$$
E \subseteq V \times V = \{(u, v) : u, v \in V\}
$$

In a simple digraph (no self-loops), we require $u \neq v$. The maximum number of edges is $n(n-1)$ since each ordered pair can appear independently.

The edge $(u, v)$ does not imply the existence of $(v, u)$. When both $(u, v)$ and $(v, u)$ exist, they are distinct edges forming an **antiparallel pair**.

!!! example "Directed Graph"
    The World Wide Web is a digraph: page $u$ may link to page $v$ without $v$ linking back. Course prerequisites form another digraph: "Calculus I must precede Calculus II" is a directed relationship.

## Structural Comparison

| Property | Undirected | Directed |
|---|---|---|
| Edge notation | $\{u, v\}$ | $(u, v)$ |
| Symmetry | $\{u,v\} = \{v,u\}$ | $(u,v) \neq (v,u)$ in general |
| Max edges (simple) | $\binom{n}{2}$ | $n(n-1)$ |
| Degree | $\deg(v)$ | $\deg^+(v)$, $\deg^-(v)$ |
| Connectivity | Connected / disconnected | Strongly / weakly connected |
| Cycle detection | Union-Find or DFS | DFS with color states |

## Important Directed Graph Classes

### Directed Acyclic Graphs (DAGs)

A digraph with no directed cycle is called a **DAG**. DAGs model dependency structures, and every DAG admits a [topological ordering](../../ch17/topological/dag.md) of its vertices: a linear sequence where every edge $(u, v)$ has $u$ appearing before $v$.

### Tournaments

A **tournament** is a directed graph obtained by assigning a direction to every edge of a complete graph $K_n$. For every pair of distinct vertices $u, v$, exactly one of $(u, v)$ or $(v, u)$ exists. Tournaments model round-robin competitions.

### Underlying Undirected Graph

Every digraph $G$ has an **underlying undirected graph** $G'$ obtained by replacing each directed edge $(u, v)$ with the undirected edge $\{u, v\}$ and removing duplicates. A digraph is **weakly connected** if its underlying undirected graph is connected.

## Converting Between Representations

```python
"""
Conversion between directed and undirected graph representations.

Demonstrates how to build directed and undirected adjacency lists,
convert a digraph to its underlying undirected graph, and check
for antiparallel edges.
"""


# === Build Adjacency Lists ===

def build_undirected(n, edges):
    """Build undirected adjacency list from edge pairs."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def build_directed(n, edges):
    """Build directed adjacency list from edge pairs."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
    return adj


# === Underlying Undirected Graph ===

def to_undirected(adj, n):
    """Convert a directed graph to its underlying undirected graph."""
    edge_set = set()
    for u in range(n):
        for v in adj[u]:
            edge_set.add((min(u, v), max(u, v)))
    undirected = [[] for _ in range(n)]
    for u, v in edge_set:
        undirected[u].append(v)
        undirected[v].append(u)
    return undirected


# === Antiparallel Detection ===

def find_antiparallel(adj, n):
    """Find all antiparallel edge pairs in a directed graph."""
    edge_set = set()
    for u in range(n):
        for v in adj[u]:
            edge_set.add((u, v))
    pairs = []
    for u, v in edge_set:
        if (v, u) in edge_set and u < v:
            pairs.append((u, v))
    return pairs


# === Main ===

if __name__ == "__main__":
    # Directed graph
    directed_edges = [(0, 1), (1, 2), (2, 0), (1, 3)]
    adj_dir = build_directed(4, directed_edges)
    print("Directed adjacency list:")
    for v in range(4):
        print(f"  {v} -> {adj_dir[v]}")

    # Convert to undirected
    adj_undir = to_undirected(adj_dir, 4)
    print("\nUnderlying undirected graph:")
    for v in range(4):
        print(f"  {v} -- {adj_undir[v]}")

    # Check antiparallel edges
    # Add reverse edge (1,0) to create antiparallel pair
    directed_edges2 = [(0, 1), (1, 0), (1, 2), (2, 0)]
    adj_dir2 = build_directed(3, directed_edges2)
    pairs = find_antiparallel(adj_dir2, 3)
    print(f"\nAntiparallel pairs: {pairs}")
```

**Output:**
```
Directed adjacency list:
  0 -> [1]
  1 -> [2, 3]
  2 -> [0]
  3 -> []
Underlying undirected graph:
  0 -- [1, 2]
  1 -- [0, 2, 3]
  2 -- [0, 1]
  3 -- [1]
Antiparallel pairs: [(0, 1)]
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 22.
- Diestel, R. (2017). *Graph Theory* (5th ed.). Springer. Chapters 1-2.
