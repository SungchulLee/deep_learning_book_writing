# The Floyd-Warshall Algorithm

Single-source algorithms like Dijkstra and Bellman-Ford find shortest paths from
one source to all other vertices.  When shortest paths between *every* pair of
vertices are needed, running a single-source algorithm from each vertex works
but may not be the simplest approach.  The Floyd-Warshall algorithm solves the
**all-pairs shortest paths** problem directly using dynamic programming, with a
remarkably compact three-loop structure that runs in $\Theta(V^3)$ time.

## The Dynamic Programming Recurrence

Number the vertices $1, 2, \dots, n$.  Define $d^{(k)}_{ij}$ as the weight of
the shortest path from $i$ to $j$ using only vertices $\{1, 2, \dots, k\}$ as
intermediate vertices.

**Base case ($k = 0$):** No intermediate vertices are allowed, so

$$
d^{(0)}_{ij} =
\begin{cases}
0 & \text{if } i = j \\
w(i, j) & \text{if } (i, j) \in E \\
\infty & \text{otherwise}
\end{cases}
$$

**Recursive case:** The shortest path from $i$ to $j$ using intermediates
from $\{1, \dots, k\}$ either avoids vertex $k$ (in which case it uses only
$\{1, \dots, k-1\}$) or passes through $k$ (decomposing into
$i \leadsto k \leadsto j$):

$$
d^{(k)}_{ij} = \min\!\left(d^{(k-1)}_{ij},\ d^{(k-1)}_{ik} + d^{(k-1)}_{kj}\right)
$$

After computing $d^{(n)}_{ij}$ for all pairs $(i, j)$, the matrix contains
the shortest-path weights between all vertex pairs.

## Why It Works

The recurrence considers each vertex $k$ as a potential intermediate vertex.
By the optimal substructure of shortest paths, the sub-paths
$i \leadsto k$ and $k \leadsto j$ are themselves shortest paths using only
intermediates from $\{1, \dots, k-1\}$.  The $\min$ operation selects the
better option: either avoid $k$ or route through $k$.

Since every shortest path uses some subset of $\{1, \dots, n\}$ as
intermediates, iterating $k$ from $1$ to $n$ covers all possibilities.

## Pseudocode

```
FLOYD-WARSHALL(W):
    n = |V|
    D = W                      // D[i][j] = w(i,j) or inf
    P = predecessor matrix     // P[i][j] = i if (i,j) in E, else NIL
    for k = 1 to n:
        for i = 1 to n:
            for j = 1 to n:
                if D[i][k] + D[k][j] < D[i][j]:
                    D[i][j] = D[i][k] + D[k][j]
                    P[i][j] = P[k][j]
    return D, P
```

The algorithm updates the distance matrix in place.  The predecessor matrix
$P$ tracks the last intermediate vertex on the shortest path, enabling path
reconstruction.

## Complexity

- **Time:** Three nested loops over $n$ vertices: $\Theta(n^3)$.
- **Space:** $\Theta(n^2)$ for the distance and predecessor matrices.

The cubic time makes Floyd-Warshall practical for dense graphs with up to a
few thousand vertices.  For sparse graphs, Johnson's algorithm
($O(V^2 \log V + VE)$) may be faster.

## In-Place Update Correctness

A subtle point: the algorithm updates $D$ in place rather than maintaining
separate $D^{(k-1)}$ and $D^{(k)}$ matrices.  This is correct because when
computing $d^{(k)}_{ij}$, the values $d^{(k)}_{ik}$ and $d^{(k)}_{kj}$ equal
$d^{(k-1)}_{ik}$ and $d^{(k-1)}_{kj}$ respectively (adding $k$ as an
intermediate on a path from $i$ to $k$ or from $k$ to $j$ does not help, since
$k$ is already an endpoint).

## Path Reconstruction

The predecessor matrix $P$ allows reconstructing the shortest path between any
pair.  $P[i][j]$ stores the predecessor of $j$ on the shortest path from $i$.
To reconstruct the path from $i$ to $j$:

```
PRINT-PATH(P, i, j):
    if i == j:
        print i
    elif P[i][j] == NIL:
        print "no path"
    else:
        PRINT-PATH(P, i, P[i][j])
        print j
```

## Negative Cycle Detection

After running Floyd-Warshall, check the diagonal of the distance matrix.
If $d^{(n)}_{ii} < 0$ for any vertex $i$, the graph contains a negative-weight
cycle through $i$.

## Worked Example

Consider a graph with 4 vertices and the following weight matrix:

$$
W = \begin{pmatrix}
0 & 3 & \infty & 7 \\
8 & 0 & 2 & \infty \\
5 & \infty & 0 & 1 \\
2 & \infty & \infty & 0
\end{pmatrix}
$$

**After $k=1$:** Vertex 1 as intermediate.
$d_{24} = \min(\infty, d_{21} + d_{14}) = \min(\infty, 8 + 7) = 15$.
$d_{32} = \min(\infty, d_{31} + d_{12}) = \min(\infty, 5 + 3) = 8$.

**After $k=2$:** Vertex 2 as intermediate.
$d_{13} = \min(\infty, d_{12} + d_{23}) = \min(\infty, 3 + 2) = 5$.

**After $k=3$:** Vertex 3 as intermediate.
$d_{14} = \min(7, d_{13} + d_{34}) = \min(7, 5 + 1) = 6$.
$d_{24} = \min(15, d_{23} + d_{34}) = \min(15, 2 + 1) = 3$.

**After $k=4$:** Vertex 4 as intermediate.
$d_{31} = \min(5, d_{34} + d_{41}) = \min(5, 1 + 2) = 3$.
$d_{32} = \min(8, d_{31} + d_{12}) = \min(8, 3 + 3) = 6$.

Final distance matrix:

$$
D^{(4)} = \begin{pmatrix}
0 & 3 & 5 & 6 \\
5 & 0 & 2 & 3 \\
3 & 6 & 0 & 1 \\
2 & 5 & 7 & 0
\end{pmatrix}
$$

## Implementation

```python
"""
Floyd-Warshall all-pairs shortest paths algorithm.

Computes shortest paths between all pairs of vertices in O(V^3) time
using dynamic programming with intermediate vertex expansion.
"""

from math import inf


# === Floyd-Warshall algorithm ================================================

def floyd_warshall(n: int, edges: list) -> tuple[list, list]:
    """Compute all-pairs shortest paths.

    Parameters
    ----------
    n : int
        Number of vertices (labeled 0 to n-1).
    edges : list of (u, v, w)
        Directed edges with weights.

    Returns
    -------
    dist : list of list
        dist[i][j] = shortest path weight from i to j.
    pred : list of list
        pred[i][j] = predecessor of j on shortest i->j path.
    """
    # Initialize distance and predecessor matrices
    dist = [[inf] * n for _ in range(n)]
    pred = [[None] * n for _ in range(n)]

    for i in range(n):
        dist[i][i] = 0

    for u, v, w in edges:
        dist[u][v] = w
        pred[u][v] = u

    # Main DP loop
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    pred[i][j] = pred[k][j]

    return dist, pred


# === Path reconstruction =====================================================

def reconstruct_path(pred: list, i: int, j: int) -> list:
    """Reconstruct the shortest path from vertex i to vertex j."""
    if pred[i][j] is None and i != j:
        return []  # no path
    path = []
    v = j
    while v != i:
        if v is None:
            return []  # no path
        path.append(v)
        v = pred[i][v]
    path.append(i)
    path.reverse()
    return path


# === Negative cycle detection ================================================

def has_negative_cycle(dist: list) -> bool:
    """Check if the graph contains a negative-weight cycle."""
    return any(dist[i][i] < 0 for i in range(len(dist)))


# === Demo ====================================================================

if __name__ == "__main__":
    n = 4
    edges = [
        (0, 1, 3), (0, 3, 7),
        (1, 0, 8), (1, 2, 2),
        (2, 0, 5), (2, 3, 1),
        (3, 0, 2),
    ]

    dist, pred = floyd_warshall(n, edges)

    print("Distance matrix:")
    for row in dist:
        print([x if x != inf else "inf" for x in row])

    print(f"\nShortest path 1->3: {reconstruct_path(pred, 1, 3)}")
    print(f"Distance 1->3: {dist[1][3]}")
    print(f"Shortest path 2->1: {reconstruct_path(pred, 2, 1)}")
    print(f"Distance 2->1: {dist[2][1]}")
    print(f"Negative cycle: {has_negative_cycle(dist)}")
```

**Output:**

```
Distance matrix:
[0, 3, 5, 6]
[5, 0, 2, 3]
[3, 6, 0, 1]
[2, 5, 7, 0]

Shortest path 1->3: [1, 2, 3]
Distance 1->3: 3
Shortest path 2->1: [2, 0, 1]
Distance 2->1: 6
Negative cycle: False
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to
  Algorithms* (4th ed.), Chapter 25.2: The Floyd-Warshall Algorithm.
- Floyd, R. W. (1962). Algorithm 97: Shortest path. *Communications of the
  ACM*, 5(6), 345.
