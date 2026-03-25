# Transitive Closure

Many graph problems do not require exact shortest-path distances — they only
need to know whether a path exists from vertex $u$ to vertex $v$.  The
**transitive closure** of a directed graph answers this reachability question
for all pairs of vertices simultaneously.  It is closely related to the
Floyd-Warshall algorithm but operates on Boolean values instead of distances,
making it simpler and more efficient in practice.

## Definition

Given a directed graph $G = (V, E)$, the **transitive closure** is a graph
$G^* = (V, E^*)$ where:

$$
(u, v) \in E^* \iff \text{there exists a path from } u \text{ to } v \text{ in } G
$$

Equivalently, the transitive closure can be represented as a Boolean matrix
$T$ where:

$$
T[i][j] =
\begin{cases}
1 & \text{if vertex } j \text{ is reachable from vertex } i \\
0 & \text{otherwise}
\end{cases}
$$

## Warshall's Algorithm

Warshall's algorithm computes the transitive closure using the same
intermediate-vertex expansion as Floyd-Warshall, but replaces arithmetic
operations ($\min$, $+$) with Boolean operations ($\lor$, $\land$):

$$
t^{(k)}_{ij} = t^{(k-1)}_{ij} \lor \left(t^{(k-1)}_{ik} \land t^{(k-1)}_{kj}\right)
$$

**Base case ($k = 0$):**

$$
t^{(0)}_{ij} =
\begin{cases}
1 & \text{if } i = j \text{ or } (i, j) \in E \\
0 & \text{otherwise}
\end{cases}
$$

**Interpretation:** Vertex $j$ is reachable from $i$ using intermediates from
$\{1, \dots, k\}$ if either it was already reachable using $\{1, \dots, k-1\}$,
or there is a path $i \leadsto k \leadsto j$ through vertex $k$.

## Pseudocode

```
TRANSITIVE-CLOSURE(G):
    n = |V|
    T = adjacency matrix of G (with diagonal set to 1)
    for k = 0 to n-1:
        for i = 0 to n-1:
            for j = 0 to n-1:
                T[i][j] = T[i][j] OR (T[i][k] AND T[k][j])
    return T
```

## Complexity

- **Time:** $\Theta(V^3)$ — three nested loops over $n$ vertices.
- **Space:** $\Theta(V^2)$ for the Boolean matrix.

Although the asymptotic time matches Floyd-Warshall, Boolean operations are
significantly cheaper than floating-point arithmetic.  Furthermore, the matrix
can be stored using bitsets (one bit per entry), reducing space by a factor of
32 or 64 and enabling bitwise parallelism in the inner loop.

### Bitset Optimization

Storing each row as a bitset of $n$ bits allows the inner loop to be replaced
by a single bitwise OR:

```
for k = 0 to n-1:
    for i = 0 to n-1:
        if T[i][k]:
            T[i] = T[i] OR T[k]
```

This reduces the practical running time by a factor of the word size (typically
64), giving an effective complexity of $O(V^3 / w)$ where $w$ is the word size.

## Connection to Floyd-Warshall

Warshall's algorithm is a Boolean specialization of Floyd-Warshall:

| Operation | Floyd-Warshall | Warshall |
|---|---|---|
| Combine | $\min$ | $\lor$ |
| Extend | $+$ | $\land$ |
| Identity | $\infty$ | $0$ (false) |
| Base value | edge weight | $1$ (true) |
| Result | shortest distances | reachability |

This connection extends to the algebraic concept of a **semiring**: both
algorithms compute a closure over a semiring, with Floyd-Warshall using the
tropical semiring $(\min, +, \infty, 0)$ and Warshall using the Boolean
semiring $(\lor, \land, 0, 1)$.

## Alternative Approaches

The transitive closure can also be computed by:

- **BFS/DFS from each vertex:** $O(V(V + E))$ time.  Better for sparse graphs
  when $E \ll V^2$.
- **Matrix multiplication:** Compute $A + A^2 + \cdots + A^{V-1}$ where $A$ is
  the adjacency matrix.  Using repeated squaring, this takes $O(V^3 \log V)$
  multiplications but can leverage fast matrix multiplication for sub-cubic
  theoretical bounds.

## Worked Example

Consider a graph with 4 vertices:

Edges: $(0, 1)$, $(1, 2)$, $(2, 3)$, $(3, 1)$.

Initial matrix (with self-loops on diagonal):

$$
T^{(0)} = \begin{pmatrix}
1 & 1 & 0 & 0 \\
0 & 1 & 1 & 0 \\
0 & 0 & 1 & 1 \\
0 & 1 & 0 & 1
\end{pmatrix}
$$

**After $k=0$:** Row 0 provides paths through vertex 0.
$T[1][0] = 0$, so no changes via vertex 0.

**After $k=1$:** Vertex 1 as intermediate.
$T[0][2] = T[0][1] \land T[1][2] = 1 \land 1 = 1$.
$T[3][2] = T[3][1] \land T[1][2] = 1 \land 1 = 1$.

**After $k=2$:** Vertex 2 as intermediate.
$T[0][3] = T[0][2] \land T[2][3] = 1 \land 1 = 1$.
$T[1][3] = T[1][2] \land T[2][3] = 1 \land 1 = 1$.
$T[3][3]$ already 1.

**After $k=3$:** Vertex 3 as intermediate.
$T[0][1]$ already 1.  $T[1][1]$ already 1.
$T[2][1] = T[2][3] \land T[3][1] = 1 \land 1 = 1$.

Final matrix:

$$
T^{(4)} = \begin{pmatrix}
1 & 1 & 1 & 1 \\
0 & 1 & 1 & 1 \\
0 & 1 & 1 & 1 \\
0 & 1 & 1 & 1
\end{pmatrix}
$$

Vertex 0 can reach all others, but no vertex can reach vertex 0 (except itself).

## Implementation

```python
"""
Transitive closure of a directed graph using Warshall's algorithm.

Determines reachability between all pairs of vertices in O(V^3) time,
using Boolean operations instead of arithmetic.
"""


# === Warshall's algorithm ====================================================

def transitive_closure(n: int, edges: list) -> list:
    """Compute the transitive closure of a directed graph.

    Parameters
    ----------
    n : int
        Number of vertices (labeled 0 to n-1).
    edges : list of (u, v)
        Directed edges.

    Returns
    -------
    T : list of list of bool
        T[i][j] is True if j is reachable from i.
    """
    # Initialize: diagonal and direct edges
    T = [[False] * n for _ in range(n)]
    for i in range(n):
        T[i][i] = True
    for u, v in edges:
        T[u][v] = True

    # Warshall's algorithm
    for k in range(n):
        for i in range(n):
            if T[i][k]:
                for j in range(n):
                    if T[k][j]:
                        T[i][j] = True

    return T


# === Display utility =========================================================

def print_matrix(T: list, label: str = "Reachability matrix") -> None:
    """Print a Boolean matrix as 0s and 1s."""
    print(f"{label}:")
    for row in T:
        print("  " + " ".join("1" if x else "0" for x in row))


# === Demo ====================================================================

if __name__ == "__main__":
    n = 4
    edges = [(0, 1), (1, 2), (2, 3), (3, 1)]

    T = transitive_closure(n, edges)
    print_matrix(T)

    # Check specific reachability
    print(f"\n0 -> 3 reachable: {T[0][3]}")
    print(f"3 -> 0 reachable: {T[3][0]}")
    print(f"1 -> 3 reachable: {T[1][3]}")
    print(f"2 -> 1 reachable: {T[2][1]}")
```

**Output:**

```
Reachability matrix:
  1 1 1 1
  0 1 1 1
  0 1 1 1
  0 1 1 1

0 -> 3 reachable: True
3 -> 0 reachable: False
1 -> 3 reachable: True
2 -> 1 reachable: True
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to
  Algorithms* (4th ed.), Chapter 25.2: Transitive Closure of a Directed Graph.
- Warshall, S. (1962). A theorem on Boolean matrices. *Journal of the ACM*,
  9(1), 11-12.
