# Implicit Graphs

Not every graph fits neatly into an adjacency list or matrix. Many important graphs are too large to store explicitly -- or are even infinite -- yet their structure follows simple rules that let us compute neighbors on demand. A chessboard where each cell connects to its cardinal neighbors, a Rubik's Cube where each state connects to states reachable by one twist, or the game tree of chess: all are graphs defined by rules rather than data. These are implicit graphs, and understanding them is essential for search, planning, and combinatorial optimization.

## Definition

An **implicit graph** $G = (V, E)$ is a graph where the vertices and edges are not stored explicitly in memory. Instead, a **successor function** (or neighbor function) $\text{neighbors}(v)$ computes the set of vertices adjacent to $v$ on demand.

Formally, the graph is specified by:

1. An initial vertex (or set of vertices) $s \in V$.
2. A function $\text{neighbors}: V \to 2^V$ that returns the neighbors of any given vertex.
3. Optionally, a goal predicate $\text{goal}: V \to \{0, 1\}$.

The edge set is defined implicitly:

$$
E = \{(u, v) : v \in \text{neighbors}(u)\}
$$

## Why Implicit Graphs Matter

Implicit graphs arise when the state space is too large to enumerate:

| Domain | Vertex | Neighbor Function | State Space Size |
|---|---|---|---|
| Grid/maze | Cell $(r, c)$ | Adjacent non-wall cells | $O(rows \times cols)$ |
| Sliding puzzle | Board configuration | Configurations reachable by one slide | $n!$ |
| Rubik's Cube | Cube state | States from one quarter-turn | $\approx 4.3 \times 10^{19}$ |
| Chess | Board position | Legal moves | $\approx 10^{47}$ |
| Word ladder | Dictionary word | Words differing by one letter | Dictionary size |

For a Rubik's Cube, storing all $4.3 \times 10^{19}$ states explicitly is impossible. BFS or DFS discovers only the states reachable from the start, generating neighbors lazily.

## Grid Graphs

The most common implicit graph in algorithmic problems is the **grid graph**. An $m \times n$ grid has vertices at integer coordinates $(r, c)$ with $0 \leq r < m$ and $0 \leq c < n$. Each cell connects to its 4 cardinal neighbors (up, down, left, right), subject to boundary conditions and obstacles.

$$
\text{neighbors}(r, c) = \{(r', c') : |r - r'| + |c - c'| = 1, \; 0 \leq r' < m, \; 0 \leq c' < n, \; \text{not blocked}\}
$$

```python
"""
Implicit graph examples: grid graph and word ladder.

Demonstrates how to define graphs via neighbor functions rather
than explicit adjacency structures, enabling BFS/DFS on large
or infinite state spaces.
"""

from collections import deque


# === Grid Graph ===

def grid_neighbors(r, c, rows, cols, blocked=None):
    """
    Compute neighbors of cell (r, c) in a grid graph.

    Returns adjacent cells within bounds that are not blocked.
    """
    if blocked is None:
        blocked = set()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    result = []
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in blocked:
            result.append((nr, nc))
    return result


def bfs_grid(rows, cols, start, goal, blocked=None):
    """
    BFS on an implicit grid graph.

    Returns the shortest path length from start to goal,
    or -1 if no path exists.
    """
    if blocked is None:
        blocked = set()
    if start == goal:
        return 0
    visited = {start}
    queue = deque([(start, 0)])

    while queue:
        (r, c), dist = queue.popleft()
        for nr, nc in grid_neighbors(r, c, rows, cols, blocked):
            if (nr, nc) == goal:
                return dist + 1
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), dist + 1))
    return -1


# === Word Ladder ===

def word_neighbors(word, dictionary):
    """
    Compute neighbors in a word ladder graph.

    A neighbor is a dictionary word that differs from the input
    by exactly one character.
    """
    result = []
    for i in range(len(word)):
        for c in 'abcdefghijklmnopqrstuvwxyz':
            if c != word[i]:
                candidate = word[:i] + c + word[i + 1:]
                if candidate in dictionary:
                    result.append(candidate)
    return result


def bfs_word_ladder(start, goal, dictionary):
    """
    BFS on the implicit word ladder graph.

    Returns the shortest transformation length, or -1 if
    no transformation exists.
    """
    if start == goal:
        return 0
    dict_set = set(dictionary)
    visited = {start}
    queue = deque([(start, 0)])

    while queue:
        word, dist = queue.popleft()
        for neighbor in word_neighbors(word, dict_set):
            if neighbor == goal:
                return dist + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return -1


# === Main ===

if __name__ == "__main__":
    # Grid BFS: 5x5 grid with obstacles
    blocked = {(1, 1), (1, 2), (1, 3), (2, 1)}
    dist = bfs_grid(5, 5, (0, 0), (4, 4), blocked)
    print(f"Grid shortest path (0,0)->(4,4): {dist} steps")

    # Unreachable goal
    wall = {(r, 2) for r in range(5)}  # full column wall
    dist2 = bfs_grid(5, 5, (0, 0), (0, 4), wall)
    print(f"Grid with wall: {dist2} (unreachable)")

    # Word ladder
    dictionary = ["hit", "hot", "dot", "dog", "lot", "log", "cog"]
    steps = bfs_word_ladder("hit", "cog", dictionary)
    print(f"\nWord ladder 'hit' -> 'cog': {steps} transformations")
```

**Output:**
```
Grid shortest path (0,0)->(4,4): 8 steps
Grid with wall: -1 (unreachable)
Word ladder 'hit' -> 'cog': 4 transformations
```

## Complexity Considerations

For implicit graphs, the standard complexity measures change:

- **Space.** We do not store the full graph. BFS or DFS stores the visited set, which grows up to $O(|V_{\text{reachable}}|)$.
- **Time.** Each vertex is visited once, and computing neighbors takes time $T_{\text{neighbors}}$. Total time is $O(|V_{\text{reachable}}| \cdot T_{\text{neighbors}})$.

For grid graphs, $T_{\text{neighbors}} = O(1)$ (at most 4 neighbors), so BFS runs in $O(V)$ where $V = m \times n$.

For word ladder graphs with words of length $L$ and dictionary of size $D$, $T_{\text{neighbors}} = O(26L)$ per vertex, and the total number of reachable vertices is at most $D$.

!!! warning "Infinite State Spaces"
    When the state space is infinite (e.g., unbounded grids, mathematical puzzles), BFS guarantees finding the shortest path but may run forever if no solution exists. Iterative deepening DFS (IDDFS) combines the optimality of BFS with the space efficiency of DFS for such problems. See [Iterative Deepening](../traversals/iddfs.md).

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 22.
- Russell, S. J., & Norvig, P. (2021). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson. Chapter 3.
