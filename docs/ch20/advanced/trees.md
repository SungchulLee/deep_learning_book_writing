# DP on Trees

Trees are naturally recursive: removing the root splits a tree into independent subtrees. This structure makes trees ideal for dynamic programming, where the optimal solution at each node combines optimal solutions from its children. DP on trees processes nodes in **post-order** (children before parents), ensuring that all child subproblems are solved before the parent's state is computed. Applications range from finding the maximum independent set and minimum vertex cover to computing tree diameters and subtree sums.

## General Framework

Given a rooted tree with $n$ nodes, define $dp[v]$ as the optimal value for the subtree rooted at $v$. The recurrence aggregates results from all children:

$$
dp[v] = f\bigl(dp[c_1], dp[c_2], \ldots, dp[c_k]\bigr)
$$

where $c_1, \ldots, c_k$ are the children of $v$ and $f$ depends on the problem.

**Base case**: for a leaf node $v$, $dp[v]$ is defined directly (often 0 or 1).

**Traversal order**: DFS post-order guarantees that $dp[c]$ is available before $dp[v]$ is computed.

## Example: Maximum Independent Set

An **independent set** is a set of vertices with no two adjacent. The maximum independent set on a tree can be solved in $O(n)$ time.

Define two states per node:

- $dp[v][0]$: maximum independent set size in the subtree of $v$, with $v$ **excluded**
- $dp[v][1]$: maximum independent set size in the subtree of $v$, with $v$ **included**

**Recurrence**:

$$
dp[v][0] = \sum_{c \in \text{children}(v)} \max\bigl(dp[c][0],\; dp[c][1]\bigr)
$$

$$
dp[v][1] = 1 + \sum_{c \in \text{children}(v)} dp[c][0]
$$

If $v$ is excluded, each child can be either included or excluded. If $v$ is included, no child can be included (adjacency constraint).

**Answer**: $\max\bigl(dp[\text{root}][0],\; dp[\text{root}][1]\bigr)$.

## Example: Minimum Vertex Cover

A **vertex cover** is a set of vertices such that every edge has at least one endpoint in the set.

- $dp[v][0]$: minimum vertex cover in the subtree of $v$, with $v$ **excluded**
- $dp[v][1]$: minimum vertex cover in the subtree of $v$, with $v$ **included**

**Recurrence**:

$$
dp[v][0] = \sum_{c \in \text{children}(v)} dp[c][1]
$$

$$
dp[v][1] = 1 + \sum_{c \in \text{children}(v)} \min\bigl(dp[c][0],\; dp[c][1]\bigr)
$$

If $v$ is excluded, every child must be included to cover the edges to $v$. If $v$ is included, each child can be either included or excluded.

## Example: Tree Diameter

The **diameter** of a tree is the length of the longest path between any two nodes. Define $\text{depth}(v)$ as the length of the longest downward path from $v$.

$$
\text{depth}(v) = \begin{cases} 0 & \text{if } v \text{ is a leaf} \\ 1 + \max_{c \in \text{children}(v)} \text{depth}(c) & \text{otherwise} \end{cases}
$$

The diameter passing through $v$ uses the two longest downward paths from $v$:

$$
\text{diameter through } v = \text{depth}_1(v) + \text{depth}_2(v)
$$

where $\text{depth}_1$ and $\text{depth}_2$ are the two largest child depths. The overall diameter is the maximum over all nodes.

## Implementation

```python
"""
DP on trees: maximum independent set, minimum vertex cover, and tree diameter.
"""

from collections import defaultdict


# ===================================================================
# Build adjacency list from edges
# ===================================================================
def build_tree(n: int, edges: list[tuple[int, int]], root: int = 0):
    """Build rooted tree from undirected edges.

    Parameters
    ----------
    n : int
        Number of nodes.
    edges : list[tuple[int, int]]
        Undirected edges.
    root : int
        Root node.

    Returns
    -------
    tuple
        Adjacency list and parent array.
    """
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    children = defaultdict(list)
    parent = [-1] * n
    visited = [False] * n
    stack = [root]
    order = []
    visited[root] = True

    while stack:
        node = stack.pop()
        order.append(node)
        for neighbor in adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                parent[neighbor] = node
                children[node].append(neighbor)
                stack.append(neighbor)

    return children, order


# ===================================================================
# Maximum independent set
# ===================================================================
def max_independent_set(n: int, edges: list[tuple[int, int]]) -> int:
    """Find maximum independent set size on a tree.

    Parameters
    ----------
    n : int
        Number of nodes.
    edges : list[tuple[int, int]]
        Tree edges.

    Returns
    -------
    int
        Size of the maximum independent set.
    """
    children, order = build_tree(n, edges)

    dp = [[0, 0] for _ in range(n)]

    # Process in reverse order (post-order)
    for v in reversed(order):
        dp[v][1] = 1
        for c in children[v]:
            dp[v][0] += max(dp[c][0], dp[c][1])
            dp[v][1] += dp[c][0]

    return max(dp[0][0], dp[0][1])


# ===================================================================
# Minimum vertex cover
# ===================================================================
def min_vertex_cover(n: int, edges: list[tuple[int, int]]) -> int:
    """Find minimum vertex cover size on a tree.

    Parameters
    ----------
    n : int
        Number of nodes.
    edges : list[tuple[int, int]]
        Tree edges.

    Returns
    -------
    int
        Size of the minimum vertex cover.
    """
    children, order = build_tree(n, edges)

    dp = [[0, 0] for _ in range(n)]

    for v in reversed(order):
        dp[v][1] = 1
        for c in children[v]:
            dp[v][0] += dp[c][1]
            dp[v][1] += min(dp[c][0], dp[c][1])

    return min(dp[0][0], dp[0][1])


# ===================================================================
# Tree diameter
# ===================================================================
def tree_diameter(n: int, edges: list[tuple[int, int]]) -> int:
    """Find the diameter of a tree.

    Parameters
    ----------
    n : int
        Number of nodes.
    edges : list[tuple[int, int]]
        Tree edges.

    Returns
    -------
    int
        Length of the longest path (number of edges).
    """
    children, order = build_tree(n, edges)

    depth = [0] * n
    diameter = 0

    for v in reversed(order):
        top_two = [0, 0]
        for c in children[v]:
            d = depth[c] + 1
            if d > top_two[0]:
                top_two[1] = top_two[0]
                top_two[0] = d
            elif d > top_two[1]:
                top_two[1] = d
        depth[v] = top_two[0]
        diameter = max(diameter, top_two[0] + top_two[1])

    return diameter


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    #       0
    #      / \
    #     1   2
    #    / \   \
    #   3   4   5
    #       |
    #       6
    n = 7
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (4, 6)]

    print(f"Max independent set: {max_independent_set(n, edges)}")
    print(f"Min vertex cover: {min_vertex_cover(n, edges)}")
    print(f"Tree diameter: {tree_diameter(n, edges)}")
```

**Output:**
```
Max independent set: 5
Min vertex cover: 2
Tree diameter: 4
```

## Complexity

All three examples run in $O(n)$ time and $O(n)$ space, making a single post-order pass over the tree.

| Problem | Time | Space | States per node |
|---------|------|-------|----------------|
| Max independent set | $O(n)$ | $O(n)$ | 2 |
| Min vertex cover | $O(n)$ | $O(n)$ | 2 |
| Tree diameter | $O(n)$ | $O(n)$ | 1 |

## Rerooting Technique

Some tree DP problems require computing the answer for **every node as root** (e.g., "for each node $v$, find the farthest node from $v$"). Naively re-rooting and re-running DP takes $O(n^2)$. The **rerooting technique** computes all $n$ answers in $O(n)$ total:

1. Run DP once with an arbitrary root to get $dp[v]$ for all $v$
2. In a second DFS, compute $dp_{\text{up}}[v]$ (contribution from the subtree above $v$) using the parent's $dp$ values minus $v$'s contribution
3. Combine $dp[v]$ and $dp_{\text{up}}[v]$ for each node

!!! tip "When to use rerooting"
    Use rerooting when the problem asks for a quantity that depends on which node is chosen as the root and the DP combines children's values using an **invertible** operation (sum, product, max with second-max tracking).

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 15. MIT Press.
