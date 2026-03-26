# Lowest Common Ancestor

Given a rooted tree, the **lowest common ancestor** (LCA) of two nodes $u$ and $v$ is the deepest node that is an ancestor of both. For example, finding the distance between two nodes in a tree requires knowing their deepest shared ancestor: the path from $u$ to $v$ always passes through $\text{LCA}(u, v)$. LCA queries appear throughout tree algorithms -- computing distances, answering path queries, building virtual trees, and solving range problems. Efficient LCA algorithms reduce these tasks from $O(n)$ per query to $O(\log n)$ or even $O(1)$.

## Definition and Properties

Let $T$ be a rooted tree with root $r$. For any node $u$, the set of ancestors of $u$ is the set of nodes on the path from $u$ to $r$ (inclusive). The LCA of $u$ and $v$ is:

$$
\text{LCA}(u, v) = \arg\max_{w} \{\text{depth}(w) : w \text{ is an ancestor of both } u \text{ and } v\}
$$

Key properties:

- $\text{LCA}(u, u) = u$ for all nodes $u$.
- $\text{LCA}(u, v) = u$ if and only if $u$ is an ancestor of $v$.
- The path from $u$ to $v$ passes through $\text{LCA}(u, v)$.
- The distance between $u$ and $v$ in the tree is $\text{depth}(u) + \text{depth}(v) - 2 \cdot \text{depth}(\text{LCA}(u, v))$.

## Binary Lifting

The most widely used LCA algorithm in competitive programming is **binary lifting** (also called the method of doubling). It preprocesses the tree in $O(n \log n)$ time and answers each LCA query in $O(\log n)$.

### Preprocessing

For each node $u$ and each power $k$, store $\text{up}[u][k]$, the $2^k$-th ancestor of $u$:

$$
\text{up}[u][k] = \text{up}[\text{up}[u][k-1]][k-1]
$$

The base case is $\text{up}[u][0] = \text{parent}(u)$. The table has $n$ rows and $\lceil \log_2 n \rceil$ columns.

### Query Algorithm

To find $\text{LCA}(u, v)$:

1. Bring $u$ and $v$ to the same depth by lifting the deeper node.
2. If $u = v$, return $u$.
3. Simultaneously lift both $u$ and $v$ by decreasing powers of 2, stopping just below the LCA.
4. Return $\text{parent}(u)$.

```python
"""
Lowest Common Ancestor using binary lifting.

Preprocesses a rooted tree in O(n log n) and answers
each LCA query in O(log n).
"""

import math
from collections import deque

# ===================================================================
# Binary Lifting LCA
# ===================================================================

class LCA:
    """LCA with binary lifting on a rooted tree."""

    def __init__(self, adj, root=0):
        """Build the binary lifting table.

        Args:
            adj: adjacency list (list of lists)
            root: root node index
        """
        self.n = len(adj)
        self.LOG = max(1, math.ceil(math.log2(self.n))) + 1
        self.depth = [0] * self.n
        self.up = [[0] * self.LOG for _ in range(self.n)]

        # BFS to compute depth and parent (up[v][0])
        visited = [False] * self.n
        visited[root] = True
        queue = deque([root])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    self.depth[v] = self.depth[u] + 1
                    self.up[v][0] = u
                    queue.append(v)

        # Fill binary lifting table
        for k in range(1, self.LOG):
            for v in range(self.n):
                self.up[v][k] = self.up[self.up[v][k - 1]][k - 1]

    def query(self, u, v):
        """Return the LCA of nodes u and v."""
        # Step 1: bring to same depth
        if self.depth[u] < self.depth[v]:
            u, v = v, u
        diff = self.depth[u] - self.depth[v]
        for k in range(self.LOG):
            if (diff >> k) & 1:
                u = self.up[u][k]

        if u == v:
            return u

        # Step 2: lift both until just below LCA
        for k in range(self.LOG - 1, -1, -1):
            if self.up[u][k] != self.up[v][k]:
                u = self.up[u][k]
                v = self.up[v][k]

        return self.up[u][0]

    def distance(self, u, v):
        """Return the distance (number of edges) between u and v."""
        w = self.query(u, v)
        return self.depth[u] + self.depth[v] - 2 * self.depth[w]

# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    # Build a tree:
    #       0
    #      / \
    #     1   2
    #    / \   \
    #   3   4   5
    #  /
    # 6
    n = 7
    adj = [[] for _ in range(n)]
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (3, 6)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    lca = LCA(adj, root=0)

    print(f"LCA(3, 4) = {lca.query(3, 4)}")  # 1
    print(f"LCA(6, 5) = {lca.query(6, 5)}")  # 0
    print(f"LCA(6, 4) = {lca.query(6, 4)}")  # 1
    print(f"LCA(6, 6) = {lca.query(6, 6)}")  # 6
    print(f"dist(6, 5) = {lca.distance(6, 5)}")  # 5
```

**Output:**
```
LCA(3, 4) = 1
LCA(6, 5) = 0
LCA(6, 4) = 1
LCA(6, 6) = 6
dist(6, 5) = 5
```

## Complexity Analysis

| Phase | Time | Space |
|---|---|---|
| Preprocessing | $O(n \log n)$ | $O(n \log n)$ |
| Query | $O(\log n)$ | -- |

The $O(n \log n)$ space for the binary lifting table is acceptable for $n \le 10^5$ (roughly 1.7 million entries at $\log n \approx 17$).

## Alternative Approaches

### Euler Tour + Sparse Table

An alternative achieves $O(1)$ per query after $O(n \log n)$ preprocessing:

1. Compute an Euler tour of the tree, recording the depth at each step.
2. LCA of $u$ and $v$ is the node with minimum depth between the first occurrences of $u$ and $v$ in the Euler tour.
3. This is a range minimum query (RMQ), solvable in $O(1)$ with a sparse table.

This approach is covered in detail in the [Euler Tour](euler_tour.md) section.

### Tarjan's Offline LCA

When all queries are known in advance, Tarjan's offline algorithm answers all $q$ queries in $O(n \cdot \alpha(n) + q)$ using a DFS with Union-Find. This is optimal but requires offline processing.

## Applications

- **Tree distance queries**: $\text{dist}(u, v) = \text{depth}(u) + \text{depth}(v) - 2 \cdot \text{depth}(\text{LCA}(u, v))$.
- **Path aggregation**: Compute sum, max, or min on the path from $u$ to $v$ by splitting at the LCA.
- **Virtual trees**: Construct a compressed tree containing only query-relevant nodes and their LCAs (see [Virtual Tree](virtual_tree.md)).
- **Heavy-light decomposition**: LCA determines which chain transitions occur on a path query (see [HLD](hld.md)).

## Reference

- [Competitive Programmer's Handbook](https://cses.fi/book/book.pdf)
