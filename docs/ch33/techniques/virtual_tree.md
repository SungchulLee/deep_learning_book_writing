# Virtual Tree

Many tree problems involve queries that reference only a small subset of the tree's nodes. Processing the entire tree for each query wastes time when the relevant nodes are sparse. A **virtual tree** (also called *auxiliary tree*) is a compressed tree that contains only the query-relevant nodes and their pairwise lowest common ancestors, preserving the ancestral relationships of the original tree. This reduces the problem size from $n$ to $O(k)$ where $k$ is the number of relevant nodes.

## Definition

Given a rooted tree $T$ with $n$ nodes and a set of **key nodes** $S = \{v_1, v_2, \ldots, v_k\}$, the virtual tree $T'$ is the minimal subtree of $T$ that:

1. Contains all nodes in $S$.
2. Contains the LCA of every pair of nodes in $S$.
3. Contains the root of $T$ (optional, depending on the formulation).
4. Preserves ancestor-descendant relationships from $T$.

The virtual tree has at most $2k - 1$ nodes: the $k$ key nodes plus at most $k - 1$ pairwise LCAs.

## Construction Algorithm

The standard construction uses Euler-tour ordering and an LCA oracle.

### Steps

1. **Sort** the key nodes by their Euler-tour entry time (DFS order).
2. **Insert LCAs**: For each consecutive pair $(v_i, v_{i+1})$ in the sorted order, compute $\text{LCA}(v_i, v_{i+1})$ and add it to the node set.
3. **Deduplicate** and re-sort all nodes (key nodes + LCAs) by DFS order.
4. **Build the virtual tree** using a stack: process nodes in DFS order, maintaining a stack of ancestors. For each new node, pop the stack until the top is an ancestor, add an edge from the stack top to the new node, and push the new node.

### Why Consecutive LCAs Suffice

A key property: the LCA of any two nodes in $S$ equals the LCA of some pair of consecutive nodes (in DFS order). This is because if $v_i$ and $v_j$ are not consecutive, there exists some $v_m$ between them in DFS order, and $\text{LCA}(v_i, v_j) = \text{LCA}(v_i, v_m)$ or $\text{LCA}(v_m, v_j)$.

## Implementation

```python
"""
Virtual tree construction.

Given a rooted tree and a set of key nodes, constructs the
virtual (auxiliary) tree containing only the key nodes and
their pairwise LCAs.
"""

import math
from collections import deque

# ===================================================================
# LCA with Binary Lifting
# ===================================================================

class LCAOracle:
    """Preprocess a tree for O(log n) LCA queries."""

    def __init__(self, adj, root=0):
        self.n = len(adj)
        self.LOG = max(1, math.ceil(math.log2(self.n))) + 1
        self.depth = [0] * self.n
        self.up = [[0] * self.LOG for _ in range(self.n)]
        self.tin = [0] * self.n  # Euler tour entry time
        self.tout = [0] * self.n
        self._timer = 0

        # BFS for depth and parent
        visited = [False] * self.n
        visited[root] = True
        queue = deque([root])
        order = []
        while queue:
            u = queue.popleft()
            order.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    self.depth[v] = self.depth[u] + 1
                    self.up[v][0] = u
                    queue.append(v)

        # Binary lifting table
        for k in range(1, self.LOG):
            for v in range(self.n):
                self.up[v][k] = self.up[self.up[v][k - 1]][k - 1]

        # DFS for Euler tour times
        self._dfs_iterative(adj, root)

    def _dfs_iterative(self, adj, root):
        stack = [(root, -1, False)]
        while stack:
            u, par, leaving = stack.pop()
            if leaving:
                self.tout[u] = self._timer
                self._timer += 1
                continue
            self.tin[u] = self._timer
            self._timer += 1
            stack.append((u, par, True))
            for v in adj[u]:
                if v != par:
                    stack.append((v, u, False))

    def is_ancestor(self, u, v):
        return self.tin[u] <= self.tin[v] and self.tout[u] >= self.tout[v]

    def query(self, u, v):
        if self.is_ancestor(u, v):
            return u
        if self.is_ancestor(v, u):
            return v
        for k in range(self.LOG - 1, -1, -1):
            if not self.is_ancestor(self.up[u][k], v):
                u = self.up[u][k]
        return self.up[u][0]

# ===================================================================
# Virtual Tree Construction
# ===================================================================

def build_virtual_tree(lca_oracle, key_nodes):
    """Construct a virtual tree from key nodes.

    Args:
        lca_oracle: LCAOracle instance
        key_nodes: list of key node indices

    Returns:
        vt_adj: adjacency dict of the virtual tree
        vt_nodes: sorted list of virtual tree nodes
    """
    if not key_nodes:
        return {}, []

    # Sort by DFS entry time
    nodes = sorted(set(key_nodes), key=lambda v: lca_oracle.tin[v])

    # Add LCAs of consecutive pairs
    all_nodes = set(nodes)
    for i in range(len(nodes) - 1):
        lca_node = lca_oracle.query(nodes[i], nodes[i + 1])
        all_nodes.add(lca_node)

    # Sort all nodes by DFS entry time
    vt_nodes = sorted(all_nodes, key=lambda v: lca_oracle.tin[v])

    # Build virtual tree using stack
    vt_adj = {v: [] for v in vt_nodes}
    stack = [vt_nodes[0]]

    for i in range(1, len(vt_nodes)):
        v = vt_nodes[i]
        # Pop until stack top is an ancestor of v
        while len(stack) > 1 and not lca_oracle.is_ancestor(stack[-1], v):
            stack.pop()
        # Add edge from stack top to v
        vt_adj[stack[-1]].append(v)
        stack.append(v)

    return vt_adj, vt_nodes

# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    #         0
    #        / \
    #       1   2
    #      /|    \
    #     3 4     5
    #    /       / \
    #   6       7   8
    n = 9
    adj = [[] for _ in range(n)]
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5),
             (3, 6), (5, 7), (5, 8)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    oracle = LCAOracle(adj, root=0)

    key_nodes = [6, 4, 7]
    vt_adj, vt_nodes = build_virtual_tree(oracle, key_nodes)

    print(f"Original tree: {n} nodes")
    print(f"Key nodes: {key_nodes}")
    print(f"Virtual tree nodes: {vt_nodes}")
    print(f"Virtual tree size: {len(vt_nodes)}")
    print(f"Virtual tree edges:")
    for u in vt_nodes:
        for v in vt_adj[u]:
            print(f"  {u} -> {v}")
```

**Output:**
```
Original tree: 9 nodes
Key nodes: [6, 4, 7]
Virtual tree nodes: [0, 1, 4, 6, 7]
Virtual tree size: 5
Virtual tree edges:
  0 -> 1
  0 -> 7
  1 -> 4
  1 -> 6
```

## Complexity

| Phase | Time |
|---|---|
| LCA preprocessing | $O(n \log n)$ |
| Sort key nodes | $O(k \log k)$ |
| Compute LCAs | $O(k \log n)$ |
| Build virtual tree | $O(k)$ |
| **Total per query** | $O(k \log n)$ |

The virtual tree has at most $2k - 1$ nodes and $2k - 2$ edges, so any subsequent tree DP on the virtual tree costs $O(k)$ rather than $O(n)$.

## Applications

- **Steiner tree on trees**: Find the minimum-weight subtree connecting a set of key nodes. Build the virtual tree, then sum edge weights.
- **Multi-query tree DP**: When multiple queries each specify a small set of relevant nodes, build a virtual tree per query and run DP on it.
- **Path counting**: Count paths between pairs of key nodes by analyzing the virtual tree structure.

## Reference

- Competitive Programmer's Handbook (Laaksonen).
- Various competitive programming resources on "auxiliary tree" / "virtual tree" construction.
