# Centroid Decomposition

Centroid decomposition recursively splits a tree at balanced pivot nodes, producing
an auxiliary tree of logarithmic depth that turns many hard path problems into
straightforward divide-and-conquer solutions.

## Intuition

Consider counting the number of paths in a tree whose length is at most $k$.
A brute-force approach takes $O(n^2)$, but if we can always split the tree into
roughly equal halves, we handle each half recursively and merge in linear time —
giving $O(n \log n)$ overall. The **centroid** of a tree is exactly the node that
guarantees this balanced split.

## Centroid of a Tree

**Definition.** A node $c$ in a tree $T$ with $n$ nodes is a **centroid** if every
subtree formed by removing $c$ contains at most $\lfloor n/2 \rfloor$ nodes.

**Theorem.** Every tree has at least one centroid, and at most two.

??? note "Proof sketch"
    Start at any node and move toward the largest subtree. At each step the
    largest-subtree size decreases while the "opposite" side grows. The process
    terminates at a node where no subtree exceeds $\lfloor n/2 \rfloor$.
    A second centroid can exist only when $n$ is even and two adjacent nodes
    each split the tree into components of size exactly $n/2$.

**Finding the centroid** requires two passes:

1. Root the tree at an arbitrary node and compute subtree sizes via DFS.
2. For each node $v$, the largest component after removing $v$ is

$$
\max\!\bigl(\,n - \text{size}(v),\;\max_{u \in \text{children}(v)} \text{size}(u)\bigr)
$$

3. Return the node that minimizes this value (it will be $\le \lfloor n/2 \rfloor$).

## The Decomposition Algorithm

Centroid decomposition builds an auxiliary **centroid tree** $T^*$:

1. Find the centroid $c$ of the current tree.
2. Mark $c$ as visited (logically remove it).
3. For each subtree $T_i$ remaining after removing $c$, recurse to find its
   centroid $c_i$.
4. Make every $c_i$ a child of $c$ in $T^*$.

The result is a rooted tree $T^*$ on the same $n$ nodes. Crucially, $T^*$ may have
a completely different shape from $T$.

### Depth Bound

**Lemma.** The centroid tree $T^*$ has depth at most $\lfloor \log_2 n \rfloor$.

??? note "Proof"
    Each time we remove a centroid from a component of size $m$, every remaining
    subtree has size at most $\lfloor m/2 \rfloor$. After $d$ levels of recursion
    the component sizes are at most $n / 2^d$, which reaches $1$ when
    $d = \lfloor \log_2 n \rfloor$.

### Time Complexity

At each level of the centroid tree the total work across all components is $O(n)$
(every node is processed exactly once per level). With $O(\log n)$ levels, the
overall construction time is $O(n \log n)$.

## Worked Example

Consider a path graph $1 - 2 - 3 - 4 - 5 - 6 - 7$ with $n = 7$.

| Step | Component | Centroid | Remaining subtrees |
|------|-----------|----------|--------------------|
| 1 | $\{1,\dots,7\}$ | $4$ | $\{1,2,3\}$, $\{5,6,7\}$ |
| 2 | $\{1,2,3\}$ | $2$ | $\{1\}$, $\{3\}$ |
| 3 | $\{5,6,7\}$ | $6$ | $\{5\}$, $\{7\}$ |

The centroid tree $T^*$ is rooted at $4$ with children $2$ and $6$;
node $2$ has children $1, 3$; node $6$ has children $5, 7$. The depth is $2 = \lfloor \log_2 7 \rfloor$.

## Applications

- **Distance queries**: For every node, store its distance to each ancestor in $T^*$.
  A path query between $u$ and $v$ can be answered by examining only their
  $O(\log n)$ ancestors — giving $O(n \log n)$ preprocessing and $O(\log n)$ per query.
- **Path counting**: Count paths satisfying a predicate (e.g., length $\le k$) in
  $O(n \log n)$ by processing each centroid's component.
- **Update queries**: Support point updates on nodes and answer path-aggregate
  queries in $O(\log^2 n)$ by maintaining data structures at each centroid ancestor.

## Implementation

```python
"""Centroid decomposition of an unrooted tree."""

import sys
from collections import defaultdict

# === Constants ===
sys.setrecursionlimit(300_000)


# === Subtree size computation ===
def compute_sizes(adj, root, removed):
    """Compute subtree sizes via iterative DFS."""
    size = defaultdict(lambda: 1)
    parent = {root: -1}
    order = []
    stack = [root]
    while stack:
        v = stack.pop()
        order.append(v)
        for u in adj[v]:
            if u != parent[v] and not removed[u]:
                parent[u] = v
                stack.append(u)
    for v in reversed(order):
        if parent[v] != -1:
            size[parent[v]] += size[v]
    return size, len(order)


# === Find centroid ===
def find_centroid(adj, root, removed):
    """Return the centroid of the component containing *root*."""
    size, n = compute_sizes(adj, root, removed)
    best, best_val = root, n
    stack = [root]
    parent = {root: -1}
    while stack:
        v = stack.pop()
        largest = n - size[v]
        for u in adj[v]:
            if u != parent[v] and not removed[u]:
                largest = max(largest, size[u])
                parent[u] = v
                stack.append(u)
        if largest < best_val:
            best, best_val = v, largest
    return best


# === Build centroid tree ===
def centroid_decomposition(adj, n):
    """Build and return the centroid tree as a parent array."""
    removed = [False] * n
    ct_parent = [-1] * n

    def decompose(root, par):
        c = find_centroid(adj, root, removed)
        removed[c] = True
        ct_parent[c] = par
        for u in adj[c]:
            if not removed[u]:
                decompose(u, c)

    decompose(0, -1)
    return ct_parent


# === Demo ===
if __name__ == "__main__":
    # Path graph: 0-1-2-3-4-5-6
    n = 7
    adj = defaultdict(list)
    for i in range(n - 1):
        adj[i].append(i + 1)
        adj[i + 1].append(i)

    ct_parent = centroid_decomposition(adj, n)
    print("Centroid tree (parent array):", ct_parent)
    # Expected root (centroid) has parent -1
    root = ct_parent.index(-1)
    print(f"Root of centroid tree: {root}")
    print(f"Depth bound: {n.bit_length() - 1}")
```

## Complexity Summary

| Operation | Time | Space |
|-----------|------|-------|
| Build centroid tree | $O(n \log n)$ | $O(n)$ |
| Depth of centroid tree | $O(\log n)$ | — |
| Distance query (with preprocessing) | $O(\log n)$ per query | $O(n \log n)$ |

## Reference

- [Competitive Programmer's Handbook](https://cses.fi/book/book.pdf)
- Bentley, J. L. (1980). *Multidimensional Divide-and-Conquer*
