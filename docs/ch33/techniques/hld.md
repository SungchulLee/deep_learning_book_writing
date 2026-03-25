# Heavy-Light Decomposition

Heavy-light decomposition (HLD) partitions a rooted tree into vertex-disjoint
chains so that any root-to-leaf path crosses at most $O(\log n)$ chains. Combined
with a segment tree over the chains, it answers path queries and updates in
$O(\log^2 n)$ time.

## Intuition

In a rooted tree, some children are "heavier" (have larger subtrees) than others.
By always extending the current chain through the heaviest child, we guarantee that
switching to a different chain at least halves the subtree size. This limits the
number of chain switches on any path to $O(\log n)$.

## Definitions

Let $T$ be a rooted tree with $n$ nodes. For each non-leaf node $v$:

- The **heavy child** of $v$ is the child $u$ with the largest subtree:
  $\text{size}(u) \ge \text{size}(w)$ for all children $w$ of $v$. Ties are broken
  arbitrarily.
- The edge $(v, u)$ to the heavy child is a **heavy edge**; all other child edges
  are **light edges**.
- A maximal path of heavy edges forms a **heavy chain**.
- The **head** of a chain is its topmost node (closest to the root).

**Key property of light edges.** If $(v, u)$ is a light edge, then

$$
\text{size}(u) \le \frac{\text{size}(v)}{2}
$$

because $u$ is not the heaviest child.

## Chain-Count Bound

**Theorem.** Any root-to-leaf path crosses at most $\lfloor \log_2 n \rfloor + 1$
chains.

??? note "Proof"
    Each time the path traverses a light edge from $v$ to child $u$, the subtree
    size at least halves: $\text{size}(u) \le \text{size}(v)/2$. Starting from
    $\text{size}(\text{root}) = n$ and ending at a leaf with size $1$, the number
    of halvings is at most $\lfloor \log_2 n \rfloor$. Each halving corresponds to
    entering a new chain, giving at most $\lfloor \log_2 n \rfloor + 1$ chains.

## Algorithm

### Step 1 — Preprocessing

1. Root the tree and compute subtree sizes via DFS.
2. For each node, identify its heavy child.
3. Run a second DFS that assigns each node a **position** $\text{pos}[v]$ in a
   flat array. Within each chain, positions are contiguous.
4. Record $\text{head}[v]$ — the head of the chain containing $v$.

### Step 2 — Path Query

To query the path from $u$ to $v$:

1. While $\text{head}[u] \ne \text{head}[v]$:
    - Let $u$ be the node whose chain head is deeper (swap if needed).
    - Query the segment tree on $[\text{pos}[\text{head}[u]],\; \text{pos}[u]]$.
    - Move $u$ up to $\text{parent}[\text{head}[u]]$.
2. When both nodes are on the same chain, query
   $[\min(\text{pos}[u], \text{pos}[v]),\; \max(\text{pos}[u], \text{pos}[v])]$.

Each iteration removes one chain ($O(\log n)$ iterations), and each segment-tree
query costs $O(\log n)$, for a total of $O(\log^2 n)$.

## Worked Example

```
        1
       /|\
      2  3  6
     / \
    4   5
```

Subtree sizes: $\text{size}(1)=6,\; \text{size}(2)=3,\; \text{size}(3)=1,\;
\text{size}(4)=1,\; \text{size}(5)=1,\; \text{size}(6)=1$.

Heavy children: $1 \to 2$ (size 3 vs 1), $2 \to 4$ (tie, pick 4).

Chains: $[1, 2, 4]$, $[5]$, $[3]$, $[6]$.

Path query $5 \to 6$:

1. $\text{head}[5] = 5$, $\text{head}[6] = 6$. Move $5$ up to $\text{parent}[5] = 2$.
2. $\text{head}[2] = 1$, $\text{head}[6] = 6$. Move $6$ up to $\text{parent}[6] = 1$.
3. Now both on chain $[1, 2, 4]$. Query $[\text{pos}[1], \text{pos}[2]]$.

Total: 3 segment-tree queries.

## Implementation

```python
"""Heavy-light decomposition with path-max queries."""

import sys
from collections import defaultdict

# === Constants ===
sys.setrecursionlimit(300_000)
INF = float("inf")


# === HLD construction ===
class HLD:
    """Heavy-light decomposition of an unrooted tree."""

    def __init__(self, adj, root, n):
        self.n = n
        self.adj = adj
        self.root = root
        self.parent = [-1] * n
        self.depth = [0] * n
        self.size = [1] * n
        self.heavy = [-1] * n
        self.head = list(range(n))
        self.pos = [0] * n
        self._timer = 0

        self._compute_sizes()
        self._decompose()

    def _compute_sizes(self):
        """Iterative DFS to compute subtree sizes and heavy children."""
        stack = [(self.root, -1, False)]
        order = []
        while stack:
            v, par, entered = stack.pop()
            if entered:
                for u in self.adj[v]:
                    if u != par:
                        self.size[v] += self.size[u]
                        if self.heavy[v] == -1 or self.size[u] > self.size[self.heavy[v]]:
                            self.heavy[v] = u
                continue
            self.parent[v] = par
            stack.append((v, par, True))
            order.append(v)
            for u in self.adj[v]:
                if u != par:
                    self.depth[u] = self.depth[v] + 1
                    stack.append((u, v, False))

    def _decompose(self):
        """Assign chain heads and flat positions."""
        stack = [(self.root, self.root)]
        while stack:
            v, h = stack.pop()
            self.head[v] = h
            self.pos[v] = self._timer
            self._timer += 1
            # Process light children first (reversed for stack order)
            children = [u for u in self.adj[v] if u != self.parent[v]]
            for u in children:
                if u != self.heavy[v]:
                    stack.append((u, u))
            # Heavy child last so it is processed first (stack LIFO)
            if self.heavy[v] != -1:
                stack.append((self.heavy[v], h))

    def path_query(self, u, v, seg_query):
        """Query the path u-v using a segment-tree query function.

        seg_query(l, r) should return the answer for range [l, r].
        """
        result = 0
        while self.head[u] != self.head[v]:
            if self.depth[self.head[u]] < self.depth[self.head[v]]:
                u, v = v, u
            result = max(result, seg_query(self.pos[self.head[u]], self.pos[u]))
            u = self.parent[self.head[u]]
        if self.depth[u] > self.depth[v]:
            u, v = v, u
        result = max(result, seg_query(self.pos[u], self.pos[v]))
        return result


# === Demo ===
if __name__ == "__main__":
    n = 6
    adj = defaultdict(list)
    edges = [(0, 1), (0, 2), (0, 5), (1, 3), (1, 4)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    hld = HLD(adj, 0, n)
    print("pos: ", hld.pos)
    print("head:", hld.head)
    print("heavy:", hld.heavy)
```

## Complexity Summary

| Operation | Time | Space |
|-----------|------|-------|
| Preprocessing | $O(n)$ | $O(n)$ |
| Path query | $O(\log^2 n)$ | — |
| Path update | $O(\log^2 n)$ | — |
| Subtree query | $O(\log n)$ | — |

## Reference

- Sleator, D. D. & Tarjan, R. E. (1983). *A Data Structure for Dynamic Trees*
- [Competitive Programmer's Handbook](https://cses.fi/book/book.pdf)
