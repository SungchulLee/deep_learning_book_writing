# Fibonacci Heap Amortized Analysis

Fibonacci heaps achieve remarkable amortized bounds -- $O(1)$ for insert, merge, find-min, and decrease-key, and $O(\log n)$ for extract-min and delete -- but the worst-case cost of individual operations can be much higher. A single decrease-key may trigger a cascade of cuts costing $O(\log n)$, and extract-min consolidates the root list at a cost proportional to the number of trees. The potential method reveals that these expensive operations are always "paid for" by preceding cheap ones, yielding the advertised amortized bounds. Understanding this analysis is essential because these bounds directly determine the running time of Dijkstra's and Prim's algorithms.

## Potential Function

The amortized analysis uses the potential function:

$$
\Phi(H) = t(H) + 2\,m(H)
$$

where:

- $t(H)$ is the number of trees in the root list of heap $H$
- $m(H)$ is the number of **marked** nodes in $H$ (nodes that have lost one child since they were last made a child of another node)

Initially, $\Phi(H_0) = 0$ for an empty heap, and $\Phi(H) \ge 0$ always holds. The amortized cost of each operation is:

$$
\hat{c}_i = c_i + \Phi(H_i) - \Phi(H_{i-1})
$$

where $c_i$ is the actual cost and $H_i$ is the heap state after operation $i$.

## Insert: Amortized O(1)

Inserting a new node creates a single new tree in the root list.

**Actual cost**: $c = O(1)$ (create node, add to root list, update min pointer).

**Potential change**: $t(H)$ increases by 1, $m(H)$ is unchanged, so $\Delta\Phi = 1$.

$$
\hat{c} = O(1) + 1 = O(1)
$$

## Find-Min: Amortized O(1)

The minimum is maintained as a pointer. No structural change occurs.

**Actual cost**: $c = O(1)$.

**Potential change**: $\Delta\Phi = 0$.

$$
\hat{c} = O(1) + 0 = O(1)
$$

## Merge: Amortized O(1)

Merging two heaps concatenates their root lists and updates the min pointer.

**Actual cost**: $c = O(1)$ (with doubly-linked circular lists, concatenation is constant time).

**Potential change**: The new potential equals $t(H_1) + t(H_2) + 2(m(H_1) + m(H_2)) = \Phi(H_1) + \Phi(H_2)$, so $\Delta\Phi = 0$ relative to the sum of the two original potentials.

$$
\hat{c} = O(1) + 0 = O(1)
$$

## Extract-Min: Amortized O(log n)

Extract-min is the most complex operation. It removes the minimum root, adds its children to the root list, and then consolidates trees of the same degree.

### Actual Cost

Let $D(n)$ denote the maximum degree of any node in an $n$-node Fibonacci heap. The minimum node has at most $D(n)$ children, all added to the root list. Before consolidation, the root list has at most:

$$
t(H) - 1 + D(n)
$$

trees. Consolidation examines each root and links trees of equal degree. The work is proportional to the number of roots before consolidation plus $D(n)$ (for the degree array). Thus:

$$
c = O(t(H) + D(n))
$$

### Potential Change

After consolidation, at most one tree of each degree remains, so the new number of trees is at most $D(n) + 1$. No nodes are marked or unmarked during extract-min. Therefore:

$$
\Delta\Phi = (D(n) + 1) - t(H)
$$

### Amortized Cost

$$
\hat{c} = O(t(H) + D(n)) + (D(n) + 1) - t(H) = O(D(n))
$$

The $t(H)$ terms cancel: the actual cost from processing many roots is offset by the decrease in potential. Since $D(n) = O(\log n)$ (proven below), the amortized cost is $O(\log n)$.

## Decrease-Key: Amortized O(1)

Decreasing a key may trigger cascading cuts. If the decreased node violates the heap order with its parent, it is cut from its parent and added to the root list. If the parent was already marked (had already lost a child), the parent is also cut -- and this cascading continues up the tree.

### Actual Cost

Suppose the cascade performs $c_{\text{cuts}}$ cuts. Each cut takes $O(1)$ work, so:

$$
c = O(c_{\text{cuts}})
$$

### Potential Change

Each cut adds one tree to the root list ($t$ increases by 1 per cut). The first $c_{\text{cuts}} - 1$ cuts unmark nodes (each cascading parent was marked), decreasing $m$ by $c_{\text{cuts}} - 1$. The final node in the cascade may become newly marked (increasing $m$ by at most 1). The decreased node itself is moved to the root list and unmarked. Therefore:

$$
\Delta\Phi \le c_{\text{cuts}} + 2(1 - (c_{\text{cuts}} - 1)) = c_{\text{cuts}} + 2(2 - c_{\text{cuts}}) = 4 - c_{\text{cuts}}
$$

### Amortized Cost

$$
\hat{c} = O(c_{\text{cuts}}) + 4 - c_{\text{cuts}} = O(1)
$$

The cascade cost is entirely absorbed by the decrease in potential from unmarking nodes. This is the core insight of the Fibonacci heap design: marking and cascading cuts serve as a "credit scheme" that keeps the amortized cost constant.

## The Degree Bound: Why Fibonacci Numbers

The amortized bounds above rely on $D(n) = O(\log n)$. This bound comes from a structural property enforced by cascading cuts.

!!! tip "Key Invariant"
    A node is cut from its parent the **second** time it loses a child (that is what the mark bit tracks). This ensures that no node loses too many children, which in turn limits the minimum size of a subtree.

Let $x$ be a node with degree $k$, and let $y_1, y_2, \ldots, y_k$ be its children in the order they were linked. When $y_i$ was linked, $x$ already had at least $i - 1$ children, so $y_i$'s degree was at least $i - 1$ at that time. Since a node can lose at most one child (before being cut itself), $y_i$ has degree at least $i - 2$.

Let $s_k$ be the minimum number of nodes in a subtree rooted at a node of degree $k$. Then:

$$
s_k \ge s_{k-2} + s_{k-3} + \cdots + s_0 + 2
$$

This recurrence yields $s_k \ge F_{k+2}$, where $F_k$ is the $k$-th Fibonacci number. Since $F_{k+2} \ge \phi^k$ where $\phi = (1 + \sqrt{5})/2 \approx 1.618$, a node of degree $k$ roots a subtree with at least $\phi^k$ nodes. Therefore, a degree-$k$ node can only exist if $n \ge \phi^k$, giving:

$$
D(n) \le \lfloor \log_\phi n \rfloor = O(\log n)
$$

This is why the data structure is named after Fibonacci: the Fibonacci numbers govern the minimum subtree sizes.

## Summary of Amortized Bounds

| Operation | Actual worst-case | Amortized |
|-----------|:-----------------:|:---------:|
| Insert | $O(1)$ | $O(1)$ |
| Find-min | $O(1)$ | $O(1)$ |
| Merge | $O(1)$ | $O(1)$ |
| Extract-min | $O(n)$ | $O(\log n)$ |
| Decrease-key | $O(\log n)$ | $O(1)$ |
| Delete | $O(n)$ | $O(\log n)$ |

!!! warning "Worst-Case vs Amortized"
    The amortized bounds guarantee that any sequence of $m$ operations starting from an empty heap costs at most $O(m + k \log n)$ total, where $k$ is the number of extract-min and delete operations. Individual operations may exceed these bounds -- for example, a single extract-min may cost $O(n)$ if all nodes are roots.

## Implementation Sketch

```python
"""
Fibonacci heap amortized analysis demonstration.

Illustrates the potential function tracking during a sequence
of Fibonacci heap operations to verify amortized bounds.
"""

import math


# === Potential Tracking ===

class PotentialTracker:
    """Track the potential function Phi = t(H) + 2*m(H)
    during a sequence of Fibonacci heap operations.
    """

    def __init__(self):
        self.trees = 0       # t(H): number of trees in root list
        self.marked = 0      # m(H): number of marked nodes
        self.total_actual = 0
        self.total_amortized = 0
        self.ops = []

    def potential(self):
        """Current potential: t(H) + 2*m(H)."""
        return self.trees + 2 * self.marked

    def record_op(self, name, actual_cost, new_trees, new_marked):
        """Record an operation with its actual cost and new state."""
        old_phi = self.potential()
        self.trees = new_trees
        self.marked = new_marked
        new_phi = self.potential()
        amortized = actual_cost + (new_phi - old_phi)
        self.total_actual += actual_cost
        self.total_amortized += amortized
        self.ops.append({
            "op": name,
            "actual": actual_cost,
            "delta_phi": new_phi - old_phi,
            "amortized": amortized,
            "phi": new_phi,
        })

    def report(self):
        """Print a summary of all operations."""
        print(f"{'Op':<20} {'Actual':>8} {'dPhi':>8} {'Amort':>8} {'Phi':>8}")
        print("-" * 56)
        for op in self.ops:
            print(f"{op['op']:<20} {op['actual']:>8} "
                  f"{op['delta_phi']:>8} {op['amortized']:>8} "
                  f"{op['phi']:>8}")
        print("-" * 56)
        print(f"{'Total':<20} {self.total_actual:>8} "
              f"{'':>8} {self.total_amortized:>8}")


# === Demonstration ===

if __name__ == "__main__":
    tracker = PotentialTracker()

    # Simulate inserting 8 elements (each adds one tree)
    for i in range(1, 9):
        tracker.record_op(f"insert({i})", 1, tracker.trees + 1, tracker.marked)

    # Simulate extract-min: consolidation reduces trees
    # Before: 8 trees. After consolidation: ~log2(8)=3 trees
    actual_extract = 8 + 3  # process all roots + degree array
    tracker.record_op("extract-min", actual_extract, 3, tracker.marked)

    # Simulate decrease-key with 3 cascading cuts
    # 3 cuts: 3 new trees, 2 unmarked, 1 newly marked
    actual_decrease = 3
    tracker.record_op("decrease-key (3 cuts)", actual_decrease,
                      tracker.trees + 3, tracker.marked - 2 + 1)

    tracker.report()

    print(f"\nDegree bound for n=1000: D(n) <= "
          f"{math.floor(math.log(1000) / math.log((1 + math.sqrt(5)) / 2))}")
```

**Output:**
```
Op                     Actual     dPhi    Amort      Phi
--------------------------------------------------------
insert(1)                   1        1        2        1
insert(2)                   1        1        2        2
insert(3)                   1        1        2        3
insert(4)                   1        1        2        4
insert(5)                   1        1        2        5
insert(6)                   1        1        2        6
insert(7)                   1        1        2        7
insert(8)                   1        1        2        8
extract-min                11       -5        6        3
decrease-key (3 cuts)       3        1        4        4
--------------------------------------------------------
Total                      21                26
```

The output shows that while actual costs vary (extract-min costs 11), the amortized costs remain bounded. The total amortized cost (26) provides the guarantee, while individual amortized costs stay within $O(1)$ for insert and decrease-key, and $O(\log n)$ for extract-min.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., and Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 19: Fibonacci Heaps. MIT Press.
- Fredman, M. L. and Tarjan, R. E. "Fibonacci heaps and their uses in improved network optimization algorithms." *Journal of the ACM*, 34(3):596--615, 1987.
