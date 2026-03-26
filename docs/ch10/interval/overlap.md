# Overlap Queries

The primary operation supported by an [interval tree](structure.md) is the **overlap query**: given a query interval $[q_{low}, q_{high}]$, find an interval in the tree that overlaps with it.  The [augmented max-endpoint field](augmented.md) stored at each node makes this possible in $O(\log n)$ time by guiding the search toward subtrees that may contain overlapping intervals and pruning those that cannot.

## When Do Two Intervals Overlap?

Two closed intervals $[a, b]$ and $[c, d]$ overlap if and only if:

$$
a \le d \quad \text{and} \quad c \le b
$$

Equivalently, they do **not** overlap when $b < c$ or $d < a$ (one ends before the other begins).

## The Overlap Search Algorithm

Given a query interval $q = [q_{low}, q_{high}]$ and an interval tree rooted at $x$:

```
INTERVAL-SEARCH(T, q):
    x = T.root
    while x != T.nil and q does not overlap x.interval:
        if x.left != T.nil and x.left.max >= q.low:
            x = x.left
        else:
            x = x.right
    return x
```

The algorithm walks down from the root, choosing left or right at each node.  It terminates either when an overlapping interval is found or when $x$ reaches a nil node (no overlap exists).

## Decision Rule

At each internal node $x$, the algorithm checks:

**Go left if** $x.left \ne \text{nil}$ and $x.left.max \ge q_{low}$.

**Go right otherwise.**

The intuition: if the left subtree's maximum endpoint is at least $q_{low}$, then some interval in the left subtree extends far enough to the right that it *might* overlap with $q$.  If the left subtree's max is less than $q_{low}$, then no interval in the left subtree can overlap with $q$ (they all end before $q$ starts), so the algorithm goes right.

## Correctness

!!! note "Correctness theorem"
    If the algorithm goes left, then either the left subtree contains an overlapping interval, or no interval in the entire tree overlaps with $q$.  If the algorithm goes right, then the left subtree contains no overlapping interval.

**Proof of the right-go case.** If $x.left.max < q_{low}$, then every interval $[a, b]$ in the left subtree satisfies $b \le x.left.max < q_{low}$, so $b < q_{low}$ and no overlap is possible. $\square$

**Proof of the left-go case.** Suppose the algorithm goes left because $x.left.max \ge q_{low}$.  Let $[a, b]$ be the interval in the left subtree that achieves the max endpoint ($b = x.left.max$).  If $q$ does not overlap $[a, b]$, then $q_{high} < a$.  Since all intervals in the left subtree have low endpoints $\le a$ (by BST order on low endpoints... actually, $a$ is just some interval's low endpoint; but the BST ordering means all intervals in the right subtree have low endpoints $\ge x.key$).  Since $q_{high} < a \le x.low$ (the node's interval low endpoint), and all right-subtree intervals have low endpoints $\ge x.low \ge a > q_{high}$, no right-subtree interval overlaps $q$ either. $\square$

## Finding All Overlapping Intervals

The basic algorithm returns a single overlapping interval.  To find **all** $k$ intervals overlapping with $q$, modify the search to explore both subtrees when overlap is possible:

1. If $x$'s interval overlaps $q$, report $x$.
2. If $x.left \ne \text{nil}$ and $x.left.max \ge q_{low}$, recurse into the left subtree.
3. If $x.right \ne \text{nil}$ and $x.right.max \ge q_{low}$ and $x.key \le q_{high}$ (the right subtree has intervals starting before $q$ ends), recurse into the right subtree.

This variant runs in $O(k \log n)$ time, where $k$ is the number of overlapping intervals.  More sophisticated structures (e.g., augmented interval trees with sorted lists at each node) can achieve $O(\log n + k)$.

## Implementation

```python
"""Interval tree with overlap query."""

from __future__ import annotations


# === Node Definition ===

class IntervalNode:
    """Interval tree node storing [low, high] with augmented max."""

    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high
        self.max = high
        self.left: IntervalNode | None = None
        self.right: IntervalNode | None = None


# === Insert ===

def insert(root: IntervalNode | None, low: int, high: int) -> IntervalNode:
    """Insert interval [low, high] into the interval tree."""
    if root is None:
        return IntervalNode(low, high)
    if low < root.low:
        root.left = insert(root.left, low, high)
    else:
        root.right = insert(root.right, low, high)
    root.max = max(root.max, high)
    return root


# === Overlap Search ===

def overlap_search(root: IntervalNode | None,
                   q_low: int, q_high: int) -> IntervalNode | None:
    """Find an interval overlapping [q_low, q_high], or None."""
    x = root
    while x is not None:
        if x.low <= q_high and q_low <= x.high:
            return x  # overlap found
        if x.left is not None and x.left.max >= q_low:
            x = x.left
        else:
            x = x.right
    return None


# === Demonstration ===

if __name__ == "__main__":
    root: IntervalNode | None = None
    intervals = [(15, 20), (10, 30), (17, 19), (5, 20), (12, 15), (30, 40)]
    for lo, hi in intervals:
        root = insert(root, lo, hi)

    query = (14, 16)
    result = overlap_search(root, *query)
    if result:
        print(f"Query {query} overlaps [{result.low}, {result.high}]")
    else:
        print(f"Query {query}: no overlap found")
```

## Complexity

| Operation | Time |
|-----------|------|
| Single overlap query | $O(\log n)$ |
| Find all $k$ overlaps | $O(k \log n)$ |
| Insert | $O(\log n)$ |
| Delete | $O(\log n)$ |

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Section 14.3. MIT Press.
- de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M. (2008). *Computational Geometry: Algorithms and Applications* (3rd ed.), Chapter 10. Springer.
