# Nearest Neighbor Search in KD-Trees

Finding the point closest to a query point is one of the most common operations in computational geometry, machine learning (k-nearest neighbors), and spatial databases.  A naive search compares the query against all $n$ points in $O(n)$ time.  A [kd-tree](construction.md) reduces this to $O(\log n)$ on average by using the tree structure to prune large portions of the search space.  The key insight is that if the closest point found so far is closer than the distance to a splitting hyperplane, the entire subtree on the far side of that hyperplane can be skipped.

## Algorithm

The nearest-neighbor search in a kd-tree is a recursive procedure that maintains a **current best** point and distance.  At each node:

1. **Compute the distance** from the query point $q$ to the current node's point $p$.  If this distance is less than the current best, update the best.
2. **Determine which child to visit first.** Compare $q$'s coordinate along the splitting axis to $p$'s coordinate.  Visit the child on the same side as $q$ first (the "near" child), because the nearest neighbor is more likely to be there.
3. **Recurse into the near child.**
4. **Check whether the far child needs to be visited.** Compute the distance from $q$ to the splitting hyperplane.  If this distance is less than the current best distance, the far subtree might contain a closer point — recurse into it.  Otherwise, prune the far subtree entirely.

## Pseudocode

```
NN-SEARCH(node, query, best, best_dist):
    if node is nil:
        return (best, best_dist)

    dist = distance(query, node.point)
    if dist < best_dist:
        best = node.point
        best_dist = dist

    axis = node.axis
    diff = query[axis] - node.point[axis]

    if diff <= 0:
        near, far = node.left, node.right
    else:
        near, far = node.right, node.left

    (best, best_dist) = NN-SEARCH(near, query, best, best_dist)

    if |diff| < best_dist:             # far side might have closer point
        (best, best_dist) = NN-SEARCH(far, query, best, best_dist)

    return (best, best_dist)
```

## Pruning Condition

The pruning step is the core of the algorithm's efficiency.  The splitting hyperplane at a node divides space into two half-spaces.  The minimum distance from the query point $q$ to any point in the far half-space is:

$$
d_{\text{hyperplane}} = |q[\text{axis}] - p[\text{axis}]|
$$

If $d_{\text{hyperplane}} \ge d_{\text{best}}$, every point in the far subtree is at least $d_{\text{hyperplane}}$ away along one coordinate, so by the triangle inequality, it cannot be closer than the current best.

??? example "Nearest neighbor search in 2D"
    Consider points: $(2,3), (5,4), (9,6), (4,7), (8,1), (7,2)$ with query $q = (6, 3)$.

    The kd-tree (built with round-robin splitting) splits first on $x$, then $y$, etc.

    1. Start at root $(7,2)$ (axis=0). Distance to $q$: $\sqrt{(6-7)^2 + (3-2)^2} = \sqrt{2} \approx 1.41$.  Best = $(7,2)$, best\_dist = 1.41.
    2. $q[0] = 6 < 7$, so go left first.
    3. Visit $(5,4)$ (axis=1). Distance = $\sqrt{1+1} = \sqrt{2} \approx 1.41$.  Tie — keep current best.
    4. Continue recursing...eventually the algorithm finds $(7,2)$ as the nearest neighbor.

## Implementation

```python
"""Nearest neighbor search in a kd-tree."""

from __future__ import annotations

import math


# === Node Definition ===

class KDNode:
    """KD-tree node with point and splitting axis."""

    def __init__(self, point: list[float], axis: int):
        self.point = point
        self.axis = axis
        self.left: KDNode | None = None
        self.right: KDNode | None = None


# === Construction (from construction.md) ===

def build_kdtree(points: list[list[float]], depth: int = 0) -> KDNode | None:
    """Build a balanced kd-tree."""
    if not points:
        return None
    k = len(points[0])
    axis = depth % k
    points.sort(key=lambda p: p[axis])
    mid = len(points) // 2
    node = KDNode(points[mid], axis)
    node.left = build_kdtree(points[:mid], depth + 1)
    node.right = build_kdtree(points[mid + 1:], depth + 1)
    return node


# === Nearest Neighbor Search ===

def nearest_neighbor(node: KDNode | None, query: list[float],
                     best: list[float] | None = None,
                     best_dist: float = math.inf
                     ) -> tuple[list[float] | None, float]:
    """Find the nearest point to *query* in the kd-tree."""
    if node is None:
        return best, best_dist

    dist = math.sqrt(sum((q - p) ** 2 for q, p in zip(query, node.point)))
    if dist < best_dist:
        best, best_dist = node.point, dist

    axis = node.axis
    diff = query[axis] - node.point[axis]
    near = node.left if diff <= 0 else node.right
    far = node.right if diff <= 0 else node.left

    best, best_dist = nearest_neighbor(near, query, best, best_dist)

    if abs(diff) < best_dist:
        best, best_dist = nearest_neighbor(far, query, best, best_dist)

    return best, best_dist


# === Demonstration ===

if __name__ == "__main__":
    pts = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    root = build_kdtree(pts)

    query = [6, 3]
    result, dist = nearest_neighbor(root, query)
    print(f"Nearest to {query}: {result} (distance={dist:.3f})")
```

## Complexity

| Metric | Average case | Worst case |
|--------|-------------|------------|
| Time | $O(\log n)$ | $O(n)$ |
| Space | $O(\log n)$ stack | $O(n)$ stack |

The average $O(\log n)$ bound assumes reasonably distributed points.  The worst case $O(n)$ occurs when the pruning condition fails at most nodes (e.g., when the query point is far from all stored points and the tree is poorly balanced).

!!! warning "Curse of dimensionality"
    In high dimensions ($k \gg \log n$), the pruning condition fails frequently because distances to hyperplanes become small relative to distances to points.  For $k \gtrsim 20$, kd-tree nearest-neighbor search degrades toward $O(n)$, and approximate methods (e.g., locality-sensitive hashing) are preferred.

## K-Nearest Neighbors

To find the $k$ nearest neighbors instead of just one, replace the single best point with a **max-heap of size $k$**.  The pruning condition uses the distance to the farthest point in the heap (the heap root).  When the heap has fewer than $k$ elements, always explore both subtrees.

## Reference

- Friedman, J. H., Bentley, J. L., & Finkel, R. A. (1977). An algorithm for finding best matches in logarithmic expected time. *ACM Transactions on Mathematical Software*, 3(3), 209–226.
- de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M. (2008). *Computational Geometry: Algorithms and Applications* (3rd ed.), Chapter 5. Springer.
