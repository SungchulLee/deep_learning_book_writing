# Range Search in KD-Trees

A **range query** asks: "which points from the stored set lie inside a given region?"  For axis-aligned rectangular regions in $k$ dimensions, a [kd-tree](construction.md) answers this query efficiently by pruning subtrees whose bounding regions do not intersect the query rectangle.  In 2D, a balanced kd-tree reports all $r$ points inside a rectangular range in $O(\sqrt{n} + r)$ time, a significant improvement over the $O(n)$ brute-force scan when the result set is small.

## Query Region

The query is an axis-aligned rectangle (hyperrectangle in $k$ dimensions) defined by lower and upper bounds in each dimension:

$$
R = [x_1^{lo}, x_1^{hi}] \times [x_2^{lo}, x_2^{hi}] \times \cdots \times [x_k^{lo}, x_k^{hi}]
$$

A point $p$ is reported if $x_i^{lo} \le p[i] \le x_i^{hi}$ for all dimensions $i = 1, \ldots, k$.

## Algorithm

The range search traverses the kd-tree recursively, using three cases at each node:

1. **Node's region is entirely inside the query:** report all points in the subtree.
2. **Node's region does not intersect the query:** prune the entire subtree.
3. **Partial overlap:** check whether the current node's point is in the range, then recurse into both children.

At each node, the splitting hyperplane determines whether the query range intersects the left subtree, the right subtree, or both.

```
RANGE-SEARCH(node, query_range):
    if node is nil:
        return

    if node.point is inside query_range:
        report node.point

    axis = node.axis
    if query_range.low[axis] <= node.point[axis]:
        RANGE-SEARCH(node.left, query_range)
    if query_range.high[axis] >= node.point[axis]:
        RANGE-SEARCH(node.right, query_range)
```

## Pruning Condition

The splitting hyperplane at a node with point $p$ and axis $d$ divides space at $p[d]$.  The left subtree contains all points with coordinate $\le p[d]$ along axis $d$; the right subtree contains points with coordinate $> p[d]$.

- **Skip the left subtree** if $q_{low}[d] > p[d]$ (the query range starts after the split).
- **Skip the right subtree** if $q_{high}[d] < p[d]$ (the query range ends before the split).

When both conditions fail (the split value falls within the query range), the algorithm must explore both subtrees.

## Implementation

```python
"""Range search in a kd-tree."""

from __future__ import annotations


# === Node Definition ===

class KDNode:
    """KD-tree node with point and splitting axis."""

    def __init__(self, point: list[float], axis: int):
        self.point = point
        self.axis = axis
        self.left: KDNode | None = None
        self.right: KDNode | None = None


# === Construction ===

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


# === Range Search ===

def range_search(node: KDNode | None,
                 low: list[float], high: list[float]) -> list[list[float]]:
    """Find all points inside the axis-aligned rectangle [low, high]."""
    result: list[list[float]] = []
    _range_helper(node, low, high, result)
    return result


def _range_helper(node: KDNode | None,
                  low: list[float], high: list[float],
                  result: list[list[float]]) -> None:
    """Recursive helper for range search."""
    if node is None:
        return

    # Check if the current point is in the range
    if all(lo <= p <= hi for lo, p, hi in zip(low, node.point, high)):
        result.append(node.point)

    axis = node.axis

    # Recurse into subtrees that might intersect the range
    if low[axis] <= node.point[axis]:
        _range_helper(node.left, low, high, result)
    if high[axis] >= node.point[axis]:
        _range_helper(node.right, low, high, result)


# === Demonstration ===

if __name__ == "__main__":
    pts = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    root = build_kdtree(pts)

    low, high = [3, 2], [8, 5]
    found = range_search(root, low, high)
    print(f"Points in [{low}, {high}]: {found}")
    # Expected: [5, 4], [7, 2] (and possibly others in range)
```

## Complexity

The complexity of range search in a balanced kd-tree depends on the number of dimensions:

| Dimensions | Query time | Space |
|------------|-----------|-------|
| 2D | $O(\sqrt{n} + r)$ | $O(n)$ |
| $k$-D | $O(n^{1 - 1/k} + r)$ | $O(n)$ |

where $r$ is the number of reported points.

The $O(\sqrt{n})$ term in 2D arises because at each level of the tree, the query range crosses at most 2 splitting lines (on the relevant axis), and the tree has $O(\log n)$ levels.  A careful analysis shows that the number of nodes visited is bounded by $O(\sqrt{n})$ when $r = 0$.

!!! note "Lower bound"
    The $O(\sqrt{n} + r)$ bound for 2D range queries is tight for kd-trees — there exist query configurations that require visiting $\Omega(\sqrt{n})$ nodes.  For faster range queries, consider **range trees** which achieve $O(\log^2 n + r)$ (or $O(\log n + r)$ with fractional cascading) at the cost of $O(n \log n)$ space.

## Circular Range Queries

For non-rectangular regions (e.g., "find all points within distance $d$ of query point $q$"), use the bounding rectangle of the circle as the query range, and add a post-filtering step that checks the actual distance for each candidate point.  The kd-tree still provides efficient pruning.

## Reference

- Bentley, J. L. (1975). Multidimensional binary search trees used for associative searching. *Communications of the ACM*, 18(9), 509–517.
- de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M. (2008). *Computational Geometry: Algorithms and Applications* (3rd ed.), Chapter 5. Springer.
