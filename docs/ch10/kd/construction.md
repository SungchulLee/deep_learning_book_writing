# KD-Tree Construction

A **kd-tree** (k-dimensional tree) partitions $k$-dimensional space by recursively splitting along alternating coordinate axes.  Each internal node defines a splitting hyperplane that divides the point set into two halves, and the recursion continues until each leaf contains at most one point.  The construction algorithm produces a balanced tree of height $O(\log n)$ that supports efficient [range queries](range.md) and [nearest-neighbor searches](nearest.md).

## Splitting Strategy

At each level of the tree, the algorithm selects a **splitting dimension** and a **splitting value**:

- **Dimension selection:** the simplest approach cycles through dimensions in round-robin order.  At depth $d$, split along dimension $d \bmod k$.
- **Splitting value:** use the **median** point along the chosen dimension.  This ensures that each split divides the points as evenly as possible, producing a balanced tree.

For $n$ points in $\mathbb{R}^k$, the recursive construction is:

1. If $n = 0$, return a nil node.
2. Choose splitting dimension $d$ (typically depth mod $k$).
3. Find the median point $p$ along dimension $d$.
4. Create a node storing $p$.
5. Recursively build the left subtree from all points with coordinate $< p[d]$.
6. Recursively build the right subtree from all points with coordinate $\ge p[d]$ (excluding $p$).

## Construction Algorithm

```python
"""KD-tree construction from a set of points."""

from __future__ import annotations


# === Node Definition ===

class KDNode:
    """Node in a kd-tree storing a point and splitting dimension."""

    def __init__(self, point: list[float], axis: int):
        self.point = point
        self.axis = axis
        self.left: KDNode | None = None
        self.right: KDNode | None = None


# === Construction ===

def build_kdtree(points: list[list[float]], depth: int = 0) -> KDNode | None:
    """Build a balanced kd-tree from a list of points.

    Points are split along alternating dimensions.  The median
    point on the current axis becomes the root of the subtree.
    """
    if not points:
        return None

    k = len(points[0])  # number of dimensions
    axis = depth % k

    # Sort by the current axis and pick the median
    points.sort(key=lambda p: p[axis])
    median_idx = len(points) // 2

    node = KDNode(points[median_idx], axis)
    node.left = build_kdtree(points[:median_idx], depth + 1)
    node.right = build_kdtree(points[median_idx + 1:], depth + 1)
    return node


# === Demonstration ===

if __name__ == "__main__":
    pts = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    root = build_kdtree(pts)

    def print_tree(node: KDNode | None, indent: int = 0) -> None:
        """Print the kd-tree structure."""
        if node is None:
            return
        print(" " * indent + f"{node.point} (axis={node.axis})")
        print_tree(node.left, indent + 4)
        print_tree(node.right, indent + 4)

    print_tree(root)
```

## Height Analysis

When the median is used as the splitting value, each level halves the point set.  The resulting tree has height:

$$
h = O(\log_2 n)
$$

The tree is perfectly balanced (or within one level of balanced), containing exactly $n$ nodes.

## Construction Cost

| Method | Time | Space |
|--------|------|-------|
| Sorting at each level | $O(n \log^2 n)$ | $O(n)$ |
| Median-of-medians | $O(n \log n)$ | $O(n)$ |
| Pre-sorted lists | $O(n \log n)$ | $O(n \log n)$ |

The $O(n \log^2 n)$ bound comes from sorting $n$ points at each of the $O(\log n)$ levels.  The $O(n \log n)$ bound uses a linear-time median selection algorithm (e.g., median-of-medians) to find the splitting point at each level.

!!! tip "Practical optimization"
    The $O(n \log^2 n)$ sorting-based approach is simple and fast enough for most applications.  For very large point sets, the $O(n \log n)$ method using pre-sorted point lists (one sorted list per dimension) avoids repeated sorting.

## Dimension Selection Variants

The round-robin dimension selection is simple but not always optimal.  Alternative strategies include:

**Maximum-spread axis:** at each node, choose the dimension with the largest range (max - min) of point coordinates.  This tends to produce more balanced spatial partitions.

**Maximum-variance axis:** choose the dimension with the largest variance.  This is common in computational geometry applications.

| Strategy | Pros | Cons |
|----------|------|------|
| Round-robin | Simple, deterministic | May produce elongated cells |
| Max-spread | Better spatial balance | $O(kn)$ extra work per node |
| Max-variance | Adapts to data distribution | $O(kn)$ extra work per node |

## Properties of the Constructed Tree

- **Balanced:** height $O(\log n)$ when using median splits.
- **Space partitioning:** each node implicitly defines an axis-aligned bounding box.
- **No rebalancing:** kd-trees are typically built once from a static point set.  Dynamic insertions and deletions can degrade balance.
- **Leaf size:** the construction can stop early and store multiple points in a leaf (a "bucket kd-tree"), which improves cache performance.

!!! warning "Dynamic kd-trees"
    Inserting or deleting points from a kd-tree without rebuilding can make it highly unbalanced.  For dynamic point sets, consider using a balanced structure like a **kd-B-tree** or rebuild the tree periodically.

## Reference

- Bentley, J. L. (1975). Multidimensional binary search trees used for associative searching. *Communications of the ACM*, 18(9), 509–517.
- de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M. (2008). *Computational Geometry: Algorithms and Applications* (3rd ed.), Chapter 5. Springer.
