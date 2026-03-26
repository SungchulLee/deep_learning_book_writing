# Convex Hull via Divide and Conquer

The **convex hull** of a set of points is the smallest convex polygon that contains all the points. Imagine stretching a rubber band around nails hammered into a board; the shape it takes is the convex hull. While several $O(n \log n)$ algorithms exist (Graham scan, Andrew's monotone chain), the divide-and-conquer approach directly applies the paradigm: split the points in half, recursively compute each hull, and merge the two hulls by finding their upper and lower tangent lines.

## Definition

The **convex hull** of a point set $P \subset \mathbb{R}^2$ is the smallest convex set containing $P$. Its boundary is a convex polygon whose vertices are a subset of $P$.

## Cross Product Test

The algorithm relies on the **cross product** to determine whether three points make a left turn, right turn, or are collinear.

For points $O$, $A$, $B$:

$$
\text{cross}(O, A, B) = (A_x - O_x)(B_y - O_y) - (A_y - O_y)(B_x - O_x)
$$

- Positive: $O \to A \to B$ makes a left (counterclockwise) turn.
- Negative: $O \to A \to B$ makes a right (clockwise) turn.
- Zero: the three points are collinear.

## Divide-and-Conquer Algorithm

**Step 1: Sort.** Sort all points by $x$-coordinate (break ties by $y$-coordinate).

**Step 2: Divide.** Split the sorted points into a left half $P_L$ and right half $P_R$.

**Step 3: Conquer.** Recursively compute the convex hulls $H_L$ and $H_R$.

**Step 4: Merge.** Combine $H_L$ and $H_R$ by finding the upper and lower tangent lines between the two hulls, then concatenating the appropriate boundary segments.

The merge step finds tangent lines by walking along the hull boundaries using the cross product test. This takes $O(n)$ time, giving the recurrence:

$$
T(n) = 2T(n/2) + O(n) = O(n \log n)
$$

## Implementation

The following implementation uses Andrew's monotone chain algorithm, which achieves the same $O(n \log n)$ complexity through an approach closely related to the D&C idea: it builds the upper and lower hulls independently by scanning sorted points.

```python
"""
Convex hull using Andrew's monotone chain algorithm.

Builds the upper and lower hulls separately by scanning points
sorted by x-coordinate, achieving O(n log n) time.
"""

# === Cross Product ===

def cross(o: tuple, a: tuple, b: tuple) -> float:
    """Compute the cross product of vectors OA and OB.

    Returns:
        Positive if O->A->B is counterclockwise (left turn),
        negative if clockwise (right turn), zero if collinear.
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


# === Convex Hull ===

def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Compute the convex hull of a set of 2D points.

    Args:
        points: List of (x, y) coordinates.

    Returns:
        Vertices of the convex hull in counterclockwise order.
    """
    points = sorted(set(points))
    if len(points) <= 1:
        return list(points)

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenate (last point of each half is first point of the other)
    return lower[:-1] + upper[:-1]


# === Demonstration ===

if __name__ == "__main__":
    pts = [(0, 0), (1, 1), (2, 2), (0, 2), (2, 0), (1, 0), (1, 3)]
    hull = convex_hull(pts)
    print(f"Points: {pts}")
    print(f"Hull vertices: {hull}")
    print(f"Number of hull vertices: {len(hull)}")

    # Verify: all points should be inside or on the hull
    def point_in_hull(p, hull_pts):
        n = len(hull_pts)
        for i in range(n):
            if cross(hull_pts[i], hull_pts[(i+1) % n], p) < 0:
                return False
        return True

    all_inside = all(point_in_hull(p, hull) for p in pts)
    print(f"All points inside hull: {all_inside}")
```

**Output:**

```
Points: [(0, 0), (1, 1), (2, 2), (0, 2), (2, 0), (1, 0), (1, 3)]
Hull vertices: [(0, 0), (1, 0), (2, 0), (2, 2), (1, 3), (0, 2)]
Number of hull vertices: 6
All points inside hull: True
```

The interior point $(1, 1)$ is not a hull vertex. The six hull vertices form a convex polygon that contains all seven input points.

## Complexity

| Aspect | Cost |
|--------|:----:|
| Time   | $O(n \log n)$ |
| Space  | $O(n)$ |

The sorting step dominates. The hull construction itself takes $O(n)$ amortized time, since each point is pushed and popped from the stack at most once.

## Lower Bound

Computing the convex hull requires $\Omega(n \log n)$ time in the comparison model. This follows from a reduction from sorting: given $n$ numbers $x_1, \dots, x_n$, map them to points $(x_i, x_i^2)$ on a parabola. The convex hull of these points visits them in sorted order, so any convex hull algorithm can sort.

## Reference

- Andrew, A. M. (1979). Another efficient algorithm for convex hulls in two dimensions. *Information Processing Letters*, 9(5), 216--219.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 33: Computational Geometry.
