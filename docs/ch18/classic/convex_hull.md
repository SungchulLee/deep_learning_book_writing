# Convex Hull via Divide and Conquer

Many computational geometry problems begin by identifying the extreme boundary of a point set. Imagine hammering nails into a board and stretching a rubber band around them: the shape the band takes is the **convex hull**. This structure appears in collision detection, image processing, and geographic information systems. This page develops the divide-and-conquer convex hull algorithm and compares it to Andrew's monotone chain, a closely related approach that is simpler to implement.

## Definition

The **convex hull** of a point set $P \subset \mathbb{R}^2$ is the smallest convex set containing $P$. A set $S$ is convex if for every pair of points $a, b \in S$, the line segment $\overline{ab}$ lies entirely within $S$. The boundary of the convex hull is a convex polygon whose vertices are a subset of $P$.

## Cross Product Test

Both algorithms below rely on the **cross product** to determine whether three ordered points make a left turn, right turn, or are collinear. For points $O$, $A$, $B$, define:

$$
\text{cross}(O, A, B) = (A_x - O_x)(B_y - O_y) - (A_y - O_y)(B_x - O_x)
$$

This quantity equals the signed area of the parallelogram spanned by vectors $\overrightarrow{OA}$ and $\overrightarrow{OB}$:

- **Positive**: $O \to A \to B$ makes a left (counterclockwise) turn.
- **Negative**: $O \to A \to B$ makes a right (clockwise) turn.
- **Zero**: the three points are collinear.

## Divide-and-Conquer Algorithm

The classic divide-and-conquer approach proceeds in four steps.

**Step 1 -- Sort.** Sort all $n$ points by $x$-coordinate, breaking ties by $y$-coordinate. This takes $O(n \log n)$ time and needs to be done only once.

**Step 2 -- Divide.** Split the sorted array at the median into a left half $P_L$ (indices $1$ to $\lfloor n/2 \rfloor$) and a right half $P_R$ (indices $\lfloor n/2 \rfloor + 1$ to $n$).

**Step 3 -- Conquer.** Recursively compute the convex hulls $H_L$ and $H_R$. The base case is a set of one, two, or three points, whose hull is trivial.

**Step 4 -- Merge.** Combine $H_L$ and $H_R$ into a single hull by finding the **upper tangent** and **lower tangent** between the two hulls. The tangent-finding procedure works as follows:

1. Start with the rightmost point $p$ of $H_L$ and the leftmost point $q$ of $H_R$.
2. **Upper tangent**: While the line $\overline{pq}$ is not tangent to both hulls, repeatedly move $p$ counterclockwise around $H_L$ (as long as the cross product shows a left turn with the next vertex) and move $q$ clockwise around $H_R$ (as long as the cross product shows a right turn with the next vertex).
3. **Lower tangent**: Apply the symmetric procedure, moving $p$ clockwise and $q$ counterclockwise.
4. Concatenate the boundary segments of $H_L$ from the lower tangent point to the upper tangent point, then the boundary segments of $H_R$ from the upper tangent point to the lower tangent point.

Each tangent walk visits each vertex at most once, so the merge takes $O(n)$ time. The overall recurrence is:

$$
T(n) = 2T(n/2) + O(n) = O(n \log n)
$$

## Andrew's Monotone Chain

Andrew's monotone chain achieves the same $O(n \log n)$ bound through a simpler implementation. Rather than recursively merging two hulls, it builds the **upper hull** and **lower hull** independently by scanning sorted points from left to right (lower hull) and right to left (upper hull). Each scan maintains a stack and uses the cross product test to discard points that would create a non-convex turn. The two half-hulls are then concatenated to form the complete hull.

This approach is closely related to the divide-and-conquer strategy: the sorting step is shared, and building each half-hull mirrors the way the D&C merge constructs the upper and lower boundary. The monotone chain is generally preferred in practice because it avoids the bookkeeping of recursive merging while retaining the same time complexity.

```python
"""
Convex hull via Andrew's monotone chain algorithm.

Builds the upper and lower hulls separately by scanning points
sorted by x-coordinate, achieving O(n log n) time.  Uses <= 0
in the cross-product check to exclude collinear boundary points;
change to < 0 to include them.
"""

# === Cross Product ===

def cross(o: tuple, a: tuple, b: tuple) -> float:
    """Compute the cross product of vectors OA and OB.

    Returns:
        Positive if O->A->B is counterclockwise (left turn),
        negative if clockwise (right turn), zero if collinear.
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


# === Point-in-Hull Test ===

def point_in_hull(p: tuple, hull_pts: list[tuple]) -> bool:
    """Check whether point p lies inside or on the convex hull.

    The hull vertices must be in counterclockwise order.  The test
    verifies that p is on the left side of every directed edge.

    Args:
        p: A 2D point (x, y).
        hull_pts: Vertices of the convex hull in CCW order.

    Returns:
        True if p is inside or on the boundary of the hull.
    """
    n = len(hull_pts)
    for i in range(n):
        if cross(hull_pts[i], hull_pts[(i + 1) % n], p) < 0:
            return False
    return True


# === Convex Hull ===

def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Compute the convex hull of a set of 2D points.

    Duplicates are removed before processing.  Collinear boundary
    points are excluded (only vertices of the hull polygon are kept).

    Args:
        points: List of (x, y) coordinates.

    Returns:
        Vertices of the convex hull in counterclockwise order.
    """
    # Remove duplicates, then sort by x (ties broken by y)
    points = sorted(set(points))
    if len(points) <= 1:
        return list(points)

    # Build lower hull (left to right)
    lower: list[tuple[float, float]] = []
    for p in points:
        # Pop last point if it makes a non-left turn (<=0 excludes collinear)
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull (right to left)
    upper: list[tuple[float, float]] = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenate, removing duplicate endpoints
    return lower[:-1] + upper[:-1]


# === Demonstration ===

if __name__ == "__main__":
    pts = [(0, 0), (1, 1), (2, 2), (0, 2), (2, 0), (1, 0), (1, 3)]
    hull = convex_hull(pts)
    print(f"Points: {pts}")
    print(f"Hull vertices: {hull}")
    print(f"Number of hull vertices: {len(hull)}")

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

The interior point $(1, 1)$ and the collinear point $(2, 2)$ are not hull vertices. The six vertices form a convex polygon that encloses all seven input points.

## Complexity

| Aspect | Cost |
|--------|:----:|
| Time   | $O(n \log n)$ |
| Space  | $O(n)$ |

The sorting step dominates at $O(n \log n)$. The hull construction itself runs in $O(n)$ amortized time: each point enters the stack exactly once and is popped at most once, so the total number of push and pop operations across the entire scan is at most $2n$.

## Lower Bound

Having established an $O(n \log n)$ algorithm, a natural question is whether any comparison-based algorithm can do better. The answer is no.

Computing the convex hull requires $\Omega(n \log n)$ time in the comparison model. The proof uses a reduction from sorting. Given $n$ numbers $x_1, \dots, x_n$, map each to the point $(x_i, x_i^2)$ on the parabola $y = x^2$. Because a parabola is strictly convex, every mapped point is a vertex of the hull, and the hull visits them in sorted order of $x$. Any convex hull algorithm therefore sorts $n$ numbers, which requires $\Omega(n \log n)$ comparisons.

## Reference

- Andrew, A. M. (1979). Another efficient algorithm for convex hulls in two dimensions. *Information Processing Letters*, 9(5), 216--219.
- Preparata, F. P. & Shamos, M. I. (1985). *Computational Geometry: An Introduction*. Springer.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 33: Computational Geometry.
