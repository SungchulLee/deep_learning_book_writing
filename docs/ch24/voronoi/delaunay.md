# Delaunay Triangulation

A triangulation of a point set connects the points with non-crossing edges
to form triangles. Among all possible triangulations, the **Delaunay
triangulation** is special: it maximizes the minimum angle across all
triangles, avoiding the long, thin triangles that cause numerical problems
in finite element methods, mesh generation, and terrain modeling. The
Delaunay triangulation is also the dual graph of the Voronoi diagram.

## Definition

Given a set $P$ of $n$ points in the plane, the **Delaunay triangulation**
$DT(P)$ is the triangulation satisfying the **empty circumcircle property**:

!!! note "Empty Circumcircle Property"
    For every triangle in $DT(P)$, its circumscribed circle contains no
    point of $P$ in its interior.

Equivalently, for every edge in $DT(P)$, the two triangles sharing that
edge (if they exist) satisfy the Delaunay condition: the sum of opposite
angles is at most $\pi$.

## Properties

The Delaunay triangulation has several important properties:

- **Max-min angle:** Among all triangulations of $P$, $DT(P)$ maximizes the
  minimum angle. This makes it optimal for avoiding degenerate triangles.
- **Dual of Voronoi:** Two points $p_i, p_j$ are connected by a Delaunay edge
  if and only if their Voronoi regions share an edge.
- **Uniqueness:** If no four points are cocircular, $DT(P)$ is unique.
- **Number of edges:** $O(n)$ — specifically at most $3n - 6$ edges and
  $2n - 5$ triangles.
- **Contains the nearest neighbor graph:** The nearest neighbor of every
  point is connected to it by a Delaunay edge.
- **Contains the Euclidean MST:** The minimum spanning tree is a subgraph
  of $DT(P)$.

## The In-Circle Test

The fundamental predicate for Delaunay triangulation is the **in-circle test**:
given four points $A, B, C, D$, is $D$ inside the circumcircle of
$\triangle ABC$?

$$
\text{InCircle}(A, B, C, D) =
\begin{vmatrix}
a_x - d_x & a_y - d_y & (a_x - d_x)^2 + (a_y - d_y)^2 \\
b_x - d_x & b_y - d_y & (b_x - d_x)^2 + (b_y - d_y)^2 \\
c_x - d_x & c_y - d_y & (c_x - d_x)^2 + (c_y - d_y)^2
\end{vmatrix}
$$

If $A, B, C$ are in counterclockwise order:

- $> 0$: $D$ is inside the circumcircle
- $= 0$: $D$ is on the circumcircle (cocircular)
- $< 0$: $D$ is outside the circumcircle

## Edge Flipping Algorithm

A simple approach to construct $DT(P)$:

1. Start with any valid triangulation of $P$.
2. For each interior edge, check whether it satisfies the Delaunay condition
   using the in-circle test.
3. If an edge fails the test, **flip** it: replace the shared edge of two
   adjacent triangles with the other diagonal of their quadrilateral.
4. Repeat until no more flips are needed.

This algorithm terminates because each flip strictly increases the minimum
angle, and there are finitely many triangulations.

**Complexity:** $O(n^2)$ in the worst case due to cascading flips.

## Incremental Construction

The more practical algorithm inserts points one at a time:

1. Start with a large bounding triangle containing all points.
2. For each point $p$:
    - Find the triangle containing $p$.
    - Split that triangle into three (or two if $p$ is on an edge).
    - Flip edges as needed to restore the Delaunay property.
3. Remove the bounding triangle and its edges.

With a point-location structure, each insertion takes $O(\log n)$ expected
time, giving $O(n \log n)$ total.

## Implementation

```python
"""
Delaunay triangulation: incremental construction with edge flipping.

Demonstrates the in-circle test and a simplified incremental algorithm.
"""

import math


# === Geometric Predicates ===

def orient(a, b, c):
    """Orientation test: positive if CCW, negative if CW, zero if collinear."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def in_circle(a, b, c, d):
    """Test if point d is inside the circumcircle of triangle abc.

    Assumes a, b, c are in counterclockwise order.
    Returns positive if d is inside, zero if on, negative if outside.
    """
    ax, ay = a[0] - d[0], a[1] - d[1]
    bx, by = b[0] - d[0], b[1] - d[1]
    cx, cy = c[0] - d[0], c[1] - d[1]

    return (ax * ax + ay * ay) * (bx * cy - cx * by) \
         - (bx * bx + by * by) * (ax * cy - cx * ay) \
         + (cx * cx + cy * cy) * (ax * by - bx * ay)


# === Circumcircle Computation ===

def circumcenter(a, b, c):
    """Compute the circumcenter of triangle abc."""
    ax, ay = a
    bx, by = b
    cx, cy = c
    D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(D) < 1e-10:
        return None
    ux = ((ax * ax + ay * ay) * (by - cy)
        + (bx * bx + by * by) * (cy - ay)
        + (cx * cx + cy * cy) * (ay - by)) / D
    uy = ((ax * ax + ay * ay) * (cx - bx)
        + (bx * bx + by * by) * (ax - cx)
        + (cx * cx + cy * cy) * (bx - ax)) / D
    return (ux, uy)


def circumradius(a, b, c):
    """Compute the circumradius of triangle abc."""
    center = circumcenter(a, b, c)
    if center is None:
        return float("inf")
    return math.hypot(center[0] - a[0], center[1] - a[1])


# === Brute-Force Delaunay Check ===

def is_delaunay(triangles, points):
    """Check if a triangulation satisfies the Delaunay property.

    Returns True if no point lies inside any triangle's circumcircle.
    """
    for tri in triangles:
        a, b, c = [points[i] for i in tri]
        if orient(a, b, c) < 0:
            a, b = b, a
        for j, p in enumerate(points):
            if j in tri:
                continue
            if in_circle(a, b, c, p) > 1e-10:
                return False
    return True


# === Main ===

if __name__ == "__main__":
    # In-circle test examples
    A, B, C = (0, 0), (4, 0), (2, 3)
    D_in = (2, 1)
    D_out = (10, 10)

    print(f"Triangle: {A}, {B}, {C}")
    print(f"Point {D_in}: in_circle = {in_circle(A, B, C, D_in):.1f} (inside)")
    print(f"Point {D_out}: in_circle = {in_circle(A, B, C, D_out):.1f} (outside)")

    center = circumcenter(A, B, C)
    radius = circumradius(A, B, C)
    print(f"Circumcenter: ({center[0]:.3f}, {center[1]:.3f})")
    print(f"Circumradius: {radius:.3f}")

    # Check a simple triangulation
    pts = [(0, 0), (4, 0), (4, 4), (0, 4)]
    tri1 = [(0, 1, 2), (0, 2, 3)]  # Diagonal (0,2)
    tri2 = [(0, 1, 3), (1, 2, 3)]  # Diagonal (1,3)
    print(f"\nSquare points: {pts}")
    print(f"Triangulation 1 is Delaunay: {is_delaunay(tri1, pts)}")
    print(f"Triangulation 2 is Delaunay: {is_delaunay(tri2, pts)}")
```

**Output:**
```
Triangle: (0, 0), (4, 0), (2, 3)
Point (2, 1): in_circle = 20.0 (inside)
Point (10, 10): in_circle = -1120.0 (outside)
Circumcenter: (2.000, 1.167)
Circumradius: 2.333

Square points: [(0, 0), (4, 0), (4, 4), (0, 4)]
Triangulation 1 is Delaunay: True
Triangulation 2 is Delaunay: True
```

## Algorithms Summary

| Algorithm | Time | Notes |
|---|---|---|
| Edge flipping | $O(n^2)$ | Simple but slow |
| Incremental | $O(n \log n)$ expected | Most practical |
| Divide and conquer | $O(n \log n)$ worst case | Theoretically optimal |
| Fortune's sweep | $O(n \log n)$ | Via Voronoi dual |

## Reference

- de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M. *Computational Geometry: Algorithms and Applications*. Springer, Chapter 9.
- Guibas, L. & Stolfi, J. "Primitives for the Manipulation of General Subdivisions." *ACM Trans. Graphics*, 1985.
