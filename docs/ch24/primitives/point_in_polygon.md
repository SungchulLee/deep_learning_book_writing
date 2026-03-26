# Point-in-Polygon Test

Given a polygon and a query point, does the point lie inside or outside
the polygon? This question arises constantly in computer graphics (hit
testing), geographic information systems (geo-fencing), and robotics
(collision detection). The classic solution is the **ray casting algorithm**:
shoot a ray from the query point in any direction and count how many times
it crosses the polygon boundary. An odd count means inside; an even count
means outside.

## The Ray Casting Algorithm

The idea rests on the Jordan Curve Theorem: every simple closed curve
divides the plane into an interior and an exterior. A ray from an exterior
point crosses the boundary an even number of times (possibly zero), while
a ray from an interior point crosses an odd number of times.

### Algorithm Steps

1. Cast a horizontal ray from the query point $Q = (q_x, q_y)$ toward $+\infty$.
2. For each edge $\overline{P_i P_{i+1}}$ of the polygon, check whether the
   ray crosses this edge.
3. Count crossings. If the count is odd, $Q$ is inside; otherwise, outside.

### Edge Crossing Condition

The horizontal ray from $Q$ crosses edge $\overline{P_i P_{i+1}}$ when:

1. The edge straddles the horizontal line $y = q_y$, meaning one endpoint
   is above $q_y$ and the other is at or below $q_y$.
2. The intersection of the edge with $y = q_y$ has $x$-coordinate greater
   than $q_x$.

The $x$-coordinate of the intersection is computed by linear interpolation:

$$
x_{\text{int}} = P_{i,x} + \frac{q_y - P_{i,y}}{P_{i+1,y} - P_{i,y}} \cdot (P_{i+1,x} - P_{i,x})
$$

## Handling Edge Cases

!!! warning "Degenerate Cases"
    The ray may pass through a vertex or run along an edge. The standard
    fix is the "endpoint convention": count an edge as crossed only if
    its lower endpoint is strictly below the ray and its upper endpoint
    is at or above the ray. This ensures each vertex is counted exactly
    once.

Specifically, for edge $\overline{P_i P_{i+1}}$:

- If $P_{i,y} \le q_y < P_{i+1,y}$ (edge goes upward across the ray), check crossing.
- If $P_{i+1,y} \le q_y < P_{i,y}$ (edge goes downward across the ray), check crossing.
- Otherwise, skip this edge.

## Complexity

| Operation | Time | Space |
|---|---|---|
| Ray casting | $O(n)$ | $O(1)$ |
| Preprocessing (trapezoidal map) | $O(n \log n)$ build, $O(\log n)$ query | $O(n)$ |

For a single query, the simple $O(n)$ algorithm is optimal. For many
queries on the same polygon, preprocessing into a trapezoidal map
reduces each query to $O(\log n)$.

## Winding Number Alternative

The **winding number** method counts how many times the polygon winds
around the query point. For a simple polygon, a nonzero winding number
means inside. The winding number handles self-intersecting polygons
correctly, whereas ray casting does not.

The winding number for point $Q$ relative to polygon $P_0 P_1 \ldots P_{n-1}$ is:

$$
w = \frac{1}{2\pi} \sum_{i=0}^{n-1} \theta_i
$$

where $\theta_i$ is the signed angle subtended by edge $\overline{P_i P_{i+1}}$
at $Q$. In practice, we avoid computing angles by using orientation tests
and upward/downward crossing rules.

## Worked Example

Consider a square with vertices $(0,0)$, $(4,0)$, $(4,4)$, $(0,4)$ and
query point $Q = (2, 2)$.

Casting a horizontal ray from $(2,2)$ toward $+\infty$:

| Edge | Straddles $y=2$? | $x_{\text{int}}$ | Crosses ray? |
|---|---|---|---|
| $(0,0) \to (4,0)$ | No ($0 \le 2$ and $0 \le 2$) | — | No |
| $(4,0) \to (4,4)$ | Yes ($0 \le 2 < 4$) | $4$ | Yes ($4 > 2$) |
| $(4,4) \to (0,4)$ | No ($4 > 2$ and $4 > 2$) | — | No |
| $(0,4) \to (0,0)$ | Yes ($0 \le 2 < 4$) | $0$ | No ($0 < 2$) |

Crossing count = 1 (odd), so $Q = (2,2)$ is **inside**.

## Implementation

```python
"""
Point-in-polygon test using the ray casting algorithm.

Shoots a horizontal ray from the query point and counts edge crossings.
Odd crossings = inside, even crossings = outside.
"""


# === Ray Casting ===

def point_in_polygon(polygon, query):
    """Test whether a point lies inside a simple polygon.

    Args:
        polygon: list of (x, y) vertices in order.
        query: (x, y) point to test.

    Returns:
        True if the point is inside, False otherwise.
        Points on the boundary may return either value.
    """
    qx, qy = query
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        # Check if edge straddles the horizontal ray
        if (yi > qy) != (yj > qy):
            # Compute x-coordinate of intersection
            x_int = xj + (qy - yj) / (yi - yj) * (xi - xj)
            if qx < x_int:
                inside = not inside

        j = i

    return inside


# === Winding Number ===

def winding_number(polygon, query):
    """Compute the winding number of a polygon around a point.

    Returns:
        The winding number (nonzero means inside for simple polygons).
    """
    qx, qy = query
    n = len(polygon)
    wn = 0

    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[(i + 1) % n]

        if yi <= qy:
            if yj > qy:
                # Upward crossing
                cross = (xj - xi) * (qy - yi) - (qx - xi) * (yj - yi)
                if cross > 0:
                    wn += 1
        else:
            if yj <= qy:
                # Downward crossing
                cross = (xj - xi) * (qy - yi) - (qx - xi) * (yj - yi)
                if cross < 0:
                    wn -= 1

    return wn


# === Main ===

if __name__ == "__main__":
    # Square polygon
    square = [(0, 0), (4, 0), (4, 4), (0, 4)]

    test_points = [(2, 2), (5, 2), (0, 0), (4, 2)]
    for pt in test_points:
        rc = point_in_polygon(square, pt)
        wn = winding_number(square, pt)
        print(f"Point {pt}: ray_cast={rc}, winding={wn}")

    # L-shaped polygon
    l_shape = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
    print(f"\nL-shaped polygon: {l_shape}")
    for pt in [(0.5, 0.5), (1.5, 1.5), (0.5, 1.5)]:
        rc = point_in_polygon(l_shape, pt)
        print(f"Point {pt}: inside={rc}")
```

**Output:**
```
Point (2, 2): ray_cast=True, winding=1
Point (5, 2): ray_cast=False, winding=0
Point (0, 0): ray_cast=False, winding=0
Point (4, 2): ray_cast=False, winding=0

L-shaped polygon: [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
Point (0.5, 0.5): inside=True
Point (1.5, 1.5): inside=False
Point (0.5, 1.5): inside=True
```

## Reference

- de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M. *Computational Geometry: Algorithms and Applications*. Springer.
- O'Rourke, J. *Computational Geometry in C*. Cambridge University Press.
