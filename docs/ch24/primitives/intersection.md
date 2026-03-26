# Line Segment Intersection

Determining whether two line segments intersect is a core primitive in
computational geometry. Map overlay, polygon clipping, motion planning,
and collision detection all reduce to this test. The key insight is that
two segments intersect if and only if certain orientation tests produce
opposite signs — no division or floating-point comparison is needed for
the basic detection step.

## Setup

A line segment is defined by two endpoints. Let segment $s_1$ have
endpoints $P_1, Q_1$ and segment $s_2$ have endpoints $P_2, Q_2$.

We use the cross product function for three points $O, A, B$:

$$
\text{cross}(O, A, B) = (A_x - O_x)(B_y - O_y) - (A_y - O_y)(B_x - O_x)
$$

The sign tells us the orientation of the triplet: positive for
counterclockwise, negative for clockwise, zero for collinear.

## General Position Test

Two segments $s_1 = \overline{P_1 Q_1}$ and $s_2 = \overline{P_2 Q_2}$
intersect in the *general* (non-collinear) case when each segment
*straddles* the line containing the other. Formally:

!!! note "Straddling Condition"
    Segment $s_1$ straddles the line through $s_2$ if $P_1$ and $Q_1$ lie
    on opposite sides of that line, i.e.,
    $\text{cross}(P_2, Q_2, P_1)$ and $\text{cross}(P_2, Q_2, Q_1)$
    have opposite signs.

The segments intersect (general case) when *both* straddle each other:

$$
d_1 = \text{cross}(P_2, Q_2, P_1), \quad d_2 = \text{cross}(P_2, Q_2, Q_1)
$$

$$
d_3 = \text{cross}(P_1, Q_1, P_2), \quad d_4 = \text{cross}(P_1, Q_1, Q_2)
$$

The segments intersect if $d_1$ and $d_2$ have opposite signs **and**
$d_3$ and $d_4$ have opposite signs.

## Collinear Special Cases

When any of $d_1, d_2, d_3, d_4$ equals zero, an endpoint lies exactly on
the line through the other segment. The segments still intersect if that
endpoint lies *on* the other segment. We check this with a bounding-box
test: point $R$ lies on segment $\overline{PQ}$ (given collinearity) when

$$
\min(P_x, Q_x) \le R_x \le \max(P_x, Q_x) \quad \text{and} \quad
\min(P_y, Q_y) \le R_y \le \max(P_y, Q_y)
$$

## Complete Algorithm

Combining both cases:

1. Compute $d_1, d_2, d_3, d_4$.
2. If $d_1 \cdot d_2 < 0$ and $d_3 \cdot d_4 < 0$: segments intersect (general case).
3. If $d_1 = 0$ and $P_1$ lies on $\overline{P_2 Q_2}$: intersect.
4. If $d_2 = 0$ and $Q_1$ lies on $\overline{P_2 Q_2}$: intersect.
5. If $d_3 = 0$ and $P_2$ lies on $\overline{P_1 Q_1}$: intersect.
6. If $d_4 = 0$ and $Q_2$ lies on $\overline{P_1 Q_1}$: intersect.
7. Otherwise: no intersection.

**Time complexity:** $O(1)$ — a constant number of cross products and comparisons.

## Finding the Intersection Point

When we know two non-parallel segments intersect, we can compute the
intersection point using parametric representation. Segment $s_1$ is
parameterized as $P_1 + t(Q_1 - P_1)$ for $t \in [0, 1]$:

$$
t = \frac{(P_2 - P_1) \times (Q_2 - P_2)}{(Q_1 - P_1) \times (Q_2 - P_2)}
$$

where $\times$ denotes the 2D cross product. The intersection point is then
$P_1 + t(Q_1 - P_1)$.

## Worked Example

Let $s_1 = \overline{(1,1)(4,4)}$ and $s_2 = \overline{(1,4)(4,1)}$.

$$
d_1 = \text{cross}((1,4),(4,1),(1,1)) = (4-1)(1-4) - (1-4)(1-1) = 3(-3) - (-3)(0) = -9
$$

$$
d_2 = \text{cross}((1,4),(4,1),(4,4)) = (4-1)(4-4) - (1-4)(4-1) = 3(0) - (-3)(3) = 9
$$

Since $d_1 < 0$ and $d_2 > 0$ (opposite signs), $s_1$ straddles the line through $s_2$.

$$
d_3 = \text{cross}((1,1),(4,4),(1,4)) = (4-1)(4-1) - (4-1)(1-1) = 9 - 0 = 9
$$

$$
d_4 = \text{cross}((1,1),(4,4),(4,1)) = (4-1)(1-1) - (4-1)(4-1) = 0 - 9 = -9
$$

Since $d_3 > 0$ and $d_4 < 0$ (opposite signs), $s_2$ also straddles $s_1$.
Both conditions hold, so the segments intersect.

## Implementation

```python
"""
Line segment intersection detection and computation.

Uses cross-product orientation tests to determine whether two segments
intersect, handling both general and collinear cases.
"""


# === Cross Product ===

def cross(o, a, b):
    """Compute the cross product of vectors OA and OB."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


# === On-Segment Test ===

def on_segment(p, q, r):
    """Check if point r lies on segment pq, given that p, q, r are collinear."""
    return (min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and
            min(p[1], q[1]) <= r[1] <= max(p[1], q[1]))


# === Intersection Detection ===

def segments_intersect(p1, q1, p2, q2):
    """Determine whether segments p1q1 and p2q2 intersect.

    Handles general position and all collinear special cases.
    Time complexity: O(1).
    """
    d1 = cross(p2, q2, p1)
    d2 = cross(p2, q2, q1)
    d3 = cross(p1, q1, p2)
    d4 = cross(p1, q1, q2)

    # General case: each segment straddles the other
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    # Collinear special cases
    if d1 == 0 and on_segment(p2, q2, p1):
        return True
    if d2 == 0 and on_segment(p2, q2, q1):
        return True
    if d3 == 0 and on_segment(p1, q1, p2):
        return True
    if d4 == 0 and on_segment(p1, q1, q2):
        return True

    return False


# === Intersection Point ===

def intersection_point(p1, q1, p2, q2):
    """Compute the intersection point of two segments (if they intersect).

    Returns (x, y) or None if segments are parallel or do not intersect.
    """
    dx1, dy1 = q1[0] - p1[0], q1[1] - p1[1]
    dx2, dy2 = q2[0] - p2[0], q2[1] - p2[1]

    denom = dx1 * dy2 - dy1 * dx2
    if denom == 0:
        return None  # Parallel or collinear

    t = ((p2[0] - p1[0]) * dy2 - (p2[1] - p1[1]) * dx2) / denom
    return (p1[0] + t * dx1, p1[1] + t * dy1)


# === Main ===

if __name__ == "__main__":
    # Intersecting segments
    s1 = ((1, 1), (4, 4))
    s2 = ((1, 4), (4, 1))
    print(f"Segment 1: {s1}")
    print(f"Segment 2: {s2}")
    print(f"Intersect: {segments_intersect(*s1, *s2)}")
    print(f"Point: {intersection_point(*s1, *s2)}")

    # Non-intersecting segments
    s3 = ((0, 0), (1, 1))
    s4 = ((2, 2), (3, 3))
    print(f"\nSegment 3: {s3}")
    print(f"Segment 4: {s4}")
    print(f"Intersect: {segments_intersect(*s3, *s4)}")

    # Collinear touching segments
    s5 = ((0, 0), (2, 0))
    s6 = ((2, 0), (4, 0))
    print(f"\nSegment 5: {s5}")
    print(f"Segment 6: {s6}")
    print(f"Intersect: {segments_intersect(*s5, *s6)}")
```

**Output:**
```
Segment 1: ((1, 1), (4, 4))
Segment 2: ((1, 4), (4, 1))
Intersect: True
Point: (2.5, 2.5)

Segment 3: ((0, 0), (1, 1))
Segment 4: ((2, 2), (3, 3))
Intersect: False

Segment 5: ((0, 0), (2, 0))
Segment 6: ((2, 0), (4, 0))
Intersect: True
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms*. MIT Press, Chapter 33.
- de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M. *Computational Geometry: Algorithms and Applications*. Springer.
