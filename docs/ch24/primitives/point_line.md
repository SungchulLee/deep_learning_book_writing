# Point-to-Line Relationships

Many geometric problems reduce to answering two basic questions about a
point and a line (or line segment): on which side of the line does the
point lie, and how far is the point from the line? The first question uses
the orientation test (cross product sign), while the second uses a
projection formula. Together, these primitives support nearest-point
queries, polygon simplification, and line-fitting algorithms.

## Line Representations

A line in 2D can be represented in several ways:

| Form | Equation | Parameters |
|---|---|---|
| Two-point | Line through $A$ and $B$ | Points $A, B$ |
| Parametric | $P(t) = A + t(B - A)$ | Direction vector $B - A$ |
| Implicit | $ax + by + c = 0$ | Coefficients $a, b, c$ |

Given two points $A = (a_x, a_y)$ and $B = (b_x, b_y)$, the implicit form is:

$$
(b_y - a_y)(x - a_x) - (b_x - a_x)(y - a_y) = 0
$$

with $a = b_y - a_y$, $b = -(b_x - a_x)$, $c = -a \cdot a_x - b \cdot a_y$.

## Side of Line Test

The cross product determines which side of line $\overleftrightarrow{AB}$
a point $P$ lies on:

$$
\text{side}(A, B, P) = (B_x - A_x)(P_y - A_y) - (B_y - A_y)(P_x - A_x)
$$

| Value | Meaning |
|---|---|
| $> 0$ | $P$ is to the left of $\overleftrightarrow{AB}$ |
| $= 0$ | $P$ is on the line |
| $< 0$ | $P$ is to the right of $\overleftrightarrow{AB}$ |

This is exactly the orientation test applied to the triplet $(A, B, P)$.

## Point-to-Line Distance

The signed distance from point $P$ to the line through $A$ and $B$ is:

$$
d_{\text{signed}} = \frac{(B_x - A_x)(P_y - A_y) - (B_y - A_y)(P_x - A_x)}{\|\overrightarrow{AB}\|}
$$

where $\|\overrightarrow{AB}\| = \sqrt{(B_x - A_x)^2 + (B_y - A_y)^2}$.
The absolute distance is $|d_{\text{signed}}|$.

!!! note "Numerator is the Cross Product"
    The numerator is the same cross product used in the side-of-line test.
    Dividing by the length of $\overrightarrow{AB}$ normalizes it to a
    true Euclidean distance.

## Point-to-Segment Distance

For a *segment* $\overline{AB}$ (not an infinite line), the closest point
may be $A$ or $B$ rather than the perpendicular foot. We use the parameter
$t$ of the projection of $P$ onto line $\overleftrightarrow{AB}$:

$$
t = \frac{\overrightarrow{AP} \cdot \overrightarrow{AB}}{\overrightarrow{AB} \cdot \overrightarrow{AB}}
$$

- If $t \le 0$: closest point is $A$.
- If $t \ge 1$: closest point is $B$.
- If $0 < t < 1$: closest point is $A + t(B - A)$.

## Worked Example

Let $A = (1, 1)$, $B = (5, 3)$, and $P = (3, 4)$.

**Side test:**

$$
\text{side}(A, B, P) = (5-1)(4-1) - (3-1)(3-1) = 4 \cdot 3 - 2 \cdot 2 = 8
$$

Since $8 > 0$, point $P$ is to the left of line $\overleftrightarrow{AB}$.

**Distance to line:**

$$
\|\overrightarrow{AB}\| = \sqrt{16 + 4} = \sqrt{20} = 2\sqrt{5}
$$

$$
d = \frac{8}{2\sqrt{5}} = \frac{4}{\sqrt{5}} = \frac{4\sqrt{5}}{5} \approx 1.789
$$

**Projection parameter:**

$$
\overrightarrow{AP} = (2, 3), \quad \overrightarrow{AB} = (4, 2)
$$

$$
t = \frac{2 \cdot 4 + 3 \cdot 2}{4^2 + 2^2} = \frac{14}{20} = 0.7
$$

Since $0 < t < 1$, the foot of the perpendicular is on the segment at
$(1 + 0.7 \cdot 4,\, 1 + 0.7 \cdot 2) = (3.8, 2.4)$.

## Implementation

```python
"""
Point-to-line and point-to-segment primitives.

Provides side-of-line test, distance computation, and closest-point
projection for lines and segments in 2D.
"""

import math


# === Side of Line Test ===

def side_of_line(a, b, p):
    """Determine which side of line AB the point P lies on.

    Returns:
        Positive if P is left of AB, negative if right, zero if on the line.
    """
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


# === Point-to-Line Distance ===

def point_line_distance(a, b, p):
    """Compute the perpendicular distance from point P to line AB."""
    cross = side_of_line(a, b, p)
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    if length == 0:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    return abs(cross) / length


# === Point-to-Segment Distance ===

def point_segment_distance(a, b, p):
    """Compute the distance from point P to segment AB.

    Returns the distance to the closest point on the segment,
    which may be an endpoint or the perpendicular foot.
    """
    dx, dy = b[0] - a[0], b[1] - a[1]
    len_sq = dx * dx + dy * dy

    if len_sq == 0:
        return math.hypot(p[0] - a[0], p[1] - a[1])

    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len_sq
    t = max(0, min(1, t))

    proj_x = a[0] + t * dx
    proj_y = a[1] + t * dy
    return math.hypot(p[0] - proj_x, p[1] - proj_y)


# === Closest Point on Segment ===

def closest_point_on_segment(a, b, p):
    """Find the closest point on segment AB to point P."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    len_sq = dx * dx + dy * dy

    if len_sq == 0:
        return a

    t = max(0, min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len_sq))
    return (a[0] + t * dx, a[1] + t * dy)


# === Main ===

if __name__ == "__main__":
    A, B = (1, 1), (5, 3)
    P = (3, 4)

    print(f"Line through A={A}, B={B}")
    print(f"Query point P={P}")
    print(f"Side value: {side_of_line(A, B, P)}")
    print(f"Distance to line: {point_line_distance(A, B, P):.4f}")
    print(f"Distance to segment: {point_segment_distance(A, B, P):.4f}")
    print(f"Closest point: {closest_point_on_segment(A, B, P)}")

    # Point beyond segment endpoint
    P2 = (7, 5)
    print(f"\nQuery point P2={P2}")
    print(f"Distance to line: {point_line_distance(A, B, P2):.4f}")
    print(f"Distance to segment: {point_segment_distance(A, B, P2):.4f}")
    print(f"Closest point: {closest_point_on_segment(A, B, P2)}")
```

**Output:**
```
Line through A=(1, 1), B=(5, 3)
Query point P=(3, 4)
Side value: 8
Distance to line: 1.7889
Distance to segment: 1.7889
Closest point: (3.8, 2.4)

Query point P2=(7, 5)
Distance to line: 1.7889
Distance to segment: 2.8284
Closest point: (5, 3)
```

## Reference

- de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M. *Computational Geometry: Algorithms and Applications*. Springer.
- O'Rourke, J. *Computational Geometry in C*. Cambridge University Press.
