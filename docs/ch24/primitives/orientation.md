# Orientation Test

When walking from point $P$ through $Q$ toward $R$, do we turn left,
turn right, or continue straight? This simple question — the *orientation
test* — is the most frequently called subroutine in computational geometry.
Convex hull construction, segment intersection, polygon triangulation,
and point-in-polygon testing all depend on it. The test reduces to
computing a single 2D cross product and checking its sign.

## Definition

Given three ordered points $P = (p_x, p_y)$, $Q = (q_x, q_y)$, and
$R = (r_x, r_y)$, the **orientation** of the triplet $(P, Q, R)$ is
determined by the sign of the cross product of vectors
$\overrightarrow{PQ}$ and $\overrightarrow{PR}$:

$$
\text{orient}(P, Q, R)
= (q_x - p_x)(r_y - p_y) - (q_y - p_y)(r_x - p_x)
$$

| Sign | Orientation | Geometric Meaning |
|---|---|---|
| $> 0$ | Counterclockwise (CCW) | Left turn at $Q$ |
| $= 0$ | Collinear | $P$, $Q$, $R$ are on one line |
| $< 0$ | Clockwise (CW) | Right turn at $Q$ |

## Determinant Interpretation

The orientation predicate can be expressed as a $3 \times 3$ determinant:

$$
\text{orient}(P, Q, R) =
\begin{vmatrix}
1 & p_x & p_y \\
1 & q_x & q_y \\
1 & r_x & r_y
\end{vmatrix}
= \begin{vmatrix}
q_x - p_x & r_x - p_x \\
q_y - p_y & r_y - p_y
\end{vmatrix}
$$

This is twice the signed area of triangle $\triangle PQR$. The triangle has
positive signed area when the vertices are in counterclockwise order.

## Connection to the Cross Product

The orientation value is exactly the 2D cross product
$\overrightarrow{PQ} \times \overrightarrow{PR}$. This connects orientation
testing to area computation: the signed area of $\triangle PQR$ is

$$
A_{\triangle} = \frac{1}{2}\,\text{orient}(P, Q, R)
$$

## Worked Example

Consider three points: $P = (0, 0)$, $Q = (4, 0)$, $R = (2, 3)$.

$$
\text{orient}(P, Q, R) = (4 - 0)(3 - 0) - (0 - 0)(2 - 0) = 12 - 0 = 12
$$

Since $12 > 0$, the triplet is counterclockwise — we make a left turn at $Q$.

Now swap $Q$ and $R$: $P = (0, 0)$, $Q = (2, 3)$, $R = (4, 0)$.

$$
\text{orient}(P, Q, R) = (2 - 0)(0 - 0) - (3 - 0)(4 - 0) = 0 - 12 = -12
$$

Since $-12 < 0$, the triplet is clockwise — we make a right turn.

Finally, collinear points: $P = (0, 0)$, $Q = (2, 2)$, $R = (4, 4)$.

$$
\text{orient}(P, Q, R) = (2)(4) - (2)(4) = 8 - 8 = 0
$$

Zero means the three points lie on the same line.

## Robustness Considerations

!!! warning "Floating-Point Pitfalls"
    With floating-point arithmetic, the orientation test can give wrong
    results when points are nearly collinear. A value close to zero may
    have the wrong sign due to rounding. Robust implementations use either
    exact arithmetic (e.g., adaptive precision as in Shewchuk's predicates)
    or a tolerance-based approach with an epsilon threshold.

For integer coordinates, the test is exact as long as the intermediate
products do not overflow. With 32-bit integer coordinates, the cross
product fits in a 64-bit integer.

## Implementation

```python
"""
Orientation test for three points in the plane.

Determines whether a triplet of points makes a left turn (CCW),
right turn (CW), or is collinear. This is the fundamental predicate
in computational geometry.
"""


# === Orientation Predicate ===

def orient(p, q, r):
    """Compute the orientation value for points p, q, r.

    Returns:
        Positive for CCW (left turn), negative for CW (right turn),
        zero for collinear.
    """
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def orientation(p, q, r):
    """Classify the orientation of triplet (p, q, r).

    Returns:
        1 for CCW, -1 for CW, 0 for collinear.
    """
    val = orient(p, q, r)
    if val > 0:
        return 1
    elif val < 0:
        return -1
    return 0


# === Helper: human-readable label ===

def orient_label(p, q, r):
    """Return a human-readable orientation label."""
    labels = {1: "CCW (left turn)", -1: "CW (right turn)", 0: "Collinear"}
    return labels[orientation(p, q, r)]


# === Main ===

if __name__ == "__main__":
    # Left turn
    P, Q, R = (0, 0), (4, 0), (2, 3)
    print(f"P={P}, Q={Q}, R={R}")
    print(f"  orient = {orient(P, Q, R)}, {orient_label(P, Q, R)}")

    # Right turn
    P, Q, R = (0, 0), (2, 3), (4, 0)
    print(f"P={P}, Q={Q}, R={R}")
    print(f"  orient = {orient(P, Q, R)}, {orient_label(P, Q, R)}")

    # Collinear
    P, Q, R = (0, 0), (2, 2), (4, 4)
    print(f"P={P}, Q={Q}, R={R}")
    print(f"  orient = {orient(P, Q, R)}, {orient_label(P, Q, R)}")
```

**Output:**
```
P=(0, 0), Q=(4, 0), R=(2, 3)
  orient = 12, CCW (left turn)
P=(0, 0), Q=(2, 3), R=(4, 0)
  orient = -12, CW (right turn)
P=(0, 0), Q=(2, 2), R=(4, 4)
  orient = 0, Collinear
```

## Applications

| Algorithm | Role of Orientation Test |
|---|---|
| Graham scan | Detect and eliminate right turns on the hull |
| Segment intersection | Check if endpoints straddle a line |
| Point-in-polygon | Determine winding number |
| Polygon triangulation | Identify ear tips |
| Delaunay triangulation | In-circle test (extended orientation) |

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms*. MIT Press, Chapter 33.
- Shewchuk, J. R. "Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates." *Discrete & Computational Geometry*, 1997.
