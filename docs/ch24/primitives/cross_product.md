# Cross Product in 2D

The cross product is the most fundamental primitive in computational geometry.
In two dimensions, it answers three questions at once: which way do we turn
at a point, what is the signed area of a triangle, and on which side of a
line does a point lie? Nearly every geometric algorithm — convex hull,
polygon triangulation, line segment intersection — relies on the 2D cross
product as its core building block.

## Definition

Given two 2D vectors $\mathbf{a} = (a_x, a_y)$ and $\mathbf{b} = (b_x, b_y)$,
their **2D cross product** (also called the *perp dot product*) is the scalar:

$$
\mathbf{a} \times \mathbf{b} = a_x \, b_y - a_y \, b_x
$$

Geometrically, this equals the signed area of the parallelogram spanned by
$\mathbf{a}$ and $\mathbf{b}$.

!!! note "2D vs 3D Cross Product"
    In 3D, the cross product produces a vector. The 2D cross product is the
    $z$-component of the 3D cross product when both vectors lie in the
    $xy$-plane: $(a_x, a_y, 0) \times (b_x, b_y, 0) = (0, 0, a_x b_y - a_y b_x)$.

## Sign Interpretation

The sign of $\mathbf{a} \times \mathbf{b}$ encodes the relative orientation:

| Value | Meaning |
|---|---|
| $> 0$ | $\mathbf{b}$ is counterclockwise from $\mathbf{a}$ (left turn) |
| $= 0$ | $\mathbf{a}$ and $\mathbf{b}$ are collinear |
| $< 0$ | $\mathbf{b}$ is clockwise from $\mathbf{a}$ (right turn) |

## Cross Product of Three Points

In practice, we often work with three points $O$, $A$, $B$ and compute the
cross product of vectors $\overrightarrow{OA}$ and $\overrightarrow{OB}$:

$$
\text{cross}(O, A, B) = (A_x - O_x)(B_y - O_y) - (A_y - O_y)(B_x - O_x)
$$

This tells us the turn direction at $O$ when traveling from $A$ to $B$:

- **Positive**: left turn (counterclockwise)
- **Zero**: straight (collinear)
- **Negative**: right turn (clockwise)

## Relationship to Area

The signed area of the triangle $\triangle OAB$ is exactly half the cross
product:

$$
A_{\triangle} = \frac{1}{2}\,\text{cross}(O, A, B)
$$

The shoelace formula for polygon area is a direct consequence: it sums
these signed triangle areas over consecutive edges.

## Worked Example

Consider points $O = (1, 1)$, $A = (4, 1)$, $B = (2, 3)$.

$$
\text{cross}(O, A, B) = (4 - 1)(3 - 1) - (1 - 1)(2 - 1) = 3 \cdot 2 - 0 \cdot 1 = 6
$$

Since $6 > 0$, the turn from $\overrightarrow{OA}$ to $\overrightarrow{OB}$
is counterclockwise. The triangle area is $6 / 2 = 3$.

Now consider $O = (1, 1)$, $A = (4, 1)$, $B = (3, 0)$:

$$
\text{cross}(O, A, B) = (4 - 1)(0 - 1) - (1 - 1)(3 - 1) = 3 \cdot (-1) - 0 \cdot 2 = -3
$$

Since $-3 < 0$, this is a clockwise (right) turn.

## Implementation

```python
"""
2D cross product: the fundamental primitive for computational geometry.

Provides cross product computation for vectors and point triples,
with applications to orientation testing and area calculation.
"""


# === Vector Cross Product ===

def cross2d(ax, ay, bx, by):
    """Compute the 2D cross product of vectors (ax, ay) and (bx, by).

    Returns:
        Positive if b is CCW from a, negative if CW, zero if collinear.
    """
    return ax * by - ay * bx


# === Three-Point Cross Product ===

def cross(o, a, b):
    """Compute the cross product of vectors OA and OB.

    Args:
        o, a, b: points as (x, y) tuples.

    Returns:
        Positive for left turn, negative for right turn, zero for collinear.
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


# === Orientation Test ===

def orientation(o, a, b):
    """Determine the orientation of the triplet (o, a, b).

    Returns:
        1 for counterclockwise, -1 for clockwise, 0 for collinear.
    """
    cp = cross(o, a, b)
    if cp > 0:
        return 1   # CCW
    elif cp < 0:
        return -1  # CW
    else:
        return 0   # Collinear


# === Triangle Area ===

def triangle_area(o, a, b):
    """Compute the absolute area of triangle OAB."""
    return abs(cross(o, a, b)) / 2.0


# === Main ===

if __name__ == "__main__":
    # Example 1: left turn
    O, A, B = (1, 1), (4, 1), (2, 3)
    print(f"Points: O={O}, A={A}, B={B}")
    print(f"Cross product: {cross(O, A, B)}")
    print(f"Orientation: {'CCW' if orientation(O, A, B) == 1 else 'CW'}")
    print(f"Triangle area: {triangle_area(O, A, B)}")

    # Example 2: right turn
    B2 = (3, 0)
    print(f"\nPoints: O={O}, A={A}, B={B2}")
    print(f"Cross product: {cross(O, A, B2)}")
    print(f"Orientation: {'CW' if orientation(O, A, B2) == -1 else 'CCW'}")

    # Example 3: collinear
    B3 = (7, 1)
    print(f"\nPoints: O={O}, A={A}, B={B3}")
    print(f"Cross product: {cross(O, A, B3)}")
    print(f"Orientation: collinear" if orientation(O, A, B3) == 0 else "")
```

**Output:**
```
Points: O=(1, 1), A=(4, 1), B=(2, 3)
Cross product: 6
Orientation: CCW
Triangle area: 3.0

Points: O=(1, 1), A=(4, 1), B=(3, 0)
Cross product: -3
Orientation: CW

Points: O=(1, 1), A=(4, 1), B=(7, 1)
Cross product: 0
Orientation: collinear
```

## Applications in Computational Geometry

| Algorithm | How It Uses the Cross Product |
|---|---|
| Convex hull (Graham scan) | Sort by polar angle; detect left/right turns |
| Line segment intersection | Test whether endpoints lie on opposite sides |
| Polygon area (shoelace) | Sum signed triangle areas |
| Point-in-polygon | Count signed crossings of a ray |
| Triangulation | Determine ear tips via orientation |

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms*. MIT Press.
- de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M. *Computational Geometry: Algorithms and Applications*. Springer.
