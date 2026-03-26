# Polygon Area via the Shoelace Formula

Computing the area of a polygon is one of the most basic operations in
computational geometry. Given the vertices of a simple polygon in order,
the shoelace formula (also called the surveyor's formula) computes its
signed area in $O(n)$ time using only additions and multiplications — no
trigonometry or square roots required.

## Signed Area of a Triangle

Before tackling general polygons, consider a triangle with vertices
$P_0 = (x_0, y_0)$, $P_1 = (x_1, y_1)$, and $P_2 = (x_2, y_2)$.
Its signed area equals half the cross product of two edge vectors:

$$
A_{\text{signed}} = \frac{1}{2}
\begin{vmatrix}
x_1 - x_0 & x_2 - x_0 \\
y_1 - y_0 & y_2 - y_0
\end{vmatrix}
= \frac{1}{2}\bigl((x_1 - x_0)(y_2 - y_0) - (x_2 - x_0)(y_1 - y_0)\bigr)
$$

The sign encodes orientation: positive when $P_0, P_1, P_2$ are in
counterclockwise order, negative when clockwise, and zero when collinear.

## The Shoelace Formula

For a simple polygon with $n$ vertices $P_0, P_1, \ldots, P_{n-1}$ listed
in order (either clockwise or counterclockwise), the signed area is:

$$
A_{\text{signed}} = \frac{1}{2} \sum_{i=0}^{n-1}
(x_i \, y_{i+1} - x_{i+1} \, y_i)
$$

where indices are taken modulo $n$, so $P_n = P_0$. The absolute area is
$|A_{\text{signed}}|$.

!!! tip "Why 'Shoelace'?"
    Writing the $x$- and $y$-coordinates in two columns and cross-multiplying
    diagonally produces a pattern that resembles lacing a shoe.

## Derivation

The formula follows from triangulating the polygon with respect to the origin.
Decompose the polygon into $n$ signed triangles $\triangle(O, P_i, P_{i+1})$
where $O$ is the origin. Each triangle has signed area:

$$
A_i = \frac{1}{2}(x_i \, y_{i+1} - x_{i+1} \, y_i)
$$

Summing over all $n$ triangles, the contributions outside the polygon cancel
in pairs (by the telescoping property of signed areas), leaving exactly the
signed area of the polygon:

$$
A_{\text{signed}} = \sum_{i=0}^{n-1} A_i
= \frac{1}{2} \sum_{i=0}^{n-1} (x_i \, y_{i+1} - x_{i+1} \, y_i)
$$

## Worked Example

Consider a quadrilateral with vertices (in counterclockwise order):
$P_0 = (1, 1)$, $P_1 = (4, 1)$, $P_2 = (4, 3)$, $P_3 = (1, 3)$.

This is a $3 \times 2$ rectangle, so the expected area is $6$.

| $i$ | $(x_i, y_i)$ | $(x_{i+1}, y_{i+1})$ | $x_i y_{i+1} - x_{i+1} y_i$ |
|-----|---------------|-----------------------|-------------------------------|
| 0   | $(1, 1)$      | $(4, 1)$              | $1 \cdot 1 - 4 \cdot 1 = -3$ |
| 1   | $(4, 1)$      | $(4, 3)$              | $4 \cdot 3 - 4 \cdot 1 = 8$  |
| 2   | $(4, 3)$      | $(1, 3)$              | $4 \cdot 3 - 1 \cdot 3 = 9$  |
| 3   | $(1, 3)$      | $(1, 1)$              | $1 \cdot 1 - 1 \cdot 3 = -2$ |

$$
A_{\text{signed}} = \frac{1}{2}(-3 + 8 + 9 + (-2)) = \frac{12}{2} = 6
$$

The positive sign confirms counterclockwise orientation.

## Implementation

```python
"""
Polygon area via the shoelace formula.

Computes the signed and absolute area of a simple polygon given its
vertices in order. Time complexity: O(n).
"""


# === Shoelace Formula ===

def signed_area(polygon):
    """Compute the signed area of a simple polygon.

    Args:
        polygon: list of (x, y) tuples in order (CW or CCW).

    Returns:
        Signed area. Positive for CCW, negative for CW.
    """
    n = len(polygon)
    area = 0.0
    for i in range(n):
        x_i, y_i = polygon[i]
        x_next, y_next = polygon[(i + 1) % n]
        area += x_i * y_next - x_next * y_i
    return area / 2.0


def polygon_area(polygon):
    """Compute the absolute area of a simple polygon.

    Args:
        polygon: list of (x, y) tuples in order.

    Returns:
        Non-negative area.
    """
    return abs(signed_area(polygon))


# === Triangle Area ===

def triangle_area(p0, p1, p2):
    """Compute the absolute area of a triangle given three vertices."""
    return abs(
        (p1[0] - p0[0]) * (p2[1] - p0[1])
        - (p2[0] - p0[0]) * (p1[1] - p0[1])
    ) / 2.0


# === Main ===

if __name__ == "__main__":
    # Rectangle example
    rect = [(1, 1), (4, 1), (4, 3), (1, 3)]
    print(f"Rectangle vertices: {rect}")
    print(f"Signed area: {signed_area(rect)}")
    print(f"Absolute area: {polygon_area(rect)}")

    # Triangle example
    tri = [(0, 0), (4, 0), (2, 3)]
    print(f"\nTriangle vertices: {tri}")
    print(f"Triangle area: {triangle_area(*tri)}")
    print(f"Shoelace area: {polygon_area(tri)}")
```

**Output:**
```
Rectangle vertices: [(1, 1), (4, 1), (4, 3), (1, 3)]
Signed area: 6.0
Absolute area: 6.0

Triangle vertices: [(0, 0), (4, 0), (2, 3)]
Triangle area: 6.0
Shoelace area: 6.0
```

## Complexity

| Operation | Time | Space |
|---|---|---|
| Shoelace formula | $O(n)$ | $O(1)$ |
| Triangle area | $O(1)$ | $O(1)$ |

The shoelace formula processes each vertex exactly once and uses only a
running accumulator, making it both time- and space-optimal.

## Practical Considerations

!!! warning "Simple Polygons Only"
    The shoelace formula assumes a *simple* polygon (no self-intersections).
    For a self-intersecting polygon, the formula computes a signed area where
    overlapping regions may cancel, giving an incorrect result.

- **Numerical precision.** For integer coordinates, the formula involves
  only integer arithmetic (multiply and add), so it is exact. For
  floating-point coordinates, accumulation errors grow with $n$; use
  compensated summation (Kahan) for large polygons.
- **Orientation detection.** The sign of the shoelace result tells whether
  the vertices are listed counterclockwise (positive) or clockwise (negative).

## Reference

- de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M. *Computational Geometry: Algorithms and Applications*. Springer.
- O'Rourke, J. *Computational Geometry in C*. Cambridge University Press.
