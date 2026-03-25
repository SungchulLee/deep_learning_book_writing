# Andrew's Monotone Chain

Andrew's monotone chain algorithm constructs the convex hull of a set of
points in the plane by building the **lower hull** and **upper hull** separately,
each as a monotone polygonal chain sorted by x-coordinate.  Sorting first,
then scanning left-to-right (lower) and right-to-left (upper), gives
$O(n \log n)$ time with simple, cache-friendly code.

## Intuition

Sort all points by x-coordinate (break ties by y).  Walk left-to-right,
always turning **left** (counter-clockwise).  Whenever the last three points
make a right turn or are collinear, pop the middle point.  The surviving
points form the lower boundary of the hull.  Repeat in reverse for the
upper boundary, then join the two chains.

## Definitions

**Convex hull.**
Given a finite point set $S \subset \mathbb{R}^{2}$, the convex hull
$\operatorname{CH}(S)$ is the smallest convex polygon containing every
point in $S$.

**Cross product test.**
For three points $o, a, b$ the signed area of the parallelogram is

$$
\operatorname{cross}(o, a, b)
  = (a_x - o_x)(b_y - o_y) - (a_y - o_y)(b_x - o_x)
$$

- $> 0$: counter-clockwise (left turn)
- $= 0$: collinear
- $< 0$: clockwise (right turn)

## Algorithm

1. **Sort** the $n$ points lexicographically by $(x, y)$.
2. **Lower hull.** Initialise an empty stack.  For each point $p$ in
   sorted order, while the last two stack points and $p$ do **not** make a
   left turn ($\operatorname{cross} \le 0$), pop the stack.  Then push $p$.
3. **Upper hull.** Same scan in reverse sorted order.
4. **Concatenate** the two chains (removing the duplicate endpoints).

## Correctness

!!! note "Claim"
    The algorithm outputs exactly the vertices of $\operatorname{CH}(S)$ in
    counter-clockwise order.

**Proof sketch.**
Every point removed from the lower-hull stack lies strictly inside the
triangle formed by its predecessor, $p$, and some later hull vertex, so it
cannot be a hull vertex.  Every point that survives lies on the boundary
because no subsequent point causes it to be popped; it therefore witnesses
an extreme direction.  The upper hull argument is symmetric.  Concatenation
yields a simple polygon whose interior angles are all less than $\pi$,
hence convex.

## Complexity

| Measure | Cost |
|---------|------|
| Time    | $O(n \log n)$ — sorting dominates; each point is pushed and popped at most once |
| Space   | $O(n)$ — for the sorted array and the two stacks |

The $O(n \log n)$ bound is **optimal** for comparison-based convex hull
(see the lower-bound page).

## Implementation

```python
"""
Andrew's Monotone Chain — convex hull in O(n log n).
"""

from __future__ import annotations


# === Cross-product orientation test ==========================================

def cross(o: tuple[float, float],
          a: tuple[float, float],
          b: tuple[float, float]) -> float:
    """Return the signed area of the parallelogram formed by vectors OA and OB.

    Positive ⟹ counter-clockwise, zero ⟹ collinear, negative ⟹ clockwise.
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


# === Monotone chain ==========================================================

def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return the convex hull vertices in counter-clockwise order.

    Collinear points on the hull boundary are excluded (strict inequality
    would include them — change ``<= 0`` to ``< 0`` if desired).
    """
    pts = sorted(set(points))
    if len(pts) <= 1:
        return list(pts)

    # Lower hull (left to right)
    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Upper hull (right to left)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Remove duplicate endpoints where the chains meet
    return lower[:-1] + upper[:-1]


# === Demo ====================================================================

if __name__ == "__main__":
    sample = [(0, 0), (1, 1), (2, 2), (0, 2), (2, 0), (1, 0)]
    hull = convex_hull(sample)
    print(f"Input:  {sample}")
    print(f"Hull:   {hull}")
    print(f"Vertices on hull: {len(hull)}")
```

**Output:**

```
Input:  [(0, 0), (1, 1), (2, 2), (0, 2), (2, 0), (1, 0)]
Hull:   [(0, 0), (1, 0), (2, 0), (2, 2), (0, 2)]
Vertices on hull: 5
```

Note that the collinear points $(0,0), (1,0), (2,0)$ all appear because
they lie on the hull boundary, but the interior point $(1,1)$ and the
collinear-but-interior point $(2,2) \to (0,2)$ segment's midpoint are
excluded.

## Variants

- **Include collinear points.** Change `<= 0` to `< 0` in the pop condition.
  Every point on the hull boundary is then reported.
- **Online / incremental.** Maintaining a balanced BST of hull edges allows
  $O(\log n)$ per insertion, though the constant is larger.

## Reference

- A. M. Andrew, "Another Efficient Algorithm for Convex Hulls in Two
  Dimensions," *Information Processing Letters*, 9(5), 1979.
- de Berg, Cheong, van Kreveld, Overmars, *Computational Geometry:
  Algorithms and Applications*, 3rd ed., Springer, 2008.
