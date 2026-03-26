# Graham Scan

Graham scan constructs the convex hull of $n$ points in $O(n \log n)$ time.
It chooses an anchor point (the lowest), sorts the remaining points by
**polar angle** around the anchor, and then processes them in order with a
stack, ensuring every consecutive triple makes a left turn.

## Intuition

Pick the bottom-most point — it is guaranteed to be on the hull.  Standing
at that anchor, sort every other point by the angle of the line from the
anchor to the point.  Walk around in increasing-angle order, maintaining a
stack of candidate hull vertices.  Whenever the top two stack entries and
the new point form a **right turn** (or are collinear), the middle point
cannot be on the hull, so pop it.  When the sweep is complete the stack
holds exactly the hull vertices in counter-clockwise order.

## Definitions

**Polar angle.**  The angle $\theta = \operatorname{atan2}(y - y_0,\;
x - x_0)$ measured from the positive $x$-axis, where $(x_0, y_0)$ is the
anchor.

**Left turn / right turn.**  Determined by the sign of the cross product
of consecutive edge vectors:

$$
\operatorname{cross}(o, a, b)
  = (a_x - o_x)(b_y - o_y) - (a_y - o_y)(b_x - o_x)
$$

Positive means left turn (counter-clockwise), negative means right turn.

## Algorithm

1. Find the point $p_0$ with the smallest $y$-coordinate (break ties by
   smallest $x$).
2. Sort the remaining points by polar angle with respect to $p_0$.  For
   points with equal angle, keep only the farthest from $p_0$
   (or sort by distance as a secondary key).
3. Push $p_0$ and the first sorted point onto a stack.
4. For each subsequent point $p_i$:
    - While $|\text{stack}| \ge 2$ and
      $\operatorname{cross}(\text{second}, \text{top}, p_i) \le 0$: pop.
    - Push $p_i$.
5. The stack contents are the hull vertices in counter-clockwise order.

## Correctness

!!! note "Theorem"
    Graham scan outputs the vertices of $\operatorname{CH}(S)$ in counter-
    clockwise order.

**Proof.**
At every stage the stack forms a convex polygon: the invariant is that all
consecutive triples on the stack make left turns.  The pop operation
removes a point that would create a right turn, and the push restores the
invariant.  Because we process points in angular order around $p_0$, no
hull vertex is ever skipped.  A point that is popped lies inside the
triangle formed by its neighbors and some later point, so it is interior.
When the last point has been processed the stack is the complete hull.

## Complexity

| Measure | Cost |
|---------|------|
| Time | $O(n \log n)$ — sorting dominates; the stack operations are amortized $O(n)$ |
| Space | $O(n)$ |

## Python Implementation

```python
"""
Graham Scan — convex hull in O(n log n) via polar-angle sort.
"""

from __future__ import annotations
import math


# === Orientation Helper ===

def cross(o: tuple[float, float],
          a: tuple[float, float],
          b: tuple[float, float]) -> float:
    """Signed area of parallelogram OA x OB."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def dist_sq(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Squared Euclidean distance."""
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


# === Graham Scan ===

def graham_scan(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return convex hull vertices in counter-clockwise order.

    Uses polar-angle sorting around the bottom-most point.
    """
    pts = list(set(points))
    if len(pts) <= 1:
        return pts

    # Step 1: find the bottom-most point (lowest y, then leftmost x)
    anchor = min(pts, key=lambda p: (p[1], p[0]))
    pts.remove(anchor)

    # Step 2: sort by polar angle, break ties by distance
    def angle_key(p: tuple[float, float]) -> tuple[float, float]:
        return (math.atan2(p[1] - anchor[1], p[0] - anchor[0]),
                dist_sq(anchor, p))

    pts.sort(key=angle_key)

    # Step 3-4: build hull with a stack
    stack: list[tuple[float, float]] = [anchor]
    for p in pts:
        while len(stack) >= 2 and cross(stack[-2], stack[-1], p) <= 0:
            stack.pop()
        stack.append(p)

    return stack


# === Main ===

if __name__ == "__main__":
    sample = [(0, 0), (1, 1), (2, 2), (0, 2), (2, 0), (1, 0)]
    hull = graham_scan(sample)
    print(f"Input:  {sample}")
    print(f"Hull:   {hull}")
    print(f"Vertices on hull: {len(hull)}")
    # Output:
    # Input:  [(0, 0), (1, 1), (2, 2), (0, 2), (2, 0), (1, 0)]
    # Hull:   [(0, 0), (2, 0), (2, 2), (0, 2)]
    # Vertices on hull: 4
```

## Comparison with Andrew's Monotone Chain

| | Graham Scan | Monotone Chain |
|---|---|---|
| Sort key | Polar angle | Lexicographic $(x, y)$ |
| Scans | Single pass | Two passes (lower + upper) |
| Collinear handling | Needs care at equal angles | Naturally handled |
| Numerical stability | Uses `atan2` (floating point) | Integer-safe with cross product |

Both run in $O(n \log n)$.  In practice Andrew's monotone chain is often
preferred because it avoids trigonometric functions.

## Reference

- R. L. Graham, "An Efficient Algorithm for Determining the Convex Hull of
  a Finite Planar Set," *Information Processing Letters*, 1(4), 1972.
- de Berg, Cheong, van Kreveld, Overmars, *Computational Geometry*, 3rd
  ed., Springer, 2008.
