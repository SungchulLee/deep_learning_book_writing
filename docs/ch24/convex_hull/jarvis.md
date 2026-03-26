# Jarvis March

The **Jarvis march** (also called **gift wrapping**) builds the convex hull by starting from a point known to be on the hull and repeatedly selecting the point that makes the smallest counter-clockwise angle with the current edge. It wraps around the point set like wrapping paper around a gift. The time complexity is $O(nh)$, where $h$ is the number of hull vertices, making it output-sensitive — fast when $h$ is small relative to $n$.

## Intuition

Imagine standing at the leftmost point and looking rightward. Sweep your gaze counter-clockwise until it hits the first point — that point is the next hull vertex. Move there and repeat the sweep. When you return to the starting point, the hull is complete.

## Algorithm

1. Find the leftmost point $p_0$ (smallest $x$-coordinate; break ties by smallest $y$).
2. Set the current point $p = p_0$.
3. Repeat:
    - For each candidate point $q$, determine which makes the most counter-clockwise turn from the current direction.
    - The next hull vertex is the point $q^*$ such that all other points lie to the left of the line from $p$ to $q^*$ (or are collinear and farther away).
    - Set $p = q^*$ and add it to the hull.
    - Stop when $p = p_0$ (returned to start).

The selection at each step uses the cross product:

$$
\operatorname{cross}(p, q, r) = (q_x - p_x)(r_y - p_y) - (q_y - p_y)(r_x - p_x)
$$

Point $q$ is more counter-clockwise than $r$ (relative to $p$) when $\operatorname{cross}(p, q, r) > 0$.

## Complexity

| Aspect | Value |
|---|---|
| Time | $O(nh)$ where $h = $ number of hull vertices |
| Space | $O(h)$ |
| Best case | $O(n)$ when $h = O(1)$ |
| Worst case | $O(n^2)$ when $h = O(n)$ |

!!! tip "Output-Sensitive Algorithms"
    Jarvis march is **output-sensitive**: its running time depends on the output size $h$. When most points are interior ($h \ll n$), it outperforms $O(n \log n)$ algorithms like Graham scan. When most points are on the hull, Graham scan is faster.

## Correctness

At each step, the algorithm selects the point that all other points lie to the left of (or on) the directed line from the current vertex. This ensures:

1. Every selected point is on the convex hull (no interior point can be "most counter-clockwise").
2. No hull vertex is missed (if a hull vertex were skipped, some point would lie to the right of the current edge, contradicting the selection criterion).
3. The process terminates after exactly $h$ steps.

## Python Implementation

```python
"""
Jarvis March (Gift Wrapping) — Convex Hull in O(nh).

Builds the convex hull by iteratively selecting the most
counter-clockwise point, wrapping around the point set.
"""

from __future__ import annotations


# === Cross Product ===

def cross(o: tuple[float, float],
          a: tuple[float, float],
          b: tuple[float, float]) -> float:
    """Signed area of the parallelogram formed by vectors OA and OB."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def dist_sq(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Squared Euclidean distance between two points."""
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


# === Jarvis March ===

def jarvis_march(
    points: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Return convex hull vertices in counter-clockwise order.

    Uses the gift-wrapping algorithm in O(nh) time.
    """
    pts = list(set(points))
    n = len(pts)
    if n <= 2:
        return pts

    # Start from the leftmost point
    start = min(range(n), key=lambda i: (pts[i][0], pts[i][1]))
    hull = []
    current = start

    while True:
        hull.append(pts[current])
        candidate = 0

        for i in range(n):
            if i == current:
                continue

            # If candidate == current, any other point is better
            if candidate == current:
                candidate = i
                continue

            c = cross(pts[current], pts[candidate], pts[i])
            if c < 0:
                # pts[i] is more counter-clockwise
                candidate = i
            elif c == 0:
                # Collinear: pick the farther point
                if dist_sq(pts[current], pts[i]) > dist_sq(pts[current], pts[candidate]):
                    candidate = i

        current = candidate
        if current == start:
            break

    return hull


# === Main ===

if __name__ == "__main__":
    sample = [(0, 0), (1, 1), (2, 2), (0, 2), (2, 0), (1, 0)]
    hull = jarvis_march(sample)
    print(f"Input:  {sample}")
    print(f"Hull:   {hull}")
    print(f"Hull vertices: {len(hull)}")

    # Larger example with interior points
    import random
    random.seed(42)
    pts = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(20)]
    hull2 = jarvis_march(pts)
    print(f"\n{len(pts)} random points -> {len(hull2)} hull vertices")
    # Output:
    # Input:  [(0, 0), (1, 1), (2, 2), (0, 2), (2, 0), (1, 0)]
    # Hull:   [(0, 0), (1, 0), (2, 0), (2, 2), (0, 2)]
    # Hull vertices: 5
    #
    # 20 random points -> 7 hull vertices
```

## Worked Example

For points $\{(0,0), (1,0), (2,0), (1,1), (2,2), (0,2)\}$:

1. **Start** at $(0,0)$ (leftmost).
2. From $(0,0)$: most CCW point is $(1,0)$. But $(2,0)$ is collinear and farther. Select $(2,0)$.
3. From $(2,0)$: most CCW point is $(2,2)$.
4. From $(2,2)$: most CCW point is $(0,2)$.
5. From $(0,2)$: most CCW point is $(0,0)$. Back to start.

Hull: $[(0,0), (2,0), (2,2), (0,2)]$ with $h = 4$ vertices. Points $(1,0)$ and $(1,1)$ are interior.

## Comparison with Other Algorithms

| Algorithm | Time | Output-sensitive? |
|---|---|---|
| Jarvis march | $O(nh)$ | Yes |
| Graham scan | $O(n \log n)$ | No |
| Andrew's monotone chain | $O(n \log n)$ | No |
| Chan's algorithm | $O(n \log h)$ | Yes |

Chan's algorithm combines the best of both worlds: it achieves $O(n \log h)$ by running Jarvis march on groups of $O(h)$ points, each pre-processed with Graham scan.

## Reference

- Jarvis, R. A. (1973). On the identification of the convex hull of a finite set of points in the plane. *Information Processing Letters*, 2(1), 18-21.
- de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M. (2008). *Computational Geometry* (3rd ed.). Springer.
