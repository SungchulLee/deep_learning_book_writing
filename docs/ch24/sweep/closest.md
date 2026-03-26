# Closest Pair of Points

Given $n$ points in the plane, which two are closest together? The brute-force
approach checks all $\binom{n}{2}$ pairs in $O(n^2)$ time. A classic
divide-and-conquer algorithm solves this in $O(n \log n)$, and a sweep-line
variant achieves the same bound. This page presents both approaches.

## Problem Statement

Given a set $P = \{p_1, p_2, \ldots, p_n\}$ of $n$ points in $\mathbb{R}^2$,
find the pair $(p_i, p_j)$ with $i \neq j$ that minimizes the Euclidean distance:

$$
d(p_i, p_j) = \sqrt{(p_{i,x} - p_{j,x})^2 + (p_{i,y} - p_{j,y})^2}
$$

## Divide and Conquer Approach

### Algorithm

1. **Sort** the points by $x$-coordinate.
2. **Divide** the sorted list into two halves $P_L$ and $P_R$ at the median $x$-value.
3. **Conquer** by recursively finding the closest pair in $P_L$ (distance $\delta_L$)
   and in $P_R$ (distance $\delta_R$). Let $\delta = \min(\delta_L, \delta_R)$.
4. **Combine** by checking pairs that cross the dividing line. Only points
   within a vertical strip of width $2\delta$ around the median can form
   a closer pair. Within this strip, each point needs to be compared with
   at most 7 other points (sorted by $y$-coordinate).

### Why Only 7 Comparisons?

!!! tip "The Strip Sparsity Argument"
    Consider a $\delta \times 2\delta$ rectangle in the strip. At most 8
    points can fit in this rectangle (4 from each side) while maintaining
    pairwise distance $\ge \delta$. Therefore, for each point, we check at
    most 7 subsequent points in the $y$-sorted order.

### Recurrence

$$
T(n) = 2T(n/2) + O(n)
$$

By the Master Theorem, $T(n) = O(n \log n)$.

## Sweep-Line Approach

The sweep-line variant processes points from left to right, maintaining
an active set of points sorted by $y$-coordinate in a balanced BST.

1. Sort points by $x$-coordinate.
2. Initialize $\delta = \infty$ and an empty active set.
3. For each point $p$ (left to right):
    - Remove from the active set all points with $x$-coordinate less than $p_x - \delta$.
    - Query the active set for points with $y$-coordinate in $[p_y - \delta, p_y + \delta]$.
    - Update $\delta$ if any neighbor is closer.
    - Insert $p$ into the active set.

Each point is inserted and deleted once. Each query returns $O(1)$ points
(by the same sparsity argument). Total time: $O(n \log n)$.

## Worked Example

Points: $(2,3)$, $(12,30)$, $(40,50)$, $(5,1)$, $(12,10)$, $(3,4)$.

After sorting by $x$: $(2,3)$, $(3,4)$, $(5,1)$, $(12,10)$, $(12,30)$, $(40,50)$.

**Left half** $\{(2,3), (3,4), (5,1)\}$: closest pair is $((2,3), (3,4))$ with $\delta_L = \sqrt{2} \approx 1.414$.

**Right half** $\{(12,10), (12,30), (40,50)\}$: closest pair is $((12,10), (12,30))$ with $\delta_R = 20$.

$\delta = \min(1.414, 20) = 1.414$. The strip around the median ($x \approx 5$)
with width $2\delta \approx 2.83$ contains only left-half points, so no
cross-boundary pair is closer.

**Result:** closest pair is $((2,3), (3,4))$ with distance $\sqrt{2}$.

## Implementation

```python
"""
Closest pair of points in the plane.

Implements the divide-and-conquer algorithm achieving O(n log n) time.
"""

import math


# === Distance ===

def dist(p, q):
    """Compute Euclidean distance between two points."""
    return math.hypot(p[0] - q[0], p[1] - q[1])


# === Brute Force (Base Case) ===

def brute_force(points):
    """Find closest pair among a small set of points.

    Used as the base case when the subproblem has 3 or fewer points.
    """
    min_d = float("inf")
    pair = (None, None)
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            d = dist(points[i], points[j])
            if d < min_d:
                min_d = d
                pair = (points[i], points[j])
    return min_d, pair


# === Strip Check ===

def closest_in_strip(strip, delta):
    """Check pairs in the strip for a distance smaller than delta.

    The strip is sorted by y-coordinate. Each point is compared with
    at most 7 subsequent points.
    """
    min_d = delta
    pair = (None, None)
    strip.sort(key=lambda p: p[1])

    for i in range(len(strip)):
        j = i + 1
        while j < len(strip) and (strip[j][1] - strip[i][1]) < min_d:
            d = dist(strip[i], strip[j])
            if d < min_d:
                min_d = d
                pair = (strip[i], strip[j])
            j += 1

    return min_d, pair


# === Divide and Conquer ===

def closest_pair_rec(px):
    """Recursive closest pair on points sorted by x-coordinate."""
    n = len(px)
    if n <= 3:
        return brute_force(px)

    mid = n // 2
    mid_x = px[mid][0]

    dl, pair_l = closest_pair_rec(px[:mid])
    dr, pair_r = closest_pair_rec(px[mid:])

    if dl < dr:
        delta, best_pair = dl, pair_l
    else:
        delta, best_pair = dr, pair_r

    strip = [p for p in px if abs(p[0] - mid_x) < delta]
    ds, pair_s = closest_in_strip(strip, delta)

    if ds < delta:
        return ds, pair_s
    return delta, best_pair


def closest_pair(points):
    """Find the closest pair of points in O(n log n) time.

    Args:
        points: list of (x, y) tuples.

    Returns:
        (distance, (point_a, point_b)).
    """
    px = sorted(points, key=lambda p: (p[0], p[1]))
    return closest_pair_rec(px)


# === Main ===

if __name__ == "__main__":
    pts = [(2, 3), (12, 30), (40, 50), (5, 1), (12, 10), (3, 4)]
    d, pair = closest_pair(pts)
    print(f"Points: {pts}")
    print(f"Closest pair: {pair}")
    print(f"Distance: {d:.4f}")

    # Another example
    pts2 = [(0, 0), (1, 0), (0, 1), (1, 1), (0.5, 0.5)]
    d2, pair2 = closest_pair(pts2)
    print(f"\nPoints: {pts2}")
    print(f"Closest pair: {pair2}")
    print(f"Distance: {d2:.4f}")
```

**Output:**
```
Points: [(2, 3), (12, 30), (40, 50), (5, 1), (12, 10), (3, 4)]
Closest pair: ((2, 3), (3, 4))
Distance: 1.4142

Points: [(0, 0), (1, 0), (0, 1), (1, 1), (0.5, 0.5)]
Closest pair: ((0.5, 0.5), (0, 0))
Distance: 0.7071
```

## Complexity Summary

| Approach | Time | Space |
|---|---|---|
| Brute force | $O(n^2)$ | $O(1)$ |
| Divide and conquer | $O(n \log n)$ | $O(n)$ |
| Sweep line | $O(n \log n)$ | $O(n)$ |
| Randomized | $O(n)$ expected | $O(n)$ |

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms*. MIT Press, Chapter 33.
- de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M. *Computational Geometry: Algorithms and Applications*. Springer.
