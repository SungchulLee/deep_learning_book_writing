# Closest Pair of Points

Given $n$ points in the plane, finding the pair with the smallest Euclidean distance by brute force requires checking all $\binom{n}{2}$ pairs in $O(n^2)$ time. A divide-and-conquer approach achieves $O(n \log n)$, matching the lower bound for comparison-based algorithms. The key challenge lies in the **combine step**, where a clever geometric argument limits the number of cross-boundary pairs to examine.

## Problem Statement

Given a set $P = \{p_1, p_2, \dots, p_n\}$ of points in $\mathbb{R}^2$, find:

$$
\min_{i \ne j} d(p_i, p_j) = \min_{i \ne j} \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}
$$

## Divide-and-Conquer Algorithm

**Step 1: Sort.** Sort all points by $x$-coordinate. Also maintain a copy sorted by $y$-coordinate.

**Step 2: Divide.** Split $P$ into two halves $P_L$ and $P_R$ at the median $x$-coordinate.

**Step 3: Conquer.** Recursively find the closest pair in $P_L$ (distance $\delta_L$) and in $P_R$ (distance $\delta_R$). Let $\delta = \min(\delta_L, \delta_R)$.

**Step 4: Combine.** Check whether any pair with one point in $P_L$ and the other in $P_R$ has distance less than $\delta$. This is where the algorithm's efficiency depends on a geometric insight.

## The Strip Argument

Only points within distance $\delta$ of the dividing line can form a closer pair. Define the **strip**:

$$
S = \{p \in P : |p.x - x_{\text{mid}}| < \delta\}
$$

Sort the points in $S$ by $y$-coordinate. For each point $p$ in $S$, compare it only to points within $\delta$ in the $y$-direction.

!!! note "Sparsity Lemma"
    For any point $p$ in the strip, at most **7** other points in $S$ lie within a $\delta \times 2\delta$ rectangle centered at $p$. Therefore, the inner loop examines at most 7 candidates per point.

The proof uses a packing argument: a $\delta \times 2\delta$ rectangle can be divided into eight $(\delta/2) \times (\delta/2)$ sub-squares. Each sub-square contains at most one point (since any two points in the same half have distance at least $\delta$), so at most $8 - 1 = 7$ other points exist in the rectangle.

This means the combine step takes $O(|S|)$ time (linear in the strip size), and the overall recurrence is:

$$
T(n) = 2T(n/2) + O(n) = O(n \log n)
$$

## Implementation

```python
"""
Closest pair of points in the plane using divide and conquer.

Achieves O(n log n) time by exploiting the sparsity of the strip
region in the combine step.
"""

import math

# === Closest Pair Algorithm ===

def closest_pair(points: list[tuple[float, float]]) -> float:
    """Find the distance of the closest pair of points.

    Args:
        points: List of (x, y) coordinates.

    Returns:
        Minimum Euclidean distance between any two points.
    """
    def dist(p1: tuple, p2: tuple) -> float:
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _solve(px: list, py: list) -> float:
        n = len(px)
        if n <= 3:
            # Brute force for small cases
            best = float('inf')
            for i in range(n):
                for j in range(i + 1, n):
                    best = min(best, dist(px[i], px[j]))
            return best

        mid = n // 2
        mid_x = px[mid][0]

        # Split py into left and right, maintaining y-sort order
        pyl = [p for p in py if p[0] <= mid_x]
        pyr = [p for p in py if p[0] > mid_x]

        # Handle ties at the midpoint
        if len(pyl) > mid:
            excess = len(pyl) - mid
            pyr = [p for p in pyl if p[0] == mid_x][-excess:] + pyr
            pyl = [p for p in pyl if p[0] < mid_x] + \
                  [p for p in pyl if p[0] == mid_x][:-excess]

        dl = _solve(px[:mid], pyl)
        dr = _solve(px[mid:], pyr)
        delta = min(dl, dr)

        # Build strip sorted by y-coordinate
        strip = [p for p in py if abs(p[0] - mid_x) < delta]

        # Check strip pairs (at most 7 comparisons per point)
        for i in range(len(strip)):
            j = i + 1
            while j < len(strip) and strip[j][1] - strip[i][1] < delta:
                delta = min(delta, dist(strip[i], strip[j]))
                j += 1

        return delta

    px = sorted(points, key=lambda p: p[0])
    py = sorted(points, key=lambda p: p[1])
    return _solve(px, py)


# === Demonstration ===

if __name__ == "__main__":
    points = [
        (2.0, 3.0), (12.0, 30.0), (40.0, 50.0),
        (5.0, 1.0), (12.0, 10.0), (3.0, 4.0),
    ]
    result = closest_pair(points)
    print(f"Closest pair distance: {result:.4f}")

    # Verify with brute force
    best = float('inf')
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = math.sqrt((points[i][0]-points[j][0])**2 +
                          (points[i][1]-points[j][1])**2)
            best = min(best, d)
    print(f"Brute force distance: {best:.4f}")
```

**Output:**

```
Closest pair distance: 1.4142
Brute force distance: 1.4142
```

The closest pair is $(2, 3)$ and $(3, 4)$ with distance $\sqrt{2} \approx 1.4142$. Both the divide-and-conquer and brute-force approaches find the same answer.

## Complexity

| Aspect | Cost |
|--------|:----:|
| Time   | $O(n \log n)$ |
| Space  | $O(n)$ |

The initial sort takes $O(n \log n)$. The recurrence $T(n) = 2T(n/2) + O(n)$ solves to $O(n \log n)$ by the master theorem.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 33: Computational Geometry.
- Shamos, M. I., & Hoey, D. (1975). Closest-point problems. *IEEE Symposium on FOCS*, pp. 151--162.
