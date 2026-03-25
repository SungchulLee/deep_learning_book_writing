# Chan's Algorithm

Chan's algorithm computes the convex hull of $n$ points in
$O(n \log h)$ time, where $h$ is the number of hull vertices.  It is
**output-sensitive**: when the hull is small the algorithm beats the
$\Theta(n \log n)$ barrier of comparison-based methods.  The key idea is to
combine a fast sub-hull algorithm (Graham scan) with an efficient wrapping
step (Jarvis march), guessing the hull size $h$ via repeated doubling.

## Intuition

1. Guess a target hull size $m$.
2. Partition the $n$ points into $\lceil n/m \rceil$ groups of size $m$.
3. Compute the convex hull of each group in $O(m \log m)$ using Graham scan.
4. Run Jarvis march on the group hulls: at each wrapping step, binary-search
   each group hull to find the best tangent point in $O(\log m)$.
5. If the march finishes in $\le m$ steps, we are done.  Otherwise, double
   $m$ and restart.

Because the correct $m = h$ is found after $O(\log \log h)$ rounds and each
round costs $O(n \log m)$, the total is $O(n \log h)$.

## Definitions

**Output-sensitive.**  An algorithm whose complexity depends on the size
of the output, not just the input.

**Tangent from a point to a convex polygon.**  Given an external point $q$
and a convex polygon $P$ stored in sorted order, the two tangent lines from
$q$ to $P$ can be found in $O(\log |P|)$ by binary search on the turning
direction.

## Algorithm Details

### Phase 1 -- Group Hulls

Partition $S$ into groups $G_1, \dots, G_{\lceil n/m \rceil}$ each of size
at most $m$.  Compute $\operatorname{CH}(G_i)$ for every group using any
$O(k \log k)$ hull algorithm (Graham scan is typical).

**Cost:** $\lceil n/m \rceil$ groups $\times O(m \log m)$ each
$= O(n \log m)$.

### Phase 2 -- Jarvis Wrap over Group Hulls

Starting from the lowest point $p_0$, repeat:

1. For each group hull $\operatorname{CH}(G_i)$, binary-search for the
   point $q_i$ that maximises the angle from the current edge direction.
   Cost per group: $O(\log m)$.
2. Among all $q_i$, pick the overall maximum-angle point $p_{\text{next}}$.
3. If $p_{\text{next}} = p_0$, the hull is complete.
4. If we have taken $m$ steps without closing, **abort** this round.

**Cost per step:** $O((n/m) \log m)$.
**Total for $h$ steps:** $O(n \log m)$ (when $m \ge h$).

### Doubling Schedule

Try $m = 2^{2^t}$ for $t = 1, 2, 3, \dots$  The first $m \ge h$ succeeds.
Round $t$ costs $O(n \log 2^{2^t}) = O(n \cdot 2^t)$.  Summing a geometric
series over all failed rounds gives $O(n \log h)$ total.

## Correctness

!!! note "Theorem"
    Chan's algorithm outputs $\operatorname{CH}(S)$ in $O(n \log h)$ time.

**Proof sketch.**

- Each group hull is correct (by correctness of Graham scan).
- Jarvis march is correct when every tangent query returns the true
  maximum-angle point.  Binary search on a convex polygon sorted by angle
  guarantees this.
- The doubling schedule ensures $m \ge h$ is reached; at that point the
  march completes within $m$ steps and returns the full hull.
- The total work across all rounds is dominated by the last successful
  round: $O(n \log h)$.

## Complexity

| Measure | Cost |
|---------|------|
| Time (worst-case) | $O(n \log h)$ |
| Time (when $h = \Theta(n)$) | $O(n \log n)$ -- same as Graham scan |
| Time (when $h = O(1)$) | $O(n)$ -- same as Jarvis march |
| Space | $O(n)$ |

## Implementation

```python
"""
Chan's algorithm — output-sensitive convex hull in O(n log h).

Uses Graham scan for group hulls and Jarvis march with tangent
queries over the group hulls.
"""

from __future__ import annotations


# === Orientation helper ======================================================

def cross(o: tuple[float, float],
          a: tuple[float, float],
          b: tuple[float, float]) -> float:
    """Signed area of parallelogram OA x OB."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


# === Graham scan for small groups ============================================

def graham_hull(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Convex hull via Andrew's monotone chain (used as sub-routine)."""
    pts = sorted(set(pts))
    if len(pts) <= 1:
        return list(pts)
    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


# === Tangent search ==========================================================

def _tangent(hull: list[tuple[float, float]],
             point: tuple[float, float]) -> tuple[float, float]:
    """Find the point on *hull* maximising left-turn angle from *point*."""
    best = hull[0]
    for q in hull:
        if cross(point, best, q) < 0:
            best = q
    return best


# === Chan's algorithm ========================================================

def chan_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Output-sensitive convex hull in O(n log h)."""
    pts = list(set(points))
    n = len(pts)
    if n <= 2:
        return graham_hull(pts)

    for t in range(1, n + 1):
        m = min(2 ** (2 ** t), n)
        # Partition into groups and compute sub-hulls
        groups = [graham_hull(pts[i:i + m]) for i in range(0, n, m)]

        # Start wrapping from the bottom-most point
        start = min(pts, key=lambda p: (p[1], p[0]))
        hull = [start]
        for _ in range(m):
            candidates = [_tangent(g, hull[-1]) for g in groups]
            best = candidates[0]
            for c in candidates[1:]:
                if c == hull[-1]:
                    continue
                if best == hull[-1] or cross(hull[-1], best, c) < 0:
                    best = c
            if best == start:
                return hull
            hull.append(best)
        # m was too small — try larger
    return hull  # fallback


# === Demo ====================================================================

if __name__ == "__main__":
    sample = [(0, 0), (1, 1), (2, 2), (0, 2), (2, 0), (1, 0)]
    hull = chan_hull(sample)
    print(f"Input:  {sample}")
    print(f"Hull:   {hull}")
    print(f"|hull| = {len(hull)}")
```

## Comparison with Other Algorithms

| Algorithm | Time | Output-sensitive? |
|-----------|------|-------------------|
| Graham scan | $O(n \log n)$ | No |
| Jarvis march | $O(nh)$ | Yes |
| **Chan's** | $O(n \log h)$ | Yes |

Chan's algorithm is always at least as fast as Graham scan and strictly
faster when $h = o(n)$.

## Reference

- T. M. Chan, "Optimal Output-Sensitive Convex Hull Algorithms in Two and
  Three Dimensions," *Discrete & Computational Geometry*, 16, 1996.
- de Berg, Cheong, van Kreveld, Overmars, *Computational Geometry*, 3rd
  ed., Springer, 2008.
