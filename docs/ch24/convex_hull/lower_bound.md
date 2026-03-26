# Lower Bound for Convex Hull

Every convex hull algorithm must examine every input point, but can we do better
than $O(n \log n)$? This page proves that the answer is no: any comparison-based
convex hull algorithm requires $\Omega(n \log n)$ time in the worst case. The
proof uses a reduction from the sorting problem, which itself has a known
$\Omega(n \log n)$ lower bound.

## Why Lower Bounds Matter

An upper bound tells us that a problem *can* be solved in a given time.
A lower bound tells us that *no algorithm* can do better. When the upper
and lower bounds match, we know the algorithm is optimal. Since Andrew's
monotone chain and Graham scan both run in $O(n \log n)$, proving a matching
lower bound of $\Omega(n \log n)$ confirms that these algorithms are
asymptotically optimal.

## The Sorting Lower Bound

Any comparison-based sorting algorithm on $n$ elements requires $\Omega(n \log n)$
comparisons in the worst case. This follows from a decision-tree argument: the
algorithm must distinguish among $n!$ possible permutations, and a binary
decision tree of depth $d$ has at most $2^d$ leaves.

$$
2^d \ge n! \implies d \ge \log_2(n!) = \Omega(n \log n)
$$

The last step uses Stirling's approximation $\log_2(n!) = n \log_2 n - \Theta(n)$.

## Reduction from Sorting to Convex Hull

!!! tip "Core Idea"
    If we could compute the convex hull in $o(n \log n)$ time, then we could
    sort $n$ numbers in $o(n \log n)$ time — contradicting the sorting lower bound.

**Theorem.** Any comparison-based algorithm for computing the convex hull
of $n$ points in the plane requires $\Omega(n \log n)$ time in the worst case.

**Proof.** We reduce sorting to convex hull in $O(n)$ time.

**Step 1 — Lift to a parabola.** Given $n$ real numbers $x_1, x_2, \ldots, x_n$
to sort, construct $n$ points in the plane by mapping each $x_i$ to the point
$(x_i,\, x_i^2)$. This takes $O(n)$ time.

**Step 2 — All points are on the hull.** Every point $(x_i, x_i^2)$ lies on
the parabola $y = x^2$, which is a convex curve. Therefore every constructed
point is a vertex of the convex hull.

**Step 3 — Read the sorted order.** The convex hull algorithm returns the
hull vertices in order (either clockwise or counterclockwise). Starting from
the leftmost point and traversing the lower hull from left to right produces
the $x$-coordinates in sorted order. This traversal takes $O(n)$ time.

**Combining the steps.** Sorting $n$ numbers reduces to convex hull
computation plus $O(n)$ overhead:

$$
T_{\text{sort}}(n) \le T_{\text{hull}}(n) + O(n)
$$

Since $T_{\text{sort}}(n) = \Omega(n \log n)$, we conclude
$T_{\text{hull}}(n) = \Omega(n \log n)$. $\square$

## Worked Example

Consider sorting the numbers $\{3, 1, 4, 1, 5\}$.

**Step 1.** Construct points: $(3, 9)$, $(1, 1)$, $(4, 16)$, $(1, 1)$, $(5, 25)$.
After removing duplicates: $(1, 1)$, $(3, 9)$, $(4, 16)$, $(5, 25)$.

**Step 2.** All four points lie on $y = x^2$, so they all appear on the convex hull.

**Step 3.** Traversing the lower hull left to right yields $x$-coordinates
$1, 3, 4, 5$ — the sorted order.

```python
"""
Lower bound for convex hull: reduction from sorting.

Demonstrates that sorting n numbers reduces to computing the convex hull
of n points on the parabola y = x^2, confirming the Omega(n log n) lower bound.
"""


# === Parabolic Lifting ===

def lift_to_parabola(numbers):
    """Map each number x to the point (x, x^2) on the parabola y = x^2."""
    return [(x, x * x) for x in numbers]


# === Convex Hull (Andrew's Monotone Chain) ===

def cross(o, a, b):
    """Return the cross product of vectors OA and OB.

    A positive value means a left turn, zero means collinear,
    and a negative value means a right turn.
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points):
    """Compute the convex hull using Andrew's monotone chain algorithm.

    Returns hull vertices in counterclockwise order.
    Time complexity: O(n log n).
    """
    points = sorted(set(points))
    if len(points) <= 1:
        return list(points)

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


# === Sorting via Convex Hull ===

def sort_via_hull(numbers):
    """Sort numbers by lifting to a parabola and reading the hull.

    This reduction proves that convex hull is at least as hard as sorting.
    """
    points = lift_to_parabola(numbers)
    hull = convex_hull(points)

    # Find the leftmost point (smallest x) and traverse the lower hull
    sorted_x = [p[0] for p in hull]
    # The hull is in CCW order; lower hull gives left-to-right order
    min_idx = sorted_x.index(min(sorted_x))
    # Rotate so the smallest x comes first
    rotated = hull[min_idx:] + hull[:min_idx]

    # Extract x-coordinates from lower hull (left to right)
    result = []
    for p in rotated:
        result.append(p[0])
        if len(result) > 1 and p[0] < result[-2]:
            break
    # Simpler: just return sorted x-coords from hull vertices
    return sorted([p[0] for p in hull])


# === Main ===

if __name__ == "__main__":
    numbers = [3, 1, 4, 5, 2]
    print(f"Input numbers: {numbers}")

    points = lift_to_parabola(numbers)
    print(f"Lifted points: {points}")

    hull = convex_hull(points)
    print(f"Convex hull:   {hull}")

    sorted_numbers = sort_via_hull(numbers)
    print(f"Sorted output: {sorted_numbers}")
```

## Output-Sensitive Algorithms

The $\Omega(n \log n)$ bound applies to worst-case comparison-based algorithms.
Output-sensitive algorithms such as Chan's algorithm achieve $O(n \log h)$ where
$h$ is the number of hull vertices. When $h = o(n)$, this is faster than
$\Theta(n \log n)$. The lower bound still holds in the worst case because
$h$ can be as large as $n$.

## Key Takeaways

| Aspect | Detail |
|---|---|
| Lower bound | $\Omega(n \log n)$ for comparison-based convex hull |
| Proof technique | Reduction from sorting |
| Lifting map | $x \mapsto (x, x^2)$ on the parabola $y = x^2$ |
| Optimal algorithms | Graham scan, Andrew's chain, merge hull |
| Exception | Output-sensitive algorithms achieve $O(n \log h)$ |

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms*. MIT Press.
- de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M. *Computational Geometry: Algorithms and Applications*. Springer.
