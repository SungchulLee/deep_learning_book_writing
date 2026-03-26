# Rectangle Union Area

Computing the total area covered by a set of axis-aligned rectangles is a
classic application of the sweep-line paradigm. Overlapping regions must be
counted only once, so simply summing individual areas gives the wrong answer.
A sweep-line combined with a segment tree (or simpler coordinate compression)
computes the exact union area in $O(n \log n)$ time.

## Problem Statement

Given $n$ axis-aligned rectangles, each defined by its lower-left corner
$(x_1, y_1)$ and upper-right corner $(x_2, y_2)$, compute the total area
of their union.

## Sweep-Line Strategy

Sweep a vertical line from left to right. At each $x$-coordinate where a
rectangle starts or ends, the set of active $y$-intervals changes.

1. **Events:** For each rectangle, create two events:
    - **Left edge** at $x = x_1$: add interval $[y_1, y_2]$.
    - **Right edge** at $x = x_2$: remove interval $[y_1, y_2]$.
2. **Sort** events by $x$-coordinate.
3. **Between events:** The total covered length along the $y$-axis
   (the union of active intervals) multiplied by $\Delta x$ gives the
   area contribution.

The key subproblem is maintaining the union length of active $y$-intervals
efficiently. A segment tree achieves $O(\log n)$ per update.

## Coordinate Compression

!!! tip "Reducing to Discrete Coordinates"
    Since we only care about $y$-values that appear as rectangle boundaries,
    we compress the $y$-coordinates to indices $0, 1, \ldots, m-1$ where $m$
    is the number of distinct $y$-values. Each "cell" in the compressed grid
    represents an interval between consecutive $y$-values.

## Complexity

| Component | Time |
|---|---|
| Sorting events | $O(n \log n)$ |
| Processing $2n$ events | $O(n \log n)$ with segment tree |
| **Total** | $O(n \log n)$ |

Space: $O(n)$.

## Worked Example

Three rectangles:
- $R_1$: $(1,1)$ to $(4,3)$ — area $= 6$
- $R_2$: $(2,2)$ to $(5,5)$ — area $= 9$
- $R_3$: $(3,0)$ to $(6,2)$ — area $= 6$

Sum of individual areas $= 21$, but the union area is smaller due to overlaps.

Events sorted by $x$: $x=1$ (add $R_1$), $x=2$ (add $R_2$), $x=3$ (add $R_3$),
$x=4$ (remove $R_1$), $x=5$ (remove $R_2$), $x=6$ (remove $R_3$).

| $x$-interval | Active intervals on $y$ | Union length | $\Delta x$ | Area |
|---|---|---|---|---|
| $[1,2)$ | $[1,3]$ | $2$ | $1$ | $2$ |
| $[2,3)$ | $[1,3] \cup [2,5]$ = $[1,5]$ | $4$ | $1$ | $4$ |
| $[3,4)$ | $[1,5] \cup [0,2]$ = $[0,5]$ | $5$ | $1$ | $5$ |
| $[4,5)$ | $[2,5] \cup [0,2]$ = $[0,5]$ | $5$ | $1$ | $5$ |
| $[5,6)$ | $[0,2]$ | $2$ | $1$ | $2$ |

Union area $= 2 + 4 + 5 + 5 + 2 = 18$.

## Implementation

```python
"""
Rectangle union area via sweep line with coordinate compression.

Sweeps left to right, maintaining a count array over compressed
y-intervals to compute the total covered y-length at each step.
"""


# === Sweep-Line Rectangle Union ===

def rectangle_union_area(rectangles):
    """Compute the area of the union of axis-aligned rectangles.

    Args:
        rectangles: list of (x1, y1, x2, y2) tuples.

    Returns:
        Total union area.
    """
    if not rectangles:
        return 0

    # Collect events
    events = []
    ys = set()
    for x1, y1, x2, y2 in rectangles:
        events.append((x1, 0, y1, y2))  # 0 = left edge (add)
        events.append((x2, 1, y1, y2))  # 1 = right edge (remove)
        ys.add(y1)
        ys.add(y2)

    events.sort()
    ys = sorted(ys)
    y_index = {y: i for i, y in enumerate(ys)}
    m = len(ys) - 1  # number of intervals

    # Count array: how many rectangles cover each y-interval
    count = [0] * m

    total_area = 0.0
    prev_x = events[0][0]

    for x, etype, y1, y2 in events:
        # Compute covered y-length
        covered = sum(
            ys[i + 1] - ys[i] for i in range(m) if count[i] > 0
        )
        total_area += covered * (x - prev_x)
        prev_x = x

        # Update counts
        lo = y_index[y1]
        hi = y_index[y2]
        delta = 1 if etype == 0 else -1
        for i in range(lo, hi):
            count[i] += delta

    return total_area


# === Main ===

if __name__ == "__main__":
    rects = [
        (1, 1, 4, 3),
        (2, 2, 5, 5),
        (3, 0, 6, 2),
    ]

    area = rectangle_union_area(rects)
    print("Rectangles:")
    for r in rects:
        print(f"  ({r[0]},{r[1]}) to ({r[2]},{r[3]})")
    print(f"Union area: {area}")

    # Non-overlapping rectangles
    rects2 = [(0, 0, 1, 1), (2, 2, 3, 3)]
    print(f"\nNon-overlapping: area = {rectangle_union_area(rects2)}")

    # Fully overlapping rectangles
    rects3 = [(0, 0, 4, 4), (1, 1, 3, 3)]
    print(f"Fully overlapping: area = {rectangle_union_area(rects3)}")
```

**Output:**
```
Rectangles:
  (1,1) to (4,3)
  (2,2) to (5,5)
  (3,0) to (6,2)
Union area: 18.0

Non-overlapping: area = 2.0
Fully overlapping: area = 16.0
```

## Optimization with Segment Tree

The simple implementation above recomputes the covered length in $O(m)$
per event, giving $O(nm)$ total. For large inputs, replace the count
array with a segment tree that maintains:

- **count[node]:** number of full covers of this interval
- **covered[node]:** total length of covered sub-intervals

Each update runs in $O(\log m)$, reducing the total to $O(n \log n)$.

## Reference

- de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M. *Computational Geometry: Algorithms and Applications*. Springer.
- Bentley, J. L. "Algorithms for Klee's Rectangle Problems." Unpublished manuscript, 1977.
