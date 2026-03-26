# Sweep-Line Segment Intersection

Given $n$ line segments in the plane, how many pairs intersect? The brute-force
approach tests all $\binom{n}{2}$ pairs in $O(n^2)$ time. The
Bentley-Ottmann sweep-line algorithm finds all $k$ intersection points in
$O((n + k) \log n)$ time by exploiting the fact that two segments can only
intersect when they are adjacent in the vertical ordering along a sweep line.

## Problem Statement

Given a set of $n$ line segments $S = \{s_1, s_2, \ldots, s_n\}$ in the plane,
report all pairs of segments that intersect. Let $k$ denote the number of
intersection points.

## Sweep-Line Paradigm

The algorithm sweeps a vertical line from left to right across the plane.
Two data structures maintain the state:

- **Event queue** (priority queue): stores the $x$-coordinates where something
  interesting happens. Events are left endpoints, right endpoints, and
  intersection points.
- **Status structure** (balanced BST): stores the segments currently crossing
  the sweep line, ordered by their $y$-coordinate at the current $x$-position.

!!! note "Key Observation"
    Two segments can only intersect if they are *adjacent* in the status
    structure at some point during the sweep. This means we only need to test
    neighboring pairs, not all pairs.

## Event Types

| Event | Action |
|---|---|
| Left endpoint of $s$ | Insert $s$ into status; check for intersections with its new neighbors |
| Right endpoint of $s$ | Remove $s$ from status; check if its former neighbors now intersect |
| Intersection of $s_i, s_j$ | Report intersection; swap $s_i$ and $s_j$ in status; check new neighbors |

## Algorithm Steps

1. Initialize the event queue with all $2n$ segment endpoints.
2. While the event queue is not empty:
    - Extract the leftmost event.
    - **Left endpoint:** Insert the segment into the status. Test it against its
      upper and lower neighbors.
    - **Right endpoint:** Before removing the segment, check if its upper and
      lower neighbors intersect. Then remove it.
    - **Intersection:** Report it. Swap the two segments in the status.
      Test the upper segment against its new upper neighbor, and the lower
      segment against its new lower neighbor.

## Correctness Argument

**Claim.** Before two segments intersect, they must become adjacent in the
status structure.

Two non-adjacent segments are separated by at least one other segment.
As the sweep line advances, segments swap positions only at intersection
points. Therefore, two segments that eventually intersect must first become
neighbors (by having the separating segments end or swap away). The algorithm
checks every pair when they become neighbors, so no intersection is missed.

## Complexity Analysis

- Each of the $2n$ endpoints generates one insertion or deletion: $O(n \log n)$.
- Each of the $k$ intersections generates one swap: $O(k \log n)$.
- Each event involves $O(1)$ neighbor checks and BST operations.

$$
T(n, k) = O((n + k) \log n)
$$

**Space:** $O(n + k)$ for the event queue and status structure.

When $k = O(n)$, this is $O(n \log n)$. In the worst case $k = O(n^2)$, so
the algorithm is $O(n^2 \log n)$ — but this is still better than brute force
for reporting intersections since we must output each one.

## Simplified Implementation

The full Bentley-Ottmann algorithm requires a balanced BST with custom
comparison. Below is a simplified sweep that detects intersections among
a small set of horizontal and vertical segments.

```python
"""
Sweep-line segment intersection (simplified).

Demonstrates the sweep-line paradigm for detecting intersections
between horizontal and vertical line segments.
"""

import heapq
from collections import defaultdict


# === Event-Based Sweep ===

def sweep_intersections(segments):
    """Find intersections among axis-aligned segments using sweep line.

    Args:
        segments: list of ((x1,y1), (x2,y2)) endpoint pairs.

    Returns:
        List of intersection points.
    """
    events = []
    horizontals = []

    for i, (p, q) in enumerate(segments):
        x1, y1 = p
        x2, y2 = q

        if y1 == y2:
            # Horizontal segment: start and end events
            lx, rx = min(x1, x2), max(x1, x2)
            events.append((lx, 0, y1, i))   # start
            events.append((rx, 2, y1, i))   # end
            horizontals.append((lx, rx, y1))
        else:
            # Vertical segment: query event
            x = x1
            ly, ry = min(y1, y2), max(y1, y2)
            events.append((x, 1, (ly, ry), i))  # vertical

    events.sort()
    active_y = defaultdict(int)
    intersections = []

    for event in events:
        x = event[0]
        etype = event[1]

        if etype == 0:
            # Horizontal start: add y to active set
            y = event[2]
            active_y[y] += 1
        elif etype == 2:
            # Horizontal end: remove y from active set
            y = event[2]
            active_y[y] -= 1
            if active_y[y] == 0:
                del active_y[y]
        else:
            # Vertical segment: check active horizontals
            ly, ry = event[2]
            for y in list(active_y.keys()):
                if ly <= y <= ry and active_y[y] > 0:
                    intersections.append((x, y))

    return intersections


# === Main ===

if __name__ == "__main__":
    # Horizontal and vertical segments
    segments = [
        ((1, 3), (6, 3)),   # horizontal
        ((2, 1), (2, 5)),   # vertical
        ((4, 1), (4, 4)),   # vertical
        ((1, 1), (5, 1)),   # horizontal
    ]

    result = sweep_intersections(segments)
    print("Segments:")
    for s in segments:
        print(f"  {s[0]} -> {s[1]}")
    print(f"Intersections: {result}")
```

**Output:**
```
Segments:
  (1, 3) -> (6, 3)
  (2, 1) -> (2, 5)
  (4, 1) -> (4, 4)
  (1, 1) -> (5, 1)
Intersections: [(2, 1), (2, 3), (4, 1), (4, 3)]
```

## Comparison of Approaches

| Algorithm | Time | Handles General Segments? |
|---|---|---|
| Brute force | $O(n^2)$ | Yes |
| Bentley-Ottmann | $O((n+k) \log n)$ | Yes |
| Shamos-Hoey | $O(n \log n)$ | Detection only (yes/no) |

## Reference

- de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M. *Computational Geometry: Algorithms and Applications*. Springer, Chapter 2.
- Bentley, J. L. & Ottmann, T. A. "Algorithms for Reporting and Counting Geometric Intersections." *IEEE Trans. Computers*, 1979.
