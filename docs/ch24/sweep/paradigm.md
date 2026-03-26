# Sweep-Line Paradigm

Many geometric problems involve objects spread across the plane with no
obvious ordering. The **sweep-line** (or plane sweep) paradigm imposes
an ordering by moving an imaginary vertical line from left to right across
the input, processing geometric events as the line encounters them. This
reduces a 2D problem to a sequence of 1D problems, often dropping the
complexity from $O(n^2)$ to $O(n \log n)$.

## Core Components

Every sweep-line algorithm has two data structures:

### Event Queue

A priority queue (typically a min-heap on $x$-coordinate) that stores
the events where the sweep line must stop. Events are processed in
left-to-right order. There are three common event types:

| Type | Trigger |
|---|---|
| **Start event** | Sweep line reaches the left endpoint of an object |
| **End event** | Sweep line passes the right endpoint of an object |
| **Interaction event** | Two objects interact (e.g., segments intersect) |

Start and end events are known in advance. Interaction events are
discovered dynamically and inserted into the queue during processing.

### Status Structure

A balanced BST (or similar ordered container) that maintains the set of
objects currently crossing the sweep line, ordered by their position along
the line. As the sweep line moves, objects are inserted, deleted, or
reordered in the status structure.

## Generic Sweep-Line Template

1. Initialize the event queue with all known events (e.g., endpoints).
2. Initialize an empty status structure.
3. While the event queue is not empty:
    - Extract the next event (smallest $x$-coordinate).
    - Update the status structure (insert, delete, or swap).
    - Check for new interaction events among neighbors in the status.
    - Insert any newly discovered events into the queue.

## Why It Works

The sweep-line paradigm works because geometric relationships are
*local*: two objects far apart in the status structure cannot interact
without first becoming neighbors. This locality means we only need to
check $O(1)$ pairs per event, even though the total number of objects
may be large.

## Complexity Pattern

Most sweep-line algorithms achieve the following:

$$
T(n) = O(n \log n + k \log n)
$$

where $n$ is the input size and $k$ is the output size (number of
interactions). The $\log n$ factor comes from priority queue and BST
operations.

## Classic Applications

| Problem | Events | Status | Time |
|---|---|---|---|
| Segment intersection | Endpoints + intersections | Segments by $y$ | $O((n+k) \log n)$ |
| Rectangle union area | Left/right edges | Active intervals | $O(n \log n)$ |
| Closest pair | Points by $x$ | Points by $y$ | $O(n \log n)$ |
| Voronoi diagram | Sites + circle events | Beach line arcs | $O(n \log n)$ |

## Example: Counting Active Intervals

As a minimal example, consider counting the maximum number of overlapping
intervals — a 1D sweep that captures the essence of the paradigm.

```python
"""
Sweep-line paradigm: interval overlap counting.

Demonstrates the sweep-line approach on a simple 1D problem:
finding the maximum number of overlapping intervals.
"""

import heapq


# === Sweep-Line Interval Counter ===

def max_overlap(intervals):
    """Find the maximum number of simultaneously active intervals.

    Args:
        intervals: list of (start, end) pairs.

    Returns:
        The maximum overlap count.
    """
    events = []
    for start, end in intervals:
        events.append((start, 1))   # start event: +1
        events.append((end, -1))    # end event: -1

    events.sort()

    max_count = 0
    current = 0
    for _, delta in events:
        current += delta
        max_count = max(max_count, current)

    return max_count


# === 2D Sweep: Active Segments ===

def sweep_active_count(segments):
    """Sweep left to right and report active segment counts at events.

    Args:
        segments: list of ((x1, y1), (x2, y2)) with x1 <= x2.

    Returns:
        List of (x, active_count) at each event.
    """
    events = []
    for i, ((x1, y1), (x2, y2)) in enumerate(segments):
        lx, rx = min(x1, x2), max(x1, x2)
        events.append((lx, 0, i))   # 0 = start
        events.append((rx, 1, i))   # 1 = end

    events.sort()
    active = set()
    result = []

    for x, etype, idx in events:
        if etype == 0:
            active.add(idx)
        else:
            active.discard(idx)
        result.append((x, len(active)))

    return result


# === Main ===

if __name__ == "__main__":
    # 1D interval overlap
    intervals = [(1, 5), (2, 7), (4, 6), (8, 10)]
    print(f"Intervals: {intervals}")
    print(f"Max overlap: {max_overlap(intervals)}")

    # 2D segment sweep
    segments = [
        ((1, 0), (5, 0)),
        ((2, 1), (7, 1)),
        ((4, 2), (6, 2)),
        ((8, 0), (10, 0)),
    ]
    print(f"\nSegment sweep events:")
    for x, count in sweep_active_count(segments):
        print(f"  x={x}: {count} active")
```

**Output:**
```
Intervals: [(1, 5), (2, 7), (4, 6), (8, 10)]
Max overlap: 3

Segment sweep events:
  x=1: 1 active
  x=2: 2 active
  x=4: 3 active
  x=5: 2 active
  x=6: 1 active
  x=7: 0 active
  x=8: 1 active
  x=10: 0 active
```

## Design Considerations

!!! tip "Choosing the Sweep Direction"
    While left-to-right is conventional, some problems benefit from a
    top-to-bottom or radial sweep. The choice depends on which direction
    reduces the problem to the simplest 1D subproblem.

!!! warning "Event Queue Discipline"
    When two events share the same $x$-coordinate, the tie-breaking
    order matters. A common convention is to process start events before
    end events (or vice versa), depending on the problem semantics.

## Reference

- de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M. *Computational Geometry: Algorithms and Applications*. Springer, Chapter 2.
- Preparata, F. P. & Shamos, M. I. *Computational Geometry: An Introduction*. Springer.
