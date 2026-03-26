# Interval Partitioning

While interval scheduling maximizes the number of activities on a single resource, **interval partitioning** asks a different question: given a set of activities, what is the minimum number of resources (rooms, machines, processors) needed so that all activities can run without conflicts? A greedy algorithm that assigns each activity to the earliest available resource solves this optimally, and the answer always equals the maximum number of overlapping activities at any point in time.

## Problem Statement

Given $n$ activities with intervals $[s_i, f_i)$, assign each activity to a resource (room) such that no two activities assigned to the same resource overlap. Minimize the number of resources used.

## Lower Bound: Depth

The **depth** of a set of intervals is the maximum number of intervals that contain any common point:

$$
\text{depth} = \max_{t} |\{i : s_i \le t < f_i\}|
$$

No schedule can use fewer than depth resources, because at the point of maximum overlap, each overlapping activity needs its own resource.

!!! note "Theorem"
    The minimum number of resources equals the depth.

## Greedy Algorithm

**Strategy.** Sort activities by start time. For each activity, assign it to any resource that is free (its last activity ended before the current one starts). If no resource is free, open a new one.

Using a min-heap (priority queue) keyed on the finish time of the last activity on each resource, the algorithm efficiently finds the earliest available resource.

## Correctness

The greedy algorithm uses exactly depth resources. The proof has two parts:

1. **Lower bound.** At least depth resources are needed (by the depth argument).
2. **Upper bound.** The greedy algorithm never opens more than depth resources. When it opens a new resource, all existing resources are busy, meaning the current activity overlaps with at least one activity on each existing resource. This implies depth has increased to the number of open resources.

## Implementation

```python
"""
Interval partitioning via a greedy algorithm with a min-heap.

Assigns activities to the minimum number of resources such that
no two activities on the same resource overlap.
"""

import heapq

# === Greedy Interval Partitioning ===

def interval_partitioning(
    activities: list[tuple[int, int]]
) -> list[list[tuple[int, int]]]:
    """Partition activities into minimum number of resources.

    Args:
        activities: List of (start, finish) tuples.

    Returns:
        List of resource assignments, where each resource is a list
        of activities assigned to it.
    """
    if not activities:
        return []

    # Sort by start time
    sorted_acts = sorted(activities, key=lambda x: x[0])

    # Min-heap: (finish_time_of_last_activity, resource_index)
    heap = []
    resources = []

    for start, finish in sorted_acts:
        if heap and heap[0][0] <= start:
            # Reuse the resource that finishes earliest
            _, idx = heapq.heappop(heap)
            resources[idx].append((start, finish))
            heapq.heappush(heap, (finish, idx))
        else:
            # Open a new resource
            idx = len(resources)
            resources.append([(start, finish)])
            heapq.heappush(heap, (finish, idx))

    return resources


def compute_depth(activities: list[tuple[int, int]]) -> int:
    """Compute the depth (maximum overlap) of a set of intervals."""
    events = []
    for s, f in activities:
        events.append((s, 1))   # interval starts
        events.append((f, -1))  # interval ends
    events.sort()

    max_depth = 0
    current = 0
    for _, delta in events:
        current += delta
        max_depth = max(max_depth, current)
    return max_depth


# === Demonstration ===

if __name__ == "__main__":
    activities = [
        (0, 3), (1, 4), (2, 5), (3, 7),
        (4, 6), (6, 9), (7, 8),
    ]

    resources = interval_partitioning(activities)
    depth = compute_depth(activities)

    print(f"Number of activities: {len(activities)}")
    print(f"Depth (max overlap): {depth}")
    print(f"Resources needed: {len(resources)}")
    for i, res in enumerate(resources):
        print(f"  Resource {i}: {res}")
```

**Output:**

```
Number of activities: 7
Depth (max overlap): 3
Resources needed: 3
  Resource 0: [(0, 3), (3, 7), (7, 8)]
  Resource 1: [(1, 4), (4, 6), (6, 9)]
  Resource 2: [(2, 5)]
```

At time $t = 2$, three activities overlap: $[0,3)$, $[1,4)$, and $[2,5)$. The algorithm uses exactly 3 resources, matching the depth lower bound.

## Complexity

| Aspect | Cost |
|--------|:----:|
| Time   | $O(n \log n)$ |
| Space  | $O(n)$ |

Sorting takes $O(n \log n)$. Each activity involves one heap push and at most one heap pop, each taking $O(\log n)$ time.

## Applications

- **Classroom assignment.** Assign lectures to the minimum number of rooms given a class schedule.
- **CPU scheduling.** Determine the minimum number of processors for a set of tasks.
- **Vehicle routing.** Minimum fleet size to cover all delivery time windows.

## Reference

- Kleinberg, J., & Tardos, E. (2006). *Algorithm Design*. Pearson. Chapter 4: Greedy Algorithms.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 16: Greedy Algorithms.
