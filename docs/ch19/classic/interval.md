# Interval Scheduling

Given a set of activities (or jobs), each with a start and finish time, the **interval scheduling** problem asks for the maximum number of non-overlapping activities. The greedy strategy of always selecting the activity that finishes earliest is optimal, and its correctness can be proved by an "exchange argument" showing that no other strategy can do better.

## Problem Statement

Given $n$ activities $\{a_1, a_2, \dots, a_n\}$ where activity $a_i$ has start time $s_i$ and finish time $f_i$, find the largest subset $S$ of mutually compatible activities. Two activities $a_i$ and $a_j$ are **compatible** if their intervals do not overlap: $f_i \le s_j$ or $f_j \le s_i$.

## Greedy Algorithm

**Strategy.** Sort activities by finish time. Greedily select each activity whose start time is at or after the finish time of the last selected activity.

**Why earliest finish time?** By finishing as early as possible, we leave the most room for subsequent activities. Alternative strategies (earliest start time, shortest duration) can be shown to fail with simple counterexamples.

## Correctness Proof

!!! note "Theorem"
    The earliest-finish-time greedy algorithm produces a maximum-size set of mutually compatible activities.

??? example "Proof (Exchange Argument)"
    Let $G = \{g_1, g_2, \dots, g_k\}$ be the greedy solution (sorted by finish time) and $O = \{o_1, o_2, \dots, o_m\}$ be an optimal solution with $m \ge k$. We show $k = m$.

    **Claim.** For each $i \le k$, $f(g_i) \le f(o_i)$ (greedy finishes at least as early at every step).

    *Base case:* $f(g_1) \le f(o_1)$ because greedy picks the earliest finish time.

    *Inductive step:* Assume $f(g_i) \le f(o_i)$. Then $s(o_{i+1}) \ge f(o_i) \ge f(g_i)$, so $o_{i+1}$ is available to greedy at step $i+1$. Greedy picks $g_{i+1}$ with the earliest finish time among all available activities, so $f(g_{i+1}) \le f(o_{i+1})$.

    Since the greedy solution stays ahead at every step, if $m > k$, then $o_{k+1}$ would be compatible with $g_k$ (because $s(o_{k+1}) \ge f(o_k) \ge f(g_k)$), contradicting the fact that greedy stopped. Therefore $m = k$. $\square$

## Implementation

```python
"""
Interval scheduling via the earliest-finish-time greedy algorithm.

Selects the maximum number of non-overlapping intervals by always
choosing the activity that finishes earliest.
"""

# === Greedy Interval Scheduling ===

def interval_scheduling(
    activities: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Find maximum set of non-overlapping activities.

    Args:
        activities: List of (start, finish) tuples.

    Returns:
        Maximum-size subset of mutually compatible activities.
    """
    # Sort by finish time
    sorted_acts = sorted(activities, key=lambda x: x[1])

    selected = []
    last_finish = -1

    for start, finish in sorted_acts:
        if start >= last_finish:
            selected.append((start, finish))
            last_finish = finish

    return selected


# === Demonstration ===

if __name__ == "__main__":
    activities = [
        (1, 4), (3, 5), (0, 6), (5, 7),
        (3, 9), (5, 9), (6, 10), (8, 11),
        (8, 12), (2, 14), (12, 16),
    ]
    result = interval_scheduling(activities)
    print(f"Total activities: {len(activities)}")
    print(f"Maximum compatible set: {len(result)} activities")
    for s, f in result:
        print(f"  [{s}, {f})")
```

**Output:**

```
Total activities: 11
Maximum compatible set: 4 activities
  [1, 4)
  [5, 7)
  [8, 11)
  [12, 16)
```

Out of 11 activities, the greedy algorithm selects 4 non-overlapping ones. Each selected activity starts at or after the previous one finishes.

## Why Other Strategies Fail

| Strategy | Counterexample |
|----------|---------------|
| Earliest start time | A long activity starting early blocks many short ones |
| Shortest duration | A short activity in the middle overlaps two non-overlapping ones |
| Fewest conflicts | An activity with few conflicts may still block optimal choices |

## Complexity

| Aspect | Cost |
|--------|:----:|
| Time   | $O(n \log n)$ |
| Space  | $O(n)$ |

Sorting dominates at $O(n \log n)$. The selection pass is a single $O(n)$ scan.

## Weighted Variant

When each activity has a weight (profit) $w_i$, the goal becomes maximizing total weight rather than count. The greedy approach no longer works, and dynamic programming is needed:

$$
\text{dp}[j] = \max\bigl(w_j + \text{dp}[p(j)],\; \text{dp}[j-1]\bigr)
$$

where $p(j)$ is the latest activity compatible with $j$ (found via binary search after sorting by finish time).

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 16: Greedy Algorithms.
- Kleinberg, J., & Tardos, E. (2006). *Algorithm Design*. Pearson. Chapter 4: Greedy Algorithms.
