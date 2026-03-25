# Activity Selection

The activity selection problem is the canonical example of greedy algorithm design. Given a set of activities that each require exclusive use of a shared resource (a lecture hall, a conference room, a CPU), the goal is to select the largest possible subset of non-overlapping activities. This problem appears in scheduling, resource allocation, and compiler optimization, and its greedy solution --- selecting by earliest finish time --- is both simple and provably optimal.

## Problem Statement

**Input.** A set $S = \{a_1, a_2, \ldots, a_n\}$ of $n$ activities. Each activity $a_i$ has a start time $s_i$ and a finish time $f_i$, with $s_i < f_i$.

**Compatibility.** Two activities $a_i$ and $a_j$ are **compatible** if their intervals do not overlap: $f_i \leq s_j$ or $f_j \leq s_i$.

**Goal.** Find a maximum-size subset $A \subseteq S$ of mutually compatible activities.

## Why Earliest Finish Time?

There are several natural greedy strategies:

| Strategy | Greedy rule | Optimal? |
|----------|-------------|----------|
| Earliest start time | Pick $\min(s_i)$ | No |
| Shortest duration | Pick $\min(f_i - s_i)$ | No |
| Fewest conflicts | Pick the activity overlapping the fewest others | No |
| **Earliest finish time** | **Pick $\min(f_i)$** | **Yes** |

The earliest finish time strategy works because it leaves the maximum amount of remaining time for future activities. By finishing as early as possible, the greedy choice maximizes the number of compatible activities that can still be selected.

## Algorithm

!!! note "Greedy Activity Selection"
    1. Sort activities by finish time: $f_1 \leq f_2 \leq \cdots \leq f_n$.
    2. Select $a_1$ (the activity with the earliest finish time).
    3. For $i = 2, \ldots, n$: if $s_i \geq f_{\text{last}}$ (where $f_{\text{last}}$ is the finish time of the most recently selected activity), select $a_i$.

## Worked Example

Consider six activities:

| Activity | $s_i$ | $f_i$ |
|----------|-------|-------|
| $a_1$    | 1     | 4     |
| $a_2$    | 3     | 5     |
| $a_3$    | 0     | 6     |
| $a_4$    | 5     | 7     |
| $a_5$    | 3     | 9     |
| $a_6$    | 5     | 9     |

**Sorted by finish time:** $a_1, a_2, a_3, a_4, a_5, a_6$.

**Greedy execution:**

1. Select $a_1$ (finishes at 4). Set $f_{\text{last}} = 4$.
2. $a_2$: $s_2 = 3 < 4$. Skip (overlaps).
3. $a_3$: $s_3 = 0 < 4$. Skip (overlaps).
4. $a_4$: $s_4 = 5 \geq 4$. Select. Set $f_{\text{last}} = 7$.
5. $a_5$: $s_5 = 3 < 7$. Skip.
6. $a_6$: $s_6 = 5 < 7$. Skip.

**Result:** $\{a_1, a_4\}$ with 2 activities. This is maximum --- no set of 3 mutually compatible activities exists.

## Correctness Proof

**Theorem.** The greedy algorithm selects a maximum-size set of mutually compatible activities.

The proof combines the greedy choice property with optimal substructure.

**Greedy choice property.** There exists an optimal solution containing $a_1$ (the activity with the earliest finish time).

*Proof.* Let $S^* = \{a_{j_1}, a_{j_2}, \ldots, a_{j_k}\}$ be an optimal solution sorted by finish time. If $a_{j_1} = a_1$, done. Otherwise, $f_1 \leq f_{j_1}$, so replacing $a_{j_1}$ with $a_1$ preserves compatibility with all subsequent activities. The resulting set has the same cardinality $k$. $\square$

**Optimal substructure.** If an optimal solution contains $a_1$, then the remaining activities $\{a_1\} \cup R$ have $R$ optimal for $S' = \{a_i \in S : s_i \geq f_1\}$.

*Proof.* Cut-and-paste: if $R$ is not optimal for $S'$, a better $R'$ would yield $\{a_1\} \cup R'$ with $|R'| > |R|$, contradicting $|S^*| = k$. $\square$

**Conclusion.** By induction on the number of activities, the greedy algorithm selects a maximum-size set.

## Python Implementation

```python
"""
Activity selection using the earliest-finish-time greedy strategy.

Given activities with start and finish times, selects the maximum number
of mutually compatible (non-overlapping) activities.
"""


# === Greedy Activity Selection ===

def activity_selection(activities):
    """Select maximum number of non-overlapping activities.

    Args:
        activities: list of (start, finish) tuples

    Returns:
        List of selected (start, finish) tuples
    """
    # Sort by finish time
    sorted_acts = sorted(activities, key=lambda x: x[1])

    selected = [sorted_acts[0]]
    last_finish = sorted_acts[0][1]

    for start, finish in sorted_acts[1:]:
        if start >= last_finish:
            selected.append((start, finish))
            last_finish = finish

    return selected


# === Recursive Version ===

def activity_selection_recursive(activities, k=0):
    """Recursive greedy activity selection.

    Args:
        activities: list of (start, finish) tuples, sorted by finish time
        k: finish time of the last selected activity

    Returns:
        List of selected (start, finish) tuples
    """
    # Find the first compatible activity
    for i, (start, finish) in enumerate(activities):
        if start >= k:
            return [(start, finish)] + activity_selection_recursive(
                activities[i + 1:], finish
            )
    return []


if __name__ == "__main__":
    # Example from the worked example above
    activities = [(1, 4), (3, 5), (0, 6), (5, 7), (3, 9), (5, 9)]

    result = activity_selection(activities)
    print(f"Activities: {activities}")
    print(f"Selected:   {result}")
    print(f"Count:      {len(result)}")

    # Recursive version
    sorted_acts = sorted(activities, key=lambda x: x[1])
    result_rec = activity_selection_recursive(sorted_acts)
    print(f"Recursive:  {result_rec}")
```

**Output:**
```
Activities: [(1, 4), (3, 5), (0, 6), (5, 7), (3, 9), (5, 9)]
Selected:   [(1, 4), (5, 7)]
Count:      2
Recursive:  [(1, 4), (5, 7)]
```

## Complexity Analysis

**Time complexity.**

- Sorting: $O(n \log n)$.
- Single scan: $O(n)$.
- **Total:** $O(n \log n)$.

If the activities are pre-sorted by finish time, the algorithm runs in $O(n)$.

**Space complexity.** $O(n)$ for the output (or $O(1)$ auxiliary if we only count selected activities).

## Weighted Activity Selection

When each activity $a_i$ has an associated weight (value) $v_i$, the goal changes to maximizing total weight rather than count. The greedy approach no longer works --- this variant requires dynamic programming:

$$
\text{OPT}(j) = \max\bigl(\text{OPT}(j-1),\; v_j + \text{OPT}(p(j))\bigr)
$$

where $p(j)$ is the largest index $i < j$ such that $a_i$ is compatible with $a_j$.

This illustrates an important lesson: a small change in the objective (from count to weighted sum) can invalidate the greedy choice property.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 16.1. MIT Press.
- Kleinberg, J. & Tardos, E. (2006). *Algorithm Design*, Chapter 4.1. Pearson.
