# Binary Search Template

Standard binary search finds a specific target in a sorted array. Many problems, however, require finding the **boundary** where a condition changes from false to true (or vice versa). A generalized binary search template handles all such problems uniformly: given a monotone predicate, find the smallest (or largest) index satisfying the predicate.

This template eliminates the error-prone details of boundary manipulation -- off-by-one errors, infinite loops, and incorrect midpoint rounding -- that make binary search notoriously tricky to implement correctly.

## The Generalized Problem

Suppose we have a search space $\{0, 1, \ldots, n-1\}$ and a boolean predicate $\text{condition}(m)$ that is **monotone**: there exists a threshold $k$ such that

$$
\text{condition}(m) = \begin{cases} \text{false} & \text{if } m < k \\ \text{true} & \text{if } m \ge k \end{cases}
$$

The goal is to find the smallest $m$ for which $\text{condition}(m)$ is true. This is the **leftmost true** problem.

## The Template

```python
def binary_search_template(lo, hi, condition):
    """
    Find the smallest value in [lo, hi] satisfying condition.

    Parameters
    ----------
    lo : int
        Lower bound of the search space (inclusive).
    hi : int
        Upper bound of the search space (inclusive).
    condition : callable
        A monotone predicate: False for values below the
        threshold, True at and above it.

    Returns
    -------
    int
        The smallest value m in [lo, hi] such that
        condition(m) is True, or hi + 1 if no such value exists.
    """
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if condition(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
```

### Key Design Choices

1. **Loop condition `lo < hi`** (not `lo <= hi`): the loop terminates when `lo == hi`, at which point the answer is `lo`.
2. **`hi = mid`** (not `hi = mid - 1`): when the condition is true, `mid` might be the answer, so we keep it in the search space.
3. **`lo = mid + 1`**: when the condition is false, `mid` is definitely not the answer, so we exclude it.
4. **`mid = lo + (hi - lo) // 2`**: rounds down, ensuring `mid < hi` when `lo < hi`, which prevents infinite loops.

!!! warning "Infinite Loop Trap"
    Using `hi = mid - 1` with `mid = lo + (hi - lo) // 2` can cause the loop to miss the answer. Using `lo = mid` with the same midpoint formula causes an infinite loop when `hi - lo == 1`. The template above avoids both pitfalls.

## Correctness Proof

**Loop invariant.** At the start of each iteration, the answer (the smallest $m$ with $\text{condition}(m)$ true) lies in $[\text{lo}, \text{hi}]$.

**Initialization.** The invariant holds because the initial range covers the entire search space.

**Maintenance.** Let $\text{mid} = \lfloor (\text{lo} + \text{hi}) / 2 \rfloor$.

- If $\text{condition}(\text{mid})$ is true, then the answer is at most $\text{mid}$, so setting $\text{hi} = \text{mid}$ preserves the invariant.
- If $\text{condition}(\text{mid})$ is false, then the answer is at least $\text{mid} + 1$, so setting $\text{lo} = \text{mid} + 1$ preserves the invariant.

**Termination.** The quantity $\text{hi} - \text{lo}$ is a non-negative integer that strictly decreases at each iteration (because $\text{mid} < \text{hi}$ when $\text{lo} < \text{hi}$). When $\text{lo} = \text{hi}$, the loop terminates, and the invariant guarantees `lo` is the answer. $\square$

**Time complexity.** The search space halves at each iteration, so the template performs $O(\log(\text{hi} - \text{lo}))$ iterations. If each call to `condition` takes $O(C)$ time, the total is $O(C \log(\text{hi} - \text{lo}))$.

## Rightmost False Variant

To find the **largest** $m$ for which $\text{condition}(m)$ is false, use a mirror template:

```python
def binary_search_rightmost_false(lo, hi, condition):
    """
    Find the largest value in [lo, hi] where condition is False.

    Returns lo - 1 if condition is True for all values.
    """
    while lo < hi:
        mid = lo + (hi - lo + 1) // 2  # round up
        if condition(mid):
            hi = mid - 1
        else:
            lo = mid
    return lo
```

Note the midpoint rounds **up** (`(hi - lo + 1) // 2`) to prevent infinite loops when `lo = mid`.

## Applications

### Search Insert Position

Find the index where `target` should be inserted in a sorted array to maintain sorted order.

```python
def search_insert(nums, target):
    """Find the insertion position for target in a sorted array."""
    return binary_search_template(
        0, len(nums),
        lambda mid: mid == len(nums) or nums[mid] >= target
    )
```

### Integer Square Root

Find the largest integer $k$ such that $k^2 \le x$.

```python
def integer_sqrt(x):
    """Compute floor(sqrt(x)) using binary search."""
    if x < 0:
        raise ValueError("Square root of negative number")
    if x == 0:
        return 0
    # Find smallest k where (k+1)^2 > x, then return k
    return binary_search_template(
        1, x,
        lambda mid: mid * mid > x
    ) - 1
```

### First Bad Version

Given $n$ versions numbered $1$ to $n$ and a function `is_bad(v)` that is monotone (all versions after the first bad one are also bad), find the first bad version.

```python
def first_bad_version(n, is_bad):
    """Find the first bad version among versions 1..n."""
    return binary_search_template(1, n, is_bad)
```

### Capacity to Ship Packages

Find the minimum ship capacity to deliver all packages within $d$ days. The predicate "can ship all packages in $d$ days with capacity $c$" is monotone in $c$.

```python
def ship_within_days(weights, days):
    """Find minimum capacity to ship all weights within given days."""
    def can_ship(capacity):
        day_count, current_load = 1, 0
        for w in weights:
            if current_load + w > capacity:
                day_count += 1
                current_load = 0
            current_load += w
        return day_count <= days

    return binary_search_template(
        max(weights), sum(weights), can_ship
    )
```

## When to Use the Template

The template applies whenever a problem has these properties:

1. **Monotone predicate**: the condition transitions from false to true exactly once across the search space.
2. **Bounded search space**: the range $[\text{lo}, \text{hi}]$ is known in advance.
3. **Efficient evaluation**: checking `condition(mid)` takes polynomial time.

!!! tip "Identifying Binary Search Problems"
    If a problem asks for the "minimum value satisfying X" or the "maximum value not exceeding Y," and X or Y is monotone in the answer, binary search on the answer is likely the right approach.

## Summary

The generalized binary search template reduces all binary search variants to a single pattern: define a monotone predicate, set the search bounds, and let the template find the transition point. The correctness proof relies on a loop invariant showing that the answer always lies within the current bounds, and termination follows from the strictly decreasing search space. The template runs in $O(\log n)$ iterations, each calling the predicate once.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 2. MIT Press.
