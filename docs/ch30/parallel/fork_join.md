# Fork-Join Framework

Many parallel algorithms share a common structure: split a problem into independent subproblems, solve them concurrently, and combine the results. The **fork-join** model formalizes this pattern, providing both a programming abstraction and an analytical framework. It underpins parallel divide-and-conquer algorithms and is the execution model behind frameworks like Cilk, Java ForkJoinPool, and Intel TBB.

## The Fork-Join Model

A fork-join computation consists of three phases:

1. **Fork**: The current task spawns one or more child tasks that can execute in parallel.
2. **Compute**: Each child task executes independently (and may recursively fork further subtasks).
3. **Join**: The parent task waits until all child tasks complete, then combines their results.

This creates a tree-structured computation DAG. Each internal node represents a fork point, and the join corresponds to a synchronization barrier.

## Work-Span Analysis

The fork-join structure naturally leads to divide-and-conquer recurrences. For a problem of size $n$ that forks into $a$ subproblems of size $n/b$ with $O(f(n))$ overhead for forking and joining:

$$
T_1(n) = a \cdot T_1(n/b) + f(n)
$$

$$
T_\infty(n) = T_\infty(n/b) + f(n)
$$

The work recurrence counts all operations (each subproblem contributes). The span recurrence follows only the critical path, so the $a$ parallel subproblems contribute just one term (the slowest, which by symmetry has the same span).

!!! tip "Applying the Master theorem"
    The work recurrence $T_1(n) = a \cdot T_1(n/b) + f(n)$ follows the standard Master theorem form. The span recurrence $T_\infty(n) = T_\infty(n/b) + f(n)$ is the special case $a = 1$.

## Example: Parallel Sum

A parallel sum of an array $A[0 \ldots n-1]$ forks the array into two halves, recursively sums each half, and joins by adding the two partial sums.

**Work recurrence**:

$$
T_1(n) = 2 \cdot T_1(n/2) + O(1) = O(n)
$$

**Span recurrence**:

$$
T_\infty(n) = T_\infty(n/2) + O(1) = O(\log n)
$$

**Parallelism**: $P = T_1 / T_\infty = O(n / \log n)$.

```python
"""
Fork-join parallel sum simulation.

Demonstrates the fork-join pattern with recursive array summation.
Tracks work (total operations) and span (critical path depth).
"""

# ===================================================================
# Fork-Join Parallel Sum
# ===================================================================

class ForkJoinStats:
    """Track work and span of a fork-join computation."""

    def __init__(self):
        self.work = 0

    def parallel_sum(self, arr, lo, hi):
        """Compute sum of arr[lo:hi] using fork-join pattern.

        Args:
            arr: input array
            lo: start index (inclusive)
            hi: end index (exclusive)

        Returns:
            Tuple of (sum, span)
        """
        if hi - lo <= 1:
            self.work += 1
            return (arr[lo] if lo < hi else 0), 1

        mid = (lo + hi) // 2

        # Fork: two subproblems (would run in parallel)
        left_sum, left_span = self.parallel_sum(arr, lo, mid)
        right_sum, right_span = self.parallel_sum(arr, mid, hi)

        # Join: combine results
        self.work += 1  # addition at join
        total = left_sum + right_sum
        # Span: max of parallel branches + 1 for the join
        span = max(left_span, right_span) + 1

        return total, span

# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    arr = list(range(1, 17))  # [1, 2, ..., 16]
    n = len(arr)

    stats = ForkJoinStats()
    total, span = stats.parallel_sum(arr, 0, n)

    print(f"Array: {arr}")
    print(f"Sum:   {total}")
    print(f"Work (T_1):   {stats.work}")
    print(f"Span (T_inf): {span}")
    print(f"Parallelism:  {stats.work / span:.1f}")
    print()

    # Brent's bound for various processor counts
    print("Brent's bound T_p <= T_1/p + T_inf:")
    for p in [1, 2, 4, 8]:
        tp = stats.work / p + span
        print(f"  p={p}: T_p <= {tp:.1f}")
```

**Output:**
```
Array: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
Sum:   136
Work (T_1):   31
Span (T_inf): 5
Parallelism:  6.2

Brent's bound T_p <= T_1/p + T_inf:
  p=1: T_p <= 36.0
  p=2: T_p <= 20.5
  p=4: T_p <= 12.8
  p=8: T_p <= 8.9
```

## Nested Fork-Join

Fork-join computations can nest arbitrarily. A common pattern in practice:

```
fork_join_outer:
    fork:
        fork_join_inner_A:
            fork: subproblem A1
            fork: subproblem A2
            join
    fork:
        fork_join_inner_B:
            fork: subproblem B1
            fork: subproblem B2
            join
    join
```

The span of the outer computation is the span of the sequential composition of fork and join overhead plus the maximum span among the parallel branches.

## Greedy Scheduler

A **greedy scheduler** assigns ready tasks to idle processors without unnecessary delays. Brent's theorem guarantees that any greedy scheduler achieves:

$$
T_p \le \frac{T_1}{p} + T_\infty
$$

This bound is within a factor of 2 of optimal, since no schedule can beat $\max(T_1/p,\, T_\infty)$.

!!! note "Work stealing implements greedy scheduling"
    The work-stealing scheduler (see [Work Stealing](work_stealing.md)) is a practical realization of a greedy scheduler. Each processor maintains a local deque of tasks, and idle processors steal from busy ones.

## Reference

- Cormen, T. H. et al. *Introduction to Algorithms*, Chapter 27 (Multithreaded Algorithms).
- Blumofe, R. D. and Leiserson, C. E. (1999). "Scheduling multithreaded computations by work stealing." *JACM*, 46(5), 720--748.
