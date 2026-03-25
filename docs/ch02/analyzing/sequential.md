# Sequential Composition

Most algorithms consist of several steps executed one after another: initialize data
structures, process the input, then produce output. To analyze the total running time,
we need a rule for combining the costs of consecutive blocks of code. Sequential
composition provides that rule and, through its interaction with asymptotic notation,
reveals which step dominates the overall cost.

## The Sum Rule

If an algorithm consists of two consecutive steps with running times $T_1(n)$ and
$T_2(n)$, the total running time is their sum:

$$
T(n) = T_1(n) + T_2(n)
$$

This extends naturally to $k$ sequential steps:

$$
T(n) = T_1(n) + T_2(n) + \cdots + T_k(n) = \sum_{i=1}^{k} T_i(n)
$$

The sum rule follows directly from the definition of running time: the total number of
operations is the sum of operations in each step, since each step executes exactly
once.

??? example "Two Consecutive Loops"

    ```
    Process(A, n):
        // Step 1: Find the maximum
        max_val = A[0]
        for i = 1 to n - 1:
            if A[i] > max_val:
                max_val = A[i]

        // Step 2: Count elements equal to maximum
        count = 0
        for i = 0 to n - 1:
            if A[i] == max_val:
                count = count + 1

        return count
    ```

    Step 1 runs in $T_1(n) = \Theta(n)$. Step 2 runs in $T_2(n) = \Theta(n)$.
    By the sum rule: $T(n) = \Theta(n) + \Theta(n) = \Theta(n)$.

## The Maximum Rule

When combining asymptotic expressions, the sum is dominated by the largest term.
For non-negative functions $f$ and $g$:

$$
O(f(n)) + O(g(n)) = O(\max(f(n), g(n)))
$$

More precisely, if $T_1(n) = O(f(n))$ and $T_2(n) = O(g(n))$, and $f(n) \leq g(n)$
for all sufficiently large $n$, then:

$$
T(n) = T_1(n) + T_2(n) = O(g(n))
$$

This follows because $f(n) + g(n) \leq 2g(n) = O(g(n))$.

!!! tip "The Dominant Term Wins"

    When sequential steps have different asymptotic costs, only the most expensive
    step matters. A $\Theta(n^2)$ step followed by a $\Theta(n)$ step costs
    $\Theta(n^2)$ overall. The linear step is "free" in the asymptotic sense.

## Applying the Maximum Rule

Consider an algorithm with three sequential phases:

```
Algorithm(A, n):
    Phase1(A, n)        # O(n)
    Phase2(A, n)        # O(n^2)
    Phase3(A, n)        # O(n log n)
```

By the sum and maximum rules:

$$
T(n) = O(n) + O(n^2) + O(n \log n) = O(n^2)
$$

The quadratic phase dominates.

??? example "Sort Then Search"

    A common algorithmic pattern is to sort the data and then perform searches:

    ```
    SortAndSearch(A, n, targets, m):
        Sort(A, n)                          # O(n log n)
        for i = 0 to m - 1:                 # m iterations
            BinarySearch(A, n, targets[i])  # O(log n) each
    ```

    Phase 1 costs $O(n \log n)$. Phase 2 costs $O(m \log n)$. Total:

    $$
    T(n, m) = O(n \log n) + O(m \log n) = O((n + m) \log n)
    $$

    If $m = O(n)$, this simplifies to $O(n \log n)$. The sort dominates because
    $m \leq n$.

## When Constants Matter

The maximum rule applies to asymptotic analysis, where constant factors are
suppressed. In practice, two sequential $O(n)$ steps with large constant factors can
be slower than a single $O(n \log n)$ step with a small constant — for realistic
input sizes. Sequential composition tells us the growth rate, not the exact time.

??? example "Constants in Practice"

    ```
    SlowLinear(A, n):       # 100n operations
    FastNlogN(A, n):        # 2n log n operations
    ```

    Asymptotically, `SlowLinear` is faster ($O(n)$ vs $O(n \log n)$). But for
    $n < 2^{50}$, the fast $n \log n$ algorithm with its small constant actually
    finishes sooner: $2n \log_2 n < 100n$ when $\log_2 n < 50$.

## Composing with Other Rules

Sequential composition combines with the loop and conditional rules:

| Construct | Rule |
|---|---|
| `S1; S2` (sequence) | $T_1(n) + T_2(n)$ |
| `for` loop | iterations $\times$ body cost |
| `if-else` | condition $+$ max of branches |
| Nested construct | Apply rules inside-out |

An algorithm's total cost is built by composing these rules. Start from the innermost
construct and work outward, applying sequential composition whenever consecutive
blocks appear at the same nesting level.

!!! warning "Order of Execution Matters for Correctness, Not Cost"

    The sum rule $T_1(n) + T_2(n)$ gives the same total regardless of which step runs
    first. However, swapping the order may change the algorithm's correctness. The sum
    rule is a statement about *cost*, not about *semantics*.

## Reference

- [Introduction to Algorithms (CLRS), Chapters 2-4](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
