# Divide, Conquer, Combine

Every divide-and-conquer algorithm follows a three-step pattern: split the input, solve the pieces, and merge the results. While the [Strategy](strategy.md) page introduced this pattern at a high level, this page examines each step in detail, showing how the design choices made at each step determine correctness and efficiency.

Understanding the mechanics of each step is essential because the same problem can often be attacked with different divide, conquer, and combine strategies, each leading to a different recurrence and a different running time.

## The Divide Step

The **divide** step partitions the input into smaller subproblems of the same type. The goal is to reduce the problem size by a constant factor at each level of recursion.

### Splitting Strategies

The most common approach splits the input in half, producing two subproblems of size $\lfloor n/2 \rfloor$ and $\lceil n/2 \rceil$. This balanced split is the default for array-based problems such as merge sort and binary search.

More generally, a divide step may produce $a$ subproblems of size $n/b$:

- **Binary split** ($a = 2$, $b = 2$): merge sort, closest pair of points.
- **Unary elimination** ($a = 1$, $b = 2$): binary search discards half the input.
- **Multi-way split** ($a > 2$): Karatsuba uses $a = 3$ subproblems from a 2-way split ($b = 2$).

!!! warning "Unbalanced Splits Hurt Performance"
    If the divide step produces subproblems of sizes $n - 1$ and $1$ (as in naive quicksort on a sorted array), the recursion depth becomes $O(n)$ and the total work is often $O(n^2)$. Balanced splits keep the recursion depth at $O(\log n)$.

### Cost of Dividing

The divide step itself takes time $D(n)$. For many algorithms, dividing is trivial:

- **Merge sort**: compute $\text{mid} = \lfloor (l + r) / 2 \rfloor$ in $O(1)$.
- **Binary search**: compute the midpoint in $O(1)$.
- **Closest pair**: sort points by coordinate or split at a median, costing $O(n)$ or $O(1)$ if presorted.

The divide cost contributes to the $f(n)$ term in the recurrence $T(n) = aT(n/b) + f(n)$.

## The Conquer Step

The **conquer** step solves each subproblem recursively by applying the same algorithm to the smaller input. Recursion continues until the input reaches a **base case** small enough to solve directly.

### Base Cases

A well-chosen base case is critical for both correctness and efficiency.

| Algorithm | Base Case | Direct Solution |
|---|---|---|
| Merge sort | $n \le 1$ | A single element is already sorted |
| Binary search | $l > r$ | The target is not in the array |
| Karatsuba | $n = 1$ | Single-digit multiplication |
| Strassen | $n = 1$ | Scalar multiplication |

### Hybrid Base Cases

In practice, switching to a simpler algorithm below a threshold $n_0$ reduces constant-factor overhead. For example, merge sort implementations typically switch to insertion sort when $n \le 16$, because insertion sort's lower overhead makes it faster on small arrays despite its $O(n^2)$ worst case.

The threshold $n_0$ does not change the asymptotic complexity but can improve real-world performance by a constant factor.

### Independence of Subproblems

A defining property of divide and conquer is that subproblems are **independent**: solving one does not require the result of another. This independence is what distinguishes divide and conquer from dynamic programming, where subproblems overlap and share solutions.

Independence also makes divide-and-conquer algorithms natural candidates for **parallelism**: all $a$ subproblems at any level can be solved concurrently.

## The Combine Step

The **combine** step merges the solutions of the subproblems into a solution for the original problem. This is often the most algorithmically interesting step and the one that determines the overall complexity.

### Examples of Combine Steps

**Merge sort.** The combine step merges two sorted halves into a single sorted array. The merge operation scans both halves simultaneously and produces the output in $O(n)$ time:

```python
def merge(left, right):
    """Merge two sorted lists into one sorted list."""
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

**Binary search.** The combine step is trivial: the answer from the single recursive call is the answer to the original problem. The combine cost is $O(1)$.

**Maximum subarray (divide and conquer).** The combine step finds the maximum crossing subarray that spans the midpoint. This requires a linear scan of the left and right halves, costing $O(n)$.

### Combine Cost and Its Impact

The combine cost $C(n)$ is part of the overhead function $f(n) = D(n) + C(n)$ in the recurrence. The relationship between $f(n)$ and the subproblem work determines the overall complexity through the Master Theorem:

$$
T(n) = a \, T\!\left(\frac{n}{b}\right) + f(n)
$$

- If $f(n) = O(n^{\log_b a - \epsilon})$ for some $\epsilon > 0$, the subproblem work dominates: $T(n) = \Theta(n^{\log_b a})$.
- If $f(n) = \Theta(n^{\log_b a})$, the work is evenly distributed: $T(n) = \Theta(n^{\log_b a} \log n)$.
- If $f(n) = \Omega(n^{\log_b a + \epsilon})$ and the regularity condition holds, the combine work dominates: $T(n) = \Theta(f(n))$.

For a detailed treatment of solving these recurrences, see the [Recurrence Analysis](recurrence.md) page.

## Putting It All Together: Merge Sort

Merge sort illustrates all three steps cleanly.

**Divide.** Split the array at the midpoint: $\text{mid} = \lfloor (l + r) / 2 \rfloor$. Cost: $O(1)$.

**Conquer.** Recursively sort the left half $A[l \,..\, \text{mid}]$ and the right half $A[\text{mid}+1 \,..\, r]$.

**Combine.** Merge the two sorted halves. Cost: $O(n)$.

The recurrence is

$$
T(n) = 2T\!\left(\frac{n}{2}\right) + \Theta(n)
$$

By the Master Theorem (case 2, with $a = 2$, $b = 2$, $f(n) = \Theta(n)$, and $\log_b a = 1$), the solution is

$$
T(n) = \Theta(n \log n)
$$

```python
def merge_sort(arr):
    """Sort an array using the merge sort algorithm."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])      # Conquer left
    right = merge_sort(arr[mid:])     # Conquer right
    return merge(left, right)          # Combine
```

## Common Pitfalls

!!! danger "Forgetting the Combine Step"
    A correct divide-and-conquer algorithm must combine subproblem solutions. Simply dividing and conquering without combining produces only solutions to subproblems, not to the original problem.

!!! warning "Overlapping Subproblems"
    If the divide step produces subproblems that share substructure (e.g., computing Fibonacci numbers recursively), the same work is repeated exponentially many times. In such cases, **dynamic programming** with memoization is the appropriate paradigm.

## Summary

The three steps of divide and conquer -- divide, conquer, combine -- form a complete recipe for algorithm design. The divide step reduces the problem size; the conquer step handles the recursion and base cases; the combine step assembles the final answer. The running time is captured by the recurrence $T(n) = aT(n/b) + f(n)$, where $f(n)$ includes both the divide and combine costs.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 4. MIT Press.
