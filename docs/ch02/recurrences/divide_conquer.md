# Recurrence from Divide and Conquer

When analyzing a recursive algorithm, we rarely know its running time directly. Instead, we express the time to solve a problem of size $n$ in terms of the time to solve smaller subproblems. The resulting equation is called a **recurrence relation**, and deriving it correctly is the first step toward understanding an algorithm's efficiency. This page shows how to translate a divide-and-conquer algorithm into a recurrence and identifies the key components that determine the recurrence's structure.

## The Divide-and-Conquer Paradigm

A divide-and-conquer algorithm follows three steps:

1. **Divide**: Split the problem of size $n$ into $a$ subproblems, each of size roughly $n/b$
2. **Conquer**: Solve each subproblem recursively (or directly if the subproblem is small enough)
3. **Combine**: Merge the subproblem solutions into a solution for the original problem

Each of these steps contributes to the total running time, and the recurrence captures that contribution precisely.

## Deriving the General Recurrence

Let $T(n)$ denote the running time of a divide-and-conquer algorithm on an input of size $n$. The three steps contribute:

- **Divide cost** $D(n)$: the time to split the input
- **Conquer cost**: $a$ recursive calls, each on a problem of size $n/b$, costing $a \cdot T(n/b)$
- **Combine cost** $C(n)$: the time to merge solutions

The total running time satisfies:

$$
T(n) = \begin{cases} \Theta(1) & \text{if } n \leq n_0 \\[4pt] a \, T(n/b) + D(n) + C(n) & \text{if } n > n_0 \end{cases}
$$

where $n_0$ is the base-case threshold. The sum $f(n) = D(n) + C(n)$ is often called the **toll function** or **driving function**, representing the non-recursive work at each level.

## Identifying the Parameters

To write down a recurrence for a specific algorithm, answer these four questions:

| Parameter | Question | Typical values |
|-----------|----------|----------------|
| $a$ | How many recursive calls does each invocation make? | 1, 2, 4, 7, ... |
| $b$ | By what factor does the problem size shrink? | 2, 3, 4, ... |
| $D(n)$ | How much work does the divide step take? | $\Theta(1)$, $\Theta(n)$ |
| $C(n)$ | How much work does the combine step take? | $\Theta(1)$, $\Theta(n)$, $\Theta(n^2)$ |

The parameters $a$ and $b$ determine the shape of the recursion tree, while $D(n) + C(n)$ determines the work done at each node.

## Classic Examples

### Merge Sort

Merge sort divides an array of $n$ elements into two halves, recursively sorts each half, and merges the sorted halves.

- **Divide**: Split the array at the midpoint. Cost: $D(n) = \Theta(1)$
- **Conquer**: Two recursive calls on arrays of size $n/2$
- **Combine**: Merge two sorted halves by scanning both. Cost: $C(n) = \Theta(n)$

Recurrence:

$$
T(n) = 2T(n/2) + \Theta(n)
$$

Solution: $T(n) = \Theta(n \log n)$, derived via the Master theorem or the recursion tree method.

### Binary Search

Binary search compares the target with the middle element and recurses on one half.

- **Divide**: Compute the midpoint. Cost: $D(n) = \Theta(1)$
- **Conquer**: One recursive call on an array of size $n/2$
- **Combine**: No merging needed. Cost: $C(n) = \Theta(1)$

Recurrence:

$$
T(n) = T(n/2) + \Theta(1)
$$

Solution: $T(n) = \Theta(\log n)$.

### Strassen's Matrix Multiplication

Strassen's algorithm multiplies two $n \times n$ matrices by reducing the problem from eight recursive multiplications to seven, at the cost of additional additions.

- **Divide**: Partition each matrix into four $n/2 \times n/2$ submatrices. Cost: $D(n) = \Theta(1)$ (index arithmetic only)
- **Conquer**: Seven recursive multiplications on $n/2 \times n/2$ matrices
- **Combine**: Compute the result submatrices from the seven products using $\Theta(n^2)$ additions

Recurrence:

$$
T(n) = 7T(n/2) + \Theta(n^2)
$$

Solution: $T(n) = \Theta(n^{\log_2 7}) \approx \Theta(n^{2.807})$, which improves on the naive $\Theta(n^3)$.

### Maximum Subarray (Divide and Conquer)

Find the contiguous subarray with the largest sum by splitting the array in half and considering three cases: the maximum subarray lies entirely in the left half, entirely in the right half, or crosses the midpoint.

- **Divide**: Split at midpoint. Cost: $D(n) = \Theta(1)$
- **Conquer**: Two recursive calls on arrays of size $n/2$
- **Combine**: Find the best crossing subarray by scanning left and right from the midpoint. Cost: $C(n) = \Theta(n)$

Recurrence:

$$
T(n) = 2T(n/2) + \Theta(n)
$$

Solution: $T(n) = \Theta(n \log n)$, the same structure as merge sort.

## Handling Floors and Ceilings

In practice, $n/b$ is not always an integer. For merge sort on an odd-length array, one half has $\lfloor n/2 \rfloor$ elements and the other has $\lceil n/2 \rceil$. The exact recurrence is:

$$
T(n) = T(\lfloor n/2 \rfloor) + T(\lceil n/2 \rceil) + \Theta(n)
$$

For asymptotic analysis, floors and ceilings do not affect the solution. The standard practice is to write $T(n) = 2T(n/2) + \Theta(n)$ with the understanding that this represents the asymptotic behavior. The [Akra-Bazzi method](akra_bazzi.md) provides a rigorous justification for ignoring floors and ceilings.

## Unequal Subproblem Sizes

Some algorithms split the input into subproblems of different sizes. The select algorithm (median of medians) produces the recurrence:

$$
T(n) = T(n/5) + T(7n/10) + \Theta(n)
$$

This does not fit the standard $T(n) = aT(n/b) + f(n)$ form because the two subproblems have different size ratios. The [Akra-Bazzi method](akra_bazzi.md) handles such recurrences directly.

## From Recurrence to Solution

Once a recurrence has been derived, several methods can solve it:

| Method | Best suited for | Page |
|--------|----------------|------|
| Substitution | Verifying a guessed solution | [Substitution Method](substitution.md) |
| Recursion tree | Building intuition, guessing the answer | [Recursion Tree Method](recursion_tree.md) |
| Master theorem | Standard $T(n) = aT(n/b) + f(n)$ form | [Master Theorem](master.md) |
| Extended Master | Logarithmic factors in $f(n)$ | [Extended Master Theorem](extended_master.md) |
| Akra-Bazzi | Unequal subproblem sizes | [Akra-Bazzi Method](akra_bazzi.md) |
| Generating functions | Non-standard or full exact solutions | [Generating Functions](generating.md) |

The choice depends on the recurrence's form and the level of detail needed.

## Common Pitfalls

!!! warning "Pitfalls When Deriving Recurrences"
    - **Forgetting the base case**: Every recurrence needs $T(n) = \Theta(1)$ for $n \leq n_0$. Without it, the recurrence is not well-defined.
    - **Miscounting recursive calls**: Count the number of recursive invocations, not the number of subproblems created by the divide step. Strassen creates many submatrices but makes exactly seven recursive calls.
    - **Ignoring the combine cost**: The combine step often dominates. Merge sort's $\Theta(n \log n)$ comes from the $\Theta(n)$ merge at each level, not from the divide step.
    - **Confusing $a$ and $b$**: The parameter $a$ is the number of subproblems; $b$ is the factor by which the problem size shrinks. For merge sort, $a = 2$ and $b = 2$.

## References

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapters 2 and 4. MIT Press.
- Kleinberg, J., & Tardos, E. (2005). *Algorithm Design*, Chapter 5. Pearson.
