# Counting Operations

To determine an algorithm's running time, we count the number of **primitive
operations** it executes as a function of the input size $n$. Each primitive
operation — an assignment, a comparison, an arithmetic operation, or an array
access — takes constant time, so the total running time is proportional to the
total count. This line-by-line accounting is the foundation on which all asymptotic
analysis rests.

## Primitive Operations

A **primitive operation** (or elementary operation) is any instruction that the
machine executes in $O(1)$ time, regardless of input size. Common primitive
operations include:

| Operation | Examples |
|---|---|
| Assignment | `x = 5`, `temp = A[i]` |
| Comparison | `x < y`, `A[i] == target` |
| Arithmetic | `x + y`, `i * 2`, `n / 2` |
| Array access | `A[i]`, `A[i] = v` |
| Return | `return x` |

We assign a cost of 1 to each primitive operation. The total cost $T(n)$ of an
algorithm is the sum of costs over all executed operations.

## Line-by-Line Counting

The standard method is to annotate each line with its cost and the number of times it
executes, then sum the products.

??? example "Sum of Array Elements"

    ```
    Sum(A, n):
    1.  total = 0                  cost c1,  runs 1 time
    2.  for i = 0 to n - 1:       cost c2,  runs n + 1 times (including final test)
    3.      total = total + A[i]   cost c3,  runs n times
    4.  return total               cost c4,  runs 1 time
    ```

    The total operation count is:

    $$
    T(n) = c_1 \cdot 1 + c_2 \cdot (n + 1) + c_3 \cdot n + c_4 \cdot 1
    $$

    Expanding:

    $$
    T(n) = (c_2 + c_3) \cdot n + (c_1 + c_2 + c_4)
    $$

    This is a linear function of $n$, so $T(n) = \Theta(n)$.

## Simplifying the Count

In practice, we do not track individual constants $c_1, c_2, \ldots$ because
asymptotic notation absorbs them. The simplified approach:

1. **Identify the dominant term.** Find which operation executes the most times.
2. **Count executions, not cost.** Since each primitive costs $O(1)$, we only need the
   execution count.
3. **Express as a function of $n$.** The dominant term's execution count gives the
   growth rate.

!!! tip "Constant Factors Do Not Matter"

    Whether an assignment takes 1 nanosecond or 5 nanoseconds does not change the
    asymptotic class. We care about *how many times* each line runs, not *how long*
    each execution takes.

## Counting in Detail: Insertion Sort

A more involved example shows how counting handles input-dependent behavior.

```
InsertionSort(A, n):
1.  for j = 1 to n - 1:              runs n times (loop test)
2.      key = A[j]                    runs n - 1 times
3.      i = j - 1                     runs n - 1 times
4.      while i >= 0 and A[i] > key:  runs t_j times for each j
5.          A[i + 1] = A[i]           runs t_j - 1 times for each j
6.          i = i - 1                 runs t_j - 1 times for each j
7.      A[i + 1] = key                runs n - 1 times
```

Here $t_j$ denotes the number of times the `while` loop test on line 4 executes for
a given value of $j$. The total cost is:

$$
T(n) = c_1 n + c_2(n-1) + c_3(n-1) + c_4 \sum_{j=1}^{n-1} t_j + c_5 \sum_{j=1}^{n-1}(t_j - 1) + c_6 \sum_{j=1}^{n-1}(t_j - 1) + c_7(n-1)
$$

The values of $t_j$ depend on the input:

- **Best case** (sorted array): $t_j = 1$ for all $j$, so $\sum t_j = n - 1$ and
  $T(n) = \Theta(n)$.
- **Worst case** (reverse-sorted): $t_j = j$ for all $j$, so
  $\sum t_j = \frac{n(n-1)}{2}$ and $T(n) = \Theta(n^2)$.

## Counting Rules Summary

| Construct | Operation Count |
|---|---|
| Single statement | $O(1)$ |
| Sequence of statements | Sum of individual costs |
| `if-else` | Cost of condition $+$ max of branches (worst case) |
| `for` loop ($n$ iterations) | $n \times$ (cost per iteration) |
| Nested loops | Product of iteration counts $\times$ cost of innermost body |
| Function call | Cost of executing the function body |

These rules compose: a loop containing a conditional containing another loop
contributes the product of all iteration counts, adjusted for which branch executes.

!!! warning "Do Not Double-Count"

    When a loop header includes a comparison (e.g., `i < n`), that comparison executes
    one more time than the loop body — the final failing test. For asymptotic analysis
    this difference is absorbed, but for exact counts it matters.

## Reference

- [Introduction to Algorithms (CLRS), Chapters 2-4](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
