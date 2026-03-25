# Loop Analysis

Loops are the primary source of nonconstant running time in algorithms. A single
statement costs $O(1)$, but wrapping it in a loop that iterates $n$ times produces
$O(n)$ total cost. Understanding how to count loop iterations — and how the loop
variable changes from one iteration to the next — is the essential skill in algorithm
analysis.

## Linear Loops

A **linear loop** increments (or decrements) its counter by a constant amount each
iteration.

```
for i = 0 to n - 1:
    body                    # O(1)
```

The loop variable $i$ takes values $0, 1, 2, \ldots, n - 1$, so the body executes
exactly $n$ times. Total cost:

$$
T(n) = n \cdot O(1) = O(n)
$$

More generally, a loop `for i = a to b` with step size $s$ runs
$\lfloor (b - a) / s \rfloor + 1$ times.

??? example "Summing Every Other Element"

    ```
    SumEven(A, n):
        total = 0
        for i = 0 to n - 1 step 2:
            total = total + A[i]
        return total
    ```

    The loop runs $\lceil n / 2 \rceil$ times. Since each iteration costs $O(1)$,
    the total is $T(n) = \Theta(n)$. The factor of $1/2$ vanishes inside
    $\Theta$-notation.

## Logarithmic Loops

A **logarithmic loop** multiplies or divides the counter by a constant factor each
iteration.

```
i = 1
while i < n:
    body                    # O(1)
    i = i * 2
```

The variable $i$ takes values $1, 2, 4, 8, \ldots$ and stops when $i \geq n$. After
$k$ iterations, $i = 2^k$. The loop ends when $2^k \geq n$, which gives:

$$
k = \lceil \log_2 n \rceil
$$

Total cost: $T(n) = O(\log n)$.

Similarly, dividing by a constant produces a logarithmic loop:

```
i = n
while i >= 1:
    body                    # O(1)
    i = i / 2
```

This also runs $O(\log n)$ times because $n / 2^k < 1$ when $k > \log_2 n$.

??? example "Repeated Halving"

    Binary search reduces the search space by half at each step:

    ```
    BinarySearch(A, target, lo, hi):
        while lo <= hi:
            mid = (lo + hi) / 2
            if A[mid] == target:
                return mid
            else if A[mid] < target:
                lo = mid + 1
            else:
                hi = mid - 1
        return -1
    ```

    The range $[\text{lo}, \text{hi}]$ halves each iteration, so the loop runs at
    most $\lfloor \log_2 n \rfloor + 1$ times, giving $T(n) = O(\log n)$.

## Square-Root Loops

Some loops have counters that grow quadratically, producing a square-root iteration
count.

```
i = 0
while i * i < n:
    body                    # O(1)
    i = i + 1
```

The loop stops when $i^2 \geq n$, so $i = \lceil \sqrt{n} \rceil$. Total cost:
$T(n) = O(\sqrt{n})$.

## While Loops with Data-Dependent Termination

When a `while` loop's termination depends on the input data rather than a simple
counter, the iteration count varies across inputs.

```
LinearSearch(A, n, target):
    i = 0
    while i < n and A[i] != target:
        i = i + 1
    if i < n:
        return i
    return -1
```

- **Best case:** $A[0]$ = target, loop runs 1 time, $T_{\text{best}}(n) = O(1)$.
- **Worst case:** target not in $A$, loop runs $n$ times,
  $T_{\text{worst}}(n) = O(n)$.

For such loops, the analysis requires considering best, worst, and average cases
separately.

## General Loop Analysis Strategy

Given any loop, follow these steps:

1. **Identify the loop variable** and how it changes each iteration (increment,
   multiply, or data-dependent).
2. **Determine the stopping condition** in terms of $n$ and the loop variable.
3. **Solve for the number of iterations** $k$ by finding when the loop variable
   reaches the stopping condition.
4. **Multiply** the iteration count by the cost per iteration.

| Loop Pattern | Variable Update | Iterations | Running Time |
|---|---|---|---|
| `i = 0; i < n; i++` | $i \leftarrow i + 1$ | $n$ | $O(n)$ |
| `i = 0; i < n; i += c` | $i \leftarrow i + c$ | $n / c$ | $O(n)$ |
| `i = 1; i < n; i *= 2` | $i \leftarrow 2i$ | $\log_2 n$ | $O(\log n)$ |
| `i = 1; i < n; i *= c` | $i \leftarrow ci$ | $\log_c n$ | $O(\log n)$ |
| `i = n; i >= 1; i /= 2` | $i \leftarrow i/2$ | $\log_2 n$ | $O(\log n)$ |
| `i = 0; i*i < n; i++` | $i \leftarrow i + 1$ | $\sqrt{n}$ | $O(\sqrt{n})$ |

!!! tip "Logarithm Base Does Not Matter"

    Since $\log_c n = \frac{\log_2 n}{\log_2 c}$ and $\log_2 c$ is a constant, all
    logarithmic bases produce the same asymptotic class $O(\log n)$. We typically omit
    the base in asymptotic notation.

## Reference

- [Introduction to Algorithms (CLRS), Chapters 2-4](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
