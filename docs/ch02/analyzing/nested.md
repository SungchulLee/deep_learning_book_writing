# Nested Loops

Nested loops are the most common source of polynomial running times. When one loop
sits inside another, the total number of operations is not the sum of their iteration
counts but — in the simplest case — their product. Analyzing nested loops precisely
requires translating the loop structure into summations and evaluating them in closed
form.

## Independent Nested Loops

When the inner loop's bounds do not depend on the outer loop variable, the total
iteration count is the product of the two ranges.

```
for i = 0 to n - 1:
    for j = 0 to n - 1:
        body                # O(1)
```

The outer loop runs $n$ times. For each outer iteration, the inner loop runs $n$
times. Total cost:

$$
T(n) = \sum_{i=0}^{n-1} \sum_{j=0}^{n-1} O(1) = n \cdot n \cdot O(1) = O(n^2)
$$

This generalizes to $k$ levels of nesting with independent bounds:

$$
T(n) = \underbrace{n \cdot n \cdots n}_{k \text{ factors}} = O(n^k)
$$

??? example "Matrix Initialization"

    ```
    InitMatrix(n):
        for i = 0 to n - 1:
            for j = 0 to n - 1:
                M[i][j] = 0
    ```

    Both loops are independent and run $n$ times each. The assignment executes
    $n \times n = n^2$ times, so $T(n) = \Theta(n^2)$.

## Dependent Nested Loops

When the inner loop's bounds depend on the outer loop variable, the iteration count
changes with each outer iteration. We must evaluate the resulting summation.

### Inner Loop Depends on Outer Variable

```
for i = 0 to n - 1:
    for j = 0 to i:
        body                # O(1)
```

For outer iteration $i$, the inner loop runs $i + 1$ times. Total cost:

$$
T(n) = \sum_{i=0}^{n-1} (i + 1) = \sum_{k=1}^{n} k = \frac{n(n+1)}{2} = O(n^2)
$$

??? example "Selection Sort Inner Loop"

    ```
    SelectionSort(A, n):
        for i = 0 to n - 2:
            min_idx = i
            for j = i + 1 to n - 1:
                if A[j] < A[min_idx]:
                    min_idx = j
            swap A[i] and A[min_idx]
    ```

    For each $i$, the inner loop runs $n - 1 - i$ times:

    $$
    T(n) = \sum_{i=0}^{n-2}(n - 1 - i) = \sum_{k=1}^{n-1} k = \frac{(n-1)n}{2} = \Theta(n^2)
    $$

### Inner Loop with Logarithmic Bound

```
for i = 1 to n:
    for j = 1 to log(i):
        body                # O(1)
```

Total cost:

$$
T(n) = \sum_{i=1}^{n} \lfloor \log_2 i \rfloor \leq \sum_{i=1}^{n} \log_2 i = \log_2(n!) = \Theta(n \log n)
$$

The last step uses Stirling's approximation: $\log(n!) = \Theta(n \log n)$.

## Mixed Nesting: Linear Outer, Logarithmic Inner

```
for i = 0 to n - 1:        # O(n) iterations
    j = n
    while j >= 1:           # O(log n) iterations
        body                # O(1)
        j = j / 2
```

The outer loop runs $n$ times. The inner loop runs $O(\log n)$ times regardless of
$i$ (since it only depends on $n$). Total cost:

$$
T(n) = n \cdot O(\log n) = O(n \log n)
$$

## Deeply Nested Loops

For three or more levels, apply the same principle: sum from the innermost loop
outward.

??? example "Three Independent Loops"

    ```
    for i = 0 to n - 1:
        for j = 0 to n - 1:
            for k = 0 to n - 1:
                body            # O(1)
    ```

    $$
    T(n) = \sum_{i=0}^{n-1} \sum_{j=0}^{n-1} \sum_{k=0}^{n-1} 1 = n^3 = \Theta(n^3)
    $$

??? example "Three Dependent Loops"

    ```
    for i = 0 to n - 1:
        for j = 0 to i:
            for k = 0 to j:
                body            # O(1)
    ```

    $$
    T(n) = \sum_{i=0}^{n-1} \sum_{j=0}^{i} (j+1) = \sum_{i=0}^{n-1} \frac{(i+1)(i+2)}{2} = \Theta(n^3)
    $$

    The exact value is $\frac{n(n+1)(n+2)}{6}$, which follows from the identity
    $\sum_{i=1}^{n} \binom{i+1}{2} = \binom{n+2}{3}$.

## Common Pitfall: Not All Nested Loops Are Quadratic

A nested loop does not always produce $O(n^2)$. If the total work across all inner
iterations is bounded, the result can be linear.

!!! warning "Amortized Inner Loop Work"

    ```
    TwoPointer(A, n):
        j = 0
        for i = 0 to n - 1:
            while j < n and A[j] < A[i]:
                j = j + 1
            process(i, j)
    ```

    Although the `while` loop is inside the `for` loop, $j$ only increases and never
    resets. Across all iterations of the outer loop, $j$ increments at most $n$ times
    total. The overall cost is $O(n) + O(n) = O(n)$, not $O(n^2)$.

## Summation Reference

| Summation | Closed Form | Growth |
|---|---|---|
| $\sum_{i=1}^{n} 1$ | $n$ | $\Theta(n)$ |
| $\sum_{i=1}^{n} i$ | $\frac{n(n+1)}{2}$ | $\Theta(n^2)$ |
| $\sum_{i=1}^{n} i^2$ | $\frac{n(n+1)(2n+1)}{6}$ | $\Theta(n^3)$ |
| $\sum_{i=0}^{k} 2^i$ | $2^{k+1} - 1$ | $\Theta(2^k)$ |
| $\sum_{i=1}^{n} \frac{1}{i}$ | $H_n \approx \ln n$ | $\Theta(\log n)$ |

These closed forms are the building blocks for evaluating nested loop summations.

## Reference

- [Introduction to Algorithms (CLRS), Chapters 2-4](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
