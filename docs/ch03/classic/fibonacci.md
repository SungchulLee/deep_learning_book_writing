# Fibonacci Numbers via Recursion

The Fibonacci sequence is one of the most natural examples of recursion in computer science. Each term is defined as the sum of the two preceding terms, making the recursive definition almost a direct translation of the mathematical formula. However, the naive recursive implementation reveals a fundamental efficiency problem that motivates key algorithm design techniques like memoization and dynamic programming.

## Definition

The Fibonacci sequence is defined recursively by

$$
F(n) = \begin{cases} 0 & \text{if } n = 0 \\ 1 & \text{if } n = 1 \\ F(n-1) + F(n-2) & \text{if } n \geq 2 \end{cases}
$$

The first several terms are: $0, 1, 1, 2, 3, 5, 8, 13, 21, 34, \ldots$

## Naive Recursive Implementation

The recursive definition translates directly into Python code:

```python
def compute_fibonacci_using_recursion(n):
    """Compute the n-th Fibonacci number using naive recursion.

    Time complexity: O(2^n) -- exponential due to repeated subproblems.
    Space complexity: O(n) -- maximum recursion depth.
    """
    if n <= 1:
        return n
    return compute_fibonacci_using_recursion(n - 1) + compute_fibonacci_using_recursion(n - 2)


# === Main ===
if __name__ == "__main__":
    n = 10
    for i in range(n):
        result = compute_fibonacci_using_recursion(i)
        print(f"fibonacci({i}) = {result}")
```

**Output:**
```
fibonacci(0) = 0
fibonacci(1) = 1
fibonacci(2) = 1
fibonacci(3) = 2
fibonacci(4) = 3
fibonacci(5) = 5
fibonacci(6) = 8
fibonacci(7) = 13
fibonacci(8) = 21
fibonacci(9) = 34
```

## Complexity Analysis

The naive recursive approach has **exponential** time complexity. Each call to `compute_fibonacci_using_recursion(n)` spawns two recursive calls, and many subproblems are solved repeatedly. The number of calls satisfies the recurrence $T(n) = T(n-1) + T(n-2) + O(1)$, which grows as

$$
T(n) = O(\phi^n) \quad \text{where } \phi = \frac{1 + \sqrt{5}}{2} \approx 1.618
$$

For $n = 40$, this results in over one billion function calls. The redundant computation makes naive recursion impractical for all but the smallest inputs.

!!! warning "Exponential Blowup"
    The naive recursive Fibonacci is a classic example of how a correct algorithm can be unusably slow. The fix -- storing previously computed results (memoization) or building the solution bottom-up (dynamic programming) -- reduces the time complexity from $O(\phi^n)$ to $O(n)$.

## References

[Introduction to Algorithms (CLRS), Section 15.1](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
