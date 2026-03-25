# Binary Recursion

Binary recursion occurs when a function makes **two** recursive calls per invocation. This pattern arises naturally in divide-and-conquer algorithms, where a problem splits into two subproblems of roughly equal size. The classic examples include merge sort, quicksort, and computing Fibonacci numbers.

## Structure

A binary recursive function has the form:

```
function f(problem):
    if base_case:
        return simple_answer
    left  = f(left_subproblem)
    right = f(right_subproblem)
    return combine(left, right)
```

The two recursive calls create a binary tree of invocations, which is why binary recursion is closely related to binary tree traversals.

## Example: Fibonacci Numbers

The Fibonacci sequence is the most direct example of binary recursion. Each call spawns two smaller calls:

```python
"""Binary recursion demonstrated with Fibonacci numbers."""


# === Naive Binary Recursive Fibonacci ===

def fib(n):
    """Compute the nth Fibonacci number using binary recursion."""
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)


# === Main ===

if __name__ == "__main__":
    for i in range(10):
        print(f"fib({i}) = {fib(i)}")
```

**Output:**
```
fib(0) = 0
fib(1) = 1
fib(2) = 1
fib(3) = 2
fib(4) = 3
fib(5) = 5
fib(6) = 8
fib(7) = 13
fib(8) = 21
fib(9) = 34
```

## Complexity

Binary recursion produces an exponential number of calls in the naive case. For Fibonacci:

$$
T(n) = T(n-1) + T(n-2) + O(1)
$$

This gives $T(n) = O(2^n)$ — exponential time, because many subproblems are recomputed. Techniques like memoization can reduce this to $O(n)$.

When the two subproblems are of equal size (as in merge sort), the recurrence becomes $T(n) = 2T(n/2) + O(n)$, which solves to $O(n \log n)$ by the master theorem.

## Reference

[Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
