# Recursive Thinking

Recursion is more than a programming technique — it is a way of thinking about problems. A function is **recursive** if it calls itself within its own definition. The power of recursive thinking lies in reducing a complex problem to a simpler version of itself, trusting that the simpler version will be solved correctly.

## The Recursive Mindset

To think recursively, follow three steps:

1. **Identify the base case**: What is the simplest version of this problem, where the answer is immediate?
2. **Assume the recursive call works**: Trust that calling the function on a smaller input produces the correct result (this is the "recursive leap of faith")
3. **Combine**: Use the result of the recursive call to solve the current problem

## Example: Sum of First n Integers

Instead of thinking about a loop that accumulates a running total, recursive thinking frames the problem as:

- The sum of the first $n$ integers equals $n$ plus the sum of the first $n - 1$ integers
- The sum of zero integers is $0$

```python
"""Recursive thinking: sum of first n integers."""


# === Recursive Sum ===

def sum_to(n):
    """Return 1 + 2 + ... + n using recursive thinking."""
    if n == 0:       # Base case: nothing to add
        return 0
    return n + sum_to(n - 1)  # n plus the answer to the smaller problem


# === Main ===

if __name__ == "__main__":
    for n in [0, 1, 5, 10]:
        print(f"sum_to({n}) = {sum_to(n)}")
```

**Output:**
```
sum_to(0) = 0
sum_to(1) = 1
sum_to(5) = 15
sum_to(10) = 55
```

## The Recursive Leap of Faith

The hardest part of recursive thinking is trusting that the recursive call returns the correct answer. Do not try to trace every call mentally. Instead, verify two things:

1. The base case is correct
2. If the recursive call returns the right answer for the smaller problem, the current function returns the right answer for the current problem

If both hold, the function is correct by induction.

## Reference

[Recursion의 개념과 기본 예제들](https://www.youtube.com/watch?v=ln7AfppN7mY&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=1)
