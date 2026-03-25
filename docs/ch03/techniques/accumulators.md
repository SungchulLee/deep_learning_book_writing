# Recursion with Accumulators

An **accumulator** is an extra parameter that carries a running result through recursive calls. Instead of building up the answer on the way back from the base case (as in standard recursion), accumulator-based recursion builds the answer on the way *down*, making the recursive call the last operation — which enables tail recursion.

## Standard vs Accumulator Style

Consider computing the factorial. Standard recursion multiplies *after* the recursive call returns:

```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)  # multiplication waits for return
```

With an accumulator, the multiplication happens *before* the recursive call:

```python
"""Accumulator-based recursion demonstrated with factorial."""


# === Accumulator Factorial ===

def factorial_acc(n, acc=1):
    """Compute n! using an accumulator to carry the running product."""
    if n <= 1:
        return acc
    return factorial_acc(n - 1, acc * n)  # result carried forward


# === Main ===

if __name__ == "__main__":
    for n in [0, 1, 5, 10]:
        print(f"{n}! = {factorial_acc(n)}")
```

**Output:**
```
0! = 1
1! = 1
5! = 120
10! = 3628800
```

## How Accumulators Work

In standard recursion, the call stack holds pending operations (multiplications, additions) that execute as frames unwind. With an accumulator, there are no pending operations — the current result is always in the accumulator parameter:

| Call | `n` | `acc` |
|---|---|---|
| `factorial_acc(5, 1)` | 5 | 1 |
| `factorial_acc(4, 5)` | 4 | 5 |
| `factorial_acc(3, 20)` | 3 | 20 |
| `factorial_acc(2, 60)` | 2 | 60 |
| `factorial_acc(1, 120)` | 1 | 120 → return |

## Benefits

Accumulator-style recursion is **tail recursive** — the recursive call is the last operation, with no pending computation. Languages that support tail call optimization can reuse the current stack frame, converting $O(n)$ stack usage to $O(1)$.

## Reference

[Recursion의 개념과 기본 예제들](https://www.youtube.com/watch?v=Vwfo_hrxuzg&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=3)
