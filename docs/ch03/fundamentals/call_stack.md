# Call Stack

When a function calls itself recursively, each invocation creates a new entry on the **call stack** — the runtime data structure that tracks active function calls. Understanding the call stack is essential for reasoning about recursion, because it explains how the computer remembers where to return after each recursive call completes.

## How the Call Stack Works

Each time a function is called, the runtime pushes a **frame** onto the call stack containing:

1. The function's local variables and parameters
2. The return address (where to resume after the call finishes)

When the function returns, its frame is popped off the stack and execution resumes at the return address.

## Example: Tracing the Call Stack

Consider a recursive countdown function. Each call adds a frame to the stack, and frames are removed as calls return:

```python
"""Demonstration of call stack behavior during recursion."""


# === Recursive Countdown with Stack Tracing ===

def countdown(n, depth=0):
    """Print countdown with indentation showing stack depth."""
    indent = "  " * depth
    print(f"{indent}countdown({n}) called  [stack depth: {depth + 1}]")
    if n == 0:
        print(f"{indent}Base case reached, returning")
        return
    countdown(n - 1, depth + 1)
    print(f"{indent}countdown({n}) returning  [stack depth: {depth + 1}]")


# === Main ===

if __name__ == "__main__":
    countdown(3)
```

**Output:**
```
countdown(3) called  [stack depth: 1]
  countdown(2) called  [stack depth: 2]
    countdown(1) called  [stack depth: 3]
      countdown(0) called  [stack depth: 4]
      Base case reached, returning
    countdown(1) returning  [stack depth: 3]
  countdown(2) returning  [stack depth: 2]
countdown(3) returning  [stack depth: 1]
```

## Stack Overflow

If a recursive function lacks a proper base case, or the base case is never reached, the call stack grows without bound until the system's stack space is exhausted. This produces a **stack overflow** error:

```python
def infinite_recursion(n):
    """This function never terminates — n grows, never reaching 0."""
    print("H", end="")
    infinite_recursion(n + 1)  # n increases, base case at 0 is unreachable

# Calling infinite_recursion(1) would produce:
# RecursionError: maximum recursion depth exceeded
```

Python limits recursion depth (default: 1000) to prevent true stack overflow. This safety mechanism converts infinite recursion into a `RecursionError`.

## Reference

[Recursion의 개념과 기본 예제들](https://www.youtube.com/watch?v=ln7AfppN7mY&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=1)
