# Working Backwards

**Working backwards** means starting from the desired result and reasoning toward the initial state. This technique is particularly powerful for construction problems, game theory, and problems where the forward direction has too many choices.

## When to Work Backwards

- The end state is well-defined but the start state has many possibilities.
- Forward simulation branches exponentially, but reverse simulation converges.
- The problem involves undoing operations.

## Example: Reducing to 1

**Problem:** Given a number $n$, find the minimum number of operations to reduce it to 1. Operations: subtract 1, divide by 2 (if even), divide by 3 (if divisible by 3).

```python
from collections import deque

def min_operations_to_one(n):
    if n == 1:
        return 0
    visited = {n}
    queue = deque([(n, 0)])
    while queue:
        val, steps = queue.popleft()
        candidates = [val - 1]
        if val % 2 == 0:
            candidates.append(val // 2)
        if val % 3 == 0:
            candidates.append(val // 3)
        for next_val in candidates:
            if next_val == 1:
                return steps + 1
            if next_val not in visited and next_val > 0:
                visited.add(next_val)
                queue.append((next_val, steps + 1))
    return -1

print(min_operations_to_one(10))  # Output: 3 (10->9->3->1)
```

## Example: Constructing a Target Array

**Problem:** Given a target array, determine if it can be built from `[1, 1, ..., 1]` by repeatedly replacing the maximum element with (max - sum_of_others).

**Key insight:** Work backwards. If the target is `[x, y]` with $x > y$, then the previous state was `[x - y, y]`. This is essentially the GCD algorithm.

```python
import heapq

def is_possible(target):
    total = sum(target)
    heap = [-x for x in target]  # max-heap via negation
    heapq.heapify(heap)
    while True:
        largest = -heapq.heappop(heap)
        rest = total - largest
        if largest == 1 or rest == 1:
            return True
        if rest == 0 or largest <= rest:
            return False
        largest %= rest  # Work backwards: undo multiple steps at once
        if largest == 0:
            largest = rest
        total = rest + largest
        heapq.heappush(heap, -largest)

print(is_possible([9, 3, 5]))  # Output: True
```

## Working Backwards in Proofs

Working backwards also helps in proof design: start from what you want to prove and determine what conditions are sufficient to reach it.

# Reference

- Polya, G. *How to Solve It*, Princeton University Press, 1945.
- LeetCode Problem 1354: Construct Target Array With Multiple Sums.
