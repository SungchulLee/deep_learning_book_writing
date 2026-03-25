# Tower of Hanoi

The Tower of Hanoi is a classic problem that demonstrates the power of recursive thinking. Given three pegs and $n$ disks of decreasing size stacked on the first peg, the goal is to move all disks to the third peg, obeying two rules: move only one disk at a time, and never place a larger disk on top of a smaller one. Despite the simple rules, the solution reveals an elegant recursive structure.

## Problem Statement

Given $n$ disks on peg A, move all disks to peg C using peg B as auxiliary:

1. Only one disk may be moved at a time
2. Each move takes the top disk from one peg and places it on another
3. No disk may be placed on top of a smaller disk

## Recursive Solution

The key insight is that moving $n$ disks reduces to three subproblems:

1. Move the top $n - 1$ disks from A to B (using C as auxiliary)
2. Move the largest disk from A to C
3. Move the $n - 1$ disks from B to C (using A as auxiliary)

```python
"""Tower of Hanoi — recursive solution with move counting."""


# === Recursive Algorithm ===

def hanoi(n, source, auxiliary, target):
    """Move n disks from source to target using auxiliary peg."""
    if n == 1:
        print(f"Move disk 1 from {source} to {target}")
        return 1
    moves = 0
    moves += hanoi(n - 1, source, target, auxiliary)
    print(f"Move disk {n} from {source} to {target}")
    moves += 1
    moves += hanoi(n - 1, auxiliary, source, target)
    return moves


# === Main ===

if __name__ == "__main__":
    n = 3
    print(f"Tower of Hanoi with {n} disks:\n")
    total = hanoi(n, "A", "B", "C")
    print(f"\nTotal moves: {total}")
```

**Output:**
```
Tower of Hanoi with 3 disks:

Move disk 1 from A to C
Move disk 2 from A to B
Move disk 1 from C to B
Move disk 3 from A to C
Move disk 1 from B to A
Move disk 2 from B to C
Move disk 1 from A to C

Total moves: 7
```

## Recurrence Relation

Let $T(n)$ denote the number of moves required for $n$ disks. The recursive structure gives:

$$
T(n) = 2\,T(n - 1) + 1, \quad T(1) = 1
$$

## Closed-Form Solution

Expanding the recurrence:

$$
T(n) = 2^n - 1
$$

This can be verified by induction. For $n = 1$, $T(1) = 2^1 - 1 = 1$. Assuming $T(k) = 2^k - 1$, then $T(k+1) = 2(2^k - 1) + 1 = 2^{k+1} - 1$. $\square$

## Complexity

The Tower of Hanoi requires exactly $2^n - 1$ moves, giving $\Theta(2^n)$ time complexity. No algorithm can solve the problem in fewer moves — this is the optimal solution.

## Reference

[Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
