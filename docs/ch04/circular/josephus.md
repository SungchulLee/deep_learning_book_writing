# Josephus Problem

The Josephus problem is a classical counting-out puzzle that arises naturally
from circular linked lists. Imagine $n$ people standing in a circle. Starting
from a designated first person, every $k$-th person is eliminated until only
one remains. The question is: **which position survives?** This problem
appears in round-robin scheduling, game theory, and cryptography. It also
serves as a compelling demonstration of why circular linked lists exist --
the circular structure directly mirrors the problem's geometry.

## Problem Statement

Given $n$ people numbered $0, 1, \ldots, n-1$ arranged in a circle and a
step size $k \geq 1$:

1. Start at person 0.
2. Count $k$ people clockwise (including the starting person).
3. Eliminate the $k$-th person.
4. Repeat from the next person until one person remains.

The **Josephus number** $J(n, k)$ is the position of the last survivor.

## Simulation with a Circular Linked List

The most intuitive approach simulates the process directly using a circular
singly linked list. Each node represents one person. When a person is
eliminated, the corresponding node is deleted and counting continues around
the ring.

```python
"""
Josephus problem solved by simulating elimination on a circular linked list.

Demonstrates circular list traversal, node deletion, and the Josephus
recurrence for verification.
"""


# === Node and Circular List ===

class Node:
    """A node in a circular singly linked list."""

    def __init__(self, data):
        self.data = data
        self.next = None


def build_circle(n):
    """Build a circular singly linked list with nodes 0, 1, ..., n-1.

    Returns the node for person 0.
    """
    head = Node(0)
    current = head
    for i in range(1, n):
        current.next = Node(i)
        current = current.next
    current.next = head       # close the ring
    return head


# === Simulation Approach ===

def josephus_simulation(n, k):
    """Return the survivor's position using circular list simulation.

    Time: O(n * k) -- each of the n-1 eliminations requires k steps.
    Space: O(n) for the circular linked list.
    """
    head = build_circle(n)
    current = head

    # Find the node just before the starting position
    prev = head
    while prev.next is not head:
        prev = prev.next

    for _ in range(n - 1):
        # Count k-1 steps forward (current is step 1)
        for _ in range(k - 1):
            prev = current
            current = current.next
        # Eliminate current
        prev.next = current.next
        current = current.next

    return current.data


# === Recurrence Approach ===

def josephus_recurrence(n, k):
    """Return the survivor's position using the mathematical recurrence.

    The Josephus recurrence for 0-indexed positions:
        J(1, k) = 0
        J(n, k) = (J(n-1, k) + k) mod n

    Time: O(n)
    Space: O(1)
    """
    survivor = 0
    for i in range(2, n + 1):
        survivor = (survivor + k) % i
    return survivor


# === Main ===

if __name__ == "__main__":
    # Example: 7 people, every 3rd eliminated
    n, k = 7, 3
    print(f"Josephus({n}, {k})")
    print(f"  Simulation:  {josephus_simulation(n, k)}")
    print(f"  Recurrence:  {josephus_recurrence(n, k)}")

    # Verify both methods agree for several inputs
    print("\nVerification:")
    for n in range(1, 11):
        sim = josephus_simulation(n, 3)
        rec = josephus_recurrence(n, 3)
        status = "OK" if sim == rec else "MISMATCH"
        print(f"  n={n:2d}, k=3: survivor={sim}  [{status}]")
```

**Output:**

```
Josephus(7, 3)
  Simulation:  3
  Recurrence:  3

Verification:
  n= 1, k=3: survivor=0  [OK]
  n= 2, k=3: survivor=1  [OK]
  n= 3, k=3: survivor=1  [OK]
  n= 4, k=3: survivor=0  [OK]
  n= 5, k=3: survivor=3  [OK]
  n= 6, k=3: survivor=0  [OK]
  n= 7, k=3: survivor=3  [OK]
  n= 8, k=3: survivor=6  [OK]
  n= 9, k=3: survivor=0  [OK]
  n=10, k=3: survivor=3  [OK]
```

## The Josephus Recurrence

Rather than simulating elimination physically, the problem can be solved with
a recurrence relation. After eliminating the first person (at position
$k - 1$), the remaining $n - 1$ people form a new circle, but the
numbering shifts by $k$ positions. This gives the 0-indexed recurrence:

$$
J(1, k) = 0
$$

$$
J(n, k) = \bigl(J(n-1, k) + k\bigr) \bmod n \quad \text{for } n \geq 2
$$

The recurrence computes the answer in $O(n)$ time and $O(1)$ space using
a simple loop (shown in the code above), compared to $O(nk)$ for the
simulation.

??? note "Derivation of the recurrence"
    After eliminating person at position $k - 1$ (0-indexed) from a circle
    of $n$, relabel the remaining $n - 1$ people starting from position $k$.
    Person at old position $j$ maps to new position $(j - k) \bmod (n-1)$.
    If $J(n-1, k)$ is the survivor in the relabeled circle, the survivor in
    the original circle is at position $(J(n-1, k) + k) \bmod n$.

## Special Case: k = 2

When $k = 2$, the Josephus problem has a closed-form solution. Write
$n = 2^m + \ell$ where $0 \leq \ell < 2^m$. Then:

$$
J(n, 2) = 2\ell + 1
$$

This can be computed in $O(\log n)$ time using bit operations: find the
highest set bit of $n$, and rotate the binary representation left by one
position.

## Complexity Summary

| Method | Time | Space |
|---|---|---|
| Circular list simulation | $O(nk)$ | $O(n)$ |
| Iterative recurrence | $O(n)$ | $O(1)$ |
| Closed form ($k = 2$ only) | $O(\log n)$ | $O(1)$ |

The simulation approach is the most intuitive and demonstrates the power of
circular linked lists, but the recurrence is vastly more efficient for large
inputs.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C.
  *Introduction to Algorithms* (4th ed.), Problem 14-2. MIT Press.
- Graham, R. L., Knuth, D. E., & Patashnik, O. *Concrete Mathematics*
  (2nd ed.), Section 1.3. Addison-Wesley.
