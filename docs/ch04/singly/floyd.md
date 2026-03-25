# Detecting Cycles (Floyd)

A linked list normally terminates when a node's `next` pointer is `None`. But if a node points back to an earlier node in the list, the structure contains a **cycle**: following `next` pointers will loop forever without reaching `None`. Cycles typically arise from bugs, but detecting them is a classic algorithmic problem with applications in duplicate detection, deadlock detection, and pseudorandom number generator analysis. **Floyd's cycle detection algorithm** (also called the tortoise and hare algorithm) solves this problem using only $O(1)$ extra space and $O(n)$ time.

## Problem Statement

Given the head of a singly linked list, determine whether the list contains a cycle. If it does, find the node where the cycle begins.

A cycle exists if some node $x$ is reachable from itself by following `next` pointers: there exists a sequence $x \to x_1 \to x_2 \to \cdots \to x$. The **cycle entry point** is the first node in the list (starting from the head) that is part of the cycle.

## The Tortoise and Hare Algorithm

Floyd's algorithm uses two pointers that traverse the list at different speeds:

- **Slow (tortoise)**: advances one node per step.
- **Fast (hare)**: advances two nodes per step.

**Phase 1 -- Detect the cycle:** both pointers start at the head. If the fast pointer reaches `None`, there is no cycle. If a cycle exists, the fast pointer will eventually lap the slow pointer, and they will meet at some node inside the cycle.

**Phase 2 -- Find the cycle entry:** reset one pointer to the head and keep the other at the meeting point. Advance both pointers one step at a time. The node where they meet is the cycle entry point.

## Proof of Correctness

Let $\lambda$ be the length of the non-cyclic prefix (distance from head to the cycle entry), and let $\mu$ be the cycle length.

### Phase 1: Meeting Inside the Cycle

When the slow pointer enters the cycle (after $\lambda$ steps), the fast pointer has taken $2\lambda$ steps and is at position $\lambda \bmod \mu$ within the cycle. The fast pointer gains one position on the slow pointer each step, so they meet after at most $\mu$ additional steps. The total number of steps is $O(\lambda + \mu) = O(n)$.

### Phase 2: Finding the Entry Point

When the two pointers meet in phase 1, the slow pointer has taken $s$ steps and is at position $s - \lambda$ within the cycle. The fast pointer has taken $2s$ steps and is at the same position, so

$$
2s - \lambda \equiv s - \lambda \pmod{\mu}
$$

which simplifies to $s \equiv 0 \pmod{\mu}$, meaning $s = k\mu$ for some integer $k \ge 1$.

Now reset one pointer to the head. Both pointers advance at speed 1. After $\lambda$ steps:

- The pointer starting from the head reaches the cycle entry.
- The pointer starting from the meeting point has moved $\lambda$ steps within the cycle from position $s - \lambda = k\mu - \lambda$, reaching position $(k\mu - \lambda + \lambda) \bmod \mu = 0$, which is the cycle entry.

They meet at the cycle entry point. $\square$

## Worked Example

??? example "Step-by-Step Trace"

    Consider a list: `1 -> 2 -> 3 -> 4 -> 5 -> 3` (node 5 points back to node 3).

    The non-cyclic prefix has length $\lambda = 2$ (nodes 1, 2), and the cycle has length $\mu = 3$ (nodes 3, 4, 5).

    **Phase 1 (detection):**

    | Step | Slow position | Fast position |
    |------|---------------|---------------|
    | 0    | 1             | 1             |
    | 1    | 2             | 3             |
    | 2    | 3             | 5             |
    | 3    | 4             | 4             |

    They meet at node 4 (step 3).

    **Phase 2 (entry finding):**

    Reset one pointer to head (node 1), keep the other at node 4.

    | Step | From head | From meeting |
    |------|-----------|--------------|
    | 0    | 1         | 4            |
    | 1    | 2         | 5            |
    | 2    | 3         | 3            |

    They meet at node 3, which is the cycle entry point.

## Implementation

```python
"""Floyd's cycle detection algorithm for singly linked lists."""


# === Node Class ===
class Node:
    """A single node in a singly linked list."""

    def __init__(self, data, next_node=None):
        self.data = data
        self.next = next_node

    def __repr__(self):
        return f"Node({self.data})"


# === Floyd's Cycle Detection ===
def has_cycle(head):
    """Detect whether a linked list contains a cycle.

    Returns True if a cycle exists, False otherwise.
    Time: O(n), Space: O(1).
    """
    slow = head
    fast = head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False


def find_cycle_entry(head):
    """Find the entry point of a cycle in a linked list.

    Returns the node where the cycle begins, or None if no cycle.
    Time: O(n), Space: O(1).
    """
    # Phase 1: detect meeting point
    slow = head
    fast = head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            break
    else:
        return None  # no cycle

    # Phase 2: find entry point
    entry = head
    while entry is not slow:
        entry = entry.next
        slow = slow.next
    return entry


def cycle_length(head):
    """Return the length of the cycle, or 0 if no cycle exists."""
    entry = find_cycle_entry(head)
    if entry is None:
        return 0
    current = entry.next
    length = 1
    while current is not entry:
        current = current.next
        length += 1
    return length


# === Demonstration ===
if __name__ == "__main__":
    # Build a list with a cycle: 1 -> 2 -> 3 -> 4 -> 5 -> (back to 3)
    nodes = [Node(i) for i in range(1, 6)]
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    nodes[4].next = nodes[2]  # create cycle: 5 -> 3

    print(f"Has cycle: {has_cycle(nodes[0])}")
    print(f"Cycle entry: {find_cycle_entry(nodes[0])}")
    print(f"Cycle length: {cycle_length(nodes[0])}")

    # Test with no cycle
    head = Node(1, Node(2, Node(3)))
    print(f"\nNo-cycle list:")
    print(f"Has cycle: {has_cycle(head)}")
    print(f"Cycle entry: {find_cycle_entry(head)}")
```

**Output:**
```
Has cycle: True
Cycle entry: Node(3)
Cycle length: 3

No-cycle list:
Has cycle: False
Cycle entry: None
```

## Complexity Analysis

| Aspect            | Complexity |
|-------------------|------------|
| Time (detection)  | $O(n)$     |
| Time (entry find) | $O(n)$     |
| Space             | $O(1)$     |

The algorithm uses only two pointer variables regardless of the list size, achieving optimal $O(1)$ space. The alternative approach -- using a hash set to track visited nodes -- uses $O(n)$ space.

## Reference

- [Find the Duplicate Number - Floyd's Cycle Detection - Leetcode 287](https://www.youtube.com/watch?v=wjYnzkAhcNk)
- [287. Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/)
