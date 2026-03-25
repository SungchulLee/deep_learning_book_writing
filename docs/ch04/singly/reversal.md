# Reversal

Reversing a singly linked list -- making the last node the new head and flipping the direction of every pointer -- is one of the most fundamental linked list operations. It appears as a subroutine in many algorithms (such as reversing the second half of a list to check for palindromes) and is a common building block in interview problems. Because singly linked list nodes only point forward, reversal requires carefully redirecting each `next` pointer to point backward, one node at a time, without losing track of the remaining list.

## Iterative Reversal

The iterative approach uses three pointers to walk through the list and reverse each link in a single pass.

**Algorithm:**

1. Initialize `prev = None`, `current = head`.
2. While `current` is not `None`:
    - Save `next_node = current.next` (so we do not lose the rest of the list).
    - Reverse the link: `current.next = prev`.
    - Advance: `prev = current`, `current = next_node`.
3. Return `prev` as the new head.

??? example "Step-by-Step Trace"

    Reversing `1 -> 2 -> 3 -> 4`:

    | Step | prev | current | next_node | Action                    |
    |------|------|---------|-----------|---------------------------|
    | 0    | None | 1       | 2         | 1.next = None             |
    | 1    | 1    | 2       | 3         | 2.next = 1                |
    | 2    | 2    | 3       | 4         | 3.next = 2                |
    | 3    | 3    | 4       | None      | 4.next = 3                |

    After step 3: `prev = Node(4)`, `current = None`. Return `Node(4)` as new head.

    Result: `4 -> 3 -> 2 -> 1`.

**Time complexity:** $O(n)$ -- each node is visited exactly once.

**Space complexity:** $O(1)$ -- only three pointer variables are used.

## Recursive Reversal

The recursive approach reverses the rest of the list first, then fixes the current node's pointer.

**Algorithm:**

1. Base case: if `head` is `None` or `head.next` is `None`, return `head`.
2. Recursively reverse the sublist starting from `head.next`: `new_head = reverse(head.next)`.
3. Make `head.next.next = head` (the node after `head` now points back to `head`).
4. Set `head.next = None` (break the old forward link).
5. Return `new_head`.

??? example "Recursive Trace"

    Reversing `1 -> 2 -> 3`:

    ```
    reverse(1)
      reverse(2)
        reverse(3)          # base case: return Node(3)
        3.next = None → set 3.next = 2, 2.next = None
        return Node(3)      # list: 3 -> 2
      2.next = None → set 2.next = 1, 1.next = None
      return Node(3)        # list: 3 -> 2 -> 1
    ```

**Time complexity:** $O(n)$ -- one recursive call per node.

**Space complexity:** $O(n)$ -- the recursion stack holds $n$ frames.

!!! warning "Stack Overflow Risk"

    The recursive approach uses $O(n)$ stack space. For lists with thousands of nodes, this can cause a stack overflow in Python (default recursion limit is 1000). The iterative approach is preferred for large lists.

## Reversing a Sublist

A useful variant reverses only the nodes between positions $m$ and $n$ (1-indexed), leaving the rest of the list unchanged.

**Algorithm:**

1. Traverse to the node at position $m - 1$ (the predecessor of the reversal segment).
2. Reverse the $n - m + 1$ nodes from position $m$ to $n$ using the iterative technique.
3. Reconnect the reversed segment to the rest of the list.

**Time complexity:** $O(n)$ -- single pass through the list.

## Implementation

```python
"""Reversal operations for a singly linked list."""


# === Node Class ===
class Node:
    """A single node in a singly linked list."""

    def __init__(self, data, next_node=None):
        self.data = data
        self.next = next_node

    def __repr__(self):
        return f"Node({self.data})"


# === Helper Functions ===
def build_list(values):
    """Create a linked list from an iterable, returning the head node."""
    head = None
    for val in reversed(values):
        head = Node(val, head)
    return head


def to_list(head):
    """Collect all node values into a Python list."""
    result = []
    current = head
    while current is not None:
        result.append(current.data)
        current = current.next
    return result


# === Iterative Reversal ===
def reverse_iterative(head):
    """Reverse a linked list iteratively.

    Time: O(n), Space: O(1).
    """
    prev = None
    current = head
    while current is not None:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    return prev


# === Recursive Reversal ===
def reverse_recursive(head):
    """Reverse a linked list recursively.

    Time: O(n), Space: O(n) due to recursion stack.
    """
    if head is None or head.next is None:
        return head
    new_head = reverse_recursive(head.next)
    head.next.next = head
    head.next = None
    return new_head


# === Reverse Sublist ===
def reverse_between(head, m, n):
    """Reverse nodes from position m to n (1-indexed).

    Time: O(n), Space: O(1).
    """
    if m == n:
        return head

    dummy = Node(0, head)
    prev = dummy
    for _ in range(m - 1):
        prev = prev.next

    # Reverse n - m + 1 nodes
    current = prev.next
    for _ in range(n - m):
        next_node = current.next
        current.next = next_node.next
        next_node.next = prev.next
        prev.next = next_node

    return dummy.next


# === Demonstration ===
if __name__ == "__main__":
    # Iterative reversal
    head = build_list([1, 2, 3, 4, 5])
    print(f"Original:    {to_list(head)}")
    head = reverse_iterative(head)
    print(f"Reversed:    {to_list(head)}")

    # Recursive reversal
    head = build_list([10, 20, 30, 40])
    head = reverse_recursive(head)
    print(f"\nRecursive:   {to_list(head)}")

    # Partial reversal (positions 2 to 4)
    head = build_list([1, 2, 3, 4, 5])
    head = reverse_between(head, 2, 4)
    print(f"\nPartial [2,4]: {to_list(head)}")
```

**Output:**
```
Original:    [1, 2, 3, 4, 5]
Reversed:    [5, 4, 3, 2, 1]

Recursive:   [40, 30, 20, 10]

Partial [2,4]: [1, 4, 3, 2, 5]
```

## Complexity Summary

| Variant               | Time   | Space  |
|-----------------------|--------|--------|
| Iterative reversal    | $O(n)$ | $O(1)$ |
| Recursive reversal    | $O(n)$ | $O(n)$ |
| Reverse sublist [m,n] | $O(n)$ | $O(1)$ |

## Reference

- [Introduction to Algorithms (CLRS), Chapter 10](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
