# Traversal

Traversal is the most basic linked list operation: visiting every node in sequence from the head to the tail. Nearly every other operation on a linked list -- searching, counting, printing, summing, finding the maximum -- is built on top of traversal. Because singly linked list nodes are scattered in memory and connected only by `next` pointers, the only way to visit all elements is to follow the chain one node at a time. This takes $O(n)$ time and $O(1)$ space in the iterative form, making it the baseline against which all other linked list operations are measured.

## Iterative Traversal

The standard traversal pattern uses a single pointer that starts at the head and advances through each node.

**Algorithm:**

1. Set `current = head`.
2. While `current` is not `None`:
    - Process `current.data` (print, accumulate, transform, etc.).
    - Advance: `current = current.next`.

**Time complexity:** $O(n)$ -- each of the $n$ nodes is visited exactly once.

**Space complexity:** $O(1)$ -- only a single pointer variable.

This pattern is the linked list analog of iterating through an array with a for-loop. The key difference is that each "step" involves a pointer dereference rather than an index increment, which means the memory accesses are non-sequential and less cache-friendly.

## Recursive Traversal

Traversal can also be expressed recursively. The recursive form processes the current node, then calls itself on the rest of the list.

**Algorithm:**

```
traverse(node):
    if node is None:
        return
    process(node.data)
    traverse(node.next)
```

**Time complexity:** $O(n)$.

**Space complexity:** $O(n)$ due to $n$ recursive stack frames.

!!! warning "Recursion Depth"

    Recursive traversal is elegant but uses $O(n)$ stack space, which causes stack overflow for long lists. Python's default recursion limit is 1000 frames. The iterative approach is always preferred for production code.

## Common Traversal Patterns

Several practical operations are direct applications of traversal with different processing steps.

### Counting Nodes

Count the number of nodes by incrementing a counter during traversal.

**Time:** $O(n)$ | **Space:** $O(1)$

### Summing Values

Accumulate the sum of all node values during a single traversal.

**Time:** $O(n)$ | **Space:** $O(1)$

### Finding the Maximum

Track the maximum value seen so far as each node is visited.

**Time:** $O(n)$ | **Space:** $O(1)$

### Collecting into a List

Build a Python list of all node values. This is useful for printing or comparing against expected output.

**Time:** $O(n)$ | **Space:** $O(n)$ for the output list

### Printing the List

Display the list in `a -> b -> c -> None` format by collecting values during traversal and joining with arrows.

## Implementation

```python
"""Traversal operations for a singly linked list."""


# === Node Class ===
class Node:
    """A single node in a singly linked list."""

    def __init__(self, data, next_node=None):
        self.data = data
        self.next = next_node

    def __repr__(self):
        return f"Node({self.data})"


# === Build Helper ===
def build_list(values):
    """Create a linked list from an iterable, returning the head node."""
    head = None
    for val in reversed(values):
        head = Node(val, head)
    return head


# === Iterative Traversal ===
def print_list(head):
    """Print the linked list in arrow notation."""
    parts = []
    current = head
    while current is not None:
        parts.append(str(current.data))
        current = current.next
    print(" -> ".join(parts) + " -> None")


# === Count Nodes ===
def count_nodes(head):
    """Return the number of nodes in the list. Time: O(n), Space: O(1)."""
    count = 0
    current = head
    while current is not None:
        count += 1
        current = current.next
    return count


# === Sum Values ===
def sum_values(head):
    """Return the sum of all node values. Time: O(n), Space: O(1)."""
    total = 0
    current = head
    while current is not None:
        total += current.data
        current = current.next
    return total


# === Find Maximum ===
def find_max(head):
    """Return the maximum value in the list. Time: O(n), Space: O(1)."""
    if head is None:
        raise ValueError("Cannot find max of empty list")
    max_val = head.data
    current = head.next
    while current is not None:
        if current.data > max_val:
            max_val = current.data
        current = current.next
    return max_val


# === Collect to Python List ===
def to_list(head):
    """Collect all node values into a Python list. Time: O(n), Space: O(n)."""
    result = []
    current = head
    while current is not None:
        result.append(current.data)
        current = current.next
    return result


# === Recursive Traversal ===
def print_recursive(node):
    """Print values recursively. Time: O(n), Space: O(n)."""
    if node is None:
        print("None")
        return
    print(f"{node.data} -> ", end="")
    print_recursive(node.next)


# === Apply Function to Each Node ===
def for_each(head, func):
    """Apply func to each node's data. Time: O(n), Space: O(1)."""
    current = head
    while current is not None:
        func(current.data)
        current = current.next


# === Demonstration ===
if __name__ == "__main__":
    head = build_list([10, 20, 30, 40, 50])

    # Print using iterative traversal
    print("Iterative print:")
    print_list(head)

    # Print using recursive traversal
    print("\nRecursive print:")
    print_recursive(head)

    # Count, sum, max
    print(f"\nCount: {count_nodes(head)}")
    print(f"Sum:   {sum_values(head)}")
    print(f"Max:   {find_max(head)}")

    # Collect to Python list
    print(f"As list: {to_list(head)}")

    # Apply function
    print("\nDoubled values:")
    for_each(head, lambda x: print(f"  {x * 2}"))
```

**Output:**
```
Iterative print:
10 -> 20 -> 30 -> 40 -> 50 -> None

Recursive print:
10 -> 20 -> 30 -> 40 -> 50 -> None

Count: 5
Sum:   150
Max:   50
As list: [10, 20, 30, 40, 50]

Doubled values:
  20
  40
  60
  80
  100
```

## Complexity Summary

| Operation         | Time   | Space  |
|-------------------|--------|--------|
| Iterative traversal | $O(n)$ | $O(1)$ |
| Recursive traversal | $O(n)$ | $O(n)$ |
| Count nodes       | $O(n)$ | $O(1)$ |
| Sum values        | $O(n)$ | $O(1)$ |
| Find maximum      | $O(n)$ | $O(1)$ |
| Collect to list   | $O(n)$ | $O(n)$ |

## Reference

- [Introduction to Algorithms (CLRS), Chapter 10](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
