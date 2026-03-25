# Bidirectional Links

In a singly linked list, each node stores only a reference to the next node.
This one-way design makes backward traversal impossible: to reach a node's
predecessor, the entire list must be re-traversed from the head. A **doubly
linked list** solves this limitation by adding a second pointer in each node,
one that points backward to the previous element. This section introduces the
doubly linked node structure and explains how bidirectional links enable
efficient two-way traversal.

## Node Structure

A doubly linked list node contains three fields:

- **data** -- the value stored in the node.
- **next** -- a reference to the successor node (or `None` if this is the tail).
- **prev** -- a reference to the predecessor node (or `None` if this is the head).

```python
"""
Doubly linked list node and basic bidirectional traversal.

Demonstrates the node structure with prev and next pointers,
list construction, and traversal in both directions.
"""


# === Node Definition ===

class Node:
    """A single node in a doubly linked list."""

    def __init__(self, data, prev=None, next_node=None):
        self.data = data
        self.prev = prev
        self.next = next_node


# === List Construction ===

def build_list(values):
    """Build a doubly linked list from a Python list of values.

    Returns the head node of the resulting doubly linked list.
    """
    if not values:
        return None
    head = Node(values[0])
    current = head
    for val in values[1:]:
        new_node = Node(val, prev=current)
        current.next = new_node
        current = new_node
    return head


# === Traversal ===

def traverse_forward(head):
    """Traverse the list from head to tail, collecting values."""
    result = []
    current = head
    while current:
        result.append(current.data)
        current = current.next
    return result


def traverse_backward(tail):
    """Traverse the list from tail to head, collecting values."""
    result = []
    current = tail
    while current:
        result.append(current.data)
        current = current.prev
    return result


def get_tail(head):
    """Return the tail node of the list starting at head."""
    current = head
    while current and current.next:
        current = current.next
    return current


# === Main ===

if __name__ == "__main__":
    head = build_list([10, 20, 30, 40])

    print("Forward: ", traverse_forward(head))
    print("Backward:", traverse_backward(get_tail(head)))
```

**Output:**

```
Forward:  [10, 20, 30, 40]
Backward: [40, 30, 20, 10]
```

## Pointer Relationships

Every adjacent pair of nodes satisfies two invariants:

1. If node $A$ has `A.next = B`, then node $B$ must have `B.prev = A`.
2. If node $B$ has `B.prev = A`, then node $A$ must have `A.next = B`.

These invariants mean the links are **symmetric**: following `next` and then
`prev` returns to the original node, and vice versa. Formally, for any
interior node $x$ (one that is neither head nor tail):

$$
x.\text{next}.\text{prev} = x = x.\text{prev}.\text{next}
$$

Maintaining this symmetry during insertion and deletion is the central
challenge of doubly linked list algorithms, as covered in the
[Insertion and Deletion](operations.md) page.

## Forward and Backward Traversal

Because each node carries a `prev` pointer, a doubly linked list supports
traversal in both directions without any additional data structure:

| Traversal | Start | Follow | Stop when |
|---|---|---|---|
| Forward | `head` | `current = current.next` | `current is None` |
| Backward | `tail` | `current = current.prev` | `current is None` |

Both forward and backward traversal visit every node exactly once, giving a
time complexity of $O(n)$ and a space complexity of $O(1)$ beyond the list
itself.

## Comparison with Singly Linked Nodes

| Feature | Singly linked | Doubly linked |
|---|---|---|
| Pointers per node | 1 (`next`) | 2 (`prev` + `next`) |
| Memory per node | Lower | Higher (one extra pointer) |
| Forward traversal | $O(n)$ | $O(n)$ |
| Backward traversal | $O(n^2)$ via restarts | $O(n)$ via `prev` |
| Delete given node | $O(n)$ (need predecessor) | $O(1)$ (predecessor is `node.prev`) |

The key trade-off is **memory versus flexibility**. Each doubly linked node
uses one additional pointer (typically 8 bytes on a 64-bit system), but this
extra pointer enables $O(1)$ deletion of a known node and efficient backward
traversal, operations that are costly or impossible with singly linked nodes.

## When to Use Bidirectional Links

Doubly linked lists are a natural choice when the application requires:

- **Backward iteration**: undo history, browser back-button, text editor
  cursor movement.
- **O(1) deletion of a known node**: LRU caches, where a hash map stores
  direct references to list nodes that must be removed without a linear scan.
- **Bidirectional searching**: meeting-in-the-middle algorithms on ordered
  lists.

When none of these requirements apply, a singly linked list saves memory and
is simpler to implement.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C.
  *Introduction to Algorithms* (4th ed.), Chapter 10. MIT Press.
