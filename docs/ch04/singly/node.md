# Node Structure

Arrays require a contiguous block of memory, which makes insertion and deletion in the middle expensive because elements must be shifted. A linked list removes this constraint by scattering elements across memory and connecting them with pointers. The fundamental building block of a linked list is the **node**: a small container that holds one data element and a reference (pointer) to the next node in the sequence. Understanding the node structure is essential before studying linked list operations, because every insertion, deletion, traversal, and search operates by manipulating these node objects and their pointers.

## Definition

A **singly linked list node** consists of two fields:

1. **Data** (also called the key, value, or element): the item stored at this position in the list.
2. **Next** (also called the link or pointer): a reference to the next node in the list, or `None` (null) if this is the last node.

Formally, each node $x$ has:

- $x.\text{data}$: the stored value
- $x.\text{next}$: a pointer to the successor node, or $\text{NIL}$ if $x$ is the tail

The list itself is represented by a single **head pointer** that references the first node. If the head is `None`, the list is empty.

## Memory Layout

Unlike arrays, linked list nodes are not stored contiguously in memory. Each node is allocated independently, and the `next` pointer provides the only way to reach the following node. This has two important consequences:

1. **No random access**: to reach the $k$-th element, you must follow $k$ pointers starting from the head. This takes $O(k)$ time compared to $O(1)$ for an array.
2. **Flexible insertion/deletion**: inserting or deleting a node requires only updating a few pointers, taking $O(1)$ time once the position is known. No elements need to be shifted.

??? example "Linked List vs Array in Memory"

    An array `[10, 20, 30]` occupies consecutive addresses:

    ```
    Address: 1000  1004  1008
    Value:   [10]  [20]  [30]
    ```

    A linked list storing the same values may have nodes scattered anywhere:

    ```
    Address 2048: [data=10 | next→5120]
    Address 5120: [data=20 | next→3072]
    Address 3072: [data=30 | next=None]
    head → 2048
    ```

    The values are the same, but the memory layout is completely different.

## Python Implementation

```python
"""Node structure for a singly linked list."""


# === Node Class ===
class Node:
    """A single node in a singly linked list.

    Attributes:
        data: The value stored in this node.
        next: Reference to the next node, or None if this is the tail.
    """

    def __init__(self, data, next_node=None):
        self.data = data
        self.next = next_node

    def __repr__(self):
        return f"Node({self.data})"


# === Helper: build a linked list from a Python list ===
def build_list(values):
    """Create a linked list from an iterable, returning the head node."""
    head = None
    for val in reversed(values):
        head = Node(val, head)
    return head


# === Helper: convert linked list to Python list for display ===
def to_list(head):
    """Traverse the linked list and collect values into a Python list."""
    result = []
    current = head
    while current is not None:
        result.append(current.data)
        current = current.next
    return result


# === Demonstration ===
if __name__ == "__main__":
    # Build a list: 10 -> 20 -> 30 -> 40
    head = build_list([10, 20, 30, 40])

    # Display the list
    print(f"List: {to_list(head)}")
    print(f"Head: {head}")
    print(f"Second node: {head.next}")
    print(f"Tail: {head.next.next.next}")
    print(f"After tail: {head.next.next.next.next}")
```

**Output:**
```
List: [10, 20, 30, 40]
Head: Node(10)
Second node: Node(20)
Tail: Node(40)
After tail: None
```

## Space Overhead

Each node stores both data and a pointer. For a list of $n$ elements, the total memory is

$$
n \cdot (w_{\text{data}} + w_{\text{pointer}})
$$

where $w_{\text{data}}$ is the size of the data field and $w_{\text{pointer}}$ is the size of a pointer (typically 8 bytes on a 64-bit system). In Python, the overhead is larger because each `Node` is a full object with a dictionary, type pointer, and reference count -- typically 50-100 bytes per node compared to 8 bytes per element in a NumPy array.

!!! warning "Linked Lists Are Memory-Hungry"

    The per-node overhead in a high-level language like Python can be 10-20x larger than the equivalent array storage. Linked lists should be chosen for their structural advantages (efficient insertion/deletion), not for memory efficiency.

## The Head Pointer

All linked list operations begin at the **head pointer**. The head is not a node itself but a variable that stores a reference to the first node. This distinction matters because:

- Inserting at the front of the list changes the head, so functions that modify the list must return the new head (or use a wrapper object).
- An empty list is represented by `head = None`, not by a special sentinel node (though sentinel-based designs exist, as covered in the doubly linked list section).

## Reference

- [Introduction to Algorithms (CLRS), Chapter 10](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
