# Deletion

Deleting a node from a singly linked list is conceptually simple -- update a pointer to bypass the removed node -- but the details depend on which node is being deleted. The main challenge arises from the singly linked structure: to remove a node, you need access to its **predecessor** so you can redirect the predecessor's `next` pointer. Since singly linked lists provide no backward references, finding the predecessor requires a traversal from the head. This makes deletion at the head $O(1)$ but deletion elsewhere $O(n)$ in the worst case.

## Deletion at the Head

Removing the first node is the simplest case. The head pointer is redirected to the second node, and the old head becomes unreachable (and is garbage-collected in Python).

**Algorithm:**

1. If the list is empty, raise an error.
2. Save a reference to the head's data (to return it).
3. Set `head = head.next`.

**Time complexity:** $O(1)$ -- no traversal needed.

## Deletion at the Tail

Removing the last node requires traversing the entire list to find the second-to-last node, then setting its `next` to `None`.

**Algorithm:**

1. If the list is empty, raise an error.
2. If the list has only one node, set `head = None`.
3. Otherwise, traverse to the node whose `next.next` is `None` (the second-to-last node).
4. Set that node's `next = None`.

**Time complexity:** $O(n)$ -- must traverse $n - 1$ nodes.

## Deletion by Value

To delete the first node containing a specific value, traverse the list while tracking the predecessor.

**Algorithm:**

1. If the head contains the target value, perform head deletion.
2. Otherwise, traverse from the head, maintaining a `prev` pointer one step behind `current`.
3. When `current.data == target`, set `prev.next = current.next`.
4. If no node matches, the list is unchanged.

**Time complexity:** $O(n)$ -- may need to scan the entire list.

## Deletion at Position k

To delete the node at zero-based index $k$, traverse $k$ nodes from the head.

**Algorithm:**

1. If $k = 0$, perform head deletion.
2. Otherwise, traverse $k - 1$ steps to reach the predecessor.
3. Set `predecessor.next = predecessor.next.next`.

**Time complexity:** $O(k)$ -- traverse $k$ nodes.

!!! warning "The Predecessor Problem"

    In a singly linked list, deleting a node when you only have a reference to that node (not its predecessor) is problematic. The standard workaround is to copy the data from the next node into the current node and then delete the next node. This trick fails for the tail node, which has no successor to copy from. Doubly linked lists (covered in the next section) solve this elegantly with backward pointers.

## Implementation

```python
"""Deletion operations for a singly linked list."""


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


# === Deletion Operations ===
def delete_head(head):
    """Delete the first node. Returns (new_head, deleted_value)."""
    if head is None:
        raise IndexError("Cannot delete from an empty list")
    return head.next, head.data


def delete_tail(head):
    """Delete the last node. Returns (new_head, deleted_value)."""
    if head is None:
        raise IndexError("Cannot delete from an empty list")
    if head.next is None:
        return None, head.data
    current = head
    while current.next.next is not None:
        current = current.next
    deleted_value = current.next.data
    current.next = None
    return head, deleted_value


def delete_by_value(head, target):
    """Delete the first node with data == target. Returns new_head."""
    if head is None:
        return None
    if head.data == target:
        return head.next
    current = head
    while current.next is not None:
        if current.next.data == target:
            current.next = current.next.next
            return head
        current = current.next
    return head  # target not found


def delete_at_position(head, k):
    """Delete the node at zero-based index k. Returns new_head."""
    if head is None:
        raise IndexError("Cannot delete from an empty list")
    if k == 0:
        return head.next
    current = head
    for _ in range(k - 1):
        if current.next is None:
            raise IndexError(f"Position {k} out of range")
        current = current.next
    if current.next is None:
        raise IndexError(f"Position {k} out of range")
    current.next = current.next.next
    return head


# === Demonstration ===
if __name__ == "__main__":
    # Build list: 10 -> 20 -> 30 -> 40 -> 50
    head = build_list([10, 20, 30, 40, 50])
    print(f"Original:          {to_list(head)}")

    # Delete head
    head, val = delete_head(head)
    print(f"After delete head: {to_list(head)}  (removed {val})")

    # Delete tail
    head, val = delete_tail(head)
    print(f"After delete tail: {to_list(head)}  (removed {val})")

    # Delete by value (30)
    head = delete_by_value(head, 30)
    print(f"After delete 30:   {to_list(head)}")

    # Rebuild and delete at position 2
    head = build_list([1, 2, 3, 4, 5])
    head = delete_at_position(head, 2)
    print(f"Delete at pos 2:   {to_list(head)}")
```

**Output:**
```
Original:          [10, 20, 30, 40, 50]
After delete head: [20, 30, 40, 50]  (removed 10)
After delete tail: [20, 30, 40]  (removed 50)
After delete 30:   [20, 40]
Delete at pos 2:   [1, 2, 4, 5]
```

## Complexity Summary

| Operation            | Time       | Space  |
|----------------------|------------|--------|
| Delete head          | $O(1)$     | $O(1)$ |
| Delete tail          | $O(n)$     | $O(1)$ |
| Delete by value      | $O(n)$     | $O(1)$ |
| Delete at position $k$ | $O(k)$  | $O(1)$ |

All deletion operations use $O(1)$ auxiliary space because they only modify pointers without allocating new nodes.

## Reference

- [Introduction to Algorithms (CLRS), Chapter 10](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
