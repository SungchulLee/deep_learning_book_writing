# Search

Searching a linked list means finding a node that satisfies some criterion -- typically matching a target value or locating the node at a given position. Unlike arrays, where sorted data enables $O(\log n)$ binary search, linked lists provide no random access: reaching the $k$-th element requires following $k$ pointers from the head. This makes **linear search** the only general-purpose option for linked lists, with a worst-case cost of $O(n)$. Understanding this limitation is important for choosing between arrays and linked lists in practice.

## Search by Value

The most common search operation traverses the list from the head, comparing each node's data to the target value.

**Algorithm:**

1. Start at the head node.
2. If the current node's data equals the target, return the node.
3. Move to the next node.
4. If the current node is `None`, the target is not in the list.

**Time complexity:** $O(n)$ in the worst case (target is the last element or absent). $O(1)$ in the best case (target is the head).

**Average case:** assuming each element is equally likely to be searched, the expected number of comparisons is

$$
\frac{1}{n} \sum_{i=1}^{n} i = \frac{n + 1}{2} = \Theta(n)
$$

## Search by Position

To access the element at zero-based index $k$, traverse exactly $k$ nodes from the head.

**Algorithm:**

1. Start at the head with a counter at 0.
2. Advance to the next node and increment the counter.
3. When the counter reaches $k$, return the current node.
4. If a `None` is reached before $k$, the index is out of bounds.

**Time complexity:** $O(k)$, which is $O(n)$ in the worst case when $k = n - 1$.

!!! warning "No Binary Search on Linked Lists"

    Even if a linked list is sorted, binary search does not apply efficiently. Binary search requires accessing the middle element in $O(1)$ time, but finding the middle of a linked list takes $O(n/2)$ traversal. Applying binary search to a sorted linked list results in $O(n \log n)$ total time -- worse than simple linear search. If sorted searches are frequent, use an array or a balanced search tree instead.

## Finding the Middle Node

A useful variant uses the **two-pointer technique**: a slow pointer advances one step at a time while a fast pointer advances two steps. When the fast pointer reaches the end, the slow pointer is at the middle.

**Time complexity:** $O(n)$ -- the fast pointer traverses the full list.

**Space complexity:** $O(1)$ -- only two pointers.

This technique avoids the need for two passes (one to count, one to reach the middle).

## Finding the k-th Node from the End

Another two-pointer technique: advance the first pointer $k$ steps ahead, then advance both pointers together until the first reaches the end. The second pointer is now $k$ positions from the end.

**Time complexity:** $O(n)$ with a single pass.

## Implementation

```python
"""Search operations for a singly linked list."""


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


# === Search by Value ===
def search_value(head, target):
    """Return the first node with data == target, or None if not found.

    Time: O(n), Space: O(1).
    """
    current = head
    while current is not None:
        if current.data == target:
            return current
        current = current.next
    return None


# === Search by Position ===
def search_position(head, k):
    """Return the node at zero-based index k, or raise IndexError.

    Time: O(k), Space: O(1).
    """
    current = head
    for _ in range(k):
        if current is None:
            raise IndexError(f"Index {k} out of range")
        current = current.next
    if current is None:
        raise IndexError(f"Index {k} out of range")
    return current


# === Find Middle Node ===
def find_middle(head):
    """Return the middle node using the slow/fast pointer technique.

    For even-length lists, returns the second of the two middle nodes.
    Time: O(n), Space: O(1).
    """
    if head is None:
        return None
    slow = head
    fast = head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
    return slow


# === Find k-th from End ===
def kth_from_end(head, k):
    """Return the k-th node from the end (1-indexed).

    Time: O(n), Space: O(1).
    """
    first = head
    for _ in range(k):
        if first is None:
            raise IndexError(f"List has fewer than {k} nodes")
        first = first.next

    second = head
    while first is not None:
        first = first.next
        second = second.next
    return second


# === Demonstration ===
if __name__ == "__main__":
    head = build_list([10, 20, 30, 40, 50])
    print(f"List: {to_list(head)}")

    # Search by value
    result = search_value(head, 30)
    print(f"Search for 30: {result}")
    result = search_value(head, 99)
    print(f"Search for 99: {result}")

    # Search by position
    result = search_position(head, 2)
    print(f"Position 2: {result}")

    # Find middle
    result = find_middle(head)
    print(f"Middle: {result}")

    # k-th from end
    result = kth_from_end(head, 2)
    print(f"2nd from end: {result}")
```

**Output:**
```
List: [10, 20, 30, 40, 50]
Search for 30: Node(30)
Search for 99: None
Position 2: Node(30)
Middle: Node(30)
2nd from end: Node(40)
```

## Complexity Summary

| Operation             | Time       | Space  |
|-----------------------|------------|--------|
| Search by value       | $O(n)$     | $O(1)$ |
| Search by position $k$| $O(k)$    | $O(1)$ |
| Find middle           | $O(n)$     | $O(1)$ |
| k-th from end         | $O(n)$     | $O(1)$ |

All search operations use constant auxiliary space because they only maintain a fixed number of pointer variables.

## Reference

- [Introduction to Algorithms (CLRS), Chapter 10](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
