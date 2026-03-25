# Circular Singly Linked

In a standard singly linked list, the last node's `next` pointer is `None`,
marking the end of the chain. A **circular singly linked list** modifies this
design by connecting the last node's `next` pointer back to the first node,
forming a closed loop. This circular structure naturally models problems where
elements cycle repeatedly -- task schedulers, multiplayer game turns, and
streaming data buffers all benefit from the ability to traverse the list
indefinitely without hitting a boundary.

## Structure

A circular singly linked list with $n$ nodes $x_0, x_1, \ldots, x_{n-1}$
satisfies:

$$
x_i.\text{next} = x_{(i+1) \bmod n} \quad \text{for all } i
$$

In particular, the last node $x_{n-1}$ has $x_{n-1}.\text{next} = x_0$.
There is no `None` pointer in the structure.

A common design choice is to maintain a reference to the **tail** node
rather than the head, because `tail.next` gives $O(1)$ access to the head,
providing efficient access to both ends.

## Implementation

```python
"""
Circular singly linked list implementation.

The last node's next pointer connects back to the head,
forming a ring. A tail reference provides O(1) access to both ends.
"""


# === Node Definition ===

class Node:
    """A node in a circular singly linked list."""

    def __init__(self, data):
        self.data = data
        self.next = None


# === Circular Singly Linked List ===

class CircularSLL:
    """Circular singly linked list with a tail reference."""

    def __init__(self):
        self.tail = None
        self.size = 0

    def is_empty(self):
        return self.size == 0

    def insert_front(self, data):
        """Insert a new node at the front of the list.

        The new node becomes tail.next (the head).
        """
        new_node = Node(data)
        if self.is_empty():
            new_node.next = new_node   # single node points to itself
            self.tail = new_node
        else:
            new_node.next = self.tail.next   # new head points to old head
            self.tail.next = new_node        # tail points to new head
        self.size += 1

    def insert_back(self, data):
        """Insert a new node at the back of the list.

        The new node becomes the new tail.
        """
        self.insert_front(data)
        self.tail = self.tail.next     # advance tail to the newly inserted node
        # insert_front placed it as the head; making it the tail rotates it

    def delete_front(self):
        """Remove and return the front (head) element."""
        if self.is_empty():
            raise IndexError("delete from empty list")
        head = self.tail.next
        if self.size == 1:
            self.tail = None
        else:
            self.tail.next = head.next
        self.size -= 1
        return head.data

    def rotate(self):
        """Advance the tail reference by one position.

        This effectively moves the head to the back, cycling the list.
        """
        if not self.is_empty():
            self.tail = self.tail.next

    def traverse(self):
        """Return all values in order starting from the head."""
        if self.is_empty():
            return []
        result = []
        current = self.tail.next       # start at the head
        while True:
            result.append(current.data)
            current = current.next
            if current is self.tail.next:
                break
        return result

    def search(self, target):
        """Search for a node containing the target value.

        Returns the node if found, None otherwise.
        """
        if self.is_empty():
            return None
        current = self.tail.next
        while True:
            if current.data == target:
                return current
            current = current.next
            if current is self.tail.next:
                return None


# === Main ===

if __name__ == "__main__":
    cll = CircularSLL()

    # Build the list
    cll.insert_back(10)
    cll.insert_back(20)
    cll.insert_back(30)
    print("List:", cll.traverse())

    # Insert at front
    cll.insert_front(5)
    print("After insert_front(5):", cll.traverse())

    # Delete front
    removed = cll.delete_front()
    print(f"Removed {removed}:", cll.traverse())

    # Rotate
    cll.rotate()
    print("After rotate:", cll.traverse())
```

**Output:**

```
List: [10, 20, 30]
After insert_front(5): [5, 10, 20, 30]
Removed 5: [10, 20, 30]
After rotate: [20, 30, 10]
```

## Traversal Termination

The fundamental difference between traversing a circular list and a standard
list is the stopping condition. In a standard list, traversal stops when
`current is None`. In a circular list, `None` never appears, so traversal
must stop when the cursor returns to the starting node.

!!! warning "Infinite loop danger"
    Forgetting to check the return-to-start condition is the most common
    bug in circular list code. Always use a `do-while` pattern: record the
    starting node before the loop and check against it at the end of each
    iteration.

## Tail-Reference Design

Maintaining a tail reference instead of a head reference is a practical
optimization:

| Operation | Head reference | Tail reference |
|---|---|---|
| Access head | $O(1)$ | $O(1)$ via `tail.next` |
| Access tail | $O(n)$ | $O(1)$ |
| Insert at front | $O(n)$ to update last node | $O(1)$ |
| Insert at back | $O(n)$ to find last node | $O(1)$ |

With a tail reference, both front and back insertion are constant time,
making the circular singly linked list competitive with doubly linked lists
for queue-like workloads.

## Complexity

| Operation | Time |
|---|---|
| Insert front | $O(1)$ |
| Insert back | $O(1)$ |
| Delete front | $O(1)$ |
| Delete arbitrary | $O(n)$ (need predecessor) |
| Search | $O(n)$ |
| Traverse | $O(n)$ |
| Rotate | $O(1)$ |

Space complexity is $O(n)$ for $n$ nodes. Compared to a standard singly
linked list, the only structural change is one pointer assignment (last
node to first node), so there is no additional space overhead.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C.
  *Introduction to Algorithms* (4th ed.), Chapter 10. MIT Press.
- Goodrich, M. T., Tamassia, R., & Goldwasser, M. H.
  *Data Structures and Algorithms in Python*, Section 7.2. Wiley.
