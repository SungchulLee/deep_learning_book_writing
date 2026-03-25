# Circular Doubly Linked

A standard doubly linked list has clear endpoints: the head's `prev` is
`None` and the tail's `next` is `None`. This means traversing from the tail
back to the head requires explicitly tracking both ends. A **circular doubly
linked list** removes these endpoints by connecting the tail's `next` to the
head and the head's `prev` to the tail, forming a closed ring. This design is
especially useful for applications that cycle through elements continuously,
such as round-robin schedulers, navigation in circular menus, and
fixed-size buffer management.

## Structure

In a circular doubly linked list with $n$ nodes, every node has valid `prev`
and `next` pointers -- no pointer is ever `None`. For nodes
$x_0, x_1, \ldots, x_{n-1}$:

$$
x_i.\text{next} = x_{(i+1) \bmod n} \quad \text{and} \quad x_i.\text{prev} = x_{(i-1) \bmod n}
$$

The list has no distinguished head or tail. Instead, an external reference
points to any one node, and the entire list is reachable from that node in
either direction.

## Implementation

```python
"""
Circular doubly linked list implementation.

Each node has prev and next pointers forming a closed ring.
Traversal in either direction returns to the starting node.
"""


# === Node Definition ===

class Node:
    """A node in a circular doubly linked list."""

    def __init__(self, data):
        self.data = data
        self.prev = self
        self.next = self


# === Circular Doubly Linked List ===

class CircularDLL:
    """Circular doubly linked list with no head/tail distinction."""

    def __init__(self):
        self.access = None     # reference to any node in the ring
        self.size = 0

    def is_empty(self):
        return self.size == 0

    def insert(self, data):
        """Insert a new node into the ring.

        If the ring is empty, the new node points to itself.
        Otherwise, the new node is inserted after the access node.
        """
        new_node = Node(data)
        if self.is_empty():
            self.access = new_node
        else:
            new_node.prev = self.access
            new_node.next = self.access.next
            self.access.next.prev = new_node
            self.access.next = new_node
        self.size += 1
        return new_node

    def delete(self, node):
        """Remove a node from the ring.

        If it is the only node, the ring becomes empty.
        """
        if self.size == 1:
            self.access = None
        else:
            node.prev.next = node.next
            node.next.prev = node.prev
            if self.access is node:
                self.access = node.next
        self.size -= 1
        return node.data

    def traverse(self):
        """Return all values by traversing forward from the access node."""
        if self.is_empty():
            return []
        result = []
        current = self.access
        while True:
            result.append(current.data)
            current = current.next
            if current is self.access:
                break
        return result

    def traverse_reverse(self):
        """Return all values by traversing backward from the access node."""
        if self.is_empty():
            return []
        result = []
        current = self.access
        while True:
            result.append(current.data)
            current = current.prev
            if current is self.access:
                break
        return result

    def search(self, target):
        """Search for a node with the given data value.

        Returns the node if found, None otherwise.
        """
        if self.is_empty():
            return None
        current = self.access
        while True:
            if current.data == target:
                return current
            current = current.next
            if current is self.access:
                return None


# === Main ===

if __name__ == "__main__":
    ring = CircularDLL()

    ring.insert(10)
    ring.insert(20)
    ring.insert(30)
    print("Forward: ", ring.traverse())
    print("Backward:", ring.traverse_reverse())

    # Delete node with value 20
    node = ring.search(20)
    if node:
        ring.delete(node)
    print("After deleting 20:", ring.traverse())
```

**Output:**

```
Forward:  [10, 20, 30]
Backward: [10, 30, 20]
Forward after deleting 20: [10, 30]
```

## Traversal Termination

In a non-circular list, traversal stops when the current pointer becomes
`None`. In a circular list, there is no `None` -- traversal must stop when
the cursor returns to the starting node. Forgetting this condition creates
an infinite loop, which is the most common bug with circular structures.

The standard pattern uses a `do-while` equivalent:

1. Record the starting node.
2. Move to the next (or previous) node.
3. Stop when the current node equals the starting node.

## Complexity

| Operation | Time | Notes |
|---|---|---|
| Insert after access | $O(1)$ | Four pointer updates |
| Delete given node | $O(1)$ | Four pointer updates |
| Search | $O(n)$ | Must traverse up to $n$ nodes |
| Traverse (full ring) | $O(n)$ | Visits every node once |

Space complexity is $O(n)$ for $n$ nodes, with each node storing one data
field and two pointers.

## Comparison with Non-Circular Doubly Linked

| Feature | Non-circular DLL | Circular DLL |
|---|---|---|
| Head's `prev` | `None` | Points to tail |
| Tail's `next` | `None` | Points to head |
| Boundary checks | Required | Not required |
| Natural fit for cycling | No | Yes |
| Traversal termination | `None` check | Start-node check |

The circular variant is strictly more general: a non-circular doubly linked
list is a circular doubly linked list where the ring has been "cut" by
setting two pointers to `None`.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C.
  *Introduction to Algorithms* (4th ed.), Chapter 10. MIT Press.
