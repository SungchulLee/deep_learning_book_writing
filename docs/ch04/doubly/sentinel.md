# Sentinel Nodes

Every doubly linked list operation -- insert at head, delete the tail, remove
an arbitrary node -- must handle boundary cases: "Is the list empty?", "Is
this the head?", "Is this the tail?". These checks clutter the code and
create opportunities for bugs. A **sentinel node** (also called a dummy node)
is a special node that sits at the boundary of the list, carrying no real
data, whose sole purpose is to guarantee that every real node always has both
a predecessor and a successor. This section shows how sentinels simplify
doubly linked list code by eliminating every `None` check.

## The Problem Without Sentinels

Consider inserting a new node after a given node `x` in a standard doubly
linked list. Without sentinels, the code must handle two special cases:

```python
# Without sentinels -- boundary checks required
def insert_after(x, new_node):
    new_node.prev = x
    new_node.next = x.next
    if x.next is not None:        # special case: x is the tail
        x.next.prev = new_node
    x.next = new_node
```

Similarly, deletion requires checking whether the deleted node is the head or
the tail. Every operation carries this overhead.

## Sentinel Design

A sentinel-based doubly linked list uses a single sentinel node `s` that
represents both the beginning and the end of the list. The sentinel is
connected in a circular fashion:

- `s.next` points to the first real node (or back to `s` if the list is
  empty).
- `s.prev` points to the last real node (or back to `s` if the list is
  empty).

An empty list is simply the sentinel pointing to itself in both directions:

$$
s.\text{next} = s \quad \text{and} \quad s.\text{prev} = s
$$

The sentinel is **never removed** and **never carries user data**. It exists
purely as a structural element.

## Implementation

```python
"""
Doubly linked list with a sentinel node.

The sentinel eliminates all None checks for boundary conditions,
making insertion and deletion code uniform and concise.
"""


# === Node Definition ===

class Node:
    """A node in a sentinel-based doubly linked list."""

    def __init__(self, data=None):
        self.data = data
        self.prev = None
        self.next = None


# === Sentinel-Based Doubly Linked List ===

class SentinelDLL:
    """Doubly linked list using a sentinel node."""

    def __init__(self):
        self.sentinel = Node()          # dummy node, no real data
        self.sentinel.next = self.sentinel
        self.sentinel.prev = self.sentinel

    def is_empty(self):
        return self.sentinel.next is self.sentinel

    def insert_after(self, x, data):
        """Insert a new node with 'data' immediately after node x."""
        new_node = Node(data)
        new_node.prev = x
        new_node.next = x.next
        x.next.prev = new_node         # no None check needed
        x.next = new_node
        return new_node

    def insert_front(self, data):
        """Insert at the front of the list (after the sentinel)."""
        return self.insert_after(self.sentinel, data)

    def insert_back(self, data):
        """Insert at the back of the list (before the sentinel)."""
        return self.insert_after(self.sentinel.prev, data)

    def delete(self, x):
        """Remove node x from the list. x must not be the sentinel."""
        x.prev.next = x.next           # no None check needed
        x.next.prev = x.prev
        x.prev = None
        x.next = None
        return x.data

    def to_list(self):
        """Return all data values as a Python list (forward order)."""
        result = []
        current = self.sentinel.next
        while current is not self.sentinel:
            result.append(current.data)
            current = current.next
        return result

    def to_list_reverse(self):
        """Return all data values in reverse order."""
        result = []
        current = self.sentinel.prev
        while current is not self.sentinel:
            result.append(current.data)
            current = current.prev
        return result


# === Main ===

if __name__ == "__main__":
    dll = SentinelDLL()

    # Insert elements
    dll.insert_back(10)
    dll.insert_back(20)
    dll.insert_back(30)
    print("Forward: ", dll.to_list())
    print("Backward:", dll.to_list_reverse())

    # Insert at front
    dll.insert_front(5)
    print("After insert_front(5):", dll.to_list())

    # Delete the second element (value 10)
    node_10 = dll.sentinel.next.next   # 5 -> 10 -> 20 -> 30
    dll.delete(node_10)
    print("After deleting 10:    ", dll.to_list())
```

**Output:**

```
Forward:  [10, 20, 30]
Backward: [30, 20, 10]
After insert_front(5): [5, 10, 20, 30]
After deleting 10:     [5, 20, 30]
```

## Why Sentinels Work

The sentinel guarantees a key invariant: **every real node has a non-sentinel
predecessor and successor that are valid `Node` objects**. More precisely,
for every real node $x$:

- $x.\text{prev}$ is either another real node or the sentinel.
- $x.\text{next}$ is either another real node or the sentinel.

Because the sentinel is a real `Node` object (not `None`), the
assignments `x.next.prev = ...` and `x.prev.next = ...` are always safe.
This transforms insertion and deletion from multi-branch conditional code
into a fixed sequence of four pointer updates.

## Complexity

Sentinels do not change the asymptotic complexity of any operation. They
eliminate constant-time conditional branches:

| Operation | Without sentinel | With sentinel |
|---|---|---|
| Insert after node | $O(1)$ with branch | $O(1)$ branchless |
| Delete given node | $O(1)$ with branch | $O(1)$ branchless |
| Search | $O(n)$ | $O(n)$ |
| Space overhead | 0 | 1 extra node |

The practical benefit is simpler, less error-prone code rather than faster
asymptotic performance.

!!! tip "When to use sentinels"
    Sentinels shine when insertion and deletion are the dominant operations
    and the list is frequently modified. For read-heavy workloads or very
    short lists, the extra sentinel node is unnecessary overhead. CLRS uses
    sentinels throughout its linked-list presentation as the standard
    implementation approach.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C.
  *Introduction to Algorithms* (4th ed.), Chapter 10.3. MIT Press.
