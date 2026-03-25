# Caches (LRU)

A cache stores a small subset of recently accessed data to avoid expensive re-computation or slow I/O. When the cache reaches its capacity limit, it must **evict** an entry to make room. The **Least Recently Used (LRU)** policy evicts the entry that has not been accessed for the longest time, exploiting the principle of temporal locality: if an item was accessed recently, it is likely to be accessed again soon.

## The Design Problem

An LRU cache must support two operations, each in $O(1)$ time:

- **get(key)**: return the associated value and mark the entry as most recently used.
- **put(key, value)**: insert or update the entry and, if the cache exceeds capacity, evict the least recently used entry.

Neither a hash table alone nor a linked list alone can achieve $O(1)$ for both operations simultaneously. The key insight is to combine them.

## Hash Map Plus Doubly-Linked List

The LRU cache uses two data structures working together:

1. **Doubly-linked list**: maintains entries in order of recency. The head is the most recently used; the tail is the least recently used. Moving a node to the head or removing the tail takes $O(1)$ time.
2. **Hash map**: maps keys to their corresponding nodes in the linked list, enabling $O(1)$ lookup by key.

This combination achieves $O(1)$ time for all operations:

| Operation | Hash map role | Linked list role |
|---|---|---|
| get(key) | Find node in $O(1)$ | Move to head in $O(1)$ |
| put(key, value) | Insert/find node in $O(1)$ | Add to head / evict tail in $O(1)$ |
| evict | Remove from map in $O(1)$ | Remove tail in $O(1)$ |

## Space Complexity

An LRU cache of capacity $C$ stores at most $C$ entries in both the hash map and the linked list. Each entry requires a linked list node (key, value, two pointers) plus a hash map entry (key, node pointer). The total space is

$$
O(C)
$$

## Eviction Policies Comparison

| Policy | Description | Strength | Weakness |
|---|---|---|---|
| LRU | Evict least recently used | Good temporal locality | Scan-resistant |
| FIFO | Evict oldest inserted | Simple | Ignores access patterns |
| LFU | Evict least frequently used | Good for skewed access | Slow to adapt |
| Random | Evict random entry | No overhead | Unpredictable |

LRU is the most widely used policy in practice, balancing simplicity with effectiveness.

## Python Implementation with OrderedDict

Python's `collections.OrderedDict` combines a hash map with a doubly-linked list internally, making it ideal for a concise LRU cache.

```python
"""
LRU Cache implementation using OrderedDict.

Demonstrates the Least Recently Used eviction policy
with O(1) get and put operations.
"""

from collections import OrderedDict


# === LRU Cache ===

class LRUCache:
    """LRU cache with O(1) get and put operations."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        """Return value for key, or -1 if absent. Marks as recently used."""
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        """Insert or update key-value pair. Evicts LRU if at capacity."""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # remove least recent


# === Demonstration ===

if __name__ == "__main__":
    cache = LRUCache(capacity=3)

    # Fill cache
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    print(f"get('a'): {cache.get('a')}")  # access 'a', now most recent

    # Insert 'd' — evicts 'b' (least recently used)
    cache.put("d", 4)
    print(f"get('b'): {cache.get('b')}")  # -1, evicted
    print(f"get('c'): {cache.get('c')}")  # 3, still present
    print(f"get('d'): {cache.get('d')}")  # 4, present
```

**Output:**
```
get('a'): 1
get('b'): -1
get('c'): 3
get('d'): 4
```

## From-Scratch Implementation

To understand the internal mechanism, here is an LRU cache built from a raw doubly-linked list and a dictionary.

```python
"""
LRU Cache built from scratch with a doubly-linked list and hash map.

Shows the explicit data structure design that underlies
the OrderedDict-based implementation above.
"""


# === Doubly-Linked List Node ===

class DLLNode:
    """Node in a doubly-linked list."""

    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key=None, value=None):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


# === LRU Cache (from scratch) ===

class LRUCacheDLL:
    """LRU cache using explicit doubly-linked list + hash map."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}  # key -> DLLNode
        # Sentinel nodes simplify edge cases
        self.head = DLLNode()  # most recent end
        self.tail = DLLNode()  # least recent end
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        """Remove node from linked list in O(1)."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_head(self, node):
        """Add node right after head sentinel in O(1)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key):
        """Return value for key, or -1 if absent."""
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._remove(node)
        self._add_to_head(node)
        return node.value

    def put(self, key, value):
        """Insert or update key-value pair."""
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self._remove(node)
            self._add_to_head(node)
        else:
            node = DLLNode(key, value)
            self.cache[key] = node
            self._add_to_head(node)
            if len(self.cache) > self.capacity:
                lru = self.tail.prev
                self._remove(lru)
                del self.cache[lru.key]


# === Demonstration ===

if __name__ == "__main__":
    cache = LRUCacheDLL(capacity=2)
    cache.put(1, 10)
    cache.put(2, 20)
    print(f"get(1): {cache.get(1)}")
    cache.put(3, 30)  # evicts key 2
    print(f"get(2): {cache.get(2)}")
    print(f"get(3): {cache.get(3)}")
```

**Output:**
```
get(1): 10
get(2): -1
get(3): 30
```

## Reference

- [Introduction to Algorithms (CLRS), Chapter 11](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
