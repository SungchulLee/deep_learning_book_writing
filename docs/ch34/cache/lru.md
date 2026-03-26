# LRU Cache

Caches store a subset of data closer to the consumer to reduce access latency. When the cache is full and a new item must be inserted, an **eviction policy** determines which item to remove. **Least Recently Used (LRU)** evicts the item that has not been accessed for the longest time, betting that recently used items are more likely to be used again. LRU is the most widely deployed cache eviction policy, used in CPU caches, operating system page replacement, and application-level caches.

## Design

An $O(1)$ LRU cache combines two data structures:

- **Hash map**: Provides $O(1)$ key lookup.
- **Doubly-linked list**: Maintains access order, with the most recently used item at the tail and the least recently used at the head.

### Operations

**Get(key)**: Look up the key in the hash map. If found, move the node to the tail of the list (mark as most recently used) and return the value. If not found, return a miss indicator.

**Put(key, value)**: If the key exists, update its value and move it to the tail. If the key is new and the cache is full, remove the head node (least recently used), then insert the new node at the tail.

Both operations run in $O(1)$ time.

## Implementation

```python
"""
LRU (Least Recently Used) Cache.

Uses Python's OrderedDict to combine hash map and doubly-linked
list behavior, providing O(1) get and put operations.
"""

from collections import OrderedDict

# ===================================================================
# LRU Cache
# ===================================================================

class LRUCache:
    """LRU cache with O(1) get and put.

    Args:
        capacity: maximum number of items in cache
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        """Get value by key. Returns -1 on miss.

        Args:
            key: lookup key

        Returns:
            Cached value or -1 if not found
        """
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        """Insert or update key-value pair.

        If the cache is at capacity, evicts the least recently
        used item before inserting the new one.

        Args:
            key: cache key
            value: value to store
        """
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    cache = LRUCache(capacity=3)

    operations = [
        ("put", "A", 1),
        ("put", "B", 2),
        ("put", "C", 3),
        ("get", "A", None),
        ("put", "D", 4),   # evicts B (least recently used)
        ("get", "B", None),
        ("get", "C", None),
        ("put", "E", 5),   # evicts D (A, C were used more recently)
        ("get", "D", None),
        ("get", "A", None),
    ]

    print("LRU Cache (capacity=3):")
    for op in operations:
        if op[0] == "put":
            cache.put(op[1], op[2])
            print(f"  put({op[1]}, {op[2]})")
        else:
            result = cache.get(op[1])
            status = "HIT" if result != -1 else "MISS"
            print(f"  get({op[1]}) = {result} [{status}]")
```

**Output:**
```
LRU Cache (capacity=3):
  put(A, 1)
  put(B, 2)
  put(C, 3)
  get(A) = 1 [HIT]
  put(D, 4)
  get(B) = -1 [MISS]
  get(C) = 3 [HIT]
  put(E, 5)
  get(D) = -1 [MISS]
  get(A) = 1 [HIT]
```

!!! tip "Why B is evicted before C"
    After `get(A)`, the order is B, C, A (least to most recent). When D is inserted, B is at the head and gets evicted. C and A remain because they were accessed more recently than B.

## Complexity

| Operation | Time | Space |
|---|---|---|
| `get` | $O(1)$ | -- |
| `put` | $O(1)$ amortized | -- |
| Total space | -- | $O(c)$ |

## LRU from Scratch

Python's `OrderedDict` hides the underlying linked list. A manual implementation uses an explicit doubly-linked list with sentinel nodes:

- **Sentinel head/tail**: Eliminate edge cases for insertion and deletion.
- **Hash map**: Maps keys to linked list nodes for $O(1)$ access.
- **Move to end**: Detach the node from its current position, attach before the tail sentinel.
- **Evict**: Remove the node after the head sentinel.

## Weaknesses

- **Scan pollution**: A sequential scan of many distinct items flushes the entire cache, evicting useful entries.
- **No frequency awareness**: An item accessed once recently ranks above an item accessed 1000 times but not in the last second.
- **Fixed policy**: LRU cannot adapt to changing access patterns.

These weaknesses motivate alternatives like [LFU](lfu.md) (frequency-based) and [ARC](arc.md) (adaptive).

## Reference

- Tanenbaum, A. S. *Modern Operating Systems*, Chapter on Memory Management.
- LeetCode Problem 146: LRU Cache.
