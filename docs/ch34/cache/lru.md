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

## Exercises

**Exercise 1.**
Design an LRU cache that supports `get(key)` and `put(key, value)` in $O(1)$ time. Describe the data structures and how they interact.

??? success "Solution to Exercise 1"
    Use a **hash map** mapping keys to nodes in a **doubly-linked list**. The list maintains access order: the most recently used item is at the head, and the least recently used is at the tail. `get(key)`: look up the node in the hash map ($O(1)$), move it to the head of the list ($O(1)$ pointer operations), return the value. `put(key, value)`: if key exists, update its value and move to head. If key does not exist, create a new node at the head, insert into the hash map. If capacity is exceeded, remove the tail node from the list and delete its entry from the hash map. All operations are $O(1)$. $\square$

---

**Exercise 2.**
Trace through an LRU cache of capacity 2 on the sequence: put(1,A), put(2,B), get(1), put(3,C). Show the cache state after each operation.

??? success "Solution to Exercise 2"
    After put(1,A): list = [1:A], map = {1}. After put(2,B): list = [2:B, 1:A], map = {1, 2}. After get(1): move 1 to head; list = [1:A, 2:B], map = {1, 2}, returns A. After put(3,C): cache full, evict tail (2:B); list = [3:C, 1:A], map = {1, 3}. Key 2 is evicted because it was the least recently used -- even though it was inserted more recently than key 1, key 1 was accessed via get after key 2's insertion. $\square$

---

**Exercise 3.**
Prove that LRU achieves the optimal hit rate for any workload where the access sequence exhibits temporal locality (the probability of accessing an item decreases monotonically with time since last access).

??? success "Solution to Exercise 3"
    Under the Independent Reference Model with a monotonically decreasing reuse probability, an item accessed $t$ steps ago has probability $p(t)$ of being accessed next, where $p$ is decreasing. A cache of size $k$ should store the $k$ items with the highest probability of near-term access. Since $p$ is decreasing in $t$, these are exactly the $k$ most recently accessed items -- which is precisely the set LRU maintains. Any other eviction policy would sometimes retain an item last accessed at time $t_1$ while evicting one last accessed at $t_2 < t_1$, accepting probability $p(t_1) \le p(t_2)$ instead of $p(t_2)$, yielding a weakly lower hit rate. Therefore LRU is optimal under this model. $\square$

---

**Exercise 4.**
Describe a workload pattern where LRU performs poorly compared to the optimal offline algorithm (Belady's). Quantify the gap in hit rates.

??? success "Solution to Exercise 4"
    Consider a cache of size $k$ and a cyclic workload accessing items $1, 2, \ldots, k+1, 1, 2, \ldots, k+1, \ldots$ LRU always evicts the item needed soonest: when accessing item $i$, item $i - 1$ (mod $k+1$) was just accessed and item $i - k$ (mod $k+1$) is evicted, but item $i + 1$ is the next request and is guaranteed to be the one just evicted. Hit rate: $0\%$ (every access is a miss). Belady's optimal algorithm, which evicts the item accessed furthest in the future, achieves hit rate $(k-1)/(k+1)$ by keeping the next $k$ items. For $k = 99$, LRU gets 0% while optimal gets $\approx 98\%$. This pathological case shows LRU's competitive ratio is $k$ against the offline optimum. $\square$

---

**Exercise 5.**
Explain how an LRU cache can be approximated without a doubly-linked list using the "clock algorithm" (also called second-chance). What is the tradeoff?

??? success "Solution to Exercise 5"
    The clock algorithm arranges cache entries in a circular buffer with a "hand" pointer. Each entry has a reference bit, set to 1 on access. On eviction: advance the hand; if the current entry's bit is 1, clear it and advance; if 0, evict that entry. This approximates LRU because frequently accessed items have their bits repeatedly set, surviving multiple passes of the hand, while inactive items are evicted. The tradeoff: the clock algorithm uses only 1 bit per entry (vs. a full linked list with pointers), making it much cheaper in memory and implementation complexity. However, it only approximates recency -- two items accessed 1 step and 100 steps ago are treated identically if both have their bit set. This coarse approximation reduces hit rates compared to true LRU, particularly on workloads with moderate temporal locality. $\square$
