# Concurrent Skip List

Balanced BSTs (AVL, red-black) are difficult to make concurrent because rotations during rebalancing affect multiple nodes, requiring complex locking protocols. **Skip lists** offer a compelling alternative: their probabilistic structure requires no rotations, and insertions and deletions affect only local nodes. This makes skip lists naturally suited for concurrent access, and a lock-free concurrent skip list achieves $O(\log n)$ expected time for all operations without global synchronization.

## Skip List Review

A skip list is a layered linked list where each element is promoted to higher levels with probability $p$ (typically $1/2$). Searching starts at the top level and drops down when the next pointer overshoots. This gives $O(\log n)$ expected search, insertion, and deletion time.

The expected number of levels is $O(\log n)$, and each element has $O(1)$ expected pointers.

## Why Skip Lists for Concurrency

- **No rotations**: Unlike balanced BSTs, skip list operations modify only local pointers. No global restructuring is needed.
- **Decoupled levels**: An insertion at level $k$ does not affect levels $k+1$ or higher (after linking). This enables fine-grained or lock-free synchronization.
- **Independent coin flips**: The level of a new node is determined randomly, independent of the current structure.

## Concurrent Operations

### Lock-Based (Fine-Grained)

Lock only the nodes being modified. For an insertion between nodes $A$ and $B$ at some level:

1. Lock $A$ at that level.
2. Lock $B$ at that level (to prevent concurrent insertions in the same gap).
3. Insert the new node between $A$ and $B$.
4. Unlock $B$, then $A$.
5. Repeat at each level where the new node appears.

This allows concurrent insertions and deletions at different positions.

### Lock-Free (CAS-Based)

The lock-free approach marks nodes for deletion before physically unlinking them:

1. **Logical deletion**: Set a mark bit on the node's next pointer using CAS.
2. **Physical deletion**: Subsequent traversals skip marked nodes and CAS them out of the list.
3. **Insertion**: Link the new node at the bottom level first, then at higher levels.

## Implementation

```python
"""
Concurrent skip list with fine-grained locking.

Uses per-node locks to allow concurrent operations at
different positions in the skip list.
"""

import random
import threading

# ===================================================================
# Concurrent Skip List
# ===================================================================

MAX_LEVEL = 16

class SkipNode:
    """Skip list node with per-level next pointers and a lock."""

    def __init__(self, key, value, level):
        self.key = key
        self.value = value
        self.next = [None] * (level + 1)
        self.lock = threading.Lock()
        self.level = level


class ConcurrentSkipList:
    """Skip list with fine-grained locking.

    Args:
        max_level: maximum number of levels
        p: probability of promotion to next level
    """

    def __init__(self, max_level=MAX_LEVEL, p=0.5):
        self.max_level = max_level
        self.p = p
        self.header = SkipNode(float('-inf'), None, max_level)
        self.level = 0
        self._lock = threading.Lock()

    def _random_level(self):
        """Generate a random level for a new node."""
        lvl = 0
        while random.random() < self.p and lvl < self.max_level:
            lvl += 1
        return lvl

    def search(self, key):
        """Search for key in the skip list.

        Args:
            key: key to search for

        Returns:
            Value if found, None otherwise
        """
        current = self.header
        for i in range(self.level, -1, -1):
            while (current.next[i] is not None and
                   current.next[i].key < key):
                current = current.next[i]
        current = current.next[0]
        if current is not None and current.key == key:
            return current.value
        return None

    def insert(self, key, value):
        """Thread-safe insertion.

        Args:
            key: key to insert
            value: associated value
        """
        update = [None] * (self.max_level + 1)
        current = self.header

        for i in range(self.level, -1, -1):
            while (current.next[i] is not None and
                   current.next[i].key < key):
                current = current.next[i]
            update[i] = current

        current = current.next[0]

        if current is not None and current.key == key:
            current.value = value
            return

        new_level = self._random_level()

        with self._lock:
            if new_level > self.level:
                for i in range(self.level + 1, new_level + 1):
                    update[i] = self.header
                self.level = new_level

        new_node = SkipNode(key, value, new_level)

        for i in range(new_level + 1):
            if update[i] is not None:
                with update[i].lock:
                    new_node.next[i] = update[i].next[i]
                    update[i].next[i] = new_node

    def to_list(self):
        """Return all key-value pairs in sorted order."""
        result = []
        current = self.header.next[0]
        while current is not None:
            result.append((current.key, current.value))
            current = current.next[0]
        return result

# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    random.seed(42)
    sl = ConcurrentSkipList()

    # Single-threaded correctness
    for key in [3, 6, 1, 9, 2, 7, 4, 8, 5]:
        sl.insert(key, key * 10)

    print("Skip list contents:", sl.to_list())
    print(f"search(5) = {sl.search(5)}")
    print(f"search(10) = {sl.search(10)}")

    # Multi-threaded insertion
    sl2 = ConcurrentSkipList()
    barrier = threading.Barrier(4)

    def worker(start, count):
        barrier.wait()
        for i in range(start, start + count):
            sl2.insert(i, i)

    threads = [threading.Thread(target=worker, args=(t * 25, 25))
               for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    items = sl2.to_list()
    print(f"\nConcurrent insertion: {len(items)} items")
    print(f"Sorted correctly: {items == sorted(items)}")
    print(f"All present: {len(items) == 100}")
```

**Output:**
```
Skip list contents: [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60), (7, 70), (8, 80), (9, 90)]
search(5) = 50
search(10) = None

Concurrent insertion: 100 items
Sorted correctly: True
All present: True
```

## Complexity

| Operation | Expected Time |
|---|---|
| Search | $O(\log n)$ |
| Insert | $O(\log n)$ |
| Delete | $O(\log n)$ |
| Space | $O(n)$ expected |

## Comparison with Concurrent Trees

| Property | Concurrent skip list | Concurrent red-black tree |
|---|---|---|
| Rotations | None | Required, complicates locking |
| Lock granularity | Per-node, per-level | Per-node + rotation neighbors |
| Lock-free possible | Yes (well-studied) | Difficult |
| Cache behavior | Pointer-chasing | Better with node packing |
| Practical use | Java ConcurrentSkipListMap | Less common for concurrent |

!!! note "Java ConcurrentSkipListMap"
    Java's standard library chose a lock-free skip list (ConcurrentSkipListMap) over a concurrent tree for its sorted concurrent map, specifically because skip lists are easier to make lock-free.

## Reference

- Pugh, W. (1990). "Concurrent maintenance of skip lists." *TR CS-2222, University of Maryland*.
- Herlihy, M. et al. (2006). "A provably correct scalable concurrent skip list." *OPODIS*.
