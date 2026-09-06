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

## Exercises

**Exercise 1.**
Explain why skip lists are more amenable to concurrent access than balanced BSTs. What structural property makes the difference?

??? success "Solution to Exercise 1"
    Balanced BSTs (AVL, red-black) require rotations during insertions and deletions to maintain balance. A single rotation modifies the parent, child, and grandchild pointers -- three nodes that may be spread across the tree. Concurrent access to these nodes requires locking a variable-size region of the tree, and the locking order is hard to predict (rotations may propagate upward). Skip lists avoid this entirely: their balance is probabilistic (random level assignment at insertion), requiring no structural adjustments after insertion or deletion. An insertion affects only the immediate predecessor at each level, and these predecessors can be locked independently (fine-grained locking) or updated via CAS (lock-free). The locality of modifications -- each operation touches $O(\log n)$ adjacent nodes in a predictable order -- makes skip lists naturally suited for fine-grained and lock-free concurrency. $\square$

---

**Exercise 2.**
Describe the lock-free skip list insertion algorithm. How does it handle the case where a concurrent deletion removes a predecessor node during insertion?

??? success "Solution to Exercise 2"
    Lock-free insertion: (1) search from the top level, recording the predecessor and successor at each level. (2) Allocate a new node with a randomly chosen height. (3) Starting from level 0 (bottom), CAS the predecessor's `next` pointer from the successor to the new node. If CAS fails (predecessor changed), re-search at that level and retry. (4) Repeat for each higher level. For concurrent deletion: deleted nodes are first logically marked (a flag in the `next` pointer) before being physically unlinked. During insertion, if a search encounters a marked (logically deleted) predecessor, the inserting thread helps unlink it (physically removes the marked node) and retries the search. This "helping" mechanism ensures progress and prevents inserting into a chain that includes deleted nodes. $\square$

---

**Exercise 3.**
Analyze the expected time complexity of a concurrent skip list search operation. Does contention from concurrent writes affect the asymptotic search time?

??? success "Solution to Exercise 3"
    A sequential skip list search takes $O(\log n)$ expected time: at each level, it traverses an expected $O(1)$ nodes before dropping down, and there are $O(\log n)$ levels. In a concurrent setting, search is read-only and does not modify any pointers, so it does not perform any CAS operations. Multiple searches proceed in parallel without interference. Concurrent writes (insertions/deletions) may modify the list structure during a search, but the search remains correct because: (1) atomically published new nodes are visible and safe to traverse, (2) deleted nodes are logically marked before unlinking, and a search encountering a marked node simply skips to the next. The expected number of nodes traversed per level remains $O(1)$ because writes change at most a constant number of pointers per level. Therefore, the asymptotic expected search time is $O(\log n)$, unaffected by concurrent writes. $\square$

---

**Exercise 4.**
Java's `ConcurrentSkipListMap` is used as a concurrent sorted map. Compare its performance characteristics with a `ConcurrentHashMap` for different access patterns: point lookups, range queries, and ordered iteration.

??? success "Solution to Exercise 4"
    **Point lookups**: `ConcurrentHashMap` provides $O(1)$ expected time (hash + bucket access), while `ConcurrentSkipListMap` provides $O(\log n)$. For pure point lookups, the hash map is 3--10x faster. **Range queries** (find all keys in $[a, b]$): `ConcurrentSkipListMap` supports this in $O(\log n + k)$ where $k$ is the number of keys in the range, by searching for $a$ and traversing the bottom-level linked list. `ConcurrentHashMap` has no efficient range query -- it requires scanning all buckets in $O(n)$. **Ordered iteration**: `ConcurrentSkipListMap` provides keys in sorted order by traversing the bottom-level list. `ConcurrentHashMap` provides no ordering guarantees. Recommendation: use `ConcurrentHashMap` for unordered key-value stores; use `ConcurrentSkipListMap` when sorted order, range queries, or operations like `ceilingKey`/`floorKey` are needed. $\square$

---

**Exercise 5.**
Prove that a skip list with $n$ elements and promotion probability $p = 1/2$ has expected height $O(\log n)$ and expected total space $O(n)$.

??? success "Solution to Exercise 5"
    **Height**: a node is promoted to level $k$ with probability $(1/2)^k$. The maximum level of any node is the height. The probability that at least one node reaches level $c \log_2 n$ is at most $n \cdot (1/2)^{c \log_2 n} = n \cdot n^{-c} = n^{1-c}$. For $c = 2$, this is $1/n$, so with high probability the height is at most $2 \log_2 n = O(\log n)$. **Space**: each node at level 0 is promoted to level 1 with probability $1/2$, to level 2 with probability $1/4$, etc. The expected number of pointers for one node is $\sum_{k=0}^{\infty} (1/2)^k = 2$. Over $n$ nodes, the expected total number of pointers is $2n = O(n)$. $\square$
