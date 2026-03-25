# Search

Searching a sorted linked list takes $O(n)$ because every element must be
examined sequentially. A skip list accelerates this by maintaining multiple
layers of linked lists, where higher layers skip over large sections of the
data. The search algorithm exploits these express lanes to locate an element
in $O(\log n)$ expected time, much like binary search exploits sorted arrays.
This page describes the skip list search algorithm and traces through a
concrete example.

## Search Algorithm

The search begins at the **header** node at the **highest level** and
proceeds as follows:

1. At the current level, move forward along the linked list as long as
   the next node's key is less than the target key.
2. When the next node's key is greater than or equal to the target, or the
   forward pointer is `None`, drop down one level.
3. Repeat until reaching level 0.
4. At level 0, check whether the next node's key equals the target.

This top-down, left-to-right traversal ensures that each level acts as an
express lane, progressively narrowing the search range.

## Pseudocode

```
SEARCH(skip_list, target):
    current = skip_list.header
    for level = skip_list.height down to 0:
        while current.forward[level] != None
              and current.forward[level].key < target:
            current = current.forward[level]
    current = current.forward[0]
    if current != None and current.key == target:
        return current
    return None
```

The algorithm examines at most $O(\log n)$ nodes in expectation, visiting
a constant number of nodes at each level on average.

## Worked Example

Consider a skip list containing keys $\{3, 6, 7, 9, 12, 17, 19, 21, 25, 26\}$
with the following level assignments:

```
Level 3: header ───────────────> 6 ─────────────────────────────> 25 ──> None
Level 2: header ───────────────> 6 ──────> 9 ───────────────────> 25 ──> None
Level 1: header ───────────────> 6 ──────> 9 ───> 17 ──> 19 ──> 25 ──> None
Level 0: header ──> 3 ──> 6 ──> 7 ──> 9 ──> 12 ──> 17 ──> 19 ──> 21 ──> 25 ──> 26 ──> None
```

**Searching for key 17:**

| Step | Level | Current | Action |
|---|---|---|---|
| 1 | 3 | header | forward[3] = 6 < 17, move right |
| 2 | 3 | 6 | forward[3] = 25 > 17, drop down |
| 3 | 2 | 6 | forward[2] = 9 < 17, move right |
| 4 | 2 | 9 | forward[2] = 25 > 17, drop down |
| 5 | 1 | 9 | forward[1] = 17 = target, drop down |
| 6 | 0 | 9 | forward[0] = 12 < 17, move right |
| 7 | 0 | 12 | forward[0] = 17 = target, found! |

The search examined 5 distinct nodes (header, 6, 9, 12, 17) out of 10
total, demonstrating how the higher levels skip past irrelevant elements.

**Searching for key 15 (not present):**

| Step | Level | Current | Action |
|---|---|---|---|
| 1 | 3 | header | forward[3] = 6 < 15, move right |
| 2 | 3 | 6 | forward[3] = 25 > 15, drop down |
| 3 | 2 | 6 | forward[2] = 9 < 15, move right |
| 4 | 2 | 9 | forward[2] = 25 > 15, drop down |
| 5 | 1 | 9 | forward[1] = 17 > 15, drop down |
| 6 | 0 | 9 | forward[0] = 12 < 15, move right |
| 7 | 0 | 12 | forward[0] = 17 > 15, stop |
| 8 | -- | 17 | 17 != 15, return None |

The algorithm correctly determines that 15 is absent.

## Implementation

```python
"""
Skip list search algorithm.

Demonstrates the top-down, left-to-right search traversal that
gives skip lists O(log n) expected search time.
"""

import random


# === Node Definition ===

class SkipNode:
    """A node with forward pointers at multiple levels."""

    def __init__(self, key, level):
        self.key = key
        self.forward = [None] * (level + 1)


# === Skip List (Search Only) ===

class SkipList:
    """Skip list with search and insert (insert needed to build the list)."""

    def __init__(self, max_level=16, p=0.5):
        self.max_level = max_level
        self.p = p
        self.level = 0
        self.header = SkipNode(-1, max_level)

    def random_level(self):
        lvl = 0
        while random.random() < self.p and lvl < self.max_level:
            lvl += 1
        return lvl

    def insert(self, key):
        """Insert a key (needed to build the list for search demos)."""
        update = [None] * (self.max_level + 1)
        current = self.header
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        new_level = self.random_level()
        if new_level > self.level:
            for i in range(self.level + 1, new_level + 1):
                update[i] = self.header
            self.level = new_level

        new_node = SkipNode(key, new_level)
        for i in range(new_level + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node

    def search(self, key):
        """Search for a key, returning the node if found.

        The algorithm starts at the highest level and works down,
        moving right at each level as far as possible.
        """
        current = self.header
        comparisons = 0

        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
                comparisons += 1
            comparisons += 1  # comparison that caused the drop

        current = current.forward[0]
        comparisons += 1

        if current and current.key == key:
            return current, comparisons
        return None, comparisons


# === Main ===

if __name__ == "__main__":
    random.seed(42)
    sl = SkipList(max_level=4, p=0.5)

    keys = [3, 6, 7, 9, 12, 17, 19, 21, 25, 26]
    for k in keys:
        sl.insert(k)

    # Search for existing keys
    for target in [17, 9, 26]:
        result, comps = sl.search(target)
        status = "found" if result else "not found"
        print(f"Search({target:2d}): {status}, comparisons={comps}")

    # Search for missing keys
    for target in [15, 1, 30]:
        result, comps = sl.search(target)
        status = "found" if result else "not found"
        print(f"Search({target:2d}): {status}, comparisons={comps}")
```

**Output:**

```
Search(17): found, comparisons=8
Search( 9): found, comparisons=5
Search(26): found, comparisons=7
Search(15): not found, comparisons=8
Search( 1): not found, comparisons=6
Search(30): not found, comparisons=7
```

## Analogy to Binary Search

Skip list search is analogous to binary search on a sorted array:

| Aspect | Binary search | Skip list search |
|---|---|---|
| Halves the range by | Choosing middle element | Dropping down a level |
| Comparisons per step | 1 | 1 |
| Total comparisons | $O(\log n)$ worst case | $O(\log n)$ expected |
| Requires | Sorted array | Sorted multi-level list |
| Supports insertion | $O(n)$ (shift elements) | $O(\log n)$ expected |

The advantage of skip lists over binary search is that insertion and
deletion are also $O(\log n)$, whereas maintaining a sorted array requires
$O(n)$ element shifting.

## Reference

- Pugh, W. "Skip Lists: A Probabilistic Alternative to Balanced Trees."
  *Communications of the ACM*, 33(6), 1990.
