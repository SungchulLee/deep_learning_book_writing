# Skip List Structure

Searching a sorted linked list requires scanning every element, making
search an $O(n)$ operation. Balanced binary search trees (AVL trees,
red-black trees) achieve $O(\log n)$ search, but their rebalancing logic
is complex. A **skip list**, introduced by William Pugh in 1990, offers an
elegant alternative: it achieves $O(\log n)$ expected time for search,
insertion, and deletion using randomization rather than rigid structural
invariants. This page describes the structure of a skip list and explains
the intuition behind its design.

## Intuition: Express Lanes

Consider a sorted linked list of $n$ elements. Searching takes $O(n)$
because every node must be examined. Now imagine adding a second "express"
linked list that contains only every other element. A search can first scan
the express list (examining $n/2$ nodes) to get close to the target, then
drop down to the full list for the final steps.

Adding a third list containing every fourth element reduces the search
further. Continuing this pattern with $\log_2 n$ levels, where level $i$
contains every $2^i$-th element, yields an idealized structure where
search visits $O(\log n)$ nodes -- one or two per level.

A skip list achieves this layered structure **probabilistically**: instead
of deterministically selecting every $2^i$-th element for level $i$,
each node is randomly promoted to higher levels with probability $p$
(typically $p = 1/2$). In expectation, this produces the same geometric
spacing as the ideal structure.

## Components

A skip list consists of:

- **Levels**: Numbered from 0 (the base level containing all elements) upward.
  Higher levels contain progressively fewer elements.
- **Nodes**: Each node stores a key and an array of forward pointers, one
  per level the node participates in.
- **Header**: A sentinel node that participates in all levels and serves as
  the starting point for every search.
- **Forward pointers**: `node.forward[i]` points to the next node at level
  $i$ whose key is greater than `node.key`.

## Visual Representation

A skip list with keys $\{3, 6, 7, 9, 12, 19, 17, 21, 25, 26\}$ might look
like:

```
Level 3: header ──────────────────────────────────────> 19 ──────────────────> None
Level 2: header ───────────> 6 ───────────────────────> 19 ──> 25 ──────────> None
Level 1: header ───────────> 6 ──> 9 ────> 17 ────────> 19 ──> 25 ──────────> None
Level 0: header ──> 3 ──> 6 ──> 7 ──> 9 ──> 12 ──> 17 ──> 19 ──> 21 ──> 25 ──> 26 ──> None
```

Level 0 is a complete sorted linked list. Each higher level is a subset
of the level below, acting as an increasingly sparse express lane.

## Node Structure

Each node contains a key and an array of forward pointers. The length of
the array equals the node's level plus one (levels are 0-indexed).

```python
"""
Skip list node and basic structure.

Demonstrates the multi-level node structure and how levels
create express lanes over a sorted linked list.
"""

import random


# === Node Definition ===

class SkipNode:
    """A node in a skip list.

    Attributes:
        key: The value stored in this node.
        forward: List of forward pointers, one per level.
                 forward[i] points to the next node at level i.
    """

    def __init__(self, key, level):
        self.key = key
        self.forward = [None] * (level + 1)

    def __repr__(self):
        return f"SkipNode({self.key}, levels={len(self.forward)})"


# === Level Generation ===

def random_level(max_level, p=0.5):
    """Generate a random level using geometric distribution.

    Each level above 0 is added independently with probability p.
    The expected level is 1/(1-p); for p=0.5, this is 2.
    """
    level = 0
    while random.random() < p and level < max_level:
        level += 1
    return level


# === Building a Skip List ===

class SkipList:
    """A skip list maintaining sorted order with probabilistic balancing."""

    def __init__(self, max_level=16, p=0.5):
        self.max_level = max_level
        self.p = p
        self.level = 0   # current maximum level in use
        self.header = SkipNode(-1, max_level)

    def insert(self, key):
        """Insert a key into the skip list."""
        update = [None] * (self.max_level + 1)
        current = self.header
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        new_level = random_level(self.max_level, self.p)
        if new_level > self.level:
            for i in range(self.level + 1, new_level + 1):
                update[i] = self.header
            self.level = new_level

        new_node = SkipNode(key, new_level)
        for i in range(new_level + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node

    def display(self):
        """Print the skip list level by level."""
        for i in range(self.level, -1, -1):
            nodes = []
            current = self.header.forward[i]
            while current:
                nodes.append(str(current.key))
                current = current.forward[i]
            print(f"Level {i}: header -> {' -> '.join(nodes)} -> None")

    def level_counts(self):
        """Return the number of nodes at each level."""
        counts = []
        for i in range(self.level + 1):
            count = 0
            current = self.header.forward[i]
            while current:
                count += 1
                current = current.forward[i]
            counts.append(count)
        return counts


# === Main ===

if __name__ == "__main__":
    random.seed(42)
    sl = SkipList(max_level=4, p=0.5)

    keys = [3, 6, 7, 9, 12, 17, 19, 21, 25, 26]
    for k in keys:
        sl.insert(k)

    print("Skip list structure:")
    sl.display()

    counts = sl.level_counts()
    print(f"\nNodes per level: {counts}")
    print(f"Total nodes: {counts[0]}")
    print(f"Height: {sl.level}")
```

**Output:**

```
Skip list structure:
Level 4: header -> 6 -> None
Level 3: header -> 6 -> 25 -> None
Level 2: header -> 6 -> 9 -> 25 -> None
Level 1: header -> 6 -> 9 -> 17 -> 19 -> 25 -> None
Level 0: header -> 3 -> 6 -> 7 -> 9 -> 12 -> 17 -> 19 -> 21 -> 25 -> 26 -> None

Nodes per level: [10, 5, 3, 2, 1]
Total nodes: 10
Height: 4
```

## Properties

A well-constructed skip list exhibits several key properties:

1. **Level 0 is complete**: Every element appears at level 0, forming a
   standard sorted linked list.

2. **Level inclusion**: If a node appears at level $i$, it also appears at
   all levels $0, 1, \ldots, i-1$.

3. **Geometric thinning**: The expected number of nodes at level $i$ is
   $n \cdot p^i$, so each level contains roughly $p$ times as many nodes
   as the level above.

4. **Logarithmic height**: The expected height (maximum level) is
   $O(\log_{1/p} n)$. See [Expected Height](height.md) for the derivation.

5. **Linear space**: The total expected number of forward pointers across
   all nodes is $n / (1 - p) = O(n)$.

## Comparison with Ideal Skip List

| Feature | Ideal (deterministic) | Randomized |
|---|---|---|
| Level assignment | Every $2^i$-th element at level $i$ | Random with probability $p^i$ |
| Search guarantee | $O(\log n)$ worst case | $O(\log n)$ expected |
| Insertion | $O(n)$ (must restructure levels) | $O(\log n)$ expected |
| Deletion | $O(n)$ (must restructure levels) | $O(\log n)$ expected |

The ideal skip list has perfect balance but is impractical because every
insertion or deletion would require repositioning elements across levels.
Randomization sacrifices worst-case guarantees for simple, efficient updates.

## Reference

- Pugh, W. "Skip Lists: A Probabilistic Alternative to Balanced Trees."
  *Communications of the ACM*, 33(6), 1990.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C.
  *Introduction to Algorithms* (4th ed.). MIT Press.
