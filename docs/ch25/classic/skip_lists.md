# Randomized Skip Lists

A sorted linked list supports search in $O(n)$ time. A balanced BST
achieves $O(\log n)$ but requires complex rebalancing. **Skip lists**
provide the same $O(\log n)$ expected time for search, insertion, and
deletion, using randomization instead of deterministic rebalancing.
Each element is randomly promoted to higher levels, creating "express
lanes" that allow the search to skip over large portions of the list.

## Structure

A skip list consists of multiple levels of linked lists. Level 0 (the
bottom) contains all $n$ elements in sorted order. Each higher level
contains a random subset of the elements from the level below.

Each element is assigned a random **height**: flip a fair coin repeatedly
and count the number of heads before the first tail. An element with
height $h$ appears in levels $0, 1, \ldots, h$.

!!! note "Expected Heights"
    With a fair coin (promotion probability $p = 1/2$):

    - Expected height of each element: $1/(1-p) = 2$
    - Expected number of elements at level $i$: $n/2^i$
    - Expected maximum height: $O(\log n)$

## Search

To search for a key $x$:

1. Start at the head node of the highest level.
2. Move right along the current level until the next node has key $> x$
   (or we reach the end).
3. Drop down one level and repeat.
4. At level 0, check if the current node has key $= x$.

This is analogous to binary search: each level halves the remaining
search space in expectation.

## Expected Search Time

**Theorem.** The expected search time in a skip list with $n$ elements
is $O(\log n)$.

**Proof sketch.** Analyze the search path *backwards* from the target
to the head. At each step going back, we either go up one level (with
probability $1/2$, since the current node was promoted) or go left on the
same level (with probability $1/2$). The expected number of left moves
at level $i$ is at most $1/p = 2$. With $O(\log n)$ levels, the total
expected path length is $O(\log n)$.

More precisely, the expected number of comparisons is:

$$
E[\text{comparisons}] = \frac{\log_2 n}{p} + \frac{1}{1-p} = O(\log n)
$$

## Insertion

1. Search for the position (keeping track of update pointers at each level).
2. Generate a random height $h$ for the new element.
3. Insert the element at levels $0, 1, \ldots, h$, splicing it into each
   level's linked list.

If $h$ exceeds the current maximum level, add new levels to the skip list.

## Deletion

1. Search for the element (keeping track of predecessors at each level).
2. Remove the element from each level where it appears.
3. If the highest level becomes empty, reduce the maximum level.

## Implementation

```python
"""
Randomized skip list: a probabilistic alternative to balanced BSTs.

Supports search, insertion, and deletion in O(log n) expected time.
"""

import random


# === Skip List Node ===

class SkipNode:
    """A node in the skip list."""

    def __init__(self, key, level):
        self.key = key
        self.forward = [None] * (level + 1)


# === Skip List ===

class SkipList:
    """A randomized skip list data structure.

    Each operation (search, insert, delete) runs in O(log n) expected time.
    """

    def __init__(self, max_level=16, p=0.5):
        self.max_level = max_level
        self.p = p
        self.level = 0
        self.header = SkipNode(-float("inf"), max_level)
        self.size = 0

    def random_level(self):
        """Generate a random level using coin flips."""
        lvl = 0
        while random.random() < self.p and lvl < self.max_level:
            lvl += 1
        return lvl

    def search(self, key):
        """Search for a key in the skip list.

        Returns True if found, False otherwise.
        """
        current = self.header
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
        current = current.forward[0]
        return current is not None and current.key == key

    def insert(self, key):
        """Insert a key into the skip list."""
        update = [None] * (self.max_level + 1)
        current = self.header

        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        current = current.forward[0]

        if current is None or current.key != key:
            new_level = self.random_level()

            if new_level > self.level:
                for i in range(self.level + 1, new_level + 1):
                    update[i] = self.header
                self.level = new_level

            new_node = SkipNode(key, new_level)
            for i in range(new_level + 1):
                new_node.forward[i] = update[i].forward[i]
                update[i].forward[i] = new_node

            self.size += 1

    def delete(self, key):
        """Delete a key from the skip list.

        Returns True if the key was found and deleted.
        """
        update = [None] * (self.max_level + 1)
        current = self.header

        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        current = current.forward[0]

        if current and current.key == key:
            for i in range(self.level + 1):
                if update[i].forward[i] != current:
                    break
                update[i].forward[i] = current.forward[i]

            while self.level > 0 and self.header.forward[self.level] is None:
                self.level -= 1

            self.size -= 1
            return True
        return False

    def display(self):
        """Print all levels of the skip list."""
        for i in range(self.level, -1, -1):
            nodes = []
            node = self.header.forward[i]
            while node:
                nodes.append(str(node.key))
                node = node.forward[i]
            print(f"  Level {i}: {' -> '.join(nodes)}")


# === Main ===

if __name__ == "__main__":
    random.seed(42)
    sl = SkipList()

    # Insert elements
    for val in [3, 6, 7, 9, 12, 19, 17, 26, 21, 25]:
        sl.insert(val)

    print(f"Skip list ({sl.size} elements):")
    sl.display()

    # Search
    for key in [7, 10, 21]:
        print(f"Search {key}: {sl.search(key)}")

    # Delete
    sl.delete(19)
    print(f"\nAfter deleting 19 ({sl.size} elements):")
    sl.display()
```

**Output:**
```
Skip list (10 elements):
  Level 2: 6 -> 17
  Level 1: 6 -> 9 -> 17 -> 21 -> 25
  Level 0: 3 -> 6 -> 7 -> 9 -> 12 -> 17 -> 19 -> 21 -> 25 -> 26
Search 7: True
Search 10: False
Search 21: True

After deleting 19 (9 elements):
  Level 2: 6 -> 17
  Level 1: 6 -> 9 -> 17 -> 21 -> 25
  Level 0: 3 -> 6 -> 7 -> 9 -> 12 -> 17 -> 21 -> 25 -> 26
```

## Complexity Summary

| Operation | Expected Time | Worst Case |
|---|---|---|
| Search | $O(\log n)$ | $O(n)$ |
| Insert | $O(\log n)$ | $O(n)$ |
| Delete | $O(\log n)$ | $O(n)$ |
| Space | $O(n)$ expected | $O(n \log n)$ |

## Skip List vs Balanced BST

| Feature | Skip List | Balanced BST |
|---|---|---|
| Implementation | Simple | Complex rotations |
| Guarantees | Expected $O(\log n)$ | Worst-case $O(\log n)$ |
| Concurrency | Lock-free variants easy | Hard to parallelize |
| Cache behavior | Poor (pointer chasing) | Better with arrays |

!!! tip "When to Choose Skip Lists"
    Skip lists shine in concurrent programming — lock-free skip lists are
    much simpler than lock-free balanced trees. Redis, LevelDB, and Java's
    ConcurrentSkipListMap all use skip lists.

## Reference

- Pugh, W. "Skip Lists: A Probabilistic Alternative to Balanced Trees." *CACM*, 1990.
- Motwani, R. & Raghavan, P. *Randomized Algorithms*. Cambridge University Press.
