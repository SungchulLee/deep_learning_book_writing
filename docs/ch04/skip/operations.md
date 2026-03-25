# Insertion and Deletion

Search in a skip list finds an element without modifying the structure.
Insertion and deletion, by contrast, must maintain the skip list invariants
while adding or removing nodes across multiple levels. Both operations
build on the search procedure: they first locate the correct position using
the same top-down traversal, then perform local pointer updates at each
affected level. This page describes both operations in detail, with a
complete implementation.

## Insertion Algorithm

Inserting a key $k$ into a skip list requires three steps:

1. **Search for position**: Traverse the skip list as in a normal search,
   but at each level record the last node visited before dropping down.
   These nodes form the **update array** -- they are the predecessors at
   each level where pointers must be redirected.

2. **Generate random level**: Determine the level of the new node by
   repeatedly flipping a biased coin with probability $p$. Start at
   level 1 and promote while the coin lands heads.

3. **Splice into each level**: For each level from 1 up to the new node's
   level, insert the new node after the corresponding update node by
   redirecting pointers.

If the new node's level exceeds the current skip list height, the header
is extended with new levels pointing directly to the new node.

## Deletion Algorithm

Deleting a key $k$ follows a similar pattern:

1. **Search for position**: Traverse the skip list, recording the update
   array as during insertion.

2. **Verify existence**: Check that the target node actually contains
   key $k$. If not, the key is not in the list.

3. **Unlink from each level**: For each level where the target node
   appears, redirect the predecessor's forward pointer to skip over
   the target node.

4. **Reduce height**: If the deletion causes the top levels to become
   empty (header points to `None`), reduce the skip list height.

## Implementation

```python
"""
Skip list insertion and deletion operations.

Both operations use the search-with-update-array pattern:
find the position, then splice or unlink at each level.
"""

import random


# === Node Definition ===

class SkipNode:
    """A node in the skip list with forward pointers at multiple levels."""

    def __init__(self, key, level):
        self.key = key
        self.forward = [None] * (level + 1)  # forward[i] = next node at level i


# === Skip List ===

class SkipList:
    """Skip list supporting search, insertion, and deletion."""

    def __init__(self, max_level=16, p=0.5):
        self.max_level = max_level
        self.p = p
        self.level = 0   # current highest level in use
        self.header = SkipNode(-1, max_level)  # sentinel header

    def random_level(self):
        """Generate a random level for a new node."""
        lvl = 0
        while random.random() < self.p and lvl < self.max_level:
            lvl += 1
        return lvl

    def search(self, key):
        """Search for a key, returning the node if found."""
        current = self.header
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
        current = current.forward[0]
        if current and current.key == key:
            return current
        return None

    def insert(self, key):
        """Insert a key into the skip list.

        Returns the newly created node.
        """
        # Step 1: Find update array (predecessors at each level)
        update = [None] * (self.max_level + 1)
        current = self.header
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        current = current.forward[0]

        # If key already exists, do not insert duplicate
        if current and current.key == key:
            return current

        # Step 2: Generate random level
        new_level = self.random_level()

        # Extend update array if new level exceeds current height
        if new_level > self.level:
            for i in range(self.level + 1, new_level + 1):
                update[i] = self.header
            self.level = new_level

        # Step 3: Create node and splice into each level
        new_node = SkipNode(key, new_level)
        for i in range(new_level + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node

        return new_node

    def delete(self, key):
        """Delete a key from the skip list.

        Returns True if the key was found and deleted, False otherwise.
        """
        # Step 1: Find update array
        update = [None] * (self.max_level + 1)
        current = self.header
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        target = current.forward[0]

        # Step 2: Verify existence
        if not target or target.key != key:
            return False

        # Step 3: Unlink from each level
        for i in range(self.level + 1):
            if update[i].forward[i] is not target:
                break
            update[i].forward[i] = target.forward[i]

        # Step 4: Reduce height if needed
        while self.level > 0 and self.header.forward[self.level] is None:
            self.level -= 1

        return True

    def to_list(self):
        """Return all keys in sorted order (level-0 traversal)."""
        result = []
        current = self.header.forward[0]
        while current:
            result.append(current.key)
            current = current.forward[0]
        return result

    def display(self):
        """Print the skip list level by level."""
        for i in range(self.level, -1, -1):
            nodes = []
            current = self.header.forward[i]
            while current:
                nodes.append(str(current.key))
                current = current.forward[i]
            print(f"Level {i}: {' -> '.join(nodes)}")


# === Main ===

if __name__ == "__main__":
    random.seed(42)
    sl = SkipList(max_level=4, p=0.5)

    # Insert elements
    for key in [3, 6, 7, 9, 12, 19, 17, 26, 21, 25]:
        sl.insert(key)

    print("After insertions:")
    sl.display()
    print("Sorted:", sl.to_list())

    # Delete elements
    sl.delete(19)
    sl.delete(3)
    print("\nAfter deleting 19 and 3:")
    sl.display()
    print("Sorted:", sl.to_list())

    # Search
    found = sl.search(12)
    print(f"\nSearch 12: {'found' if found else 'not found'}")
    found = sl.search(19)
    print(f"Search 19: {'found' if found else 'not found'}")
```

**Output:**

```
After insertions:
Level 4: 6
Level 3: 6 -> 25
Level 2: 6 -> 9 -> 25
Level 1: 6 -> 9 -> 17 -> 19 -> 25
Level 0: 3 -> 6 -> 7 -> 9 -> 12 -> 17 -> 19 -> 21 -> 25 -> 26
Sorted: [3, 6, 7, 9, 12, 17, 19, 21, 25, 26]

After deleting 19 and 3:
Level 4: 6
Level 3: 6 -> 25
Level 2: 6 -> 9 -> 25
Level 1: 6 -> 9 -> 17 -> 25
Level 0: 6 -> 7 -> 9 -> 12 -> 17 -> 21 -> 25 -> 26
Sorted: [6, 7, 9, 12, 17, 21, 25, 26]

Search 12: found
Search 19: not found
```

## The Update Array

The update array is the central bookkeeping structure for both insertion
and deletion. For each level $i$, `update[i]` stores the rightmost node
at level $i$ whose key is less than the target key. This node is the
predecessor of the insertion or deletion point at level $i$.

Building the update array costs the same as a search: $O(\log n)$ expected
time. The splice or unlink step at each level takes $O(1)$ per level, and
the number of levels for a single node is $O(\log n)$ in expectation.

## Complexity

| Operation | Expected time | Worst-case time |
|---|---|---|
| Insert | $O(\log n)$ | $O(n)$ |
| Delete | $O(\log n)$ | $O(n)$ |

Both operations are dominated by the search step. The pointer updates
themselves take $O(\ell)$ where $\ell$ is the level of the affected node,
which is $O(1)$ in expectation.

The worst case $O(n)$ occurs only if all nodes happen to be at level 1
(probability decreasing exponentially with $n$), degrading the skip list
to a plain sorted linked list.

## Reference

- Pugh, W. "Skip Lists: A Probabilistic Alternative to Balanced Trees."
  *Communications of the ACM*, 33(6), 1990.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C.
  *Introduction to Algorithms* (4th ed.). MIT Press.
