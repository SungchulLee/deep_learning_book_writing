# Skip Lists

Balanced BSTs (AVL, red-black) guarantee $O(\log n)$ operations but require complex rebalancing logic. A **skip list** achieves the same expected bounds using randomization instead of structural invariants. Each element is promoted to higher "express lanes" with probability $p$ (typically $1/2$), creating a hierarchy of linked lists that enables binary-search-like traversal over a linked list.

## Structure

A skip list is a collection of sorted linked lists $L_0, L_1, \ldots, L_h$:

- $L_0$ contains all $n$ elements.
- Each element in $L_i$ is independently promoted to $L_{i+1}$ with probability $p$.
- Sentinel nodes $-\infty$ and $+\infty$ appear at every level.

The expected number of elements at level $i$ is $n p^i$, and the expected height is:

$$
E[h] = \log_{1/p} n = O(\log n)
$$

## Search

To search for key $k$, start at the topmost level and the leftmost sentinel:

1. Move right while the next element is less than $k$.
2. When the next element is $\ge k$, drop down one level.
3. Repeat until reaching level 0.
4. If the element at level 0 equals $k$, return it; otherwise $k$ is absent.

**Expected time**: At each level, we make at most $1/p$ comparisons on average before dropping down. With $O(\log n)$ levels:

$$
E[T_{\text{search}}] = O\!\left(\frac{\log n}{p}\right) = O(\log n) \text{ for constant } p
$$

## Insert

To insert key $k$:

1. Search for $k$'s position, recording the predecessors at each level.
2. Determine the new node's height by flipping coins: $\ell = 0$; while a coin flip is heads (probability $p$), increment $\ell$.
3. Insert the node into levels $0$ through $\ell$, splicing it after the recorded predecessors.

$$
E[T_{\text{insert}}] = O(\log n), \quad E[S_{\text{insert}}] = O\!\left(\frac{1}{1-p}\right) = O(1)
$$

## Delete

To delete key $k$:

1. Search for $k$, recording predecessors at each level.
2. Remove $k$ from every level where it appears by adjusting pointers.
3. If the topmost non-empty level decreased, reduce the height.

$$
E[T_{\text{delete}}] = O(\log n)
$$

## Expected Space

Each element has expected height $1/(1-p)$, so the total expected space is:

$$
E[S] = \frac{n}{1-p} = O(n) \text{ for constant } p
$$

With $p = 1/2$, the expected space is $2n$ pointers.

## Complexity Summary

| Operation | Expected Time | Worst Case |
|---|---|---|
| Search | $O(\log n)$ | $O(n)$ |
| Insert | $O(\log n)$ | $O(n)$ |
| Delete | $O(\log n)$ | $O(n)$ |
| Space | $O(n)$ | $O(n \log n)$ |

The worst case (all elements promoted to the maximum level) occurs with negligibly small probability.

## Implementation

```python
"""
Skip List -- randomized sorted data structure.

Achieves O(log n) expected time for search, insert, and delete
using probabilistic promotion instead of rebalancing.
"""

from __future__ import annotations
import random
import math


# === Skip List Node ===========================================================

class SkipNode:
    """Node with forward pointers at multiple levels."""

    def __init__(self, key: float, level: int):
        self.key = key
        self.forward: list[SkipNode | None] = [None] * (level + 1)


# === Skip List ================================================================

class SkipList:
    """Sorted probabilistic data structure with O(log n) expected operations."""

    def __init__(self, max_level: int = 16, p: float = 0.5):
        self.max_level = max_level
        self.p = p
        self.level = 0
        self.header = SkipNode(float("-inf"), max_level)
        self.size = 0

    def _random_level(self) -> int:
        """Generate a random level by coin flipping."""
        lvl = 0
        while random.random() < self.p and lvl < self.max_level:
            lvl += 1
        return lvl

    def search(self, key: float) -> bool:
        """Return True if *key* is in the skip list."""
        current = self.header
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
        current = current.forward[0]
        return current is not None and current.key == key

    def insert(self, key: float) -> None:
        """Insert *key* into the skip list."""
        update = [None] * (self.max_level + 1)
        current = self.header
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        current = current.forward[0]
        if current and current.key == key:
            return  # duplicate

        new_level = self._random_level()
        if new_level > self.level:
            for i in range(self.level + 1, new_level + 1):
                update[i] = self.header
            self.level = new_level

        new_node = SkipNode(key, new_level)
        for i in range(new_level + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node
        self.size += 1

    def to_list(self) -> list[float]:
        """Return all elements in sorted order."""
        result = []
        current = self.header.forward[0]
        while current:
            result.append(current.key)
            current = current.forward[0]
        return result


# === Main =====================================================================

if __name__ == "__main__":
    random.seed(42)
    sl = SkipList()
    for val in [3, 6, 7, 9, 12, 19, 17, 26, 21, 25]:
        sl.insert(val)

    print(f"Elements: {sl.to_list()}")
    print(f"Levels used: {sl.level}")
    print(f"Search 19: {sl.search(19)}")
    print(f"Search 15: {sl.search(15)}")
```

**Output:**

```
Elements: [3, 6, 7, 9, 12, 17, 19, 21, 25, 26]
Levels used: 3
Search 19: True
Search 15: False
```

The elements are maintained in sorted order across the multi-level structure, and search correctly distinguishes present from absent keys.

## Reference

- Pugh, W. "Skip Lists: A Probabilistic Alternative to Balanced Trees." *CACM*, 1990
- [Advanced Data Structures (Brass)](https://www.cambridge.org/core/books/advanced-data-structures/D56E2269D7CEE969A3B8105D3541F601)
