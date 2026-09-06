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

## Exercises

**Exercise 1.**
Describe the search algorithm in a skip list. How does the multi-level structure provide $O(\log n)$ expected search time?

??? success "Solution to Exercise 1"
    Search for key $k$: start at the top-left corner (highest level, head sentinel). At each level, move right along the linked list while the next node's key is less than $k$. When the next key is $\ge k$ (or the list ends), drop down one level. Repeat until reaching level 0. If the node to the right at level 0 has key $= k$, return it; otherwise, $k$ is absent. The multi-level structure acts like a binary search: at each level, roughly half the elements are skipped (since each element is promoted with probability $1/2$). The expected number of comparisons at each level is $O(1)$ (on average, 2 comparisons before dropping down). With $O(\log n)$ levels, total expected comparisons are $O(\log n)$. $\square$

---

**Exercise 2.**
Analyze the expected space usage of a skip list with $n$ elements and promotion probability $p = 1/2$. What is the expected number of total pointers across all levels?

??? success "Solution to Exercise 2"
    Each element at level 0 has one pointer. An element promoted to level $j$ has $j + 1$ pointers (one per level from 0 to $j$). The probability of being at level $j$ or higher is $(1/2)^j$. Expected total pointers for one element: $\sum_{j=0}^{\infty} (1/2)^j = 2$. Over $n$ elements: expected total pointers = $2n$. This means a skip list uses roughly twice the space of a simple linked list, and each additional level adds approximately $n/2^j$ pointers. The space is $O(n)$ in expectation with a small constant. In the worst case (all elements promoted to every level), space is $O(n \log n)$, but this event has exponentially small probability. $\square$

---

**Exercise 3.**
Compare skip lists with AVL trees and red-black trees in terms of: expected/worst-case time, space, implementation complexity, and cache performance.

??? success "Solution to Exercise 3"
    | Property | Skip List | AVL Tree | Red-Black Tree |
    |---|---|---|---|
    | Search (expected) | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ |
    | Search (worst) | $O(n)$ (rare) | $O(\log n)$ | $O(\log n)$ |
    | Insert/Delete | $O(\log n)$ exp. | $O(\log n)$ | $O(\log n)$ |
    | Space | $O(n)$ expected | $O(n)$ | $O(n)$ |
    | Implementation | Simple | Moderate | Complex |
    | Cache perf. | Poor (pointers) | Moderate | Moderate |

    Skip lists win on implementation simplicity (no rotations, no color/balance maintenance). AVL/RB trees win on worst-case guarantees and cache performance (nodes are contiguous if allocated together). Skip lists are preferred for concurrent implementations (no global rebalancing). $\square$

---

**Exercise 4.**
Prove that the expected height of a skip list with $n$ elements and promotion probability $p$ is $O(\log_{1/p} n)$.

??? success "Solution to Exercise 4"
    The height is the maximum level of any element. Element $i$ reaches level $\ge j$ with probability $p^j$. By a union bound, $P(\text{height} \ge j) \le n \cdot p^j$. Setting $n \cdot p^j \le 1$ gives $j \ge \log_{1/p} n$. More precisely, $P(\text{height} \ge c \log_{1/p} n) \le n \cdot p^{c \log_{1/p} n} = n \cdot n^{-c} = n^{1-c}$. For $c = 2$, the probability of height exceeding $2 \log_{1/p} n$ is $\le 1/n$. Therefore, the expected height is $O(\log_{1/p} n)$. For $p = 1/2$: $O(\log_2 n)$. For $p = 1/4$: $O(\log_4 n) = O(\log_2 n / 2)$, half as many levels but more traversal per level. $\square$

---

**Exercise 5.**
Design a skip list that supports an order-statistic operation: find the $k$-th smallest element in $O(\log n)$ expected time. What augmentation is needed?

??? success "Solution to Exercise 5"
    Augment each forward pointer with a **span**: the number of elements it skips over (including the destination node). At level 0, every span is 1. At higher levels, spans are the sum of the spans of the lower-level pointers they skip over. To find the $k$-th element: start at the top-left. At each level, if the span of the next pointer is $\le k$, subtract the span from $k$ and move right. Otherwise, drop down. When $k = 0$ (or we are at the target), the current node is the answer. This is $O(\log n)$ expected time (same as search). Insert/delete must update spans along the insertion/deletion path: subtract 1 from bypassed pointers above the insertion level, add 1 to the new node's pointers. This augmentation is used in Redis's sorted sets (`ZRANGEBYSCORE` and `ZRANK` commands). $\square$
