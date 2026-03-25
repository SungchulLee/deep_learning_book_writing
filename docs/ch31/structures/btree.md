# B-Trees for External Memory

Binary search trees achieve $O(\log_2 N)$ height, but each level requires one disk access -- resulting in roughly 30 I/O operations for a billion elements. B-trees solve this problem by packing hundreds or thousands of keys into each node, so that one disk block holds an entire node. This increases the branching factor from 2 to $\Theta(B)$, reducing the tree height and the number of I/O operations to $O(\log_B N)$.

## B-Tree Properties

A B-tree of minimum degree $t \ge 2$ satisfies the following properties:

1. Every node stores between $t - 1$ and $2t - 1$ keys (except the root, which may have as few as 1 key).
2. Every internal node with $k$ keys has exactly $k + 1$ children.
3. All leaves are at the same depth.
4. Keys within each node are sorted in increasing order.

In the external memory context, we choose $t$ so that each node fits in one disk block. Since a node with $2t - 1$ keys and $2t$ child pointers must fit in $B$ elements:

$$
t = \Theta(B)
$$

This gives each internal node a branching factor of $\Theta(B)$, which is the key to I/O efficiency.

## Height Bound

For a B-tree storing $N$ keys with minimum degree $t$, the height $h$ satisfies:

$$
h \le \log_t \frac{N + 1}{2}
$$

Since $t = \Theta(B)$, this gives:

$$
h = O(\log_B N)
$$

At height $h$, the tree contains at least $2t^h - 1$ keys, so the height grows logarithmically with base $B$ rather than base 2.

## I/O Complexity of Operations

Each operation traverses a root-to-leaf path, performing one I/O per level:

| Operation | I/O Complexity | Description |
|---|---|---|
| Search | $O(\log_B N)$ | Follow one root-to-leaf path |
| Insert | $O(\log_B N)$ | Search + possible node splits |
| Delete | $O(\log_B N)$ | Search + possible merges/redistributions |
| Range query ($K$ results) | $O(\log_B N + K/B)$ | Search + scan consecutive leaves |

The range query bound reflects the B-tree's locality: once we find the starting leaf, consecutive keys are packed into adjacent blocks.

## Search

Searching for a key $k$ starts at the root and descends through the tree. At each node, a binary search among the $O(B)$ keys determines which child to follow. Since binary search within a node is performed entirely in memory (the whole node was loaded with one I/O), the total cost is one I/O per level:

$$
\text{Search I/O} = O(\log_B N)
$$

## Insertion with Node Splitting

To insert a key, first search for the correct leaf. If the leaf has fewer than $2t - 1$ keys, insert directly. If the leaf is full (has $2t - 1$ keys), split it into two nodes of $t - 1$ keys each and push the median key up to the parent. Splitting may cascade upward, but each split costs $O(1)$ I/O operations (read the full node, write two new nodes, update parent).

The total insertion cost is:

$$
\text{Insert I/O} = O(\log_B N)
$$

## Deletion with Rebalancing

Deletion mirrors insertion: if removing a key causes a node to have fewer than $t - 1$ keys, the tree rebalances by borrowing from a sibling or merging two nodes. Each rebalancing step costs $O(1)$ I/O operations, and at most $O(\log_B N)$ rebalancing steps occur along the path to the root.

## Example: B-Tree Node and Search

```python
"""
B-tree node structure and search for external memory.

Demonstrates how B-tree nodes pack multiple keys into a single block
and how search traverses O(log_B N) levels.
"""

import bisect
import math

# ===================================================================
# B-Tree Node
# ===================================================================

class BTreeNode:
    """A B-tree node with minimum degree t."""

    def __init__(self, t: int, leaf: bool = True):
        self.t = t
        self.leaf = leaf
        self.keys: list = []
        self.children: list[BTreeNode] = []

    def is_full(self) -> bool:
        return len(self.keys) == 2 * self.t - 1


# ===================================================================
# B-Tree
# ===================================================================

class BTree:
    """B-tree with configurable minimum degree t."""

    def __init__(self, t: int):
        self.t = t
        self.root = BTreeNode(t, leaf=True)
        self.io_count = 0  # Track I/O operations

    def search(self, node: BTreeNode, key: int) -> bool:
        """Search for key, counting I/O operations."""
        self.io_count += 1  # Reading node = 1 I/O
        i = bisect.bisect_left(node.keys, key)
        if i < len(node.keys) and node.keys[i] == key:
            return True
        if node.leaf:
            return False
        return self.search(node.children[i], key)

    def _split_child(self, parent: BTreeNode, idx: int):
        """Split a full child node."""
        t = self.t
        child = parent.children[idx]
        new_node = BTreeNode(t, leaf=child.leaf)

        # Move upper half of keys to new node
        parent.keys.insert(idx, child.keys[t - 1])
        parent.children.insert(idx + 1, new_node)
        new_node.keys = child.keys[t:]
        child.keys = child.keys[:t - 1]

        if not child.leaf:
            new_node.children = child.children[t:]
            child.children = child.children[:t]

    def insert(self, key: int):
        """Insert a key into the B-tree."""
        root = self.root
        if root.is_full():
            new_root = BTreeNode(self.t, leaf=False)
            new_root.children.append(self.root)
            self._split_child(new_root, 0)
            self.root = new_root
        self._insert_non_full(self.root, key)

    def _insert_non_full(self, node: BTreeNode, key: int):
        """Insert into a node that is guaranteed not full."""
        i = bisect.bisect_left(node.keys, key)
        if node.leaf:
            node.keys.insert(i, key)
        else:
            if node.children[i].is_full():
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            self._insert_non_full(node.children[i], key)

    def height(self) -> int:
        """Compute tree height."""
        h = 0
        node = self.root
        while not node.leaf:
            node = node.children[0]
            h += 1
        return h


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    import random

    t = 50  # Minimum degree (simulates block size B ~ 100 keys)
    N = 100_000
    tree = BTree(t)

    # Insert N keys
    keys = list(range(N))
    random.shuffle(keys)
    for k in keys:
        tree.insert(k)

    # Measure search I/O
    tree.io_count = 0
    tree.search(tree.root, N // 2)
    actual_ios = tree.io_count

    theoretical = math.ceil(math.log(N) / math.log(t))
    print(f"B-tree with t={t}, N={N:,}")
    print(f"  Height:         {tree.height()}")
    print(f"  Search I/Os:    {actual_ios}")
    print(f"  Theoretical:    O(log_{t}({N})) = {theoretical}")
```

??? example "Sample Output"

    ```
    B-tree with t=50, N=100,000
      Height:         2
      Search I/Os:    3
      Theoretical:    O(log_50(100000)) = 3
    ```

    With a branching factor of 50, a B-tree over 100,000 keys has height only 2-3, requiring just 3 I/O operations per search.

## B-Trees vs Binary Search Trees

| Property | BST | B-Tree ($t = \Theta(B)$) |
|---|---|---|
| Branching factor | 2 | $\Theta(B)$ |
| Height | $O(\log_2 N)$ | $O(\log_B N)$ |
| I/O per search | $O(\log_2 N)$ | $O(\log_B N)$ |
| Node size | 1 key | $\Theta(B)$ keys |
| Disk block utilization | Low | High (50%--100% full) |

For $B = 1000$ and $N = 10^9$, a BST requires about 30 I/O operations per search, while a B-tree requires only 3.

## Reference

- Bayer, R. & McCreight, E. "Organization and Maintenance of Large Ordered Indexes," *Acta Informatica*, 1(3), 1972.
- Cormen, T. et al. *Introduction to Algorithms*, Chapter 18 (B-Trees), MIT Press, 2022.
- Vitter, J. S. *Algorithms and Data Structures for External Memory*, 2008.
