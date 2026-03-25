# Fibonacci Heaps

Binomial heaps achieve $O(\log n)$ for all priority queue operations, but decrease-key still costs $O(\log n)$, which becomes a bottleneck in graph algorithms that call decrease-key far more often than extract-min. **Fibonacci heaps** improve decrease-key to $O(1)$ amortized by adopting a lazy approach: instead of immediately restoring structural invariants, they defer cleanup to the next extract-min. This lazy design yields $O(1)$ amortized insert, merge, find-min, and decrease-key, while keeping extract-min at $O(\log n)$ amortized. These bounds make Fibonacci heaps theoretically optimal for Dijkstra's algorithm ($O(|V|\log|V| + |E|)$) and Prim's MST algorithm.

## Structure

A Fibonacci heap is a collection of heap-ordered trees (a forest), but unlike binomial heaps, it imposes no constraint on tree shapes or the number of trees of each degree. The key structural elements are:

- **Root list**: a doubly-linked circular list of tree roots.
- **Min pointer**: points to the root with the minimum key.
- **Node fields**: each node stores its key, degree (number of children), a pointer to its parent, a doubly-linked circular list of children, and a **mark** bit.

The mark bit is central to the Fibonacci heap's design. A node is marked when it loses its first child after being made a child of another node. If a marked node loses a second child, it is cut from its parent and moved to the root list. This **cascading cut** mechanism prevents any node from losing too many children, which bounds the maximum degree.

## Operations

### Insert

Insert creates a new single-node tree and adds it to the root list. The min pointer is updated if necessary. No consolidation occurs -- this is the "lazy" philosophy.

**Cost**: $O(1)$ worst-case and amortized.

### Find-Min

The min pointer provides direct access.

**Cost**: $O(1)$ worst-case and amortized.

### Merge (Union)

Merging two Fibonacci heaps concatenates their root lists (constant time with circular doubly-linked lists) and updates the min pointer.

**Cost**: $O(1)$ worst-case and amortized.

### Extract-Min

This is where deferred work is performed. The algorithm:

1. Remove the min node from the root list.
2. Add all children of the min node to the root list (and clear their parent pointers).
3. **Consolidate**: repeatedly link trees of the same degree until no two roots share the same degree. This is analogous to the merge step in binomial heaps.

Consolidation uses a degree-indexed array. For each root, if another root of the same degree exists in the array, the two are linked (the larger root becomes a child of the smaller). This continues until all degrees are distinct.

**Cost**: $O(n)$ worst-case, $O(\log n)$ amortized.

### Decrease-Key

To decrease the key of node $x$:

1. If the new key does not violate the heap order with $x$'s parent, simply update the key and possibly update the min pointer.
2. If the heap order is violated, **cut** $x$ from its parent and add $x$ to the root list.
3. **Cascading cut**: if $x$'s parent $y$ is already marked, cut $y$ from its parent as well (and unmark $y$). Repeat up the tree until reaching an unmarked node or the root. Mark the first unmarked ancestor that lost a child.

**Cost**: $O(\log n)$ worst-case (cascade length), $O(1)$ amortized.

### Delete

Delete reduces to decrease-key followed by extract-min: set the key to $-\infty$, which moves the node to the min position, then extract it.

**Cost**: $O(n)$ worst-case, $O(\log n)$ amortized.

## Cascading Cuts in Detail

The cascading cut mechanism is the defining feature that distinguishes Fibonacci heaps from binomial heaps. Consider the following scenario:

```
Before decrease-key(x):          After cuts:
      a (unmarked)                    a (now marked)
     / | \                           / \
    b  c  d                         b   d
   / \
  x   e (both unmarked)           c → root list (was marked)
                                  x → root list
```

If $x$'s key is decreased below $c$'s key (its parent), $x$ is cut. If $c$ was already marked, $c$ is also cut and added to the root list. Node $a$ becomes marked because it lost child $c$.

!!! warning "Why Cascading Cuts Are Necessary"
    Without cascading cuts, a node could lose arbitrarily many children. This would break the degree bound $D(n) = O(\log n)$, which is essential for the $O(\log n)$ extract-min. The cascading cut rule ensures each node loses at most one child before being cut itself, maintaining the Fibonacci-number-based size guarantee.

## Complexity Summary

| Operation | Worst-case | Amortized |
|-----------|:----------:|:---------:|
| Insert | $O(1)$ | $O(1)$ |
| Find-min | $O(1)$ | $O(1)$ |
| Merge | $O(1)$ | $O(1)$ |
| Extract-min | $O(n)$ | $O(\log n)$ |
| Decrease-key | $O(\log n)$ | $O(1)$ |
| Delete | $O(n)$ | $O(\log n)$ |

## Implementation

```python
"""
Fibonacci heap implementation.

Supports O(1) amortized insert, merge, find-min, and decrease-key.
Extract-min is O(log n) amortized. All operations use lazy
consolidation deferred to extract-min.
"""

import math


# === Fibonacci Heap Node ===

class FibNode:
    """A node in a Fibonacci heap.

    Attributes:
        key: the priority value
        degree: number of children
        mark: True if the node lost a child since becoming a child itself
        parent: pointer to parent node (None if in root list)
        child: pointer to one child (head of circular child list)
        left, right: siblings in the doubly-linked circular list
    """

    def __init__(self, key):
        self.key = key
        self.degree = 0
        self.mark = False
        self.parent = None
        self.child = None
        self.left = self
        self.right = self

    def __repr__(self):
        return f"FibNode(key={self.key}, deg={self.degree}, mark={self.mark})"


# === Fibonacci Heap ===

class FibonacciHeap:
    """A min-Fibonacci-heap with lazy consolidation."""

    def __init__(self):
        self.min_node = None
        self.n = 0

    def _add_to_root_list(self, node):
        """Add a node to the root list."""
        node.parent = None
        if self.min_node is None:
            node.left = node
            node.right = node
            self.min_node = node
        else:
            node.left = self.min_node
            node.right = self.min_node.right
            self.min_node.right.left = node
            self.min_node.right = node

    def _remove_from_list(self, node):
        """Remove a node from its doubly-linked circular list."""
        node.left.right = node.right
        node.right.left = node.left

    def insert(self, key):
        """Insert a key into the heap. O(1)."""
        node = FibNode(key)
        self._add_to_root_list(node)
        if node.key < self.min_node.key:
            self.min_node = node
        self.n += 1
        return node

    def find_min(self):
        """Return the minimum key. O(1)."""
        if self.min_node is None:
            raise IndexError("find_min from empty heap")
        return self.min_node.key

    def merge(self, other):
        """Merge another Fibonacci heap into this one. O(1)."""
        if other.min_node is None:
            return
        if self.min_node is None:
            self.min_node = other.min_node
            self.n = other.n
            return
        # Concatenate root lists
        self_right = self.min_node.right
        other_left = other.min_node.left
        self.min_node.right = other.min_node
        other.min_node.left = self.min_node
        self_right.left = other_left
        other_left.right = self_right
        # Update min
        if other.min_node.key < self.min_node.key:
            self.min_node = other.min_node
        self.n += other.n

    def extract_min(self):
        """Remove and return the minimum key. O(log n) amortized."""
        z = self.min_node
        if z is None:
            raise IndexError("extract_min from empty heap")

        # Add all children of z to the root list
        if z.child is not None:
            children = []
            c = z.child
            while True:
                children.append(c)
                c = c.right
                if c is z.child:
                    break
            for c in children:
                self._add_to_root_list(c)

        # Remove z from the root list
        self._remove_from_list(z)

        if z == z.right:
            # z was the only root
            self.min_node = None
        else:
            self.min_node = z.right
            self._consolidate()

        self.n -= 1
        return z.key

    def _consolidate(self):
        """Consolidate trees so no two roots have the same degree."""
        max_degree = int(math.log(self.n) / math.log(1.618)) + 2
        degree_table = [None] * (max_degree + 1)

        # Collect all roots
        roots = []
        curr = self.min_node
        while True:
            roots.append(curr)
            curr = curr.right
            if curr is self.min_node:
                break

        for w in roots:
            x = w
            d = x.degree
            while d < len(degree_table) and degree_table[d] is not None:
                y = degree_table[d]
                if x.key > y.key:
                    x, y = y, x
                self._link(y, x)
                degree_table[d] = None
                d += 1
            if d >= len(degree_table):
                degree_table.extend([None] * (d - len(degree_table) + 1))
            degree_table[d] = x

        # Rebuild root list from degree_table
        self.min_node = None
        for node in degree_table:
            if node is not None:
                node.left = node
                node.right = node
                self._add_to_root_list(node)
                if node.key < self.min_node.key:
                    self.min_node = node

    def _link(self, child, parent):
        """Make child a child of parent."""
        self._remove_from_list(child)
        child.parent = parent
        if parent.child is None:
            parent.child = child
            child.left = child
            child.right = child
        else:
            child.left = parent.child
            child.right = parent.child.right
            parent.child.right.left = child
            parent.child.right = child
        parent.degree += 1
        child.mark = False

    def decrease_key(self, node, new_key):
        """Decrease the key of a node. O(1) amortized."""
        if new_key > node.key:
            raise ValueError("New key is greater than current key")
        node.key = new_key
        parent = node.parent
        if parent is not None and node.key < parent.key:
            self._cut(node, parent)
            self._cascading_cut(parent)
        if node.key < self.min_node.key:
            self.min_node = node

    def _cut(self, child, parent):
        """Cut child from parent and add to root list."""
        if child.right == child:
            parent.child = None
        else:
            if parent.child == child:
                parent.child = child.right
            self._remove_from_list(child)
        parent.degree -= 1
        self._add_to_root_list(child)
        child.mark = False

    def _cascading_cut(self, node):
        """Perform cascading cuts up the tree."""
        parent = node.parent
        if parent is not None:
            if not node.mark:
                node.mark = True
            else:
                self._cut(node, parent)
                self._cascading_cut(parent)

    def is_empty(self):
        """Check if the heap is empty."""
        return self.min_node is None


# === Demonstration ===

if __name__ == "__main__":
    h = FibonacciHeap()
    values = [7, 3, 8, 1, 5, 2, 9, 4, 6]
    nodes = {}

    print("Inserting values:")
    for v in values:
        nodes[v] = h.insert(v)
        print(f"  Inserted {v}, min = {h.find_min()}")

    print(f"\nExtract min: {h.extract_min()}")
    print(f"New min: {h.find_min()}")

    # Decrease key demonstration
    print(f"\nDecrease key 9 -> 0:")
    h.decrease_key(nodes[9], 0)
    print(f"New min: {h.find_min()}")

    print(f"\nExtracting all:")
    extracted = []
    while not h.is_empty():
        extracted.append(h.extract_min())
    print(f"Extracted: {extracted}")
```

**Output:**
```
Inserting values:
  Inserted 7, min = 7
  Inserted 3, min = 3
  Inserted 8, min = 3
  Inserted 1, min = 1
  Inserted 5, min = 1
  Inserted 2, min = 1
  Inserted 9, min = 1
  Inserted 4, min = 1
  Inserted 6, min = 1

Extract min: 1
New min: 2

Decrease key 9 -> 0:
New min: 0

Extracting all:
Extracted: [0, 2, 3, 4, 5, 6, 7, 8]
```

## Comparison with Other Heaps

| Operation | Binary Heap | Binomial Heap | Fibonacci Heap |
|-----------|:-----------:|:------------:|:--------------:|
| Insert | $O(\log n)$ | $O(\log n)$ amort. $O(1)$ | $O(1)$ |
| Find-min | $O(1)$ | $O(\log n)$ | $O(1)$ |
| Merge | $O(n)$ | $O(\log n)$ | $O(1)$ |
| Extract-min | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ amort. |
| Decrease-key | $O(\log n)$ | $O(\log n)$ | $O(1)$ amort. |

!!! tip "When to Use Fibonacci Heaps"
    Fibonacci heaps are theoretically optimal for algorithms that perform many decrease-key operations relative to extract-min (like Dijkstra and Prim). In practice, the constant factors and pointer overhead often make simpler heaps (binary or $d$-ary) faster for moderate input sizes. Fibonacci heaps become practical advantages in very large graphs or when asymptotic optimality is required.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., and Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 19: Fibonacci Heaps. MIT Press.
- Fredman, M. L. and Tarjan, R. E. "Fibonacci heaps and their uses in improved network optimization algorithms." *Journal of the ACM*, 34(3):596--615, 1987.
