# Pairing Heaps

Fibonacci heaps achieve optimal amortized bounds but are notoriously complex to implement, with high constant factors from pointer manipulation and marking logic. **Pairing heaps** offer a dramatically simpler alternative: they achieve the same $O(1)$ amortized insert and merge, and $O(\log n)$ amortized extract-min, with a much simpler structure that performs well in practice. The decrease-key bound is $O(2^{O(\sqrt{\log \log n})})$ amortized -- not quite $O(1)$ like Fibonacci heaps, but close enough that pairing heaps are often the preferred choice in practice.

## Structure

A pairing heap is a heap-ordered multi-way tree. Each node may have arbitrarily many children. The standard representation uses the **left-child, right-sibling** encoding:

- **child**: pointer to the leftmost child
- **sibling**: pointer to the next sibling
- **prev**: pointer to the previous sibling (or parent, for the leftmost child)

The only structural invariant is the **heap-order property**: every node's key is at most the key of each of its children. There is no balance condition, no degree constraint, and no marking -- this simplicity is the main appeal of pairing heaps.

## Merge (Pairing)

Merging two pairing heaps takes constant time: compare the two roots, and make the loser a child of the winner.

```
MERGE(h1, h2):
    if h1 is None: return h2
    if h2 is None: return h1
    if h1.key <= h2.key:
        make h2 the leftmost child of h1
        return h1
    else:
        make h1 the leftmost child of h2
        return h2
```

**Cost**: $O(1)$.

## Insert

Insert creates a single-node tree and merges it with the existing heap.

**Cost**: $O(1)$ (a single merge).

## Find-Min

The root holds the minimum.

**Cost**: $O(1)$.

## Extract-Min (Delete-Min)

Extract-min removes the root, leaving a collection of subtrees (the root's children). The critical question is: how should these subtrees be recombined? The **two-pass pairing** strategy gives the best bounds:

### Two-Pass Pairing

Given children $c_1, c_2, c_3, \ldots, c_k$ (left to right):

1. **Left-to-right pass**: pair the children sequentially: merge $c_1$ with $c_2$, merge $c_3$ with $c_4$, and so on. If $k$ is odd, the last child is unpaired.
2. **Right-to-left pass**: merge the resulting trees from right to left into a single tree.

```
EXTRACT-MIN(H):
    if H is empty: error
    min_key = H.root.key
    children = list of H.root's children [c1, c2, ..., ck]

    # Left-to-right pairing pass
    paired = []
    for i in 0, 2, 4, ...:
        if i + 1 < k:
            paired.append(MERGE(children[i], children[i+1]))
        else:
            paired.append(children[i])

    # Right-to-left combining pass
    result = paired[-1]
    for i in len(paired) - 2 down to 0:
        result = MERGE(paired[i], result)

    H.root = result
    return min_key
```

**Cost**: $O(k)$ where $k$ is the number of children of the min node. Amortized $O(\log n)$.

!!! tip "Why Two Passes?"
    A single left-to-right merging (without the right-to-left pass) can create pathologically unbalanced trees. The two-pass strategy ensures that each subtree is paired with a subtree of comparable "weight," analogous to how merge sort achieves balance through recursive halving. This is what gives the $O(\log n)$ amortized bound.

## Decrease-Key

To decrease the key of node $x$:

1. If $x$ is the root, simply update its key.
2. Otherwise, cut $x$ from its parent (detach it from the sibling list) and merge the resulting subtree with the main heap.

**Cost**: $O(1)$ worst-case for the operation itself. The amortized bound is $O(2^{O(\sqrt{\log \log n})})$, which is sub-logarithmic but not constant.

## Implementation

```python
"""
Pairing heap implementation.

A pairing heap is a simple heap-ordered multi-way tree that
supports O(1) merge and insert, and O(log n) amortized
extract-min using two-pass pairing.
"""


# === Pairing Heap Node ===

class PairingNode:
    """A node in a pairing heap using left-child, right-sibling representation.

    Attributes:
        key: the priority value
        child: leftmost child
        sibling: next sibling
        prev: previous sibling or parent (for cuts)
    """

    def __init__(self, key):
        self.key = key
        self.child = None
        self.sibling = None
        self.prev = None

    def __repr__(self):
        return f"PairingNode(key={self.key})"


# === Pairing Heap ===

class PairingHeap:
    """A min-pairing-heap with two-pass delete-min."""

    def __init__(self):
        self.root = None
        self.size = 0

    @staticmethod
    def _link(h1, h2):
        """Link two trees: the smaller root becomes the parent."""
        if h1 is None:
            return h2
        if h2 is None:
            return h1
        if h1.key <= h2.key:
            # h2 becomes leftmost child of h1
            h2.sibling = h1.child
            if h1.child is not None:
                h1.child.prev = h2
            h1.child = h2
            h2.prev = h1
            return h1
        else:
            # h1 becomes leftmost child of h2
            h1.sibling = h2.child
            if h2.child is not None:
                h2.child.prev = h1
            h2.child = h1
            h1.prev = h2
            return h2

    def merge(self, other):
        """Merge another pairing heap into this one. O(1)."""
        self.root = self._link(self.root, other.root)
        self.size += other.size

    def insert(self, key):
        """Insert a key. O(1)."""
        node = PairingNode(key)
        self.root = self._link(self.root, node)
        self.size += 1
        return node

    def find_min(self):
        """Return the minimum key. O(1)."""
        if self.root is None:
            raise IndexError("find_min from empty heap")
        return self.root.key

    def extract_min(self):
        """Remove and return the minimum key. O(log n) amortized."""
        if self.root is None:
            raise IndexError("extract_min from empty heap")
        min_key = self.root.key

        # Collect all children
        children = []
        child = self.root.child
        while child is not None:
            nxt = child.sibling
            child.sibling = None
            child.prev = None
            children.append(child)
            child = nxt

        # Two-pass pairing
        self.root = self._two_pass_merge(children)
        self.size -= 1
        return min_key

    @staticmethod
    def _two_pass_merge(children):
        """Merge a list of trees using two-pass pairing."""
        if not children:
            return None
        if len(children) == 1:
            return children[0]

        # Left-to-right pairing pass
        paired = []
        i = 0
        while i + 1 < len(children):
            paired.append(PairingHeap._link(children[i], children[i + 1]))
            i += 2
        if i < len(children):
            paired.append(children[i])

        # Right-to-left combining pass
        result = paired[-1]
        for j in range(len(paired) - 2, -1, -1):
            result = PairingHeap._link(paired[j], result)

        return result

    def decrease_key(self, node, new_key):
        """Decrease the key of a node. O(1) worst-case."""
        if new_key > node.key:
            raise ValueError("New key is greater than current key")
        node.key = new_key
        if node is self.root:
            return
        # Cut node from its parent
        if node.prev is not None:
            if node.prev.child is node:
                node.prev.child = node.sibling
            else:
                node.prev.sibling = node.sibling
        if node.sibling is not None:
            node.sibling.prev = node.prev
        node.prev = None
        node.sibling = None
        self.root = self._link(self.root, node)

    def is_empty(self):
        """Check if the heap is empty."""
        return self.root is None


# === Demonstration ===

if __name__ == "__main__":
    h = PairingHeap()
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
    print("Correctness verified." if extracted == sorted(extracted) else "ERROR!")
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
Correctness verified.
```

## Complexity Comparison

| Operation | Binary Heap | Fibonacci Heap | Pairing Heap |
|-----------|:-----------:|:--------------:|:------------:|
| Insert | $O(\log n)$ | $O(1)$ amort. | $O(1)$ |
| Find-min | $O(1)$ | $O(1)$ | $O(1)$ |
| Merge | $O(n)$ | $O(1)$ | $O(1)$ |
| Extract-min | $O(\log n)$ | $O(\log n)$ amort. | $O(\log n)$ amort. |
| Decrease-key | $O(\log n)$ | $O(1)$ amort. | $O(2^{O(\sqrt{\log \log n})})$ amort. |

!!! tip "Practical Performance"
    Despite the theoretically weaker decrease-key bound, pairing heaps consistently outperform Fibonacci heaps in benchmarks. The simpler pointer structure means lower constant factors, better cache behavior, and far less code. For most practical applications, pairing heaps are the recommended mergeable priority queue.

## Reference

- Fredman, M. L., Sedgewick, R., Sleator, D. D., and Tarjan, R. E. "The pairing heap: a new form of self-adjusting heap." *Algorithmica*, 1(1):111--129, 1986.
- Iacono, J. "Improved upper bounds for pairing heaps." *Scandinavian Workshop on Algorithm Theory*, 2000.
