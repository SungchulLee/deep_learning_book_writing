# Binomial Heaps

Binary heaps support insert and extract-min in $O(\log n)$ but merging two heaps requires $O(n)$ time because one must rebuild the heap from scratch. **Binomial heaps** solve this by representing a heap as a collection of binomial trees, enabling merge (and therefore insert) in $O(\log n)$ worst-case time and $O(1)$ amortized time for insert. This makes binomial heaps the natural choice when frequent merging is required, such as in parallel algorithms that merge priority queues from different processors, or in graph algorithms like Prim's MST and Dijkstra's shortest paths where efficient decrease-key is needed.

## Binomial Trees

A **binomial tree** $B_k$ is defined recursively:

- $B_0$ is a single node.
- $B_k$ is formed by linking two copies of $B_{k-1}$: one becomes the leftmost child of the other's root.

### Properties of Binomial Trees

A binomial tree $B_k$ has the following properties:

1. **Height**: $k$
2. **Number of nodes**: $2^k$
3. **Degree of root**: $k$
4. **Children of root**: the root has children $B_{k-1}, B_{k-2}, \ldots, B_0$ (in some order)
5. **Nodes at depth $d$**: $\binom{k}{d}$ (this is why they are called *binomial* trees)

!!! example "First Few Binomial Trees"
    ```
    B_0:  o        B_1:  o        B_2:    o          B_3:        o
                          |              / |                    / | \
                          o            o   o                 o   o   o
                                       |                   / |   |
                                       o                 o   o   o
                                                         |
                                                         o
    Nodes:  1             2              4                    8
    Height: 0             1              2                    3
    ```

## Binomial Heap Structure

A **binomial heap** is a collection (forest) of binomial trees satisfying two properties:

1. **Heap order**: each tree satisfies the min-heap (or max-heap) property -- every node's key is at most its children's keys.
2. **Uniqueness**: for each order $k$, there is at most one binomial tree $B_k$ in the collection.

The uniqueness property creates a direct analogy with binary representation. Since each $B_k$ contains exactly $2^k$ nodes, and at most one copy of each $B_k$ is present, a binomial heap with $n$ nodes contains $B_k$ if and only if bit $k$ is set in the binary representation of $n$. For example, $n = 13 = 1101_2$ contains trees $B_3, B_2, B_0$ with $8 + 4 + 1 = 13$ nodes total.

The minimum element is always the root of one of the trees in the forest. Since a heap with $n$ nodes contains at most $\lfloor \log_2 n \rfloor + 1$ trees (bounded by the number of bits in $n$), finding the minimum requires checking at most $\lfloor \log_2 n \rfloor + 1$ roots.

## Merge Operation

Merging is the central operation of binomial heaps -- all other operations reduce to it. The algorithm is analogous to binary addition: walk through tree orders from smallest to largest, combining trees of the same order just as one carries in binary arithmetic.

### Algorithm

```
BINOMIAL-HEAP-MERGE(H1, H2):
    carry = None
    result = empty heap
    for k = 0, 1, 2, ...:
        trees at order k: t1 from H1, t2 from H2, tc from carry
        count = number of non-None trees among {t1, t2, tc}

        if count == 0: continue
        if count == 1: add the single tree to result at order k
        if count == 2: link the two trees to form B_{k+1}, set as carry
        if count == 3: add one tree to result at order k,
                       link the other two as carry
    return result
```

**Linking** two trees of order $k$: compare their roots. The tree with the larger root becomes the leftmost child of the other root, producing a tree of order $k+1$. This preserves the heap-order property.

### Complexity

Merge walks through at most $O(\log n)$ orders, performing constant work at each. Therefore:

$$
T_{\text{merge}} = O(\log n)
$$

## Other Operations via Merge

The elegance of binomial heaps lies in their merge-centric design: every operation either calls merge directly or performs $O(\log n)$ work followed by a merge. This unifying pattern simplifies both the implementation and the complexity analysis.

| Operation | How It Uses Merge | Time |
|-----------|------------------|------|
| Insert | Create a single-node heap $B_0$, merge with existing heap | $O(\log n)$ worst, $O(1)$ amortized |
| Find-min | Check all tree roots | $O(\log n)$ |
| Extract-min | Remove min root, its children form a new heap, merge | $O(\log n)$ |
| Decrease-key | Sift up within the binomial tree | $O(\log n)$ |
| Delete | Decrease key to $-\infty$, then extract-min | $O(\log n)$ |

!!! tip "Amortized O(1) Insert"
    Although a single insert may cascade through $O(\log n)$ tree merges (like binary carries), a sequence of $n$ inserts into an initially empty binomial heap performs a total of at most $2n$ link operations. By an argument analogous to the binary counter analysis, the amortized cost per insert is $O(1)$.

## Implementation

```python
"""
Binomial heap implementation.

A binomial heap is a forest of binomial trees supporting
merge in O(log n) time. All operations reduce to merge.
"""


# === Binomial Tree Node ===

class BinomialNode:
    """A node in a binomial tree.

    Each node stores a key, a pointer to its leftmost child,
    and a pointer to its next sibling (for the forest linked list).
    """

    def __init__(self, key):
        self.key = key
        self.order = 0          # order of the binomial tree rooted here
        self.child = None       # leftmost child
        self.sibling = None     # next sibling in the forest

    def __repr__(self):
        return f"BinomialNode(key={self.key}, order={self.order})"


# === Binomial Heap ===

class BinomialHeap:
    """A min-binomial-heap implemented as a linked list of binomial trees."""

    def __init__(self):
        self.head = None  # linked list of tree roots, ordered by tree order

    def _link(self, t1, t2):
        """Link two trees of the same order.

        Compares roots and makes the larger-key root a child of
        the smaller-key root. After the swap (if needed), t1 is
        always the winner (smaller key) and t2 becomes its child.
        """
        if t1.key > t2.key:
            t1, t2 = t2, t1
        t2.sibling = t1.child
        t1.child = t2
        t1.order += 1
        return t1

    def merge(self, other):
        """Merge another binomial heap into this one. O(log n)."""
        # Merge the two sorted linked lists by order
        merged = self._merge_lists(self.head, other.head)

        if merged is None:
            self.head = None
            return

        # Walk through and combine trees of the same order
        prev = None
        curr = merged
        nxt = curr.sibling

        while nxt is not None:
            if curr.order != nxt.order or \
               (nxt.sibling is not None and nxt.sibling.order == curr.order):
                # Different orders, or three trees of same order: advance
                prev = curr
                curr = nxt
            else:
                # Two trees of same order: link them
                linked = self._link(curr, nxt)
                linked.sibling = nxt.sibling
                if prev is None:
                    merged = linked
                else:
                    prev.sibling = linked
                curr = linked
            nxt = curr.sibling

        self.head = merged

    def _merge_lists(self, h1, h2):
        """Merge two root lists sorted by order into one sorted list."""
        if h1 is None:
            return h2
        if h2 is None:
            return h1

        if h1.order <= h2.order:
            head = h1
            h1 = h1.sibling
        else:
            head = h2
            h2 = h2.sibling

        tail = head
        while h1 is not None and h2 is not None:
            if h1.order <= h2.order:
                tail.sibling = h1
                h1 = h1.sibling
            else:
                tail.sibling = h2
                h2 = h2.sibling
            tail = tail.sibling

        tail.sibling = h1 if h1 is not None else h2
        return head

    def insert(self, key):
        """Insert a key by creating a single-node heap and merging. O(log n)."""
        node = BinomialNode(key)
        temp = BinomialHeap()
        temp.head = node
        self.merge(temp)

    def find_min(self):
        """Return the minimum key. O(log n)."""
        if self.head is None:
            raise IndexError("find_min from empty heap")
        min_key = self.head.key
        curr = self.head.sibling
        while curr is not None:
            if curr.key < min_key:
                min_key = curr.key
            curr = curr.sibling
        return min_key

    def extract_min(self):
        """Remove and return the minimum key. O(log n)."""
        if self.head is None:
            raise IndexError("extract_min from empty heap")

        # Find minimum root and its predecessor
        min_node = self.head
        min_prev = None
        prev = None
        curr = self.head
        while curr is not None:
            if curr.key < min_node.key:
                min_node = curr
                min_prev = prev
            prev = curr
            curr = curr.sibling

        # Remove min_node from the root list
        if min_prev is None:
            self.head = min_node.sibling
        else:
            min_prev.sibling = min_node.sibling

        # Reverse the children of min_node to form a new heap
        child_heap = BinomialHeap()
        child = min_node.child
        prev_child = None
        while child is not None:
            nxt = child.sibling
            child.sibling = prev_child
            prev_child = child
            child = nxt
        child_heap.head = prev_child

        # Merge the children back
        self.merge(child_heap)
        return min_node.key

    def is_empty(self):
        """Check if the heap is empty."""
        return self.head is None

    def _collect_keys(self):
        """Collect all keys in the heap (for testing)."""
        keys = []
        self._collect_from_node(self.head, keys)
        return keys

    def _collect_from_node(self, node, keys):
        while node is not None:
            keys.append(node.key)
            self._collect_from_node(node.child, keys)
            node = node.sibling


# === Demonstration ===

if __name__ == "__main__":
    h = BinomialHeap()
    values = [7, 3, 8, 1, 5, 2, 9, 4, 6]

    print("Inserting values:")
    for v in values:
        h.insert(v)
        print(f"  Inserted {v}, min = {h.find_min()}")

    print(f"\nExtracting in order:")
    extracted = []
    while not h.is_empty():
        val = h.extract_min()
        extracted.append(val)
        print(f"  Extracted {val}")

    print(f"\nExtracted sequence: {extracted}")
    assert extracted == sorted(values), "Extraction order incorrect!"
    print("Correctness verified.")

    # Demonstrate merge
    print("\n--- Merge Demo ---")
    h1 = BinomialHeap()
    for v in [5, 3, 7]:
        h1.insert(v)

    h2 = BinomialHeap()
    for v in [2, 8, 1]:
        h2.insert(v)

    print(f"H1 min: {h1.find_min()}, H2 min: {h2.find_min()}")
    h1.merge(h2)
    print(f"After merge, min: {h1.find_min()}")

    merged = []
    while not h1.is_empty():
        merged.append(h1.extract_min())
    print(f"Merged extraction: {merged}")
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

Extracting in order:
  Extracted 1
  Extracted 2
  Extracted 3
  Extracted 4
  Extracted 5
  Extracted 6
  Extracted 7
  Extracted 8
  Extracted 9

Extracted sequence: [1, 2, 3, 4, 5, 6, 7, 8, 9]
Correctness verified.

--- Merge Demo ---
H1 min: 3, H2 min: 1
After merge, min: 1
Merged extraction: [1, 2, 3, 5, 7, 8]
```

## Complexity Summary

| Operation | Binary Heap | Binomial Heap (worst-case) | Binomial Heap (amortized) |
|-----------|:-----------:|:--------------------------:|:-------------------------:|
| Insert | $O(\log n)$ | $O(\log n)$ | $O(1)$ |
| Find-min | $O(1)$ | $O(\log n)$ | $O(\log n)$ |
| Extract-min | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ |
| Merge | $O(n)$ | $O(\log n)$ | $O(\log n)$ |
| Decrease-key | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ |

The key advantage of binomial heaps over binary heaps is the $O(\log n)$ merge. The amortized $O(1)$ insert follows from the binary counter argument: just as incrementing a binary counter flips $O(1)$ bits amortized, inserting into a binomial heap links $O(1)$ trees amortized. The cost relative to binary heaps is a slightly more complex implementation and $O(\log n)$ find-min instead of $O(1)$.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., and Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 19: Binomial Heaps. MIT Press.
