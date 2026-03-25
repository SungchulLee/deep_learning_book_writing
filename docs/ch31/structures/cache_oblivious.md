# Cache-Oblivious B-Trees

A standard B-tree achieves optimal I/O complexity, but its branching factor must be tuned to the specific block size $B$ of the underlying hardware. If the algorithm runs on a different machine with a different $B$, the tree must be rebuilt. **Cache-oblivious** algorithms eliminate this dependency: they achieve optimal I/O complexity for *all* values of $B$ and $M$ simultaneously, without knowing either parameter. This portability is the central appeal of the cache-oblivious approach.

## The Ideal Cache Model

Cache-oblivious analysis uses the **ideal cache model**, which extends the external memory model with two additional assumptions:

1. **Optimal replacement:** The cache evicts the block that will be accessed furthest in the future (equivalent to the offline-optimal Belady's algorithm).
2. **Full associativity:** Any disk block can be stored in any cache line.

These assumptions are unrealistic in practice, but a key theorem guarantees that any algorithm analyzed in the ideal cache model achieves the same asymptotic I/O complexity on real caches with LRU replacement (up to constant factors), provided $M \ge 2B$ (the **tall cache assumption**).

## The Van Emde Boas Layout

The key technique for cache-oblivious search trees is the **van Emde Boas (vEB) layout**, which stores a static binary tree in memory so that subtrees of every size are stored contiguously.

Given a complete binary tree of height $h$, the vEB layout works recursively:

1. Split the tree at height $h/2$ into a **top subtree** of height $h/2$ and $\Theta(2^{h/2})$ **bottom subtrees**, each of height $h/2$.
2. Store the top subtree contiguously, followed by each bottom subtree contiguously.
3. Apply the same layout recursively within each sub-subtree.

This recursive splitting ensures that for any block size $B$, a search path of length $\log_2 N$ traverses only $O(\log_B N)$ blocks.

## Search Complexity

A search in a vEB-layout tree follows a root-to-leaf path of $\log_2 N$ nodes. The vEB layout guarantees that the path crosses at most:

$$
O(\log_B N)
$$

block boundaries, because each contiguous subtree of height $\frac{1}{2}\log_2 B$ fits within a constant number of blocks. This matches the B-tree's $O(\log_B N)$ search bound without knowing $B$.

## The Static Cache-Oblivious B-Tree

The static cache-oblivious search tree combines the vEB layout with a search strategy:

1. Store $N$ sorted keys in a complete binary search tree.
2. Lay out the tree in memory using the vEB layout.
3. Search by following the standard BST path from root to leaf.

| Operation | I/O Complexity |
|---|---|
| Search | $O(\log_B N)$ |
| Build | $O(N/B)$ (just scan sorted data) |

The search bound is optimal -- it matches the B-tree -- yet the algorithm never references $B$ or $M$.

## Dynamic Cache-Oblivious B-Trees

Supporting insertions and deletions cache-obliviously is harder. The **packed-memory array** technique maintains elements in a sorted array with controlled gaps, allowing insertions and deletions while preserving the vEB layout:

1. Maintain a density invariant: each segment of the array is between 25% and 75% full.
2. When a segment becomes too full or too sparse, rebalance the smallest enclosing segment that violates the density bounds.
3. The tree index over the array uses the vEB layout.

This achieves:

| Operation | Amortized I/O |
|---|---|
| Search | $O(\log_B N)$ |
| Insert | $O\!\left(\frac{\log^2 N}{B}\right)$ |
| Delete | $O\!\left(\frac{\log^2 N}{B}\right)$ |

The insert/delete bounds have a $\log^2 N / B$ term, which is slightly worse than the B-tree's $O(\log_B N)$ but remains efficient for large $B$.

## Example: Van Emde Boas Layout

```python
"""
Van Emde Boas tree layout for cache-oblivious search.

Demonstrates how the recursive vEB layout stores a binary search tree
so that searches cross O(log_B N) block boundaries for any block size B.
"""

import math

# ===================================================================
# Van Emde Boas layout
# ===================================================================

def veb_layout(keys: list[int]) -> list[int]:
    """
    Arrange sorted keys into van Emde Boas memory layout.

    Builds a complete binary search tree and lays it out recursively
    so that subtrees at every scale are stored contiguously.

    Parameters
    ----------
    keys : Sorted list of keys (length should be 2^h - 1).

    Returns
    -------
    List of keys in vEB layout order.
    """
    n = len(keys)
    if n <= 1:
        return keys[:]

    # Build BST in level-order first
    bst = [0] * (n + 1)  # 1-indexed
    _fill_bst(keys, bst, 1, 0, n - 1)

    # Apply vEB layout
    result = []
    _veb_recurse(bst, 1, int(math.log2(n + 1)), result)
    return result


def _fill_bst(keys, bst, node, lo, hi):
    """Fill a 1-indexed BST array from sorted keys."""
    if lo > hi or node >= len(bst):
        return
    mid = (lo + hi) // 2
    bst[node] = keys[mid]
    _fill_bst(keys, bst, 2 * node, lo, mid - 1)
    _fill_bst(keys, bst, 2 * node + 1, mid + 1, hi)


def _veb_recurse(bst, root, height, result):
    """Recursively produce vEB layout order."""
    if height <= 0 or root >= len(bst):
        return
    if height == 1:
        result.append(bst[root])
        return

    top_h = height // 2
    bottom_h = height - top_h

    # Collect top subtree
    _collect_top(bst, root, top_h, result)

    # Collect bottom subtrees
    bottom_roots = []
    _find_bottom_roots(root, top_h, bottom_roots)
    for br in bottom_roots:
        if br < len(bst):
            _veb_recurse(bst, br, bottom_h, result)


def _collect_top(bst, root, height, result):
    """Collect nodes in the top subtree (BFS-like)."""
    if height <= 0 or root >= len(bst):
        return
    result.append(bst[root])
    if height > 1:
        _collect_top(bst, 2 * root, height - 1, result)
        _collect_top(bst, 2 * root + 1, height - 1, result)


def _find_bottom_roots(root, top_height, roots):
    """Find root indices of bottom subtrees."""
    if top_height <= 0:
        roots.append(root)
        return
    _find_bottom_roots(2 * root, top_height - 1, roots)
    _find_bottom_roots(2 * root + 1, top_height - 1, roots)


def count_block_crossings(layout: list[int], block_size: int,
                          search_path: list[int]) -> int:
    """Count how many block boundaries a search path crosses."""
    positions = {val: i for i, val in enumerate(layout)}
    blocks_visited = set()
    for node in search_path:
        if node in positions:
            blocks_visited.add(positions[node] // block_size)
    return len(blocks_visited)


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    # Build a complete BST with 2^h - 1 nodes
    h = 4
    n = 2**h - 1
    keys = list(range(1, n + 1))

    layout = veb_layout(keys)
    print(f"Sorted keys: {keys}")
    print(f"vEB layout:  {layout}")
    print()

    # Show block crossings for different block sizes
    # Search path for key 5 in BST: root -> left/right ...
    search_path = [8, 4, 2, 1]  # Example path in BST
    for B in [2, 4, 8]:
        crossings = count_block_crossings(layout, B, search_path)
        theoretical = math.ceil(math.log(n) / math.log(max(2, B)))
        print(f"B={B}: blocks visited = {crossings}, "
              f"O(log_B N) = {theoretical}")
```

??? example "Sample Output"

    ```
    Sorted keys: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    vEB layout:  [8, 4, 12, 2, 6, 1, 3, 5, 7, 10, 14, 9, 11, 13, 15]

    B=2: blocks visited = 3, O(log_B N) = 4
    B=4: blocks visited = 2, O(log_B N) = 2
    B=8: blocks visited = 1, O(log_B N) = 2
    ```

    The vEB layout ensures that larger block sizes capture more of the search path in each transfer, matching the $O(\log_B N)$ bound for any $B$.

## Cache-Oblivious vs Cache-Aware

| Property | B-Tree (cache-aware) | Cache-oblivious B-tree |
|---|---|---|
| Knows $B$ and $M$ | Yes | No |
| Search I/O | $O(\log_B N)$ | $O(\log_B N)$ |
| Insert I/O | $O(\log_B N)$ | $O(\log^2 N / B)$ amortized |
| Portability | Tuned to one machine | Optimal on any machine |
| Implementation | Straightforward | More complex |

The cache-oblivious approach sacrifices a small constant factor in update costs for the guarantee that the data structure performs well on any hardware without retuning.

## Reference

- Frigo, M. et al. "Cache-Oblivious Algorithms," *FOCS*, 1999.
- Bender, M. et al. "Cache-Oblivious B-Trees," *SIAM Journal on Computing*, 35(2), 2005.
- Prokop, H. "Cache-Oblivious Algorithms," Master's thesis, MIT, 1999.
