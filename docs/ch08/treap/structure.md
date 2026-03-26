# Treap Structure

Balanced binary search trees like AVL trees and red-black trees maintain balance through explicit invariants (height difference, node coloring) and complex rebalancing code.  A **treap** (tree + heap) achieves balance through a completely different mechanism: each node carries a **random priority**, and the tree simultaneously satisfies BST order on keys and heap order on priorities.  The resulting structure is provably equivalent to a random BST, giving expected $O(\log n)$ height with remarkably simple code.

## Definition

A treap is a binary tree where each node stores a pair $(k, p)$ — a **key** $k$ and a **priority** $p$ — satisfying two simultaneous ordering constraints:

1. **BST property on keys:** for every node $x$, all keys in $x$'s left subtree are less than $x.key$, and all keys in the right subtree are greater.
2. **Heap property on priorities:** for every node $x$ with parent $y$, we have $x.priority \le y.priority$ (max-heap convention).

The name "treap" comes from combining "tree" (for BST) and "heap."

## Uniqueness

!!! note "Uniqueness theorem"
    For a set of $n$ key–priority pairs with distinct keys and distinct priorities, there exists exactly one treap.

**Proof.**  The node with the highest priority must be the root (heap property).  Its key $k^*$ partitions the remaining pairs into those with keys $< k^*$ (forming the left subtree) and those with keys $> k^*$ (forming the right subtree).  By induction on $n$, each subtree has a unique treap structure.  The base case $n = 0$ is trivially unique. $\square$

??? example "A treap with 5 nodes"
    Key–priority pairs: $(3, 7),\; (1, 5),\; (5, 6),\; (2, 1),\; (4, 3)$

    The node with the highest priority is $(3, 7)$, which becomes the root.  Keys less than 3 are $\{1, 2\}$; keys greater than 3 are $\{4, 5\}$.

    ```
            (3, 7)
           /       \
       (1, 5)     (5, 6)
          \        /
        (2, 1)  (4, 3)
    ```

    - BST order: inorder traversal gives keys 1, 2, 3, 4, 5.
    - Heap order: every child has a lower priority than its parent.

## Node Structure

Each treap node contains:

| Field | Description |
|-------|-------------|
| `key` | The search key (determines BST ordering) |
| `priority` | Random value (determines heap ordering) |
| `left` | Pointer to left child |
| `right` | Pointer to right child |

Priorities are assigned at node creation and never change.  They are typically drawn uniformly at random from a large range (e.g., 64-bit integers or floating-point values in $[0, 1]$).

## Connection to Random BSTs

The central insight behind treaps is their equivalence to random BSTs.  When priorities are drawn independently from a continuous distribution:

- The node with the highest priority becomes the root, and each of the $n$ nodes is equally likely to have the highest priority.
- This is exactly the same as inserting keys in a random order, where each key is equally likely to be inserted first.

Therefore, treaps inherit all statistical properties of random BSTs, including:

- **Expected height:** $O(\log n)$
- **Expected search time:** $O(\log n)$
- **Expected number of rotations per insert:** less than 2

See [randomized priorities](priorities.md) for the detailed analysis.

## Operations Overview

All treap operations maintain both the BST and heap properties:

| Operation | Approach | Expected time |
|-----------|----------|---------------|
| Search | Standard BST search (ignore priorities) | $O(\log n)$ |
| Insert | BST insert + rotate up to restore heap order | $O(\log n)$ |
| Delete | Rotate down to a leaf, then remove | $O(\log n)$ |
| [Split](split_merge.md) | Recursive decomposition by key | $O(\log n)$ |
| [Merge](split_merge.md) | Recursive combination by priority | $O(\log n)$ |

!!! tip "Two implementation styles"
    Treaps can be implemented using either **(1) rotations** (insert via BST insert + rotate up, delete via rotate down) or **(2) split/merge** (all operations decompose into split and merge). The split/merge style is often preferred for its simplicity and natural extension to implicit treaps.

## Comparison with Other Balanced BSTs

| Property | Treap | AVL | Red-black | Splay |
|----------|-------|-----|-----------|-------|
| Balance type | Probabilistic | Strict | Strict | Amortized |
| Height guarantee | Expected $O(\log n)$ | $\le 1.44 \log n$ | $\le 2 \log n$ | Amortized $O(\log n)$ |
| Implementation | Simple | Moderate | Complex | Simple |
| Persistent version | Easy (split/merge) | Moderate | Hard | Hard |
| Worst case | $O(n)$ | $O(\log n)$ | $O(\log n)$ | $O(n)$ |

## Reference

- Aragon, C. R., & Seidel, R. (1989). Randomized search trees. *30th IEEE Symposium on Foundations of Computer Science*, 540–545.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Problem 13-4. MIT Press.
