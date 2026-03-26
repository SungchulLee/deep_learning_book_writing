# B-Tree Deletion

[Insertion](insertion.md) into a B-tree splits full nodes to maintain the B-tree properties.  Deletion is the inverse operation: it removes a key and may need to **merge** or **borrow** from sibling nodes when a node falls below the minimum number of keys ($t - 1$, where $t$ is the [minimum degree](definition.md)).  The deletion algorithm handles three distinct cases depending on where the target key resides and whether surrounding nodes have keys to spare.

## Precondition: Ensuring Minimum Keys

Before descending into a child during deletion, the algorithm ensures that the child contains at least $t$ keys (one more than the minimum $t - 1$).  This guarantee means that when a key is eventually removed from a leaf, the leaf still satisfies the B-tree property without needing to backtrack.  The technique is analogous to the single-pass approach used in [B-tree insertion](insertion.md), where full nodes are split proactively during descent.

## Case 1: Key in a Leaf Node

If the key $k$ is found in a leaf node $x$ and $x$ has at least $t$ keys, simply remove $k$ from $x$.

This is the simplest case — no structural changes are needed.

## Case 2: Key in an Internal Node

If $k$ is found in an internal node $x$, the key cannot simply be removed because it serves as a separator between two child subtrees.  Three sub-cases arise:

**Case 2a.** The child $y$ that precedes $k$ has at least $t$ keys.  Find the **predecessor** $k'$ of $k$ in the subtree rooted at $y$ (the rightmost key in $y$'s subtree).  Replace $k$ with $k'$ in $x$, then recursively delete $k'$ from $y$'s subtree.

**Case 2b.** The child $z$ that follows $k$ has at least $t$ keys.  Find the **successor** $k'$ of $k$ in the subtree rooted at $z$ (the leftmost key in $z$'s subtree).  Replace $k$ with $k'$ in $x$, then recursively delete $k'$ from $z$'s subtree.

**Case 2c.** Both $y$ and $z$ have exactly $t - 1$ keys.  **Merge** $k$ and all keys of $z$ into $y$, so that $y$ now contains $2t - 1$ keys.  Remove the child pointer to $z$ from $x$.  Then recursively delete $k$ from $y$.

## Case 3: Key Not in Current Node (Descending)

If $k$ is not in the current internal node $x$, determine the child $c_i$ whose subtree must contain $k$.  Before descending into $c_i$, ensure it has at least $t$ keys:

**Case 3a.** If $c_i$ has only $t - 1$ keys but an immediate sibling has at least $t$ keys, perform a **rotation**: move a separator key from $x$ down into $c_i$, and move a key from the sibling up into $x$.  If borrowing from the left sibling, the sibling's largest key moves up and $x$'s separator moves down.

**Case 3b.** If $c_i$ and both its immediate siblings have exactly $t - 1$ keys, **merge** $c_i$ with one sibling: move the separator key from $x$ down into the merged node, which now has $2t - 1$ keys.

After ensuring $c_i$ has at least $t$ keys, recurse into $c_i$.

??? example "Deleting from a B-tree of minimum degree t = 3"
    Consider a B-tree with $t = 3$ (each non-root node holds 2–5 keys):

    ```
              [G, M, P, X]
             /   |   |   |   \
       [A,C] [D,E] [J,K] [N,O] [R,S,T,U,V]
    ```

    **Delete G (Case 2a):** G is in the root (internal node). The left child `[D,E]` has only 2 keys ($= t - 1$), but the predecessor of G in the left subtree is E. Since `[D,E]` has $t - 1$ keys, check the right child `[J,K]` (Case 2b): the successor is J. Replace G with J in the root, delete J from `[J,K]`, yielding `[K]`.

    **Delete D (Case 1):** D is in a leaf. After removing it, the leaf `[E]` has only 1 key, which is still $\ge t - 1 = 2$... wait, $t - 1 = 2$ and `[E]` has 1 key. This triggers Case 3 on the next deletion that descends through this node, requiring a borrow or merge.

## Single-Pass Deletion

The CLRS algorithm performs deletion in a **single downward pass** by proactively fixing nodes that have only $t - 1$ keys before descending into them.  This avoids the need to backtrack up the tree after removing a key.

The invariant maintained during descent is:

$$
\text{Every node visited (except possibly the root) has at least } t \text{ keys}
$$

This ensures that when the key is finally found and removed, the node still satisfies the minimum-key requirement.

## Complexity

| Operation | Time | Disk accesses |
|-----------|------|---------------|
| Delete | $O(t \log_t n)$ | $O(\log_t n)$ |

Each level of the tree requires $O(t)$ work to shift keys within a node, and the height is $O(\log_t n)$.  The number of disk accesses matches the height because each merge or rotation involves at most a constant number of neighboring nodes.

!!! warning "Shrinking the tree"
    The tree height decreases only when the root has a single key and both its children are merged.  The merged node becomes the new root.  This is the only operation that reduces the height of a B-tree.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Section 18.3. MIT Press.
- Bayer, R., & McCreight, E. (1972). Organization and maintenance of large ordered indexes. *Acta Informatica*, 1(3), 173–189.
