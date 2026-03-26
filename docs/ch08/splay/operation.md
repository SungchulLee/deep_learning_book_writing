# Splay Tree Operations

A splay tree is a self-adjusting binary search tree where every access to a node triggers a sequence of [rotations](rotations.md) that moves the accessed node to the root.  This **splay** step is the single primitive on which all standard BST operations — search, insert, delete, split, and join — are built.  Although no explicit balance condition is maintained, the [amortized analysis](amortized.md) guarantees $O(\log n)$ cost per operation over any sequence of operations.

## The Splay Primitive

The `splay(x)` procedure moves node $x$ to the root by repeatedly applying rotation steps based on the relationship between $x$, its parent $p$, and its grandparent $g$ (if it exists).  Three cases govern the rotations:

1. **Zig:** $p$ is the root.  Perform a single rotation at $p$ to make $x$ the root.
2. **Zig-zig:** $x$ and $p$ are both left children (or both right children).  Rotate at $g$ first, then rotate at $p$.
3. **Zig-zag:** $x$ is a left child and $p$ is a right child (or vice versa).  Rotate at $p$ first, then rotate at $g$.

The splay terminates when $x$ reaches the root.  See [rotations](rotations.md) for detailed diagrams and proofs of each step.

## Search

To search for a key $k$ in a splay tree:

1. Perform a standard BST search for $k$.
2. If $k$ is found at node $x$, splay $x$ to the root.
3. If $k$ is not found, splay the **last node visited** (the deepest node on the search path) to the root.

The splay step ensures that recently accessed elements are near the root, providing fast access on repeated queries.  The return value is the root after splaying: if $\text{root.key} = k$, the search succeeded.

## Insert

To insert a key $k$:

1. Splay the tree with key $k$.  After splaying, the root is either $k$ (if it already exists) or the predecessor/successor of $k$.
2. If the root's key equals $k$, the key already exists — done.
3. Otherwise, create a new node with key $k$ and make it the new root:
      - If $k$ is greater than the root's key, the new node's left child is the old root, and its right child is the old root's right subtree (detached from the old root).
      - If $k$ is less than the root's key, the new node's right child is the old root, and its left child is the old root's left subtree.

## Delete

To delete a key $k$:

1. Splay the tree with key $k$.  If the root's key is not $k$, the key does not exist — done.
2. If the root's key is $k$, remove the root.  This leaves two subtrees $L$ (left) and $R$ (right).
3. If $L$ is empty, $R$ becomes the new tree.
4. Otherwise, splay the **maximum element** of $L$ to the root of $L$.  Since this element has no right child (it is the maximum), attach $R$ as its right child.

## Split and Join

Splay trees support two powerful structural operations that are the basis for many advanced algorithms.

### Split

`split(T, k)` divides tree $T$ into two trees $L$ and $R$ such that all keys in $L$ are $\le k$ and all keys in $R$ are $> k$:

1. Splay $k$ (or its predecessor) to the root.
2. Detach the root's right subtree as $R$.
3. The remaining tree (root + left subtree) is $L$.

### Join

`join(L, R)` merges two trees where every key in $L$ is less than every key in $R$:

1. Splay the maximum element of $L$ to the root.
2. Set the root's right child to $R$.

Both split and join run in $O(\log n)$ amortized time.

## Complexity

| Operation | Amortized | Worst case |
|-----------|-----------|------------|
| Splay | $O(\log n)$ | $O(n)$ |
| Search | $O(\log n)$ | $O(n)$ |
| Insert | $O(\log n)$ | $O(n)$ |
| Delete | $O(\log n)$ | $O(n)$ |
| Split | $O(\log n)$ | $O(n)$ |
| Join | $O(\log n)$ | $O(n)$ |

!!! warning "No worst-case guarantee"
    Unlike AVL or red-black trees, splay trees do not guarantee $O(\log n)$ worst-case time for a single operation.  Applications that require strict worst-case bounds (e.g., real-time systems) should use a balanced BST instead.

## When to Use Splay Trees

Splay trees excel in scenarios with **temporal locality** — when recently accessed elements are likely to be accessed again soon.  The working set theorem shows that the cost of accessing an element is $O(\log w)$ amortized, where $w$ is the number of distinct elements accessed since the last access to the same element.

Common use cases include:

- **Cache implementations** where hot keys are accessed repeatedly.
- **Network routing tables** where popular destinations are queried frequently.
- **Undo/redo buffers** where recent operations are revisited.

## Reference

- Sleator, D. D., & Tarjan, R. E. (1985). Self-adjusting binary search trees. *Journal of the ACM*, 32(3), 652–686.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Problem 13-2. MIT Press.
