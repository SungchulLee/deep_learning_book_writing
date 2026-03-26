# BST Family

Binary search trees maintain the invariant that every node's left subtree contains
only smaller keys and its right subtree contains only larger keys. This property
enables $O(h)$ search, insertion, and deletion, where $h$ is the tree height. The
various BST variants differ in how they bound $h$ to prevent degeneration to $O(n)$.

## Unbalanced BST

A plain BST provides no height guarantee. If keys are inserted in sorted order, the
tree degenerates into a linked list.

| Operation | Best | Average (random) | Worst | Space |
|---|---|---|---|---|
| Search | $O(1)$ | $O(\log n)$ | $O(n)$ | $O(n)$ |
| Insert | $O(1)$ | $O(\log n)$ | $O(n)$ | $O(n)$ |
| Delete | $O(\log n)$ | $O(\log n)$ | $O(n)$ | $O(n)$ |
| Min / Max | $O(\log n)$ | $O(\log n)$ | $O(n)$ | -- |
| Successor / Predecessor | $O(\log n)$ | $O(\log n)$ | $O(n)$ | -- |
| In-order traversal | $O(n)$ | $O(n)$ | $O(n)$ | $O(h)$ |

The expected height of a randomly built BST with $n$ keys is $O(\log n)$, specifically
$E[h] = 4.311 \ln n$ asymptotically.

## Self-Balancing BSTs

Self-balancing BSTs guarantee $O(\log n)$ height through rotations or restructuring
after each modification.

| Tree Type | Search | Insert | Delete | Height Bound | Rotations per Op |
|---|---|---|---|---|---|
| AVL | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | $1.44 \log_2 n$ | $O(1)$ insert, $O(\log n)$ delete |
| Red-Black | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | $2 \log_2(n+1)$ | $O(1)$ |
| Splay | $O(\log n)$ amort. | $O(\log n)$ amort. | $O(\log n)$ amort. | $O(n)$ worst | $O(\log n)$ amort. |
| Treap | $O(\log n)$ exp. | $O(\log n)$ exp. | $O(\log n)$ exp. | $O(\log n)$ expected | $O(1)$ expected |
| Scapegoat | $O(\log n)$ | $O(\log n)$ amort. | $O(\log n)$ amort. | $O(\log n)$ | Rebuild subtree |

!!! tip "AVL vs Red-Black"
    AVL trees have tighter balance (height at most $1.44 \log n$) and are faster
    for lookup-heavy workloads. Red-Black trees allow slightly worse balance
    ($2 \log n$) but require fewer rotations on insertion and deletion, making
    them preferred for modification-heavy workloads. C++ `std::map` and Java
    `TreeMap` use Red-Black trees.

## B-Trees and Variants

B-trees generalize BSTs to have multiple keys per node, reducing tree height and disk
I/O for external storage.

| Tree Type | Search | Insert | Delete | Height | Node Size |
|---|---|---|---|---|---|
| B-tree (order $m$) | $O(\log_m n)$ | $O(\log_m n)$ | $O(\log_m n)$ | $O(\log_m n)$ | $m - 1$ keys |
| B+ tree | $O(\log_m n)$ | $O(\log_m n)$ | $O(\log_m n)$ | $O(\log_m n)$ | Leaves linked |
| 2-3 tree ($m = 3$) | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | 1--2 keys |
| 2-3-4 tree ($m = 4$) | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | 1--3 keys |

For databases with block size $B$, setting $m = B$ minimizes disk accesses:

$$
\text{disk accesses} = O(\log_{B} n)
$$

## Augmented BSTs

Augmenting BST nodes with extra information enables specialized queries without
changing the asymptotic complexity of basic operations.

| Augmentation | Extra Field | Supported Query | Query Time |
|---|---|---|---|
| Order statistics | Subtree size | $k$-th smallest element | $O(\log n)$ |
| Interval tree | Max endpoint in subtree | Overlapping intervals | $O(\log n + k)$ |
| Range tree | -- (uses nested trees) | 2D range query | $O(\log^2 n + k)$ |

Here $k$ is the number of results returned.

## Space Comparison

| Tree Type | Space per Node | Total Space | Notes |
|---|---|---|---|
| Unbalanced BST | 2 pointers + key | $O(n)$ | Minimal overhead |
| AVL | 2 pointers + key + balance factor | $O(n)$ | 1 extra byte |
| Red-Black | 2 pointers + key + color bit | $O(n)$ | 1 extra bit |
| Splay | 2 pointers + key + parent | $O(n)$ | Parent pointer needed |
| B-tree (order $m$) | $m$ pointers + $m-1$ keys | $O(n)$ | Large nodes |

## When to Use Each Variant

| Use Case | Recommended Tree | Why |
|---|---|---|
| General-purpose ordered map | Red-Black | Good balance of lookup and modification |
| Read-heavy, few writes | AVL | Tighter balance gives faster lookup |
| Recently accessed keys are hot | Splay | Amortized $O(\log n)$ with working set property |
| Database index | B+ tree | Minimizes disk I/O |
| Need randomized balance | Treap | Simple implementation, expected $O(\log n)$ |

## Reference

- [Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Sedgewick, R. and Wayne, K. *Algorithms*. 4th ed. Addison-Wesley, 2011.
