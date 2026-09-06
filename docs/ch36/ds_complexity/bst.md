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

## Exercises

**Exercise 1.**
Compare unbalanced BST, AVL tree, red-black tree, and B-tree in terms of worst-case search, insert, and delete time.

??? success "Solution to Exercise 1"
    | Structure | Search | Insert | Delete | Balance |
    |---|---|---|---|---|
    | Unbalanced BST | $O(n)$ | $O(n)$ | $O(n)$ | None |
    | AVL tree | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | Height $\le 1.44 \log n$ |
    | Red-black tree | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | Height $\le 2 \log n$ |
    | B-tree (order $m$) | $O(\log_m n)$ | $O(\log_m n)$ | $O(\log_m n)$ | All leaves same depth |

    AVL trees are more strictly balanced (faster lookups) but require more rotations on insertion. Red-black trees allow slightly taller trees but need fewer rotations (at most 2 per insert), making them faster for insert-heavy workloads. B-trees minimize disk I/O by maximizing branching factor. $\square$

---

**Exercise 2.**
Prove that an AVL tree with $n$ nodes has height at most $1.44 \log_2(n + 2)$.

??? success "Solution to Exercise 2"
    Let $N(h)$ be the minimum number of nodes in an AVL tree of height $h$. The AVL property requires subtree heights to differ by at most 1, so the minimum tree of height $h$ has subtrees of heights $h-1$ and $h-2$: $N(h) = N(h-1) + N(h-2) + 1$ with $N(0) = 1$, $N(1) = 2$. This recurrence is similar to Fibonacci: $N(h) > F(h+2) - 1$ where $F$ is the Fibonacci sequence. Since $F(k) \approx \phi^k / \sqrt{5}$ where $\phi = (1 + \sqrt{5})/2$: $n \ge N(h) > \phi^{h+2}/\sqrt{5} - 2$. Solving for $h$: $h < \log_\phi(n+2) \cdot \sqrt{5} \approx 1.44 \log_2(n+2)$. Therefore, the height is $O(\log n)$ with constant $\approx 1.44$. $\square$

---

**Exercise 3.**
Explain why red-black trees are preferred over AVL trees in standard library implementations (e.g., C++ `std::map`, Java `TreeMap`).

??? success "Solution to Exercise 3"
    Red-black trees perform at most 2 rotations per insertion and at most 3 per deletion, with recolorings propagating upward in $O(\log n)$. AVL trees may perform up to $O(\log n)$ rotations per deletion (one at each level). For workloads with frequent insertions and deletions, red-black trees have lower overhead per modification. AVL trees have a tighter height bound (1.44 vs. 2 times $\log n$), making lookups $\sim$30% faster in the worst case. But the difference is small in practice (one or two fewer comparisons for $n = 10^6$). Standard libraries prioritize balanced performance across all operations, making red-black trees the better default. AVL trees are preferred in read-heavy applications (databases, lookup tables) where the tighter height bound matters. $\square$

---

**Exercise 4.**
An order-statistic tree augments a BST with subtree sizes. Describe how to find the $k$-th smallest element in $O(\log n)$.

??? success "Solution to Exercise 4"
    Each node stores `size`: the number of nodes in its subtree ($\text{size} = 1 + \text{left.size} + \text{right.size}$). To find the $k$-th smallest: start at root. Let $r = \text{left.size} + 1$ (the rank of the root). If $k = r$: return root. If $k < r$: recurse on left subtree with same $k$. If $k > r$: recurse on right subtree with $k - r$. Each step descends one level, so time is $O(h) = O(\log n)$ for a balanced BST. The size field is updated in $O(1)$ per node during insertions, deletions, and rotations, so all operations remain $O(\log n)$. $\square$

---

**Exercise 5.**
A splay tree has $O(\log n)$ amortized time per operation but $O(n)$ worst case for a single operation. Explain when splay trees are preferable to balanced BSTs.

??? success "Solution to Exercise 5"
    Splay trees move accessed nodes to the root via rotations (splaying). This provides two advantages: (1) **Temporal locality**: recently accessed elements are near the root, so repeated accesses are $O(1)$. If the access pattern has a small working set, splay trees outperform balanced BSTs. (2) **Simplicity**: no balance metadata (no colors, no heights) -- just the splay operation. Easier to implement and lower memory overhead. Splay trees are preferable when: (a) the access pattern is skewed (some elements accessed much more than others); (b) simplicity of implementation matters; (c) amortized bounds are acceptable. They are not suitable when: (a) worst-case $O(\log n)$ per operation is required (real-time systems); (b) the access pattern is uniform (all elements equally likely) -- splay's constant factor is higher than red-black trees. $\square$
