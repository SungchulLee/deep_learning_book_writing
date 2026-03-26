# Augmented BST for Interval Trees

An **augmented binary search tree** extends a standard BST by storing additional information at each node — information that can be maintained efficiently during insertions, deletions, and rotations.  For [interval trees](structure.md), the key augmentation is the **maximum endpoint** in each subtree.  This extra field enables [overlap queries](overlap.md) in $O(\log n)$ time by allowing the search to prune entire subtrees that cannot contain overlapping intervals.

## The Augmentation Framework

CLRS describes a general four-step process for augmenting a balanced BST (e.g., a red-black tree):

1. **Choose an underlying data structure** — typically a red-black tree or AVL tree.
2. **Determine the additional information** to store at each node.
3. **Verify that the information can be maintained** during insertions, deletions, and rotations without increasing the asymptotic cost.
4. **Develop new operations** that use the augmented information.

For interval trees, the augmentation is straightforward and satisfies all four requirements.

## Augmented Node Structure

Each node $x$ stores an interval $[x.low, x.high]$ and an additional field:

$$
x.max = \max(x.high,\; x.left.max,\; x.right.max)
$$

The $x.max$ field records the maximum endpoint among all intervals stored in $x$'s subtree (including $x$ itself).  If a child is nil, its `.max` is treated as $-\infty$.

| Field | Description |
|-------|-------------|
| $x.low$ | Left endpoint of the interval (used as BST key) |
| $x.high$ | Right endpoint of the interval |
| $x.max$ | Maximum $high$ value in $x$'s entire subtree |

## Maintaining the Augmentation

The max field can be computed from a node's own interval and its children's max values:

$$
x.max = \max(x.high,\; x.left.max,\; x.right.max)
$$

This is a **local computation**: it depends only on information stored at $x$ and its two children.  Consequently:

**Insertion:** after inserting a new node and performing any rebalancing rotations, update $x.max$ for each node on the path from the new node to the root.  Cost: $O(\log n)$.

**Deletion:** after removing a node and performing fixup rotations, update $x.max$ along the affected path.  Cost: $O(\log n)$.

**Rotation:** when a rotation changes the parent–child relationship between two nodes, recompute $x.max$ for both nodes involved (in bottom-up order).  Cost: $O(1)$ per rotation.

!!! note "Why this works for any balanced BST"
    The augmentation theorem (CLRS, Theorem 14.1) states that any augmented field that can be computed from the node's own data and its children's augmented fields can be maintained during rotations in $O(1)$ time.  The max-endpoint field satisfies this condition.

## Example

Consider the following intervals inserted into an augmented BST (ordered by low endpoint):

| Interval | low (key) | high |
|----------|-----------|------|
| $[0, 3]$ | 0 | 3 |
| $[5, 8]$ | 5 | 8 |
| $[6, 10]$ | 6 | 10 |
| $[8, 9]$ | 8 | 9 |
| $[15, 23]$ | 15 | 23 |
| $[16, 21]$ | 16 | 21 |
| $[25, 30]$ | 25 | 30 |

An augmented red-black tree might look like:

```
              [8, 9] max=30
             /              \
     [5, 8] max=10     [15, 23] max=30
     /       \          /         \
[0,3]m=3 [6,10]m=10 [16,21]m=21 [25,30]m=30
```

Each node's max field equals the maximum high endpoint in its subtree.  The root's max (30) equals the global maximum endpoint.

## Other Common Augmentations

The same framework supports many other augmentations:

| Augmentation | Application |
|-------------|-------------|
| Subtree size | Order-statistics trees (find $k$-th smallest) |
| Subtree sum | Range-sum queries |
| Max endpoint | Interval trees (overlap queries) |
| Height | AVL trees (balance checking) |
| Black height | Red-black trees (verification) |

Each of these satisfies the requirement that the augmented field is computable from the node's data and its children's augmented fields.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 14. MIT Press.
- de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M. (2008). *Computational Geometry: Algorithms and Applications* (3rd ed.), Chapter 10. Springer.
