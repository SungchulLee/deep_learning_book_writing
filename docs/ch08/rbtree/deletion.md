# Red-Black Tree Deletion

Deletion from a red-black tree begins with the standard BST deletion procedure and then applies a [fixup](delete_fixup.md) to restore the red-black [properties](properties.md).  The challenge is that removing a black node reduces the black-height on one side of the tree, violating the property that every root-to-leaf path has the same number of black nodes.  Understanding when this violation occurs — and when it does not — is the key to the deletion algorithm.

## BST Deletion Recap

Standard BST deletion handles three cases based on the number of children of the node $z$ to be deleted:

1. **$z$ has no children:** remove $z$ directly.
2. **$z$ has one child:** replace $z$ with its child.
3. **$z$ has two children:** find $z$'s successor $y$ (the leftmost node in $z$'s right subtree), copy $y$'s key and satellite data into $z$, then delete $y$ (which has at most one child).

In the red-black tree version, the algorithm tracks both the **actually removed node** $y$ and the node $x$ that takes $y$'s place.

## The RB-DELETE Procedure

The algorithm uses a `TRANSPLANT` helper that replaces one subtree with another:

```
RB-TRANSPLANT(T, u, v):
    if u.parent == T.nil:
        T.root = v
    elif u == u.parent.left:
        u.parent.left = v
    else:
        u.parent.right = v
    v.parent = u.parent
```

The main deletion procedure:

```
RB-DELETE(T, z):
    y = z
    y-original-color = y.color
    if z.left == T.nil:
        x = z.right
        RB-TRANSPLANT(T, z, z.right)
    elif z.right == T.nil:
        x = z.left
        RB-TRANSPLANT(T, z, z.left)
    else:
        y = TREE-MINIMUM(z.right)       # z's successor
        y-original-color = y.color
        x = y.right
        if y.parent == z:
            x.parent = y                # needed when x is T.nil
        else:
            RB-TRANSPLANT(T, y, y.right)
            y.right = z.right
            y.right.parent = y
        RB-TRANSPLANT(T, z, y)
        y.left = z.left
        y.left.parent = y
        y.color = z.color
    if y-original-color == BLACK:
        RB-DELETE-FIXUP(T, x)
```

## Key Observations

**When is fixup needed?**  Only when `y-original-color` is BLACK.  If the removed/moved node was red, no black-height changes, no red-red violations are introduced, and the tree remains valid.

**What is $x$?**  The node $x$ is the child that moved into $y$'s position.  It may be `T.nil` (the sentinel), which is treated as a black node.

**Why track the original color of $y$?**  When $z$ has two children, $y$ is $z$'s successor and is physically moved into $z$'s position.  The successor $y$ takes on $z$'s color, so any potential violation comes from $y$'s original color, not its new color.

## Which Properties Can Be Violated

After RB-DELETE (before fixup), at most one property is violated:

| Property | Can it be violated? | When? |
|----------|-------------------|-------|
| 1. Every node is red or black | No | Colors are not changed to invalid values |
| 2. Root is black | Yes | If the root was deleted and replaced by a red node |
| 3. Leaves (T.nil) are black | No | T.nil is always black |
| 4. Red node has black children | Yes | If $x$ is red and $x.parent$ is red |
| 5. Equal black-height | Yes | If $y$ was black, paths through $x$ have one fewer black node |

The [fixup procedure](delete_fixup.md) resolves these violations using at most three rotations and $O(\log n)$ recolorings.

## Complexity

| Operation | Time |
|-----------|------|
| RB-DELETE | $O(\log n)$ |
| TRANSPLANT | $O(1)$ |
| TREE-MINIMUM | $O(\log n)$ |
| RB-DELETE-FIXUP | $O(\log n)$ |
| **Total** | $O(\log n)$ |

The total deletion cost is $O(\log n)$: $O(\log n)$ to find the successor, $O(1)$ for the transplant, and $O(\log n)$ for the fixup.

!!! tip "Deletion is the hardest RB-tree operation"
    While [insertion fixup](insert_fixup.md) has three symmetric cases, deletion fixup has four.  The additional complexity arises because deletion can create a black-height deficit that is harder to resolve than the red-red violation produced by insertion.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Section 13.4. MIT Press.
- Sedgewick, R., & Wayne, K. (2011). *Algorithms* (4th ed.), Section 3.3. Addison-Wesley.
