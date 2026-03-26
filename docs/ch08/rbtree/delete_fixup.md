# Red-Black Tree Deletion Fixup

When a [black node is removed](deletion.md) from a red-black tree, the black-height property may be violated: one path in the tree has one fewer black node than all others.  The **deletion fixup** procedure restores the red-black properties by performing rotations and recoloring along the path from the replacement node to the root.  The fixup considers four symmetric cases (and their mirror images), each designed to either resolve the violation locally or push it one level closer to the root.

## When Fixup Is Needed

After the standard BST deletion and transplant, fixup is invoked only when the **removed or moved node was black**.  Removing a red node does not violate any red-black property.  When a black node is removed, the subtree that lost the node has a black-height deficit of 1.

Let $x$ denote the node that moved into the deleted node's position (or the sentinel `T.nil` if the deleted node had no children).  The fixup procedure treats $x$ as carrying an "extra black" — conceptually, $x$ is either **double-black** (if $x$ is black) or **red-and-black** (if $x$ is red).  The goal is to remove this extra black by transferring it upward or absorbing it through rotations.

## The Four Cases

Let $x$ be the node with the extra black, and let $w$ be $x$'s sibling.  The cases below assume $x$ is a left child; the symmetric cases apply when $x$ is a right child.

### Case 1: Sibling w is Red

Since $w$ is red, both of $w$'s children must be black (by the red-black property).

- Recolor $w$ to black and $x.parent$ to red.
- Left-rotate on $x.parent$.
- The new sibling of $x$ is now one of $w$'s former children, which is black.

This transforms the situation into Case 2, 3, or 4 with a black sibling.

### Case 2: Sibling w is Black, Both Children of w are Black

Neither of $w$'s children can donate blackness through rotation.

- Remove one black from both $x$ (eliminating the extra black) and $w$ (recolor $w$ to red).
- Move the extra black up to $x.parent$.
- If $x.parent$ is red, color it black and terminate.  Otherwise, repeat the fixup with $x.parent$ as the new $x$.

### Case 3: Sibling w is Black, w's Left Child is Red, w's Right Child is Black

The sibling's red child is on the "wrong side" for a single rotation to fix the problem.

- Recolor $w$'s left child to black and $w$ to red.
- Right-rotate on $w$.
- The new sibling of $x$ now has a red right child, transitioning to Case 4.

### Case 4: Sibling w is Black, w's Right Child is Red

This is the terminal case — it resolves the extra black in one step.

- Set $w$'s color to $x.parent$'s color.
- Set $x.parent$ to black.
- Set $w$'s right child to black.
- Left-rotate on $x.parent$.
- Set $x$ to the root (terminating the loop).

After the rotation, the extra black is absorbed and all red-black properties are restored.

## Pseudocode

```
RB-DELETE-FIXUP(T, x):
    while x != T.root and x.color == BLACK:
        if x == x.parent.left:
            w = x.parent.right            # sibling
            if w.color == RED:             # Case 1
                w.color = BLACK
                x.parent.color = RED
                LEFT-ROTATE(T, x.parent)
                w = x.parent.right
            if w.left.color == BLACK and w.right.color == BLACK:  # Case 2
                w.color = RED
                x = x.parent
            else:
                if w.right.color == BLACK:  # Case 3
                    w.left.color = BLACK
                    w.color = RED
                    RIGHT-ROTATE(T, w)
                    w = x.parent.right
                w.color = x.parent.color   # Case 4
                x.parent.color = BLACK
                w.right.color = BLACK
                LEFT-ROTATE(T, x.parent)
                x = T.root
        else:
            # symmetric: swap left/right
            ...
    x.color = BLACK
```

## Case Flow

The cases are not independent — they form a directed progression:

- **Case 1** always leads to Case 2, 3, or 4 (with a black sibling).
- **Case 2** may repeat at a higher level (moving $x$ toward the root) or terminate if the parent is red.
- **Case 3** always leads to Case 4.
- **Case 4** always terminates the fixup.

This means the fixup performs at most **three rotations** total and $O(\log n)$ recolorings.

## Complexity

| Metric | Bound |
|--------|-------|
| Rotations | At most 3 |
| Recolorings | $O(\log n)$ |
| Total time | $O(\log n)$ |

!!! warning "Sentinel node"
    The fixup procedure references `x.parent` and sibling colors even when $x$ is `T.nil`.  The sentinel node must have its `color` set to BLACK and its `parent` pointer set correctly by the deletion procedure for the fixup to work.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Section 13.4. MIT Press.
- Sedgewick, R., & Wayne, K. (2011). *Algorithms* (4th ed.), Section 3.3. Addison-Wesley.
