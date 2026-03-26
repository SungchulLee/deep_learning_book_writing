# Splay Rotations: Zig, Zig-Zig, Zig-Zag

The [splay operation](operation.md) moves a target node $x$ to the root of the tree through a sequence of **rotation steps**.  Each step is chosen based on the relationship between $x$, its parent $p$, and its grandparent $g$ (if one exists).  The three cases — zig, zig-zig, and zig-zag — are designed not just to move $x$ upward, but to simultaneously improve the balance of the tree along the access path.  This restructuring property is what gives splay trees their $O(\log n)$ [amortized performance](amortized.md).

## Why Not Simple Rotations?

A naive approach would rotate $x$ with its parent repeatedly until $x$ reaches the root.  This "move-to-root" strategy does move $x$ to the top, but it can leave the tree just as unbalanced as before — it merely shifts the imbalance to a different path.  The zig-zig and zig-zag steps fix this by rotating at the **grandparent first** (in the zig-zig case), which compresses the access path and improves balance for future operations.

## Zig Step

The zig step applies when $x$'s parent $p$ is the **root** of the tree.  Since there is no grandparent, a single rotation suffices.

**$x$ is the left child of $p$:** right-rotate at $p$.

```
      p               x
     / \             / \
    x   C    →      A   p
   / \                 / \
  A   B               B   C
```

**$x$ is the right child of $p$:** left-rotate at $p$.

The zig step occurs at most once per splay operation (as the final step when $x$ is one level below the root).

## Zig-Zig Step

The zig-zig step applies when $x$ and its parent $p$ are **both left children** (or both right children) of their respective parents.  The critical detail is the **order of rotations**: rotate at the grandparent $g$ first, then at $p$.

**Both left children:**

```
        g                 p               x
       / \              / \             / \
      p   D    →       x   g    →      A   p
     / \              / \ / \             / \
    x   C            A  B C  D           B   g
   / \                                      / \
  A   B                                    C   D
```

Step 1: right-rotate at $g$ (bringing $p$ up).
Step 2: right-rotate at $p$ (bringing $x$ up).

**Both right children:** the mirror image — left-rotate at $g$, then left-rotate at $p$.

!!! warning "Rotation order matters"
    Rotating at $p$ first (instead of $g$) gives the naive move-to-root heuristic, which does NOT achieve $O(\log n)$ amortized cost.  The grandparent-first order is essential because it compresses the path from $g$ downward, reducing the depth of all nodes on the access path.

## Zig-Zag Step

The zig-zag step applies when $x$ and $p$ are on **opposite sides**: $x$ is a right child and $p$ is a left child (or vice versa).  Two rotations in opposite directions bring $x$ to $g$'s position.

**$x$ is right child of $p$, $p$ is left child of $g$:**

```
      g                 g               x
     / \              / \             / \
    p   D    →       x   D    →      p   g
   / \              / \             / \ / \
  A   x            p   C          A  B C  D
     / \          / \
    B   C        A   B
```

Step 1: left-rotate at $p$ (bringing $x$ above $p$).
Step 2: right-rotate at $g$ (bringing $x$ above $g$).

**$x$ is left child of $p$, $p$ is right child of $g$:** the mirror image — right-rotate at $p$, then left-rotate at $g$.

## Summary of Cases

| Case | Condition | Rotations | When it occurs |
|------|-----------|-----------|----------------|
| Zig | $p$ is the root | 1 rotation at $p$ | At most once (last step) |
| Zig-zig | $x$ and $p$ same side | 2 rotations: $g$ then $p$ | Any non-root step |
| Zig-zag | $x$ and $p$ opposite sides | 2 rotations: $p$ then $g$ | Any non-root step |

Each zig-zig or zig-zag step moves $x$ up by **two levels**.  The zig step moves $x$ up by one level.  Therefore, if $x$ starts at depth $d$, the splay operation performs at most $\lfloor d/2 \rfloor$ zig-zig/zig-zag steps plus at most one zig step.

## Effect on Tree Structure

The key structural property of splaying is **path compression**: after splaying node $x$, every node on the original access path from the root to $x$ has its depth roughly halved.  This ensures that a sequence of accesses to deep nodes does not repeatedly pay $O(n)$ cost — the first access restructures the tree so that subsequent accesses are cheaper.

## Complexity Per Step

Each individual rotation is an $O(1)$ operation (rearranging a constant number of pointers).  The entire splay operation performs $O(d)$ rotations where $d$ is the depth of $x$.  The amortized cost is $O(\log n)$ by the [access lemma](amortized.md).

## Reference

- Sleator, D. D., & Tarjan, R. E. (1985). Self-adjusting binary search trees. *Journal of the ACM*, 32(3), 652–686.
- Goodrich, M. T., & Tamassia, R. (2014). *Data Structures and Algorithms in Java* (6th ed.), Section 11.4. Wiley.
