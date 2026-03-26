# Amortized Analysis of Splay Trees

A single [splay operation](operation.md) can take $O(n)$ time if the tree is highly unbalanced.  Despite this worst case, Sleator and Tarjan proved that any sequence of $m$ splay operations on a tree with at most $n$ nodes takes $O(m \log n)$ total time, giving an **amortized cost of $O(\log n)$ per operation**.  The proof uses the potential method with a carefully chosen potential function that captures how "unbalanced" the tree is.

## The Potential Function

Assign each node $x$ a **rank** equal to the logarithm of the size of its subtree:

$$
r(x) = \log_2 s(x)
$$

where $s(x)$ is the number of nodes in the subtree rooted at $x$ (including $x$ itself).

The **potential** of the entire tree $T$ is the sum of all ranks:

$$
\Phi(T) = \sum_{x \in T} r(x)
$$

The potential is always non-negative (since $s(x) \ge 1$ for every node) and satisfies $0 \le \Phi(T) \le n \log_2 n$.

## The Access Lemma

The key result bounding the amortized cost of a splay operation:

!!! note "Access Lemma"
    The amortized cost of splaying a node $x$ to the root of a tree with root $t$ is at most:

    $$
    \hat{c} \le 3(r(t) - r(x)) + 1 = 3\log_2 \frac{s(t)}{s(x)} + 1
    $$

Since $s(t) = n$ (the root's subtree contains all $n$ nodes) and $s(x) \ge 1$, this gives:

$$
\hat{c} \le 3 \log_2 n + 1 = O(\log n)
$$

## Proof Sketch

The proof analyzes each [rotation](rotations.md) step (zig, zig-zig, zig-zag) separately and shows that the amortized cost of each step is bounded in terms of rank changes.

Let $r(x)$ and $r'(x)$ denote the rank of node $x$ before and after a single splay step.

**Zig step** (single rotation at the root): the amortized cost is at most:

$$
\hat{c}_{\text{zig}} \le 1 + 3(r'(x) - r(x))
$$

**Zig-zig step** (two rotations in the same direction): the amortized cost is at most:

$$
\hat{c}_{\text{zig-zig}} \le 3(r'(x) - r(x))
$$

**Zig-zag step** (two rotations in opposite directions): the amortized cost is at most:

$$
\hat{c}_{\text{zig-zag}} \le 3(r'(x) - r(x))
$$

The proof of each bound uses the concavity of the logarithm function. For the zig-zig case, the critical inequality is:

$$
\log a + \log b \le 2 \log \frac{a + b}{2} - 2
$$

which holds when $a + b \le c$ for appropriate subtree sizes.

Summing the amortized costs over all steps in a splay operation yields a telescoping sum.  The intermediate rank terms cancel, leaving only $3(r'(\text{root}) - r(x)) + 1$, which proves the access lemma.

## Amortized Bound for a Sequence

For a sequence of $m$ splay operations on a tree that never exceeds $n$ nodes:

$$
\sum_{i=1}^{m} c_i = \sum_{i=1}^{m} \hat{c}_i + \Phi_0 - \Phi_m \le m(3 \log_2 n + 1) + \Phi_0
$$

Since $\Phi_0 \le n \log_2 n$ and $\Phi_m \ge 0$, the total cost is:

$$
O(m \log n + n \log n) = O((m + n) \log n)
$$

When $m \ge n$ (which is typical), this simplifies to $O(m \log n)$.

## Weighted Analysis

The potential function can be generalized by assigning **weights** $w(x)$ to nodes, letting $s(x) = \sum_{y \in \text{subtree}(x)} w(x)$ be the weighted subtree size.  This leads to the **dynamic optimality conjecture**: splay trees are within a constant factor of any binary search tree on any access sequence.

The weighted access lemma gives:

$$
\hat{c} \le 3 \log_2 \frac{W}{w(x)} + 1
$$

where $W = \sum_{x} w(x)$ is the total weight.  Setting $w(x) = 1/n$ for all $x$ recovers the unweighted bound.

## Complexity Summary

| Metric | Bound |
|--------|-------|
| Worst-case single splay | $O(n)$ |
| Amortized single splay | $O(\log n)$ |
| Sequence of $m$ operations | $O((m + n) \log n)$ |

!!! tip "Why amortized analysis matters for splay trees"
    Unlike AVL or red-black trees, splay trees have no explicit balance condition.  Their good performance is a purely amortized phenomenon: expensive operations restructure the tree enough to prepay for future cheap operations.

## Reference

- Sleator, D. D., & Tarjan, R. E. (1985). Self-adjusting binary search trees. *Journal of the ACM*, 32(3), 652–686.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Problem 13-2. MIT Press.
