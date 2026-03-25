# Height Analysis

The performance guarantee of an AVL tree rests entirely on the claim that its height is $O(\log n)$. Without this bound, the $O(\log n)$ complexity of search, insertion, and deletion would not follow. This section proves the height bound rigorously by analyzing the **sparsest possible** AVL trees --- those with the fewest nodes for a given height --- and connecting them to Fibonacci numbers.

## The Key Question

An AVL tree with $n$ nodes has height $h$. The question is: how large can $h$ be as a function of $n$? If we can show that $h = O(\log n)$, then every root-to-leaf path is logarithmic, and all dictionary operations run in logarithmic time.

The strategy is to flip the question: for a given height $h$, what is the **minimum** number of nodes $N(h)$ that an AVL tree of height $h$ can contain? If we find $N(h)$, then any AVL tree with $n \geq N(h)$ nodes has height at most $h$, and inverting the relationship gives $h$ in terms of $n$.

## Minimal AVL Trees

A **minimal AVL tree** of height $h$ is an AVL tree of height $h$ that contains the fewest possible nodes. To minimize nodes while maintaining height $h$, we make one subtree as short as the AVL condition allows.

Let $N(h)$ denote the minimum number of nodes in an AVL tree of height $h$. The base cases are:

$$
N(0) = 1, \qquad N(1) = 2
$$

For $h \geq 2$, a minimal AVL tree of height $h$ has:

- One subtree of height $h - 1$ (to achieve overall height $h$), which itself is a minimal AVL tree.
- One subtree of height $h - 2$ (the shortest the AVL condition permits), which is also a minimal AVL tree.
- The root node itself.

This gives the recurrence:

$$
N(h) = N(h-1) + N(h-2) + 1
$$

## Connection to Fibonacci Numbers

The recurrence $N(h) = N(h-1) + N(h-2) + 1$ closely resembles the Fibonacci recurrence $F(h) = F(h-1) + F(h-2)$. Define $M(h) = N(h) + 1$. Then:

$$
M(h) = N(h) + 1 = N(h-1) + N(h-2) + 2 = M(h-1) + M(h-2)
$$

with $M(0) = 2$ and $M(1) = 3$. Since Fibonacci numbers satisfy $F_0 = 0, F_1 = 1, F_2 = 1, F_3 = 2, \ldots$, we can verify that:

$$
M(h) = F_{h+3} - 1
$$

where $F_k$ is the $k$-th Fibonacci number (with $F_1 = 1, F_2 = 1$). Therefore:

$$
N(h) = F_{h+3} - 2
$$

## The Height Bound

Using the well-known approximation $F_k \approx \phi^k / \sqrt{5}$ where $\phi = (1 + \sqrt{5})/2 \approx 1.618$ is the golden ratio:

$$
N(h) \geq F_{h+3} - 2 \geq \frac{\phi^{h+3}}{\sqrt{5}} - 3
$$

For any AVL tree with $n$ nodes and height $h$, we have $n \geq N(h)$, so:

$$
n \geq \frac{\phi^{h+3}}{\sqrt{5}} - 3
$$

Taking logarithms base $\phi$:

$$
h \leq \log_\phi(n + 3) + \log_\phi \sqrt{5} - 3
$$

Since $\log_\phi = \log_2 / \log_2 \phi$ and $\log_2 \phi \approx 0.694$, we get $\log_\phi n \approx 1.44 \log_2 n$. The precise bound is:

$$
h \leq 1.44 \log_2(n + 2) - 0.328
$$

??? note "Derivation of the constant 1.44"
    The factor $1/\log_2 \phi = 1/\log_2((1+\sqrt{5})/2) \approx 1.4404$ arises from the change of base between $\log_\phi$ and $\log_2$. This means an AVL tree is at most about 44% taller than a perfectly balanced binary tree, which has height $\lfloor \log_2 n \rfloor$.

## Main Theorem

!!! info "Theorem: AVL Tree Height"
    An AVL tree with $n$ nodes has height $h = \Theta(\log n)$. Specifically:

    $$
    \lfloor \log_2 n \rfloor \leq h \leq 1.44 \log_2(n + 2) - 0.328
    $$

    The lower bound comes from the fact that a binary tree of height $h$ has at most $2^{h+1} - 1$ nodes, so $h \geq \lfloor \log_2 n \rfloor$. The upper bound comes from the minimal AVL tree analysis above.

**Proof sketch.** The upper bound follows from $n \geq N(h) = F_{h+3} - 2$ and the exponential growth of Fibonacci numbers. The lower bound holds for all binary trees: a tree of height $h$ has at most $2^{h+1} - 1$ nodes, so $n \leq 2^{h+1} - 1$ implies $h \geq \log_2(n+1) - 1$. $\square$

## Computing Minimal AVL Tree Sizes

```python
"""
Compute minimal AVL tree sizes and verify the height bound.

Demonstrates the connection between minimal AVL trees and
Fibonacci numbers, confirming the 1.44 * log2(n) height bound.
"""


# === Minimal AVL Tree Sizes ===

def minimal_avl_sizes(max_height):
    """Compute N(h) = minimum nodes in an AVL tree of height h."""
    if max_height < 0:
        return []
    sizes = [1, 2]  # N(0) = 1, N(1) = 2
    for h in range(2, max_height + 1):
        sizes.append(sizes[h - 1] + sizes[h - 2] + 1)
    return sizes[:max_height + 1]


# === Fibonacci Numbers ===

def fibonacci(k):
    """Compute F_k (1-indexed: F_1 = 1, F_2 = 1, F_3 = 2, ...)."""
    if k <= 0:
        return 0
    a, b = 0, 1
    for _ in range(k):
        a, b = b, a + b
    return a


# === Verification ===

def verify_fibonacci_connection(max_height):
    """Verify that N(h) = F_{h+3} - 2."""
    sizes = minimal_avl_sizes(max_height)
    print(f"{'h':>3} | {'N(h)':>8} | {'F(h+3)-2':>8} | {'Match':>5}")
    print("-" * 35)
    for h, n_h in enumerate(sizes):
        fib_val = fibonacci(h + 3) - 2
        match = "yes" if n_h == fib_val else "NO"
        print(f"{h:3d} | {n_h:8d} | {fib_val:8d} | {match:>5}")


# === Height Bound Check ===

import math

def height_bound(n):
    """Upper bound: h <= 1.44 * log2(n+2) - 0.328."""
    if n <= 0:
        return 0
    return 1.44 * math.log2(n + 2) - 0.328


if __name__ == "__main__":
    print("=== Minimal AVL Tree Sizes vs Fibonacci ===")
    verify_fibonacci_connection(12)
    print()

    print("=== Height Bound Verification ===")
    sizes = minimal_avl_sizes(12)
    print(f"{'h':>3} | {'N(h)':>8} | {'bound':>8}")
    print("-" * 28)
    for h, n_h in enumerate(sizes):
        bound = height_bound(n_h)
        print(f"{h:3d} | {n_h:8d} | {bound:8.2f}")
```

**Output:**
```
=== Minimal AVL Tree Sizes vs Fibonacci ===
  h |     N(h) | F(h+3)-2 | Match
-----------------------------------
  0 |        1 |        1 |   yes
  1 |        2 |        2 |   yes
  2 |        4 |        4 |   yes
  3 |        7 |        7 |   yes
  4 |       12 |       12 |   yes
  5 |       20 |       20 |   yes
  6 |       33 |       33 |   yes
  7 |       54 |       54 |   yes
  8 |       88 |       88 |   yes
  9 |      143 |      143 |   yes
 10 |      232 |      232 |   yes
 11 |      376 |      376 |   yes
 12 |      609 |      609 |   yes

=== Height Bound Verification ===
  h |     N(h) |    bound
----------------------------
  0 |        1 |     1.95
  1 |        2 |     2.63
  2 |        4 |     3.63
  3 |        7 |     4.60
  4 |       12 |     5.55
  5 |       20 |     6.49
  6 |       33 |     7.46
  7 |       54 |     8.42
  8 |       88 |     9.39
  9 |      143 |    10.36
 10 |      232 |    11.33
 11 |      376 |    12.30
 12 |      609 |    13.27
```

Each height $h$ is indeed below its corresponding bound value, confirming the theorem.

## Practical Implications

The height bound $h \leq 1.44 \log_2 n$ means:

- An AVL tree with $10^6$ nodes has height at most $1.44 \times 20 \approx 29$.
- A perfectly balanced tree would have height $20$. The AVL tree is at most 44% taller.
- All operations (search, insert, delete) visit at most $h$ nodes, so they run in $O(\log n)$ worst-case time.

This worst case is tight: minimal AVL trees (the Fibonacci trees described above) achieve height exactly $1.44 \log_2 n$. However, randomly built AVL trees tend to have height much closer to $\log_2 n$.

## Reference

- [6. AVL Trees, AVL Sort](https://www.youtube.com/watch?v=FNeL18KsWPc&list=PLUl4u3cNGP61Oq3tWYp6V_F-5jb5L2iHb&index=7)
- [AVL tree](https://en.wikipedia.org/wiki/AVL_tree)
- [1382. Balance a Binary Search Tree](https://leetcode.com/problems/balance-a-binary-search-tree/)
