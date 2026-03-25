# Probabilistic Analysis

Skip lists rely on randomization rather than strict structural invariants
to achieve balanced performance. Unlike AVL trees or red-black trees, which
maintain balance through rotations after every insertion, skip lists assign
random levels to new nodes and let probability ensure that the resulting
structure is well-balanced **in expectation**. This page provides the
probabilistic analysis that justifies the $O(\log n)$ expected time for
skip list operations.

## The Randomization Model

Each node in a skip list is assigned a random level. Level 1 is guaranteed;
each additional level is added independently with promotion probability $p$
(typically $p = 1/2$). Thus, the probability that a node reaches level $i$
is:

$$
\Pr[\text{level} \geq i] = p^{i-1}
$$

The level of a node follows a geometric distribution with parameter $1 - p$.
The expected level of a single node is:

$$
E[\text{level}] = \frac{1}{1 - p}
$$

For $p = 1/2$, this gives $E[\text{level}] = 2$, meaning the average node
has just two pointers -- one for level 1 and one for level 2.

## Expected Number of Nodes per Level

In a skip list with $n$ nodes, the expected number of nodes at level $i$ is:

$$
E[\text{nodes at level } i] = n \cdot p^{i-1}
$$

This means approximately $n$ nodes at level 1, $np$ at level 2,
$np^2$ at level 3, and so on. The levels thin out geometrically, creating
the layered structure that enables fast search.

## Expected Search Cost

The key result for skip lists is that search takes $O(\log n)$ expected
time. The analysis proceeds by counting the number of nodes examined during
a search, working **backward** from the found node to the header.

At each step of the backward analysis, the search path either:

1. **Moves up** one level (with probability $p$, since the current node was
   promoted), or
2. **Moves left** on the same level (with probability $1 - p$).

Let $C(h)$ be the expected number of nodes examined while climbing $h$
levels. The backward analysis gives the recurrence:

$$
C(h) = (1 - p) \cdot \bigl(1 + C(h)\bigr) + p \cdot \bigl(1 + C(h - 1)\bigr)
$$

The first term accounts for moving left (still $h$ levels to climb), and the
second for moving up (now $h - 1$ levels remain). Solving:

$$
C(h) = \frac{h}{p}
$$

Since the maximum level is $O(\log n)$ in expectation (see
[Expected Height](height.md)), the expected search cost is:

$$
E[\text{search cost}] = O\!\left(\frac{\log n}{p}\right) = O(\log n)
$$

For $p = 1/2$, the expected number of comparisons is approximately
$2 \log_2 n$.

## Expected Space

The total expected space across all levels is the sum of expected nodes at
each level:

$$
E[\text{total pointers}] = \sum_{i=1}^{\infty} n \cdot p^{i-1} = \frac{n}{1 - p}
$$

For $p = 1/2$, this gives $2n$ expected total pointers, meaning the skip
list uses $O(n)$ expected space -- only a constant factor more than a
standard linked list.

## Choice of Promotion Probability

The parameter $p$ controls a trade-off between search speed and space:

| $p$ | Expected search cost | Expected space per node |
|---|---|---|
| $1/2$ | $2 \log_2 n$ | 2 pointers |
| $1/3$ | $1.5 \log_3 n \approx 1.5 \log_2 n / \log_2 3$ | 1.5 pointers |
| $1/4$ | $1.33 \log_4 n$ | 1.33 pointers |
| $1/e$ | $\approx 1.58 \log_2 n$ (minimizes $\frac{\log n}{p}$) | $\approx 1.58$ pointers |

The value $p = 1/e \approx 0.368$ minimizes the expected number of
comparisons per search, but $p = 1/2$ is the most common choice because it
allows level generation using a single coin flip per level and simplifies
the analysis.

## Comparison with Balanced BSTs

| Property | Skip list | AVL / Red-black tree |
|---|---|---|
| Search | $O(\log n)$ expected | $O(\log n)$ worst case |
| Insertion | $O(\log n)$ expected | $O(\log n)$ worst case |
| Deletion | $O(\log n)$ expected | $O(\log n)$ worst case |
| Space | $O(n)$ expected | $O(n)$ worst case |
| Implementation | Simple (no rotations) | Complex (rotation cases) |
| Concurrency | Easy fine-grained locking | Difficult (rotations span nodes) |

Skip lists trade worst-case guarantees for simplicity. Their randomized
structure avoids the complex rebalancing logic of deterministic trees,
and their layered pointer structure supports fine-grained locking for
concurrent access more naturally than tree rotations do.

!!! note "High-probability bounds"
    The $O(\log n)$ bounds for skip lists hold not only in expectation but
    with high probability. Specifically, the probability that a search
    examines more than $c \log n$ nodes decreases polynomially in $n$ for
    sufficiently large constant $c$. This means worst-case degenerate
    behavior is astronomically unlikely for any reasonable $n$.

## Reference

- Pugh, W. "Skip Lists: A Probabilistic Alternative to Balanced Trees."
  *Communications of the ACM*, 33(6), 1990.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C.
  *Introduction to Algorithms* (4th ed.). MIT Press.
