# Expected Height

The height of a skip list -- the maximum level among all nodes -- determines
the number of layers the search algorithm must descend. If the height grew
linearly with $n$, skip lists would be no faster than a plain linked list.
The probabilistic analysis shows that the expected height is $O(\log n)$,
which is what makes $O(\log n)$ search possible. This page derives the
expected height and establishes high-probability bounds.

## Setup

Consider a skip list with $n$ nodes and promotion probability $p$. Each
node is independently assigned a random level: level 1 is guaranteed, and
each additional level is added with probability $p$. The **height** $H$ of
the skip list is the maximum level across all $n$ nodes:

$$
H = \max_{i=1}^{n} L_i
$$

where $L_i$ is the level of node $i$, and each $L_i$ follows a geometric
distribution with parameter $1 - p$.

## Probability of a Single Node Reaching Level k

A single node reaches level $k$ or higher if it is promoted $k - 1$ times,
each with probability $p$:

$$
\Pr[L_i \geq k] = p^{k-1}
$$

## Expected Maximum Level

The height $H \geq k$ if and only if at least one of the $n$ nodes reaches
level $k$. By the union bound:

$$
\Pr[H \geq k] = \Pr\!\left[\bigcup_{i=1}^{n} \{L_i \geq k\}\right] \leq n \cdot p^{k-1}
$$

Setting $k = \log_{1/p} n + 1$ (so that $p^{k-1} = 1/n$), this bound
becomes:

$$
\Pr\!\left[H \geq \log_{1/p} n + 1\right] \leq n \cdot \frac{1}{n} = 1
$$

This is trivial, but for $k = c \log_{1/p} n$ with $c > 1$:

$$
\Pr[H \geq c \log_{1/p} n] \leq n \cdot p^{c \log_{1/p} n - 1} = \frac{n}{p} \cdot n^{-c} = \frac{1}{p \cdot n^{c-1}}
$$

For any constant $c > 1$, this probability vanishes as $n$ grows.

The expected height can be computed exactly:

$$
E[H] = \sum_{k=1}^{\infty} \Pr[H \geq k] \leq \sum_{k=1}^{\infty} \min(1, \, n \cdot p^{k-1})
$$

The first $\log_{1/p} n$ terms contribute at most $\log_{1/p} n$. The
remaining terms form a geometric series:

$$
\sum_{k = \lfloor\log_{1/p} n\rfloor + 1}^{\infty} n \cdot p^{k-1} \leq \sum_{j=0}^{\infty} p^{j} = \frac{1}{1 - p}
$$

Therefore:

$$
E[H] \leq \log_{1/p} n + \frac{1}{1 - p} = O(\log n)
$$

For $p = 1/2$, this gives $E[H] \leq \log_2 n + 2$.

## High-Probability Bound

The union bound analysis above shows that the height exceeds
$c \log_{1/p} n$ with probability at most $O(n^{-(c-1)})$. This is a
**high-probability** bound: for any desired polynomial confidence
$1 - 1/n^d$, choose $c = d + 1$.

??? note "Concrete example"
    For $p = 1/2$ and $n = 10^6$ nodes, the expected height is at most
    $\log_2(10^6) + 2 \approx 22$. The probability that the height
    exceeds $40$ (roughly $2 \log_2 n$) is at most
    $10^6 \cdot 2^{-39} \approx 1.8 \times 10^{-6}$, or about one in
    half a million.

## Maximum Level Cap

In practice, skip list implementations set a maximum level cap
$\text{MaxLevel}$ to bound memory usage and avoid extremely tall (but
astronomically unlikely) nodes. A common choice is:

$$
\text{MaxLevel} = \lfloor \log_{1/p} n \rfloor + 1
$$

For $p = 1/2$ and an anticipated maximum of $n = 2^{16} = 65{,}536$
elements, $\text{MaxLevel} = 17$ is sufficient.

!!! warning "Setting MaxLevel too low"
    If MaxLevel is set below $\log_{1/p} n$, the skip list degrades: the
    top levels become overcrowded, and search approaches $O(n)$.
    Always set MaxLevel based on the expected maximum number of elements.

## Comparison with Deterministic Trees

| Property | Skip list height | AVL tree height | Red-black tree height |
|---|---|---|---|
| Bound type | Expected / high probability | Worst case | Worst case |
| Formula | $\leq \log_{1/p} n + O(1)$ | $\leq 1.44 \log_2 n$ | $\leq 2 \log_2 n$ |
| For $n = 10^6$ | $\approx 22$ ($p = 1/2$) | $\approx 29$ | $\approx 40$ |

The expected skip list height is competitive with deterministic tree
heights, and the high-probability bound ensures that pathological heights
are vanishingly unlikely.

## Reference

- Pugh, W. "Skip Lists: A Probabilistic Alternative to Balanced Trees."
  *Communications of the ACM*, 33(6), 1990.
- Motwani, R. & Raghavan, P. *Randomized Algorithms*, Chapter 3.
  Cambridge University Press.
