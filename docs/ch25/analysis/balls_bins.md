# Balls into Bins

When a hash function maps $n$ keys to $m$ slots, or when $n$ jobs are randomly assigned to $m$ servers, the resulting distribution follows the **balls-into-bins** model. Understanding how uniformly random assignments distribute load — and in particular, how heavily the most loaded bin is filled — is fundamental to hashing, load balancing, and the analysis of randomized algorithms.

## The Model

Throw $n$ balls independently and uniformly at random into $m$ bins. Let $B_j$ denote the number of balls in bin $j$ for $j = 1, 2, \ldots, m$. The key questions are:

1. **Expected load**: What is $E[B_j]$?
2. **Maximum load**: What is $E[\max_j B_j]$?
3. **Empty bins**: How many bins are empty?
4. **Collisions**: How many bins contain more than one ball?

## Expected Load per Bin

By symmetry and linearity of expectation, each bin receives an equal expected number of balls:

$$
E[B_j] = \frac{n}{m}
$$

This follows immediately from defining $X_{ij} = I\{\text{ball } i \text{ lands in bin } j\}$ and noting $E[X_{ij}] = 1/m$, so $E[B_j] = \sum_{i=1}^{n} 1/m = n/m$.

## Maximum Load (Birthday Regime: n = m)

The most important case is $n = m$ (as many balls as bins). Even though the expected load per bin is 1, some bins are much more heavily loaded due to random fluctuation.

**Theorem.** When $n$ balls are thrown into $n$ bins uniformly at random, the maximum load satisfies

$$
\Pr\left[\max_j B_j \geq \frac{3 \ln n}{\ln \ln n}\right] \leq \frac{1}{n}
$$

The proof uses a union bound and a careful binomial estimate. For a single bin $j$, the probability that it receives at least $k$ balls is

$$
\Pr[B_j \geq k] \leq \binom{n}{k} \left(\frac{1}{n}\right)^k \leq \left(\frac{e}{k}\right)^k
$$

Setting $k = 3 \ln n / \ln \ln n$ and applying a union bound over all $n$ bins yields the result.

!!! tip "The Log n / Log Log n Threshold"
    The maximum load $\Theta(\ln n / \ln \ln n)$ is a fundamental result that appears throughout randomized algorithm analysis. It governs the worst-case lookup time in hash tables with chaining and the maximum queue length in random load balancing.

## Empty Bins

The expected number of empty bins when throwing $n$ balls into $m$ bins is

$$
E[\text{empty bins}] = m \left(1 - \frac{1}{m}\right)^n
$$

For $n = m$, this becomes $m \cdot (1 - 1/m)^m \approx m/e \approx 0.368m$. About $37\%$ of bins are empty, even though the average load is 1 ball per bin.

## Collisions

The expected number of bins with at least 2 balls (collisions) relates to the birthday paradox. A bin has a collision if at least 2 of the $n$ balls land in it. By inclusion-exclusion on pairs, the expected number of colliding pairs is $\binom{n}{2}/m$.

## Power of Two Choices

A remarkable improvement occurs with the "two choices" paradigm: instead of placing each ball in a uniformly random bin, place it in the **less loaded** of two randomly chosen bins. This reduces the maximum load dramatically.

**Theorem (Azar et al., 1999).** With the two-choices strategy, $n$ balls and $n$ bins, the maximum load is

$$
\max_j B_j = \frac{\ln \ln n}{\ln 2} + O(1)
$$

with high probability. This is an exponential improvement over the $\Theta(\ln n / \ln \ln n)$ bound for a single random choice.

| Strategy | Maximum load (whp) |
|---|---|
| One random choice | $\Theta(\ln n / \ln \ln n)$ |
| Two random choices | $\Theta(\ln \ln n)$ |
| $d$ random choices ($d \geq 2$) | $\Theta(\ln \ln n / \ln d)$ |

## Applications

### Hash Tables

With $n$ keys hashed into $m = n$ slots using chaining, the longest chain has expected length $\Theta(\ln n / \ln \ln n)$. With universal hashing and two hash functions (cuckoo hashing), the maximum probe sequence length drops to $O(\log \log n)$.

### Load Balancing

Assigning $n$ tasks to $n$ servers uniformly at random gives maximum load $\Theta(\ln n / \ln \ln n)$. The power-of-two-choices strategy, where each task probes two random servers and joins the shorter queue, reduces maximum load to $\Theta(\ln \ln n)$.

### Random Graphs

In the Erdos-Renyi random graph $G(n, p)$, the degree distribution relates to balls-into-bins: each potential edge is a "ball" assigned to vertex pairs. Maximum degree analysis uses the same techniques.

## Reference

- Motwani, R. & Raghavan, P. *Randomized Algorithms*. Cambridge University Press, 1995.
- Mitzenmacher, M. & Upfal, E. *Probability and Computing*. Cambridge University Press, 2017.
