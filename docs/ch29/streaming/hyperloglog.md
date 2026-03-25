# HyperLogLog

Counting the number of distinct elements in a data stream is one of the most fundamental streaming problems. Exact computation requires $\Omega(n)$ space, but for many applications an approximate count suffices. **HyperLogLog** (Flajolet et al., 2007) estimates the distinct count using only $O(\log \log U + \log(1/\delta))$ bits of memory, making it practical for cardinality estimation over billions of elements. It is deployed in production systems at Google, Redis, and numerous database engines.

## The Distinct Elements Problem

Given a stream $\sigma = a_1, a_2, \ldots, a_n$ with elements from $[U]$, the **distinct count** (or cardinality) is:

$$
D = |\{j \in [U] : f_j > 0\}|
$$

where $f_j$ is the frequency of element $j$.

!!! note "Exact Counting Is Expensive"
    Maintaining a set of all seen elements requires $O(D \log U)$ bits. When $D$ is in the billions and $U$ is large (e.g., IPv6 addresses, user IDs), this is prohibitively expensive. HyperLogLog achieves a $(1 \pm \epsilon)$-approximation using only $O(1/\epsilon^2)$ registers of $O(\log \log U)$ bits each.

## Intuition: The Flajolet-Martin Idea

The foundational idea (Flajolet and Martin, 1985) uses a simple observation: if you hash each stream element to a uniformly random binary string, the maximum number of leading zeros observed is roughly $\log_2 D$.

Let $h : [U] \to [0, 1)$ be a hash function that produces uniformly distributed values. Define:

$$
R = \max_{j : f_j > 0} \rho(h(j))
$$

where $\rho(x)$ denotes the position of the leftmost 1-bit in the binary representation of $x$ (i.e., the number of leading zeros plus one). If there are $D$ distinct elements, the expected value of $R$ is approximately $\log_2 D$.

The estimator $\hat{D} = 2^R$ is highly variable (its standard deviation is comparable to its mean), so HyperLogLog uses **stochastic averaging** to reduce variance.

## The HyperLogLog Algorithm

### Structure

Maintain $m = 2^b$ **registers** $M[1], M[2], \ldots, M[m]$, each initialized to 0. The parameter $b$ uses the first $b$ bits of the hash to select a register.

### Update

For each stream element $a$:

1. Compute $x = h(a)$, a hash value viewed as a binary string.
2. Use the first $b$ bits of $x$ to select register index $j = 1 + \langle x_1 x_2 \ldots x_b \rangle_2$.
3. Compute $w = \rho(x_{b+1} x_{b+2} \ldots)$, the position of the first 1-bit in the remaining bits.
4. Update: $M[j] \leftarrow \max(M[j], w)$.

### Estimate

The raw HyperLogLog estimate uses the **harmonic mean** of the $2^{M[j]}$ values:

$$
\hat{D} = \alpha_m \cdot m^2 \cdot \left(\sum_{j=1}^{m} 2^{-M[j]}\right)^{-1}
$$

where $\alpha_m$ is a bias correction constant:

$$
\alpha_m = \left(m \int_0^{\infty} \left(\log_2 \frac{2 + u}{1 + u}\right)^m du\right)^{-1}
$$

For practical values: $\alpha_{16} = 0.673$, $\alpha_{32} = 0.697$, $\alpha_{64} = 0.709$, and $\alpha_m = 0.7213 / (1 + 1.079/m)$ for $m \geq 128$.

## Accuracy Analysis

**Theorem.** The HyperLogLog estimator with $m$ registers provides:

$$
\frac{\sigma[\hat{D}]}{\mathbb{E}[\hat{D}]} \approx \frac{1.04}{\sqrt{m}}
$$

This means using $m$ registers gives a relative standard error of approximately $1.04/\sqrt{m}$:

| Registers ($m$) | Memory | Relative Error |
|---|---|---|
| 16 | 80 bytes | 26% |
| 256 | 1.3 KB | 6.5% |
| 1024 | 5 KB | 3.25% |
| 16384 | 82 KB | 0.81% |

!!! tip "Practical Memory Usage"
    Each register stores a value up to $\log_2 \log_2 U + O(1)$ bits. For 64-bit hash values, each register needs at most 6 bits. With $m = 16384$ registers, the total memory is about 12 KB — enough to count billions of distinct elements with less than 1% error.

## Corrections for Small and Large Ranges

The raw HyperLogLog estimate is biased for very small and very large cardinalities:

- **Small range correction**: when $\hat{D} < 5m/2$, use **linear counting** if there are empty registers: $\hat{D}_{\text{LC}} = m \ln(m / V)$ where $V$ is the number of registers equal to 0.
- **Large range correction**: when $\hat{D} > 2^{32}/30$ (for 32-bit hashes), apply a correction for hash collisions: $\hat{D}_{\text{corr}} = -2^{32} \ln(1 - \hat{D}/2^{32})$.

The HyperLogLog++ variant (Heule et al., 2013) combines these corrections with empirical bias estimation for improved accuracy across all ranges.

## Mergeability

HyperLogLog sketches are **mergeable**: given two sketches $S_1$ and $S_2$ with the same hash function and $m$ registers, the merged sketch is:

$$
M_{\text{merged}}[j] = \max(M_1[j], M_2[j]) \quad \text{for } j = 1, \ldots, m
$$

This enables distributed cardinality estimation: each node maintains a local HyperLogLog, and sketches are merged to estimate the global distinct count.

## Connection to Deep Learning

- **Vocabulary size estimation**: before training a language model, estimate the number of unique tokens in a large corpus using HyperLogLog to plan embedding table sizes.
- **Deduplication**: detect and remove duplicate training examples by hashing each example and using HyperLogLog to estimate the fraction of unique samples.
- **Feature cardinality**: in recommendation systems, estimate the number of distinct users, items, or feature values to size embedding tables appropriately.

## Summary

HyperLogLog estimates the number of distinct elements in a data stream using $O(m \log \log U)$ bits, achieving relative standard error $1.04/\sqrt{m}$. The algorithm combines hashing, stochastic averaging over $m$ registers, and the harmonic mean estimator with bias correction. Its mergeability makes it suitable for distributed systems, and its extreme space efficiency has made it the standard algorithm for cardinality estimation in practice.

## References

- [HyperLogLog: The Analysis of a Near-Optimal Cardinality Estimation Algorithm (Flajolet et al., 2007)](http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf)
- [Data Streams: Algorithms and Applications (Muthukrishnan)](https://www.cs.rutgers.edu/~muthu/stream-1-1.ps)
