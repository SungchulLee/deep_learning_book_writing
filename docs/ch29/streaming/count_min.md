# Count-Min Sketch

How can we estimate the frequency of any item in a massive data stream using only a small, fixed amount of memory? The **Count-Min Sketch** (CMS), introduced by Cormode and Muthukrishnan in 2005, answers this question with an elegant probabilistic data structure. It uses a two-dimensional array of counters with pairwise independent hash functions to provide one-sided error guarantees: estimates never undercount, and overcounting is bounded with high probability. The CMS has become one of the most widely deployed streaming data structures, used in databases, network monitoring, and machine learning systems.

## Data Structure

A Count-Min Sketch consists of a $d \times w$ array of counters $C[1 \ldots d][1 \ldots w]$, initialized to zero, together with $d$ pairwise independent hash functions $h_1, h_2, \ldots, h_d : [U] \to [w]$.

The parameters are chosen as:

$$
w = \left\lceil \frac{e}{\epsilon} \right\rceil, \quad d = \left\lceil \ln \frac{1}{\delta} \right\rceil
$$

where $\epsilon$ is the desired accuracy and $\delta$ is the failure probability.

**Total space**: $O\left(\frac{1}{\epsilon} \log \frac{1}{\delta}\right)$ counters plus $O(d \log U)$ bits for the hash functions.

## Operations

### Update

When element $a$ arrives in the stream, increment the corresponding counter in each row:

$$
C[i][h_i(a)] \leftarrow C[i][h_i(a)] + 1 \quad \text{for } i = 1, 2, \ldots, d
$$

For weighted updates with weight $c$, add $c$ instead of 1.

### Point Query

To estimate the frequency $f_j$ of item $j$, return the minimum counter across all rows:

$$
\hat{f}_j = \min_{i=1}^{d} C[i][h_i(j)]
$$

The minimum operation mitigates the effect of hash collisions: while some rows may have inflated counts due to collisions, at least one row is likely to have a count close to the true frequency.

## Accuracy Guarantees

**Theorem.** For any item $j$, the Count-Min Sketch estimate satisfies:

1. **No underestimation**: $\hat{f}_j \geq f_j$ always.
2. **Bounded overestimation**: $P(\hat{f}_j > f_j + \epsilon n) < \delta$, where $n$ is the stream length.

*Proof.*

(1) Each counter $C[i][h_i(j)]$ accumulates $f_j$ from item $j$ itself plus non-negative contributions from other items that hash to the same bucket. Therefore $C[i][h_i(j)] \geq f_j$ for all $i$, and the minimum is at least $f_j$.

(2) For a fixed row $i$, define the **collision noise**:

$$
X_i = C[i][h_i(j)] - f_j = \sum_{k \neq j} f_k \cdot \mathbf{1}[h_i(k) = h_i(j)]
$$

By pairwise independence of $h_i$:

$$
\mathbb{E}[X_i] = \sum_{k \neq j} f_k \cdot \frac{1}{w} \leq \frac{n}{w} = \frac{\epsilon n}{e} \leq \frac{\epsilon n}{e}
$$

By Markov's inequality:

$$
P(X_i > \epsilon n) \leq \frac{\mathbb{E}[X_i]}{\epsilon n} \leq \frac{1}{e}
$$

Since the $d$ hash functions are independent:

$$
P(\hat{f}_j - f_j > \epsilon n) = P(\min_i X_i > \epsilon n) = \prod_{i=1}^{d} P(X_i > \epsilon n) \leq \left(\frac{1}{e}\right)^d = e^{-d} \leq \delta
$$

$\square$

!!! tip "Conservative Update"
    The **conservative update** optimization only increments $C[i][h_i(a)]$ to $\max(C[i][h_i(a)], \hat{f}_a + 1)$ rather than blindly adding 1. This reduces overcounting in practice without changing the worst-case guarantees.

## Applications

### Heavy Hitter Detection

To find all items with frequency exceeding $\epsilon n$:

1. Maintain a Count-Min Sketch.
2. After each update, check if $\hat{f}_{a_i} > \epsilon n$.
3. If so, add $a_i$ to the candidate heavy hitters set.

The no-underestimation property ensures no heavy hitter is missed (no false negatives). False positives are bounded by the $\delta$ parameter.

### Range Queries

By building a Count-Min Sketch over a dyadic decomposition, range queries $\sum_{j=l}^{r} f_j$ can be answered with error $\epsilon n \log U$ using $O(\frac{\log U}{\epsilon} \log \frac{1}{\delta})$ space.

### Inner Product Estimation

Given two streams with frequency vectors $f$ and $g$, the inner product $\langle f, g \rangle = \sum_j f_j g_j$ can be estimated by maintaining two Count-Min Sketches and computing:

$$
\widehat{\langle f, g \rangle} = \min_{i=1}^{d} \sum_{k=1}^{w} C_f[i][k] \cdot C_g[i][k]
$$

## Mergeability

Count-Min Sketches are **mergeable**: given two sketches $S_1$ and $S_2$ built with the same hash functions, their element-wise sum $S_1 + S_2$ is a valid sketch for the combined stream. This property enables:

- **Distributed computation**: each node maintains a local sketch, and sketches are periodically merged at a central coordinator.
- **Parallel processing**: partition the stream across workers, sketch independently, and merge results.

## Comparison with Count Sketch

| Property | Count-Min Sketch | Count Sketch |
|---|---|---|
| Error type | One-sided ($\hat{f}_j \geq f_j$) | Two-sided |
| Error bound | $\epsilon n$ | $\epsilon \sqrt{F_2}$ |
| Space | $O(\frac{1}{\epsilon} \log \frac{1}{\delta})$ | $O(\frac{1}{\epsilon^2} \log \frac{1}{\delta})$ |
| Aggregation | Min across rows | Median across rows |
| Heavy hitters | No false negatives | Possible false negatives |

## Connection to Deep Learning

- **Feature hashing**: the hashing trick in machine learning maps features to a fixed-size vector using hash functions, directly paralleling the CMS update step. Count-Min Sketch analysis provides error bounds for feature hashing.
- **Gradient sketching**: in distributed deep learning, gradient vectors are compressed using sketches for communication-efficient aggregation.
- **Frequency-based vocabulary pruning**: language models use frequency estimates from sketches to determine which tokens to include in the vocabulary during preprocessing.

## Summary

The Count-Min Sketch provides space-efficient frequency estimation with one-sided error guarantees, using $O(\frac{1}{\epsilon} \log \frac{1}{\delta})$ space. Its simplicity, mergeability, and strong theoretical guarantees have made it one of the most practically important streaming data structures. The key insight is that taking the minimum across multiple independent hash projections controls the collision noise that inflates individual counter values.

## References

- [An Improved Data Stream Summary: The Count-Min Sketch (Cormode and Muthukrishnan, 2005)](https://doi.org/10.1016/j.jalgor.2003.12.001)
- [Data Streams: Algorithms and Applications (Muthukrishnan)](https://www.cs.rutgers.edu/~muthu/stream-1-1.ps)
