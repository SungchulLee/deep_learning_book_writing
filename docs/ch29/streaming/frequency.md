# Frequency Estimation

Given a stream of elements, how often does a particular item appear? Answering this **point query** exactly requires storing the entire frequency vector, which may be prohibitively large. Frequency estimation algorithms maintain compact summaries that answer point queries approximately, with provable error guarantees. These algorithms form the backbone of streaming analytics, powering applications from network monitoring to natural language processing.

## Problem Statement

Given a data stream $\sigma = a_1, a_2, \ldots, a_n$ where each $a_i \in [U] = \{1, 2, \ldots, U\}$, define the **frequency** of item $j$ as:

$$
f_j = |\{i : a_i = j\}|
$$

The goal is to maintain a data structure of size $o(U)$ that, given a query item $j$, returns an estimate $\hat{f}_j$ satisfying:

$$
|\hat{f}_j - f_j| \leq \epsilon n
$$

with probability at least $1 - \delta$, where $\epsilon$ is the accuracy parameter and $\delta$ is the failure probability.

## Frequency Moments

Frequency estimation is closely related to **frequency moments**. The $k$-th frequency moment is:

$$
F_k = \sum_{j=1}^{U} f_j^k
$$

Important special cases:

- $F_0$: the number of **distinct elements** in the stream
- $F_1 = n$: the **stream length**
- $F_2 = \sum_j f_j^2$: the **self-join size** or sum of squared frequencies, measuring how skewed the frequency distribution is

!!! note "Why $F_2$ Matters"
    $F_2$ quantifies the "surprise" of the frequency distribution. If all items appear equally, $F_2 = n^2/U$. If one item dominates, $F_2 \approx n^2$. The ratio $F_2 / F_1^2$ is sometimes called the **repeat rate** and measures the predictability of the stream.

## Approaches to Frequency Estimation

### Exact Counting

Maintaining a hash table with exact counts requires $O(d)$ space where $d$ is the number of distinct elements. When $d$ is large (e.g., IP addresses, n-grams), this is impractical.

### Count-Min Sketch

The **Count-Min Sketch** (Cormode and Muthukrishnan, 2005) uses $w \times d$ counters with $d$ independent hash functions. Each hash function maps items to one of $w$ counters:

$$
\hat{f}_j = \min_{i=1}^{d} C[i][h_i(j)]
$$

Space: $O\left(\frac{e}{\epsilon} \cdot \ln \frac{1}{\delta}\right)$ with $w = \lceil e/\epsilon \rceil$ and $d = \lceil \ln(1/\delta) \rceil$.

The estimate always satisfies $\hat{f}_j \geq f_j$ (no underestimation) and:

$$
P(\hat{f}_j - f_j > \epsilon n) < \delta
$$

### Count Sketch

The **Count Sketch** (Charikar, Chen, and Farach-Colton, 2004) uses sign functions $s_i : [U] \to \{-1, +1\}$ in addition to hash functions. Updates add $s_i(a)$ to the counter, and the estimate is the **median** across rows:

$$
\hat{f}_j = \text{median}_{i=1}^{d} \; s_i(j) \cdot C[i][h_i(j)]
$$

The Count Sketch provides a two-sided error guarantee:

$$
P(|\hat{f}_j - f_j| > \epsilon \sqrt{F_2}) < \delta
$$

This is stronger than Count-Min Sketch for skewed distributions because $\sqrt{F_2} \leq n$.

### Comparison

| Property | Count-Min Sketch | Count Sketch |
|---|---|---|
| Error bound | $\epsilon n$ | $\epsilon \sqrt{F_2}$ |
| Error type | One-sided (overestimate) | Two-sided |
| Space | $O(\frac{1}{\epsilon} \log \frac{1}{\delta})$ | $O(\frac{1}{\epsilon^2} \log \frac{1}{\delta})$ |
| Best for | Heavy hitters, point queries | Skewed distributions |

## The AMS Sketch for $F_2$

The Alon-Matias-Szegedy (AMS) sketch estimates $F_2$ using random projections. Maintain a counter $Z$ using a 4-wise independent hash function $h : [U] \to \{-1, +1\}$:

$$
Z = \sum_{i=1}^{n} h(a_i)
$$

Then $\mathbb{E}[Z^2] = F_2$. By maintaining $O(1/\epsilon^2)$ independent copies and taking the mean, the estimate achieves:

$$
P(|\hat{F}_2 - F_2| > \epsilon F_2) < \delta
$$

using $O(\frac{1}{\epsilon^2} \log \frac{1}{\delta})$ space.

## Heavy Hitters

An item $j$ is an **$\epsilon$-heavy hitter** if $f_j > \epsilon n$. Finding all heavy hitters is a fundamental streaming problem:

- **Misra-Gries** finds all $\epsilon$-heavy hitters using $O(1/\epsilon)$ counters
- **Count-Min Sketch** with threshold $\epsilon n$ identifies heavy hitters with no false negatives
- At most $1/\epsilon$ items can be $\epsilon$-heavy hitters (since $\sum_j f_j = n$)

## Connection to Deep Learning

Frequency estimation techniques are used in deep learning systems:

- **Feature hashing**: maps high-dimensional sparse features to a fixed-size vector using hash functions, directly analogous to Count-Min Sketch
- **Gradient compression**: in distributed training, gradient sketches compress communication by keeping only the most significant (heavy hitter) gradient components
- **Vocabulary management**: tracking token frequencies in language model training uses streaming frequency estimation over corpora too large to fit in memory

## Summary

Frequency estimation in the streaming model trades exact counts for compact approximate representations. The Count-Min Sketch and Count Sketch provide different tradeoffs between space, error guarantees, and distribution sensitivity. The AMS sketch extends these ideas to estimate frequency moments, particularly $F_2$. Together, these tools enable efficient processing of massive data streams with provable accuracy guarantees.

## References

- [Data Streams: Algorithms and Applications (Muthukrishnan)](https://www.cs.rutgers.edu/~muthu/stream-1-1.ps)
- [An Improved Data Stream Summary: The Count-Min Sketch (Cormode and Muthukrishnan, 2005)](https://doi.org/10.1016/j.jalgor.2003.12.001)
