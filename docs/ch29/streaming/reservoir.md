# Reservoir Sampling

When data arrives as a stream of unknown length, maintaining a uniform random sample is non-trivial: we cannot simply pick elements with probability $k/n$ because $n$ is unknown in advance. **Reservoir sampling** solves this problem elegantly, maintaining a sample of exactly $k$ elements such that at every point in the stream, each element seen so far has equal probability of being in the sample. This technique is fundamental to streaming algorithms and has direct applications in data loading for machine learning.

## Problem Statement

Given a stream $\sigma = a_1, a_2, \ldots$ of unknown length, maintain a **reservoir** $R$ of size $k$ such that after processing $n$ elements, each element $a_i$ for $i \leq n$ is in $R$ with probability exactly $k/n$.

The key challenge is that $n$ is not known in advance, so the inclusion probability must be maintained dynamically as new elements arrive.

## Algorithm R (Vitter, 1985)

The classic algorithm, due to Vitter, processes each element as follows:

1. **Initialization**: for $i = 1, 2, \ldots, k$, place $a_i$ directly into $R[i]$.
2. **Sampling phase**: for each subsequent element $a_i$ (where $i > k$):
    - Generate a random integer $j$ uniformly from $\{1, 2, \ldots, i\}$.
    - If $j \leq k$, replace $R[j]$ with $a_i$.
    - Otherwise, discard $a_i$.

!!! note "Space Complexity"
    The algorithm uses $O(k)$ space for the reservoir plus $O(1)$ additional space for the counter and random number generation.

## Correctness Proof

**Theorem.** After processing $n$ elements, each element $a_i$ (for $i \leq n$) is in the reservoir with probability exactly $k/n$.

*Proof by induction on $n$.*

**Base case**: $n = k$. All $k$ elements are in the reservoir, each with probability $k/k = 1$.

**Inductive step**: assume after processing $n - 1$ elements, each is in $R$ with probability $k/(n-1)$. When element $a_n$ arrives:

- $a_n$ is included with probability $k/n$ (since $j \leq k$ with probability $k/n$).
- For any previous element $a_i$ (with $i < n$), it remains in $R$ if it was in $R$ after step $n - 1$ **and** it is not replaced by $a_n$:

$$
P(a_i \in R \text{ after step } n) = \frac{k}{n-1} \cdot \left(1 - \frac{1}{n}\right) = \frac{k}{n-1} \cdot \frac{n-1}{n} = \frac{k}{n}
$$

The second factor accounts for the probability that, if $a_n$ replaces an element, it does not replace $a_i$ specifically: it replaces $a_i$ with probability $1/k \cdot k/n = 1/n$, so it does not replace $a_i$ with probability $(n-1)/n$.  $\square$

## Weighted Reservoir Sampling

When elements have weights $w_i$ and we want the inclusion probability proportional to the weight, the **A-Res** (Algorithm with Reservoir) method works as follows:

1. For each element $a_i$ with weight $w_i$, compute a key $k_i = u_i^{1/w_i}$ where $u_i \sim \text{Uniform}(0, 1)$.
2. Maintain the $k$ elements with the largest keys.

**Theorem (Efraimidis and Spirakis, 2006).** A-Res produces a weighted random sample without replacement, where each element's inclusion probability is proportional to its weight.

## Optimized Variants

### Algorithm L (Vitter, 1985)

Algorithm R generates a random number for every element in the stream, which is wasteful when $n \gg k$. Algorithm L computes the **gap** (number of elements to skip) directly:

1. After including the current element, generate the next gap $G$:

$$
G = \left\lfloor \frac{\ln(U)}{\ln(1 - k/n)} \right\rfloor
$$

where $U \sim \text{Uniform}(0, 1)$ and $n$ is the current stream position.

2. Skip $G$ elements, then replace a random reservoir element with the next element.

This reduces the expected number of random numbers generated from $O(n)$ to $O(k(1 + \ln(n/k)))$.

### Merge-Based Reservoir Sampling

For distributed streams, each node maintains a local reservoir. Merging two reservoirs of size $k$ from streams of lengths $n_1$ and $n_2$ produces a valid reservoir for the combined stream by:

1. Combine both reservoirs into a pool of $2k$ candidates.
2. Sample $k$ elements from the pool, where elements from stream $i$ are included with probability proportional to $n_i$.

## Applications

### Streaming Data Analysis

When the full dataset cannot fit in memory, reservoir sampling provides a representative sample for:

- Estimating statistics (mean, variance, quantiles)
- Training approximate models
- Generating visualizations

### Experience Replay in Reinforcement Learning

In reinforcement learning, the **experience replay buffer** stores transitions $(s, a, r, s')$ sampled from the agent's interactions. Reservoir sampling provides a principled way to maintain a uniform sample from the entire training history when the buffer has a fixed size.

### Data Loading for Deep Learning

- **Shuffled sampling**: reservoir sampling can produce a shuffled subset of training data when the dataset is too large to shuffle in memory.
- **Active learning**: when selecting data points for labeling from a stream, reservoir sampling ensures unbiased candidate selection.

## Summary

Reservoir sampling maintains a uniform random sample of $k$ elements from a stream of unknown length using $O(k)$ space. The inclusion probability $k/n$ is maintained exactly as new elements arrive. Weighted variants handle non-uniform sampling, and gap-based optimizations reduce the number of random number generations. The algorithm's simplicity and strong theoretical guarantees make it a cornerstone of streaming computation.

## References

- [Random Sampling with a Reservoir (Vitter, 1985)](https://doi.org/10.1145/3147.3165)
- [Data Streams: Algorithms and Applications (Muthukrishnan)](https://www.cs.rutgers.edu/~muthu/stream-1-1.ps)
