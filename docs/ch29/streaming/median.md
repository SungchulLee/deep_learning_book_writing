# Approximate Median

Finding the exact median of a data stream requires storing all elements, which is infeasible for massive streams. **Approximate quantile algorithms** maintain compact summaries that answer rank queries with bounded error, returning an element whose rank is within $\epsilon n$ of the true median. These algorithms enable real-time percentile monitoring of latency distributions, training loss values, and other metrics that would be impractical to compute exactly over large-scale data.

## Problem Statement

Given a stream $\sigma = a_1, a_2, \ldots, a_n$ of comparable elements, the **$\phi$-quantile** (for $\phi \in [0, 1]$) is the element with rank $\lceil \phi n \rceil$ in the sorted order. The **median** is the $0.5$-quantile.

An **$\epsilon$-approximate $\phi$-quantile** is any element whose rank $r$ satisfies:

$$
(\phi - \epsilon) n \leq r \leq (\phi + \epsilon) n
$$

The goal is to answer such queries using space $o(n)$.

**Theorem (Munro-Paterson, 1980).** Any single-pass deterministic algorithm that computes the exact median requires $\Omega(n)$ space.

This lower bound motivates approximate solutions.

## The Greenwald-Khanna Algorithm

The **Greenwald-Khanna (GK) algorithm** (2001) maintains a summary of $O(\frac{1}{\epsilon} \log(\epsilon n))$ entries that supports $\epsilon$-approximate quantile queries for all $\phi$ simultaneously.

### Summary Structure

The summary is an ordered sequence of tuples $(v_i, g_i, \Delta_i)$ where:

- $v_i$: a stored value from the stream, with $v_1 < v_2 < \cdots < v_s$.
- $g_i$: the difference between the minimum possible rank of $v_i$ and the minimum possible rank of $v_{i-1}$.
- $\Delta_i$: the maximum uncertainty in the rank of $v_i$, satisfying $g_i + \Delta_i \leq 2\epsilon n$.

The rank of $v_i$ lies in the interval $[\text{rmin}(v_i), \text{rmax}(v_i)]$ where:

$$
\text{rmin}(v_i) = \sum_{j \leq i} g_j, \quad \text{rmax}(v_i) = \sum_{j \leq i} g_j + \Delta_i
$$

### Operations

**Insert**: when element $a$ arrives, insert a new tuple at the appropriate position in the sorted order. Periodically **compress** the summary by merging adjacent tuples whose combined uncertainty stays below $2\epsilon n$.

**Query**: to find the $\epsilon$-approximate $\phi$-quantile, find the smallest $i$ such that $\text{rmax}(v_i) \geq \phi n - \epsilon n$ and $\text{rmax}(v_i) + g_{i+1} + \Delta_{i+1} > \phi n + \epsilon n$.

**Space**: $O(\frac{1}{\epsilon} \log(\epsilon n))$ tuples.

## Random Sampling Approach

A simpler approach uses **reservoir sampling** to maintain a sample of size $s$:

1. Maintain a uniform random sample of $s$ elements using reservoir sampling.
2. To answer a $\phi$-quantile query, sort the sample and return the element at position $\lceil \phi s \rceil$.

By the Dvoretzky-Kiefer-Wolfowitz (DKW) inequality, a sample of size:

$$
s = O\left(\frac{1}{\epsilon^2} \log \frac{1}{\delta}\right)
$$

guarantees that all quantile estimates are $\epsilon$-approximate with probability $1 - \delta$.

!!! tip "When to Use Which"
    The GK algorithm provides deterministic guarantees with $O(\frac{1}{\epsilon} \log(\epsilon n))$ space, while random sampling gives probabilistic guarantees with $O(\frac{1}{\epsilon^2})$ space (independent of $n$). For small $\epsilon$, random sampling can be more space-efficient; for streaming updates with many queries, GK is more flexible.

## The t-Digest

The **t-digest** (Dunning, 2019) is a practical approximate quantile data structure that uses cluster centroids to represent the distribution:

- The stream is partitioned into clusters, each summarized by its centroid and count.
- Clusters near the tails ($\phi$ close to 0 or 1) are kept small for high accuracy, while clusters near the median can be larger.
- A scale function $k(\phi)$ controls the maximum cluster size at quantile $\phi$:

$$
k(\phi) = \frac{1}{2\pi} \arcsin(2\phi - 1)
$$

The t-digest uses $O(1/\epsilon)$ clusters and is **mergeable**, making it suitable for distributed systems.

## Two-Heap Approach for Exact Streaming Median

For the special case of maintaining the exact median over a stream (when memory permits), the **two-heap** approach uses a max-heap for the lower half and a min-heap for the upper half:

- The max-heap stores elements $\leq$ median; the min-heap stores elements $>$ median.
- Balance the heaps so their sizes differ by at most 1.
- The median is the top of the larger heap (or the average of both tops if equal size).

This requires $O(n)$ space but provides exact answers with $O(\log n)$ update time.

## Biased Quantiles

Some applications require high accuracy for extreme quantiles (e.g., 99th percentile latency) but tolerate lower accuracy for the median. **Biased quantile algorithms** achieve error $\epsilon \phi n$ instead of $\epsilon n$:

$$
|\hat{r} - \phi n| \leq \epsilon \phi n
$$

This provides relative rather than absolute error, using $O(\frac{1}{\epsilon} \log^2(\epsilon n))$ space.

## Connection to Deep Learning

- **Training monitoring**: tracking percentiles of loss values, gradient norms, or activation magnitudes over millions of training steps uses approximate quantile algorithms.
- **Latency profiling**: in model serving, monitoring the p50, p95, and p99 latency of inference requests uses streaming quantile estimation.
- **Batch normalization statistics**: maintaining running statistics of activation distributions across mini-batches is a form of streaming quantile computation.

## Summary

Approximate median and quantile estimation in the streaming model requires $o(n)$ space, with the Greenwald-Khanna algorithm achieving $O(\frac{1}{\epsilon} \log(\epsilon n))$ deterministic space and random sampling achieving $O(\frac{1}{\epsilon^2})$ probabilistic space. The t-digest provides a practical, mergeable alternative. These algorithms enable real-time percentile monitoring over massive data streams where storing all elements is infeasible.

## References

- [Space-Efficient Online Computation of Quantile Summaries (Greenwald and Khanna, 2001)](https://doi.org/10.1145/375663.375670)
- [Data Streams: Algorithms and Applications (Muthukrishnan)](https://www.cs.rutgers.edu/~muthu/stream-1-1.ps)
