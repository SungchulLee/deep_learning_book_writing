# Paging

Every computer system with a cache faces the same fundamental question: when the cache is full and a new item must be loaded, which existing item should be evicted? This is the **online paging problem**, one of the most studied problems in competitive analysis. It connects directly to CPU cache management, web proxy caching, and GPU memory management in deep learning workloads, making it a cornerstone example of online algorithms with immediate practical relevance.

## Problem Formulation

A cache holds at most $k$ pages from a universe of $N > k$ pages. A request sequence $\sigma = p_1, p_2, \ldots, p_n$ specifies pages to access:

- **Cache hit**: page $p_i$ is already in the cache; cost is 0.
- **Cache miss (page fault)**: page $p_i$ is not in the cache; it must be loaded (cost 1), and if the cache is full, one existing page must be evicted.

The goal is to minimize the total number of page faults. The online algorithm must decide which page to evict without knowing future requests.

## Common Eviction Strategies

### LRU (Least Recently Used)

Evict the page whose most recent access is farthest in the past. LRU exploits temporal locality: recently used pages are likely to be used again soon.

### FIFO (First In, First Out)

Evict the page that has been in the cache the longest, regardless of access pattern.

### LFU (Least Frequently Used)

Evict the page with the fewest accesses. LFU adapts to frequency-based access patterns but can be slow to respond to distribution changes.

### Optimal Offline (Belady's Algorithm)

Evict the page whose next request is farthest in the future. This requires complete knowledge of the future request sequence and serves as the offline optimum $\text{OPT}$.

## Deterministic Competitive Ratio

!!! note "LRU and FIFO Are $k$-Competitive"
    Both LRU and FIFO achieve a competitive ratio of exactly $k$, where $k$ is the cache size. This is the best possible for any deterministic online paging algorithm.

**Theorem.** LRU is $k$-competitive for the paging problem.

*Proof sketch.* Partition the request sequence into **phases**. A new phase begins whenever LRU incurs its $(k+1)$-th distinct page request since the last phase boundary. Within each phase, LRU faults on at most $k$ pages, while OPT must fault on at least 1 page (since the phase contains $k + 1$ distinct pages but OPT's cache holds only $k$). Therefore:

$$
\frac{\text{faults}_{\text{LRU}}}{\text{faults}_{\text{OPT}}} \leq \frac{k \cdot (\text{number of phases})}{1 \cdot (\text{number of phases})} = k
$$

**Theorem (Lower Bound).** No deterministic online paging algorithm can achieve a competitive ratio better than $k$.

*Proof sketch.* An adversary maintains a set of $k + 1$ pages and always requests the page not currently in the algorithm's cache. The algorithm faults on every request, while OPT (with $k$ pages) faults at most once per $k$ requests. $\square$

## Randomized Competitive Ratio

Randomization dramatically improves the competitive ratio. Against an oblivious adversary, the optimal randomized competitive ratio is $H_k$, the $k$-th harmonic number:

$$
H_k = \sum_{i=1}^{k} \frac{1}{i} = \Theta(\ln k)
$$

### The Marker Algorithm

The **Marker algorithm** achieves the optimal randomized competitive ratio of $H_k$:

1. Mark all pages as **unmarked** at the start of each phase.
2. On a cache hit, mark the accessed page.
3. On a cache miss:
    - If all pages in the cache are marked, start a new phase: unmark all pages.
    - Evict a page chosen **uniformly at random** from the unmarked pages in the cache.
    - Load the requested page and mark it.

**Theorem.** The Marker algorithm is $H_k$-competitive against an oblivious adversary.

The gap between $k$ (deterministic) and $H_k \approx \ln k$ (randomized) demonstrates the substantial power of randomization in online computation.

## The $h$-$k$ Paging Problem

A generalization allows the online algorithm a cache of size $k$ while OPT uses a smaller cache of size $h \leq k$. This is called **resource augmentation**. With this advantage:

$$
\text{LRU is } \frac{k}{k - h + 1}\text{-competitive}
$$

When $k = 2h$, LRU becomes 2-competitive, showing that a constant-factor larger cache eliminates most of the disadvantage of being online.

!!! tip "Resource Augmentation in Practice"
    Resource augmentation is often a more realistic model: in practice, caches can be made somewhat larger than strictly necessary. The $h$-$k$ framework shows that modest extra resources dramatically reduce the competitive ratio.

## Working Set and Locality

Real-world request sequences exhibit **locality of reference**: recent pages are likely to be requested again. The **working set** at time $t$ consists of the $w(t)$ distinct pages accessed since the last request to the current page.

LRU is particularly well-suited to sequences with locality because it automatically adapts its cache contents to the working set. When the working set size $w$ satisfies $w \leq k$, LRU incurs no faults.

## Connection to Deep Learning

Paging algorithms are directly relevant to deep learning systems:

- **GPU memory management**: deep learning frameworks like PyTorch and TensorFlow use caching allocators that decide which tensors to keep in GPU memory. The eviction policy directly affects training throughput.
- **Gradient checkpointing**: during backpropagation, intermediate activations can be recomputed rather than stored, creating a paging-like tradeoff between memory and computation.
- **Data loading**: prefetching training batches from disk into memory is an online caching problem where the access pattern follows the training data order.

## Summary

The paging problem demonstrates both the power and limitations of online algorithms. Deterministic algorithms like LRU and FIFO achieve the optimal ratio of $k$, while randomized algorithms achieve $H_k \approx \ln k$, an exponential improvement. Resource augmentation further reduces the ratio, providing a practical bridge between theoretical bounds and real-world system design.

## References

- [Online Computation and Competitive Analysis (Borodin and El-Yaniv)](https://www.amazon.com/dp/0521619467)
- [Data Streams: Algorithms and Applications (Muthukrishnan)](https://www.cs.rutgers.edu/~muthu/stream-1-1.ps)
