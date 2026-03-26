# Reservoir Sampling

In many applications — processing log streams, analyzing network traffic,
or sampling from a database query — data arrives one element at a time and
the total size is unknown in advance. We need to maintain a random sample
of exactly $k$ elements from the stream, using only $O(k)$ memory.
**Reservoir sampling** solves this problem: it guarantees that at every
point in the stream, each element seen so far has equal probability of
being in the sample.

## Algorithm R (Vitter, 1985)

Maintain a reservoir array $R[1 \ldots k]$. Process the stream elements $s_1, s_2, \ldots$ one at a time:

1. **Initialization**: Place the first $k$ elements into the reservoir: $R[i] = s_i$ for $i = 1, \ldots, k$.
2. **Streaming phase**: For each subsequent element $s_j$ (where $j > k$):
    - Generate a random integer $r$ uniformly from $\{1, 2, \ldots, j\}$.
    - If $r \leq k$, replace $R[r]$ with $s_j$. Otherwise, discard $s_j$.

## Correctness Proof

**Claim.** After processing $n$ elements, each element $s_i$ (for $i = 1, \ldots, n$) is in the reservoir with probability exactly $k/n$.

*Proof by induction on $n$.*

**Base case** ($n = k$): All $k$ elements are in the reservoir, each with probability $k/k = 1$.

**Inductive step**: Assume after processing $n - 1$ elements, each is in the reservoir with probability $k/(n-1)$. When element $s_n$ arrives:

- $s_n$ enters the reservoir with probability $k/n$ (it replaces a random position if $r \leq k$, which happens with probability $k/n$).

- For any element $s_i$ already in the reservoir, it survives if $s_n$ does not replace it. The probability that $s_i$ is replaced is $(k/n) \cdot (1/k) = 1/n$. So $s_i$ survives with probability $1 - 1/n = (n-1)/n$.

- The probability that $s_i$ is in the reservoir after step $n$ is

$$
\frac{k}{n-1} \cdot \frac{n-1}{n} = \frac{k}{n}
$$

$\square$

## Implementation

```python
"""
Reservoir sampling: maintain a uniform random sample of k elements
from a stream of unknown length.
"""

import random

# === Reservoir Sampling ===

def reservoir_sample(stream, k):
    """Return a list of k elements sampled uniformly from the stream."""
    reservoir = []
    for i, element in enumerate(stream):
        if i < k:
            reservoir.append(element)
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = element
    return reservoir

# === Main ===

if __name__ == "__main__":
    random.seed(42)
    stream = range(1, 101)  # Stream of 1 to 100
    sample = reservoir_sample(stream, 5)
    print(f"Sample of 5 from 1..100: {sample}")

    # Verify uniformity: each element should appear ~k/n fraction of time
    from collections import Counter
    counts = Counter()
    trials = 100000
    for _ in range(trials):
        for elem in reservoir_sample(range(10), 3):
            counts[elem] += 1
    print("\nUniformity test (k=3, n=10, expected freq=0.300):")
    for elem in sorted(counts):
        freq = counts[elem] / trials
        print(f"  Element {elem}: freq={freq:.3f}")
```

**Output:**
```
Sample of 5 from 1..100: [68, 99, 80, 58, 87]
```

(Output varies across runs due to randomness.)

## Time and Space Complexity

| Operation | Complexity |
|---|---|
| Time per element | $O(1)$ |
| Total time for $n$ elements | $O(n)$ |
| Space | $O(k)$ |

The algorithm makes a single pass over the stream, requires no knowledge of $n$ in advance, and uses only $O(k)$ memory regardless of stream length.

## Weighted Reservoir Sampling

When elements have non-uniform weights $w_i$, we want element $s_i$ to appear in the sample with probability proportional to $w_i$. The **Efraimidis-Spirakis algorithm** achieves this:

1. For each element $s_i$, compute a key $u_i^{1/w_i}$ where $u_i \sim \text{Uniform}(0, 1)$.
2. Keep the $k$ elements with the largest keys.

This can be implemented in a single pass using a min-heap of size $k$.

!!! tip "Practical Considerations"
    In practice, generating a random number for every stream element is expensive for large streams. Vitter's Algorithm Z computes the number of elements to skip between replacements, reducing the expected number of random variates from $n$ to $O(k(1 + \log(n/k)))$.

## Applications

- **Database sampling**: SELECT queries on large tables without knowing the row count.
- **Stream analytics**: Maintaining representative samples of network packets or log entries.
- **Machine learning**: Stochastic gradient descent with uniform mini-batch sampling from a data stream.
- **A/B testing**: Randomly assigning users to treatment groups in a streaming setting.

## Reference

- Vitter, J. S. "Random Sampling with a Reservoir." *ACM Transactions on Mathematical Software*, 1985.
- Efraimidis, P. S. & Spirakis, P. G. "Weighted Random Sampling with a Reservoir." *Information Processing Letters*, 2006.
