# Count-Min Sketch

Tracking exact frequencies of every element in a high-volume data stream requires space proportional to the number of distinct elements, which can be prohibitively large.  The **Count-Min sketch** (Cormode and Muthukrishnan, 2005) trades exact counts for approximate ones, using only $O(\frac{1}{\varepsilon} \log \frac{1}{\delta})$ space to guarantee that every frequency estimate overshoots the true count by at most $\varepsilon n$ with probability at least $1 - \delta$, where $n$ is the total number of elements seen.

## Data Structure

A Count-Min sketch consists of a two-dimensional array of counters $\text{CM}[1 \ldots d][1 \ldots w]$ initialized to zero, together with $d$ pairwise-independent hash functions $h_1, h_2, \ldots, h_d$, each mapping the universe $U$ to $\{1, 2, \ldots, w\}$.

The dimensions are chosen based on the desired accuracy and confidence:

$$
w = \left\lceil \frac{e}{\varepsilon} \right\rceil, \qquad d = \left\lceil \ln \frac{1}{\delta} \right\rceil
$$

where $e = 2.718\ldots$ is Euler's number, $\varepsilon$ controls the approximation error, and $\delta$ controls the failure probability.

## Operations

### Update

When element $x$ arrives in the stream, increment the counter in each row at the position determined by the corresponding hash function:

$$
\text{CM}[i][h_i(x)] \leftarrow \text{CM}[i][h_i(x)] + 1 \quad \text{for } i = 1, 2, \ldots, d
$$

Each update takes $O(d)$ time.

### Query

To estimate the frequency $\hat{f}_x$ of element $x$, take the minimum across all $d$ rows:

$$
\hat{f}_x = \min_{1 \leq i \leq d} \text{CM}[i][h_i(x)]
$$

The minimum operation is essential: each individual counter $\text{CM}[i][h_i(x)]$ may be inflated by collisions with other elements, but taking the minimum across independent hash functions reduces this inflation.

### Properties of the Estimate

The Count-Min sketch is a **one-sided estimator**: it never underestimates the true frequency.

**Theorem.** For any element $x$ with true frequency $f_x$:

$$
f_x \leq \hat{f}_x
$$

and

$$
\Pr[\hat{f}_x \leq f_x + \varepsilon n] \geq 1 - \delta
$$

where $n = \sum_x f_x$ is the total count of all elements.

*Proof sketch.* For each row $i$, element $x$ contributes exactly $f_x$ to counter $\text{CM}[i][h_i(x)]$, so $\text{CM}[i][h_i(x)] \geq f_x$.  The excess comes from other elements $y \neq x$ that collide with $x$ in row $i$, i.e., $h_i(y) = h_i(x)$.  By the pairwise independence of $h_i$ and Markov's inequality:

$$
\Pr[\text{CM}[i][h_i(x)] - f_x > \varepsilon n] \leq \frac{\mathbb{E}[\text{CM}[i][h_i(x)] - f_x]}{\varepsilon n} = \frac{(n - f_x)/w}{\varepsilon n} \leq \frac{1}{w \varepsilon} \leq \frac{1}{e}
$$

Since the $d$ rows use independent hash functions, the probability that *all* rows have excess greater than $\varepsilon n$ is at most $(1/e)^d \leq \delta$. $\square$

## Point Query vs Range Query

The basic Count-Min sketch answers **point queries** (frequency of a single element).  It can be extended to answer **range queries** by maintaining $\log U$ sketches at different granularities (a dyadic range decomposition), where $U$ is the universe size.

## Implementation

```python
"""Count-Min sketch for approximate frequency estimation in data streams."""

import hashlib
import math


# === Count-Min Sketch ===

class CountMinSketch:
    """Approximate frequency counter using sub-linear space."""

    def __init__(self, epsilon: float = 0.01, delta: float = 0.01):
        self.w = math.ceil(math.e / epsilon)
        self.d = math.ceil(math.log(1.0 / delta))
        self.table = [[0] * self.w for _ in range(self.d)]
        self.n = 0  # total count

    def _hash(self, x: str, i: int) -> int:
        """Hash element x for row i."""
        h = hashlib.md5(f"{i}:{x}".encode()).hexdigest()
        return int(h, 16) % self.w

    def update(self, x: str, count: int = 1) -> None:
        """Record count occurrences of element x."""
        self.n += count
        for i in range(self.d):
            self.table[i][self._hash(x, i)] += count

    def query(self, x: str) -> int:
        """Estimate the frequency of element x."""
        return min(self.table[i][self._hash(x, i)] for i in range(self.d))


# === Demonstration ===

if __name__ == "__main__":
    cms = CountMinSketch(epsilon=0.001, delta=0.01)

    # Simulate a stream with known frequencies
    frequencies = {"apple": 500, "banana": 300, "cherry": 100, "date": 50}
    for item, freq in frequencies.items():
        for _ in range(freq):
            cms.update(item)

    print(f"Sketch dimensions: {cms.d} rows x {cms.w} columns")
    print(f"Total elements: {cms.n}")
    print()
    for item, true_freq in frequencies.items():
        est = cms.query(item)
        print(f"{item}: true={true_freq}, estimate={est}, error={est - true_freq}")

    # Query for an absent element
    est_absent = cms.query("elderberry")
    print(f"elderberry (absent): estimate={est_absent}")
```

## Comparison with Other Sketches

| Data structure | Query type | Space | Error type |
|---|---|---|---|
| Count-Min sketch | Frequency | $O(\frac{1}{\varepsilon} \log \frac{1}{\delta})$ | One-sided (overestimate) |
| Count sketch | Frequency | $O(\frac{1}{\varepsilon^2} \log \frac{1}{\delta})$ | Two-sided (unbiased) |
| [Bloom filter](bloom.md) | Membership | $O(n \log \frac{1}{\delta})$ | One-sided (false positives) |
| [HyperLogLog](hyperloglog.md) | Cardinality | $O(\frac{1}{\varepsilon^2})$ | Two-sided |

The Count-Min sketch is preferred when one-sided error is acceptable (e.g., never underestimating a frequency is important for anomaly detection or heavy-hitter identification).

## Applications

- **Network traffic monitoring:** Estimate per-flow packet counts to detect heavy hitters (flows consuming disproportionate bandwidth).
- **Natural language processing:** Approximate word frequencies in large corpora without storing the full vocabulary.
- **Database query optimization:** Estimate join sizes and selectivities from streaming data.
- **Anomaly detection:** Flag elements whose estimated frequency exceeds a threshold.

## Reference

- Cormode, G., & Muthukrishnan, S. (2005). An improved data stream summary: The Count-Min sketch and its applications. *Journal of Algorithms*, 55(1), 58--75.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 11. MIT Press.
