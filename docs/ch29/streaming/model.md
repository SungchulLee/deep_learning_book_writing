# Streaming Model

Modern data systems routinely process volumes far too large to store in memory: network traffic logs generating terabytes per day, social media feeds producing millions of events per second, and sensor networks streaming continuous measurements. The **streaming model** of computation formalizes algorithm design for this setting, requiring that data elements are processed in a single pass (or a small number of passes) using memory that is sublinear in the input size. This model fundamentally changes what is computable, forcing the use of approximate answers and randomized data structures.

## Formal Definition

A **data stream** is a sequence $\sigma = a_1, a_2, \ldots, a_n$ of elements drawn from a universe $[U] = \{1, 2, \ldots, U\}$. A streaming algorithm processes elements one at a time in order, maintaining a **summary** (or **sketch**) $S$ in working memory of size:

$$
\text{Space}(S) = O(\text{polylog}(n, U)) \quad \text{or} \quad O(n^\epsilon) \text{ for some } \epsilon < 1
$$

The algorithm cannot revisit past elements. After processing the entire stream, it must answer queries about the stream using only the summary $S$.

!!! note "Why Sublinear Memory?"
    If the algorithm could store the entire stream, any query could be answered exactly. The streaming model's constraint of sublinear memory forces algorithms to make tradeoffs between space, accuracy, and the number of passes over the data.

## Stream Types

### Cash Register Model

Each element $a_i$ represents an **insertion**: the frequency $f_j$ of item $j$ increases by 1 (or by a positive weight $w_i$). All frequencies are non-negative:

$$
f_j = \sum_{i : a_i = j} w_i \geq 0
$$

This models counting occurrences of events (e.g., packet counts, word frequencies).

### Turnstile Model

Each element $a_i = (j, \Delta)$ represents an **update**: item $j$'s frequency changes by $\Delta \in \mathbb{Z}$ (which can be negative). The frequency vector is:

$$
f_j = \sum_{i : a_i = (j, \Delta_i)} \Delta_i
$$

In the **strict turnstile model**, $f_j \geq 0$ at all times. In the **general turnstile model**, $f_j$ can be negative. The turnstile model captures settings where deletions occur (e.g., items leaving a window, corrections to counts).

### Sliding Window Model

Only the most recent $W$ elements matter. The algorithm must answer queries about the substream $a_{n-W+1}, \ldots, a_n$. This models time-decaying data where old observations become irrelevant.

## Fundamental Limitations

The streaming model imposes inherent limitations on what can be computed exactly.

### Communication Complexity Lower Bounds

Lower bounds on streaming space come from **communication complexity**. If computing a function $f$ on a stream requires distinguishing between many possible frequency vectors, the algorithm needs enough memory to encode these distinctions.

**Theorem.** Any deterministic algorithm that computes the number of distinct elements in a stream exactly requires $\Omega(n)$ space.

*Proof sketch.* Consider a stream where each element is distinct. After processing $k$ elements, the algorithm must distinguish $\binom{U}{k}$ possible sets to determine whether the next element is new. This requires $\Omega(k \log U)$ bits of state. $\square$

This impossibility result motivates approximate solutions like HyperLogLog, which estimates the distinct count using only $O(\log \log U + \log(1/\delta))$ space.

### The Space-Accuracy Tradeoff

For most streaming problems, an $(\epsilon, \delta)$-approximation (multiplicative error $\epsilon$ with failure probability $\delta$) can be achieved with space:

$$
O\left(\frac{1}{\epsilon^2} \log \frac{1}{\delta}\right)
$$

This tradeoff is often tight: halving the error requires quadrupling the space.

## Query Types

Common queries answered by streaming algorithms:

| Query | Example Algorithm | Space |
|---|---|---|
| Point query: $f_j$ | Count-Min Sketch | $O(\frac{1}{\epsilon} \log \frac{1}{\delta})$ |
| Distinct count: $|\{j : f_j > 0\}|$ | HyperLogLog | $O(\frac{1}{\epsilon^2} + \log U)$ |
| Frequency moments: $F_k = \sum_j f_j^k$ | AMS Sketch | $O(\frac{1}{\epsilon^2} \log U)$ |
| Heavy hitters: $\{j : f_j > \epsilon n\}$ | Misra-Gries | $O(\frac{1}{\epsilon})$ |
| Quantiles | Greenwald-Khanna | $O(\frac{1}{\epsilon} \log(\epsilon n))$ |
| Random sample | Reservoir Sampling | $O(k)$ for $k$ samples |

## Multi-Pass Streaming

Some algorithms make a small number of passes over the data, using the sketch from one pass to guide the next:

- **2-pass algorithms**: can achieve exact answers for some problems where single-pass requires approximation.
- **Semi-streaming**: allows $O(n \cdot \text{polylog}(n))$ space, useful for graph problems where the stream consists of edges.

## Connection to Deep Learning

The streaming model connects to deep learning in several ways:

- **Online training**: stochastic gradient descent processes training examples in a streaming fashion, maintaining model parameters (the "sketch") in memory far smaller than the full dataset.
- **Feature hashing**: the hashing trick maps high-dimensional feature spaces to fixed-size representations, analogous to streaming sketches.
- **Monitoring training metrics**: tracking running statistics (loss, gradient norms, activation distributions) over millions of training steps uses streaming algorithms to maintain summaries without storing all historical values.

## Summary

The streaming model formalizes computation under severe memory constraints, processing data in one or few passes using sublinear space. The cash register, turnstile, and sliding window models capture different update semantics. Communication complexity lower bounds establish fundamental space requirements, while randomized sketching algorithms achieve near-optimal space-accuracy tradeoffs for a wide range of queries.

## References

- [Data Streams: Algorithms and Applications (Muthukrishnan)](https://www.cs.rutgers.edu/~muthu/stream-1-1.ps)
- [Sketch Techniques for Approximate Query Processing (Cormode)](https://doi.org/10.1561/0400000060)
