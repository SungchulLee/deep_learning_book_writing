# Algorithms as Technology

When building software systems, engineers routinely make technology choices -- which database to use, how much memory to provision, what hardware to deploy. Algorithm selection belongs in the same category. A well-chosen algorithm can transform an intractable computation into one that finishes in seconds, often delivering gains that no amount of hardware can match.

## Hardware vs Algorithms

A faster algorithm on a slower computer often beats a slower algorithm on a faster computer. The following comparison illustrates why.

Consider sorting $n = 10^7$ elements. Insertion sort runs in $O(n^2)$ time, while merge sort runs in $O(n \log n)$ time. Even if insertion sort executes on a supercomputer performing $10^{10}$ operations per second and merge sort runs on a laptop performing $10^7$ operations per second, the wall-clock times are:

- **Insertion sort on supercomputer**: $\dfrac{(10^7)^2}{10^{10}} = 10{,}000$ seconds (nearly 3 hours)
- **Merge sort on laptop**: $\dfrac{10^7 \times 23}{10^7} = 23$ seconds

The ratio of operations is

$$
\frac{n^2}{n \log_2 n} = \frac{10^{14}}{10^7 \times 23} \approx 4.3 \times 10^5
$$

Merge sort performs roughly 430,000 times fewer operations. As datasets scale from thousands to millions to billions of entries, the gap between an $O(n^2)$ and an $O(n \log n)$ algorithm widens dramatically -- making algorithmic efficiency one of the highest-leverage investments in system design.

!!! tip "The Takeaway"
    When input sizes are large, the **growth rate** of an algorithm's running time matters far more than the constant factors or the speed of the hardware. This is why asymptotic analysis ($O$-notation) is the primary tool for comparing algorithms.

## References

[Introduction to Algorithms (CLRS), Section 1.1](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
