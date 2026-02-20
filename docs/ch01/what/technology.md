# Algorithms as Technology

Algorithms are a **technology** — like hardware, networking, or machine learning. The choice of algorithm can make the difference between solving a problem in seconds or not at all.

$$
\text{Total performance} = \text{Hardware speed} \times \text{Algorithm efficiency}
$$

## Hardware vs Algorithms

A faster algorithm on a slower computer often beats a slower algorithm on a faster computer.

**Example**: Insertion sort ($O(n^2)$) on a supercomputer vs merge sort ($O(n \log n)$) on a laptop:

$$
\text{For } n = 10^7: \quad \frac{n^2}{n \log n} = \frac{10^{14}}{10^7 \times 23} \approx 4.3 \times 10^5
$$

The algorithm advantage grows with input size, eventually dominating any constant-factor hardware advantage.


# Reference

[Introduction to Algorithms (CLRS), Section 1.1](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
