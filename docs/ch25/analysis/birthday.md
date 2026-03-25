# Birthday Paradox

How many people must be in a room before there is a better-than-even chance that two share the same birthday? The answer — approximately 23 — is surprisingly small compared to the 365 possible birthdays. The **birthday paradox** formalizes this phenomenon: when $n$ items are drawn uniformly at random from $m$ categories, a collision (two items in the same category) occurs after only $\Theta(\sqrt{m})$ draws. This result is central to hash table analysis, cryptographic security, and the study of random mappings.

## Exact Probability

Suppose $n$ people have birthdays chosen independently and uniformly from $\{1, 2, \ldots, m\}$ (with $m = 365$ for the classic problem). The probability that all $n$ birthdays are distinct is

$$
\Pr[\text{no collision}] = \frac{m}{m} \cdot \frac{m-1}{m} \cdot \frac{m-2}{m} \cdots \frac{m-n+1}{m} = \prod_{k=0}^{n-1} \left(1 - \frac{k}{m}\right)
$$

The probability of at least one collision is

$$
\Pr[\text{collision}] = 1 - \prod_{k=0}^{n-1} \left(1 - \frac{k}{m}\right)
$$

## Asymptotic Analysis

Using the approximation $1 - x \leq e^{-x}$ for $x \geq 0$,

$$
\Pr[\text{no collision}] \leq \prod_{k=0}^{n-1} e^{-k/m} = e^{-\sum_{k=0}^{n-1} k/m} = e^{-n(n-1)/(2m)}
$$

The collision probability exceeds $1/2$ when $e^{-n(n-1)/(2m)} \leq 1/2$, which gives

$$
n(n-1) \geq 2m \ln 2
$$

Solving for $n$,

$$
n \approx \sqrt{2m \ln 2} = \Theta(\sqrt{m})
$$

For $m = 365$, this yields $n \approx \sqrt{2 \cdot 365 \cdot 0.693} \approx 22.5$, confirming that 23 people suffice.

## Indicator Variable Proof

The expected number of colliding pairs provides an alternative approach. Define $X_{ij} = I\{\text{person } i \text{ and person } j \text{ share a birthday}\}$ for $1 \leq i < j \leq n$. Since each pair collides with probability $1/m$,

$$
E\left[\sum_{i < j} X_{ij}\right] = \binom{n}{2} \cdot \frac{1}{m} = \frac{n(n-1)}{2m}
$$

When $n = \Theta(\sqrt{m})$, the expected number of collisions is $\Theta(1)$, which is consistent with the collision probability becoming non-negligible.

!!! tip "The Square-Root Rule"
    The birthday paradox establishes a general principle: in a space of size $m$, expect collisions after $O(\sqrt{m})$ random samples. This applies to hash functions, random number generators, and cryptographic attack complexity.

## Exact Values for the Classic Problem

| People ($n$) | $\Pr[\text{collision}]$ |
|---|---|
| 10 | 0.117 |
| 20 | 0.411 |
| 23 | 0.507 |
| 30 | 0.706 |
| 50 | 0.970 |
| 70 | 0.999 |

The probability rises sharply around $n = 23$ because there are $\binom{n}{2}$ pairs, each contributing to the collision probability.

## Applications in Algorithm Analysis

### Hash Table Collisions

When inserting $n$ keys into a hash table with $m$ slots using a uniform hash function, the birthday paradox predicts collisions after about $\sqrt{m}$ insertions. For collision-free hashing, the table must have $m = \Omega(n^2)$ slots.

### Cryptographic Attacks

A birthday attack on a hash function with $b$-bit output requires approximately $2^{b/2}$ hash evaluations to find a collision. This is why cryptographic hash functions use output sizes at least twice the desired security level (e.g., 256 bits for 128-bit security).

### Randomized Algorithms

Many randomized algorithms involve random assignments to bins or categories. The birthday paradox governs when collisions become likely, influencing the design of algorithms for duplicate detection, database join estimation, and network protocol analysis.

## Generalization: $k$-way Collisions

The birthday paradox generalizes to $k$-way collisions (where $k$ items share a category). The threshold for a $k$-way collision among $m$ categories is

$$
n = \Theta(m^{(k-1)/k})
$$

For pairwise collisions ($k = 2$), this recovers the $\Theta(\sqrt{m})$ result.

## Reference

- Motwani, R. & Raghavan, P. *Randomized Algorithms*. Cambridge University Press, 1995.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L. & Stein, C. *Introduction to Algorithms*. MIT Press, 2022.
