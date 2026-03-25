# Coupon Collector

Imagine collecting trading cards: each pack contains one card drawn uniformly at random from $n$ types. How many packs must you buy before you have at least one of every type? The **coupon collector problem** shows that the answer is $\Theta(n \ln n)$ — substantially more than $n$ because the last few missing types become increasingly hard to find. This problem models hashing completeness, randomized covering, and the performance of randomized algorithms that must "hit" every element.

## Problem Setup

A collector draws coupons one at a time, each independently and uniformly from $n$ types. Let $T$ denote the number of draws until all $n$ types have been collected at least once.

We decompose $T$ into phases. **Phase $i$** begins when exactly $i - 1$ distinct types have been collected and ends when the $i$-th new type appears, for $i = 1, 2, \ldots, n$. Let $T_i$ denote the number of draws during phase $i$. Then

$$
T = \sum_{i=1}^{n} T_i
$$

## Analysis of Each Phase

During phase $i$, exactly $i - 1$ types have been seen, so $n - (i - 1) = n - i + 1$ types remain unseen. Each draw independently hits an unseen type with probability

$$
p_i = \frac{n - i + 1}{n}
$$

Since draws are independent and each has success probability $p_i$, the number of draws $T_i$ follows a geometric distribution with parameter $p_i$:

$$
E[T_i] = \frac{1}{p_i} = \frac{n}{n - i + 1}
$$

## Expected Collection Time

By linearity of expectation,

$$
E[T] = \sum_{i=1}^{n} E[T_i] = \sum_{i=1}^{n} \frac{n}{n - i + 1} = n \sum_{j=1}^{n} \frac{1}{j} = n H_n
$$

where $H_n = \sum_{j=1}^{n} 1/j$ is the $n$-th harmonic number. Since $H_n = \ln n + \gamma + O(1/n)$ where $\gamma \approx 0.5772$ is the Euler-Mascheroni constant,

$$
E[T] = n \ln n + \gamma n + O(1)
$$

??? example "Numerical Example"
    For $n = 100$ coupon types, the expected number of draws is

    $$
    E[T] = 100 \cdot H_{100} \approx 100 \cdot 5.187 = 518.7
    $$

    To collect all 100 types, you need about 519 draws on average — roughly 5 times the number of types.

## Variance and Concentration

The variance of $T$ is

$$
\text{Var}(T) = \sum_{i=1}^{n} \text{Var}(T_i) = \sum_{i=1}^{n} \frac{1 - p_i}{p_i^2} = n^2 \sum_{j=1}^{n} \frac{1}{j^2} - n \sum_{j=1}^{n} \frac{1}{j}
$$

Since $\sum_{j=1}^{\infty} 1/j^2 = \pi^2/6$,

$$
\text{Var}(T) \leq \frac{\pi^2}{6} n^2
$$

The standard deviation is $O(n)$, which is small relative to the mean of $\Theta(n \log n)$.

!!! tip "Tail Bound"
    By Markov's inequality, $\Pr[T \geq 2n \ln n] \leq 1/2$. A sharper bound uses the independence of the $T_i$: for $c > 0$,

    $$
    \Pr[T \geq n \ln n + cn] \leq e^{-c}
    $$

    This shows that the collection time concentrates around its mean with exponential tails.

## The Last Few Coupons

The expected time to collect the last coupon (when $n - 1$ types are known) is $n$. The second-to-last costs $n/2$ in expectation. More generally, the last $k$ coupons require

$$
\sum_{j=1}^{k} \frac{n}{j} = n H_k \approx n \ln k
$$

draws in expectation. This logarithmic blowup in the final stages is the essence of the coupon collector phenomenon.

## Applications

### Hashing Completeness

When randomly assigning $n$ items to $n$ hash buckets, the coupon collector result governs how many items must be inserted before every bucket has at least one item: $\Theta(n \ln n)$ items are needed.

### Randomized Broadcasting

In a network with $n$ nodes, if each round a random node receives a message, all nodes receive it after $\Theta(n \ln n)$ rounds in expectation.

### Testing Coverage

Random testing of $n$ code paths requires $\Theta(n \ln n)$ random test cases to cover every path at least once.

## Generalization: Collecting Multiple Copies

If each type must be collected at least $k$ times, the expected number of draws is

$$
E[T_k] = n H_n + (k-1) n \ln \ln n + O(n)
$$

for $k$ growing slowly with $n$. For fixed $k$, the leading term remains $n \ln n$.

## Reference

- Motwani, R. & Raghavan, P. *Randomized Algorithms*. Cambridge University Press, 1995.
- Mitzenmacher, M. & Upfal, E. *Probability and Computing*. Cambridge University Press, 2017.
