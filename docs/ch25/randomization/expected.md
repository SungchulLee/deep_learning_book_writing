# Expected Running Time

Randomized algorithms make random choices during execution, so their running time varies across different runs on the same input. Rather than analyzing a single deterministic path, we characterize performance through the **expected running time** — the average over all possible random choices the algorithm could make. This measure provides a meaningful guarantee because, by linearity of expectation and concentration inequalities, the actual running time on any given run is typically close to the expectation.

## Definition

Let $A$ be a randomized algorithm and $x$ an input of size $n$. The algorithm's execution depends on random bits $r$ drawn from some probability space $\Omega$. Denote by $T(x, r)$ the number of elementary operations performed on input $x$ with random choices $r$.

The **expected running time on input $x$** is

$$
E[T(x)] = \sum_{r \in \Omega} T(x, r) \cdot \Pr[r]
$$

When the random choices are continuous (e.g., choosing a real-valued pivot), the sum becomes an integral.

The **worst-case expected running time** over all inputs of size $n$ is

$$
T_{\text{exp}}(n) = \max_{|x| = n} E[T(x)]
$$

This is the standard complexity measure for randomized algorithms: it takes the worst case over inputs but averages over the algorithm's internal randomness.

## Linearity of Expectation

The most powerful tool for analyzing expected running time is **linearity of expectation**: for any random variables $X_1, X_2, \ldots, X_n$,

$$
E\left[\sum_{i=1}^{n} X_i\right] = \sum_{i=1}^{n} E[X_i]
$$

This holds regardless of whether the $X_i$ are independent. The strategy is to decompose the total running time into a sum of simpler random variables — often indicator random variables — and compute each expectation separately.

??? example "Decomposition Strategy"
    To analyze a randomized algorithm with expected running time:

    1. **Identify elementary events**: define indicator random variables $X_{ij}$ for small events (e.g., "element $i$ is compared to element $j$")
    2. **Express the total cost** as $T = \sum_{i,j} X_{ij}$
    3. **Compute each expectation**: $E[X_{ij}] = \Pr[X_{ij} = 1]$
    4. **Sum**: $E[T] = \sum_{i,j} \Pr[X_{ij} = 1]$

## Expected vs Worst-Case vs Average-Case

It is important to distinguish three different notions of running time:

| Measure | Randomness source | Input assumption |
|---|---|---|
| Worst-case | None (deterministic) | Adversarial |
| Average-case | None (deterministic) | Random input distribution |
| Expected (randomized) | Algorithm's coin flips | Adversarial |

The expected running time of a randomized algorithm does **not** assume a distribution on inputs. The randomness is internal to the algorithm, so the guarantee holds for every input. This contrasts with average-case analysis, which assumes inputs are drawn from a specific distribution.

## Worked Example: Randomized Search

Consider searching for a target value $t$ in an unsorted array $A[1 \ldots n]$ by repeatedly picking a random index and checking it. Let $T$ denote the number of probes until $t$ is found. Assume $t$ appears exactly once.

Each probe succeeds with probability $1/n$, so $T$ follows a geometric distribution:

$$
\Pr[T = k] = \left(1 - \frac{1}{n}\right)^{k-1} \cdot \frac{1}{n}
$$

The expected number of probes is

$$
E[T] = \sum_{k=1}^{\infty} k \cdot \left(1 - \frac{1}{n}\right)^{k-1} \cdot \frac{1}{n} = n
$$

Although this is no better than linear scan in expectation, it illustrates how internal randomness defines the expected running time: the input is fixed, and only the algorithm's random choices vary.

## Conditional Expectation and Recurrences

Many randomized algorithms are naturally recursive, and their expected running time satisfies a recurrence. The **law of total expectation** is the key tool:

$$
E[T(n)] = \sum_{i} \Pr[\text{event } i] \cdot E[T(n) \mid \text{event } i]
$$

For randomized quicksort, if the pivot lands at rank $i$ (each rank equally likely), the expected running time satisfies

$$
E[T(n)] = \sum_{i=1}^{n} \frac{1}{n} \cdot \bigl(E[T(i-1)] + E[T(n-i)] + \Theta(n)\bigr)
$$

Solving this recurrence yields $E[T(n)] = O(n \log n)$.

## Tail Bounds and Concentration

Knowing the expectation alone does not guarantee that the running time is close to its mean on any particular execution. **Markov's inequality** provides a basic tail bound: for a non-negative random variable $T$,

$$
\Pr[T \geq c \cdot E[T]] \leq \frac{1}{c}
$$

Stronger bounds come from **Chebyshev's inequality** (requires the variance) and **Chernoff bounds** (requires independence). For many randomized algorithms, the running time concentrates sharply around its expectation, making the expected running time a reliable performance predictor.

!!! tip "When Expected Running Time Suffices"
    If a randomized algorithm's running time concentrates well (e.g., the variance is $o(E[T]^2)$), then the expected running time is a strong practical guarantee. Algorithms like randomized quicksort and randomized selection exhibit this concentration.

## Reference

- Motwani, R. & Raghavan, P. *Randomized Algorithms*. Cambridge University Press, 1995.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L. & Stein, C. *Introduction to Algorithms*. MIT Press, 2022.
