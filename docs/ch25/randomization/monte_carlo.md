# Monte Carlo Algorithms

Some computational problems admit algorithms that trade absolute correctness for a guaranteed time bound. **Monte Carlo algorithms** use randomness and always terminate within a fixed running time, but may produce an incorrect answer with some bounded probability. This paradigm is essential in settings where a hard deadline matters more than a perfect answer, or where no efficient deterministic algorithm is known.

## Definition

A randomized algorithm $A$ is a **Monte Carlo algorithm** if, for every input $x$:

1. $A(x)$ runs in deterministic time $T(n)$ (no randomness in the running time), and
2. $A(x)$ produces the correct answer with probability at least $1 - \epsilon$ for some error bound $\epsilon < 1/2$.

The error bound $\epsilon$ is over the algorithm's internal random choices, not over the input distribution.

## One-Sided vs Two-Sided Error

Monte Carlo algorithms are classified by their error behavior:

**One-sided error (false negatives only).** If the correct answer is YES, the algorithm may incorrectly output NO with probability at most $\epsilon$. If the correct answer is NO, the algorithm always outputs NO correctly. This defines the complexity class **RP** (Randomized Polynomial time):

$$
x \in L \implies \Pr[A(x) = \text{YES}] \geq \frac{1}{2}
$$

$$
x \notin L \implies \Pr[A(x) = \text{YES}] = 0
$$

**Two-sided error.** The algorithm may err on both YES and NO instances, but with bounded probability. This defines the complexity class **BPP**:

$$
x \in L \implies \Pr[A(x) = \text{YES}] \geq \frac{2}{3}
$$

$$
x \notin L \implies \Pr[A(x) = \text{NO}] \geq \frac{2}{3}
$$

The constants $1/2$ and $2/3$ are conventional; any constant bounded away from $1/2$ suffices because of probability amplification.

## Probability Amplification

A central technique reduces the error probability exponentially by independent repetition. Run the Monte Carlo algorithm $k$ times independently and take the **majority vote** (for two-sided error) or **accept if any run accepts** (for one-sided error).

For a two-sided error algorithm with success probability $p > 1/2$ per run, the probability that the majority of $k$ independent runs is wrong is bounded by

$$
\Pr[\text{majority wrong}] \leq e^{-2k(p - 1/2)^2}
$$

by a Chernoff bound. Setting $k = O(\log(1/\delta))$ reduces the error to any desired $\delta > 0$.

??? example "Amplification in Practice"
    Suppose a Monte Carlo algorithm has error probability $1/3$. Running it $k = 100$ times and taking the majority vote gives error probability at most

    $$
    e^{-2 \cdot 100 \cdot (1/6)^2} = e^{-100/18} \approx 3.7 \times 10^{-3}
    $$

    With $k = 1000$ independent runs, the error drops below $10^{-24}$, which is negligible for any practical purpose.

## Classic Examples

### Miller-Rabin Primality Test

Given an integer $n$, the Miller-Rabin test determines whether $n$ is prime. It is a one-sided Monte Carlo algorithm: if it declares $n$ composite, it is always correct; if it declares $n$ prime, there is at most a $1/4$ probability of error per round. After $k$ rounds, the error probability is at most $4^{-k}$.

### Randomized Min-Cut (Karger's Algorithm)

Karger's contraction algorithm finds a minimum cut in a graph. A single run succeeds with probability at least $\binom{n}{2}^{-1} = 2/(n(n-1))$. By repeating $O(n^2 \log n)$ times, the probability of finding the min-cut approaches 1.

### Freivalds' Algorithm

To verify whether $AB = C$ for $n \times n$ matrices, choose a random vector $r \in \{0, 1\}^n$ and check whether $A(Br) = Cr$. This runs in $O(n^2)$ time (versus $O(n^3)$ for direct multiplication). If $AB \neq C$, the check fails with probability at least $1/2$. After $k$ repetitions, the error drops to $2^{-k}$.

## Monte Carlo vs Las Vegas

| Property | Monte Carlo | Las Vegas |
|---|---|---|
| Correctness | Probabilistic ($\geq 1 - \epsilon$) | Always correct |
| Running time | Deterministic bound | Random variable |
| Error reduction | Repeat and vote | Restart with fresh randomness |
| Complexity class | BPP, RP | ZPP |

!!! tip "Conversion Between Paradigms"
    A Las Vegas algorithm with expected time $E[T]$ can be converted to a Monte Carlo algorithm by imposing a time cutoff of $c \cdot E[T]$ and outputting "failure" if unfinished. Conversely, a Monte Carlo algorithm whose output is efficiently verifiable can be converted to Las Vegas by repeating until verification succeeds.

## Reference

- Motwani, R. & Raghavan, P. *Randomized Algorithms*. Cambridge University Press, 1995.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L. & Stein, C. *Introduction to Algorithms*. MIT Press, 2022.
