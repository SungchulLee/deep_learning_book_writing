# Las Vegas Algorithms

When designing a randomized algorithm, a fundamental design choice is whether to guarantee correctness or running time. **Las Vegas algorithms** always produce the correct answer but have a random running time. This contrasts with Monte Carlo algorithms, which may produce an incorrect answer but run within a deterministic time bound. The Las Vegas guarantee is particularly attractive in settings where an incorrect result is unacceptable, such as sorting or searching.

## Definition

A randomized algorithm $A$ is a **Las Vegas algorithm** if, for every input $x$:

1. $A(x)$ always outputs the correct answer, and
2. the running time $T(x)$ is a random variable (depending on the algorithm's internal coin flips).

The performance measure for a Las Vegas algorithm is its **expected running time**:

$$
T_{\text{LV}}(n) = \max_{|x|=n} E[T(x)]
$$

The expectation is over the algorithm's random choices, not over the input. For every input, the output is correct with probability 1.

## Formal Properties

A Las Vegas algorithm can be viewed as a distribution over deterministic algorithms, all of which are correct:

$$
\Pr[\text{output is correct}] = 1
$$

$$
E[T(x)] < \infty \quad \text{for all inputs } x
$$

The second condition ensures that the algorithm terminates in finite expected time. In practice, Las Vegas algorithms often have small variance around the expectation, so the actual running time rarely deviates far from $E[T]$.

## Classic Examples

### Randomized Quicksort

Randomized quicksort chooses a pivot uniformly at random, then partitions and recurses. It always produces a correctly sorted array, but the number of comparisons depends on pivot choices:

$$
E[\text{comparisons}] = 2n \ln n + O(n) = O(n \log n)
$$

The worst case is $O(n^2)$, but this occurs with negligible probability (exponentially small for a random pivot).

### Randomized Selection

The randomized select algorithm finds the $k$-th smallest element by random pivoting. It always returns the correct element, with expected running time $O(n)$ and worst case $O(n^2)$.

### Randomized Search in a Hash Table

Universal hashing with chaining is a Las Vegas approach to dictionary operations: every lookup returns the correct result, but the time depends on the random hash function chosen.

## Las Vegas vs Monte Carlo Conversion

Las Vegas and Monte Carlo algorithms are interconvertible under mild conditions.

**Las Vegas to Monte Carlo.** Run the Las Vegas algorithm for a fixed time budget $t$. If it finishes, output the result. Otherwise, output "failure" or a random guess. The resulting algorithm runs in time at most $t$, but may be incorrect.

By Markov's inequality, if the Las Vegas expected time is $E[T]$, then running for $t = c \cdot E[T]$ steps gives a correct answer with probability at least $1 - 1/c$.

**Monte Carlo to Las Vegas.** If the Monte Carlo algorithm's output can be verified in polynomial time (as with NP problems), run the Monte Carlo algorithm repeatedly until verification succeeds. The expected number of repetitions is $1/p$, where $p$ is the Monte Carlo success probability.

!!! tip "Choosing Between Paradigms"
    Prefer a Las Vegas algorithm when correctness is paramount and the expected running time is acceptable. Prefer Monte Carlo when a hard time bound is required and occasional errors are tolerable (e.g., probabilistic primality testing).

## Worst-Case Elimination

A useful technique converts a Las Vegas algorithm with bad worst-case behavior into one with controlled running time. Run the algorithm for $2 \cdot E[T]$ steps. If it has not finished, restart with fresh random bits. The expected total time is

$$
E[T_{\text{restart}}] \leq 2 \cdot E[T] \cdot \frac{1}{1 - 1/2} = 4 \cdot E[T]
$$

This restart strategy ensures that the probability of running longer than $c \cdot E[T]$ decreases exponentially in $c$, giving strong tail concentration even when the original algorithm's running time has high variance.

## Reference

- Motwani, R. & Raghavan, P. *Randomized Algorithms*. Cambridge University Press, 1995.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L. & Stein, C. *Introduction to Algorithms*. MIT Press, 2022.
