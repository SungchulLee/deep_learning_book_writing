# Indicator Random Variables

Many randomized algorithm analyses require counting how many "events" occur during execution — how many comparisons, how many collisions, how many elements land in a particular bin. **Indicator random variables** provide a systematic way to convert such counting problems into probability calculations. Combined with linearity of expectation, they yield elegant proofs that avoid complicated conditional reasoning.

## Definition

Given a probability space and an event $A$, the **indicator random variable** for $A$ is

$$
I_A = \begin{cases} 1 & \text{if } A \text{ occurs} \\ 0 & \text{otherwise} \end{cases}
$$

The expectation of an indicator is simply the probability of the event:

$$
E[I_A] = 1 \cdot \Pr[A] + 0 \cdot \Pr[\bar{A}] = \Pr[A]
$$

This simple identity is the foundation for all indicator-based analyses.

## Properties

Indicator random variables satisfy several useful algebraic properties:

**Complement.** $I_{\bar{A}} = 1 - I_A$.

**Intersection.** $I_{A \cap B} = I_A \cdot I_B$.

**Union.** $I_{A \cup B} = I_A + I_B - I_A \cdot I_B$.

**Independence.** Events $A$ and $B$ are independent if and only if $E[I_A \cdot I_B] = E[I_A] \cdot E[I_B]$.

**Variance.** Since $I_A^2 = I_A$ (it takes only values 0 and 1),

$$
\text{Var}(I_A) = E[I_A^2] - (E[I_A])^2 = \Pr[A] - \Pr[A]^2 = \Pr[A](1 - \Pr[A])
$$

The variance is maximized at $\Pr[A] = 1/2$, where $\text{Var}(I_A) = 1/4$.

## The Counting Technique

The main application of indicator random variables is expressing a count as a sum of indicators. Suppose we want to count the total number of occurrences of some event type across $n$ trials. Define $X_i = I_{A_i}$ for each trial $i$. The total count is

$$
X = \sum_{i=1}^{n} X_i
$$

By linearity of expectation (which holds regardless of dependence),

$$
E[X] = \sum_{i=1}^{n} E[X_i] = \sum_{i=1}^{n} \Pr[A_i]
$$

!!! tip "No Independence Required"
    Linearity of expectation holds even when the $X_i$ are dependent. This is what makes the indicator technique so powerful: we can decompose a complex random variable into dependent pieces and still compute the expectation by summing individual probabilities.

## Worked Examples

### Counting Inversions

An **inversion** in a permutation $\pi$ of $\{1, 2, \ldots, n\}$ is a pair $(i, j)$ with $i < j$ but $\pi(i) > \pi(j)$. For a uniformly random permutation, define

$$
X_{ij} = I\{\pi(i) > \pi(j)\} \quad \text{for } 1 \leq i < j \leq n
$$

By symmetry, $\Pr[\pi(i) > \pi(j)] = 1/2$, so $E[X_{ij}] = 1/2$. The total number of inversions is

$$
E\left[\sum_{i < j} X_{ij}\right] = \binom{n}{2} \cdot \frac{1}{2} = \frac{n(n-1)}{4}
$$

### Hat-Check Problem

Each of $n$ people checks a hat, and the hats are returned uniformly at random. How many people get their own hat? Define $X_i = I\{\text{person } i \text{ gets own hat}\}$. Then $\Pr[X_i = 1] = 1/n$, and

$$
E\left[\sum_{i=1}^{n} X_i\right] = n \cdot \frac{1}{n} = 1
$$

The expected number of fixed points in a random permutation is exactly 1, regardless of $n$.

### Randomized Quicksort Comparisons

Define $X_{ij} = I\{z_i \text{ is compared with } z_j\}$ where $z_i$ is the $i$-th smallest element. The total comparisons satisfy

$$
E[X] = \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} \frac{2}{j - i + 1} = 2n \ln n + O(n)
$$

This classic result follows entirely from computing $\Pr[X_{ij} = 1] = 2/(j - i + 1)$ and summing.

## Variance of a Sum of Indicators

When independence holds, the variance of a sum of indicators has a simple form. If $X = \sum_{i} X_i$ with independent $X_i$,

$$
\text{Var}(X) = \sum_{i} \text{Var}(X_i) = \sum_{i} p_i(1 - p_i)
$$

where $p_i = \Pr[A_i]$. Without independence, the variance includes covariance terms:

$$
\text{Var}(X) = \sum_{i} \text{Var}(X_i) + 2 \sum_{i < j} \text{Cov}(X_i, X_j)
$$

where $\text{Cov}(X_i, X_j) = E[X_i X_j] - E[X_i] E[X_j] = \Pr[A_i \cap A_j] - \Pr[A_i]\Pr[A_j]$.

## Reference

- Motwani, R. & Raghavan, P. *Randomized Algorithms*. Cambridge University Press, 1995.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L. & Stein, C. *Introduction to Algorithms*. MIT Press, 2022.
