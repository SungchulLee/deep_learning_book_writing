# Secretary Problem

Imagine interviewing candidates one by one for a position, where after each interview you must immediately accept or reject the candidate with no possibility of recalling rejected ones. How many candidates should you observe before starting to select? The **secretary problem** (also called the optimal stopping problem) provides a beautiful answer: observe the first $n/e$ candidates, then select the next candidate who is better than all previously seen. This strategy selects the best candidate with probability converging to $1/e \approx 0.368$, and no strategy can do better.

## Problem Formulation

A set of $n$ candidates arrives in uniformly random order. After interviewing candidate $i$, the algorithm observes the **relative rank** of candidate $i$ among the first $i$ candidates (but not the absolute quality). The algorithm must immediately and irrevocably decide to accept or reject candidate $i$. The goal is to maximize the probability of selecting the single best candidate.

Key assumptions:

- There are exactly $n$ candidates, and $n$ is known in advance.
- Candidates arrive in a uniformly random permutation.
- Only relative rankings are observable (no cardinal scores).
- Decisions are irrevocable: rejected candidates cannot be recalled.
- The objective is to select the absolute best candidate (not just a good one).

## The Optimal Stopping Rule

The optimal strategy belongs to the class of **threshold rules**: reject the first $r - 1$ candidates unconditionally (the **observation phase**), then accept the first subsequent candidate who is better than all candidates seen so far.

Let $P(r, n)$ denote the probability of selecting the best candidate when the observation phase has length $r - 1$:

$$
P(r, n) = \sum_{i=r}^{n} P(\text{best is at position } i \text{ and is selected}) = \sum_{i=r}^{n} \frac{1}{n} \cdot \frac{r-1}{i-1}
$$

The term $1/n$ is the probability that the best candidate is at position $i$, and $(r-1)/(i-1)$ is the probability that the best among the first $i-1$ candidates falls within the observation phase (so the algorithm does not stop earlier).

## Asymptotic Optimality

To find the optimal $r$, maximize $P(r, n)$:

$$
P(r, n) = \frac{r-1}{n} \sum_{i=r}^{n} \frac{1}{i-1}
$$

As $n \to \infty$, set $r = \lfloor \alpha n \rfloor$ for some $\alpha \in (0, 1)$. The sum becomes a Riemann integral:

$$
P(\alpha) = \alpha \int_{\alpha}^{1} \frac{1}{x} \, dx = -\alpha \ln \alpha
$$

Maximizing over $\alpha$ by taking the derivative and setting it to zero:

$$
\frac{d}{d\alpha}(-\alpha \ln \alpha) = -\ln \alpha - 1 = 0 \implies \alpha^* = \frac{1}{e}
$$

**Theorem.** The optimal strategy observes the first $\lfloor n/e \rfloor$ candidates, then selects the next candidate better than all previous ones. This strategy selects the best candidate with probability:

$$
P^* = \frac{1}{e} \approx 0.3679
$$

No online algorithm can achieve a higher probability. $\square$

!!! note "Probability Interpretation"
    The probability $1/e$ means that even the best possible strategy fails to select the best candidate about 63% of the time. This is the inherent cost of making irrevocable decisions with incomplete information.

## Variants

### Multiple Choices

If the algorithm can select $k$ candidates (e.g., hiring $k$ positions), the success probability increases. The optimal threshold shifts, and the analysis generalizes using multiple stopping rules.

### Unknown $n$

When the number of candidates $n$ is unknown, the algorithm cannot compute $n/e$ directly. Variants use:

- **Time-based thresholds**: if interview times are uniform on $[0, 1]$, reject until time $1/e$.
- **Adaptive strategies**: maintain a running estimate of $n$ and adjust the threshold dynamically.

### Cardinal Payoff

If the objective changes from selecting the best to maximizing expected rank or expected value, different strategies emerge. For maximizing expected rank, the optimal threshold decreases as the penalty for non-best selections is less severe.

### Secretary Problem with Recall

If rejected candidates can be recalled with some probability $p$, the optimal threshold shifts to the right, allowing a shorter observation phase.

## Analysis of the $1/e$ Strategy

The proof that $1/e$ is optimal uses the following steps:

1. **Restrict to threshold strategies**: any optimal strategy can be expressed as a threshold rule (by the structure of sufficient statistics).
2. **Compute success probability**: for threshold $r$, the probability is $P(r,n) = \frac{r-1}{n}\sum_{i=r}^n \frac{1}{i-1}$.
3. **Take the continuous limit**: as $n \to \infty$, $P(\alpha) = -\alpha \ln \alpha$.
4. **Optimize**: the unique maximum occurs at $\alpha = 1/e$ with value $1/e$.

??? example "Numerical Verification for Small $n$"
    For $n = 10$, the optimal threshold is $r = 4$ (observe 3, then select), yielding success probability $\approx 0.399$. As $n$ grows, the optimal $r/n$ approaches $1/e$ and the success probability approaches $1/e \approx 0.368$.

## Connection to Deep Learning

The secretary problem framework appears in several deep learning contexts:

- **Hyperparameter search**: when evaluating model configurations sequentially with a limited budget, the explore-then-exploit structure mirrors the secretary problem's observation-then-selection phases.
- **Early stopping**: deciding when to stop training resembles an optimal stopping problem where the "candidates" are model checkpoints at different epochs.
- **Neural architecture search**: evaluating architectures sequentially under budget constraints involves similar accept/reject decisions.

## Summary

The secretary problem demonstrates that with the optimal $1/e$-threshold strategy, an online algorithm selects the best of $n$ candidates with probability $1/e$, and this is the best achievable. The elegant interplay between the observation phase length and selection probability produces one of the most celebrated results in optimal stopping theory, with applications ranging from hiring decisions to hyperparameter optimization.

## References

- [Online Computation and Competitive Analysis (Borodin and El-Yaniv)](https://www.amazon.com/dp/0521619467)
- [Who Solved the Secretary Problem? (Ferguson, 1989)](https://doi.org/10.1214/ss/1177012493)
