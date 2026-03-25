# Competitive Ratio

The competitive ratio is the central quantitative measure in online algorithm analysis. Just as approximation ratio measures how well a polynomial-time algorithm approaches an NP-hard optimum, the competitive ratio measures how well an online algorithm performs relative to an omniscient offline optimum. Understanding how to compute and prove competitive ratios is essential for designing and evaluating algorithms that must operate under uncertainty.

## Formal Definition

Let $A$ be an online algorithm and $\text{OPT}$ the optimal offline algorithm. For a minimization problem, $A$ is **$c$-competitive** if there exists a constant $b$ such that for every request sequence $\sigma$:

$$
C_A(\sigma) \leq c \cdot C_{\text{OPT}}(\sigma) + b
$$

The **competitive ratio** of $A$ is the infimum over all $c$ satisfying this inequality:

$$
\rho_A = \inf \left\{ c \geq 1 : \exists\, b \text{ s.t. } C_A(\sigma) \leq c \cdot C_{\text{OPT}}(\sigma) + b, \; \forall \sigma \right\}
$$

For a **maximization** problem, the direction reverses:

$$
C_A(\sigma) \geq \frac{1}{c} \cdot C_{\text{OPT}}(\sigma) - b
$$

!!! note "Strict Competitive Ratio"
    When $b = 0$, the algorithm is **strictly $c$-competitive**. For many problems, the additive constant does not affect the ratio for sufficiently long sequences, so strict and non-strict ratios coincide asymptotically.

## Computing the Competitive Ratio

### Upper Bound (Algorithm Analysis)

To show an algorithm $A$ has competitive ratio at most $c$, prove that for every input sequence $\sigma$:

$$
\frac{C_A(\sigma)}{C_{\text{OPT}}(\sigma)} \leq c
$$

Common proof techniques include:

1. **Direct analysis**: bound $C_A(\sigma)$ explicitly as a function of $C_{\text{OPT}}(\sigma)$ for every possible $\sigma$.
2. **Potential function method**: define a potential $\Phi$ mapping algorithm states to non-negative reals, and show the amortized cost per request satisfies $a_i \leq c \cdot c_i^{\text{OPT}}$.
3. **Charging argument**: assign each unit of cost incurred by $A$ to a unit of cost incurred by $\text{OPT}$, showing at most $c$ charges per OPT unit.

### Lower Bound (Adversary Argument)

To prove no algorithm can achieve ratio better than $c$, construct an adversary that forces any algorithm to pay at least $c$ times OPT.

!!! tip "Yao's Minimax Principle"
    For randomized algorithms against an oblivious adversary, Yao's principle states: the expected competitive ratio of the best randomized algorithm against a worst-case input equals the competitive ratio of the best deterministic algorithm against the worst-case input distribution. Formally:

    $$
    \max_{\sigma} \frac{\mathbb{E}[C_A(\sigma)]}{C_{\text{OPT}}(\sigma)} \geq \min_A \frac{\mathbb{E}_\sigma[C_A(\sigma)]}{\mathbb{E}_\sigma[C_{\text{OPT}}(\sigma)]}
    $$

## Classical Competitive Ratios

The following table summarizes competitive ratios for well-known online problems:

| Problem | Deterministic | Randomized (oblivious) |
|---|---|---|
| Paging ($k$-cache) | $k$ (LRU, FIFO) | $H_k = \Theta(\ln k)$ (Marker) |
| Ski rental | $2$ (buy on day $b$) | $e/(e-1) \approx 1.58$ |
| List accessing | $2$ (Move-to-Front) | $1.5$ |
| Online scheduling ($m$ machines) | $2 - 1/m$ (List) | $\Theta(\log m)$ |
| $k$-Server ($k$ servers, metric space) | $2k - 1$ (conjectured tight) | $\Theta(\log k)$ (conjectured) |

## Example: Ski Rental Competitive Ratio

Consider the ski rental problem with rental cost 1 per day and purchase cost $b$. The deterministic strategy "rent for $b - 1$ days, then buy" costs:

- If the season lasts $n \leq b - 1$ days: $C_A = n = C_{\text{OPT}}$, so ratio is 1
- If the season lasts $n \geq b$ days: $C_A = (b - 1) + b = 2b - 1$, while $C_{\text{OPT}} = b$, giving ratio $(2b - 1)/b < 2$

Therefore this strategy is strictly 2-competitive:

$$
\frac{C_A(\sigma)}{C_{\text{OPT}}(\sigma)} \leq \frac{2b - 1}{b} < 2
$$

## Competitive Ratio for Randomized Algorithms

A randomized algorithm $A$ is **$c$-competitive against an oblivious adversary** if:

$$
\mathbb{E}[C_A(\sigma)] \leq c \cdot C_{\text{OPT}}(\sigma) + b \quad \forall \sigma
$$

??? example "Randomized Ski Rental"
    Choose a random day $D$ to buy, drawn from a carefully designed distribution. With probability $p_i$ for day $i$, the expected cost becomes a function of the distribution. The optimal distribution achieves a competitive ratio of $e/(e-1) \approx 1.58$, strictly better than the deterministic ratio of 2.

## Connection to Regret in Online Learning

In online learning, the analogous concept is **regret**: the difference between the learner's cumulative loss and the best fixed strategy in hindsight:

$$
R_T = \sum_{t=1}^{T} \ell_t(a_t) - \min_{a^*} \sum_{t=1}^{T} \ell_t(a^*)
$$

While competitive ratio measures multiplicative overhead, regret measures additive overhead. An algorithm with sublinear regret $R_T = o(T)$ achieves average cost approaching the best fixed strategy, analogous to a competitive ratio approaching 1.

## Summary

The competitive ratio provides a worst-case guarantee for online algorithms, quantifying the price of ignorance about future inputs. Upper bounds come from algorithm analysis (potential functions, charging arguments), while lower bounds come from adversary constructions and Yao's minimax principle. Randomization provably improves competitive ratios for many fundamental problems.

## References

- [Online Computation and Competitive Analysis (Borodin and El-Yaniv)](https://www.amazon.com/dp/0521619467)
- [Data Streams: Algorithms and Applications (Muthukrishnan)](https://www.cs.rutgers.edu/~muthu/stream-1-1.ps)
