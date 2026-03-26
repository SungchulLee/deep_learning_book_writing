# Randomized Rounding

When solving an LP relaxation of a combinatorial optimization problem, the fractional solution must be converted to an integer one. **Randomized rounding** uses the fractional values as probabilities, setting each variable to 1 independently with probability equal to (or derived from) its LP value. This elegant technique, introduced by Raghavan and Thompson (1987), produces solutions that are near-optimal in expectation.

## The Randomized Rounding Framework

Given an integer program relaxed to an LP with optimal fractional solution $x^*$:

1. **Solve** the LP relaxation to obtain $x^* \in [0, 1]^n$.
2. **Round** each variable independently: set $X_j = 1$ with probability $p_j = f(x_j^*)$, where $f$ is a rounding function.
3. **Analyze** the expected objective value and constraint satisfaction using concentration inequalities.

The simplest choice is $p_j = x_j^*$ (direct rounding). More sophisticated schemes scale probabilities to improve feasibility.

## Application: MAX-SAT

Consider a MAX-SAT instance with $m$ clauses $C_1, \ldots, C_m$ and $n$ Boolean variables $x_1, \ldots, x_n$, where clause $C_i$ has weight $w_i$.

### LP Relaxation

Let $y_j \in [0, 1]$ represent variable $x_j$ and $z_i \in [0, 1]$ represent whether clause $C_i$ is satisfied:

$$
\max \sum_{i=1}^{m} w_i z_i
$$

subject to:

$$
z_i \leq \sum_{j \in C_i^+} y_j + \sum_{j \in C_i^-} (1 - y_j) \quad \forall i
$$

$$
0 \leq y_j \leq 1, \quad 0 \leq z_i \leq 1
$$

where $C_i^+$ and $C_i^-$ are the positive and negative literals in clause $C_i$.

### Rounding and Analysis

Set each variable $x_j = 1$ with probability $y_j^*$ independently.

!!! tip "Theorem: Expected Approximation for MAX-SAT"
    The randomized rounding procedure satisfies each clause $C_i$ with $k_i$ literals with probability at least $(1 - (1 - 1/k_i)^{k_i}) \cdot z_i^*$.

**Proof.** For a clause with $k$ literals, the probability it is unsatisfied is maximized when all LP values equal $z^*/k$. By the AM-GM inequality:

$$
\Pr[\text{clause unsatisfied}] \leq \prod_{j=1}^{k} (1 - y_j^*) \leq \left(1 - \frac{z^*}{k}\right)^k
$$

Since $1 - (1 - 1/k)^k \geq 1 - 1/e$ for all $k \geq 1$:

$$
\Pr[\text{clause satisfied}] \geq \left(1 - \left(1 - \frac{1}{k}\right)^k\right) z_i^* \geq \left(1 - \frac{1}{e}\right) z_i^*
$$

The expected total weight of satisfied clauses is at least $(1 - 1/e) \cdot \text{OPT}_{\text{LP}} \geq (1 - 1/e) \cdot \text{OPT}$. $\square$

### Best of Two Strategies

Combining randomized rounding with a simple random assignment (which satisfies each clause with probability $1 - 2^{-k}$) gives a $3/4$-approximation:

$$
\max\left(1 - \frac{1}{e}, \frac{1}{2}\right) \cdot z_i^* \quad \text{for the better strategy per clause}
$$

Taking the better of both gives expected weight at least $\frac{3}{4} \cdot \text{OPT}$.

## Application: Set Cover

For universe $U$ with $n = |U|$ and sets $S_1, \ldots, S_m$ with costs $c_j$:

### LP and Rounding

Solve the LP relaxation and round: include $S_j$ independently with probability $\min(1,\; c \cdot x_j^* \cdot \ln n)$ for an appropriate constant $c$.

!!! tip "Theorem"
    Randomized rounding with $O(\ln n)$ repetitions achieves an expected cost of $O(\ln n) \cdot \text{OPT}$.

Each element $i$ is covered with high probability because the sum of fractional values covering it is at least 1 (from the LP constraint), and amplification via $O(\ln n)$ independent trials drives the failure probability below $1/n$.

## Concentration Bounds

Randomized rounding analyses frequently use **Chernoff bounds** to convert expected guarantees into high-probability statements.

For independent random variables $X_1, \ldots, X_n \in [0, 1]$ with $\mu = \mathbb{E}[\sum_j X_j]$:

$$
\Pr\left[\sum_j X_j > (1 + \delta) \mu\right] \leq \exp\left(-\frac{\delta^2 \mu}{2 + \delta}\right)
$$

$$
\Pr\left[\sum_j X_j < (1 - \delta) \mu\right] \leq \exp\left(-\frac{\delta^2 \mu}{2}\right)
$$

These bounds convert the expected approximation ratio into a guarantee that holds with high probability.

## Derandomization

Randomized rounding can often be **derandomized** using the method of conditional expectations:

1. Fix variables one at a time: $x_1, x_2, \ldots, x_n$.
2. For each $x_j$, choose the value (0 or 1) that maximizes the conditional expectation of the objective.
3. This produces a deterministic solution at least as good as the expected randomized solution.

??? example "Worked Example: MAX-SAT Rounding"
    **Instance:** Variables $x_1, x_2, x_3$. Clauses: $C_1 = (x_1 \lor x_2)$, $C_2 = (\bar{x}_1 \lor x_3)$, $C_3 = (\bar{x}_2 \lor \bar{x}_3)$, all with weight 1.

    **LP solution:** $y_1^* = 0.5$, $y_2^* = 0.5$, $y_3^* = 0.5$, all $z_i^* = 1$.

    **Rounding:** Set each $x_j = 1$ with probability 0.5.

    - $\Pr[C_1 \text{ sat}] = 1 - (0.5)(0.5) = 0.75$
    - $\Pr[C_2 \text{ sat}] = 1 - (0.5)(0.5) = 0.75$
    - $\Pr[C_3 \text{ sat}] = 1 - (0.5)(0.5) = 0.75$

    **Expected satisfied clauses:** $3 \times 0.75 = 2.25$. Lower bound: $(1 - 1/e) \times 3 \approx 1.90$. The actual expected value exceeds the bound.

## Reference

- Raghavan, P., & Thompson, C. D. (1987). Randomized rounding: a technique for provably good algorithms. *Combinatorica*, 7(4), 365--374.
- Vazirani, V. V. (2001). *Approximation Algorithms*. Springer, Chapters 14, 16.
- Williamson, D. P., & Shmoys, D. B. (2011). *The Design of Approximation Algorithms*. Cambridge University Press.
