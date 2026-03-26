# Approximation Algorithms Overview

When an NP-hard problem must be solved in practice, exact algorithms require exponential time. **Approximation algorithms** offer a middle ground: polynomial-time algorithms with provable guarantees on solution quality. This page surveys the key ideas, contrasts approximation with other coping strategies, and summarizes the landscape of approximability.

## The Approximation Approach

An approximation algorithm for an optimization problem:

1. Runs in **polynomial time**.
2. Always produces a **feasible** solution.
3. Guarantees the solution value is within a **bounded ratio** of the optimum.

For a minimization problem with algorithm output $A(I)$ and optimum $\text{OPT}(I)$:

$$
A(I) \leq \rho \cdot \text{OPT}(I)
$$

For maximization:

$$
A(I) \geq \frac{1}{\rho} \cdot \text{OPT}(I)
$$

The ratio $\rho \geq 1$ is the **approximation ratio**. Smaller $\rho$ means better quality.

## Design Techniques

### Greedy Algorithms

The simplest approach: make locally optimal choices. Often yields constant-factor approximations.

- **Vertex Cover:** Pick any edge, add both endpoints, remove incident edges. Achieves ratio 2.
- **Set Cover:** Greedily pick the set covering the most uncovered elements. Achieves ratio $H_n = \ln n + O(1)$.

### LP Relaxation and Rounding

Formulate the problem as an integer linear program, solve the LP relaxation, then round the fractional solution to integers.

- **Threshold rounding:** Round $x_j^* \geq \theta$ to 1. Vertex cover with $\theta = 1/2$ gives ratio 2.
- **Randomized rounding:** Set $x_j = 1$ with probability $x_j^*$. MAX-SAT achieves ratio $(1 - 1/e)$.

### Primal-Dual Method

Build primal and dual solutions simultaneously. The dual provides a lower bound, and the primal-dual gap bounds the approximation ratio.

### Local Search

Start with a feasible solution and iteratively improve it through local modifications. For MAX-CUT, flipping vertices to increase the cut gives a $1/2$-approximation (ratio 2).

## Approximability Landscape

Problems fall into distinct categories based on their best achievable approximation ratio:

| Category | Best Ratio | Examples |
|----------|-----------|---------|
| Exact in P | 1 | Shortest paths, matching, MST |
| FPTAS | $(1 + \epsilon)$ for any $\epsilon$ | Knapsack |
| PTAS | $(1 + \epsilon)$ for any $\epsilon$ | Euclidean TSP |
| Constant-factor (APX) | Fixed $\rho$ | Vertex Cover (2), Metric TSP (3/2) |
| Logarithmic | $O(\log n)$ | Set Cover |
| Polynomial | $O(n^c)$ | Independent Set (planar graphs) |
| Inapproximable | No finite ratio | General TSP, Chromatic Number ($n^{1-\epsilon}$) |

## Key Results Summary

| Problem | Algorithm | Ratio | Lower Bound |
|---------|-----------|-------|-------------|
| Vertex Cover | Maximal Matching | 2 | $2 - \epsilon$ (UGC) |
| Set Cover | Greedy | $\ln n$ | $(1-\epsilon) \ln n$ |
| Metric TSP | Christofides | 3/2 | Open (no better than APX-hard) |
| MAX-CUT | Goemans-Williamson | $\approx 0.878$ | $\approx 0.878$ (UGC) |
| MAX-3SAT | Semidefinite | 7/8 | $7/8 + \epsilon$ |
| Knapsack | FPTAS | $1 + \epsilon$ | No FPTAS for strongly NP-hard |

## Approximation vs Other Coping Strategies

| Strategy | Guarantee | Trade-off |
|----------|-----------|-----------|
| **Approximation** | Provable ratio on solution quality | Worst-case ratio may be loose |
| **Parameterized** | Exact solution, FPT in parameter | Exponential in parameter |
| **Heuristic** | No worst-case guarantee | Often works well in practice |
| **Exponential exact** | Optimal solution | Exponential time |
| **Pseudo-polynomial** | Exact for bounded inputs | Not truly polynomial |

Approximation is the preferred approach when a provable quality guarantee matters and the problem admits a reasonable ratio.

## When Approximation Fails

Some problems resist approximation entirely:

!!! warning "Inapproximability Examples"
    - **General TSP:** No finite approximation ratio unless P = NP.
    - **Clique:** Cannot be approximated within $n^{1-\epsilon}$ for any $\epsilon > 0$ unless P = NP.
    - **Chromatic Number:** Same hardness as Clique.

For these problems, other coping strategies (heuristics, parameterized algorithms, special-case structure) are necessary.

??? example "Example: Greedy Set Cover"
    **Universe:** $U = \{1, 2, 3, 4, 5, 6\}$.

    **Sets:** $S_1 = \{1, 2, 3\}$, $S_2 = \{2, 4, 5\}$, $S_3 = \{3, 5, 6\}$, $S_4 = \{1, 6\}$.

    **Greedy execution:**

    1. $S_1$ covers 3 elements (most). Select $S_1$. Uncovered: $\{4, 5, 6\}$.
    2. $S_2$ covers $\{4, 5\}$ (2 elements), $S_3$ covers $\{5, 6\}$ (2 elements). Pick $S_2$. Uncovered: $\{6\}$.
    3. $S_3$ covers $\{6\}$. Select $S_3$.

    **Greedy solution:** $\{S_1, S_2, S_3\}$, size 3.

    **Optimal:** $\{S_1, S_3\}$ covers everything, size 2.

    **Ratio:** $3/2 = 1.5 \leq H_6 \approx 2.45$. Guarantee holds.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press, Chapter 35.
- Vazirani, V. V. (2001). *Approximation Algorithms*. Springer.
- Williamson, D. P., & Shmoys, D. B. (2011). *The Design of Approximation Algorithms*. Cambridge University Press.
