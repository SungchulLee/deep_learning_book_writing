# LP Relaxation

Many NP-hard combinatorial optimization problems can be formulated as **integer linear programs (ILPs)**. Solving ILPs exactly is NP-hard, but replacing the integrality constraint $x \in \{0, 1\}$ with $x \in [0, 1]$ yields a **linear program (LP)** solvable in polynomial time. The LP optimal value provides a bound on the true optimum, and the fractional solution serves as a starting point for rounding techniques that produce near-optimal integer solutions.

## From ILP to LP Relaxation

### Integer Linear Program

A combinatorial optimization problem is often expressed as:

$$
\min \mathbf{c}^\top \mathbf{x} \quad \text{s.t.} \quad A\mathbf{x} \geq \mathbf{b}, \quad \mathbf{x} \in \{0, 1\}^n
$$

Each binary variable $x_j$ represents a yes/no decision (include item $j$ or not).

### LP Relaxation

Replace $x_j \in \{0, 1\}$ with $0 \leq x_j \leq 1$:

$$
\min \mathbf{c}^\top \mathbf{x} \quad \text{s.t.} \quad A\mathbf{x} \geq \mathbf{b}, \quad \mathbf{0} \leq \mathbf{x} \leq \mathbf{1}
$$

The feasible region of the ILP is a subset of the LP feasible region. Therefore:

- **Minimization:** $\text{OPT}_{\text{LP}} \leq \text{OPT}_{\text{ILP}}$ (LP provides a lower bound).
- **Maximization:** $\text{OPT}_{\text{LP}} \geq \text{OPT}_{\text{ILP}}$ (LP provides an upper bound).

## Integrality Gap

The **integrality gap** measures how much the relaxation can deviate from the integer optimum.

!!! tip "Definition: Integrality Gap"
    For a minimization problem, the integrality gap is:

    $$
    \text{IG} = \sup_{I} \frac{\text{OPT}_{\text{ILP}}(I)}{\text{OPT}_{\text{LP}}(I)}
    $$

    For maximization, the ratio is inverted.

The integrality gap limits the approximation ratio achievable by any LP-rounding algorithm using this particular relaxation. If IG $= \alpha$, then no rounding scheme can guarantee a ratio better than $\alpha$.

### Example: Vertex Cover

The vertex cover LP has integrality gap 2. The worst case occurs on odd cycles: for a triangle $K_3$, the LP sets $x_v = 1/2$ for all vertices, giving $\text{OPT}_{\text{LP}} = 3/2$, while $\text{OPT}_{\text{ILP}} = 2$. More generally, odd complete graphs achieve gap approaching 2.

### Example: Set Cover

The set cover LP has integrality gap $\Theta(\log n)$, matching the greedy algorithm's ratio. This means the LP relaxation is tight enough to support an $O(\log n)$-approximation but not better.

## LP Relaxation for Vertex Cover

**ILP:**

$$
\min \sum_{v \in V} x_v \quad \text{s.t.} \quad x_u + x_v \geq 1 \;\; \forall (u,v) \in E, \quad x_v \in \{0, 1\}
$$

**LP relaxation:**

$$
\min \sum_{v \in V} x_v \quad \text{s.t.} \quad x_u + x_v \geq 1 \;\; \forall (u,v) \in E, \quad 0 \leq x_v \leq 1
$$

**Half-integrality property.** The vertex cover LP always has an optimal solution where every $x_v \in \{0, 1/2, 1\}$. This structure simplifies rounding: set $x_v = 1$ whenever $x_v^* \geq 1/2$.

## LP Relaxation for Set Cover

Given universe $U$ and sets $S_1, \ldots, S_m$ with costs $c_j$:

$$
\min \sum_{j=1}^{m} c_j x_j \quad \text{s.t.} \quad \sum_{j : i \in S_j} x_j \geq 1 \;\; \forall i \in U, \quad 0 \leq x_j \leq 1
$$

The LP dual assigns a price $y_i$ to each element:

$$
\max \sum_{i \in U} y_i \quad \text{s.t.} \quad \sum_{i \in S_j} y_i \leq c_j \;\; \forall j, \quad y_i \geq 0
$$

This dual interpretation says: assign prices to elements such that no set is "overpriced" (the sum of element prices in any set does not exceed its cost).

## Strengthening LP Relaxations

When the integrality gap is too large, the LP can be tightened:

1. **Adding valid inequalities.** Constraints that are redundant for the LP but tighten the feasible region closer to the integer hull. For example, clique inequalities for independent set.

2. **Lift-and-project.** Systematic methods (Sherali-Adams, Lasserre hierarchies) that add auxiliary variables and constraints to narrow the gap.

3. **Semidefinite relaxation (SDP).** Replace LP with a semidefinite program. The Goemans-Williamson MAX-CUT algorithm uses an SDP relaxation to achieve ratio $\approx 0.878$, beating any LP-based approach.

??? example "Worked Example: LP Relaxation for Vertex Cover"
    **Graph:** $V = \{a, b, c\}$, edges $\{(a,b), (b,c), (a,c)\}$ (triangle $K_3$).

    **LP constraints:**

    - $x_a + x_b \geq 1$
    - $x_b + x_c \geq 1$
    - $x_a + x_c \geq 1$
    - $0 \leq x_a, x_b, x_c \leq 1$

    **LP optimum:** $x_a = x_b = x_c = 1/2$, cost $= 3/2$.

    **Rounding:** All values $\geq 1/2$, so all vertices are included. $|C| = 3$.

    **ILP optimum:** Any two vertices suffice. $\text{OPT} = 2$.

    **Integrality gap for this instance:** $2 / (3/2) = 4/3$.

    **Approximation ratio:** $3 / 2 = 1.5 \leq 2$. The 2-approximation guarantee holds.

## Reference

- Vazirani, V. V. (2001). *Approximation Algorithms*. Springer, Chapters 12--14.
- Williamson, D. P., & Shmoys, D. B. (2011). *The Design of Approximation Algorithms*. Cambridge University Press, Chapter 1.
- Bertsimas, D., & Tsitsiklis, J. N. (1997). *Introduction to Linear Optimization*. Athena Scientific.
