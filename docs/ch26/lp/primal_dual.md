# Primal-Dual Method

The **primal-dual method** is a powerful framework for designing approximation algorithms. Rather than solving an LP and rounding, it constructs a feasible integer solution and a feasible dual solution simultaneously, using complementary slackness conditions to guide the construction. The dual solution provides a lower bound on OPT, enabling us to prove approximation guarantees without actually solving an LP.

## LP Duality Background

Consider a minimization problem formulated as an integer program, relaxed to an LP:

**Primal LP:**

$$
\min \sum_{j} c_j x_j \quad \text{s.t.} \quad \sum_{j} a_{ij} x_j \geq b_i \;\; \forall i, \quad x_j \geq 0
$$

**Dual LP:**

$$
\max \sum_{i} b_i y_i \quad \text{s.t.} \quad \sum_{i} a_{ij} y_i \leq c_j \;\; \forall j, \quad y_i \geq 0
$$

By weak duality, any feasible dual solution provides a lower bound on the primal optimum (and hence on the integer optimum):

$$
\sum_{i} b_i y_i \leq \text{OPT}_{\text{LP}} \leq \text{OPT}_{\text{IP}}
$$

## Complementary Slackness

At optimality, the primal and dual LP solutions satisfy **complementary slackness**:

- **Primal CS:** If $x_j > 0$, then $\sum_{i} a_{ij} y_i = c_j$ (the dual constraint for $j$ is tight).
- **Dual CS:** If $y_i > 0$, then $\sum_{j} a_{ij} x_j = b_i$ (the primal constraint for $i$ is tight).

The primal-dual method relaxes these conditions, allowing a bounded violation:

- **Relaxed Primal CS:** If $x_j > 0$, then $\frac{c_j}{\rho} \leq \sum_{i} a_{ij} y_i \leq c_j$.

If we can construct an integer primal solution $x$ and a feasible dual $y$ satisfying this relaxed condition, then:

$$
\sum_j c_j x_j \leq \rho \sum_i b_i y_i \leq \rho \cdot \text{OPT}
$$

giving a $\rho$-approximation.

## Generic Primal-Dual Schema

**Input:** An LP relaxation of a covering/packing problem.

1. Initialize: set all dual variables $y_i = 0$, primal $x_j = 0$.
2. While some primal constraint is violated (the solution is infeasible):
    - Select a violated constraint $i$.
    - Raise $y_i$ until some dual constraint $j$ becomes tight.
    - Set $x_j = 1$ (add element $j$ to the solution).
    - Update: mark constraints newly satisfied by $x_j$.
3. Return the primal solution $x$ and dual solution $y$.

## Application: Weighted Vertex Cover

### LP Formulation

For a graph $G = (V, E)$ with vertex weights $w_v$:

**Primal:**

$$
\min \sum_{v \in V} w_v x_v \quad \text{s.t.} \quad x_u + x_v \geq 1 \;\; \forall (u,v) \in E, \quad x_v \geq 0
$$

**Dual:**

$$
\max \sum_{(u,v) \in E} y_{uv} \quad \text{s.t.} \quad \sum_{(u,v) \in E} y_{uv} \leq w_v \;\; \forall v \in V, \quad y_{uv} \geq 0
$$

### Primal-Dual Algorithm

1. Set all $y_{uv} = 0$.
2. For each uncovered edge $(u, v)$:
    - Raise $y_{uv}$ until the dual constraint for $u$ or $v$ becomes tight: $\sum_{e \ni v} y_e = w_v$.
    - Add the tight vertex to the cover $C$.
3. Return $C$.

### Analysis

!!! tip "Theorem: 2-Approximation for Weighted Vertex Cover"
    The primal-dual algorithm produces a vertex cover $C$ with $w(C) \leq 2 \cdot \text{OPT}$.

**Proof.** The dual solution $y$ is feasible by construction (we stop raising when a constraint becomes tight). Each vertex $v \in C$ has its dual constraint tight:

$$
w(C) = \sum_{v \in C} w_v = \sum_{v \in C} \sum_{e \ni v} y_e \leq 2 \sum_{e \in E} y_e
$$

The factor 2 arises because each edge contributes to at most two vertex constraints. By weak duality:

$$
\sum_{e \in E} y_e \leq \text{OPT}
$$

Therefore $w(C) \leq 2 \cdot \text{OPT}$. $\square$

## Application: Set Cover

For universe $U$ and sets $S_1, \ldots, S_m$ with costs $c_j$, the primal-dual method yields an $f$-approximation where $f = \max_i |\{j : i \in S_j\}|$ is the maximum frequency of any element.

The dual has a variable $y_i$ for each element $i \in U$:

$$
\max \sum_{i \in U} y_i \quad \text{s.t.} \quad \sum_{i \in S_j} y_i \leq c_j \;\; \forall j, \quad y_i \geq 0
$$

The algorithm raises dual variables for uncovered elements until some set becomes tight, then adds that set.

## Advantages of the Primal-Dual Method

1. **No LP solver needed.** The algorithm is combinatorial --- it never solves the LP explicitly.
2. **Strong guarantees.** The dual provides a certificate of near-optimality.
3. **Efficiency.** Typically runs in nearly linear time in the problem size.

??? example "Worked Example: Weighted Vertex Cover"
    **Graph:** $V = \{a, b, c, d\}$, edges $\{(a,b), (b,c), (c,d)\}$, weights $w_a = 3, w_b = 2, w_c = 4, w_d = 1$.

    **Step 1:** Edge $(a,b)$ uncovered. Raise $y_{ab}$ until tight. $w_b = 2$ is smaller, so $y_{ab} = 2$, vertex $b$ becomes tight. Add $b$ to $C$.

    **Step 2:** Edge $(b,c)$ is now covered by $b$. Edge $(c,d)$ uncovered. Raise $y_{cd}$ until tight. $w_d = 1$ is smaller, so $y_{cd} = 1$, vertex $d$ becomes tight. Add $d$ to $C$.

    **Result:** $C = \{b, d\}$, cost $= 2 + 1 = 3$. Dual value $= y_{ab} + y_{cd} = 2 + 1 = 3$.

    **OPT** $\geq 3$ (dual bound). Ratio: $3/3 = 1 \leq 2$.

## Reference

- Vazirani, V. V. (2001). *Approximation Algorithms*. Springer, Chapters 12--15.
- Williamson, D. P., & Shmoys, D. B. (2011). *The Design of Approximation Algorithms*. Cambridge University Press, Chapter 7.
