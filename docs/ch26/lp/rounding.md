# Deterministic LP Rounding

After solving an LP relaxation, the fractional solution must be converted to an integer solution. **Deterministic rounding** applies fixed rules to map fractional values to integers. Unlike randomized rounding, these schemes are deterministic and produce the same output on every run. This page covers threshold rounding, iterative rounding, and their applications.

## Threshold Rounding

The simplest deterministic rounding strategy sets each variable to 1 if its LP value exceeds a threshold $\theta$.

!!! tip "Definition: Threshold Rounding"
    Given an LP optimal solution $x^*$, define the rounded solution:

    $$
    \hat{x}_j = \begin{cases} 1 & \text{if } x_j^* \geq \theta \\ 0 & \text{otherwise} \end{cases}
    $$

    The threshold $\theta$ is chosen to guarantee feasibility and a bounded approximation ratio.

### Application: Vertex Cover

For vertex cover, $\theta = 1/2$ works. Every edge constraint $x_u^* + x_v^* \geq 1$ ensures at least one endpoint has $x^* \geq 1/2$, so the rounded solution is a valid cover.

**Ratio analysis.** Each rounded variable satisfies $\hat{x}_v \leq 2 x_v^*$:

$$
\sum_v \hat{x}_v \leq 2 \sum_v x_v^* = 2 \cdot \text{OPT}_{\text{LP}} \leq 2 \cdot \text{OPT}
$$

This gives a 2-approximation.

### Application: Weighted Set Cover

For set cover with element frequency $f$ (maximum number of sets containing any element), use $\theta = 1/f$.

Each element $i$ has $\sum_{j : i \in S_j} x_j^* \geq 1$, so at least one set containing $i$ has $x_j^* \geq 1/f$, ensuring element $i$ is covered.

The ratio becomes $f$: each $\hat{x}_j \leq f \cdot x_j^*$.

## Iterative Rounding

**Iterative rounding**, introduced by Jain (2001) for the Steiner network problem, is a more sophisticated technique that repeatedly solves the LP, rounds variables, and re-solves.

### Algorithm

1. Solve the LP relaxation.
2. If any variable $x_j^*$ is integral (0 or 1), fix it and remove it from the LP.
3. If a structural argument guarantees some variable has $x_j^* \geq 1/2$ (or another threshold), round it up and fix it.
4. Re-solve the reduced LP and repeat.
5. Return the accumulated integer solution.

### Key Insight

The power of iterative rounding comes from the **rank lemma**: in a basic feasible solution (vertex of the LP polytope), the number of non-zero variables equals the number of tight constraints. As variables are fixed, the LP shrinks, and the remaining basic feasible solution often has "large" fractional values that can be rounded with small loss.

### Application: Degree-Bounded Spanning Tree

Iterative rounding achieves a spanning tree whose maximum degree exceeds the optimal by at most 1 (an additive guarantee). At each iteration, the basic feasible solution has some edge variable $x_e^* = 1$ or a degree constraint that can be dropped (relaxed). This structural property ensures progress.

## Pipage Rounding

**Pipage rounding** (Ageev and Sviridenko, 2004) converts a fractional solution to an integer solution by iteratively modifying pairs of variables while maintaining (or improving) the objective value.

### Procedure

Given fractional $x^*$ with $x_i^*, x_j^* \in (0, 1)$:

1. Choose two fractional variables $x_i, x_j$.
2. Define $\epsilon_1 = \min(x_i^*, 1 - x_j^*)$ and $\epsilon_2 = \min(1 - x_i^*, x_j^*)$.
3. Either set $(x_i, x_j) \leftarrow (x_i^* - \epsilon_1, x_j^* + \epsilon_1)$ or $(x_i, x_j) \leftarrow (x_i^* + \epsilon_2, x_j^* - \epsilon_2)$, choosing the option that does not decrease the objective.
4. Repeat until all variables are integral.

Each step reduces the number of fractional variables by at least one, so the procedure terminates in at most $n$ steps.

## Comparison of Rounding Methods

| Method | Type | Key Property | Example Application |
|--------|------|--------------|-------------------|
| Threshold | Deterministic | Simple, one-pass | Vertex Cover (ratio 2) |
| Randomized | Probabilistic | Uses LP values as probabilities | MAX-SAT (ratio $1 - 1/e$) |
| Iterative | Deterministic | Re-solves LP after each fix | Network design |
| Pipage | Deterministic | Pairs of variables, maintains objective | Submodular maximization |

??? example "Worked Example: Threshold Rounding for Vertex Cover"
    **Graph:** Path $a - b - c - d$ with edges $\{(a,b), (b,c), (c,d)\}$.

    **LP solution:** $x_a^* = 0, x_b^* = 1, x_c^* = 0.5, x_d^* = 0.5$.

    **Verify constraints:**

    - $(a,b)$: $0 + 1 = 1 \geq 1$
    - $(b,c)$: $1 + 0.5 = 1.5 \geq 1$
    - $(c,d)$: $0.5 + 0.5 = 1 \geq 1$

    **LP cost:** $0 + 1 + 0.5 + 0.5 = 2$.

    **Rounding with $\theta = 1/2$:** $\hat{x}_a = 0, \hat{x}_b = 1, \hat{x}_c = 1, \hat{x}_d = 1$. Cover $= \{b, c, d\}$, cost $= 3$.

    **Optimal:** $\{b, c\}$ covers all edges, cost $= 2$.

    **Ratio:** $3/2 = 1.5 \leq 2$. The guarantee holds.

## Reference

- Vazirani, V. V. (2001). *Approximation Algorithms*. Springer.
- Jain, K. (2001). A factor 2 approximation algorithm for the generalized Steiner network problem. *Combinatorica*, 21(1), 39--60.
- Williamson, D. P., & Shmoys, D. B. (2011). *The Design of Approximation Algorithms*. Cambridge University Press.
