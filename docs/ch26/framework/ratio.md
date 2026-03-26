# Approximation Ratio

When an NP-hard optimization problem cannot be solved exactly in polynomial time, we settle for approximate solutions. The **approximation ratio** quantifies how close an algorithm's output is to the true optimum. This measure serves as the primary yardstick for comparing approximation algorithms and classifying problems by their approximability.

## Formal Definition

Consider an optimization problem $\Pi$ with instances $I$. Let $\text{OPT}(I)$ denote the optimal objective value and $A(I)$ denote the value returned by algorithm $A$.

!!! tip "Definition: Approximation Ratio"
    An algorithm $A$ has **approximation ratio** $\rho \geq 1$ if for every instance $I$:

    **Minimization:**

    $$
    A(I) \leq \rho \cdot \text{OPT}(I)
    $$

    **Maximization:**

    $$
    A(I) \geq \frac{1}{\rho} \cdot \text{OPT}(I)
    $$

A $\rho$-approximation algorithm is one that achieves ratio $\rho$. The ratio is always at least 1, with $\rho = 1$ corresponding to an exact solution.

### Alternative Convention

Some texts define the ratio as a value in $(0, 1]$ for maximization:

$$
\alpha = \frac{A(I)}{\text{OPT}(I)} \geq \alpha \quad \text{(maximization, } \alpha \leq 1\text{)}
$$

For example, the Goemans-Williamson MAX-CUT algorithm achieves $\alpha \approx 0.878$, which corresponds to $\rho \approx 1.139$ in the $\rho \geq 1$ convention.

## Worst-Case vs Expected

The ratio can be defined in two ways:

**Worst-case ratio.** For every instance $I$ of size $n$:

$$
\rho_n = \max_{|I| = n} \frac{A(I)}{\text{OPT}(I)} \quad \text{(minimization)}
$$

The overall approximation ratio is $\rho = \sup_n \rho_n$.

**Expected ratio.** For randomized algorithms, the guarantee becomes:

$$
\mathbb{E}[A(I)] \leq \rho \cdot \text{OPT}(I)
$$

where the expectation is over the algorithm's random choices. The instance $I$ is worst-case (this is not an average-case guarantee).

## Absolute vs Relative Guarantees

Beyond multiplicative ratios, some algorithms provide **additive** guarantees:

$$
|A(I) - \text{OPT}(I)| \leq c
$$

for a constant $c$. For example, edge coloring can always be done with at most $\Delta + 1$ colors (Vizing's theorem), where $\Delta$ is the maximum degree and OPT $\geq \Delta$.

Multiplicative guarantees are more common because they scale with instance size.

## Key Examples

| Problem | Algorithm | Ratio $\rho$ | Type |
|---------|-----------|-------------|------|
| Vertex Cover | Maximal Matching | 2 | Minimization |
| Metric TSP | Christofides | 3/2 | Minimization |
| Set Cover | Greedy | $H_n \approx \ln n$ | Minimization |
| MAX-CUT | Goemans-Williamson | $\approx 1.139$ | Maximization |
| Knapsack | FPTAS | $1 + \epsilon$ | Maximization |
| MAX-3SAT | Random Assignment | $8/7$ | Maximization |

## Approximation Classes

The ratio determines which approximation class a problem belongs to:

| Class | Ratio | Example |
|-------|-------|---------|
| **FPTAS** | $(1 + \epsilon)$ for any $\epsilon$, poly in $n$ and $1/\epsilon$ | Knapsack |
| **PTAS** | $(1 + \epsilon)$ for any $\epsilon$, poly in $n$ | Euclidean TSP |
| **APX** | Constant $\rho$ | Vertex Cover ($\rho = 2$) |
| **Log-APX** | $O(\log n)$ | Set Cover |
| **Poly-APX** | $O(n^c)$ for some $c < 1$ | Independent Set (planar) |

A problem that is **APX-hard** admits no PTAS unless P = NP. This classification helps determine the best achievable ratio.

## Proving Approximation Ratios

The standard proof strategy has two components:

1. **Lower bound on OPT.** Find a quantity $L$ computable in polynomial time such that $L \leq \text{OPT}$. Common choices include LP relaxation values, matching sizes, and spanning tree costs.

2. **Upper bound on $A(I)$.** Show that the algorithm's output satisfies $A(I) \leq \rho \cdot L$.

Combining gives $A(I) \leq \rho \cdot L \leq \rho \cdot \text{OPT}$.

??? example "Example: Vertex Cover Ratio Proof"
    **Algorithm.** Find a maximal matching $M$, return both endpoints.

    **Lower bound.** Any vertex cover must include at least one endpoint of each edge in $M$. Since $M$ is a matching (no shared endpoints): $\text{OPT} \geq |M|$.

    **Upper bound.** The algorithm returns $2|M|$ vertices.

    **Ratio.** $\frac{A(I)}{\text{OPT}(I)} \leq \frac{2|M|}{|M|} = 2$.

    This ratio is tight: consider a graph consisting of $k$ disjoint edges. The optimal cover picks one endpoint per edge ($|M| = k$), while the algorithm picks both ($2k$).

## Asymptotic vs Absolute Ratios

For some problems, the ratio depends on the input size $n$:

- **Set Cover:** best ratio is $\Theta(\log n)$, meaning both the algorithm and the lower bound scale logarithmically.
- **Clique:** best polynomial-time ratio is $\Theta(n)$, meaning essentially no useful approximation is possible.

The distinction between constant-ratio and growing-ratio problems is fundamental in approximation theory.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press, Chapter 35.
- Vazirani, V. V. (2001). *Approximation Algorithms*. Springer.
- Williamson, D. P., & Shmoys, D. B. (2011). *The Design of Approximation Algorithms*. Cambridge University Press.
