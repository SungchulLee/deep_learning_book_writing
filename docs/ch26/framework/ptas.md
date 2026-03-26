# PTAS and FPTAS

For some NP-hard optimization problems, we can get arbitrarily close to the optimum --- at the cost of increased running time. A **Polynomial-Time Approximation Scheme (PTAS)** lets the user choose any desired accuracy $\epsilon > 0$, and a **Fully Polynomial-Time Approximation Scheme (FPTAS)** additionally guarantees that the running time scales polynomially in $1/\epsilon$. This page defines these concepts, relates them to the approximation hierarchy, and illustrates with the knapsack FPTAS.

## PTAS Definition

!!! tip "Definition: PTAS"
    A **Polynomial-Time Approximation Scheme** for a minimization problem $\Pi$ is a family of algorithms $\{A_\epsilon\}_{\epsilon > 0}$ such that for every $\epsilon > 0$ and every instance $I$:

    $$
    A_\epsilon(I) \leq (1 + \epsilon) \cdot \text{OPT}(I)
    $$

    and $A_\epsilon$ runs in time polynomial in $|I|$ (but not necessarily in $1/\epsilon$).

For a maximization problem, the guarantee becomes $A_\epsilon(I) \geq (1 - \epsilon) \cdot \text{OPT}(I)$.

The running time may be $O(n^{1/\epsilon})$ or $O(n^{2^{1/\epsilon}})$ --- polynomial in $n$ for each fixed $\epsilon$, but potentially impractical for small $\epsilon$.

## FPTAS Definition

!!! tip "Definition: FPTAS"
    A **Fully Polynomial-Time Approximation Scheme** is a PTAS whose running time is polynomial in both $|I|$ and $1/\epsilon$.

Typical FPTAS running times look like $O(n^2 / \epsilon)$ or $O(n^3 / \epsilon^2)$. An FPTAS is the strongest type of approximation result: for any desired accuracy, the algorithm runs in practical polynomial time.

## Approximation Hierarchy

The containment structure from strongest to weakest:

$$
\text{FPTAS} \subset \text{PTAS} \subset \text{APX} \subset \text{NPO}
$$

- **FPTAS:** Polynomial in $n$ and $1/\epsilon$.
- **PTAS:** Polynomial in $n$ for each fixed $\epsilon$, but may be exponential in $1/\epsilon$.
- **APX:** Admits a constant-factor approximation for some fixed ratio.
- **NPO:** The class of all NP optimization problems.

| Class | Example Problems |
|-------|-----------------|
| FPTAS | Knapsack, Scheduling on identical machines |
| PTAS (not FPTAS) | Euclidean TSP, Bin Packing |
| APX (not PTAS) | MAX-3SAT, Vertex Cover, Metric TSP |
| NPO (not APX) | Clique, Chromatic Number |

## FPTAS for Knapsack

The 0/1 Knapsack problem is NP-hard, yet it admits an FPTAS via a **scaling and rounding** technique applied to the exact dynamic programming solution.

### Setup

Given $n$ items with values $v_1, \ldots, v_n$ and weights $w_1, \ldots, w_n$, and a capacity $W$, maximize $\sum_{i \in S} v_i$ subject to $\sum_{i \in S} w_i \leq W$.

The exact DP runs in $O(n \cdot V)$ time where $V = \sum_i v_i$ --- pseudo-polynomial, not polynomial in the input size.

### Scaling Strategy

The idea is to round down the values to reduce $V$ while controlling the error.

1. Let $v_{\max} = \max_i v_i$.
2. Set the scaling factor $K = \frac{\epsilon \cdot v_{\max}}{n}$.
3. Define scaled values $\hat{v}_i = \lfloor v_i / K \rfloor$ for each item $i$.
4. Run the exact DP on the scaled instance $(\hat{v}_1, \ldots, \hat{v}_n, w_1, \ldots, w_n, W)$.
5. Return the solution found.

### Analysis

**Running time.** The maximum scaled value is:

$$
\hat{v}_{\max} = \left\lfloor \frac{v_{\max}}{K} \right\rfloor = \left\lfloor \frac{n}{\epsilon} \right\rfloor
$$

The DP runs in $O(n \cdot n \cdot \hat{v}_{\max}) = O(n^3 / \epsilon)$, which is polynomial in both $n$ and $1/\epsilon$.

**Approximation ratio.** Let $S^*$ be the optimal solution and $\hat{S}$ be the solution found by the scaled DP.

For each item $i$, the rounding error satisfies $v_i - K \hat{v}_i < K$. Summing over items in $S^*$:

$$
\sum_{i \in S^*} v_i - K \sum_{i \in S^*} \hat{v}_i < n \cdot K = \epsilon \cdot v_{\max}
$$

Since $\hat{S}$ is optimal for the scaled instance, $\sum_{i \in \hat{S}} \hat{v}_i \geq \sum_{i \in S^*} \hat{v}_i$, so:

$$
\sum_{i \in \hat{S}} v_i \geq K \sum_{i \in \hat{S}} \hat{v}_i \geq K \sum_{i \in S^*} \hat{v}_i > \sum_{i \in S^*} v_i - \epsilon \cdot v_{\max} \geq (1 - \epsilon) \cdot \text{OPT}
$$

The last inequality uses $\text{OPT} \geq v_{\max}$. $\square$

## PTAS for Euclidean TSP

Arora (1998) and Mitchell (1999) independently showed that TSP in Euclidean space admits a PTAS. Given $n$ points in $\mathbb{R}^2$ with Euclidean distances, for any $\epsilon > 0$ there exists an algorithm producing a tour of length at most $(1 + \epsilon) \cdot \text{OPT}$ in time $n \cdot (\log n)^{O(1/\epsilon)}$.

The key idea uses a **randomly shifted quadtree** decomposition. The algorithm:

1. Enclose all points in a bounding square.
2. Apply a random shift and recursively subdivide into quadrants.
3. Restrict crossings at each level to a bounded number of "portals."
4. Solve the resulting structured subproblem via dynamic programming.

This is a PTAS but not an FPTAS: the running time is exponential in $1/\epsilon$.

## When FPTAS Is Impossible

Not every problem with a PTAS has an FPTAS. Under standard complexity assumptions:

!!! warning "Theorem"
    If a **strongly NP-hard** problem has an FPTAS, then P = NP.

A problem is strongly NP-hard if it remains NP-hard even when all numbers in the input are bounded by a polynomial in $n$. Examples include Bin Packing and 3-Partition. Since an FPTAS would solve the pseudo-polynomial DP in truly polynomial time, it would resolve the strongly NP-hard instances.

??? example "Worked Example: Knapsack FPTAS"
    **Instance:** 4 items, capacity $W = 10$.

    | Item | Value | Weight |
    |------|-------|--------|
    | 1    | 100   | 5      |
    | 2    | 60    | 3      |
    | 3    | 120   | 7      |
    | 4    | 80    | 4      |

    Set $\epsilon = 0.2$. Then $v_{\max} = 120$, $K = \frac{0.2 \times 120}{4} = 6$.

    **Scaled values:** $\hat{v}_1 = \lfloor 100/6 \rfloor = 16$, $\hat{v}_2 = \lfloor 60/6 \rfloor = 10$, $\hat{v}_3 = \lfloor 120/6 \rfloor = 20$, $\hat{v}_4 = \lfloor 80/6 \rfloor = 13$.

    **Scaled DP optimum:** Select items 1 and 4 (weight $5+4=9 \leq 10$, scaled value $16+13=29$).

    **True value:** $100 + 80 = 180$. **OPT:** Items 1 and 3 have weight $12 > 10$; items 1 and 4 give $180$; items 2 and 3 give weight $10$, value $180$. So OPT $= 180$.

    **Ratio:** $180 / 180 = 1.0 \geq 1 - \epsilon = 0.8$. The guarantee holds.

## Reference

- Vazirani, V. V. (2001). *Approximation Algorithms*. Springer.
- Arora, S. (1998). Polynomial time approximation schemes for Euclidean traveling salesman and other geometric problems. *JACM*, 45(5), 753--782.
- Williamson, D. P., & Shmoys, D. B. (2011). *The Design of Approximation Algorithms*. Cambridge University Press.
