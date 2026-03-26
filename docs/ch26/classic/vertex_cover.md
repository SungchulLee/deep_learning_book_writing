# Vertex Cover Approximation

Given a graph, a **vertex cover** is a set of vertices that touches every edge. Finding a minimum vertex cover is NP-hard, yet a simple greedy algorithm achieves a 2-approximation --- and this ratio is essentially the best possible unless P = NP. This page presents the matching-based algorithm and an LP-relaxation approach, both achieving ratio 2.

## Problem Definition

Given an undirected graph $G = (V, E)$, a **vertex cover** is a subset $S \subseteq V$ such that for every edge $(u, v) \in E$, at least one of $u$ or $v$ belongs to $S$. The **Minimum Vertex Cover** problem asks for a vertex cover of minimum cardinality $|S|$.

Let $\text{OPT}$ denote the size of an optimal vertex cover.

## Matching-Based 2-Approximation

The key insight is that a **maximal matching** --- a matching to which no edge can be added --- immediately yields a vertex cover. Every matched edge requires at least one of its endpoints in any cover, but we take both, at most doubling the optimal.

### Algorithm

**Input:** Undirected graph $G = (V, E)$.

1. Initialize $C \leftarrow \emptyset$ and $E' \leftarrow E$.
2. While $E' \neq \emptyset$:
    - Pick any edge $(u, v) \in E'$.
    - Add both $u$ and $v$ to $C$.
    - Remove from $E'$ all edges incident to $u$ or $v$.
3. Return $C$.

### Correctness

Every edge in $E$ is either picked in step 2 (and both endpoints are in $C$) or is incident to a vertex already in $C$ (removed in the same step). Therefore $C$ is a vertex cover.

### Approximation Ratio

!!! tip "Theorem: 2-Approximation"
    The matching-based algorithm returns a vertex cover $C$ with $|C| \leq 2 \cdot \text{OPT}$.

**Proof.** Let $M$ be the set of edges picked by the algorithm. These edges form a matching (no two share an endpoint, since we remove all incident edges after each pick). The algorithm outputs $|C| = 2|M|$.

Any vertex cover must include at least one endpoint of every edge in $M$ (since no two edges in $M$ share an endpoint). Therefore:

$$
\text{OPT} \geq |M|
$$

Combining:

$$
|C| = 2|M| \leq 2 \cdot \text{OPT}
$$

$\square$

The algorithm runs in $O(|V| + |E|)$ time --- simply scan the edge list once.

## LP Relaxation Approach

An alternative path to the same ratio uses linear programming.

### Integer Program Formulation

Assign a variable $x_v \in \{0, 1\}$ to each vertex. The minimum vertex cover is:

$$
\min \sum_{v \in V} x_v \quad \text{subject to} \quad x_u + x_v \geq 1 \;\; \forall (u,v) \in E, \quad x_v \in \{0, 1\}
$$

### LP Relaxation

Relax the integrality constraint to $x_v \in [0, 1]$:

$$
\min \sum_{v \in V} x_v \quad \text{subject to} \quad x_u + x_v \geq 1 \;\; \forall (u,v) \in E, \quad 0 \leq x_v \leq 1
$$

Let $\text{OPT}_{\text{LP}}$ denote the LP optimum. Since the LP is a relaxation, $\text{OPT}_{\text{LP}} \leq \text{OPT}$.

### Rounding

Solve the LP and round:

$$
\hat{x}_v = \begin{cases} 1 & \text{if } x_v^* \geq 1/2 \\ 0 & \text{otherwise} \end{cases}
$$

**Correctness.** For every edge $(u, v)$, the constraint $x_u^* + x_v^* \geq 1$ ensures at least one of $x_u^*, x_v^* \geq 1/2$, so at least one endpoint is rounded to 1.

**Ratio.** Each $\hat{x}_v \leq 2 x_v^*$, so:

$$
\sum_{v} \hat{x}_v \leq 2 \sum_{v} x_v^* = 2 \cdot \text{OPT}_{\text{LP}} \leq 2 \cdot \text{OPT}
$$

$\square$

## Integrality Gap

The integrality gap of the vertex cover LP equals 2, achieved by the complete graph $K_n$ on odd $n$. The LP optimum assigns $x_v = 1/2$ for all vertices, giving $\text{OPT}_{\text{LP}} = n/2$, while the integer optimum is $\text{OPT} = (n-1)/2 \cdot 2/(n-1) \cdot \lceil (n-1)/2 \rceil = n - 1$ for $K_n$.

This means no rounding scheme for this LP can beat ratio 2.

## Hardness of Improvement

!!! warning "Inapproximability"
    Under the Unique Games Conjecture, no polynomial-time algorithm achieves a ratio better than $2 - \epsilon$ for any constant $\epsilon > 0$. Unconditionally, it is NP-hard to approximate within a factor of $1.3606$ (Dinur and Safra, 2005).

??? example "Worked Example"
    Consider the graph with $V = \{1, 2, 3, 4, 5\}$ and edges $E = \{(1,2), (2,3), (3,4), (4,5), (1,3)\}$.

    **Matching-based algorithm:**

    1. Pick edge $(1,2)$: add $\{1, 2\}$ to $C$. Remove edges $(1,2), (2,3), (1,3)$.
    2. Remaining: $\{(3,4), (4,5)\}$. Pick $(3,4)$: add $\{3, 4\}$ to $C$. Remove $(3,4), (4,5)$.
    3. $E' = \emptyset$. Return $C = \{1, 2, 3, 4\}$, $|C| = 4$.

    **Optimal:** $C^* = \{2, 3, 4\}$ covers all edges with $|C^*| = 3$.

    **Ratio:** $4/3 \approx 1.33 \leq 2$. The guarantee holds.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.
- Vazirani, V. V. (2001). *Approximation Algorithms*. Springer.
- Dinur, I., & Safra, S. (2005). On the hardness of approximating minimum vertex cover. *Annals of Mathematics*, 162(1), 439--485.
