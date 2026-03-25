# Exchange Argument

How can we prove that a greedy algorithm --- one that commits irrevocably to each choice --- produces an optimal solution? The **exchange argument** is the most widely used technique. The idea is to take an arbitrary optimal solution and gradually transform it into the greedy solution, one swap at a time, without ever making it worse. If every optimal solution can be reshaped to match the greedy output, then the greedy output must itself be optimal.

## Core Idea

The exchange argument works by contradiction combined with construction. We do not prove the greedy solution is unique; instead we prove that an optimal solution agreeing with the greedy choices exists:

1. Start with any optimal solution $S^*$.
2. Find the first point where $S^*$ and the greedy solution $G$ differ.
3. Swap one element of $S^*$ to match $G$ at that point.
4. Show the swap preserves feasibility and does not worsen the objective.
5. Repeat until $S^* = G$.

If each swap produces a solution that is still optimal (or at least no worse), then $G$ is optimal.

## Formal Template

!!! note "Exchange Argument Template"
    **Goal**: prove that the greedy algorithm $G$ produces an optimal solution.

    **Step 1 (Setup).** Let $S^* = (s_1^*, s_2^*, \ldots, s_m^*)$ be any optimal solution. Let $G = (g_1, g_2, \ldots, g_k)$ be the greedy solution. Both are ordered by the algorithm's selection order.

    **Step 2 (Find first difference).** Let $i$ be the smallest index where $s_i^* \neq g_i$.

    **Step 3 (Exchange).** Construct $S' = (S^* \setminus \{s_i^*\}) \cup \{g_i\}$ (swap $s_i^*$ for $g_i$).

    **Step 4 (Feasibility).** Show $S'$ satisfies all constraints of the problem.

    **Step 5 (Quality).** Show $\text{cost}(S') \leq \text{cost}(S^*)$ (minimization) or $\text{value}(S') \geq \text{value}(S^*)$ (maximization).

    **Step 6 (Conclude).** $S'$ is optimal and agrees with $G$ on one more choice. Repeat until full agreement.

## Example 1: Activity Selection

**Problem.** Select the maximum number of mutually compatible (non-overlapping) activities from a set $\{a_1, \ldots, a_n\}$, where each activity $a_i$ has start time $s_i$ and finish time $f_i$.

**Greedy rule.** Always pick the activity with the earliest finish time.

**Theorem.** The greedy algorithm produces an optimal solution.

??? example "Proof by Exchange Argument"
    Let $S^* = \{a_{j_1}, a_{j_2}, \ldots, a_{j_k}\}$ be an optimal set of compatible activities, sorted by finish time. Let $a_1$ be the activity with the globally earliest finish time (the greedy first choice).

    **Case 1:** $a_{j_1} = a_1$. The optimal solution already includes the greedy choice.

    **Case 2:** $a_{j_1} \neq a_1$. Since $a_1$ has the earliest finish time among all activities, $f_1 \leq f_{j_1}$. Construct:

    $$
    S' = (S^* \setminus \{a_{j_1}\}) \cup \{a_1\}
    $$

    **Feasibility:** Activity $a_1$ finishes at time $f_1 \leq f_{j_1}$, so $a_1$ is compatible with $a_{j_2}$ (which starts at $s_{j_2} \geq f_{j_1} \geq f_1$). All other pairwise compatibilities in $S^*$ are unchanged.

    **Quality:** $|S'| = |S^*| = k$, so $S'$ is also a maximum-size compatible set.

    Therefore, $S'$ is optimal and includes $a_1$. By optimal substructure, the subproblem $\{a_i : s_i \geq f_1\}$ has an optimal solution that combines with $a_1$ to give an optimal overall solution. The greedy algorithm solves this subproblem recursively, so by induction it is correct. $\square$

## Example 2: Fractional Knapsack

**Problem.** Given items with weights $w_i$ and values $v_i$, maximize total value in a knapsack of capacity $W$, where fractions of items may be taken.

**Greedy rule.** Sort items by value-to-weight ratio $r_i = v_i / w_i$ in decreasing order. Fill the knapsack greedily from the highest-ratio item.

**Theorem.** The greedy algorithm produces an optimal solution.

??? example "Proof by Exchange Argument"
    Without loss of generality, assume $r_1 \geq r_2 \geq \cdots \geq r_n$. Let $S^* = (x_1^*, x_2^*, \ldots, x_n^*)$ be any optimal solution, where $x_i^* \in [0, 1]$ is the fraction of item $i$ taken.

    Let the greedy solution be $G = (x_1^G, x_2^G, \ldots, x_n^G)$. In $G$, items are taken greedily: $x_1^G = \min(1, W/w_1)$, then fill remaining capacity with item 2, and so on.

    Suppose $S^* \neq G$. Let $j$ be the first index where $x_j^* \neq x_j^G$. Since the greedy algorithm takes as much of item $j$ as possible, $x_j^G > x_j^*$. Let $\delta = x_j^G - x_j^*$. The greedy solution takes $\delta$ more of item $j$.

    In $S^*$, the capacity freed by reducing item $j$ must be allocated to items $k > j$ (which have lower ratios). Construct $S'$ by increasing $x_j$ by some amount and decreasing later items by the same weight. Since $r_j \geq r_k$ for all $k > j$:

    $$
    \text{value}(S') - \text{value}(S^*) = \delta \cdot w_j \cdot r_j - \sum_{k>j} \Delta_k \cdot w_k \cdot r_k \geq 0
    $$

    because we replace lower-ratio capacity usage with higher-ratio usage. So $S'$ is at least as good and agrees with $G$ on one more item. Repeat. $\square$

## Example 3: Huffman Coding

**Problem.** Construct a prefix-free binary code minimizing the weighted path length $\sum_i f_i \cdot d_i$, where $f_i$ is the frequency of character $i$ and $d_i$ is its code length.

**Greedy rule.** Repeatedly merge the two characters with the lowest frequencies.

The exchange argument for Huffman coding shows that in any optimal tree, the two lowest-frequency characters can be made siblings at the maximum depth without increasing cost. This is a more involved exchange, as it swaps positions in a tree rather than elements in a set.

## Structural Patterns

Several common patterns appear across exchange argument proofs:

### Single-Swap Exchange
Replace one element and show the solution improves or stays the same. Used in activity selection and fractional knapsack.

### Adjacent-Swap Exchange
Swap two adjacent elements in an ordering and show it improves. Common in scheduling problems.

??? example "Adjacent Swap: Minimizing Weighted Completion Time"
    Given jobs with processing times $p_i$ and weights $w_i$, schedule to minimize $\sum w_i C_i$ where $C_i$ is the completion time. The greedy rule is to sort by $w_i / p_i$ in decreasing order.

    Consider two adjacent jobs $j$ and $k$ in some schedule. If $w_j / p_j < w_k / p_k$, swapping them so $k$ comes first reduces the total weighted completion time:

    $$
    w_j(p_j + p_k) + w_k \cdot p_k > w_k(p_k + p_j) + w_j \cdot p_j
    $$

    simplifies to $w_j \cdot p_k > w_k \cdot p_j$, i.e., $w_j / p_j < w_k / p_k$, which is our assumption. So any inversion increases cost, and the greedy order is optimal. $\square$

### Tree Exchange
Swap positions in a tree structure. Used in Huffman coding proofs.

## When Exchange Arguments Fail

The exchange argument requires that each swap preserves feasibility. In problems where swapping one element has cascading effects on other constraints, the argument becomes difficult or impossible:

- **0-1 Knapsack**: swapping items can violate the capacity constraint, and there is no way to patch the solution with a single exchange.
- **Graph coloring**: changing one vertex's color can force changes throughout the graph.
- **Traveling salesman**: removing one city from a tour and inserting another creates a different tour structure entirely.

In these cases, the greedy choice property does not hold, and the exchange argument correctly fails to go through.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 16. MIT Press.
- Kleinberg, J. & Tardos, E. (2006). *Algorithm Design*, Chapter 4.1. Pearson.
