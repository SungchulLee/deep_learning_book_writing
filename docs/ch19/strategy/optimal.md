# Optimal Substructure

Greedy algorithms work by making a single locally optimal choice and then solving a smaller version of the same problem. For this recursive decomposition to produce a globally optimal solution, the problem must exhibit **optimal substructure**: the optimal solution to the original problem must contain within it optimal solutions to its subproblems. Without this property, the greedy choice --- no matter how clever --- could leave behind a subproblem whose best solution, combined with that choice, falls short of the global optimum.

## Intuition

Optimal substructure is a "compositional" property. It says that excellence at the global level is built from excellence at the local level. If you solve each subproblem optimally and combine the results correctly, you get an optimal solution overall.

Consider scheduling activities on a lecture hall. After selecting the first activity $a_1$ (the one that finishes earliest), the remaining subproblem is to select the maximum number of non-overlapping activities that start after $a_1$ finishes. If the overall optimal solution contains $a_1$, then the remaining activities in that solution must themselves be an optimal solution to the residual subproblem --- otherwise, we could swap in a better set and improve the whole solution, contradicting optimality.

## Formal Definition

!!! note "Optimal Substructure for Greedy Problems"
    A problem $\mathcal{P}$ exhibits **optimal substructure** if an optimal solution to $\mathcal{P}$ can be constructed from the greedy choice $g$ combined with an optimal solution to the subproblem $\mathcal{P}'$ that remains after making choice $g$:

    $$
    \text{OPT}(\mathcal{P}) = \{g\} \cup \text{OPT}(\mathcal{P}')
    $$

This recursive decomposition is what enables the greedy algorithm to make one choice at a time, confident that solving the residual problem optimally will produce a globally optimal result.

## Proving Optimal Substructure

The standard proof technique is a **cut-and-paste argument**:

1. **Assume** $S^*$ is an optimal solution to $\mathcal{P}$ that includes the greedy choice $g$.
2. **Define** $S^* \setminus \{g\}$ as a candidate solution to the subproblem $\mathcal{P}'$.
3. **Suppose for contradiction** that $S^* \setminus \{g\}$ is not optimal for $\mathcal{P}'$. Then there exists a strictly better solution $T'$ for $\mathcal{P}'$.
4. **Paste**: form $S' = \{g\} \cup T'$. Since $T'$ is better than $S^* \setminus \{g\}$ for $\mathcal{P}'$, the combined solution $S'$ is better than $S^*$ for $\mathcal{P}$.
5. **Contradiction**: this contradicts the optimality of $S^*$.

Therefore, $S^* \setminus \{g\}$ must be optimal for $\mathcal{P}'$.

## Example: Activity Selection

**Problem.** Given $n$ activities with start times $s_i$ and finish times $f_i$, select the maximum number of mutually compatible (non-overlapping) activities.

**Greedy choice.** Select the activity $a_1$ with the smallest finish time $f_1$.

**Subproblem.** Let $\mathcal{P}' = \{a_i : s_i \geq f_1\}$ be the set of activities that start after $a_1$ finishes.

**Optimal substructure claim.** If $S^* = \{a_1\} \cup R$ is an optimal solution containing $a_1$, then $R$ is an optimal solution to $\mathcal{P}'$.

??? example "Cut-and-Paste Proof"
    **Proof.** Suppose $R$ is not optimal for $\mathcal{P}'$. Then there exists a compatible set $R'$ for $\mathcal{P}'$ with $|R'| > |R|$. Since every activity in $R'$ starts after $f_1$, the set $\{a_1\} \cup R'$ is a compatible set for $\mathcal{P}$ with $|\{a_1\} \cup R'| = 1 + |R'| > 1 + |R| = |S^*|$. This contradicts the optimality of $S^*$, so $R$ must be optimal for $\mathcal{P}'$. $\square$

## Example: Fractional Knapsack

**Problem.** Given $n$ items with weights $w_i$ and values $v_i$, and a knapsack of capacity $W$, maximize total value by taking fractions of items.

**Greedy choice.** Take as much as possible of the item with the highest value-to-weight ratio $v_i / w_i$.

**Optimal substructure.** After filling the knapsack with the greedy choice (fully or partially taking the best-ratio item), the remaining capacity $W' = W - \min(w_1, W)$ defines a subproblem. An optimal solution to the original problem restricted to the greedy choice decomposes as:

$$
\text{OPT}(W) = v_{\text{greedy}} + \text{OPT}(W')
$$

where $v_{\text{greedy}}$ is the value gained from the greedy choice.

## Contrast with Dynamic Programming

Both greedy algorithms and dynamic programming rely on optimal substructure, but they differ in how subproblems are explored:

| Aspect | Greedy | Dynamic Programming |
|--------|--------|---------------------|
| Number of subproblems considered | One (after greedy choice) | All possible choices |
| Subproblem dependency | Single chain | Overlapping DAG |
| Proof of correctness | Greedy choice + optimal substructure | Bellman equation + induction |

In dynamic programming, the algorithm must consider all possible first choices and pick the best. In greedy algorithms, the greedy choice property guarantees that considering only one choice suffices.

## Common Pitfall

A frequent mistake is assuming that optimal substructure alone justifies a greedy approach. It does not. The **greedy choice property** is also required --- it ensures that a locally optimal choice is part of some globally optimal solution. The 0-1 knapsack problem has optimal substructure (it can be solved by dynamic programming), but it lacks the greedy choice property, so greedy fails.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 16. MIT Press.
- Kleinberg, J. & Tardos, E. (2006). *Algorithm Design*, Chapter 4. Pearson.
