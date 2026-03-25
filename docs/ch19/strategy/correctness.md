# Proving Correctness

Greedy algorithms are deceptively simple to design: pick the best-looking option at each step. The real challenge lies in proving that this myopic strategy actually produces a globally optimal solution. Unlike dynamic programming, where correctness follows naturally from the Bellman equation and induction on subproblem size, greedy algorithms require a dedicated argument showing that the irrevocable choices made at each step never lead to a dead end. This page presents the general framework and the two principal proof techniques.

## Why Greedy Correctness Is Hard

The difficulty stems from a fundamental asymmetry: greedy algorithms commit to choices without examining alternatives. A proof must show that no alternative choice could have led to a better outcome --- even though the algorithm never considers those alternatives.

Consider three possible greedy rules for activity selection:

1. Pick the activity with the **earliest start time**.
2. Pick the **shortest** activity.
3. Pick the activity with the **earliest finish time**.

Rules 1 and 2 are intuitively reasonable but provably wrong. Only rule 3 yields an optimal solution. Without a formal proof, we cannot distinguish correct greedy strategies from incorrect ones.

## The Two-Part Framework

Every greedy correctness proof establishes two properties:

!!! note "Requirements for Greedy Correctness"
    1. **Greedy choice property**: there exists an optimal solution that includes the greedy algorithm's first choice.
    2. **Optimal substructure**: after making the greedy choice, the remaining subproblem has an optimal solution that combines with the greedy choice to form an optimal solution to the original problem.

Given both properties, correctness follows by induction on the number of choices:

- **Base case**: the empty solution is trivially optimal for an empty problem.
- **Inductive step**: assume the greedy algorithm produces an optimal solution for any subproblem of size $< k$. For a problem of size $k$, the greedy choice property guarantees the first choice is safe, and optimal substructure ensures the remaining subproblem (size $< k$) is solved optimally by the inductive hypothesis.

$$
\text{OPT}(\mathcal{P}) = \{g\} \cup \text{OPT}(\mathcal{P}')
$$

where $g$ is the greedy choice and $\mathcal{P}'$ is the residual subproblem.

## Proof Technique 1: Exchange Argument

The **exchange argument** proves the greedy choice property by showing that any optimal solution can be transformed into one that agrees with the greedy choice, without worsening the objective.

**General template:**

1. Let $S^*$ be an arbitrary optimal solution.
2. If $S^*$ already includes the greedy choice $g$, we are done.
3. Otherwise, identify an element $x \in S^*$ that can be replaced by $g$.
4. Show that the modified solution $S' = (S^* \setminus \{x\}) \cup \{g\}$ is feasible.
5. Show that $\text{cost}(S') \leq \text{cost}(S^*)$ (for minimization) or $\text{value}(S') \geq \text{value}(S^*)$ (for maximization).
6. Conclude that $S'$ is optimal and contains $g$.

The exchange argument is the most widely used technique. It works well when there is a natural "swap" between the greedy choice and any non-greedy choice that preserves feasibility.

??? example "Exchange Argument: Activity Selection"
    **Greedy rule**: always pick the activity with the earliest finish time.

    Let $S^* = \{a_{j_1}, \ldots, a_{j_k}\}$ be optimal, sorted by finish time. Let $a_1$ be the activity with the globally earliest finish time. If $a_{j_1} = a_1$, done. Otherwise, $f_1 \leq f_{j_1}$, so replacing $a_{j_1}$ with $a_1$ preserves compatibility with $a_{j_2}, \ldots, a_{j_k}$. The resulting set has the same cardinality $k$, so it is optimal and contains $a_1$. $\square$

## Proof Technique 2: Greedy Stays Ahead

The **greedy stays ahead** technique shows that at every step of the algorithm, the greedy solution is at least as good as the corresponding prefix of any other solution.

**General template:**

1. Let $G = (g_1, g_2, \ldots, g_k)$ be the greedy solution and $O = (o_1, o_2, \ldots, o_m)$ be any feasible solution, both ordered by the algorithm's processing order.
2. Define a measure of progress (e.g., finish time, partial cost).
3. Prove by induction on $i$ that the greedy solution's $i$-th choice is at least as good as $O$'s $i$-th choice under this measure.
4. Conclude that $k \geq m$ (for maximization of count) or that the greedy cost is no worse.

??? example "Greedy Stays Ahead: Activity Selection"
    Let $G = (g_1, \ldots, g_k)$ be the greedy solution (sorted by finish time) and $O = (o_1, \ldots, o_m)$ be any optimal solution (sorted by finish time).

    **Claim**: for all $i \leq \min(k, m)$, the finish time of $g_i$ is at most the finish time of $o_i$: $f(g_i) \leq f(o_i)$.

    **Proof by induction on** $i$:

    - *Base case* ($i = 1$): the greedy algorithm picks the activity with the earliest finish time, so $f(g_1) \leq f(o_1)$.
    - *Inductive step*: assume $f(g_{i-1}) \leq f(o_{i-1})$. Since $o_i$ starts after $o_{i-1}$ finishes, we have $s(o_i) \geq f(o_{i-1}) \geq f(g_{i-1})$. So $o_i$ is available when the greedy algorithm makes its $i$-th choice. The greedy algorithm picks the available activity with the earliest finish time, so $f(g_i) \leq f(o_i)$.

    Since $f(g_i) \leq f(o_i)$ for all $i$, the greedy solution is at least as long as any other: $k \geq m$. $\square$

## Choosing the Right Technique

| Criterion | Exchange Argument | Greedy Stays Ahead |
|-----------|-------------------|--------------------|
| Best for | Problems where one swap suffices | Problems with a natural ordering |
| Proof structure | Modify an arbitrary optimal solution | Induction on step index |
| Common in | Huffman coding, fractional knapsack | Activity selection, scheduling |
| Difficulty | Finding the right exchange | Defining the "ahead" measure |

Both techniques are equivalent in power: any greedy correctness proof using one can be rewritten using the other. The choice is usually a matter of which yields a cleaner argument for the specific problem.

## Common Pitfalls

!!! warning "Mistakes to Avoid"
    1. **Assuming correctness from examples.** Running a greedy algorithm on a few test cases and observing correct output does not constitute a proof.
    2. **Confusing greedy with heuristic.** A greedy heuristic may produce a good-but-not-optimal solution. Only algorithms with a correctness proof are truly "greedy algorithms" in the formal sense.
    3. **Forgetting optimal substructure.** The greedy choice property alone does not suffice --- you must also verify that the residual subproblem has the right structure.
    4. **Wrong greedy criterion.** Activity selection sorted by start time, duration, or number of conflicts all fail. Only earliest finish time works, and the proof reveals why.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 16. MIT Press.
- Kleinberg, J. & Tardos, E. (2006). *Algorithm Design*, Chapter 4. Pearson.
