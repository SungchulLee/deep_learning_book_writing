# Greedy Stays Ahead

The exchange argument transforms an arbitrary optimal solution into the greedy solution one swap at a time. The **greedy stays ahead** technique takes a different perspective: instead of modifying an optimal solution, it directly compares the greedy solution against any other feasible solution step by step, showing that after each choice the greedy algorithm is at least as far along as any competitor. This inductive invariant leads to a clean proof that the greedy solution is globally optimal.

## Intuition

Imagine two runners on parallel tracks, both starting at the same time. If runner G (greedy) is always at least as far ahead as runner O (any other strategy) at every checkpoint, then G finishes at least as early --- or, in the case of maximization, accumulates at least as much value. The "stays ahead" invariant captures exactly this: the greedy algorithm never falls behind.

The technique works best when there is a natural ordering of choices and a progress measure that can be compared step by step.

## Formal Template

!!! note "Greedy Stays Ahead Template"
    **Setup.** Let $G = (g_1, g_2, \ldots, g_k)$ be the greedy solution and $O = (o_1, o_2, \ldots, o_m)$ be any feasible solution, both sorted in the order the algorithm processes them.

    **Define a measure.** Choose a function $\mu$ that captures "progress" after $i$ steps. For activity selection, $\mu(i) = f(g_i)$ (finish time of the $i$-th selected activity).

    **Stays-ahead invariant.** For all $i \leq \min(k, m)$:

    $$
    \mu_G(i) \leq \mu_O(i) \quad \text{(for earliest-finish-time problems)}
    $$

    or the appropriate inequality for the problem's objective.

    **Prove by induction on** $i$:

    - *Base case* ($i = 1$): follows from the greedy rule.
    - *Inductive step*: assume $\mu_G(i-1) \leq \mu_O(i-1)$; show $\mu_G(i) \leq \mu_O(i)$.

    **Conclude:** Since $G$ stays ahead at every step, $G$ is at least as good as $O$ overall: $k \geq m$ (for maximization of count) or $\text{cost}(G) \leq \text{cost}(O)$ (for minimization).

## Example 1: Activity Selection

**Problem.** Select the maximum number of mutually compatible activities from $\{a_1, \ldots, a_n\}$ with start times $s_i$ and finish times $f_i$.

**Greedy rule.** Always pick the unselected compatible activity with the earliest finish time.

**Theorem.** The greedy algorithm produces a maximum-size compatible set.

**Proof.**

Let $G = (g_1, g_2, \ldots, g_k)$ be the greedy solution, sorted by finish time. Let $O = (o_1, o_2, \ldots, o_m)$ be any maximum-size compatible set, also sorted by finish time.

**Stays-ahead invariant:** for all $1 \leq i \leq \min(k, m)$, we have $f(g_i) \leq f(o_i)$.

*Base case* ($i = 1$): The greedy algorithm picks the activity with the globally earliest finish time, so $f(g_1) \leq f(o_1)$.

*Inductive step:* Assume $f(g_{i-1}) \leq f(o_{i-1})$ for some $i \geq 2$. Activity $o_i$ is compatible with $o_{i-1}$, so:

$$
s(o_i) \geq f(o_{i-1}) \geq f(g_{i-1})
$$

This means $o_i$ is available (compatible with $g_{i-1}$) when the greedy algorithm makes its $i$-th choice. The greedy algorithm picks the available activity with the smallest finish time, so:

$$
f(g_i) \leq f(o_i)
$$

**Conclusion.** Suppose for contradiction that $k < m$. Then $o_{k+1}$ exists and satisfies $s(o_{k+1}) \geq f(o_k) \geq f(g_k)$, so $o_{k+1}$ is compatible with $g_k$. But then the greedy algorithm would have selected at least one more activity after $g_k$, contradicting $|G| = k$. Therefore $k \geq m$, and since $O$ is a maximum-size set, $k = m$. $\square$

## Example 2: Minimizing Maximum Lateness

**Problem.** Given $n$ jobs with processing times $p_i$ and deadlines $d_i$, schedule all jobs on a single machine (no idle time) to minimize the maximum lateness $L_{\max} = \max_i (C_i - d_i)$, where $C_i$ is the completion time of job $i$.

**Greedy rule.** Schedule jobs in order of increasing deadline: $d_1 \leq d_2 \leq \cdots \leq d_n$ (Earliest Deadline First, EDF).

**Theorem.** EDF minimizes $L_{\max}$.

??? example "Proof by Greedy Stays Ahead"
    Let $G$ be the EDF schedule and $O$ be any other schedule. We show $L_{\max}(G) \leq L_{\max}(O)$.

    **Observation.** In $G$, since there is no idle time, the completion time of the $j$-th job is:

    $$
    C_j^G = \sum_{i=1}^{j} p_{\sigma_G(i)}
    $$

    where $\sigma_G$ is the EDF permutation. The same formula holds for $O$ with permutation $\sigma_O$.

    **Key insight.** Both schedules process the same set of jobs with no idle time, so the completion time of the $j$-th job in any schedule equals the sum of the first $j$ processing times under that schedule's permutation.

    Since EDF processes jobs in deadline order, the job completing at position $j$ has the smallest possible deadline among jobs in positions $1, \ldots, n$. This means the lateness $C_j^G - d_{\sigma_G(j)}$ is minimized for the "worst-positioned" job.

    More precisely, any schedule with an inversion (job $a$ before job $b$ with $d_a > d_b$) can be improved by swapping $a$ and $b$, reducing the maximum lateness. Since EDF has no inversions, it is optimal. $\square$

## Why Stays Ahead Works

The power of the technique lies in the **inductive strengthening**: we prove not just that the final result is optimal, but that the greedy solution dominates at every intermediate step. This stronger claim makes the induction go through cleanly.

The measure $\mu$ must be chosen carefully. Good measures satisfy:

1. **Monotonicity**: $\mu_G(i) \leq \mu_G(i+1)$ (progress always advances).
2. **Comparability**: $\mu_G(i)$ and $\mu_O(i)$ measure the same quantity for the same step index.
3. **Terminal implication**: the stays-ahead invariant at $i = \min(k, m)$ implies $G$ is at least as good as $O$.

## Comparison with Exchange Argument

| Aspect | Greedy Stays Ahead | Exchange Argument |
|--------|--------------------|--------------------|
| Proof structure | Induction on step $i$ | Transform optimal $\to$ greedy |
| What is compared | Greedy vs any solution, step by step | Two solutions at one swap point |
| Strongest when | Natural progress measure exists | Single swap preserves feasibility |
| Examples | Activity selection, scheduling | Huffman, fractional knapsack |
| Typical invariant | $f(g_i) \leq f(o_i)$ | $\|S'\| \geq \|S^*\|$ after swap |

Both techniques are logically equivalent for proving greedy correctness, but stays-ahead often produces more elegant proofs for problems where the greedy algorithm processes items in a sorted order.

## Common Pitfalls

!!! warning "Mistakes in Stays-Ahead Proofs"
    1. **Wrong measure.** Using total value instead of per-step finish time (or vice versa). The measure must be comparable at each step.
    2. **Off-by-one in the contradiction.** Forgetting to handle the case where $k < m$ separately at the end of the induction.
    3. **Assuming $k = m$.** The proof must establish this, not assume it. In activity selection, the stays-ahead invariant implies $k \geq m$ and feasibility gives $k \leq m$.

## Reference

- Kleinberg, J. & Tardos, E. (2006). *Algorithm Design*, Chapter 4.1. Pearson.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 16. MIT Press.
