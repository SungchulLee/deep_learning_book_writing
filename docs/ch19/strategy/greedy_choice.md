# Greedy Choice Property

When designing an algorithm that builds a solution incrementally, a natural question arises: can we commit to each decision as we make it, without ever reconsidering? The **greedy choice property** provides the theoretical guarantee that this strategy works. If a problem satisfies this property, then at every step there exists a locally optimal choice that can be extended to a globally optimal solution --- no backtracking and no exhaustive search required.

## Informal Intuition

Consider selecting activities to maximize the number of non-overlapping events on a single stage. A greedy approach picks the activity that finishes earliest, then repeats for the remaining compatible activities. The greedy choice property asserts that the earliest-finishing activity is always safe to include: some optimal solution must contain it, so we lose nothing by committing to it immediately.

The key insight is that we do not need to explore all possible subsets. Instead, we make one irrevocable choice, reduce the problem, and recurse.

## Formal Definition

Let $\mathcal{P}$ be an optimization problem in which a solution is built by making a sequence of choices $c_1, c_2, \ldots, c_k$. The problem satisfies the **greedy choice property** if the following holds:

!!! note "Greedy Choice Property"
    For every instance of $\mathcal{P}$, there exists an optimal solution that includes the greedy (locally optimal) first choice. That is, making the locally optimal choice at the current step does not preclude reaching a globally optimal solution.

More precisely, let $S^*$ be any optimal solution. If the greedy choice is $g$, then there exists an optimal solution $S'$ such that $g \in S'$.

## Relationship to Optimal Substructure

The greedy choice property alone is not sufficient to guarantee greedy correctness. It must be paired with **optimal substructure**: after making the greedy choice, the remaining subproblem must itself have an optimal solution that, combined with the greedy choice, yields an optimal solution to the original problem.

Together, these two properties justify the greedy paradigm:

1. **Greedy choice property** --- a locally optimal choice is globally safe.
2. **Optimal substructure** --- the residual problem after the greedy choice is itself an optimization problem whose optimal solution combines with the greedy choice to form an overall optimum.

$$
\text{OPT}(\mathcal{P}) = g \cup \text{OPT}(\mathcal{P}')
$$

where $g$ is the greedy choice and $\mathcal{P}'$ is the subproblem remaining after committing to $g$.

## Proof Template

To establish the greedy choice property for a specific problem, the standard approach is a **"cut-and-paste"** or **exchange argument**:

1. **Assume** an optimal solution $S^*$ exists.
2. **If** $S^*$ already contains the greedy choice $g$, we are done.
3. **If not**, construct a new solution $S'$ by replacing some element of $S^*$ with $g$.
4. **Show** that $S'$ is feasible (satisfies all constraints).
5. **Show** that $S'$ is at least as good as $S^*$ (the objective value does not worsen).
6. **Conclude** that $S'$ is an optimal solution containing $g$.

??? example "Activity Selection: Greedy Choice Proof Sketch"
    **Claim.** There exists an optimal solution that includes the activity $a_1$ with the earliest finish time.

    **Proof sketch.** Let $S^* = \{a_{j_1}, a_{j_2}, \ldots, a_{j_k}\}$ be an optimal set of non-overlapping activities sorted by finish time. If $a_{j_1} = a_1$, we are done. Otherwise, $a_1$ finishes no later than $a_{j_1}$, so replacing $a_{j_1}$ with $a_1$ preserves non-overlap with $a_{j_2}, \ldots, a_{j_k}$. The resulting set $S' = \{a_1, a_{j_2}, \ldots, a_{j_k}\}$ has the same size $k$, so it is also optimal. $\square$

## When the Greedy Choice Property Fails

Not every optimization problem has the greedy choice property. The classic counterexample is the **0-1 knapsack problem**: choosing the item with the highest value-to-weight ratio first can lead to a suboptimal solution because items cannot be split.

??? warning "Counterexample: 0-1 Knapsack"
    Suppose the knapsack capacity is $W = 50$ and there are three items:

    | Item | Weight | Value | Ratio |
    |------|--------|-------|-------|
    | A    | 10     | 60    | 6.0   |
    | B    | 20     | 100   | 5.0   |
    | C    | 30     | 120   | 4.0   |

    The greedy strategy (by ratio) picks A first, then B, for total value 160 with weight 30. But the optimal solution is B + C with total value 220 and weight 50. The greedy first choice (item A) is not part of the optimal solution.

## Comparison with Dynamic Programming

Both greedy algorithms and dynamic programming exploit optimal substructure. The critical difference is whether the greedy choice property holds:

| Property | Greedy | Dynamic Programming |
|----------|--------|---------------------|
| Optimal substructure | Required | Required |
| Greedy choice property | Required | Not needed |
| Overlapping subproblems | Not required | Exploited |
| Choices reconsidered | Never | Yes (all subproblems solved) |
| Time complexity | Often $O(n \log n)$ | Often $O(n^2)$ or $O(nW)$ |

When the greedy choice property holds, the greedy algorithm is preferred because it avoids the overhead of solving overlapping subproblems.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 16. MIT Press.
- Kleinberg, J. & Tardos, E. (2006). *Algorithm Design*, Chapter 4. Pearson.
