# Greedy vs DP

Two of the most powerful algorithm design paradigms are **greedy algorithms** and **dynamic programming (DP)**. Choosing between them is one of the most common classification decisions. Both solve optimization problems, but they differ fundamentally in how they make choices.

## Key Differences

| Property | Greedy | Dynamic Programming |
|---|---|---|
| Choice | Locally optimal at each step | Considers all subproblems |
| Guarantee | Not always globally optimal | Always globally optimal (if applicable) |
| Subproblems | Does not revisit decisions | Solves overlapping subproblems |
| Requirements | Greedy-choice property + optimal substructure | Optimal substructure + overlapping subproblems |
| Time complexity | Often $O(n \log n)$ | Often $O(n^2)$ or $O(nW)$ |
| Space complexity | Often $O(1)$ or $O(n)$ | Often $O(n)$ to $O(n^2)$ |

## When Greedy Works

A greedy algorithm works when:

1. **Greedy-choice property**: A locally optimal choice leads to a globally optimal solution.
2. **Optimal substructure**: An optimal solution contains optimal solutions to subproblems.

**Classic example -- Activity Selection:**

$$\text{Maximize the number of non-overlapping activities}$$

Sort by finish time, always pick the earliest-finishing compatible activity.

```python
def activity_selection(activities):
    # Sort by finish time
    activities.sort(key=lambda x: x[1])
    selected = [activities[0]]
    for i in range(1, len(activities)):
        if activities[i][0] >= selected[-1][1]:
            selected.append(activities[i])
    return selected

activities = [(1, 3), (2, 5), (3, 4), (5, 7), (6, 8), (8, 10)]
print(activity_selection(activities))
# Output: [(1, 3), (3, 4), (5, 7), (8, 10)]
```

## When DP Is Needed

DP is needed when greedy fails -- typically because a locally optimal choice can lead to a globally suboptimal solution.

**Classic example -- 0/1 Knapsack:**

$$\text{Maximize } \sum v_i x_i \text{ subject to } \sum w_i x_i \le W, \quad x_i \in \{0,1\}$$

Greedy by value-to-weight ratio fails for discrete items.

```python
def knapsack_01(weights, values, W):
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(W + 1):
            dp[i][w] = dp[i-1][w]
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w],
                               dp[i-1][w - weights[i-1]] + values[i-1])
    return dp[n][W]

print(knapsack_01([2, 3, 4, 5], [3, 4, 5, 6], 8))  # Output: 10
```

## Decision Flowchart

1. Does the problem have **optimal substructure**? If no, neither approach works directly.
2. Does making the locally best choice always lead to a global optimum? If yes, use **Greedy**.
3. Are there **overlapping subproblems**? If yes, use **DP**.
4. If subproblems are independent, use **Divide and Conquer**.

# Reference

- Cormen, T. et al. *Introduction to Algorithms*, Chapter 15-16, MIT Press, 2022.
- Kleinberg, J. & Tardos, E. *Algorithm Design*, Pearson, 2005.
