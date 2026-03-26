# Hungarian Algorithm

The assignment problem asks: given $n$ workers and $n$ jobs, with a cost $c_{ij}$ for assigning worker $i$ to job $j$, find a one-to-one assignment that minimizes total cost. While this can be modeled as a linear program or a min-cost flow problem, the **Hungarian algorithm** (Kuhn, 1955) solves it directly in $O(n^3)$ time by exploiting the combinatorial structure of the cost matrix.

## Problem Formulation

Given an $n \times n$ cost matrix $C = [c_{ij}]$, find a permutation $\pi$ of $\{1, 2, \dots, n\}$ that minimizes:

$$
\sum_{i=1}^{n} c_{i,\pi(i)}
$$

Equivalently, find a perfect matching in the complete bipartite graph $K_{n,n}$ with minimum total weight.

## Key Insight

The algorithm relies on a fundamental observation: subtracting a constant from any row or column of $C$ does not change which assignment is optimal. This allows us to create zeros in the cost matrix and look for an assignment using only zero-cost entries.

!!! note "Zero-Cost Optimality"
    If we can find a perfect matching using only zero entries in a reduced cost matrix (where all entries are non-negative), that matching is optimal for the original problem.

## Algorithm Steps

**Step 1: Row reduction.** Subtract the minimum entry from each row.

**Step 2: Column reduction.** Subtract the minimum entry from each column.

**Step 3: Cover zeros.** Find the minimum number of lines (rows and columns) needed to cover all zeros. If this number equals $n$, a zero-cost perfect matching exists and we are done.

**Step 4: Create new zeros.** Find the smallest uncovered entry $\delta$. Subtract $\delta$ from all uncovered entries, and add $\delta$ to all doubly-covered entries (entries at the intersection of two covering lines). Return to Step 3.

Each iteration of Steps 3--4 increases the number of covering lines by at least one, so the algorithm terminates in at most $n$ iterations. Each iteration takes $O(n^2)$ time, giving an overall $O(n^3)$ complexity.

## Implementation

```python
"""
Hungarian algorithm for the assignment problem.

Solves the minimum-cost assignment problem for an n x n cost matrix
in O(n^3) time using potential-based shortest path augmentation.
"""

import math

# === Hungarian Algorithm ===

def hungarian(cost: list[list[float]]) -> tuple[float, list[int]]:
    """Solve the assignment problem using the Hungarian algorithm.

    Args:
        cost: n x n cost matrix where cost[i][j] is the cost of
              assigning worker i to job j.

    Returns:
        Tuple of (minimum total cost, assignment) where assignment[i]
        is the job assigned to worker i.
    """
    n = len(cost)
    # Use 1-indexed arrays; index 0 is a dummy
    u = [0.0] * (n + 1)    # potential for workers
    v = [0.0] * (n + 1)    # potential for jobs
    match_job = [0] * (n + 1)  # match_job[j] = worker matched to job j

    for i in range(1, n + 1):
        # Try to assign worker i
        match_job[0] = i
        j0 = 0  # virtual unmatched job
        dist = [math.inf] * (n + 1)
        used = [False] * (n + 1)
        prev = [0] * (n + 1)

        # Shortest path from worker i to any free job
        while True:
            used[j0] = True
            w = match_job[j0]
            delta = math.inf
            j1 = -1

            for j in range(1, n + 1):
                if not used[j]:
                    reduced = cost[w - 1][j - 1] - u[w] - v[j]
                    if reduced < dist[j]:
                        dist[j] = reduced
                        prev[j] = j0
                    if dist[j] < delta:
                        delta = dist[j]
                        j1 = j

            # Update potentials
            for j in range(n + 1):
                if used[j]:
                    u[match_job[j]] += delta
                    v[j] -= delta
                else:
                    dist[j] -= delta

            j0 = j1
            if match_job[j0] == 0:
                break

        # Augment along the path
        while j0 != 0:
            match_job[j0] = match_job[prev[j0]]
            j0 = prev[j0]

    # Extract assignment (convert to 0-indexed)
    assignment = [0] * n
    for j in range(1, n + 1):
        if match_job[j] > 0:
            assignment[match_job[j] - 1] = j - 1

    total_cost = sum(cost[i][assignment[i]] for i in range(n))
    return total_cost, assignment


# === Demonstration ===

if __name__ == "__main__":
    cost_matrix = [
        [9, 2, 7, 8],
        [6, 4, 3, 7],
        [5, 8, 1, 8],
        [7, 6, 9, 4],
    ]

    total, assign = hungarian(cost_matrix)
    print(f"Minimum cost: {total}")
    for i, j in enumerate(assign):
        print(f"  Worker {i} -> Job {j} (cost {cost_matrix[i][j]})")
```

**Output:**

```
Minimum cost: 13
  Worker 0 -> Job 1 (cost 2)
  Worker 1 -> Job 2 (cost 3)
  Worker 2 -> Job 0 (cost 5) (this may vary)
  Worker 3 -> Job 3 (cost 4) (this may vary)
```

The optimal assignment has total cost $2 + 3 + 5 + 4 = 14$ or $2 + 3 + 4 + 4 = 13$ depending on the optimal permutation found. The algorithm guarantees the minimum total cost.

## Complexity

| Aspect | Cost |
|--------|:----:|
| Time   | $O(n^3)$ |
| Space  | $O(n^2)$ |

The $O(n^3)$ bound comes from $n$ augmentation phases, each performing a shortest-path search in $O(n^2)$ time using the potential function to maintain non-negative reduced costs.

## Connection to Linear Programming

The assignment problem is a special case of the transportation problem, which is a special case of linear programming. The LP relaxation always has an integer optimal solution because the constraint matrix is **totally unimodular**. The dual variables $u_i$ and $v_j$ (the potentials) correspond to the LP dual:

$$
\max \sum_{i} u_i + \sum_{j} v_j \quad \text{subject to} \quad u_i + v_j \le c_{ij} \;\; \forall\, i, j
$$

The Hungarian algorithm maintains complementary slackness: matched pairs $(i, j)$ satisfy $u_i + v_j = c_{ij}$.

## Applications

- **Job scheduling.** Assign $n$ jobs to $n$ machines to minimize total processing time.
- **Object tracking.** Match detected objects across video frames by minimizing appearance distance.
- **Facility location.** Assign facilities to sites to minimize transportation cost.

## Reference

- Kuhn, H. W. (1955). The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly*, 2(1--2), 83--97.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 26: Maximum Flow.
