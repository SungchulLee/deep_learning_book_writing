# Job Scheduling

In many real-world settings --- manufacturing plants, operating systems, project management --- a set of jobs must be processed on a single machine, each with a deadline. Missing a deadline incurs a penalty, so the scheduler wants to arrange jobs to minimize the worst-case tardiness. The greedy solution, **Earliest Deadline First (EDF)**, simply processes jobs in order of their deadlines. Despite its simplicity, EDF is provably optimal for minimizing maximum lateness, and the proof illustrates the exchange argument beautifully.

## Problem Statement: Minimizing Maximum Lateness

**Input.** A set of $n$ jobs $\{1, 2, \ldots, n\}$. Job $i$ has:

- Processing time $p_i > 0$ (the time required to complete it).
- Deadline $d_i$ (the time by which it should ideally finish).

**Constraints.** A single machine processes one job at a time. All jobs are available at time 0. No preemption (once started, a job runs to completion). No idle time.

**Schedule.** A permutation $\sigma$ of $\{1, 2, \ldots, n\}$. Job $\sigma(j)$ is the $j$-th job processed. Its completion time is:

$$
C_{\sigma(j)} = \sum_{k=1}^{j} p_{\sigma(k)}
$$

**Lateness.** The lateness of job $i$ is $L_i = C_i - d_i$. A positive value means the job finishes after its deadline.

**Objective.** Minimize the **maximum lateness**:

$$
L_{\max} = \max_{1 \leq i \leq n} (C_i - d_i)
$$

## Greedy Algorithm: Earliest Deadline First

!!! note "EDF Scheduling"
    1. Sort jobs by deadline: $d_1 \leq d_2 \leq \cdots \leq d_n$.
    2. Process jobs in this order with no idle time.
    3. The $j$-th job completes at time $C_j = \sum_{k=1}^{j} p_k$.

The algorithm ignores processing times entirely when determining the order --- only deadlines matter.

## Worked Example

Consider four jobs:

| Job | $p_i$ | $d_i$ |
|-----|--------|--------|
| 1   | 3      | 6      |
| 2   | 2      | 8      |
| 3   | 1      | 9      |
| 4   | 4      | 9      |

**EDF order** (sorted by deadline): $1, 2, 3, 4$.

| Position | Job | $C_i$ | $d_i$ | $L_i = C_i - d_i$ |
|----------|-----|--------|--------|--------------------|
| 1        | 1   | 3      | 6      | $-3$               |
| 2        | 2   | 5      | 8      | $-3$               |
| 3        | 3   | 6      | 9      | $-3$               |
| 4        | 4   | 10     | 9      | $1$                |

$$
L_{\max} = 1
$$

**Alternative order** $2, 1, 3, 4$:

| Position | Job | $C_i$ | $d_i$ | $L_i$ |
|----------|-----|--------|--------|--------|
| 1        | 2   | 2      | 8      | $-6$   |
| 2        | 1   | 5      | 6      | $-1$   |
| 3        | 3   | 6      | 9      | $-3$   |
| 4        | 4   | 10     | 9      | $1$    |

This also gives $L_{\max} = 1$, which matches EDF. But no schedule achieves $L_{\max} < 1$, since the total processing time is 10 and the latest deadline is 9.

## Correctness Proof

**Theorem.** EDF minimizes the maximum lateness $L_{\max}$.

The proof uses the exchange argument, showing that any **inversion** in the schedule can be removed without increasing $L_{\max}$.

**Definition.** An **inversion** in a schedule is a pair of adjacent jobs $(i, j)$ where job $i$ is scheduled before job $j$ but $d_i > d_j$.

**Claim 1.** There exists an optimal schedule with no idle time.

*Proof.* Removing idle time shifts jobs earlier, which can only decrease lateness. $\square$

**Claim 2.** There exists an optimal schedule with no inversions.

??? example "Proof by Exchange"
    Suppose schedule $\sigma$ has an inversion: job $i$ immediately precedes job $j$ with $d_i > d_j$. Let $\sigma'$ be the schedule obtained by swapping $i$ and $j$.

    Before the swap, both jobs start at the same time $t$:

    - In $\sigma$: $C_i = t + p_i$, $C_j = t + p_i + p_j$
    - In $\sigma'$: $C_j' = t + p_j$, $C_i' = t + p_j + p_i$

    The completion times of all other jobs are unchanged.

    **Job $j$ improves:** $C_j' = t + p_j < t + p_i + p_j = C_j$, so $L_j' < L_j$.

    **Job $i$'s new lateness:** $L_i' = t + p_i + p_j - d_i$.

    **Key comparison:** Before the swap, $L_j = t + p_i + p_j - d_j$. After the swap, $L_i' = t + p_i + p_j - d_i$. Since $d_i > d_j$, we have $L_i' < L_j$.

    Therefore:

    $$
    L_{\max}(\sigma') = \max(L_j', L_i', \ldots) \leq \max(L_j, \ldots) = L_{\max}(\sigma)
    $$

    Swapping the inversion does not increase $L_{\max}$. $\square$

**Claim 3.** A schedule with no inversions processes jobs in EDF order.

**Conclusion.** Since inversions can be eliminated without increasing $L_{\max}$, and a schedule with no inversions is the EDF schedule, EDF is optimal.

## Python Implementation

```python
"""
Job scheduling to minimize maximum lateness using Earliest Deadline First.

Demonstrates that sorting jobs by deadline minimizes the worst-case
tardiness on a single machine with no preemption.
"""


# === EDF Scheduling ===

def edf_schedule(jobs):
    """Schedule jobs by Earliest Deadline First.

    Args:
        jobs: list of (processing_time, deadline) tuples

    Returns:
        Tuple of (schedule, max_lateness) where schedule is a list of
        (job_index, completion_time, lateness) tuples
    """
    # Sort by deadline, keep original indices
    indexed_jobs = sorted(enumerate(jobs), key=lambda x: x[1][1])

    schedule = []
    current_time = 0

    for original_idx, (proc_time, deadline) in indexed_jobs:
        current_time += proc_time
        lateness = current_time - deadline
        schedule.append((original_idx, current_time, lateness))

    max_lateness = max(entry[2] for entry in schedule)
    return schedule, max_lateness


if __name__ == "__main__":
    # Example: (processing_time, deadline)
    jobs = [(3, 6), (2, 8), (1, 9), (4, 9)]

    schedule, max_lateness = edf_schedule(jobs)

    print("EDF Schedule:")
    print(f"{'Job':>4} {'C_i':>6} {'d_i':>6} {'L_i':>6}")
    print("-" * 24)
    for job_idx, completion, lateness in schedule:
        proc, deadline = jobs[job_idx]
        print(f"{job_idx + 1:>4} {completion:>6} {deadline:>6} {lateness:>6}")
    print(f"\nMaximum lateness: {max_lateness}")
```

**Output:**
```
EDF Schedule:
 Job    C_i    d_i    L_i
------------------------
   1      3      6     -3
   2      5      8     -3
   3      6      9     -3
   4     10      9      1

Maximum lateness: 1
```

## Complexity Analysis

- **Sorting:** $O(n \log n)$.
- **Scheduling:** $O(n)$ single pass.
- **Total:** $O(n \log n)$.

## Variant: Minimizing Weighted Completion Time

A related problem minimizes the **total weighted completion time** $\sum_{i=1}^{n} w_i C_i$, where $w_i$ is the weight (priority) of job $i$.

**Greedy rule.** Process jobs in decreasing order of $w_i / p_i$ (weight-to-processing-time ratio).

This variant is also solvable by a greedy algorithm, with correctness proved by an adjacent-swap exchange argument: swapping two adjacent jobs $i$ and $j$ with $w_i/p_i < w_j/p_j$ strictly improves the objective.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 16. MIT Press.
- Kleinberg, J. & Tardos, E. (2006). *Algorithm Design*, Chapter 4.2. Pearson.
