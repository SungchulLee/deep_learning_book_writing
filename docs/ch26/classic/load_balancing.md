# Load Balancing Approximation

Suppose you have $m$ identical machines and $n$ jobs with processing times
$p_1, p_2, \dots, p_n$. You want to assign every job to exactly one machine so
that the **makespan** — the maximum total load on any machine — is minimized.
This is the **minimum makespan scheduling** (or load balancing) problem, and it
is NP-hard even for $m = 2$. Greedy algorithms provide surprisingly good
approximations.

## Problem Definition

Given $m$ machines and $n$ jobs with processing times $p_j > 0$, find an
assignment $\sigma: \{1, \dots, n\} \to \{1, \dots, m\}$ minimizing the
makespan

$$
C_{\max} = \max_{i=1}^{m} \sum_{\substack{j : \sigma(j) = i}} p_j
$$

Let $\text{OPT}$ denote the optimal makespan. Two simple lower bounds hold:

$$
\text{OPT} \ge \frac{1}{m} \sum_{j=1}^n p_j
\qquad \text{and} \qquad
\text{OPT} \ge \max_j\, p_j
$$

The first follows because the average load is $\sum p_j / m$, and the max is
at least the average. The second holds because some machine must process the
largest job.

## List Scheduling (Graham's Algorithm)

**Intuition.** Process jobs in arbitrary order. Assign each job to the machine
with the smallest current load. This greedy rule never lets any machine sit
idle while work is available.

!!! tip "Theorem (Graham, 1966)"
    List Scheduling produces a makespan at most $(2 - 1/m) \cdot \text{OPT}$.

**Proof.** Let machine $i^*$ achieve the makespan $C_{\max}$, and let $j^*$
be the last job assigned to $i^*$. At the time $j^*$ was assigned, $i^*$ had
the smallest load among all machines, so

$$
C_{\max} - p_{j^*} \le \frac{1}{m} \sum_{j=1}^n p_j
$$

Using both lower bounds:

$$
C_{\max} = (C_{\max} - p_{j^*}) + p_{j^*}
\le \frac{1}{m} \sum_{j=1}^n p_j + \max_j\, p_j
\le \text{OPT} + \text{OPT} - \frac{\text{OPT}}{m}
$$

Wait — let us be more precise. We have $C_{\max} - p_{j^*} \le \text{OPT}$
(from the average bound) and $p_{j^*} \le \text{OPT}$ (from the max-job
bound), but we need a tighter combination. Since

$$
C_{\max} - p_{j^*} \le \frac{1}{m}\sum p_j \le \text{OPT}
$$

we get $C_{\max} \le \text{OPT} + p_{j^*} \le \text{OPT} + \text{OPT} = 2 \cdot \text{OPT}$.

For the refined $(2 - 1/m)$ bound: the load on $i^*$ before $j^*$ was at most
the average of *all* machines' loads, and $p_{j^*} \le \text{OPT}$, so

$$
C_{\max} \le \frac{1}{m}\sum_{j=1}^{n} p_j + p_{j^*}
- \frac{p_{j^*}}{m}
= \frac{1}{m}\sum p_j + \left(1 - \frac{1}{m}\right)p_{j^*}
\le \text{OPT} + \left(1 - \frac{1}{m}\right)\text{OPT}
= \left(2 - \frac{1}{m}\right)\text{OPT} \qquad \blacksquare
$$

## Longest Processing Time (LPT)

**Intuition.** Sorting jobs in decreasing order before applying list scheduling
prevents large jobs from landing on already-loaded machines late in the process.

!!! tip "Theorem (Graham, 1969)"
    LPT scheduling achieves a makespan at most $(4/3 - 1/(3m)) \cdot
    \text{OPT}$.

The improved ratio comes from the fact that if $p_{j^*}$ (the last job placed)
satisfies $p_{j^*} \le \text{OPT}/3$ (since it is the smallest remaining job
in sorted order and at least $m + 1$ jobs exist), the bound tightens to $4/3$.

## Implementation

```python
"""
Load Balancing: List Scheduling and LPT approximation algorithms.
"""

import heapq


# === List Scheduling ==========================================================

def list_scheduling(jobs, m):
    """
    Greedy list scheduling: assign each job to least-loaded machine.

    Returns (makespan, assignment).
    Approximation ratio: 2 - 1/m.
    """
    # Min-heap: (current_load, machine_id)
    machines = [(0, i) for i in range(m)]
    assignment = [0] * len(jobs)

    for j, p in enumerate(jobs):
        load, mid = heapq.heappop(machines)
        assignment[j] = mid
        heapq.heappush(machines, (load + p, mid))

    makespan = max(load for load, _ in machines)
    return makespan, assignment


# === LPT Scheduling ===========================================================

def lpt_scheduling(jobs, m):
    """
    Longest Processing Time first scheduling.

    Returns (makespan, assignment using original indices).
    Approximation ratio: 4/3 - 1/(3m).
    """
    indexed = sorted(enumerate(jobs), key=lambda x: -x[1])
    machines = [(0, i) for i in range(m)]
    assignment = [0] * len(jobs)

    for orig_idx, p in indexed:
        load, mid = heapq.heappop(machines)
        assignment[orig_idx] = mid
        heapq.heappush(machines, (load + p, mid))

    makespan = max(load for load, _ in machines)
    return makespan, assignment


# === Demo =====================================================================

if __name__ == "__main__":
    jobs = [6, 3, 8, 5, 2, 7, 4, 1]
    m = 3

    ms_list, assign_list = list_scheduling(jobs, m)
    print(f"List Scheduling: makespan={ms_list}")
    for i in range(m):
        task_indices = [j for j in range(len(jobs)) if assign_list[j] == i]
        load = sum(jobs[j] for j in task_indices)
        print(f"  Machine {i}: jobs={task_indices}, load={load}")

    print()
    ms_lpt, assign_lpt = lpt_scheduling(jobs, m)
    print(f"LPT Scheduling:  makespan={ms_lpt}")
    for i in range(m):
        task_indices = [j for j in range(len(jobs)) if assign_lpt[j] == i]
        load = sum(jobs[j] for j in task_indices)
        print(f"  Machine {i}: jobs={task_indices}, load={load}")

    lb = sum(jobs) / m
    print(f"\nLower bound (avg): {lb:.1f}")
    print(f"Lower bound (max): {max(jobs)}")
```

## Summary

| Algorithm | Ratio | Time |
|---|---|---|
| List Scheduling | $2 - 1/m$ | $O(n \log m)$ |
| LPT | $4/3 - 1/(3m)$ | $O(n \log n)$ |
| PTAS (Hochbaum-Shmoys) | $1 + \epsilon$ | $O(n \cdot (n/\epsilon)^{O(m)})$ |

For fixed $m$, a PTAS exists (Hochbaum and Shmoys, 1987), but for variable $m$
no FPTAS is possible unless P = NP, since the problem is strongly NP-hard.

## Reference

- Graham, R. L. "Bounds on Multiprocessing Timing Anomalies." *SIAM J. Appl. Math.*, 1969.
- Hochbaum, D. S. and Shmoys, D. B. "Using Dual Approximation Algorithms for
  Scheduling Problems." *JACM*, 1987.
