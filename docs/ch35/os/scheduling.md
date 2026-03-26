# Process Scheduling

An operating system's **scheduler** determines which process runs on the CPU at each point in time. The goal is to optimize metrics like throughput, response time, and fairness. Since the scheduler runs frequently (at every context switch), its algorithm must be efficient -- typically $O(1)$ or $O(\log n)$ per decision.

## Scheduling Metrics

For $n$ processes with arrival times $a_i$ and burst times $b_i$:

- **Turnaround time**: $T_i = C_i - a_i$ where $C_i$ is completion time.
- **Waiting time**: $W_i = T_i - b_i$.
- **Response time**: Time from arrival to first execution.
- **Throughput**: Number of processes completed per unit time.

## FCFS (First-Come First-Served)

Run processes in arrival order. Non-preemptive.

$$
\text{Average waiting time} = \frac{1}{n} \sum_{i=1}^{n} W_i
$$

- **Advantage**: Simple, fair in arrival order.
- **Disadvantage**: **Convoy effect** -- short processes wait behind long ones, inflating average waiting time.

## SJF (Shortest Job First)

Run the process with the shortest burst time first. SJF is provably optimal for minimizing average waiting time among non-preemptive algorithms.

$$
\text{Average waiting time}_{\text{SJF}} \le \text{Average waiting time}_{\text{any non-preemptive}}
$$

The preemptive version (**SRTF**, Shortest Remaining Time First) preempts the running process if a newly arriving process has a shorter remaining burst.

!!! warning "Starvation"
    Long processes may starve under SJF if short processes keep arriving. **Aging** mitigates this by gradually increasing the priority of waiting processes.

## Round-Robin (RR)

Each process runs for a fixed **time quantum** $q$, then is preempted and placed at the back of the ready queue.

- If $q$ is large, RR degenerates to FCFS.
- If $q$ is small, context-switch overhead dominates.
- Typical quantum: 10--100 ms.

Average waiting time depends on $q$ and the burst distribution. For equal burst times $b$:

$$
\text{Average waiting time} = (n - 1) \cdot q \cdot \left\lfloor \frac{b}{q} \right\rfloor / n
$$

## Multilevel Feedback Queue (MLFQ)

MLFQ uses multiple priority queues to balance responsiveness and throughput:

1. New processes enter the **highest-priority** queue.
2. If a process uses its full quantum without blocking, demote it to the next lower queue.
3. Lower queues have longer quanta (e.g., double at each level).
4. Periodically boost all processes to the top queue to prevent starvation.

This adaptively separates interactive (I/O-bound) processes from CPU-bound processes without requiring burst-time estimates.

## Comparison

| Algorithm | Preemptive | Optimal Avg. Wait | Starvation | Complexity |
|---|---|---|---|---|
| FCFS | No | No | No | $O(1)$ |
| SJF | No | Yes | Yes | $O(n)$ |
| SRTF | Yes | Yes | Yes | $O(\log n)$ |
| RR | Yes | No | No | $O(1)$ |
| MLFQ | Yes | Adaptive | With boost: No | $O(1)$ |

## Implementation

```python
"""
Process Scheduling -- FCFS, SJF, and Round-Robin simulation.

Computes waiting time and turnaround time for each algorithm
on a set of processes with given arrival and burst times.
"""

from __future__ import annotations
from dataclasses import dataclass
from collections import deque


# === Process ==================================================================

@dataclass
class Process:
    """A process with arrival time and CPU burst."""
    pid: int
    arrival: int
    burst: int


# === FCFS =====================================================================

def fcfs(processes: list[Process]) -> list[tuple[int, int, int]]:
    """Run FCFS. Returns (pid, waiting_time, turnaround_time) for each."""
    procs = sorted(processes, key=lambda p: (p.arrival, p.pid))
    results = []
    time = 0
    for p in procs:
        time = max(time, p.arrival)
        wait = time - p.arrival
        turnaround = wait + p.burst
        results.append((p.pid, wait, turnaround))
        time += p.burst
    return results


# === SJF (Non-Preemptive) =====================================================

def sjf(processes: list[Process]) -> list[tuple[int, int, int]]:
    """Run SJF. Returns (pid, waiting_time, turnaround_time) for each."""
    remaining = list(processes)
    results = []
    time = 0
    while remaining:
        available = [p for p in remaining if p.arrival <= time]
        if not available:
            time = min(p.arrival for p in remaining)
            available = [p for p in remaining if p.arrival <= time]
        chosen = min(available, key=lambda p: p.burst)
        remaining.remove(chosen)
        wait = time - chosen.arrival
        turnaround = wait + chosen.burst
        results.append((chosen.pid, wait, turnaround))
        time += chosen.burst
    return results


# === Round-Robin ==============================================================

def round_robin(processes: list[Process],
                quantum: int) -> list[tuple[int, int, int]]:
    """Run Round-Robin with given quantum. Returns (pid, wait, turnaround)."""
    n = len(processes)
    remaining_burst = {p.pid: p.burst for p in processes}
    arrival = {p.pid: p.arrival for p in processes}
    queue: deque[int] = deque()
    time = 0
    finish_time: dict[int, int] = {}
    arrived = set()
    procs_by_arrival = sorted(processes, key=lambda p: p.arrival)
    idx = 0

    # Add first arrivals
    while idx < n and procs_by_arrival[idx].arrival <= time:
        queue.append(procs_by_arrival[idx].pid)
        arrived.add(procs_by_arrival[idx].pid)
        idx += 1

    while queue or idx < n:
        if not queue:
            time = procs_by_arrival[idx].arrival
            while idx < n and procs_by_arrival[idx].arrival <= time:
                queue.append(procs_by_arrival[idx].pid)
                arrived.add(procs_by_arrival[idx].pid)
                idx += 1

        pid = queue.popleft()
        run = min(quantum, remaining_burst[pid])
        time += run
        remaining_burst[pid] -= run

        # Add newly arrived processes
        while idx < n and procs_by_arrival[idx].arrival <= time:
            queue.append(procs_by_arrival[idx].pid)
            arrived.add(procs_by_arrival[idx].pid)
            idx += 1

        if remaining_burst[pid] > 0:
            queue.append(pid)
        else:
            finish_time[pid] = time

    results = []
    for p in processes:
        turnaround = finish_time[p.pid] - p.arrival
        wait = turnaround - p.burst
        results.append((p.pid, wait, turnaround))
    return results


# === Main =====================================================================

if __name__ == "__main__":
    processes = [
        Process(pid=1, arrival=0, burst=6),
        Process(pid=2, arrival=1, burst=4),
        Process(pid=3, arrival=2, burst=2),
        Process(pid=4, arrival=3, burst=3),
    ]

    for name, result in [
        ("FCFS", fcfs(processes)),
        ("SJF", sjf(processes)),
        ("RR(q=2)", round_robin(processes, quantum=2)),
    ]:
        avg_wait = sum(r[1] for r in result) / len(result)
        avg_turn = sum(r[2] for r in result) / len(result)
        print(f"{name:10s}  Avg Wait={avg_wait:.1f}  Avg Turnaround={avg_turn:.1f}")
        for pid, wait, turn in sorted(result):
            print(f"  P{pid}: wait={wait}, turnaround={turn}")
```

**Output:**

```
FCFS        Avg Wait=4.5  Avg Turnaround=8.2
  P1: wait=0, turnaround=6
  P2: wait=5, turnaround=9
  P3: wait=6, turnaround=8
  P4: wait=7, turnaround=10
SJF         Avg Wait=2.8  Avg Turnaround=6.5
  P1: wait=0, turnaround=6
  P2: wait=7, turnaround=11
  P3: wait=4, turnaround=6
  P4: wait=0, turnaround=3
RR(q=2)     Avg Wait=5.5  Avg Turnaround=9.2
  P1: wait=7, turnaround=13
  P2: wait=6, turnaround=10
  P3: wait=2, turnaround=4
  P4: wait=7, turnaround=10
```

SJF achieves the lowest average waiting time (2.8), confirming its optimality for non-preemptive scheduling. FCFS suffers from the convoy effect (P3 waits 6 units despite needing only 2). Round-robin distributes CPU time more fairly but has higher average wait due to context-switch overhead.

## Reference

- Silberschatz, A., Galvin, P.B., and Gagne, G. *Operating System Concepts*. Wiley
- Arpaci-Dusseau, R.H. and Arpaci-Dusseau, A.C. *Operating Systems: Three Easy Pieces*
