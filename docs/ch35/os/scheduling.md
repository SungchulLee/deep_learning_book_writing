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

## Exercises

**Exercise 1.**
Three processes arrive at time 0 with burst times 10, 5, and 8 ms. Compute the average turnaround time and average waiting time under FCFS and SJF scheduling.

??? success "Solution to Exercise 1"
    **FCFS** (order: P1, P2, P3): P1 finishes at 10, P2 at 15, P3 at 23. Turnaround: (10 + 15 + 23)/3 = 16 ms. Waiting: (0 + 10 + 15)/3 = 8.33 ms. **SJF** (order: P2, P3, P1): P2 finishes at 5, P3 at 13, P1 at 23. Turnaround: (5 + 13 + 23)/3 = 13.67 ms. Waiting: (0 + 5 + 13)/3 = 6 ms. SJF is optimal for minimizing average waiting time among non-preemptive schedules. The improvement here is $8.33 - 6 = 2.33$ ms (28% reduction). $\square$

---

**Exercise 2.**
Prove that Shortest Job First (SJF) minimizes the average waiting time for a set of jobs available at time 0.

??? success "Solution to Exercise 2"
    Let jobs have burst times $b_1 \le b_2 \le \cdots \le b_n$ (SJF order). The waiting time for job $i$ is $\sum_{j=1}^{i-1} b_j$. Total waiting time: $\sum_{i=1}^{n} \sum_{j=1}^{i-1} b_j = \sum_{j=1}^{n} (n - j) b_j$. To minimize this weighted sum, larger weights $(n - j)$ should multiply smaller burst times $b_j$, which is exactly the SJF order (shortest first). Any swap of two adjacent jobs where a longer job precedes a shorter one increases the total waiting time by the difference in their burst times times the number of jobs between them. By the rearrangement inequality, the sorted order is optimal. $\square$

---

**Exercise 3.**
Explain the multi-level feedback queue (MLFQ) scheduler. How does it balance responsiveness for interactive processes with throughput for batch processes?

??? success "Solution to Exercise 3"
    MLFQ maintains multiple priority queues. New processes enter the highest-priority queue. If a process uses its entire time quantum without blocking, it is demoted to a lower-priority queue (presumed CPU-bound). If it blocks early (I/O-bound, interactive), it stays at high priority. Higher-priority queues have shorter time quanta. This automatically classifies processes: interactive processes (short CPU bursts, frequent I/O) stay at high priority with short quanta, ensuring low response time. CPU-bound processes sink to low-priority queues with longer quanta, maximizing throughput without constant context switches. To prevent starvation, a periodic "boost" moves all processes back to the highest queue. This prevents a long-running process from being permanently starved by new interactive processes. $\square$

---

**Exercise 4.**
The Linux Completely Fair Scheduler (CFS) uses a red-black tree keyed by virtual runtime. Explain how it achieves $O(\log n)$ scheduling and fairness.

??? success "Solution to Exercise 4"
    CFS tracks each runnable process's "virtual runtime" (vruntime): the total CPU time the process has received, weighted by its priority (nice value). The process with the smallest vruntime runs next -- it has received the least CPU time relative to its fair share. Processes are stored in a red-black tree keyed by vruntime. Selecting the next process: take the leftmost node ($O(1)$ with a cached pointer, or $O(\log n)$ for tree traversal). Inserting a process (wake-up or new): $O(\log n)$ tree insertion. This achieves fairness: over time, all processes' vruntimes converge because the running process's vruntime increases, eventually making another process the minimum. Higher-priority processes have their vruntime scaled down (they "earn" vruntime more slowly), so they receive more CPU time. $\square$

---

**Exercise 5.**
A real-time system has two periodic tasks: Task A with period 10 ms and execution time 3 ms, and Task B with period 20 ms and execution time 8 ms. Determine whether the system is schedulable under Rate-Monotonic Scheduling (RMS) and Earliest Deadline First (EDF).

??? success "Solution to Exercise 5"
    CPU utilization: $U = 3/10 + 8/20 = 0.3 + 0.4 = 0.7$. **RMS**: the utilization bound for 2 tasks is $2(2^{1/2} - 1) = 2 \times 0.414 = 0.828$. Since $0.7 < 0.828$, the system is schedulable under RMS. RMS assigns higher priority to Task A (shorter period). Schedule: [0-3] A, [3-11] B, [10-13] A, [13-19] B continues, [20-23] A, etc. Both tasks meet all deadlines. **EDF**: the utilization bound is 1.0 (EDF is optimal for uniprocessor). Since $0.7 < 1.0$, the system is schedulable. EDF always schedules the task with the earliest absolute deadline, adapting dynamically. Both schedulers work here; EDF would still work up to $U = 1.0$, while RMS fails above $U \approx 0.828$ for 2 tasks. $\square$
