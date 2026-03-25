# Task Scheduling

Operating systems must decide which of the waiting processes gets the CPU next. The simplest policy --- **first-come, first-served (FCFS)** --- uses a queue: processes are served in the order they arrive, and each process runs to completion before the next one starts. A more interactive policy --- **round-robin (RR)** --- also uses a queue but gives each process a fixed time slice (quantum) before moving it to the back of the queue. Both algorithms rely on the queue's FIFO property to ensure fairness. This page describes FCFS and round-robin scheduling, defines the key performance metrics, and compares the two approaches.

## Scheduling Metrics

Three metrics are used to evaluate scheduling algorithms:

**Turnaround time** is the total time from when a process arrives until it completes:

$$
T_{\text{turnaround}} = T_{\text{completion}} - T_{\text{arrival}}
$$

**Waiting time** is the total time a process spends in the ready queue, not running:

$$
T_{\text{waiting}} = T_{\text{turnaround}} - T_{\text{burst}}
$$

where $T_{\text{burst}}$ is the CPU time the process actually needs.

**Average turnaround time** and **average waiting time** are computed across all processes and provide aggregate performance measures.

## FCFS Scheduling

In FCFS (also called FIFO scheduling), processes are served in arrival order. Each process runs to completion before the next one begins.

**Advantages**: simple to implement, fair in the sense of arrival order.

**Disadvantage**: the **convoy effect** --- a long process at the front of the queue delays all subsequent processes, inflating average waiting time.

## Round-Robin Scheduling

Round-robin assigns each process a fixed **time quantum** $q$. When a process's quantum expires, it is preempted and moved to the back of the queue. If a process completes within its quantum, it leaves the queue.

Round-robin reduces the convoy effect by limiting how long any single process can monopolize the CPU. The choice of quantum is critical:

- **Too large** ($q \to \infty$): degenerates to FCFS
- **Too small** ($q \to 0$): excessive context-switching overhead

```python
"""
Task scheduling — FCFS and round-robin scheduling using queues.

Demonstrates how the queue's FIFO property supports fair process
scheduling, with computation of turnaround and waiting times.
"""
from collections import deque


# === Process Representation ===================================================

class Process:
    """Represents a process with an arrival time and burst time."""

    def __init__(self, name, arrival, burst):
        self.name = name
        self.arrival = arrival
        self.burst = burst
        self.remaining = burst
        self.completion = 0

    def __repr__(self):
        return f"{self.name}(arr={self.arrival}, burst={self.burst})"


# === FCFS Scheduling ==========================================================

def fcfs_schedule(processes):
    """First-Come, First-Served scheduling.

    Processes are sorted by arrival time and served in that order.
    Each runs to completion before the next begins.

    Time:  O(n log n) for sorting + O(n) for scheduling.
    Space: O(n).
    """
    procs = sorted(processes, key=lambda p: p.arrival)
    clock = 0
    results = []

    for p in procs:
        if clock < p.arrival:
            clock = p.arrival  # CPU idles until process arrives
        clock += p.burst
        p.completion = clock
        turnaround = p.completion - p.arrival
        waiting = turnaround - p.burst
        results.append((p.name, p.arrival, p.burst, p.completion, turnaround, waiting))

    return results


# === Round-Robin Scheduling ===================================================

def round_robin_schedule(processes, quantum):
    """Round-Robin scheduling with a fixed time quantum.

    Each process runs for at most `quantum` time units before being
    preempted and placed at the back of the queue.

    Time:  O(n * max_burst / quantum) in the worst case.
    Space: O(n).
    """
    # Create copies to avoid mutating originals
    procs = [Process(p.name, p.arrival, p.burst) for p in processes]
    procs.sort(key=lambda p: p.arrival)

    queue = deque()
    clock = 0
    idx = 0  # next process to arrive
    results = {}
    completed_order = []

    # Start with the first process
    if procs:
        clock = procs[0].arrival
        queue.append(procs[0])
        idx = 1

    while queue:
        p = queue.popleft()
        run_time = min(quantum, p.remaining)
        clock += run_time
        p.remaining -= run_time

        # Check for new arrivals during this time slice
        while idx < len(procs) and procs[idx].arrival <= clock:
            queue.append(procs[idx])
            idx += 1

        if p.remaining > 0:
            queue.append(p)  # not finished, re-enqueue
        else:
            p.completion = clock
            turnaround = p.completion - p.arrival
            waiting = turnaround - p.burst
            results[p.name] = (p.name, p.arrival, p.burst, p.completion, turnaround, waiting)
            completed_order.append(p.name)

        # If queue is empty but processes remain, jump to next arrival
        if not queue and idx < len(procs):
            clock = procs[idx].arrival
            queue.append(procs[idx])
            idx += 1

    return [results[name] for name in completed_order]


# === Display ==================================================================

def print_schedule(title, results):
    """Print scheduling results in a formatted table."""
    print(f"\n{title}")
    print(f"  {'Process':<10s} {'Arrival':>8s} {'Burst':>6s} {'Finish':>7s} {'Turnaround':>11s} {'Waiting':>8s}")
    print(f"  {'-'*52}")
    total_ta, total_wt = 0, 0
    for name, arr, burst, comp, ta, wt in results:
        print(f"  {name:<10s} {arr:>8d} {burst:>6d} {comp:>7d} {ta:>11d} {wt:>8d}")
        total_ta += ta
        total_wt += wt
    n = len(results)
    print(f"  {'-'*52}")
    print(f"  {'Average':<10s} {'':>8s} {'':>6s} {'':>7s} {total_ta/n:>11.1f} {total_wt/n:>8.1f}")


# === Demonstration ============================================================

if __name__ == "__main__":
    processes = [
        Process("P1", arrival=0, burst=8),
        Process("P2", arrival=1, burst=4),
        Process("P3", arrival=2, burst=2),
        Process("P4", arrival=3, burst=1),
    ]

    # FCFS
    fcfs_results = fcfs_schedule(processes)
    print_schedule("FCFS Scheduling:", fcfs_results)

    # Round-Robin with quantum = 3
    rr_results = round_robin_schedule(processes, quantum=3)
    print_schedule("Round-Robin Scheduling (quantum=3):", rr_results)
```

**Output:**
```
FCFS Scheduling:
  Process     Arrival  Burst  Finish  Turnaround  Waiting
  ----------------------------------------------------
  P1                0      8        8           8        0
  P2                1      4       12          11        7
  P3                2      2       14          12       10
  P4                3      1       15          12       11
  ----------------------------------------------------
  Average                                    10.8      7.0

Round-Robin Scheduling (quantum=3):
  Process     Arrival  Burst  Finish  Turnaround  Waiting
  ----------------------------------------------------
  P3                2      2        8           6        4
  P4                3      1        9           6        5
  P2                1      4       13          12        8
  P1                0      8       15          15        7
  ----------------------------------------------------
  Average                                     9.8      6.0
```

The FCFS schedule shows the convoy effect: P1 runs for 8 time units, forcing P2, P3, and P4 to wait even though they have short burst times. Round-robin with quantum 3 reduces average turnaround time from 10.8 to 9.8 by interleaving the processes.

## Comparison

| Property | FCFS | Round-Robin |
|----------|------|-------------|
| Fairness | Arrival order | Equal CPU share |
| Preemption | No | Yes (at quantum boundary) |
| Starvation | No | No |
| Convoy effect | Yes | Mitigated |
| Overhead | None | Context-switch at each quantum |
| Best for | Batch processing | Interactive systems |

!!! warning "Context-Switch Overhead"
    Round-robin introduces a context-switch cost each time a process is preempted. If the quantum is much smaller than the average burst time, the overhead can dominate actual computation. A typical quantum in modern systems is 10-100 milliseconds.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 10. MIT Press.
- Silberschatz, A., Galvin, P. B., & Gagne, G. (2018). *Operating System Concepts* (10th ed.), Chapter 5. Wiley.
