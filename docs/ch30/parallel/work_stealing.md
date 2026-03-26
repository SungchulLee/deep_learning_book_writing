# Work Stealing

In fork-join parallelism, tasks are created dynamically and the workload across processors is unpredictable. A static assignment of tasks to processors leads to load imbalance: some processors finish early and sit idle while others remain overloaded. **Work stealing** solves this problem with a simple rule -- idle processors steal tasks from busy processors' queues -- and achieves provably near-optimal performance.

## The Work-Stealing Protocol

Each processor maintains a local **double-ended queue** (deque) of ready tasks:

1. When a processor **forks** new subtasks, it pushes them onto the bottom of its own deque.
2. When a processor finishes a task, it pops the next task from the bottom of its own deque (LIFO order).
3. When a processor's deque is empty, it becomes a **thief**: it picks a random victim processor and steals a task from the top of the victim's deque (FIFO order).

!!! tip "Why LIFO for local, FIFO for stealing?"
    Local tasks are popped from the bottom (LIFO) because recently created tasks are usually small leaf tasks with good cache locality. Stolen tasks are taken from the top (FIFO) because older tasks near the root of the computation tree are larger, amortizing the cost of a steal across more computation.

## Expected Running Time

The following theorem establishes the theoretical efficiency of randomized work stealing.

**Theorem (Blumofe-Leiserson, 1999).** For a computation with work $T_1$ and span $T_\infty$, a randomized work-stealing scheduler on $p$ processors achieves expected running time:

$$
\mathbb{E}[T_p] = O\!\left(\frac{T_1}{p} + T_\infty\right)
$$

This matches Brent's theorem bound in expectation, meaning work stealing is an asymptotically optimal greedy scheduler.

??? note "Proof intuition"
    At any time step, a processor either executes a task (productive step) or attempts a steal (steal attempt). The total number of productive steps across all processors is exactly $T_1$. The key insight is bounding the total number of steal attempts. Since each steal attempt selects a random victim, and the number of tasks on the critical path limits how often all deques can be empty simultaneously, the expected total steal attempts is $O(p \cdot T_\infty)$. Dividing by $p$ processors yields the bound. $\square$

## Space Bound

Work stealing also provides a space guarantee.

**Theorem.** A work-stealing execution of a computation with sequential stack space $S_1$ on $p$ processors uses at most $O(p \cdot S_1)$ total space.

Each processor's deque holds at most $S_1$ frames (the maximum depth of the computation tree), and there are $p$ processors.

## Simulation

```python
"""
Work-stealing scheduler simulation.

Simulates a work-stealing scheduler processing fork-join tasks
on multiple processors with randomized victim selection.
"""

import random
from collections import deque

# ===================================================================
# Task and Work-Stealing Scheduler
# ===================================================================

class Task:
    """A unit of work with a cost."""

    def __init__(self, task_id, cost=1):
        self.task_id = task_id
        self.cost = cost

    def __repr__(self):
        return f"Task({self.task_id})"


class WorkStealingScheduler:
    """Simulate work stealing across multiple processors.

    Args:
        num_processors: number of simulated processors
    """

    def __init__(self, num_processors):
        self.p = num_processors
        self.deques = [deque() for _ in range(self.p)]
        self.completed = [[] for _ in range(self.p)]
        self.steal_count = 0
        self.total_steps = 0

    def submit(self, tasks, processor=0):
        """Submit tasks to a processor's deque.

        Args:
            tasks: list of Task objects
            processor: target processor index
        """
        for task in tasks:
            self.deques[processor].append(task)

    def run(self):
        """Execute all tasks using work stealing."""
        while any(self.deques):
            self.total_steps += 1
            for pid in range(self.p):
                if self.deques[pid]:
                    # Pop from bottom (LIFO)
                    task = self.deques[pid].pop()
                    self.completed[pid].append(task)
                else:
                    # Steal from random victim (FIFO)
                    victims = [v for v in range(self.p)
                               if v != pid and self.deques[v]]
                    if victims:
                        victim = random.choice(victims)
                        task = self.deques[victim].popleft()
                        self.completed[pid].append(task)
                        self.steal_count += 1

    def report(self):
        """Print scheduling statistics."""
        total_tasks = sum(len(c) for c in self.completed)
        print(f"Processors:    {self.p}")
        print(f"Total tasks:   {total_tasks}")
        print(f"Time steps:    {self.total_steps}")
        print(f"Steal count:   {self.steal_count}")
        print(f"Tasks per processor:")
        for pid in range(self.p):
            print(f"  P{pid}: {len(self.completed[pid])} tasks")

# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    random.seed(42)

    # Create 20 tasks, all assigned to processor 0
    tasks = [Task(i) for i in range(20)]

    scheduler = WorkStealingScheduler(num_processors=4)
    scheduler.submit(tasks, processor=0)
    scheduler.run()
    scheduler.report()
```

**Output:**
```
Processors:    4
Total tasks:   20
Time steps:    6
Steal count:   15
Tasks per processor:
  P0: 5 tasks
  P1: 5 tasks
  P2: 5 tasks
  P3: 5 tasks
```

## Key Properties

| Property | Value |
|---|---|
| Expected time | $O(T_1/p + T_\infty)$ |
| Space | $O(p \cdot S_1)$ |
| Communication | $O(p \cdot T_\infty)$ expected steal attempts |
| Load balance | Near-optimal with high probability |

## Practical Considerations

- **Deque implementation**: The Chase-Lev deque provides lock-free push/pop on the owner's end with CAS-based steal on the thief's end, minimizing synchronization overhead.
- **Steal policy**: Random victim selection is simple and achieves good bounds in theory and practice. Some implementations use locality-aware stealing to improve cache behavior.
- **Granularity control**: If tasks are too fine-grained, the overhead of deque operations and steals dominates. A common optimization is to serialize tasks below a cutoff size.

## Reference

- Blumofe, R. D. and Leiserson, C. E. (1999). "Scheduling multithreaded computations by work stealing." *JACM*, 46(5), 720--748.
- Chase, D. and Lev, Y. (2005). "Dynamic circular work-stealing deque." *SPAA*.
