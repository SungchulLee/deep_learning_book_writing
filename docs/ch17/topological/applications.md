# Applications of Topological Sort

Topological sorting is not merely a graph algorithm exercise -- it appears whenever we need to process items in an order that respects dependencies. Build systems compile source files in dependency order, schedulers assign tasks to workers respecting prerequisites, and compilers resolve symbol definitions before their uses. This page surveys the most important applications, each of which reduces to computing a topological order on an appropriate [DAG](dag.md).

## Task Scheduling

The most direct application of topological sorting is task scheduling with precedence constraints. Given a set of tasks with dependencies, a topological order provides a valid execution sequence.

**Problem.** Given $n$ tasks and a set of precedence constraints $(i, j)$ meaning "task $i$ must complete before task $j$ starts," find a valid execution order.

This is exactly the topological sort problem: model tasks as vertices and constraints as directed edges, then apply [Kahn's algorithm](kahn.md) or [DFS-based sort](dfs.md).

```python
"""
Task scheduling using topological sort.

Demonstrates how topological ordering determines a valid execution
sequence for tasks with precedence constraints.
"""

from collections import deque


# === Task Scheduler ===
def schedule_tasks(tasks, dependencies):
    """
    Compute a valid task execution order respecting all dependencies.

    Parameters
    ----------
    tasks : list[str]
        List of task names.
    dependencies : list[tuple[str, str]]
        Each (a, b) means task a must complete before task b.

    Returns
    -------
    list[str]
        Tasks in a valid execution order, or empty if circular dependency.
    """
    idx = {t: i for i, t in enumerate(tasks)}
    n = len(tasks)
    graph = {i: [] for i in range(n)}
    in_degree = [0] * n

    for a, b in dependencies:
        graph[idx[a]].append(idx[b])
        in_degree[idx[b]] += 1

    queue = deque(i for i in range(n) if in_degree[i] == 0)
    order = []

    while queue:
        u = queue.popleft()
        order.append(tasks[u])
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    return order if len(order) == n else []


# === Main ===
if __name__ == "__main__":
    tasks = ["design", "implement", "test", "document", "deploy"]
    deps = [
        ("design", "implement"),
        ("implement", "test"),
        ("implement", "document"),
        ("test", "deploy"),
        ("document", "deploy"),
    ]
    result = schedule_tasks(tasks, deps)
    print(f"Execution order: {result}")
```

**Output:**
```
Execution order: ['design', 'implement', 'test', 'document', 'deploy']
```

## Shortest and Longest Paths in DAGs

In a general weighted graph, shortest path algorithms like Dijkstra or Bellman-Ford have complexities of $O((V + E) \log V)$ or $O(VE)$. In a DAG, topological ordering enables a single-pass solution in $O(V + E)$ time, even with negative edge weights.

**Algorithm.** Process vertices in topological order. For each vertex $u$, relax all outgoing edges:

$$
d[v] = \min(d[v],\ d[u] + w(u, v))
$$

Since $u$ appears before $v$ in the topological order, $d[u]$ is finalized when we process edge $(u, v)$.

For the **longest path**, simply negate all weights or replace $\min$ with $\max$. The longest path in a general graph is NP-hard, but in a DAG it is solvable in linear time.

```python
"""
Shortest and longest paths in a DAG using topological sort.

Processes vertices in topological order for a single-pass O(V + E)
solution, which works even with negative edge weights.
"""

from collections import deque


# === DAG Shortest Path ===
def dag_shortest_path(graph, n, source):
    """
    Compute shortest distances from source in a weighted DAG.

    Parameters
    ----------
    graph : dict[int, list[tuple[int, float]]]
        Adjacency list with (neighbor, weight) pairs.
    n : int
        Number of vertices.
    source : int
        Source vertex.

    Returns
    -------
    list[float]
        Shortest distance from source to each vertex.
    """
    # Topological sort via Kahn's
    in_degree = [0] * n
    adj = {i: [] for i in range(n)}
    for u in range(n):
        for v, w in graph.get(u, []):
            adj.setdefault(u, [])
            in_degree[v] += 1

    queue = deque(i for i in range(n) if in_degree[i] == 0)
    topo = []
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v, w in graph.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # Relax edges in topological order
    dist = [float("inf")] * n
    dist[source] = 0
    for u in topo:
        if dist[u] == float("inf"):
            continue
        for v, w in graph.get(u, []):
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    return dist


# === Main ===
if __name__ == "__main__":
    # Weighted DAG: 0->1(2), 0->2(4), 1->2(1), 1->3(7), 2->3(3)
    g = {
        0: [(1, 2), (2, 4)],
        1: [(2, 1), (3, 7)],
        2: [(3, 3)],
        3: [],
    }
    dist = dag_shortest_path(g, 4, 0)
    print(f"Shortest distances from 0: {dist}")
```

**Output:**
```
Shortest distances from 0: [0, 2, 3, 6]
```

## Build Systems

Build systems like Make, Gradle, and Bazel model source file dependencies as a DAG. Compiling files in topological order ensures every dependency is built before the files that depend on it.

**Structure:**

- Each file or module is a vertex.
- A directed edge $(A, B)$ means "file $A$ must be compiled before file $B$."
- Topological sort determines the build order.
- Circular dependencies are detected when topological sort fails (the output contains fewer vertices than the total count).

## Course Prerequisites

Universities often model course prerequisite structures as DAGs. A topological order gives a valid semester-by-semester plan. The "parallel layers" variant of [Kahn's algorithm](kahn.md) naturally identifies which courses can be taken simultaneously: all sources in the same round of Kahn's algorithm have no mutual prerequisites.

## Critical Path Method

In project management, the **critical path** is the longest path through a task dependency DAG. Each task has a duration, and the critical path determines the minimum project completion time.

**Algorithm:**

1. Topologically sort the tasks.
2. Compute the earliest start time for each task using forward pass (longest path from the source).
3. Compute the latest start time using backward pass.
4. Tasks where earliest start equals latest start lie on the critical path.

The total time equals the longest path length, computable in $O(V + E)$ as described above.

## Data Processing Pipelines

Modern data pipelines (Apache Airflow, Dagster, Prefect) represent workflow steps as DAGs. Topological sort determines the execution order, and the layered structure from Kahn's algorithm identifies which steps can run in parallel.

!!! tip "Connection to Deep Learning"
    Neural network computation graphs are DAGs where nodes represent operations and edges represent data flow. Frameworks like PyTorch and TensorFlow use topological ordering to schedule forward and backward passes. The reverse topological order of the forward pass gives the correct order for backpropagation.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 20.
- Kahn, A. B. (1962). Topological sorting of large networks. *Communications of the ACM*, 5(11), 558-562.
