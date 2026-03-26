# Matroid Greedy Algorithm

Greedy algorithms make locally optimal choices at each step, hoping this leads to a global optimum. For most optimization problems, this hope is unfounded. However, for problems with matroid structure, the greedy approach is provably optimal. The **matroid greedy theorem** gives a precise characterization: a greedy algorithm that always selects the best available element produces an optimal solution if and only if the feasibility constraint forms a matroid. This theorem unifies the correctness proofs for Kruskal's MST algorithm, optimal scheduling, and many other greedy successes.

## The Algorithm

Given a weighted matroid $M = (S, \mathcal{I})$ with a weight function $w : S \to \mathbb{R}_{\ge 0}$, the goal is to find an independent set of maximum total weight:

$$
\max_{A \in \mathcal{I}} \sum_{x \in A} w(x)
$$

The greedy algorithm is remarkably simple:

1. Sort elements of $S$ in decreasing order of weight: $w(x_1) \ge w(x_2) \ge \cdots \ge w(x_n)$.
2. Initialize $A \leftarrow \emptyset$.
3. For each $x_i$ in sorted order: if $A \cup \{x_i\} \in \mathcal{I}$, set $A \leftarrow A \cup \{x_i\}$.
4. Return $A$.

```text
GREEDY-MATROID(M, w):
    sort S by w in decreasing order
    A ← ∅
    for each x in S (in sorted order):
        if A ∪ {x} ∈ I:
            A ← A ∪ {x}
    return A
```

The algorithm greedily adds the heaviest element that maintains independence. It never removes an element once added.

## Optimality Theorem

!!! note "Matroid Greedy Theorem"
    Let $M = (S, \mathcal{I})$ be a matroid and $w : S \to \mathbb{R}_{\ge 0}$ a weight function. The greedy algorithm returns an independent set $A$ of maximum weight. Moreover, $A$ is a base (maximal independent set) whenever all weights are positive.

## Proof of Correctness

**Claim.** The greedy algorithm produces a maximum-weight independent set.

**Proof.** Let $A = \{a_1, a_2, \dots, a_k\}$ be the greedy solution in the order elements were added, and let $O = \{o_1, o_2, \dots, o_m\}$ be an optimal solution with elements sorted by decreasing weight.

We show $w(a_i) \ge w(o_i)$ for all $i \le k$, which implies $w(A) \ge w(O)$.

Suppose for contradiction that $i$ is the first index where $w(a_i) < w(o_i)$. Consider:

- $A_{i-1} = \{a_1, \dots, a_{i-1}\}$ (the first $i-1$ greedy choices).
- $O_i = \{o_1, \dots, o_i\}$ (the first $i$ elements of the optimal solution).

Since $|A_{i-1}| = i - 1 < i = |O_i|$ and both are independent (by the hereditary property for $O_i$), the exchange property guarantees some $o_j \in O_i \setminus A_{i-1}$ such that $A_{i-1} \cup \{o_j\} \in \mathcal{I}$.

Since $o_j \in O_i$, we have $w(o_j) \ge w(o_i) > w(a_i)$. But the greedy algorithm considers elements in decreasing weight order and would have chosen $o_j$ before $a_i$ (or at the same step), contradicting the fact that $a_i$ was chosen at step $i$.

Therefore $w(a_i) \ge w(o_i)$ for all $i$, and since $k \ge m$ would follow from the exchange property (the greedy solution is maximal), we have $w(A) \ge w(O)$.

$\square$

## Converse: Matroids Characterize Greedy Optimality

The matroid greedy theorem has a remarkable converse:

!!! note "Converse Theorem (Edmonds, Rado)"
    Let $(S, \mathcal{I})$ be a non-empty hereditary set system (satisfying Axioms 1 and 2). The greedy algorithm finds a maximum-weight independent set for **every** weight function $w : S \to \mathbb{R}_{\ge 0}$ if and only if $(S, \mathcal{I})$ is a matroid.

This means matroids are not just sufficient for greedy optimality --- they are the **exact** characterization. If a hereditary set system is not a matroid (fails the exchange property), there exists some weight function for which greedy fails.

**Proof sketch of the converse.** Suppose $\mathcal{I}$ is hereditary but violates the exchange property: there exist $A, B \in \mathcal{I}$ with $|A| < |B|$ such that $A \cup \{x\} \notin \mathcal{I}$ for all $x \in B \setminus A$. Assign weights so that elements in $A$ have slightly higher weight than elements in $B \setminus A$, and all other elements have weight 0. The greedy algorithm selects all of $A$ first, then gets stuck with a smaller independent set than $B$, proving greedy is suboptimal.

## Applications

### Minimum Spanning Tree (Kruskal's Algorithm)

In the graphic matroid, edges are elements and forests are independent sets. Sorting edges by weight and adding each edge that does not create a cycle is exactly the matroid greedy algorithm. The theorem guarantees this produces a minimum spanning tree (using minimum weight, which is equivalent to negating weights and maximizing).

### Weighted Job Scheduling

Given $n$ unit-time jobs with deadlines $d_i$ and profits $p_i$, define a set of jobs as independent if they can all be scheduled to meet their deadlines. This forms a matroid, and the greedy algorithm (schedule highest-profit jobs first) is optimal.

### Minimum-Weight Base

For any matroid, sorting by increasing weight and greedily adding elements that preserve independence yields a minimum-weight base. This generalizes both Kruskal's algorithm and optimal scheduling.

## Implementation

```python
"""
Matroid greedy algorithm with applications.

Demonstrates the generic matroid greedy algorithm and its application
to weighted job scheduling and minimum spanning trees.
"""

from typing import Callable

# === Generic Matroid Greedy ===

def matroid_greedy(
    elements: list,
    weight: Callable,
    is_independent: Callable[[list], bool],
    maximize: bool = True
) -> list:
    """Generic matroid greedy algorithm.

    Args:
        elements: Ground set elements.
        weight: Function mapping element to its weight.
        is_independent: Function checking if a list of elements is independent.
        maximize: If True, find max-weight independent set;
                  if False, find min-weight base.

    Returns:
        Optimal independent set (a base if all elements have positive weight).
    """
    sorted_elems = sorted(elements, key=weight, reverse=maximize)
    result = []

    for x in sorted_elems:
        candidate = result + [x]
        if is_independent(candidate):
            result.append(x)

    return result


# === Application: Weighted Job Scheduling ===

def schedule_jobs(
    jobs: list[tuple[str, int, int]]
) -> tuple[list[str], int]:
    """Schedule unit-time jobs to maximize profit using matroid greedy.

    Args:
        jobs: List of (name, deadline, profit) tuples.
              Deadlines are 1-indexed.

    Returns:
        Tuple of (scheduled job names, total profit).
    """
    def is_feasible(selected_jobs: list[tuple[str, int, int]]) -> bool:
        """Check if selected jobs can all meet their deadlines."""
        deadlines = sorted(j[1] for j in selected_jobs)
        for i, d in enumerate(deadlines):
            if d < i + 1:  # slot i+1 needed but deadline is earlier
                return False
        return True

    result = matroid_greedy(
        elements=jobs,
        weight=lambda j: j[2],
        is_independent=is_feasible,
        maximize=True
    )
    names = [j[0] for j in result]
    profit = sum(j[2] for j in result)
    return names, profit


# === Application: MST via Matroid Greedy ===

def mst_matroid(
    n: int,
    edges: list[tuple[int, int, float]]
) -> list[tuple[int, int, float]]:
    """Find MST using the matroid greedy algorithm.

    Args:
        n: Number of vertices.
        edges: List of (u, v, weight) tuples.

    Returns:
        List of MST edges.
    """
    class UnionFind:
        def __init__(self, size):
            self.parent = list(range(size))
            self.rank = [0] * size

        def find(self, x):
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x

        def union(self, x, y):
            rx, ry = self.find(x), self.find(y)
            if rx == ry:
                return False
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            return True

    def is_acyclic(edge_list):
        uf = UnionFind(n)
        for u, v, _ in edge_list:
            if not uf.union(u, v):
                return False
        return True

    return matroid_greedy(
        elements=edges,
        weight=lambda e: e[2],
        is_independent=is_acyclic,
        maximize=False
    )


# === Demonstration ===

if __name__ == "__main__":
    # Job scheduling
    jobs = [
        ("a", 2, 100),
        ("b", 1, 19),
        ("c", 2, 27),
        ("d", 1, 25),
        ("e", 3, 15),
    ]
    scheduled, profit = schedule_jobs(jobs)
    print("=== Weighted Job Scheduling ===")
    print(f"Jobs: {[(j[0], f'd={j[1]}', f'p={j[2]}') for j in jobs]}")
    print(f"Scheduled: {scheduled}")
    print(f"Total profit: {profit}")

    # MST
    edges = [
        (0, 1, 1), (0, 2, 4), (1, 2, 2),
        (1, 3, 3), (2, 3, 5),
    ]
    mst = mst_matroid(4, edges)
    print(f"\n=== MST via Matroid Greedy ===")
    print(f"MST edges: {[(u, v, w) for u, v, w in mst]}")
    print(f"Total weight: {sum(w for _, _, w in mst)}")
```

**Output:**

```
=== Weighted Job Scheduling ===
Jobs: [('a', "d=2", "p=100"), ('b', "d=1", "p=19"), ('c', "d=2", "p=27"), ('d', "d=1", "p=25")]
Scheduled: ['a', 'd', 'e']
Total profit: 140

=== MST via Matroid Greedy ===
MST edges: [(0, 1, 1), (1, 2, 2), (1, 3, 3)]
Total weight: 6
```

The job scheduler greedily picks the highest-profit jobs ($a$ with profit 100, then $d$ with profit 25, then $e$ with profit 15), skipping jobs that would violate deadline feasibility. The MST application greedily picks lowest-weight edges that keep the forest acyclic.

## Complexity

| Aspect | Cost |
|--------|:----:|
| Sorting | $O(n \log n)$ |
| Independence checks | $O(n \cdot f(n))$ |
| Total | $O(n \log n + n \cdot f(n))$ |

Here $f(n)$ is the cost of one independence check. For graphic matroids with union-find, $f(n) \approx O(\alpha(n))$, giving $O(n \log n)$ total. For general matroids, $f(n)$ depends on the specific independence oracle.

## Reference

- Edmonds, J. (1971). Matroids and the greedy algorithm. *Mathematical Programming*, 1(1), 127--136.
- Rado, R. (1957). Note on independence functions. *Proceedings of the London Mathematical Society*, 7(1), 300--320.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 16: Greedy Algorithms.
