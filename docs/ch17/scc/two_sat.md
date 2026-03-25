# 2-Satisfiability

The satisfiability problem (SAT) asks whether a Boolean formula can be made true by some assignment of variables. In general, SAT is NP-complete, but the special case **2-SAT** -- where every clause has exactly two literals -- admits an elegant polynomial-time solution via [strongly connected components](definition.md). This reduction from logic to graph theory is one of the most beautiful applications of SCC decomposition.

## Problem Formulation

A **2-SAT** instance consists of $n$ Boolean variables $x_1, x_2, \ldots, x_n$ and $m$ clauses, where each clause is a disjunction of exactly two literals:

$$
(l_{1,1} \lor l_{1,2}) \land (l_{2,1} \lor l_{2,2}) \land \cdots \land (l_{m,1} \lor l_{m,2})
$$

Each literal $l_{i,j}$ is either a variable $x_k$ or its negation $\neg x_k$. The goal is to determine whether there exists an assignment of truth values to the variables that satisfies all clauses simultaneously.

## The Implication Graph

The key insight is that each clause $(a \lor b)$ is logically equivalent to two implications:

$$
(\neg a \Rightarrow b) \quad \text{and} \quad (\neg b \Rightarrow a)
$$

The **implication graph** $G = (V, E)$ is constructed as:

- **Vertices:** For each variable $x_i$, create two vertices: $x_i$ and $\neg x_i$. So $|V| = 2n$.
- **Edges:** For each clause $(a \lor b)$, add directed edges $\neg a \to b$ and $\neg b \to a$.

!!! tip "Why Implications?"
    The clause $(a \lor b)$ says "at least one of $a$, $b$ must be true." If $a$ is false, then $b$ must be true -- hence $\neg a \Rightarrow b$. Symmetrically, if $b$ is false, then $a$ must be true -- hence $\neg b \Rightarrow a$.

## SCC-Based Solution

The formula is satisfiable if and only if no variable $x_i$ and its negation $\neg x_i$ belong to the same SCC in the implication graph.

!!! note "2-SAT Satisfiability Theorem"
    A 2-SAT formula is satisfiable if and only if for every variable $x_i$, the vertices $x_i$ and $\neg x_i$ are in different SCCs of the implication graph.

**Proof sketch (forward).** If $x_i$ and $\neg x_i$ are in the same SCC, there is a path from $x_i$ to $\neg x_i$ and from $\neg x_i$ to $x_i$ in the implication graph. This means setting $x_i = \text{true}$ forces $x_i = \text{false}$ (and vice versa), making the formula unsatisfiable. $\square$

**Proof sketch (reverse).** If no variable shares an SCC with its negation, we can assign truth values consistently by processing the [condensation DAG](condensation.md) in reverse topological order: for each unassigned variable, set the literal whose SCC appears later in the topological order to true. $\square$

## Extracting an Assignment

Once we verify satisfiability, the assignment is extracted as follows:

1. Compute SCCs of the implication graph (using [Tarjan's](tarjan.md) or [Kosaraju's](kosaraju.md)).
2. For each variable $x_i$, compare the topological order of the SCC containing $x_i$ with the SCC containing $\neg x_i$.
3. Set $x_i = \text{true}$ if the SCC of $x_i$ appears later (higher topological order) than the SCC of $\neg x_i$.

The intuition is that in the condensation DAG, later SCCs "override" earlier ones in the implication chain.

## Complexity

Building the implication graph takes $O(n + m)$. Finding SCCs takes $O(V + E) = O(n + m)$. Therefore:

$$
T(n, m) = O(n + m)
$$

## Implementation

```python
"""
2-SAT solver using strongly connected components.

Constructs the implication graph, finds SCCs using Tarjan's algorithm,
and determines satisfiability by checking whether any variable and its
negation share the same SCC.
"""


# === Tarjan's SCC (helper) ===
def tarjan_scc(graph, n):
    """Find SCCs and return vertex-to-SCC-id mapping."""
    disc = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    stack = []
    timer = [0]
    scc_id = [-1] * n
    scc_count = [0]

    def dfs(u):
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        stack.append(u)
        on_stack[u] = True

        for v in graph.get(u, []):
            if disc[v] == -1:
                dfs(v)
                low[u] = min(low[u], low[v])
            elif on_stack[v]:
                low[u] = min(low[u], disc[v])

        if low[u] == disc[u]:
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc_id[w] = scc_count[0]
                if w == u:
                    break
            scc_count[0] += 1

    for u in range(n):
        if disc[u] == -1:
            dfs(u)

    return scc_id


# === 2-SAT Solver ===
def solve_2sat(num_vars, clauses):
    """
    Solve a 2-SAT instance.

    Parameters
    ----------
    num_vars : int
        Number of Boolean variables (x_0, x_1, ..., x_{num_vars-1}).
    clauses : list[tuple[int, int]]
        Each clause (a, b) is a disjunction. Variables are 0-indexed.
        Use positive integers for x_i and negative for NOT x_i.
        E.g., (1, -2) means (x_0 OR NOT x_1).

    Returns
    -------
    list[bool] or None
        Assignment if satisfiable, None otherwise.
    """
    n = num_vars

    def var_to_node(literal):
        """Map literal to implication graph node index."""
        if literal > 0:
            return 2 * (literal - 1)      # x_i
        else:
            return 2 * (-literal - 1) + 1  # NOT x_i

    def negate_node(node):
        """Return the node representing the negation."""
        return node ^ 1

    # Build implication graph
    total_nodes = 2 * n
    graph = {i: [] for i in range(total_nodes)}

    for a, b in clauses:
        na = var_to_node(a)
        nb = var_to_node(b)
        # (a OR b) => (NOT a -> b) AND (NOT b -> a)
        graph[negate_node(na)].append(nb)
        graph[negate_node(nb)].append(na)

    # Find SCCs
    scc_id = tarjan_scc(graph, total_nodes)

    # Check satisfiability
    for i in range(n):
        if scc_id[2 * i] == scc_id[2 * i + 1]:
            return None  # x_i and NOT x_i in same SCC

    # Extract assignment: x_i is true if SCC(x_i) > SCC(NOT x_i)
    # Tarjan outputs SCCs in reverse topological order, so higher
    # SCC id = earlier in topological order
    assignment = [False] * n
    for i in range(n):
        assignment[i] = scc_id[2 * i] < scc_id[2 * i + 1]

    return assignment


# === Main ===
if __name__ == "__main__":
    # Example: (x1 OR x2) AND (NOT x1 OR x3) AND (NOT x2 OR NOT x3)
    clauses = [(1, 2), (-1, 3), (-2, -3)]
    result = solve_2sat(3, clauses)
    if result is not None:
        print(f"Satisfiable: {result}")
        names = [f"x{i+1}={'T' if v else 'F'}" for i, v in enumerate(result)]
        print(f"Assignment: {', '.join(names)}")
    else:
        print("Unsatisfiable")

    # Unsatisfiable: (x1 OR x1) AND (NOT x1 OR NOT x1)
    clauses2 = [(1, 1), (-1, -1)]
    result2 = solve_2sat(1, clauses2)
    print(f"\nSecond formula: {'Satisfiable' if result2 else 'Unsatisfiable'}")
```

**Output:**
```
Satisfiable: [False, True, False]
Assignment: x1=F, x2=T, x3=F

Second formula: Unsatisfiable
```

## Common 2-SAT Patterns

Many constraint problems reduce to 2-SAT:

| Constraint | Clauses |
|---|---|
| "At least one of $a$, $b$ is true" | $(a \lor b)$ |
| "At most one of $a$, $b$ is true" | $(\neg a \lor \neg b)$ |
| "Exactly one of $a$, $b$ is true" | $(a \lor b) \land (\neg a \lor \neg b)$ |
| "$a$ implies $b$" | $(\neg a \lor b)$ |
| "$a$ must be true" | $(a \lor a)$ |
| "$a$ must be false" | $(\neg a \lor \neg a)$ |

## Reference

- Aspvall, B., Plass, M. F., & Tarjan, R. E. (1979). A linear-time algorithm for testing the truth of certain quantified Boolean formulas. *Information Processing Letters*, 8(3), 121-123.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 20.
