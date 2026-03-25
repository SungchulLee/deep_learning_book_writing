# Query Optimization

A SQL query can be executed in many equivalent ways, each with vastly different performance. **Query optimization** is the process by which a database management system (DBMS) transforms a declarative SQL statement into an efficient execution plan. The optimizer explores alternative plans, estimates their costs using statistics about the data, and selects the cheapest one. Understanding query optimization helps both database designers and application developers write queries that the optimizer can handle well.

## From SQL to Execution Plan

A query goes through several stages before execution:

1. **Parsing**: SQL text is parsed into a syntax tree.
2. **Logical plan**: The syntax tree is converted into a relational algebra expression.
3. **Optimization**: Equivalent logical plans are explored and the cheapest physical plan is selected.
4. **Execution**: The chosen physical plan runs against the storage engine.

The optimizer's job is stage 3: finding the plan with the lowest estimated cost.

## Relational Algebra Equivalences

The optimizer rewrites logical plans using algebraic equivalences. Key rules include:

**Selection pushdown** -- apply filters as early as possible to reduce intermediate result sizes:

$$
\sigma_{\theta}(R \bowtie S) \equiv \sigma_{\theta}(R) \bowtie S
$$

when $\theta$ involves only attributes of $R$.

**Projection pushdown** -- discard unneeded columns early:

$$
\pi_L(R \bowtie S) \equiv \pi_L(\pi_{L_1}(R) \bowtie \pi_{L_2}(S))
$$

where $L_1$ and $L_2$ include the join columns plus the columns in $L$.

**Join commutativity and associativity**:

$$
R \bowtie S \equiv S \bowtie R
$$

$$
(R \bowtie S) \bowtie T \equiv R \bowtie (S \bowtie T)
$$

These allow the optimizer to reorder joins freely, which is critical because join order dominates plan cost.

## Cost Estimation

The optimizer assigns a cost to each candidate plan based on:

- **I/O cost**: Number of disk page reads and writes (often the dominant factor).
- **CPU cost**: Comparison and hashing operations.
- **Network cost**: Data transfer in distributed databases.

### Selectivity

The **selectivity** of a predicate is the fraction of tuples that satisfy it. For a predicate $\sigma_{A=v}(R)$:

$$
\text{sel}(A = v) = \frac{1}{V(A, R)}
$$

where $V(A, R)$ is the number of distinct values of attribute $A$ in relation $R$ (assuming uniform distribution).

For a range predicate $\sigma_{A \leq v}(R)$:

$$
\text{sel}(A \leq v) = \frac{v - \min(A)}{\max(A) - \min(A)}
$$

### Cardinality Estimation

The estimated output size of a join is:

$$
|R \bowtie_A S| = \frac{|R| \cdot |S|}{\max(V(A, R),\; V(A, S))}
$$

Accurate cardinality estimates are crucial: an error can propagate through multiple joins, causing the optimizer to choose a plan that is orders of magnitude slower than optimal.

!!! warning "Estimation errors"
    Real data rarely follows the uniform-distribution assumption. Histograms, sampling, and sketches improve estimates, but cardinality estimation remains one of the hardest problems in query optimization.

## Join Order Optimization

For a query joining $n$ tables, the number of possible join orderings is:

$$
\frac{(2(n-1))!}{(n-1)!}
$$

which is the $n$-th Catalan number and grows super-exponentially. Optimizers use two main strategies:

**Dynamic programming (System R style)**: Build optimal plans bottom-up. For each subset of tables, find the cheapest way to join them by considering all ways to split the subset into two parts. Time complexity is $O(3^n)$ for $n$ tables -- feasible for $n \leq 15$ or so.

**Greedy / heuristic**: Start with the two-table join of lowest cost, then repeatedly add the next cheapest table. Fast but may miss the global optimum.

??? example "DP join enumeration for 4 tables"
    Given tables $A, B, C, D$, the optimizer considers:

    - All 2-table joins: $A \bowtie B$, $A \bowtie C$, ..., $C \bowtie D$ (6 pairs)
    - All 3-table plans built from the best 2-table plans
    - The final 4-table plan built from the best 3-table plans

    Each subproblem is solved once and cached, avoiding redundant computation.

## Physical Plan Selection

Once the logical plan and join order are fixed, the optimizer selects **physical operators**:

| Logical Operator | Physical Options |
|-----------------|-----------------|
| Selection ($\sigma$) | Sequential scan, index scan, bitmap scan |
| Join ($\bowtie$) | Nested-loop, sort-merge, hash join |
| Sort | External merge sort, in-memory quicksort |
| Aggregation | Hash aggregation, sort-based aggregation |

The optimizer combines I/O cost models for each operator to estimate total plan cost.

## Indexes and Query Performance

Indexes dramatically affect query plans:

- **B-tree indexes** support equality and range queries: $O(\log n)$ lookup.
- **Hash indexes** support equality only: $O(1)$ expected lookup.
- **Covering indexes** include all columns needed by the query, avoiding table access entirely.

The optimizer decides whether to use an index based on selectivity. A rule of thumb: an index scan is worthwhile when selectivity is below 10-15%; for less selective predicates, a sequential scan is faster due to sequential I/O.

## Implementation

```python
"""
Query Optimization -- selectivity estimation and join order enumeration.

Demonstrates cardinality estimation and dynamic-programming-based
join order optimization for small numbers of tables.
"""

from itertools import combinations


# === Selectivity Estimation ===================================================

def selectivity_equality(n_distinct: int) -> float:
    """Estimate selectivity of an equality predicate under uniform assumption.

    Args:
        n_distinct: Number of distinct values for the attribute.

    Returns:
        Estimated fraction of tuples satisfying A = v.
    """
    if n_distinct <= 0:
        return 1.0
    return 1.0 / n_distinct


def estimate_join_cardinality(card_r: int, card_s: int,
                               distinct_r: int, distinct_s: int) -> float:
    """Estimate output cardinality of an equi-join.

    Uses the formula |R join S| = |R| * |S| / max(V(A,R), V(A,S)).
    """
    max_distinct = max(distinct_r, distinct_s)
    if max_distinct == 0:
        return 0.0
    return (card_r * card_s) / max_distinct


# === Join Order Optimization (DP) =============================================

def dp_join_order(tables: dict[str, int],
                  join_costs: dict[tuple[str, str], float]) -> tuple[float, list]:
    """Find optimal join order using dynamic programming.

    Args:
        tables: Mapping from table name to cardinality.
        join_costs: Mapping from (table_i, table_j) to join cost.
                    Missing pairs are assumed not directly joinable.

    Returns:
        Tuple of (minimum total cost, join sequence).
    """
    table_names = sorted(tables.keys())
    n = len(table_names)
    name_to_idx = {name: i for i, name in enumerate(table_names)}

    # dp[frozenset] = (cost, cardinality, join_sequence)
    dp: dict[frozenset, tuple[float, int, list]] = {}

    # Base case: single tables
    for name in table_names:
        dp[frozenset([name])] = (0, tables[name], [name])

    # Enumerate subsets of increasing size
    for size in range(2, n + 1):
        for subset in combinations(table_names, size):
            fs = frozenset(subset)
            best = (float("inf"), 0, [])

            # Try all ways to split subset into two non-empty parts
            for split_size in range(1, size):
                for left in combinations(subset, split_size):
                    left_set = frozenset(left)
                    right_set = fs - left_set
                    if left_set not in dp or right_set not in dp:
                        continue

                    l_cost, l_card, l_seq = dp[left_set]
                    r_cost, r_card, r_seq = dp[right_set]

                    # Check if there is a join edge between the two sides
                    join_cost = 0
                    for lt in left_set:
                        for rt in right_set:
                            key = (min(lt, rt), max(lt, rt))
                            if key in join_costs:
                                join_cost = join_costs[key]
                                break
                        if join_cost > 0:
                            break

                    if join_cost == 0:
                        continue

                    total = l_cost + r_cost + join_cost
                    if total < best[0]:
                        out_card = l_card + r_card  # simplified
                        best = (total, out_card, l_seq + r_seq)

            if best[0] < float("inf"):
                dp[fs] = best

    full_set = frozenset(table_names)
    if full_set in dp:
        cost, _, sequence = dp[full_set]
        return cost, sequence
    return float("inf"), []


# === Main =====================================================================

if __name__ == "__main__":
    # Selectivity examples
    print("=== Selectivity Estimation ===")
    print(f"Equality (100 distinct): {selectivity_equality(100):.4f}")
    print(f"Join cardinality (1000 x 500, 50 vs 80 distinct): "
          f"{estimate_join_cardinality(1000, 500, 50, 80):.0f}")
    print()

    # Join order optimization
    print("=== Join Order Optimization (DP) ===")
    tables = {"A": 1000, "B": 5000, "C": 200, "D": 3000}
    costs = {
        ("A", "B"): 150,
        ("A", "C"): 30,
        ("B", "D"): 200,
        ("C", "D"): 80,
    }
    best_cost, order = dp_join_order(tables, costs)
    print(f"Tables: {tables}")
    print(f"Best join order: {' -> '.join(order)}")
    print(f"Estimated cost: {best_cost}")
```

## Reference

- Selinger, P. G. et al. "Access Path Selection in a Relational Database Management System." *SIGMOD*, 1979
- [Database System Concepts (Silberschatz, Korth, Sudarshan)](https://www.db-book.com/)
- [Designing Data-Intensive Applications (Kleppmann)](https://dataintensive.net/)
