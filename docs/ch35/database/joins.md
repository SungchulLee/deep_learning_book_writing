# Join Algorithms

Relational databases normalize data across multiple tables, so answering most queries requires **joining** tables by matching rows on a shared key. Think of it as solving a matching problem: given two lists, find all pairs that agree on some attribute. A brute-force comparison of every pair costs $O(|R| \cdot |S|)$, but smarter algorithms exploit sorting or hashing to bring the cost down to near-linear in the number of disk pages. The choice of join algorithm is one of the most consequential decisions a query optimizer makes.

This page covers the three fundamental join algorithms: nested-loop join, sort-merge join, and hash join.

## Setup and Notation

Consider joining two relations $R$ (outer) and $S$ (inner) on column $a$:

- $|R|$, $|S|$: number of tuples.
- $b_R$, $b_S$: number of disk pages occupied by each relation.
- $M$: number of available memory pages (buffer pool size).
- $B$: page size (tuples per page).

The **I/O cost** measures the number of disk page reads and writes, which dominates query execution time in practice.

## Nested-Loop Join

The simplest approach: for each tuple in $R$, scan all of $S$ looking for matches.

### Tuple-at-a-Time

```
for each tuple r in R:
    for each tuple s in S:
        if r.a == s.a:
            output (r, s)
```

**I/O cost:**

$$
b_R + |R| \cdot b_S
$$

Each $R$ tuple triggers a full scan of $S$. For $b_R = 1{,}000$ and $b_S = 500$ with $B = 100$ tuples per page, this is $1{,}000 + 100{,}000 \times 500 = 50{,}001{,}000$ I/Os.

### Block Nested-Loop Join

Read $R$ in blocks of $M - 2$ pages (reserving one page for $S$ input and one for output):

```
for each block B_r of R (M - 2 pages):
    for each page p_s of S:
        for each tuple r in B_r and s in p_s:
            if r.a == s.a:
                output (r, s)
```

**I/O cost:**

$$
b_R + \left\lceil \frac{b_R}{M - 2} \right\rceil \cdot b_S
$$

With enough memory to hold all of $R$ (i.e., $M \geq b_R + 2$), the cost drops to $b_R + b_S$ -- a single scan of each relation.

### Index Nested-Loop Join

When an index exists on $S.a$, replace the inner scan with an index lookup:

```
for each tuple r in R:
    use index on S.a to find matching tuples in S
    for each match s:
        output (r, s)
```

**I/O cost:**

$$
b_R + |R| \cdot c
$$

where $c$ is the cost of one index lookup (typically $O(\log |S|)$ for a B-tree or $O(1)$ for a hash index). This is often the fastest option when an index already exists.

## Sort-Merge Join

If both relations are sorted on the join column, matching rows can be found in a single linear scan.

### Algorithm

1. **Sort phase**: Sort $R$ and $S$ on column $a$ using external merge sort.
2. **Merge phase**: Scan both sorted relations simultaneously, advancing pointers to find matching keys.

**I/O cost (with external merge sort):**

$$
\underbrace{2 \, b_R \left\lceil \log_{M-1}\!\left\lceil \frac{b_R}{M} \right\rceil \right\rceil + b_R}_{\text{sort } R} \;+\; \underbrace{2 \, b_S \left\lceil \log_{M-1}\!\left\lceil \frac{b_S}{M} \right\rceil \right\rceil + b_S}_{\text{sort } S} \;+\; \underbrace{b_R + b_S}_{\text{merge}}
$$

Each sort pass reads and writes every page once (hence the factor of 2 per pass), and the final merge reads each page once more. For already-sorted inputs (common when an index exists), the cost is simply $b_R + b_S$.

!!! tip "When sort-merge wins"
    Sort-merge join is particularly effective when (1) both inputs are already sorted via an index, (2) the query requires sorted output (`ORDER BY`), or (3) the join involves inequality conditions ($<$, $\leq$) rather than equality.

## Hash Join

Hash join partitions both relations by the join key's hash value, then probes matching partitions.

### Algorithm

1. **Build (partition) phase**: Hash each tuple of both $R$ and $S$ into $h$ partitions using hash function $h_1$. Each partition is written to disk.
2. **Probe (match) phase**: For each partition $i$, load $R_i$ into an in-memory hash table (using a second hash function $h_2 \neq h_1$) and probe with each tuple from $S_i$.

**I/O cost:**

$$
3 \, (b_R + b_S)
$$

The factor of 3 comes from three I/O passes over the data:

1. **Read** both $R$ and $S$ during partitioning: $b_R + b_S$ reads.
2. **Write** all partitions to disk: $b_R + b_S$ writes.
3. **Read** all partitions back during probing: $b_R + b_S$ reads.

This assumes each partition of the build relation fits in memory.

### Memory Requirement

The build relation must partition into at most $M - 1$ buckets, each fitting in memory:

$$
h \leq M - 1 \quad \text{and} \quad \frac{b_R}{h} \leq M - 2
$$

This gives the requirement $b_R \leq (M - 1)(M - 2) \approx M^2$, meaning hash join works well when the smaller relation is at most $M^2$ pages.

!!! warning "Partition overflow"
    If a partition exceeds memory (e.g., due to skewed key distributions), **recursive partitioning** applies a different hash function to split the oversized partition further. This adds extra I/O passes for the affected partitions.

## Comparison

| Algorithm | I/O Cost | Memory Needed | Best When |
|-----------|----------|---------------|-----------|
| Block nested-loop | $b_R + \lceil b_R/(M{-}2) \rceil \cdot b_S$ | $M \geq 3$ | Small $R$ fits in memory |
| Index nested-loop | $b_R + |R| \cdot c$ | Minimal | Index exists on inner relation |
| Sort-merge | $O(b_R \log b_R + b_S \log b_S)$ | $O(\sqrt{b})$ | Inputs pre-sorted or sorted output needed |
| Hash join | $3(b_R + b_S)$ | $O(\sqrt{b_R})$ | Large unsorted inputs, equality joins |

??? example "Choosing the right algorithm"
    **Scenario 1**: Both tables have B-tree indexes on the join column, and the query includes `ORDER BY`. Use **sort-merge join** -- the inputs are already sorted, and the output order is free.

    **Scenario 2**: One table has 100 pages, the other has 1,000,000, and memory holds 200 pages. The small table fits entirely in memory, so **block nested-loop join** costs just $100 + 1{,}000{,}000 = 1{,}000{,}100$ I/Os.

    **Scenario 3**: Two large unsorted tables with an equality join and no useful indexes. **Hash join** at $3(b_R + b_S)$ is typically the fastest choice.

## Implementation

```python
"""
Join Algorithms -- nested-loop, sort-merge, and hash join demonstrations.

Compares the three fundamental join strategies on small in-memory
tables to illustrate their mechanics.
"""

# === Sample Data ==============================================================

R = [("a", 1), ("b", 2), ("c", 3), ("d", 4), ("a", 5)]
S = [("a", 10), ("c", 30), ("a", 20), ("e", 50)]


# === Nested-Loop Join =========================================================

def nested_loop_join(r_table, s_table, r_col=0, s_col=0):
    """Simple nested-loop join on specified columns."""
    result = []
    for r in r_table:
        for s in s_table:
            if r[r_col] == s[s_col]:
                result.append((r, s))
    return result


# === Sort-Merge Join ==========================================================

def sort_merge_join(r_table, s_table, r_col=0, s_col=0):
    """Sort-merge join on specified columns.

    Sorts both tables, then performs a linear merge to find all
    matching pairs, correctly handling duplicate keys.
    """
    r_sorted = sorted(r_table, key=lambda x: x[r_col])
    s_sorted = sorted(s_table, key=lambda x: x[s_col])

    result = []
    i, j = 0, 0
    while i < len(r_sorted) and j < len(s_sorted):
        if r_sorted[i][r_col] < s_sorted[j][s_col]:
            i += 1
        elif r_sorted[i][r_col] > s_sorted[j][s_col]:
            j += 1
        else:
            # Collect all matches for this key
            key = r_sorted[i][r_col]
            r_group = []
            while i < len(r_sorted) and r_sorted[i][r_col] == key:
                r_group.append(r_sorted[i])
                i += 1
            s_group = []
            while j < len(s_sorted) and s_sorted[j][s_col] == key:
                s_group.append(s_sorted[j])
                j += 1
            for r in r_group:
                for s in s_group:
                    result.append((r, s))
    return result


# === Hash Join ================================================================

def hash_join(r_table, s_table, r_col=0, s_col=0):
    """Hash join: build hash table on R, probe with S.

    Build phase indexes the smaller relation (R) by join key.
    Probe phase scans the larger relation (S) and looks up matches.
    """
    # Build phase
    hash_table: dict[str, list] = {}
    for r in r_table:
        key = r[r_col]
        hash_table.setdefault(key, []).append(r)

    # Probe phase
    result = []
    for s in s_table:
        key = s[s_col]
        if key in hash_table:
            for r in hash_table[key]:
                result.append((r, s))
    return result


# === Main =====================================================================

if __name__ == "__main__":
    print("R:", R)
    print("S:", S)
    print()

    for name, fn in [("Nested-Loop", nested_loop_join),
                     ("Sort-Merge", sort_merge_join),
                     ("Hash Join", hash_join)]:
        results = fn(R, S)
        print(f"{name} ({len(results)} matches):")
        for r, s in results:
            print(f"  {r} <-> {s}")
        print()
```

## Reference

- [Database System Concepts (Silberschatz, Korth, Sudarshan)](https://www.db-book.com/)
- [Designing Data-Intensive Applications (Kleppmann)](https://dataintensive.net/)
- Ramakrishnan, R. & Gehrke, J. *Database Management Systems*, Chapter 14
