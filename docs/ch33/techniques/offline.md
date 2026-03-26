# Offline Processing

Many competitive programming problems provide all queries upfront before any answers are required. **Offline processing** exploits this by reordering or batching queries to achieve better time complexity than answering each query independently. Instead of maintaining complex online data structures, offline techniques sort queries cleverly, process them in a favorable order, and output answers in the original query order.

## When to Use Offline Processing

Offline processing applies when:

- All queries are known in advance (no query depends on a previous answer).
- Processing queries in a different order reduces redundant computation.
- The problem has no updates interleaved with queries, or updates can be handled in batches.

The key tradeoff: offline algorithms cannot answer queries incrementally, but they often achieve significantly better asymptotic complexity or simpler implementations.

## Mo's Algorithm

Mo's algorithm answers range queries $[l, r]$ on a static array in $O((n + q) \sqrt{n})$ time by processing queries in a carefully chosen order.

### Idea

Maintain a "current window" $[l_{\text{cur}}, r_{\text{cur}}]$ with a running answer. To answer a new query $[l, r]$, extend or shrink the window by adding or removing elements one at a time. The total movement cost depends on the query order.

### Query Ordering

1. Divide indices into $\sqrt{n}$ blocks of size $\sqrt{n}$.
2. Sort queries by: primary key = block number of $l$, secondary key = $r$ (alternating direction for even/odd blocks).

This ensures:
- Within a block, $r$ moves at most $O(n)$ total.
- Between blocks, $l$ moves at most $O(\sqrt{n})$ per query.
- Total movement: $O((n + q)\sqrt{n})$.

### Complexity

$$
T(n, q) = O\!\left((n + q)\sqrt{n}\right)
$$

```python
"""
Mo's algorithm for offline range queries.

Processes range queries on a static array by reordering queries
to minimize the total pointer movement, achieving O((n+q)*sqrt(n)).
"""

import math

# ===================================================================
# Mo's Algorithm
# ===================================================================

def mos_algorithm(arr, queries):
    """Answer range sum queries using Mo's algorithm.

    Args:
        arr: input array of integers
        queries: list of (left, right) pairs (0-indexed, inclusive)

    Returns:
        List of answers in original query order
    """
    n = len(arr)
    block_size = max(1, int(math.sqrt(n)))

    # Attach original index and sort by Mo's order
    indexed_queries = [(l, r, i) for i, (l, r) in enumerate(queries)]
    indexed_queries.sort(key=lambda q: (q[0] // block_size,
                                        q[1] if (q[0] // block_size) % 2 == 0
                                        else -q[1]))

    answers = [0] * len(queries)
    cur_l, cur_r = 0, -1
    cur_sum = 0

    for l, r, idx in indexed_queries:
        # Expand or shrink the window
        while cur_r < r:
            cur_r += 1
            cur_sum += arr[cur_r]
        while cur_l > l:
            cur_l -= 1
            cur_sum += arr[cur_l]
        while cur_r > r:
            cur_sum -= arr[cur_r]
            cur_r -= 1
        while cur_l < l:
            cur_sum -= arr[cur_l]
            cur_l += 1

        answers[idx] = cur_sum

    return answers

# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    arr = [1, 3, 5, 2, 7, 6, 3, 8, 4, 2]
    queries = [(0, 3), (1, 5), (4, 8), (0, 9), (2, 6)]

    answers = mos_algorithm(arr, queries)

    print("Array:", arr)
    print("\nRange sum queries:")
    for (l, r), ans in zip(queries, answers):
        expected = sum(arr[l:r+1])
        print(f"  sum[{l}..{r}] = {ans}  (check: {expected})")
```

**Output:**
```
Array: [1, 3, 5, 2, 7, 6, 3, 8, 4, 2]

Range sum queries:
  sum[0..3] = 11  (check: 11)
  sum[1..5] = 23  (check: 23)
  sum[4..8] = 28  (check: 28)
  sum[0..9] = 41  (check: 41)
  sum[2..6] = 23  (check: 23)
```

## Offline LCA (Tarjan's Algorithm)

When all LCA queries are known in advance, Tarjan's offline algorithm answers all $q$ queries during a single DFS traversal using Union-Find.

### Algorithm

1. Perform a DFS from the root.
2. When node $u$ finishes (all children processed), union $u$ with its parent.
3. For each query $(u, v)$ attached to $u$: if $v$ is already visited, then $\text{LCA}(u, v) = \text{find}(v)$.

### Complexity

$$
T(n, q) = O\!\left((n + q) \cdot \alpha(n)\right)
$$

where $\alpha$ is the inverse Ackermann function. This is nearly linear.

## Other Offline Techniques

| Technique | Idea | Complexity |
|---|---|---|
| Mo's algorithm | Reorder range queries by blocks | $O((n+q)\sqrt{n})$ |
| Tarjan's offline LCA | DFS + Union-Find for LCA queries | $O((n+q)\alpha(n))$ |
| CDQ divide and conquer | Split queries by time, solve recursively | Problem-dependent |
| Offline to online (retroactive) | Process updates in reverse order | Problem-dependent |

## When Not to Use Offline

Offline processing is not applicable when:

- Queries arrive one at a time and must be answered immediately.
- A query's parameters depend on previous answers (adaptive queries).
- The problem requires real-time response guarantees.

In these cases, online data structures (segment trees, balanced BSTs, etc.) are necessary.

## Reference

- Mo, H. "Mo's algorithm." *Competitive programming folklore*.
- Tarjan, R. E. (1979). "Applications of path compression on balanced trees." *JACM*, 26(4), 690--715.
