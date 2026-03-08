# Mo's Algorithm

**Mo's algorithm** answers offline range queries in $O((n + q) \sqrt{n})$ time by processing queries in a clever order that minimizes pointer movement. It is particularly useful when maintaining a data structure over a sliding range.

## Key Idea

1. Sort all queries by $(\lfloor l / \sqrt{n} \rfloor, r)$ -- block of left endpoint first, then right endpoint.
2. Maintain a current range $[cur\_l, cur\_r]$ and expand/contract it one element at a time to reach each query range.
3. The total pointer movement across all queries is $O((n + q)\sqrt{n})$.

## Implementation

**Problem:** Given an array of integers, answer $q$ queries: how many distinct elements in $[l, r]$?

```python
import math
from collections import defaultdict

def mos_algorithm(arr, queries):
    n = len(arr)
    block = max(1, int(math.isqrt(n)))
    q = len(queries)

    # Sort queries by (block of l, r)
    indexed_queries = sorted(
        enumerate(queries),
        key=lambda x: (x[1][0] // block,
                       x[1][1] if (x[1][0] // block) % 2 == 0 else -x[1][1])
    )

    freq = defaultdict(int)
    distinct = 0
    cur_l, cur_r = 0, -1
    answers = [0] * q

    def add(idx):
        nonlocal distinct
        freq[arr[idx]] += 1
        if freq[arr[idx]] == 1:
            distinct += 1

    def remove(idx):
        nonlocal distinct
        freq[arr[idx]] -= 1
        if freq[arr[idx]] == 0:
            distinct -= 1

    for qi, (l, r) in indexed_queries:
        while cur_l > l:
            cur_l -= 1
            add(cur_l)
        while cur_r < r:
            cur_r += 1
            add(cur_r)
        while cur_l < l:
            remove(cur_l)
            cur_l += 1
        while cur_r > r:
            remove(cur_r)
            cur_r -= 1
        answers[qi] = distinct

    return answers

arr = [1, 2, 1, 3, 2, 1, 4]
queries = [(0, 3), (1, 5), (2, 6), (0, 6)]
print(mos_algorithm(arr, queries))
# Output: [3, 3, 4, 4]
```

## Complexity

| Component | Complexity |
|---|---|
| Sorting queries | $O(q \log q)$ |
| Right pointer movement | $O(n \sqrt{n})$ |
| Left pointer movement | $O(n \sqrt{n})$ |
| Total | $O((n + q)\sqrt{n})$ |

## When to Use

- Queries are **offline** (all known in advance).
- Adding/removing a single element from the range is $O(1)$ or $O(\log n)$.
- No efficient online data structure exists for the specific operation.

# Reference

- [Mo's Algorithm -- CP-Algorithms](https://cp-algorithms.com/data_structures/sqrt_decomposition.html)
- Halim, S. & Halim, F. *Competitive Programming 4*, 2020.
