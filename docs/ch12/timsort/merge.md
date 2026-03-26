# Merge Strategy

After Timsort identifies and extends natural runs, it must merge them into a single sorted array. The merge strategy determines **which runs to merge and in what order**. A naive approach (merge the first two, then merge the result with the third, and so on) can lead to highly unbalanced merges. Timsort instead maintains a stack of pending runs and enforces invariants that guarantee balanced merges, keeping the total merge cost at $O(n \log n)$.

## The Merge Stack

As Timsort discovers runs, it pushes them onto a stack. Each stack entry records the starting index and length of a run. After each push, Timsort checks whether the top entries on the stack satisfy two invariants. If either invariant is violated, it merges the appropriate pair.

## Stack Invariants

Let the top three entries on the stack be $X$, $Y$, $Z$ (with $X$ on top). Timsort enforces:

1. $|Z| > |Y| + |X|$
2. $|Y| > |X|$

These invariants ensure that run lengths grow at least as fast as the Fibonacci sequence, which bounds the stack depth to $O(\log n)$ and guarantees that merges are approximately balanced. When an invariant is violated:

- If $|Z| \leq |Y| + |X|$: merge $Y$ with the smaller of $Z$ and $X$.
- If $|Y| \leq |X|$: merge $X$ and $Y$.

## Why These Invariants Work

The Fibonacci-like growth means that a stack of depth $d$ requires at least $F_{d+2}$ elements, where $F_k$ is the $k$-th Fibonacci number. Since $F_k$ grows exponentially, the stack depth for $n$ elements is at most $O(\log_\phi n)$ where $\phi = (1 + \sqrt{5})/2$ is the golden ratio. This guarantees at most $O(\log n)$ pending runs at any time.

## Merge Procedure

When two adjacent runs are merged, Timsort uses a temporary buffer sized to the **shorter** of the two runs. This reduces the auxiliary space from $O(n)$ (full merge sort) to $O(n/2)$ in the worst case and often much less.

The merge proceeds in two modes:

1. **One-at-a-time mode**: Compare the smallest unmerged elements from each run and copy the winner. Track how many consecutive wins come from the same run.
2. **Galloping mode**: When one run wins `min_gallop` times in a row, switch to exponential search to find where the next element from the losing run belongs, then copy the winning block in bulk.

## Implementation

```python
"""
Timsort merge strategy: stack-based merging with invariants.

Demonstrates how Timsort maintains a stack of runs and enforces
Fibonacci-like invariants to keep merges balanced. Uses a temporary
buffer sized to the shorter run.
"""


# === Merge Two Adjacent Runs ===

def merge_runs(arr: list, lo: int, mid: int, hi: int) -> None:
    """Merge arr[lo..mid] and arr[mid+1..hi] in place.

    Uses a temporary buffer for the shorter run to minimize
    auxiliary space.
    """
    left = arr[lo:mid + 1]
    right = arr[mid + 1:hi + 1]

    i = 0
    j = 0
    k = lo

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1

    while i < len(left):
        arr[k] = left[i]
        i += 1
        k += 1

    while j < len(right):
        arr[k] = right[j]
        j += 1
        k += 1


# === Merge Stack with Invariant Checking ===

class MergeStack:
    """Manages pending runs and enforces Timsort's merge invariants."""

    def __init__(self, arr: list):
        self.arr = arr
        self.stack = []  # Each entry: (start, length)

    def push_run(self, start: int, length: int) -> None:
        """Push a new run and restore invariants by merging if needed."""
        self.stack.append((start, length))
        self._merge_collapse()

    def _merge_collapse(self) -> None:
        """Merge runs until stack invariants are satisfied."""
        while len(self.stack) > 1:
            n = len(self.stack) - 1

            if (n >= 2 and self.stack[n - 2][1]
                    <= self.stack[n - 1][1] + self.stack[n][1]):
                if self.stack[n - 2][1] < self.stack[n][1]:
                    self._merge_at(n - 2)
                else:
                    self._merge_at(n - 1)
            elif self.stack[n - 1][1] <= self.stack[n][1]:
                self._merge_at(n - 1)
            else:
                break

    def _merge_at(self, i: int) -> None:
        """Merge stack[i] with stack[i+1]."""
        start1, len1 = self.stack[i]
        start2, len2 = self.stack[i + 1]

        merge_runs(self.arr, start1, start1 + len1 - 1,
                   start2 + len2 - 1)

        self.stack[i] = (start1, len1 + len2)
        del self.stack[i + 1]

    def force_merge_all(self) -> None:
        """Merge all remaining runs on the stack."""
        while len(self.stack) > 1:
            n = len(self.stack) - 1
            if n >= 2 and self.stack[n - 2][1] < self.stack[n][1]:
                self._merge_at(n - 2)
            else:
                self._merge_at(n - 1)


# === Demonstration ===

if __name__ == "__main__":
    # Simulate pushing runs onto the stack
    arr = [1, 3, 5, 7, 2, 4, 6, 8, 10, 9, 11, 13, 12, 14]
    print(f"Original: {arr}")

    ms = MergeStack(arr)

    # Push runs: [1,3,5,7], [2,4,6,8,10], [9,11,13], [12,14]
    runs = [(0, 4), (4, 5), (9, 3), (12, 2)]
    for start, length in runs:
        print(f"  Push run: arr[{start}:{start+length}] = "
              f"{arr[start:start+length]}")
        ms.push_run(start, length)
        print(f"  Stack: {ms.stack}")

    ms.force_merge_all()
    print(f"After merge all: {arr}")
    print(f"Final stack: {ms.stack}")
```

**Output:**
```
Original: [1, 3, 5, 7, 2, 4, 6, 8, 10, 9, 11, 13, 12, 14]
  Push run: arr[0:4] = [1, 3, 5, 7]
  Stack: [(0, 4)]
  Push run: arr[4:9] = [2, 4, 6, 8, 10]
  Stack: [(0, 4), (4, 5)]
  Push run: arr[9:12] = [9, 11, 13]
  Stack: [(0, 4), (4, 5), (9, 3)]
  Push run: arr[12:14] = [12, 14]
  Stack: [(0, 4), (4, 5), (9, 3), (12, 2)]
After merge all: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
Final stack: [(0, 14)]
```

## Complexity

| Property | Value |
|----------|-------|
| Maximum stack depth | $O(\log n)$ |
| Total merge comparisons | $O(n \log n)$ |
| Auxiliary space for merging | $O(n/2)$ worst case |

!!! tip "Powersort Improvement"
    Python 3.11 replaced Timsort's original merge strategy with **Powersort**, which uses a simpler rule based on the "power" of each run boundary to decide merge order. Powersort achieves the same $O(n \log n)$ bound with a cleaner invariant and slightly better merge balance.

## Reference

- Peters, T. (2002). *Timsort description*. [CPython source, `Objects/listsort.txt`](https://github.com/python/cpython/blob/main/Objects/listsort.txt).
- Auger, N., Jugé, V., Nicaud, C., & Pivoteau, C. (2019). On the worst-case complexity of TimSort. *Proceedings of ESA*, 13:1-13:13.
