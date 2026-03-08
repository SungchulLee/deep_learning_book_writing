# Square Root Decomposition

**Square root decomposition** divides an array of $n$ elements into blocks of size $\sqrt{n}$, enabling $O(\sqrt{n})$ queries and updates -- a useful middle ground between brute force $O(n)$ and complex tree structures $O(\log n)$.

## Key Idea

Divide the array into $\lceil n / B \rceil$ blocks of size $B = \lfloor \sqrt{n} \rfloor$. Precompute an aggregate (sum, min, etc.) for each block.

- **Query** $[l, r]$: Process partial blocks at the edges ($O(B)$ each) and complete blocks in the middle ($O(n/B)$).
- **Update** index $i$: Recompute the block containing $i$ ($O(1)$ or $O(B)$).

Total query time: $O(B + n/B)$, minimized at $B = \sqrt{n}$ giving $O(\sqrt{n})$.

## Implementation: Range Sum with Point Updates

```python
import math

class SqrtDecomposition:
    def __init__(self, arr):
        self.arr = arr[:]
        self.n = len(arr)
        self.block_size = max(1, int(math.isqrt(self.n)))
        self.num_blocks = (self.n + self.block_size - 1) // self.block_size
        self.blocks = [0] * self.num_blocks
        for i in range(self.n):
            self.blocks[i // self.block_size] += arr[i]

    def update(self, i, val):
        self.blocks[i // self.block_size] += val - self.arr[i]
        self.arr[i] = val

    def query(self, l, r):
        total = 0
        bl = l // self.block_size
        br = r // self.block_size
        if bl == br:
            for i in range(l, r + 1):
                total += self.arr[i]
        else:
            for i in range(l, (bl + 1) * self.block_size):
                total += self.arr[i]
            for b in range(bl + 1, br):
                total += self.blocks[b]
            for i in range(br * self.block_size, r + 1):
                total += self.arr[i]
        return total

arr = [1, 3, 5, 2, 7, 6, 3, 1, 4, 8]
sd = SqrtDecomposition(arr)
print(sd.query(1, 6))   # Output: 26
sd.update(3, 10)
print(sd.query(1, 6))   # Output: 34
```

## Complexity

| Operation | Time |
|---|---|
| Build | $O(n)$ |
| Point update | $O(1)$ |
| Range query | $O(\sqrt{n})$ |

## When to Use

- When $O(\log n)$ structures (segment tree, BIT) are too complex to implement.
- When the problem involves operations not easily handled by trees.
- As a building block for Mo's algorithm.

# Reference

- [Sqrt Decomposition -- CP-Algorithms](https://cp-algorithms.com/data_structures/sqrt_decomposition.html)
- Halim, S. & Halim, F. *Competitive Programming 4*, 2020.
