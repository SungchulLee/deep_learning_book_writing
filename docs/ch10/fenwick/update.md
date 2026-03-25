# Point Updates

When an element in the underlying array changes, the Binary Indexed Tree must update every tree node whose stored range includes that element. The **point update** operation adds a value $\delta$ to position $i$, propagating the change upward through the BIT in $O(\log n)$ time. This page details how the update traversal works and why it is correct.

## Update Algorithm

To add $\delta$ to position $i$, the algorithm walks up the implicit BIT by repeatedly adding the lowest set bit:

1. Add $\delta$ to `tree[i]`.
2. Move to the next ancestor: $i \leftarrow i + \text{lowbit}(i)$, where $\text{lowbit}(i) = i \;\&\; (-i)$.
3. Repeat until $i > n$.

The key insight is that **adding** the lowest set bit moves to the nearest tree node whose range strictly contains the current range. This is the mirror operation of the prefix query, which **subtracts** the lowest set bit to move to the adjacent non-overlapping block.

## Why Adding the Lowest Set Bit Moves to the Parent

Consider index $i$ in binary. The BIT node at $i$ covers a range of $\text{lowbit}(i)$ elements. When we compute $i + \text{lowbit}(i)$, we "carry" the lowest set bit to the next higher position. The resulting index covers a strictly larger range that includes the range of $i$.

!!! example "Update Traversal for i = 3"
    | Step | $i$ (binary) | $\text{lowbit}(i)$ | Next $i$ |
    |:----:|:---:|:---:|:---:|
    | 1 | 3 = `011` | 1 | 3 + 1 = 4 |
    | 2 | 4 = `100` | 4 | 4 + 4 = 8 |
    | 3 | 8 = `1000` | 8 | 8 + 8 = 16 > $n$ |

    So updating position 3 in a BIT of size 8 touches nodes 3, 4, and 8. Node 3 covers $[3,3]$, node 4 covers $[1,4]$, and node 8 covers $[1,8]$ — all ranges that include position 3.

## Step-by-Step Trace

Starting from the array $a = [0, 0, 0, 0, 0, 0, 0, 0]$ (all zeros), let us add 5 to position 3.

**Before update:** all `tree[i] = 0`.

**Update(3, 5):**

| Step | $i$ | `tree[i]` before | `tree[i]` after |
|:----:|:---:|:---:|:---:|
| 1 | 3 | 0 | 5 |
| 2 | 4 | 0 | 5 |
| 3 | 8 | 0 | 5 |

After the update, a prefix query for any $i \geq 3$ will include the value 5, while queries for $i < 3$ will not — exactly reflecting the addition of 5 at position 3.

## Correctness Argument

The update is correct if and only if every node whose range includes position $i$ gets modified. The set of such nodes is exactly the **ancestor chain** of $i$ in the implicit BIT. The iteration $i \leftarrow i + \text{lowbit}(i)$ visits every ancestor because:

1. Each ancestor covers a strictly larger range containing position $i$.
2. The lowest set bit grows at each step (the bit position increases by at least one).
3. Eventually $i$ exceeds $n$, terminating the loop after at most $\lfloor \log_2 n \rfloor$ steps.

## Implementation

```python
"""
Point updates in a Binary Indexed Tree.

Demonstrates the update traversal that propagates a change
from a leaf position up through all ancestor nodes, with
a detailed trace of which nodes are visited.
"""


# === Fenwick Tree with Traced Update ===

class FenwickTree:
    """BIT with optional traced updates for educational purposes."""

    def __init__(self, n: int):
        self.n = n
        self.tree = [0] * (n + 1)

    def update(self, i: int, delta: int) -> None:
        """Add delta to position i, propagating to ancestors."""
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)

    def update_traced(self, i: int, delta: int) -> None:
        """Update with printed trace showing ancestor traversal."""
        step = 1
        print(f"  Update position {i} by {delta}:")
        while i <= self.n:
            self.tree[i] += delta
            print(f"    Step {step}: tree[{i}] updated "
                  f"(lowbit={i & (-i)}, next i={i + (i & (-i))})")
            i += i & (-i)
            step += 1

    def query(self, i: int) -> int:
        """Return prefix sum from 1 to i."""
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s


# === Demonstration ===

if __name__ == "__main__":
    n = 8
    ft = FenwickTree(n)

    # Traced updates
    ft.update_traced(3, 5)
    print()
    ft.update_traced(5, 7)
    print()

    # Verify with prefix queries
    print("Prefix sums after updates:")
    for i in range(1, n + 1):
        print(f"  prefix({i}) = {ft.query(i)}")

    # Update and observe changes
    print()
    ft.update_traced(3, 10)
    print()
    print("Prefix sums after adding 10 more to position 3:")
    for i in range(1, n + 1):
        print(f"  prefix({i}) = {ft.query(i)}")
```

**Output:**
```
  Update position 3 by 5:
    Step 1: tree[3] updated (lowbit=1, next i=4)
    Step 2: tree[4] updated (lowbit=4, next i=8)
    Step 3: tree[8] updated (lowbit=8, next i=16)

  Update position 5 by 7:
    Step 1: tree[5] updated (lowbit=1, next i=6)
    Step 2: tree[6] updated (lowbit=2, next i=8)
    Step 3: tree[8] updated (lowbit=8, next i=16)

Prefix sums after updates:
  prefix(1) = 0
  prefix(2) = 0
  prefix(3) = 5
  prefix(4) = 5
  prefix(5) = 12
  prefix(6) = 12
  prefix(7) = 12
  prefix(8) = 12

  Update position 3 by 10:
    Step 1: tree[3] updated (lowbit=1, next i=4)
    Step 2: tree[4] updated (lowbit=4, next i=8)
    Step 3: tree[8] updated (lowbit=8, next i=16)

Prefix sums after adding 10 more to position 3:
  prefix(1) = 0
  prefix(2) = 0
  prefix(3) = 15
  prefix(4) = 15
  prefix(5) = 22
  prefix(6) = 22
  prefix(7) = 22
  prefix(8) = 22
```

## Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Point update | $O(\log n)$ | $O(1)$ |

The update visits at most $\lfloor \log_2 n \rfloor$ nodes, each requiring $O(1)$ work (one addition and one bit operation).

!!! tip "Update vs Set"
    The BIT update operation adds a **delta** (difference) to a position. To **set** position $i$ to a new value $v$, first compute the current value as $a[i] = \text{prefix}(i) - \text{prefix}(i-1)$, then call `update(i, v - a[i])`. Alternatively, maintain a separate array to store current values for $O(1)$ lookup.

## Reference

- Fenwick, P. M. (1994). A New Data Structure for Cumulative Frequency Tables. *Software: Practice and Experience*, 24(3), 327-336.
