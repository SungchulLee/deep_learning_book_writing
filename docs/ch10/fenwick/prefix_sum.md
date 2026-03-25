# Prefix Sum Queries

Given an array $a[1..n]$ that may receive point updates, a **prefix sum query** asks for the cumulative sum $\text{prefix}(i) = \sum_{j=1}^{i} a[j]$. In a Binary Indexed Tree (BIT), this query runs in $O(\log n)$ time by exploiting the binary representation of the index $i$. This page traces the prefix query algorithm in detail and shows why it visits at most $\lfloor \log_2 n \rfloor$ nodes.

## Algorithm

Recall that each BIT entry `tree[i]` stores the sum of a contiguous block of $\text{lowbit}(i) = i \;\&\; (-i)$ elements ending at position $i$. A prefix query collects these non-overlapping blocks to cover positions $1$ through $i$.

The procedure is:

1. Initialize an accumulator $s = 0$.
2. Add `tree[i]` to $s$.
3. Remove the lowest set bit: $i \leftarrow i - \text{lowbit}(i)$.
4. Repeat until $i = 0$.

Each step strips one bit from $i$, so the loop runs at most $\lfloor \log_2 n \rfloor$ times.

## Step-by-Step Trace

Consider the array $a = [1, 3, 5, 7, 9, 2, 4, 6]$ with $n = 8$. The BIT stores:

| Index $i$ | Binary | $\text{lowbit}(i)$ | Range covered | `tree[i]` |
|:---------:|:------:|:------------------:|:-------------:|:---------:|
| 1 | `001` | 1 | $[1,1]$ | 1 |
| 2 | `010` | 2 | $[1,2]$ | 4 |
| 3 | `011` | 1 | $[3,3]$ | 5 |
| 4 | `100` | 4 | $[1,4]$ | 16 |
| 5 | `101` | 1 | $[5,5]$ | 9 |
| 6 | `110` | 2 | $[5,6]$ | 11 |
| 7 | `111` | 1 | $[7,7]$ | 4 |
| 8 | `1000` | 8 | $[1,8]$ | 37 |

**Query: prefix(7)**

We want $a[1] + a[2] + \cdots + a[7] = 1+3+5+7+9+2+4 = 31$.

| Step | $i$ (binary) | `tree[i]` | Accumulator $s$ |
|:----:|:------------:|:---------:|:---------------:|
| 1 | 7 (`111`) | 4 | 4 |
| 2 | 6 (`110`) | 11 | 15 |
| 3 | 4 (`100`) | 16 | 31 |
| 4 | 0 (`000`) | — | done |

The query correctly returns 31 after visiting just 3 nodes (since 7 has three set bits in binary).

!!! note "Number of Steps Equals Number of Set Bits"
    The prefix query visits exactly as many BIT entries as there are set bits in the binary representation of $i$. Since $i \leq n$, the worst case is $\lfloor \log_2 n \rfloor$ steps.

## Why the Ranges Cover the Prefix Exactly

Each BIT node at index $i$ covers positions $[i - \text{lowbit}(i) + 1, \; i]$. After stripping the lowest set bit to get $i' = i - \text{lowbit}(i)$, the next node covers positions ending at $i'$. The ranges are contiguous and non-overlapping because stripping the lowest bit "jumps" to the end of the immediately preceding block.

Formally, let $i = b_k b_{k-1} \cdots b_1$ in binary with set bits at positions $p_1 < p_2 < \cdots < p_m$. Then the query decomposes the prefix $[1, i]$ into exactly $m$ blocks, one for each set bit.

## Implementation

```python
"""
Prefix sum queries on a Binary Indexed Tree.

Demonstrates the prefix query algorithm with a detailed
step-by-step trace showing how set bits determine the
number of BIT nodes visited.
"""


# === Fenwick Tree with Traced Query ===

class FenwickTree:
    """BIT with an optional trace mode for prefix queries."""

    def __init__(self, n: int):
        self.n = n
        self.tree = [0] * (n + 1)

    def update(self, i: int, delta: int) -> None:
        """Add delta to position i."""
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)

    def query(self, i: int) -> int:
        """Return prefix sum a[1] + a[2] + ... + a[i]."""
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s

    def query_traced(self, i: int) -> int:
        """Prefix query with printed trace of each step."""
        s = 0
        step = 1
        print(f"  Query prefix({i}):")
        while i > 0:
            print(f"    Step {step}: i={i} (bin={bin(i)}), "
                  f"tree[{i}]={self.tree[i]}, s={s}+{self.tree[i]}={s + self.tree[i]}")
            s += self.tree[i]
            i -= i & (-i)
            step += 1
        print(f"    Result: {s}")
        return s


# === Demonstration ===

if __name__ == "__main__":
    data = [1, 3, 5, 7, 9, 2, 4, 6]
    ft = FenwickTree(len(data))
    for i, v in enumerate(data, 1):
        ft.update(i, v)

    print(f"Array: {data}")
    print()

    # Traced queries
    ft.query_traced(7)
    print()
    ft.query_traced(5)
    print()

    # Verify all prefix sums
    print("All prefix sums:")
    for i in range(1, len(data) + 1):
        expected = sum(data[:i])
        actual = ft.query(i)
        print(f"  prefix({i}) = {actual}  (expected {expected})  "
              f"{'OK' if actual == expected else 'MISMATCH'}")
```

**Output:**
```
Array: [1, 3, 5, 7, 9, 2, 4, 6]

  Query prefix(7):
    Step 1: i=7 (bin=0b111), tree[7]=4, s=0+4=4
    Step 2: i=6 (bin=0b110), tree[6]=11, s=4+11=15
    Step 3: i=4 (bin=0b100), tree[4]=16, s=15+16=31
    Result: 31

  Query prefix(5):
    Step 1: i=5 (bin=0b101), tree[5]=9, s=0+9=9
    Step 2: i=4 (bin=0b100), tree[4]=16, s=9+16=25
    Result: 25

All prefix sums:
  prefix(1) = 1  (expected 1)  OK
  prefix(2) = 4  (expected 4)  OK
  prefix(3) = 9  (expected 9)  OK
  prefix(4) = 16  (expected 16)  OK
  prefix(5) = 25  (expected 25)  OK
  prefix(6) = 27  (expected 27)  OK
  prefix(7) = 31  (expected 31)  OK
  prefix(8) = 37  (expected 37)  OK
```

## Complexity

The prefix query visits exactly $\text{popcount}(i)$ nodes — the number of set bits in $i$. Since $i \leq n$:

$$
\text{Time} = O(\log n) \qquad \text{Space} = O(1)
$$

In practice, the average number of steps is approximately $\frac{1}{2} \log_2 n$, since a random integer has about half its bits set on average.

## Reference

- Fenwick, P. M. (1994). A New Data Structure for Cumulative Frequency Tables. *Software: Practice and Experience*, 24(3), 327-336.
