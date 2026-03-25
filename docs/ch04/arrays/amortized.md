# Amortized Growth

A static array cannot grow beyond its initial capacity, so dynamic arrays must periodically allocate a larger block and copy all existing elements when the current block fills up. A single copy-everything operation costs $O(n)$ time, which seems expensive. However, these costly resizes happen infrequently enough that the **average cost per append**, measured over a long sequence of operations, is only $O(1)$. This is the essence of amortized analysis applied to array growth: occasional expensive operations are "paid for" by the many cheap operations that precede them.

## The Doubling Strategy

The most common growth policy doubles the array capacity each time it fills up. Starting with capacity $c = 1$, the array grows through capacities $1, 2, 4, 8, 16, \ldots$ as elements are appended.

**Resize rule:** when the number of elements $n$ equals the current capacity $c$, allocate a new array of capacity $2c$, copy all $n$ elements, and free the old array.

The cost of each append operation is:

- **No resize needed:** $O(1)$ to write the new element into the next open slot.
- **Resize needed:** $O(n)$ to copy $n$ elements to the new array, plus $O(1)$ to write the new element.

## Aggregate Analysis

The aggregate method computes the total cost of $n$ appends and divides by $n$.

Consider appending $n$ elements to an initially empty array. Resizes occur when the size reaches powers of 2. At size $2^k$, the array copies $2^k$ elements. The total copying cost over $n$ appends is

$$
\sum_{k=0}^{\lfloor \log_2 n \rfloor} 2^k = 2^{\lfloor \log_2 n \rfloor + 1} - 1 < 2n
$$

Each append also pays $O(1)$ for the write itself, contributing $n$ to the total. Therefore the total cost for $n$ operations is at most $3n$, giving an amortized cost per operation of

$$
\frac{3n}{n} = 3 = O(1)
$$

## Accounting Method

The accounting method assigns each operation a fixed **amortized charge** and shows that the accumulated credits always cover the actual costs.

Assign each append an amortized charge of **3 units**:

- **1 unit** pays for writing the new element.
- **1 unit** is saved as credit on the new element itself.
- **1 unit** is saved as credit on an earlier element that has not yet been copied in this growth phase.

When a resize doubles from capacity $c$ to $2c$, there are $c$ elements that need copying. Since the last resize (or the start), exactly $c/2$ new elements were appended, each contributing 2 credits (1 for itself, 1 for an older element). This provides $c/2 \times 2 = c$ credits, which is exactly enough to pay for copying all $c$ elements.

!!! tip "Intuition Behind the Charges"

    The key insight is that each new element pays not only for its own future relocation but also sponsors the relocation of one older element. By the time the array fills up, enough credits have accumulated to cover the entire copy operation.

## Potential Method

The potential method defines a potential function $\Phi$ on the data structure state and expresses the amortized cost as

$$
\hat{c}_i = c_i + \Phi(D_i) - \Phi(D_{i-1})
$$

where $c_i$ is the actual cost of operation $i$, and $D_i$ is the state after operation $i$.

Define the potential function as

$$
\Phi(D) = 2n - c
$$

where $s$ is the current number of elements and $c$ is the current capacity. Initially the array is empty with $s = 0$ and $c = 1$, so $\Phi(D_0) = 0$. After every resize the array is exactly half full ($s = c/2 + 1$), so $\Phi \ge 0$ always holds.

**Case 1: Normal append** ($s < c$). The actual cost is 1. The potential increases by 2:

$$
\hat{c} = 1 + \bigl[2(s+1) - c\bigr] - \bigl[2s - c\bigr] = 1 + 2 = 3
$$

**Case 2: Resize append** ($s = c$). The actual cost is $s + 1$ (copy $s$ elements plus write). Before the operation, $\Phi_{\text{before}} = 2s - c = 2s - s = s$. After the resize, the capacity doubles to $2s$ and the size becomes $s + 1$, giving $\Phi_{\text{after}} = 2(s + 1) - 2s = 2$. Thus

$$
\hat{c} = (s + 1) + (2 - s) = 3
$$

In both cases the amortized cost is exactly 3, confirming $O(1)$ amortized per append.

## Growth Factor Tradeoffs

The doubling strategy uses a growth factor of 2, but other factors work as well. Any constant factor $\alpha > 1$ yields $O(1)$ amortized appends, but the choice affects the constant and memory waste.

| Growth Factor $\alpha$ | Max Wasted Space | Amortized Cost Constant | Used By          |
|------------------------|------------------|-------------------------|------------------|
| 2                      | 50%              | 3                       | Java ArrayList   |
| 1.5                    | 33%              | Higher                  | C++ std::vector  |
| $\approx 1.125$        | 12.5%            | Higher                  | Python list      |

!!! warning "Additive Growth is Not Amortized O(1)"

    Growing by a fixed additive constant (e.g., adding 10 slots each time) results in $O(n)$ amortized cost per append, not $O(1)$. The geometric (multiplicative) growth policy is essential for the amortized constant-time guarantee.

## Python Demonstration

```python
"""Demonstrate amortized growth by tracking capacity changes in a Python list."""

import sys

# === Track capacity growth ===
sizes = []
data = []

for i in range(64):
    data.append(i)
    current_size = sys.getsizeof(data)
    if not sizes or current_size != sizes[-1][1]:
        sizes.append((i + 1, current_size))

print("Length | sys.getsizeof (bytes)")
print("-------|---------------------")
for length, size in sizes:
    print(f"  {length:4d} | {size}")
```

**Output:**
```
Length | sys.getsizeof (bytes)
-------|---------------------
     1 | 88
     5 | 120
     9 | 184
    17 | 248
    25 | 312
    33 | 376
    41 | 472
    53 | 568
```

The output shows that Python does not double exactly but uses a growth factor of approximately 1.125, trading a higher amortized constant for lower memory waste.

## Reference

- [Introduction to Algorithms (CLRS), Chapter 10](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
