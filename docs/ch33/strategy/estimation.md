# Time and Space Estimation

Before writing a single line of code, competitive programmers estimate whether a candidate algorithm will fit within the time and memory limits. A rough calculation takes seconds but can prevent minutes of wasted implementation. This section develops the estimation techniques that separate experienced competitors from beginners.

## The Fundamental Rule of Thumb

Modern online judges allow approximately $10^8$ simple operations per second for C++ and roughly $10^7$ for Python. This gives a quick decision rule:

$$
\text{operations} = f(n) \quad \Longrightarrow \quad \text{fits in time if } f(n) \lesssim 10^8
$$

where $f(n)$ is the time complexity evaluated at the worst-case input size.

!!! tip "The 100-Million Rule"
    If your algorithm performs at most $10^8$ elementary operations (comparisons, additions, array accesses) on the largest input, it will almost certainly pass within a 1--2 second time limit in C++. For Python, target $10^7$ or fewer operations, or use PyPy.

## Complexity-to-Operations Table

Given a constraint $n$, the following table shows the approximate operation count for common complexities.

| Complexity | $n = 10^3$ | $n = 10^5$ | $n = 10^6$ | Fits in 1s? (C++) |
|---|---|---|---|---|
| $O(\log n)$ | 10 | 17 | 20 | Always |
| $O(n)$ | $10^3$ | $10^5$ | $10^6$ | Always |
| $O(n \log n)$ | $10^4$ | $1.7 \times 10^6$ | $2 \times 10^7$ | Yes |
| $O(n \sqrt{n})$ | $3.2 \times 10^4$ | $3.2 \times 10^7$ | $10^9$ | Borderline at $10^6$ |
| $O(n^2)$ | $10^6$ | $10^{10}$ | $10^{12}$ | Only if $n \le 10^4$ |
| $O(n^2 \log n)$ | $10^7$ | $1.7 \times 10^{11}$ | $2 \times 10^{13}$ | Only if $n \le 5000$ |
| $O(n^3)$ | $10^9$ | $10^{15}$ | $10^{18}$ | Only if $n \le 500$ |
| $O(2^n)$ | $10^{300}$ | -- | -- | Only if $n \le 25$ |
| $O(n!)$ | $10^{2567}$ | -- | -- | Only if $n \le 12$ |

## Estimation Procedure

### Step 1 -- Identify the Constraint

Read the problem for the primary input size $n$ and any secondary sizes ($m$ edges, $q$ queries, string length $L$, etc.).

### Step 2 -- Determine the Required Complexity

Match the constraint to the viable complexity class from the table above. For instance, if $n = 2 \times 10^5$, you need $O(n \log n)$ or better.

### Step 3 -- Compute the Constant Factor

Not all $O(n \log n)$ algorithms are equal. Consider:

- **Cache friendliness**: Merge sort accesses memory sequentially; quicksort is cache-friendly on average but has recursive overhead.
- **Operations per iteration**: An $O(n \log n)$ segment tree query with heavy node processing may have a constant factor of 10--20, making the effective count $2 \times 10^7$ rather than $10^6$.
- **Language overhead**: Python's interpreter adds a factor of roughly 10--100 compared to C++.

### Step 4 -- Account for Multiple Test Cases

If the problem has $T$ test cases, the total operation count is:

$$
\text{total} = T \times f(n_{\max})
$$

But if the problem says "the sum of $n$ over all test cases does not exceed $S$," then the bound is:

$$
\text{total} = f(S)
$$

This distinction is crucial -- the first allows $T$ independent worst cases, while the second amortizes across all cases.

## Space Estimation

Memory limits are typically 256 MB or 512 MB. Key memory costs:

| Data type | Size | Array of $10^6$ elements |
|---|---|---|
| `int` (32-bit) | 4 bytes | 4 MB |
| `long long` (64-bit) | 8 bytes | 8 MB |
| `double` | 8 bytes | 8 MB |
| `bool` | 1 byte | 1 MB |
| `pair<int,int>` | 8 bytes | 8 MB |

### Common Space Traps

**2D arrays**: An $n \times n$ integer array with $n = 10^4$ uses $10^8 \times 4 = 400$ MB -- this exceeds typical limits. Consider whether you can reduce to $O(n)$ space using DP rolling arrays.

**Adjacency lists**: A graph with $m$ edges stored as an adjacency list uses roughly $12m$ bytes (each edge stores a target vertex and weight). For $m = 5 \times 10^5$, this is about 6 MB -- well within limits.

**Recursive stack depth**: Each recursive call uses roughly 100--1000 bytes of stack space. A recursion depth of $10^5$ may cause stack overflow. Use iterative approaches or increase the stack size when recursion depth is large.

## Worked Examples

### Example 1 -- Sorting

**Problem**: Sort $n \le 10^6$ integers. Time limit: 2 seconds.

**Estimate**: `std::sort` is $O(n \log n)$ with a small constant. Operations: $10^6 \times 20 = 2 \times 10^7$. This is well within the $2 \times 10^8$ budget for 2 seconds.

### Example 2 -- All-Pairs Shortest Path

**Problem**: Given $n \le 400$ vertices, find shortest paths between all pairs. Time limit: 3 seconds.

**Estimate**: Floyd--Warshall is $O(n^3)$. Operations: $400^3 = 6.4 \times 10^7$. With a time limit of 3 seconds, the budget is $3 \times 10^8$. The algorithm fits comfortably.

### Example 3 -- Subset Sum

**Problem**: Given $n \le 20$ integers, determine if any subset sums to target $S$. Time limit: 1 second.

**Estimate**: Brute-force enumeration of all $2^n$ subsets: $2^{20} = 1,048,576 \approx 10^6$. This fits easily. For $n \le 40$, meet-in-the-middle splits into two halves of size 20: $2 \times 2^{20} + $ sorting gives roughly $2 \times 10^7$ -- still viable.

## When Estimates Are Borderline

If your estimate is close to the limit (within a factor of 2--3), consider:

1. **Optimize constants**: Avoid `std::map` (use arrays or unordered maps), minimize allocations, use `scanf`/`printf` instead of `cin`/`cout`.
2. **Try a different algorithm**: Often a borderline $O(n^2)$ can be replaced with a clean $O(n \log n)$ approach.
3. **Test locally**: Time your solution on a maximum-size random input before submitting.

## Reference

- [Competitive Programmer's Handbook](https://cses.fi/book/book.pdf)
