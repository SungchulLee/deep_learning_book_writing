# Time Limit to Complexity

In competitive programming and coding interviews, the time limit tells you which
algorithmic complexity class is required. A modern computer executes roughly $10^8$
simple operations per second (additions, comparisons, array accesses). By comparing
this budget to the input size, you can estimate the maximum affordable time complexity
before writing any code.

## The Fundamental Rule

For a time limit of $T$ seconds:

$$
\text{max operations} \approx T \times 10^8
$$

A 1-second limit gives $10^8$ operations, a 2-second limit gives $2 \times 10^8$, and
so on. This is a rough estimate -- constant factors, cache behavior, and language
overhead all matter -- but it provides the right order of magnitude.

## Input Size to Required Complexity

Given a time limit of 1--2 seconds, the following table maps input size $n$ to the
maximum complexity class that will pass.

| $n$ | Max Complexity | Operations (approx) | Example Algorithms |
|---|---|---|---|
| $\le 10$ | $O(n!)$ | $3.6 \times 10^6$ | Permutation enumeration |
| $\le 15$ | $O(2^n \cdot n)$ | $5 \times 10^5$ | Bitmask DP with per-state work |
| $\le 20$ | $O(2^n)$ | $10^6$ | Subset enumeration, bitmask DP |
| $\le 25$ | $O(2^{n/2})$ | $6 \times 10^3$ | Meet in the middle |
| $\le 100$ | $O(n^4)$ | $10^8$ | Small matrix DP |
| $\le 500$ | $O(n^3)$ | $1.25 \times 10^8$ | Floyd-Warshall, interval DP |
| $\le 5000$ | $O(n^2)$ | $2.5 \times 10^7$ | Quadratic DP, pairwise comparison |
| $\le 10^5$ | $O(n \log n)$ | $1.7 \times 10^6$ | Sorting, segment tree, balanced BST |
| $\le 10^6$ | $O(n)$ | $10^6$ | Linear scan, BFS/DFS, hash table |
| $\le 10^7$ | $O(n)$ | $10^7$ | Sieve of Eratosthenes |
| $\le 10^8$ | $O(n)$ tight | $10^8$ | Simple loop with minimal work |
| $\le 10^{12}$ | $O(\sqrt{n})$ or $O(\log n)$ | $10^6$ or $40$ | Math, binary search |

!!! warning "Language Multipliers"
    Python is roughly 10--50x slower than C++ for tight loops. If the time limit
    is 1 second in C++, a Python solution may need an algorithm that is 1--2
    orders of magnitude faster, or you must use PyPy.

## Adjusting for Language

Different languages have very different constant factors.

| Language | Relative Speed | Effective ops/sec | Notes |
|---|---|---|---|
| C / C++ | 1x | $10^8$ -- $10^9$ | Baseline; compiler optimizations |
| Java | 2--3x slower | $3 \times 10^7$ -- $5 \times 10^7$ | JIT warmup helps |
| Python (CPython) | 30--100x slower | $10^6$ -- $3 \times 10^6$ | Interpreted |
| PyPy | 5--10x slower than C++ | $10^7$ -- $2 \times 10^7$ | JIT-compiled Python |
| JavaScript (V8) | 2--5x slower | $2 \times 10^7$ -- $5 \times 10^7$ | JIT-compiled |

## Reading the Constraints

Competitive programming problems embed the required complexity in their constraints.

| Constraint Pattern | Inferred Complexity | Approach |
|---|---|---|
| $1 \le n \le 10$ | $O(n!)$ or $O(2^n \cdot n^2)$ | Brute force or backtracking |
| $1 \le n \le 1000, 1 \le m \le 1000$ | $O(nm)$ | 2D DP |
| $1 \le n \le 10^5$ | $O(n \log n)$ | Sort-based, tree, or divide and conquer |
| $1 \le n \le 10^6$ | $O(n)$ | Linear algorithm required |
| $1 \le n \le 10^{18}$ | $O(\log n)$ or $O(1)$ | Math formula, matrix exponentiation |

!!! tip "Multiple Variables"
    When the problem has two size parameters $n$ and $m$, the product $n \times m$
    determines whether $O(nm)$ fits. For $n = m = 10^4$, the product is $10^8$ --
    right at the boundary.

## Memory Limits

Memory limits complement time limits. A typical limit of 256 MB constrains data
structure sizes.

| Data Type | Bytes | Max Elements in 256 MB |
|---|---|---|
| `int` (32-bit) | 4 | $6.4 \times 10^7$ |
| `long long` (64-bit) | 8 | $3.2 \times 10^7$ |
| `double` (64-bit) | 8 | $3.2 \times 10^7$ |
| `bool` | 1 | $2.56 \times 10^8$ |
| `int[5000][5000]` | 100 MB | Fits in 256 MB |
| `int[10000][10000]` | 400 MB | Does not fit |

This means a 2D DP table of size $n \times m$ with 32-bit integers requires
$4nm$ bytes. For $n = m = 8000$, this is 256 MB -- right at the limit.

## Practical Examples

| Problem | $n$ | Time Limit | Required | Algorithm |
|---|---|---|---|---|
| Two Sum | $10^5$ | 1s | $O(n)$ | Hash map |
| Merge Sort | $10^6$ | 2s | $O(n \log n)$ | Merge sort |
| All-Pairs Shortest Path | 400 | 2s | $O(n^3)$ | Floyd-Warshall |
| Subset Sum | 20 | 1s | $O(2^n)$ | Bitmask enumeration |
| LCS of two strings | 5000 | 1s | $O(n^2)$ | 2D DP |

## Reference

- [Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Halim, S. and Halim, F. *Competitive Programming*. 4th ed. 2020.
