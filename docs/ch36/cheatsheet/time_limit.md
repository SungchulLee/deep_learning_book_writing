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

## Exercises

**Exercise 1.**
A problem has $n \le 10^5$ and a 2-second time limit. List the complexities that are likely to pass and those that will fail.

??? success "Solution to Exercise 1"
    Pass: $O(n) = 10^5$ (trivially). $O(n \log n) = 10^5 \times 17 \approx 1.7 \times 10^6$ (easy). $O(n \sqrt{n}) = 10^5 \times 316 \approx 3.2 \times 10^7$ (likely passes). $O(n \log^2 n) \approx 3 \times 10^7$ (likely passes). Borderline: $O(n^{1.5} \log n) \approx 10^8$ (tight). Fail: $O(n^2) = 10^{10}$ (50x over budget). $O(n^2 \log n) \approx 1.7 \times 10^{11}$. The practical threshold for C++ is $\sim 2 \times 10^8$ operations in 2 seconds. For Python, divide by 10. $\square$

---

**Exercise 2.**
The constraint is $n \le 20$. This strongly suggests exponential-time algorithms. List three algorithmic paradigms suitable for $n \le 20$.

??? success "Solution to Exercise 2"
    (1) **Bitmask DP**: represent subsets of $n$ elements as bitmasks. State space: $2^n = 10^6$ subsets. With $O(n)$ transitions per state: $O(n \cdot 2^n) = 2 \times 10^7$. Example: TSP in $O(n^2 \cdot 2^n)$. (2) **Backtracking with pruning**: explore the $2^n$ or $n!$ solution space, pruning branches that cannot improve the current best. Effective when the feasible region is small. (3) **Brute force over all subsets**: enumerate all $2^{20} \approx 10^6$ subsets and check each. Time: $O(2^n \cdot \text{check})$. For $n = 20$ with an $O(n)$ check: $2 \times 10^7$, feasible. The key insight: $2^{20} \approx 10^6$ is very manageable, but $2^{30} \approx 10^9$ is borderline, and $2^{40} \approx 10^{12}$ is infeasible. $\square$

---

**Exercise 3.**
A problem has $n \le 10^6$ and $q \le 10^5$ queries. The naive approach processes each query in $O(n)$. Is this feasible? If not, what complexity should you target?

??? success "Solution to Exercise 3"
    Naive: $O(nq) = 10^{11}$ -- infeasible (1000 seconds). Target: preprocessing in $O(n \log n)$ or $O(n)$, then each query in $O(\log n)$ or $O(1)$. Total: $O(n \log n + q \log n) \approx 2 \times 10^7$ (fast). Examples: (1) prefix sums: $O(n)$ preprocessing, $O(1)$ per range sum query. (2) Segment tree: $O(n)$ build, $O(\log n)$ per query and update. (3) Sparse table: $O(n \log n)$ build, $O(1)$ per range minimum query. The constraint pattern "$n$ large, $q$ moderate" signals that $O(\sqrt{n})$ per query might also work: $10^5 \times 10^3 = 10^8$ (borderline but possible with Mo's algorithm). $\square$

---

**Exercise 4.**
Map the following $n$ constraints to the most likely expected complexities: (a) $n \le 10$, (b) $n \le 1000$, (c) $n \le 10^5$, (d) $n \le 10^7$.

??? success "Solution to Exercise 4"
    (a) $n \le 10$: $O(n!)$ is feasible ($10! = 3.6 \times 10^6$). Also $O(2^n \cdot n^2) = 10^4$. These constraints suggest brute force, backtracking, or bitmask DP. (b) $n \le 1000$: $O(n^2) = 10^6$ (easy). $O(n^3) = 10^9$ (borderline). Suggests DP with $O(n^2)$ states or pairwise algorithms. (c) $n \le 10^5$: $O(n \log n)$ is comfortable. $O(n \sqrt{n}) \approx 3 \times 10^7$ (feasible). $O(n^2) = 10^{10}$ (too slow). Suggests sorting, binary search, segment trees, or balanced BSTs. (d) $n \le 10^7$: only $O(n)$ or $O(n \log \log n)$ works. $O(n \log n) \approx 2.3 \times 10^8$ (borderline). Suggests linear-time algorithms: sieve, counting sort, single-pass with hash map. $\square$

---

**Exercise 5.**
In Python, the effective operations-per-second is roughly $10^7$. A problem has $n \le 10^5$ with a 5-second limit. What is the maximum feasible complexity in Python? How can PyPy help?

??? success "Solution to Exercise 5"
    Budget: $5 \times 10^7 = 5 \times 10^7$ operations. $O(n \log n) = 10^5 \times 17 = 1.7 \times 10^6$ (fast, well within budget). $O(n \sqrt{n}) = 10^5 \times 316 = 3.2 \times 10^7$ (feasible but tight). $O(n^2) = 10^{10}$ (200x over budget, impossible). Maximum feasible: $O(n \sqrt{n})$ or $O(n \log^2 n)$. PyPy (JIT-compiled Python) runs 5--10x faster than CPython, effectively giving $5 \times 10^7$ to $10^8$ operations/second. With PyPy: $O(n \sqrt{n})$ is comfortable, and $O(n^{1.5} \log n)$ may pass. Strategies for Python: avoid per-element function calls (use list comprehensions, `map()`), use `sys.stdin` for input, and consider implementing the inner loop in a compiled extension or using NumPy for vectorized operations. $\square$
