# Common Mistakes

Even experienced programmers make systematic errors when implementing algorithms under
time pressure. This page catalogs the most frequent mistakes, explains why they occur,
and provides concrete fixes. Reviewing these patterns before a contest or interview
significantly reduces debugging time.

## Off-by-One Errors

Off-by-one errors account for more wrong submissions than any other bug category. They
arise from confusion about inclusive vs. exclusive bounds and 0-indexed vs. 1-indexed
arrays.

| Mistake | Wrong | Correct | Why |
|---|---|---|---|
| Loop bound (0-indexed) | `for i in range(1, n)` | `for i in range(n)` | Misses index 0 |
| Loop bound (inclusive end) | `for i in range(l, r)` | `for i in range(l, r + 1)` | Misses last element |
| Binary search mid | `mid = (lo + hi) / 2` | `mid = lo + (hi - lo) // 2` | Also avoids overflow |
| Substring length | `s[i:j]` has length `j - i + 1` | Length is `j - i` | Python slicing is exclusive |
| Array allocation | `int arr[n]` | `int arr[n + 1]` | DP often needs index $n$ |

!!! warning "Binary Search Infinite Loop"
    When searching for the leftmost or rightmost element, using `lo < hi` vs
    `lo <= hi` and `mid = lo + (hi - lo) // 2` vs `mid = lo + (hi - lo + 1) // 2`
    must be matched correctly. A mismatch causes an infinite loop on arrays of
    size 2.

## Integer Overflow

Languages with fixed-width integers (C, C++, Java) silently overflow when intermediate
computations exceed the type's range.

| Operation | Risk | Safe Alternative |
|---|---|---|
| $a + b$ where $a, b \le 10^9$ | 32-bit overflow | Use `long long` (64-bit) |
| $a \times b$ where $a, b \le 10^9$ | 64-bit overflow possible | Use `__int128` or modular arithmetic |
| $n!$ for $n \ge 21$ | Exceeds 64-bit | Use modular factorial |
| Sum of $n$ elements | Overflow if $n \times \max > 2^{63}$ | Accumulate in 64-bit |

In Python, integers have arbitrary precision, so overflow is not an issue. However,
using Python's big integers in tight loops is slow compared to fixed-width arithmetic.

## Forgetting Base Cases

Recursive and DP solutions require explicit base cases. Missing them leads to infinite
recursion or wrong answers.

| Algorithm | Common Missing Base Case | Consequence |
|---|---|---|
| Binary search | Empty range ($lo > hi$) | Infinite loop or segfault |
| DFS/BFS | Visited check | Infinite loop on cyclic graphs |
| DP (memoization) | $dp[0]$ or $dp[1]$ initialization | Wrong values propagate |
| Tree recursion | Null node check | Null pointer exception |
| Divide and conquer | Single-element subproblem | Stack overflow |

!!! tip "Defensive Base Cases"
    Always handle the base case first in any recursive function. Check for null,
    empty, or size-1 inputs before any other logic.

## Wrong Complexity Analysis

Misjudging complexity leads to solutions that time out or, worse, pass small test
cases but fail large ones.

| Mistake | What You Think | Reality |
|---|---|---|
| Nested loop with `break` | $O(n)$ | Still $O(n^2)$ worst case |
| Hash map access in loop | $O(n)$ | $O(n^2)$ if hash collisions |
| Sorting inside a loop | $O(n \log n)$ | $O(n^2 \log n)$ if loop runs $n$ times |
| String concatenation in loop | $O(n)$ | $O(n^2)$ due to copying (in some languages) |
| `set.add()` in Python | $O(1)$ amortized | $O(n)$ worst case per operation |

## Graph Algorithm Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Forgetting to mark visited | Infinite loop | Mark as visited before/when enqueueing |
| BFS with wrong initial distance | Off-by-one in shortest path | Initialize source distance to 0, not 1 |
| Dijkstra with negative weights | Wrong shortest paths | Use Bellman-Ford instead |
| Modifying adjacency list during traversal | Skipped or repeated edges | Iterate over a copy |
| Directed vs. undirected confusion | Missing edges or double-counting | Add edges in both directions for undirected |

## DP Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Wrong recurrence direction | Accessing uncomputed states | Verify topological order of subproblems |
| Missing dimension in state | Wrong answer on some inputs | Check if state uniquely identifies subproblem |
| Initializing DP table to 0 | Wrong for min-cost problems | Use $\infty$ for minimization problems |
| Not handling the empty subsequence | Off-by-one | Add a dummy row/column of size 0 |
| Forgetting to reconstruct solution | Can report cost but not the actual answer | Track parent pointers alongside DP values |

## Sorting and Comparison Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Unstable sort changes relative order | Wrong output for ties | Use a stable sort or break ties explicitly |
| Custom comparator not transitive | Undefined behavior or crash | Ensure $a < b$ and $b < c$ implies $a < c$ |
| Sorting by wrong key | Correct complexity, wrong answer | Double-check the sort key |
| Forgetting that sort is $O(n \log n)$ | TLE in tight loops | Sort once, query many times |

## Language-Specific Pitfalls

| Language | Pitfall | Fix |
|---|---|---|
| Python | Recursion limit (default 1000) | `sys.setrecursionlimit(N)` |
| Python | Slow I/O with `input()` | Use `sys.stdin.readline()` |
| C++ | Uninitialized variables | Always initialize, especially arrays |
| C++ | `endl` flushes buffer | Use `"\n"` instead for speed |
| Java | `==` compares references, not values | Use `.equals()` for strings and objects |
| Java | Auto-boxing overhead | Use primitive arrays, not `ArrayList<Integer>` |

!!! warning "Python Recursion Limit"
    Python's default recursion limit of 1000 is insufficient for most algorithmic
    problems. Set it explicitly with `sys.setrecursionlimit(10**6)` and be aware
    that deep recursion may still cause a segfault due to stack size limits.

## Reference

- [Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- McDowell, G. *Cracking the Coding Interview*. 6th ed. CareerCup, 2015.
