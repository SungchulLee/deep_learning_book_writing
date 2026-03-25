# Common Pitfalls

Competitive programming submissions fail for reasons that are often predictable and preventable. Understanding the most frequent pitfalls -- and developing habits to avoid them -- can turn a 50% acceptance rate into an 80%+ rate. This section catalogs the pitfalls that account for the vast majority of Wrong Answer, Runtime Error, and Time Limit Exceeded verdicts.

## Integer Overflow

Integer overflow is arguably the single most common pitfall in competitive programming. It occurs silently in C++ and produces wrong results without any error message.

### When It Happens

- Multiplying two `int` values each up to $10^9$: the product $10^{18}$ exceeds the 32-bit range of $2^{31} - 1 \approx 2.15 \times 10^9$.
- Summing $n = 10^5$ values each up to $10^9$: the sum can reach $10^{14}$.
- Computing $\binom{n}{k}$ or factorials for even moderate $n$.

### Prevention

- Use `long long` (64-bit) whenever intermediate results can exceed $2 \times 10^9$.
- In Python, integers have arbitrary precision, so overflow is not an issue -- but converting to `float` loses precision for large values.
- When performing modular arithmetic, reduce after every multiplication:

$$
(a \times b) \bmod m = ((a \bmod m) \times (b \bmod m)) \bmod m
$$

!!! danger "The Silent Killer"
    In C++, `int a = 100000; int b = a * a;` silently overflows. The result is undefined behavior, not an error message. Always cast before multiplication: `long long b = (long long)a * a;`.

## Off-by-One Errors

Off-by-one errors arise from confusion about inclusive vs exclusive boundaries, 0-indexed vs 1-indexed arrays, and loop termination conditions.

### Common Manifestations

- Loop runs $n - 1$ times instead of $n$ (or vice versa).
- Binary search returns the wrong boundary (the last element satisfying a condition vs the first element violating it).
- Array allocated with size $n$ but accessed at index $n$ (valid indices are $0$ to $n - 1$).
- Fence-post errors: $n$ items have $n - 1$ gaps between them.

### Prevention

- Always write out the loop invariant explicitly, even if just as a comment.
- For binary search, use a template with well-defined semantics (e.g., "find the smallest index $i$ such that $f(i)$ is true").
- Allocate arrays with a small buffer: `int a[MAXN + 5]` avoids boundary issues.

## Uninitialized Variables

Using a variable before assigning a value produces undefined behavior in C/C++ and can give different results on different machines or compiler settings.

### Common Scenarios

- Global arrays in C++ are zero-initialized, but local arrays are not.
- Forgetting to initialize `ans = 0` (or `ans = INF` for minimization) before a loop.
- Reusing a variable from a previous test case without resetting.

### Prevention

- Initialize all variables at declaration.
- For multi-test-case problems, reset all global state at the start of each test case, not at the end.

## Wrong Data Types

### Floating-Point Precision

Floating-point arithmetic introduces rounding errors that accumulate across operations. Comparing `double` values with `==` is almost always wrong.

- Use epsilon-based comparison: $|a - b| < \varepsilon$ with $\varepsilon = 10^{-9}$.
- Prefer integer arithmetic when possible. For instance, comparing $\frac{a}{b}$ vs $\frac{c}{d}$ is safer as $a \times d$ vs $c \times b$ (watching for overflow).

### Signed vs Unsigned

Mixing signed and unsigned integers in C++ causes implicit conversion bugs. A common trap: `for (int i = v.size() - 1; i >= 0; i--)` fails if `v.size()` returns `size_t` (unsigned) and the vector is empty, because `0u - 1` wraps to a huge positive number.

## Array and Memory Errors

### Out-of-Bounds Access

Accessing `a[n]` in an array of size $n$, or `a[-1]` in C++, is undefined behavior. It may work on your machine but crash on the judge.

### Stack Overflow

Deep recursion (depth $> 10^4$ in C++ without stack size adjustment) causes stack overflow. Solutions:

- Convert recursion to iteration using an explicit stack.
- Increase the stack size with compiler flags or OS settings.
- Use iterative DP instead of memoized recursion.

### Memory Limit Exceeded

A 2D array of size $10^4 \times 10^4$ with `int` uses 400 MB -- exceeding typical 256 MB limits. Use rolling arrays for DP or sparse representations for graphs.

## Multi-Test-Case Errors

### Forgetting to Reset State

When a problem has $T$ test cases, global data structures must be cleared between cases. Common items to reset:

- Visited arrays for BFS/DFS.
- Union-Find parent and rank arrays.
- Adjacency lists (clear or rebuild).
- Counters and accumulators.

### Wrong Output Format

- Missing or extra newline between test cases.
- Forgetting `"Case #X: "` prefix when required.
- Printing `"Yes"` instead of `"YES"` (case sensitivity).

## Algorithm-Specific Pitfalls

### Sorting

- Using an unstable sort when stability is required.
- Incorrect comparator: a comparator must define a strict weak ordering. If `comp(a, b)` and `comp(b, a)` can both be true, the sort produces undefined behavior.

### Graph Algorithms

- Forgetting to handle disconnected components.
- Using Dijkstra with negative edge weights (use Bellman--Ford instead).
- Confusing node indices (0-indexed vs 1-indexed) between the input and your data structure.

### Dynamic Programming

- Wrong base case initialization.
- Iterating in the wrong order for a knapsack-type DP (items before capacity vs capacity before items).
- Forgetting that the answer might not be at `dp[n]` but at `max(dp[0..n])`.

### Modular Arithmetic

- Forgetting to take the modulus at intermediate steps, causing overflow.
- Subtracting modular values without adding $m$ first: $(a - b) \bmod m$ should be computed as $((a - b) \bmod m + m) \bmod m$ to avoid negative results in C++.
- Using the wrong modulus ($10^9 + 7$ vs $998244353$).

## Pitfall Prevention Checklist

Before submitting, review this checklist:

- [ ] All variables initialized.
- [ ] Integer types are large enough for worst-case values.
- [ ] Array sizes include a small buffer.
- [ ] Multi-test-case state is reset.
- [ ] Output format matches exactly (case, spacing, newlines).
- [ ] Modular arithmetic applied at every multiplication and addition.
- [ ] Edge cases tested ($n = 0$, $n = 1$, maximum $n$).
- [ ] Comparator defines a strict weak ordering.

## Reference

- [Competitive Programmer's Handbook](https://cses.fi/book/book.pdf)
