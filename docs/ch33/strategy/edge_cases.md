# Edge Cases

An algorithm that solves the general case correctly but fails on boundary inputs will receive a Wrong Answer verdict. Edge cases are the extreme or degenerate inputs that exercise corner conditions in the logic -- empty collections, single elements, maximum values, and structural degeneracies. Systematically generating and testing these inputs is one of the most reliable ways to improve submission accuracy.

## Why Edge Cases Break Solutions

Most algorithms are designed with a "typical" input in mind. Edge cases violate the implicit assumptions behind that design.

- **Empty input** ($n = 0$): Loops that assume at least one iteration produce undefined behavior.
- **Single element** ($n = 1$): Comparisons between adjacent elements fail or become vacuously true.
- **Identical values**: Sorting-based logic that assumes distinct elements may produce wrong partitions.
- **Extreme values**: Arithmetic at $10^9$ or $10^{18}$ triggers integer overflow or precision loss.

## Systematic Edge Case Checklist

### Numeric Edge Cases

| Category | Specific cases to test |
|---|---|
| Zero | $n = 0$, value $= 0$, empty array |
| One | Single node, single edge, array of length 1 |
| Two | Minimum case for pairwise operations |
| Maximum | $n$ at constraint limit, values at $10^9$ or $10^{18}$ |
| Negative | Negative values, negative indices, all-negative arrays |
| Overflow | Products near $2^{31}$ or $2^{63}$, sums of large values |

### Structural Edge Cases

| Category | Specific cases to test |
|---|---|
| All same | Array of identical elements |
| Sorted | Already sorted input (ascending and descending) |
| Reverse sorted | Worst case for insertion sort, some pivot strategies |
| Alternating | Values alternating high/low |
| Star graph | One node connected to all others |
| Path graph | Tree degenerating into a linked list |
| Disconnected | Graph with multiple components or isolated nodes |
| Self-loops | Edge from a node to itself |
| Parallel edges | Multiple edges between the same pair of nodes |

### String Edge Cases

| Category | Specific cases to test |
|---|---|
| Empty string | Length 0 |
| Single character | Length 1 |
| All same character | `"aaaaaa"` |
| Palindrome | Even and odd length palindromes |
| Maximum length | String at the constraint boundary |

## Edge Cases by Problem Type

### Arrays and Sequences

- Array with one element: is the answer that element itself, or is it undefined?
- All elements equal: does your algorithm handle duplicate keys correctly?
- Array already sorted: does your algorithm perform in $O(n)$ or degenerate to $O(n^2)$?
- Maximum and minimum values appearing at the boundaries (first or last position).

### Graphs

- Graph with no edges ($m = 0$): is the answer trivially defined?
- Disconnected graph: does your BFS/DFS handle unreachable nodes?
- Tree (connected, $n - 1$ edges): does your cycle-detection code handle acyclic graphs?
- Complete graph ($m = \binom{n}{2}$): does the algorithm stay within time/space limits?
- Bipartite vs non-bipartite: does the algorithm depend on this distinction?

### Dynamic Programming

- Base case $n = 0$ or $n = 1$: is the DP table initialized correctly?
- Target value $= 0$: is the empty selection valid?
- All weights exceed capacity: is the answer correctly reported as 0 or impossible?
- Negative values in the DP transition: does the recurrence still hold?

### Geometry

- Collinear points: do cross-product comparisons handle zero correctly?
- Coincident points: two points at the same location.
- Points on coordinate axes: $x = 0$ or $y = 0$.
- Very large coordinates: potential overflow in distance or cross-product computations.

## Constructing Edge Cases

A systematic approach to constructing edge cases follows three principles.

**Principle 1 -- Boundary values.** For every variable with constraints $a \le x \le b$, test $x = a$ and $x = b$.

**Principle 2 -- Empty and minimal.** If the problem accepts zero-length input, test it. If not, test the smallest valid input.

**Principle 3 -- Degenerate structure.** For trees, test a path (depth $= n$) and a star (depth $= 1$). For strings, test single-character repetition. For numbers, test all-zero and all-maximum.

## Worked Example

**Problem**: Find the maximum subarray sum in an array of $n$ integers ($1 \le n \le 10^5$, $|a_i| \le 10^9$).

Edge cases to construct:

1. **$n = 1$**: The answer is $a_1$ itself. Kadane's algorithm must handle this.
2. **All negative**: $[-3, -2, -5]$. The answer is $-2$ (the maximum single element). Some implementations incorrectly return 0 by starting with `max_sum = 0`.
3. **All positive**: $[1, 2, 3]$. The answer is 6 (the entire array).
4. **Overflow**: $n = 10^5$ with all $a_i = 10^9$. The sum is $10^{14}$, which exceeds 32-bit integer range. Use 64-bit integers.
5. **Alternating signs**: $[10, -1, 10, -1, 10]$. The answer is 28 (the entire array). Tests that the algorithm correctly bridges small negatives.

## Anti-Patterns

!!! danger "Assuming Distinct Values"
    Many problems do not guarantee distinct input. If your algorithm uses a `set` or assumes unique keys in a sort, duplicates may cause incorrect behavior.

!!! danger "Forgetting the Zero Case"
    If the problem allows $n = 0$ (check the constraints carefully), an unguarded access to `a[0]` causes a runtime error.

!!! danger "Hardcoding Small Cases"
    Hardcoding returns for $n \le 2$ is fragile and error-prone. A robust algorithm handles small cases naturally through its general logic.

## Reference

- [Competitive Programmer's Handbook](https://cses.fi/book/book.pdf)
