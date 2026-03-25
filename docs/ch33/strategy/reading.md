# Reading Problems

In competitive programming and algorithmic problem-solving, the most common source of incorrect submissions is not a flawed algorithm but a misunderstanding of the problem statement. Careful, systematic reading prevents wasted time on wrong approaches and edge-case surprises during testing. This section presents a structured methodology for extracting all necessary information from a problem statement before writing any code.

## Why Careful Reading Matters

A competitive programming contest allocates 2--5 hours for 5--12 problems, so every minute spent re-reading or debugging a misunderstood constraint is costly. Studies of contest logs show that roughly 30--40% of wrong submissions stem from misinterpreted requirements rather than algorithmic bugs. The upfront investment in careful reading pays for itself many times over.

## The Four-Pass Reading Strategy

A reliable approach breaks problem reading into four sequential passes, each with a distinct focus.

### Pass 1 -- High-Level Understanding

Read the entire problem once without taking notes. The goal is to classify the problem type and form an initial mental model.

- **What category?** Graph, dynamic programming, greedy, math, string, geometry, data structure?
- **What is being optimized or decided?** Minimum cost, maximum count, yes/no feasibility?
- **Is the problem asking for a single answer, multiple answers, or all answers?**

### Pass 2 -- Constraints and Bounds

Re-read with a focus on numerical limits. These determine the algorithmic complexity you can afford.

| Constraint range | Typical viable complexity |
|---|---|
| $n \le 10$ | $O(n!)$, brute-force permutations |
| $n \le 20$ | $O(2^n)$, bitmask DP |
| $n \le 500$ | $O(n^3)$, Floyd--Warshall |
| $n \le 5000$ | $O(n^2)$, quadratic DP |
| $n \le 10^5$ | $O(n \log n)$, sorting or segment tree |
| $n \le 10^6$ | $O(n)$, linear scan |
| $n \le 10^{18}$ | $O(\log n)$ or $O(\sqrt{n})$, binary search or math |

Record every constraint explicitly: ranges of $n$, $m$, edge weights, string lengths, coordinate bounds, and the number of test cases $T$. If the problem says "the sum of $n$ over all test cases does not exceed $10^5$," this is a fundamentally different constraint from "$n \le 10^5$ per test case."

### Pass 3 -- Input and Output Format

Parse the exact I/O specification line by line.

- **Indexing**: Are vertices numbered from 0 or from 1?
- **Edge direction**: Is the graph directed or undirected? (An undirected edge often requires adding two directed entries.)
- **Output precision**: "Print the answer with at most $10^{-6}$ absolute or relative error" means you need to control floating-point formatting.
- **Multiple test cases**: Does the first line contain $T$? Is there a blank line between outputs?
- **Special output values**: Should you print $-1$, `"impossible"`, or `"NO"` when no solution exists?

!!! warning "Whitespace and Newline Traps"
    Some judges are strict about trailing spaces, trailing newlines, or blank lines between test cases. Always match the exact format shown in the sample output.

### Pass 4 -- Sample Cases and Hidden Assumptions

Work through every sample input/output by hand. For each sample:

1. Trace the expected output step by step.
2. Identify which constraints are exercised (minimum $n$, maximum $n$, special structure).
3. Note any implicit assumptions the problem does not state but the samples reveal.

Often, samples do not cover corner cases. After verifying samples, mentally construct your own:

- Empty or minimal input ($n = 0$ or $n = 1$).
- Maximum input at the constraint boundary.
- Inputs that break greedy assumptions.

## Annotating the Problem

Develop a personal shorthand for marking up the problem during reading.

| Symbol | Meaning |
|---|---|
| `[C]` | Constraint -- record exact bound |
| `[I]` | Input detail -- indexing, direction, format |
| `[O]` | Output detail -- format, precision, special values |
| `[E]` | Edge case -- potential corner case to test |
| `[?]` | Ambiguity -- need to resolve from samples or clarification |

After annotation, write a one-paragraph summary in your own words before coding. If you cannot summarize the problem clearly, you have not understood it.

## Common Misreading Patterns

The following mistakes recur across all skill levels.

??? warning "1-indexed vs 0-indexed"
    Many problems use 1-indexed vertices or array positions. Submitting a solution that assumes 0-indexing produces wrong answers on most inputs. Always check the sample input to confirm.

??? warning "Directed vs Undirected"
    The statement may say "roads connect cities" (undirected) or "flights go from A to B" (directed). Treating a directed graph as undirected -- or vice versa -- is one of the most common structural errors.

??? warning "Multi-test Resets"
    When a problem has multiple test cases, forgetting to reset global arrays, visited flags, or accumulators between cases causes cascading errors that pass the first test case but fail subsequent ones.

??? warning "Overflow from Constraint Combinations"
    A problem with $n \le 10^5$ and values up to $10^9$ may require products or sums that exceed 32-bit integer range. Read constraints multiplicatively: if two values can be multiplied, check whether the product fits your data type.

## Worked Example

Consider the following problem excerpt:

> Given an undirected weighted graph with $n$ vertices ($1 \le n \le 2 \times 10^5$) and $m$ edges ($0 \le m \le 5 \times 10^5$), find the shortest path from vertex 1 to vertex $n$. Edge weights are positive integers not exceeding $10^9$. If no path exists, print $-1$.

Applying the four-pass strategy:

1. **Category**: Shortest path on weighted graph. Positive weights suggest Dijkstra.
2. **Constraints**: $n$ up to $2 \times 10^5$, $m$ up to $5 \times 10^5$, weights up to $10^9$. Dijkstra with a binary heap runs in $O(m \log n)$, which is roughly $5 \times 10^5 \times 17 \approx 8.5 \times 10^6$ operations -- well within limits. However, distances can reach $n \times 10^9 \approx 2 \times 10^{14}$, requiring 64-bit integers.
3. **I/O**: Vertices are 1-indexed. Output $-1$ if unreachable.
4. **Edge cases**: $m = 0$ and $n > 1$ means no path. $n = 1$ means distance is 0.

## Reference

- [Competitive Programmer's Handbook](https://cses.fi/book/book.pdf)
