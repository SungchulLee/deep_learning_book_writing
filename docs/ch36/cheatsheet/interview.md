# Interview Preparation

Algorithmic interviews test not just knowledge of data structures and algorithms but
also the ability to decompose problems under pressure, communicate reasoning clearly,
and estimate complexity on the fly. This page provides a structured preparation
framework covering the key phases of a coding interview.

## The UMPIRE Framework

A systematic approach prevents the most common interview failure mode: jumping straight
into code before understanding the problem.

| Step | Action | Time Budget |
|---|---|---|
| **U**nderstand | Restate the problem; clarify edge cases and constraints | 3--5 min |
| **M**atch | Identify which pattern or data structure applies | 2--3 min |
| **P**lan | Describe the algorithm in plain English; state time/space complexity | 3--5 min |
| **I**mplement | Write clean code with meaningful variable names | 10--15 min |
| **R**eview | Walk through with a small example; check off-by-one errors | 3--5 min |
| **E**valuate | Discuss complexity and potential optimizations | 2--3 min |

!!! tip "Communicate Before Coding"
    Interviewers evaluate your thought process as much as your final code. Explain
    your approach before writing a single line. If you get stuck, narrate what you
    are trying and what is blocking you.

## Complexity Estimation Checklist

Before coding, estimate the required time complexity from the input constraints.

| Input size $n$ | Max affordable complexity | Typical pattern |
|---|---|---|
| $n \le 10$ | $O(n!)$, $O(2^n)$ | Brute force, backtracking |
| $n \le 20$ | $O(2^n)$, $O(n \cdot 2^n)$ | Bitmask DP |
| $n \le 500$ | $O(n^3)$ | Floyd-Warshall, interval DP |
| $n \le 5000$ | $O(n^2)$ | Quadratic DP, nested loops |
| $n \le 10^5$ | $O(n \log n)$ | Sorting, balanced BST, segment tree |
| $n \le 10^6$ | $O(n)$ | Linear scan, hash table, BFS/DFS |
| $n \le 10^8$ | $O(n)$ or $O(\sqrt{n})$ | Math formula, number theory |

!!! warning "Off-by-Ten Errors"
    A common mistake is assuming $O(n^2)$ is fast enough for $n = 10^5$. Since
    $10^{10}$ operations take roughly 10 seconds, this exceeds typical 1--2 second
    time limits.

## Problem Classification

Recognizing the problem type is the critical first step toward a solution.

| Problem Signal | Likely Category | Key Data Structures |
|---|---|---|
| "Find shortest path" | Graph BFS/Dijkstra | Queue, priority queue |
| "Find all subsets" | Backtracking/bitmask | Recursion, bit manipulation |
| "Optimal substructure" + overlap | Dynamic programming | Array, hash map |
| "Sorted array" + "find target" | Binary search | Array |
| "Stream of elements" | Sliding window / heap | Deque, priority queue |
| "Connected components" | Union-Find or DFS | Disjoint set, stack |
| "Prefix/suffix query" | Prefix sum / segment tree | Array |
| "Parentheses/brackets" | Stack | Stack |

## Essential Data Structure Operations

Quick reference for the operations you need during interviews.

| Structure | Access | Search | Insert | Delete |
|---|---|---|---|---|
| Array | $O(1)$ | $O(n)$ | $O(n)$ | $O(n)$ |
| Hash map | -- | $O(1)$ avg | $O(1)$ avg | $O(1)$ avg |
| Sorted array | $O(1)$ | $O(\log n)$ | $O(n)$ | $O(n)$ |
| Heap | $O(1)$ top | $O(n)$ | $O(\log n)$ | $O(\log n)$ |
| BST (balanced) | -- | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ |
| Stack | $O(1)$ top | $O(n)$ | $O(1)$ | $O(1)$ |
| Queue | $O(1)$ front | $O(n)$ | $O(1)$ | $O(1)$ |

## Edge Cases Checklist

Before submitting your solution, verify these common edge cases.

- **Empty input**: array of length 0, empty string, null node
- **Single element**: array of length 1, tree with one node
- **All same elements**: array of identical values
- **Already sorted / reverse sorted**: tests for worst-case behavior
- **Negative numbers**: especially in sum problems and DP
- **Integer overflow**: products or sums exceeding 32-bit range
- **Duplicate values**: especially in BST, hash, and sorting problems
- **Disconnected graph**: multiple components

## Time Management

A 45-minute interview leaves no room for backtracking on approach.

| Phase | Minutes | Red Flag |
|---|---|---|
| Problem understanding | 5 | Still unclear after 5 min -- ask more questions |
| Algorithm design | 5--8 | No approach after 8 min -- try brute force first |
| Coding | 15--20 | Not done coding after 25 min total -- simplify |
| Testing and debugging | 5--10 | Bug found with 2 min left -- explain the fix verbally |

!!! tip "Start with Brute Force"
    If the optimal solution does not come to mind within 3 minutes, state and code
    the brute-force solution first. This shows competence and often reveals the
    structure needed for optimization.

## Reference

- [Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- McDowell, G. *Cracking the Coding Interview*. 6th ed. CareerCup, 2015.
