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

## Exercises

**Exercise 1.**
You are given an unsorted array of $n$ integers and asked to find the $k$-th smallest element. Describe two approaches with different time-space tradeoffs.

??? success "Solution to Exercise 1"
    **Approach 1 -- Sort**: sort the array in $O(n \log n)$, return the $k$-th element. Simple, uses $O(1)$ extra space (in-place sort). **Approach 2 -- Quickselect**: partition the array around a pivot (like quicksort). If the pivot's position equals $k$, return it. Otherwise, recurse on the relevant half. Expected time: $O(n)$ (each recursion halves the problem on average). Worst case: $O(n^2)$ (bad pivots). The Median of Medians algorithm guarantees $O(n)$ worst case but has a large constant. For interviews, Quickselect is the expected answer: it demonstrates understanding of partitioning, expected-time analysis, and the tradeoff between simplicity and worst-case guarantees. $\square$

---

**Exercise 2.**
During an interview, you realize your initial approach is $O(n^2)$ but the expected complexity is $O(n \log n)$. Describe how to communicate this transition to the interviewer.

??? success "Solution to Exercise 2"
    Say: "My current approach is $O(n^2)$ because [explain the nested loop or repeated scan]. I can improve this by [specific technique: sorting, using a hash map, binary search on the answer, divide and conquer]. Let me walk through the optimized approach." Key principles: (1) Acknowledge the suboptimal solution before the interviewer asks -- it shows awareness. (2) Explain *why* it is slow (identify the bottleneck). (3) State the target complexity and the technique that achieves it. (4) Ask if the interviewer wants you to code the brute force first or jump to the optimized solution. Never silently abandon your first approach; the transition demonstrates algorithmic thinking, which is what the interview evaluates. $\square$

---

**Exercise 3.**
Given a binary tree, find the lowest common ancestor (LCA) of two nodes. Describe the recursive algorithm, its time complexity, and the edge cases to handle.

??? success "Solution to Exercise 3"
    Recursive algorithm: if the current node is null, return null. If the current node equals either target, return it. Recurse on left and right children. If both return non-null, the current node is the LCA. If only one returns non-null, propagate it upward. Time: $O(n)$ (visit each node once). Space: $O(h)$ for recursion stack where $h$ is the tree height. Edge cases: (1) one node is an ancestor of the other -- the ancestor itself is the LCA; (2) one or both nodes are not in the tree -- the algorithm may return incorrect results; add a check by verifying both nodes were found; (3) tree has only one node; (4) both target nodes are the same -- the LCA is the node itself. $\square$

---

**Exercise 4.**
Explain the "STAR" method (Situation, Task, Action, Result) for answering behavioral questions in technical interviews. Give an example related to debugging a performance issue.

??? success "Solution to Exercise 4"
    **Situation**: "Our production system's response time spiked from 50ms to 2 seconds during peak hours." **Task**: "I was responsible for identifying and fixing the performance regression before the next trading session." **Action**: "I profiled the application and found that a new feature was performing N+1 database queries (one per item in a list of 10,000 items). I refactored the code to batch queries using a single JOIN, added an index on the lookup column, and implemented response caching with a 5-second TTL." **Result**: "Response time dropped to 30ms (below the original baseline), database load decreased by 95%, and the fix was deployed within 4 hours." The STAR method structures the answer so the interviewer can follow the reasoning, and the quantified result demonstrates impact. $\square$

---

**Exercise 5.**
You are stuck on an interview problem after 10 minutes. List four strategies for making progress without asking for a direct hint.

??? success "Solution to Exercise 5"
    (1) **Work through small examples**: manually solve the problem for $n = 1, 2, 3, 4$. Patterns often emerge from concrete instances. (2) **Consider related problems**: "This looks like a variant of [known problem]. In that problem, we use [technique]. Can I adapt it here?" This shows breadth of knowledge. (3) **Simplify the problem**: solve a restricted version first (e.g., assume sorted input, or $k = 1$), then generalize. (4) **Think about the output structure**: what does the answer look like? If it is a subset, consider DP or greedy. If it is a sequence, consider sorting or BFS/DFS. If it is a count, consider combinatorics or DP. Verbalize your thought process throughout -- interviewers give credit for structured reasoning even without a complete solution. Saying "I am considering approach X because..." is better than silent thinking. $\square$
