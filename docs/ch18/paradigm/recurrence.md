# Recurrence Analysis

Divide-and-conquer algorithms solve a problem by breaking it into subproblems, solving each recursively, and combining the results. Because the algorithm calls itself on smaller inputs, its running time satisfies a **recurrence relation** -- an equation that expresses $T(n)$ in terms of $T$ evaluated at smaller arguments. Solving this recurrence yields the algorithm's asymptotic complexity.

This page shows how divide-and-conquer algorithms produce recurrences and surveys the three main techniques for solving them: the recursion tree method, the substitution method, and the Master Theorem.

## From Algorithm to Recurrence

Consider a divide-and-conquer algorithm that:

- divides a problem of size $n$ into $a$ subproblems, each of size $n/b$,
- spends $D(n)$ time dividing and $C(n)$ time combining.

Its running time satisfies

$$
T(n) = \begin{cases} \Theta(1) & \text{if } n \le n_0 \\ a \, T\!\left(\dfrac{n}{b}\right) + f(n) & \text{if } n > n_0 \end{cases}
$$

where $f(n) = D(n) + C(n)$ is the total non-recursive work at each level. The parameters $a$, $b$, and $f(n)$ completely determine the asymptotic behavior of $T(n)$.

### Example: Merge Sort

Merge sort splits the array in half ($a = 2$, $b = 2$), spending $O(1)$ to divide and $O(n)$ to merge. Its recurrence is

$$
T(n) = 2T\!\left(\frac{n}{2}\right) + \Theta(n)
$$

### Example: Binary Search

Binary search examines one half of the array ($a = 1$, $b = 2$) and spends $O(1)$ per call:

$$
T(n) = T\!\left(\frac{n}{2}\right) + \Theta(1)
$$

## The Recursion Tree Method

The **recursion tree** method converts a recurrence into a tree where each node represents the cost of a single subproblem. Summing the costs across all levels gives the total running time.

For the recurrence $T(n) = aT(n/b) + f(n)$:

- **Level 0** (root): one problem of size $n$ contributes $f(n)$.
- **Level 1**: $a$ problems of size $n/b$ each contribute $f(n/b)$, total $a \cdot f(n/b)$.
- **Level $k$**: $a^k$ problems of size $n/b^k$ each contribute $f(n/b^k)$, total $a^k \cdot f(n/b^k)$.
- **Depth**: the recursion bottoms out when $n/b^k \le n_0$, giving depth $k = \log_b n$ (ignoring constant $n_0$).

The total cost is

$$
T(n) = \sum_{k=0}^{\log_b n} a^k \cdot f\!\left(\frac{n}{b^k}\right)
$$

### Worked Example: Merge Sort

For $T(n) = 2T(n/2) + cn$:

| Level | Number of nodes | Size per node | Cost per node | Level cost |
|---|---|---|---|---|
| $0$ | $1$ | $n$ | $cn$ | $cn$ |
| $1$ | $2$ | $n/2$ | $cn/2$ | $cn$ |
| $2$ | $4$ | $n/4$ | $cn/4$ | $cn$ |
| $k$ | $2^k$ | $n/2^k$ | $cn/2^k$ | $cn$ |

Every level costs $cn$, and there are $\log_2 n$ levels, so

$$
T(n) = cn \cdot \log_2 n = \Theta(n \log n)
$$

## The Substitution Method

The **substitution method** involves two steps:

1. **Guess** the form of the solution (often informed by a recursion tree).
2. **Prove** the guess correct by mathematical induction.

### Example: Proving Merge Sort is $O(n \log n)$

**Claim.** $T(n) \le cn \log n$ for some constant $c > 0$ and all $n \ge 2$.

**Inductive step.** Assume $T(k) \le ck \log k$ for all $k < n$. Then

$$
T(n) = 2T\!\left(\frac{n}{2}\right) + \Theta(n) \le 2 \cdot c \cdot \frac{n}{2} \cdot \log \frac{n}{2} + dn
$$

$$
= cn(\log n - 1) + dn = cn \log n - cn + dn \le cn \log n
$$

provided $c \ge d$. $\square$

!!! warning "Common Substitution Pitfall"
    A frequent mistake is guessing $T(n) \le cn$ for merge sort. The inductive step yields $T(n) \le cn + dn$, which does not prove $T(n) \le cn$ because the extra $dn$ term cannot be absorbed. The guess must match the asymptotic form exactly, including logarithmic factors.

## The Master Theorem

The **Master Theorem** provides a direct formula for recurrences of the form

$$
T(n) = aT\!\left(\frac{n}{b}\right) + f(n)
$$

where $a \ge 1$ and $b > 1$. The key quantity is the **critical exponent** $\log_b a$, which represents the growth rate of the number of subproblems.

### Three Cases

**Case 1.** If $f(n) = O(n^{\log_b a - \epsilon})$ for some $\epsilon > 0$, then the leaf work dominates:

$$
T(n) = \Theta(n^{\log_b a})
$$

**Case 2.** If $f(n) = \Theta(n^{\log_b a})$, then work is evenly distributed across levels:

$$
T(n) = \Theta(n^{\log_b a} \log n)
$$

**Case 3.** If $f(n) = \Omega(n^{\log_b a + \epsilon})$ for some $\epsilon > 0$, and $af(n/b) \le cf(n)$ for some $c < 1$ (regularity condition), then the root work dominates:

$$
T(n) = \Theta(f(n))
$$

### Applying the Master Theorem

| Algorithm | Recurrence | $a$ | $b$ | $\log_b a$ | Case | $T(n)$ |
|---|---|---|---|---|---|---|
| Binary search | $T(n) = T(n/2) + O(1)$ | $1$ | $2$ | $0$ | 2 | $\Theta(\log n)$ |
| Merge sort | $T(n) = 2T(n/2) + O(n)$ | $2$ | $2$ | $1$ | 2 | $\Theta(n \log n)$ |
| Karatsuba | $T(n) = 3T(n/2) + O(n)$ | $3$ | $2$ | $1.585$ | 1 | $\Theta(n^{1.585})$ |
| Strassen | $T(n) = 7T(n/2) + O(n^2)$ | $7$ | $2$ | $2.807$ | 1 | $\Theta(n^{2.807})$ |

### When the Master Theorem Does Not Apply

The Master Theorem requires $f(n)$ to be **polynomially smaller or larger** than $n^{\log_b a}$. It does not cover cases where $f(n)$ differs by a logarithmic factor. For example, the recurrence

$$
T(n) = 2T\!\left(\frac{n}{2}\right) + n \log n
$$

falls in the gap between cases 2 and 3. The **extended Master Theorem** (Akra-Bazzi method) handles such cases; see the [detailed recurrence chapter](../../ch02/recurrences/akra_bazzi.md) for coverage.

## Practical Considerations

### Floors and Ceilings

Real algorithms split arrays at $\lfloor n/2 \rfloor$ and $\lceil n/2 \rceil$, not exactly $n/2$. The standard approach is to solve the recurrence assuming exact division and then verify that floors and ceilings do not change the asymptotic result. For the Master Theorem, this assumption is provably safe.

### Constant Factors in Base Cases

The base case $T(n_0) = \Theta(1)$ absorbs implementation-dependent constants. Changing the base case threshold (e.g., switching to insertion sort for $n \le 32$) does not alter the asymptotic solution but can significantly affect practical performance.

## Summary

Every divide-and-conquer algorithm produces a recurrence $T(n) = aT(n/b) + f(n)$. Three methods solve such recurrences:

1. **Recursion tree**: visualize the cost at each level and sum across all levels.
2. **Substitution**: guess the answer and prove it by induction.
3. **Master Theorem**: compare $f(n)$ to $n^{\log_b a}$ and read off the answer.

The Master Theorem is the fastest when it applies, but the recursion tree method provides intuition that the Master Theorem does not, and the substitution method handles cases the Master Theorem misses.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 4. MIT Press.
