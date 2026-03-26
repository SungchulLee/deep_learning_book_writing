# Pseudo-Polynomial Time

The 0/1 Knapsack problem is NP-hard, yet a simple dynamic programming algorithm solves it in $O(nW)$ time, where $W$ is the capacity. This seems polynomial --- until we realize that $W$ is a number, not a count. The input size required to encode $W$ is only $\log W$ bits, making $O(nW)$ exponential in the encoding length. An algorithm whose running time is polynomial in the **numeric values** of the input (rather than the encoding size) is called **pseudo-polynomial**.

## Input Size vs Numeric Value

The distinction hinges on how we measure input size:

- **Encoding size:** The number of bits to write down the input. An integer $W$ requires $\lceil \log_2(W + 1) \rceil$ bits.
- **Numeric value:** The integer $W$ itself.

For an algorithm running in time $O(nW)$:

- If we measure input size as $n + \log W$, the running time is $O(n \cdot 2^{\log W})$ --- exponential in input size.
- If we measure by numeric value, it appears polynomial.

!!! tip "Definition: Pseudo-Polynomial Time"
    An algorithm runs in **pseudo-polynomial time** if its running time is polynomial in the **numeric value** of the input and the number of items, but potentially exponential in the **encoding length** of the input.

## Weak vs Strong NP-Hardness

The concept of pseudo-polynomial time connects to a fundamental classification of NP-hard problems:

!!! tip "Definition: Weakly NP-Hard"
    A problem is **weakly NP-hard** if it is NP-hard but admits a pseudo-polynomial time algorithm.

!!! tip "Definition: Strongly NP-Hard"
    A problem is **strongly NP-hard** if it remains NP-hard even when all numbers in the input are bounded by a polynomial in $n$.

| Classification | Pseudo-Polynomial? | FPTAS? | Examples |
|---------------|-------------------|--------|---------|
| Weakly NP-hard | Yes | Often yes | Knapsack, Subset Sum, Partition |
| Strongly NP-hard | No (unless P = NP) | No (unless P = NP) | 3-Partition, Bin Packing, TSP |
| In P | Yes (truly polynomial) | N/A | Sorting, shortest paths |

**Theorem:** If a strongly NP-hard problem has a pseudo-polynomial algorithm, then P = NP.

## Example: Knapsack DP

Given $n$ items with values $v_1, \ldots, v_n$ and weights $w_1, \ldots, w_n$, and capacity $W$:

**DP recurrence:**

$$
\text{dp}[i][j] = \max(\text{dp}[i-1][j], \; v_i + \text{dp}[i-1][j - w_i])
$$

for $i = 1, \ldots, n$ and $j = 0, \ldots, W$.

**Time:** $O(nW)$. **Space:** $O(nW)$ (reducible to $O(W)$).

### Encoding Analysis

The input consists of $n$ integers of magnitude at most $W$. The encoding size is:

$$
L = O(n \log W)
$$

The running time $O(nW) = O(n \cdot 2^{\log W})$ is exponential in $L$ when $W$ is exponential in $n$.

**Concrete example:** With $n = 30$ items and $W = 2^{30} \approx 10^9$, the DP table has $30 \times 10^9 = 3 \times 10^{10}$ entries --- infeasible. But the input is only about 1000 bits.

## Example: Subset Sum

Given integers $a_1, \ldots, a_n$ and target $t$, determine if a subset sums to $t$.

**DP:** $O(nt)$ time. This is pseudo-polynomial because $t$ could be exponentially large relative to the encoding size.

**Strong NP-hardness:** Subset Sum is weakly NP-hard. It becomes polynomial when all numbers are bounded by $\text{poly}(n)$.

## Example: Partition Problem

Given integers $a_1, \ldots, a_n$, determine if they can be split into two subsets of equal sum.

This is equivalent to Subset Sum with target $t = \sum a_i / 2$. The DP runs in $O(n \cdot \sum a_i)$ time, which is pseudo-polynomial.

## Connection to FPTAS

Pseudo-polynomial algorithms often serve as the foundation for FPTAS design:

1. Start with a pseudo-polynomial DP running in $O(n \cdot V)$ time ($V$ = total value or capacity).
2. **Scale and round** the numeric values to reduce $V$ to $O(n/\epsilon)$.
3. The resulting algorithm runs in $O(n^2/\epsilon)$ or similar --- polynomial in both $n$ and $1/\epsilon$.

This path from pseudo-polynomial to FPTAS works precisely for weakly NP-hard problems.

!!! warning "Strongly NP-Hard Problems"
    Strongly NP-hard problems like 3-Partition and Bin Packing have no pseudo-polynomial algorithm (unless P = NP) and consequently no FPTAS. For Bin Packing, the best result is an additive $+1$ approximation (OPT + 1), which is an asymptotic PTAS.

## Summary of Relationships

$$
\text{Polynomial} \subset \text{Pseudo-polynomial} \subset \text{Exponential}
$$

A pseudo-polynomial algorithm is polynomial in the numbers but exponential in their bit-lengths. It sits between truly polynomial algorithms and fully exponential ones.

??? example "Example: When Pseudo-Polynomial Fails"
    **Subset Sum instance:** $a_1 = 1, a_2 = 2, a_3 = 4, \ldots, a_{40} = 2^{39}$, target $t = 2^{40} - 1$.

    **DP table size:** $O(n \cdot t) = O(40 \cdot 2^{40}) \approx 4.4 \times 10^{13}$.

    **Input encoding:** 40 integers, each needing about 40 bits $= 1600$ bits total.

    The DP is exponential in the input size (1600 bits), despite being "polynomial" in $t$. This is why pseudo-polynomial is not truly polynomial.

    For this instance, meet-in-the-middle ($O(2^{20})$) is much faster.

## Reference

- Garey, M. R., & Johnson, D. S. (1979). *Computers and Intractability: A Guide to the Theory of NP-Completeness*. W. H. Freeman.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.
