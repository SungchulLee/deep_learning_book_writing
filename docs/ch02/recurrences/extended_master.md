# Extended Master Theorem

The standard [Master theorem](master.md) solves recurrences of the form $T(n) = aT(n/b) + f(n)$ in three cases, but it leaves a gap: when $f(n)$ is "close" to the watershed function $n^{\log_b a}$, differing by only a logarithmic factor, the standard theorem may not apply. The extended Master theorem fills this gap by adding cases that handle toll functions involving powers of $\log n$. This makes it the go-to tool for many practical recurrences that fall through the cracks of the basic version.

## Recap of the Standard Master Theorem

For reference, the standard Master theorem considers:

$$
T(n) = aT(n/b) + f(n)
$$

with $a \geq 1$ and $b > 1$, and compares $f(n)$ to the critical function $n^{\log_b a}$:

- **Case 1**: $f(n) = O(n^{\log_b a - \epsilon})$ for some $\epsilon > 0$ implies $T(n) = \Theta(n^{\log_b a})$
- **Case 2**: $f(n) = \Theta(n^{\log_b a})$ implies $T(n) = \Theta(n^{\log_b a} \log n)$
- **Case 3**: $f(n) = \Omega(n^{\log_b a + \epsilon})$ for some $\epsilon > 0$, with regularity condition, implies $T(n) = \Theta(f(n))$

The gap arises when $f(n) = \Theta(n^{\log_b a} \log^k n)$ for $k \neq 0$. Standard Case 2 only handles $k = 0$, and Cases 1 and 3 require a polynomial separation. The extended theorem resolves this.

## Extended Case 2

The key extension replaces the standard Case 2 with a more general version that accommodates logarithmic factors in the toll function.

!!! note "Extended Master Theorem (Case 2 Generalization)"
    Given $T(n) = aT(n/b) + f(n)$ with $a \geq 1$, $b > 1$, and $f(n) = \Theta(n^{\log_b a} \log^k n)$ for some constant $k \geq 0$:

    $$
    T(n) = \Theta(n^{\log_b a} \log^{k+1} n)
    $$

When $k = 0$, this reduces to the standard Case 2: $T(n) = \Theta(n^{\log_b a} \log n)$.

### Intuition

At each level of the recursion tree, the total work is $\Theta(n^{\log_b a} \log^k n)$ (adjusted for the level). There are $\Theta(\log n)$ levels, and the logarithmic factor accumulates across levels, adding one power of $\log n$ to the result.

## Full Extended Master Theorem

The complete extended version covers all relationships between $f(n)$ and $n^{\log_b a}$, including the sub-logarithmic gap.

!!! note "Full Extended Master Theorem"
    Given $T(n) = aT(n/b) + f(n)$ with $a \geq 1$, $b > 1$:

    **Case 1** (recursive work dominates): If $f(n) = O(n^{\log_b a - \epsilon})$ for some $\epsilon > 0$, then

    $$
    T(n) = \Theta(n^{\log_b a})
    $$

    **Case 2** (balanced with logarithmic factor): If $f(n) = \Theta(n^{\log_b a} \log^k n)$ for some $k \geq 0$, then

    $$
    T(n) = \Theta(n^{\log_b a} \log^{k+1} n)
    $$

    **Case 3** (toll function dominates): If $f(n) = \Omega(n^{\log_b a + \epsilon})$ for some $\epsilon > 0$, and $a f(n/b) \leq c f(n)$ for some $c < 1$ and all sufficiently large $n$, then

    $$
    T(n) = \Theta(f(n))
    $$

Some references present an even more refined version that handles $k < 0$ (inverse logarithmic factors) and sub-polynomial gaps, but the three cases above cover essentially all recurrences encountered in practice.

### Handling Negative Logarithmic Powers

A further extension addresses the case where $f(n) = \Theta(n^{\log_b a} / \log^k n)$ for $k > 0$ (equivalently, $f(n) = \Theta(n^{\log_b a} \log^{-k} n)$):

- If $k > 1$: the integral converges and $T(n) = \Theta(n^{\log_b a})$
- If $k = 1$: $T(n) = \Theta(n^{\log_b a} \log \log n)$
- If $0 < k < 1$: $T(n) = \Theta(n^{\log_b a} \log^{1-k} n)$

These sub-cases are rarely needed in practice but are important for completeness.

## Worked Examples

### Example 1: Logarithmic Toll

Consider:

$$
T(n) = 2T(n/2) + n \log n
$$

Here $a = 2$, $b = 2$, and $\log_b a = 1$. The toll function is $f(n) = n \log n = n^1 \cdot \log^1 n$, so $f(n) = \Theta(n^{\log_b a} \log^k n)$ with $k = 1$.

By Extended Case 2:

$$
T(n) = \Theta(n \log^{1+1} n) = \Theta(n \log^2 n)
$$

The standard Master theorem cannot handle this recurrence because $f(n) = n \log n$ is neither polynomially smaller nor polynomially larger than $n^{\log_b a} = n$.

### Example 2: Standard Case 2 as Special Case

Consider:

$$
T(n) = 4T(n/2) + n^2
$$

Here $a = 4$, $b = 2$, and $\log_b a = 2$. The toll function is $f(n) = n^2 = n^{\log_b a} \log^0 n$, giving $k = 0$.

By Extended Case 2:

$$
T(n) = \Theta(n^2 \log^{0+1} n) = \Theta(n^2 \log n)
$$

This is exactly what the standard Case 2 gives, confirming consistency.

### Example 3: Higher Logarithmic Power

Consider:

$$
T(n) = 2T(n/2) + n \log^3 n
$$

Here $a = 2$, $b = 2$, $\log_b a = 1$, and $f(n) = n \log^3 n = n^1 \cdot \log^3 n$, so $k = 3$.

By Extended Case 2:

$$
T(n) = \Theta(n \log^4 n)
$$

### Example 4: Non-Case-2 Recurrence (Verification)

Consider:

$$
T(n) = 9T(n/3) + n
$$

Here $a = 9$, $b = 3$, $\log_b a = 2$, and $f(n) = n$. Since $f(n) = O(n^{2 - \epsilon})$ with $\epsilon = 1$, this falls under Case 1:

$$
T(n) = \Theta(n^2)
$$

The extended theorem's Case 1 is identical to the standard theorem's Case 1.

## When to Use Which Theorem

| Toll function $f(n)$ relative to $n^{\log_b a}$ | Standard Master | Extended Master |
|--------------------------------------------------|----------------|-----------------|
| $f(n) = O(n^{\log_b a - \epsilon})$ | Case 1 applies | Case 1 (same) |
| $f(n) = \Theta(n^{\log_b a})$ | Case 2 applies | Case 2 with $k=0$ |
| $f(n) = \Theta(n^{\log_b a} \log^k n)$, $k > 0$ | Does not apply | Case 2 applies |
| $f(n) = \Omega(n^{\log_b a + \epsilon})$ with regularity | Case 3 applies | Case 3 (same) |
| $f(n)$ in the sub-logarithmic gap | Does not apply | May apply (see negative $k$ cases) |

## Proof Sketch for Extended Case 2

The proof follows the recursion tree analysis. At level $j$ of the recursion tree (for $j = 0, 1, \ldots, \log_b n - 1$), there are $a^j$ nodes, each processing a subproblem of size $n/b^j$. The work at level $j$ is:

$$
a^j \cdot f\!\left(\frac{n}{b^j}\right) = a^j \cdot \Theta\!\left(\left(\frac{n}{b^j}\right)^{\log_b a} \log^k \frac{n}{b^j}\right)
$$

Since $a^j / (b^j)^{\log_b a} = a^j / a^j = 1$, this simplifies to:

$$
\Theta\!\left(n^{\log_b a} \log^k \frac{n}{b^j}\right) = \Theta\!\left(n^{\log_b a} (\log n - j \log b)^k\right)
$$

Summing over all $\log_b n$ levels:

$$
T(n) = \Theta\!\left(n^{\log_b a} \sum_{j=0}^{\log_b n - 1} (\log n - j \log b)^k\right)
$$

The sum is $\Theta(\log^{k+1} n)$ (a standard result from summing polynomial powers), giving:

$$
T(n) = \Theta(n^{\log_b a} \log^{k+1} n)
$$

## Connections to Other Topics

- **[Master Theorem](master.md)**: The standard version that the extended theorem generalizes
- **[Akra-Bazzi Method](akra_bazzi.md)**: An even more general approach using integral evaluation
- **[Recursion Tree Method](recursion_tree.md)**: Provides the geometric intuition underlying the proof
- **[Recurrence from Divide and Conquer](divide_conquer.md)**: How to derive the recurrences that these theorems solve

## References

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 4. MIT Press.
- Leighton, T. (1996). Notes on better master theorems for divide-and-conquer recurrences. MIT CSAIL.
- Roura, S. (2001). An improved master theorem for divide-and-conquer recurrences. *Automata, Languages and Programming*, LNCS 2076, 449-459.
