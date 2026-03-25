# Catalan Numbers

The Catalan numbers form a sequence that appears in a remarkable variety of counting problems: the number of valid parenthesizations, the number of distinct binary search trees, the number of lattice paths that stay below the diagonal, and many more. Understanding this sequence equips an algorithm designer with a powerful counting tool.

## Intuition

Consider the problem of matching $n$ pairs of parentheses. For $n = 3$, the valid arrangements are `((()))`, `(()())`, `(())()`, `()(())`, and `()()()` — exactly 5 strings. This count is the third Catalan number $C_3 = 5$. The Catalan numbers capture the idea of "balanced" or "non-crossing" structures.

## Definition

The $n$-th Catalan number is defined by the closed-form expression:

$$
C_n = \frac{1}{n+1}\binom{2n}{n} = \frac{(2n)!}{(n+1)!\,n!}
$$

The first several values are:

$$
C_0 = 1,\; C_1 = 1,\; C_2 = 2,\; C_3 = 5,\; C_4 = 14,\; C_5 = 42,\; C_6 = 132
$$

## Recurrence

The Catalan numbers satisfy the recurrence:

$$
C_0 = 1, \quad C_{n+1} = \sum_{i=0}^{n} C_i \, C_{n-i} \quad (n \ge 0)
$$

This recurrence arises naturally: in a valid parenthesization, the first opening parenthesis matches some closing parenthesis at position $2i+1$, splitting the remaining string into two independent sub-problems of sizes $i$ and $n - i$.

## Proof of the Closed Form

We prove $C_n = \frac{1}{n+1}\binom{2n}{n}$ using the **reflection principle** (ballot problem approach).

**Setup.** Count lattice paths from $(0,0)$ to $(2n, 0)$ using steps $+1$ (up) and $-1$ (down), that never go below the $x$-axis. Each such path corresponds to a sequence of $n$ up-steps and $n$ down-steps where every prefix has at least as many ups as downs.

**Total paths.** Without the non-negativity constraint, there are $\binom{2n}{n}$ paths (choosing which $n$ of $2n$ steps go up).

**Bad paths.** A "bad" path touches $y = -1$ at some point. Reflect the portion of the path after the first touch of $y = -1$ across the line $y = -1$. This creates a bijection between bad paths from $(0,0)$ to $(2n,0)$ and all paths from $(0,0)$ to $(2n,-2)$, which have $n+1$ down-steps and $n-1$ up-steps. There are $\binom{2n}{n+1}$ such paths.

**Good paths.** Therefore:

$$
C_n = \binom{2n}{n} - \binom{2n}{n+1} = \binom{2n}{n} - \frac{n}{n+1}\binom{2n}{n} = \frac{1}{n+1}\binom{2n}{n}
$$

## Combinatorial Interpretations

The Catalan number $C_n$ counts each of the following:

| Structure | Description |
|---|---|
| Parenthesizations | Valid arrangements of $n$ pairs of parentheses |
| Binary trees | Distinct full binary trees with $n+1$ leaves |
| BST shapes | Structurally distinct BSTs on $n$ keys |
| Dyck paths | Lattice paths from $(0,0)$ to $(2n,0)$ that stay non-negative |
| Triangulations | Triangulations of a convex $(n+2)$-gon |
| Non-crossing partitions | Non-crossing partitions of $\{1, \ldots, n\}$ |
| Stack-sortable permutations | Permutations of $[n]$ sortable by a single stack |

## Asymptotics

Using Stirling's approximation on $\binom{2n}{n}$:

$$
C_n \sim \frac{4^n}{n^{3/2}\sqrt{\pi}}
$$

So $C_n$ grows exponentially with base 4, modulated by a polynomial factor.

## Computing Catalan Numbers

### Direct Formula

```python
def catalan_direct(n: int) -> int:
    """Compute the n-th Catalan number using the multiplicative formula.

    Runs in O(n) time and O(1) space.
    """
    if n <= 1:
        return 1
    result = 1
    for i in range(n):
        result = result * (2 * n - i) // (i + 1)
    return result // (n + 1)
```

### Dynamic Programming

```python
def catalan_dp(n: int) -> list[int]:
    """Compute Catalan numbers C_0 through C_n using the recurrence.

    Runs in O(n^2) time and O(n) space.
    """
    C = [0] * (n + 1)
    C[0] = 1
    for i in range(1, n + 1):
        for j in range(i):
            C[i] += C[j] * C[i - 1 - j]
    return C


if __name__ == "__main__":
    # === Example: first 10 Catalan numbers ===
    cats = catalan_dp(9)
    for i, c in enumerate(cats):
        print(f"C_{i} = {c}")

    # === Verify against direct formula ===
    for i in range(10):
        assert catalan_direct(i) == cats[i]
    print("Direct formula matches DP for C_0 through C_9.")
```

## Application: Matrix Chain Multiplication

The number of ways to fully parenthesize a product of $n+1$ matrices is $C_n$. For $n = 3$ (four matrices $A_1 A_2 A_3 A_4$), there are $C_3 = 5$ parenthesizations:

$$
((A_1 A_2)(A_3 A_4)),\; (A_1((A_2 A_3) A_4)),\; (A_1(A_2(A_3 A_4))),\; (((A_1 A_2) A_3) A_4),\; ((A_1(A_2 A_3)) A_4)
$$

The recurrence $C_{n+1} = \sum_{i=0}^{n} C_i C_{n-i}$ directly mirrors the dynamic programming decomposition used in the matrix chain multiplication algorithm.

## Generating Function

The ordinary generating function for the Catalan numbers is:

$$
C(x) = \sum_{n=0}^{\infty} C_n x^n = \frac{1 - \sqrt{1 - 4x}}{2x}
$$

This satisfies the functional equation $C(x) = 1 + x \cdot C(x)^2$, which encodes the recurrence.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 31.
- Stanley, R. P. (2015). *Catalan Numbers*. Cambridge University Press.
