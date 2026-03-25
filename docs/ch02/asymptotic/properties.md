# Properties of Asymptotic Notation

Asymptotic notations obey a number of algebraic properties that make them practical to work with.  Instead of returning to the formal definition every time we need to combine or manipulate bounds, we can apply these properties as rewrite rules.  This page collects the most important ones: transitivity, reflexivity, symmetry, transpose symmetry, and the sum and product rules.

## Transitivity

All five asymptotic notations are transitive: if the relationship holds from $f$ to $g$ and from $g$ to $h$, then it holds from $f$ to $h$.

!!! info "Property -- Transitivity"

    - $f(n) = O(g(n))$ and $g(n) = O(h(n))$ imply $f(n) = O(h(n))$
    - $f(n) = \Omega(g(n))$ and $g(n) = \Omega(h(n))$ imply $f(n) = \Omega(h(n))$
    - $f(n) = \Theta(g(n))$ and $g(n) = \Theta(h(n))$ imply $f(n) = \Theta(h(n))$
    - $f(n) = o(g(n))$ and $g(n) = o(h(n))$ imply $f(n) = o(h(n))$
    - $f(n) = \omega(g(n))$ and $g(n) = \omega(h(n))$ imply $f(n) = \omega(h(n))$

Transitivity is what lets us chain comparisons.  For example, knowing $n = O(n \log n)$ and $n \log n = O(n^2)$ immediately gives $n = O(n^2)$.

??? example "Proof sketch for Big-O transitivity"

    Suppose $f(n) \leq c_1 \cdot g(n)$ for $n \geq n_1$ and $g(n) \leq c_2 \cdot h(n)$ for $n \geq n_2$.  Then for $n \geq \max(n_1, n_2)$:

    $$
    f(n) \leq c_1 \cdot g(n) \leq c_1 c_2 \cdot h(n)
    $$

    Choose $c = c_1 c_2$ and $n_0 = \max(n_1, n_2)$.  $\square$

## Reflexivity

A function is always bounded by itself.

!!! info "Property -- Reflexivity"

    - $f(n) = O(f(n))$
    - $f(n) = \Omega(f(n))$
    - $f(n) = \Theta(f(n))$

Reflexivity holds because we can choose $c = 1$ (or $c_1 = c_2 = 1$) and $n_0 = 1$.  Note that little-o and little-omega are **not reflexive**: $f(n) \neq o(f(n))$ and $f(n) \neq \omega(f(n))$, because a function cannot grow strictly slower (or faster) than itself.

## Symmetry

Among the five notations, only $\Theta$ is symmetric.

!!! info "Property -- Symmetry"

    $$
    f(n) = \Theta(g(n)) \iff g(n) = \Theta(f(n))
    $$

This follows from the sandwich definition: if $c_1 g \leq f \leq c_2 g$, then $(1/c_2) f \leq g \leq (1/c_1) f$.

Big-O and Big-Omega are not symmetric.  For example, $n = O(n^2)$ but $n^2 \neq O(n)$.

## Transpose Symmetry

Although Big-O and Big-Omega are not symmetric, they are related by swapping $f$ and $g$.

!!! info "Property -- Transpose symmetry"

    - $f(n) = O(g(n)) \iff g(n) = \Omega(f(n))$
    - $f(n) = o(g(n)) \iff g(n) = \omega(f(n))$

This duality converts any upper-bound statement into a lower-bound statement and vice versa.  It is the asymptotic analogue of the fact that $a \leq b$ is equivalent to $b \geq a$.

## Sum Rule

When two functions are added, the faster-growing one dominates.

!!! info "Property -- Sum rule"

    If $f_1(n) = O(g_1(n))$ and $f_2(n) = O(g_2(n))$, then

    $$
    f_1(n) + f_2(n) = O(\max(g_1(n), g_2(n)))
    $$

The analogous rules hold for $\Omega$, $\Theta$, $o$, and $\omega$.

In practice, this rule is the reason we can drop lower-order terms: $3n^2 + 7n + 4 = O(n^2)$ because $n^2$ dominates $n$ and the constant.

??? example "Application"

    Suppose algorithm A takes $O(n \log n)$ time and algorithm B takes $O(n^2)$ time.  Running both sequentially takes $O(n \log n + n^2) = O(n^2)$ time, since $n^2$ is the dominant term.

## Product Rule

Constants and other factors multiply through asymptotic bounds.

!!! info "Property -- Product rule"

    If $f_1(n) = O(g_1(n))$ and $f_2(n) = O(g_2(n))$, then

    $$
    f_1(n) \cdot f_2(n) = O(g_1(n) \cdot g_2(n))
    $$

A common special case: if $f(n) = O(g(n))$ and $k > 0$ is a constant, then $k \cdot f(n) = O(g(n))$.  This is why we ignore constant factors in asymptotic analysis.

??? example "Application"

    A loop that runs $n$ times, each iteration doing $O(\log n)$ work, takes $O(n) \cdot O(\log n) = O(n \log n)$ total time.

## Constant Factors and Lower-Order Terms

Two rules that follow directly from the sum and product rules:

1. **Constant factors can be dropped.** $c \cdot f(n) = \Theta(f(n))$ for any constant $c > 0$.
2. **Lower-order terms can be dropped.** If $f(n) = o(g(n))$, then $f(n) + g(n) = \Theta(g(n))$.

These rules justify the informal practice of simplifying $5n^2 + 3n \log n + 7$ to $\Theta(n^2)$.

## Summary Table

| Property | $O$ | $\Omega$ | $\Theta$ | $o$ | $\omega$ |
|---|---|---|---|---|---|
| Transitive | Yes | Yes | Yes | Yes | Yes |
| Reflexive | Yes | Yes | Yes | No | No |
| Symmetric | No | No | Yes | No | No |
| Transpose symmetric | $O \leftrightarrow \Omega$ | $\Omega \leftrightarrow O$ | -- | $o \leftrightarrow \omega$ | $\omega \leftrightarrow o$ |

For the formal definitions of all five notations, see [Formal Definitions](formal.md).  For techniques to compare specific functions, see [Growth Rate Comparison](comparison.md).

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 3. MIT Press.
