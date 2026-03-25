# Formal Definitions

The preceding pages introduced Big-O, Big-Omega, and Big-Theta through examples and intuition.  This page collects all five asymptotic notations in one place, presents their precise set-based definitions, and discusses the notational conventions that textbooks use.  It serves as a quick-reference card for the entire asymptotic notation family.

## Notation as Sets

Mathematically, each asymptotic notation defines a **set of functions**.  When we write $f(n) = O(g(n))$, the equals sign does not mean ordinary equality -- it means $f$ *belongs to* the set $O(g(n))$.  The standard convention uses $=$ instead of $\in$ for readability, but the set interpretation is the correct one.

!!! warning "Abuse of notation"

    The expression $n^2 = O(n^3)$ is valid, but $O(n^3) = n^2$ is not.  The $=$ sign in asymptotic notation is **one-directional**: it means "is a member of," not "equals."  Always read it left to right.

## The Five Definitions

Throughout this section, $f$ and $g$ are functions from $\mathbb{N}$ (or $\mathbb{R}^+$) to $\mathbb{R}_{\geq 0}$.

### Big-O (Asymptotic Upper Bound)

$$
O(g(n)) = \{ f(n) : \exists\, c > 0,\, n_0 > 0 \text{ such that } 0 \leq f(n) \leq c \cdot g(n) \;\forall\, n \geq n_0 \}
$$

$f(n) = O(g(n))$ means $f$ grows **at most as fast as** $g$, up to a constant factor.  See [Big-O Notation](big_o.md) for examples and proof techniques.

### Big-Omega (Asymptotic Lower Bound)

$$
\Omega(g(n)) = \{ f(n) : \exists\, c > 0,\, n_0 > 0 \text{ such that } 0 \leq c \cdot g(n) \leq f(n) \;\forall\, n \geq n_0 \}
$$

$f(n) = \Omega(g(n))$ means $f$ grows **at least as fast as** $g$.  See [Big-Omega Notation](big_omega.md).

### Big-Theta (Asymptotically Tight Bound)

$$
\Theta(g(n)) = \{ f(n) : \exists\, c_1, c_2 > 0,\, n_0 > 0 \text{ such that } c_1 \cdot g(n) \leq f(n) \leq c_2 \cdot g(n) \;\forall\, n \geq n_0 \}
$$

$f(n) = \Theta(g(n))$ means $f$ and $g$ grow at the **same rate**, up to constant factors.  Equivalently, $f(n) = O(g(n))$ and $f(n) = \Omega(g(n))$.  See [Big-Theta Notation](big_theta.md).

### Little-o (Strictly Slower Growth)

$$
o(g(n)) = \{ f(n) : \forall\, c > 0,\, \exists\, n_0 > 0 \text{ such that } 0 \leq f(n) < c \cdot g(n) \;\forall\, n \geq n_0 \}
$$

$f(n) = o(g(n))$ means $f$ grows **strictly slower** than $g$.  Equivalently, $\lim_{n \to \infty} f(n) / g(n) = 0$.  See [Little-o and Little-omega](little.md).

### Little-omega (Strictly Faster Growth)

$$
\omega(g(n)) = \{ f(n) : \forall\, c > 0,\, \exists\, n_0 > 0 \text{ such that } 0 \leq c \cdot g(n) < f(n) \;\forall\, n \geq n_0 \}
$$

$f(n) = \omega(g(n))$ means $f$ grows **strictly faster** than $g$.  Equivalently, $\lim_{n \to \infty} f(n) / g(n) = \infty$.  See [Little-o and Little-omega](little.md).

## Key Differences at a Glance

| Notation | Quantifier on $c$ | Inequality | Intuitive meaning |
|---|---|---|---|
| $O$ | $\exists\, c > 0$ | $f \leq cg$ | Upper bound (possibly loose) |
| $\Omega$ | $\exists\, c > 0$ | $f \geq cg$ | Lower bound (possibly loose) |
| $\Theta$ | $\exists\, c_1, c_2 > 0$ | $c_1 g \leq f \leq c_2 g$ | Tight bound |
| $o$ | $\forall\, c > 0$ | $f < cg$ | Strictly below |
| $\omega$ | $\forall\, c > 0$ | $f > cg$ | Strictly above |

The critical distinction between Big-O and little-o (and between Big-Omega and little-omega) is the quantifier on $c$: existential ($\exists$) for the "big" versions versus universal ($\forall$) for the "little" versions.

## Relationships Between the Notations

The five notations are related by inclusion and duality.

**Inclusion:**

$$
f(n) = \Theta(g(n)) \implies f(n) = O(g(n)) \text{ and } f(n) = \Omega(g(n))
$$

$$
f(n) = o(g(n)) \implies f(n) = O(g(n)) \text{ but not } f(n) = \Theta(g(n))
$$

$$
f(n) = \omega(g(n)) \implies f(n) = \Omega(g(n)) \text{ but not } f(n) = \Theta(g(n))
$$

**Duality (transpose symmetry):**

$$
f(n) = O(g(n)) \iff g(n) = \Omega(f(n))
$$

$$
f(n) = o(g(n)) \iff g(n) = \omega(f(n))
$$

These dualities let us convert any upper-bound statement into a lower-bound statement by swapping $f$ and $g$.

## Domain and Non-negativity Conventions

Different textbooks make slightly different assumptions about the domain of $f$ and $g$:

- **CLRS convention:** $f$ and $g$ are **asymptotically non-negative** -- they are non-negative for all sufficiently large $n$.  This ensures that the inequalities in the definitions are meaningful.
- **Alternative convention:** Some sources allow $f$ to take negative values and define $O(g(n))$ as $|f(n)| \leq c \cdot g(n)$.  This is less common in algorithm analysis.

In this book, we follow the CLRS convention: $f(n) \geq 0$ and $g(n) > 0$ for all sufficiently large $n$.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 3. MIT Press.
- Knuth, D. E. (1976). Big Omicron and Big Omega and Big Theta. *SIGACT News*, 8(2), 18--24.
