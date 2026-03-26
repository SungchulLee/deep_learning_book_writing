# Sharp-P: Counting Complexity

NP asks whether a solution **exists**. A harder question is: **how many** solutions exist? The class **#P** (pronounced "sharp P") captures these counting problems. Remarkably, counting can be hard even when the corresponding decision problem is easy --- counting the number of perfect matchings in a bipartite graph is #P-complete, yet deciding whether one exists is in P.

## Definition

!!! tip "Definition: #P"
    A function $f : \{0,1\}^* \to \mathbb{N}$ is in **#P** if there exists a polynomial-time nondeterministic Turing machine $M$ such that for every input $x$:

    $$
    f(x) = \text{number of accepting computation paths of } M \text{ on } x
    $$

Equivalently, $f(x) = |\{w : V(x, w) = 1\}|$ where $V$ is a polynomial-time verifier and $w$ ranges over polynomial-length witnesses.

#P is a class of **functions** (not decision problems), outputting a non-negative integer rather than yes/no.

## Canonical Problems

| #P Problem | Corresponding Decision Problem | Decision Complexity |
|------------|-------------------------------|-------------------|
| #SAT: count satisfying assignments | SAT | NP-complete |
| #3-COLORING: count valid 3-colorings | 3-COLORING | NP-complete |
| #MATCHING: count perfect matchings | MATCHING | P |
| #DNF-SAT: count satisfying assignments of DNF | DNF-SAT | P |
| #HAMILTONIAN: count Hamiltonian cycles | HAMILTONIAN CYCLE | NP-complete |

## #P-Completeness

!!! tip "Definition: #P-Complete"
    A function $f$ is **#P-complete** if:

    1. $f \in$ #P
    2. Every function in #P is polynomial-time Turing reducible to $f$

A polynomial-time oracle for any #P-complete function would solve every problem in #P.

## Valiant's Theorem: Permanent Is #P-Complete

The **permanent** of an $n \times n$ matrix $A = (a_{ij})$ is:

$$
\text{perm}(A) = \sum_{\sigma \in S_n} \prod_{i=1}^{n} a_{i\sigma(i)}
$$

This looks like the determinant but without the sign factor $(-1)^{\text{sgn}(\sigma)}$. While the determinant is computable in $O(n^3)$ via Gaussian elimination, the permanent is much harder.

!!! tip "Theorem (Valiant, 1979)"
    Computing the permanent of a 0/1 matrix is #P-complete.

**Significance.** For bipartite graphs, the permanent of the biadjacency matrix equals the number of perfect matchings. Deciding whether a perfect matching exists is in P (e.g., Hopcroft-Karp), but counting how many exist is #P-complete. This demonstrates that counting can be dramatically harder than deciding.

## Toda's Theorem

!!! tip "Theorem (Toda, 1991)"
    The entire polynomial hierarchy is contained in P$^{\text{#P}}$:

    $$
    \text{PH} \subseteq \text{P}^{\text{\#P}}
    $$

This means a single call to a #P oracle is enough to solve any problem in the polynomial hierarchy. Since PH includes NP, co-NP, $\Sigma_2^p$, etc., this shows #P is extraordinarily powerful.

**Corollary.** If any #P-complete problem has a polynomial-time algorithm, then PH collapses to P.

## Relationship to Other Classes

$$
\text{P} \subseteq \text{NP} \subseteq \text{P}^{\text{\#P}} \subseteq \text{PSPACE}
$$

- **NP $\subseteq$ P$^{\text{#P}}$:** To decide if a solution exists, count solutions and check if the count is positive.
- **P$^{\text{#P}} \subseteq$ PSPACE:** A #P function can be computed in polynomial space by enumerating all witnesses.

## Approximate Counting

While exact counting is hard, **approximate counting** is sometimes tractable:

- **#DNF-SAT:** Admits an FPRAS (fully polynomial randomized approximation scheme) using Monte Carlo sampling, since DNF formulas have many satisfying assignments relative to the total.
- **Permanent:** Jerrum, Sinclair, and Vigoda (2004) gave an FPRAS for the permanent of non-negative matrices using Markov chain Monte Carlo.
- **#SAT:** No FPRAS unless NP = RP, since even approximate counting of SAT solutions is hard.

An **FPRAS** produces a $(1 \pm \epsilon)$-multiplicative approximation in time polynomial in $n$ and $1/\epsilon$.

## Connection to Statistical Physics

#P problems arise naturally in statistical physics. The **partition function** of a spin system:

$$
Z = \sum_{\sigma} e^{-\beta H(\sigma)}
$$

counts (with weights) the number of configurations. Computing $Z$ exactly is typically #P-hard, connecting computational counting to phase transitions and equilibrium properties.

??? example "Example: Counting Matchings"
    **Bipartite graph:** $L = \{1, 2, 3\}$, $R = \{a, b, c\}$ with edges $\{(1,a), (1,b), (2,a), (2,c), (3,b), (3,c)\}$.

    **Biadjacency matrix:**

    $$
    A = \begin{pmatrix} 1 & 1 & 0 \\ 1 & 0 & 1 \\ 0 & 1 & 1 \end{pmatrix}
    $$

    **Permanent:**

    $\text{perm}(A) = 1 \cdot 0 \cdot 1 + 1 \cdot 1 \cdot 0 + 0 \cdot 1 \cdot 1 + 0 \cdot 0 \cdot 0 + 1 \cdot 1 \cdot 1 + 1 \cdot 1 \cdot 1 = 0 + 0 + 0 + 0 + 1 + 1 = 2$

    The two perfect matchings are: $\{(1,b), (2,a), (3,c)\}$ and $\{(1,a), (2,c), (3,b)\}$. Deciding existence is easy; counting both required computing the permanent.

## Reference

- Valiant, L. G. (1979). The complexity of computing the permanent. *Theoretical Computer Science*, 8(2), 189--201.
- Arora, S., & Barak, B. (2009). *Computational Complexity: A Modern Approach*. Cambridge University Press, Chapter 17.
- Sipser, M. (2012). *Introduction to the Theory of Computation* (3rd ed.). Cengage Learning.
