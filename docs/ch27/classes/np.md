# The Class NP

Many computational problems share a striking asymmetry: finding a solution appears to require exhaustive search, yet *checking* a proposed solution is straightforward.  Sorting a list is easy, but factoring a large integer seems hard -- although verifying that two given factors multiply to the original number is trivial.  The complexity class **NP** formalizes this asymmetry by capturing every decision problem for which a "yes" answer comes with a short proof that can be verified quickly.

## Intuition

Consider the problem of determining whether a Boolean formula is satisfiable.  No one knows how to solve this in polynomial time for arbitrary formulas, yet if someone hands you a satisfying assignment, you can plug in the values and verify correctness in linear time.  NP is precisely the class of problems that admit this kind of efficient verification.

The name "NP" stands for **Nondeterministic Polynomial time**, reflecting an equivalent characterization through nondeterministic Turing machines.  Despite the name, NP is *not* "not polynomial" -- in fact, every problem in P is also in NP.

## Formal Definition

### Verifier-Based Definition

A language $L$ is in **NP** if there exists a deterministic polynomial-time Turing machine $V$ (the **verifier**) and a polynomial $p$ such that:

$$
x \in L \iff \exists\, c \in \Sigma^*,\; |c| \leq p(|x|),\; V(x, c) = \text{accept}
$$

The string $c$ is called a **certificate** (or **witness**).  The definition requires:

1. **Completeness**: every yes-instance has at least one valid certificate.
2. **Soundness**: no no-instance has any valid certificate.
3. **Efficiency**: the verifier runs in time polynomial in $|x|$.

### Nondeterministic Turing Machine Definition

Equivalently, $L \in \mathbf{NP}$ if there exists a nondeterministic Turing machine $N$ that decides $L$ in polynomial time:

$$
\mathbf{NP} = \bigcup_{k=0}^{\infty} \text{NTIME}(n^k)
$$

where $\text{NTIME}(T(n))$ is the class of languages decidable by a nondeterministic TM in $O(T(n))$ steps.  On a yes-instance, *at least one* computation path accepts; on a no-instance, *all* paths reject.

??? info "Equivalence of the two definitions"
    The two definitions capture the same class.  Given a verifier $V$, construct an NTM that nondeterministically guesses a certificate $c$ and then runs $V(x,c)$.  Conversely, given an NTM $N$, the certificate encodes the sequence of nondeterministic choices along an accepting path, and the verifier simulates $N$ following those choices.

## Key Properties

### P is Contained in NP

Every problem in P is also in NP: a polynomial-time algorithm can serve as its own verifier (ignoring the certificate entirely).

$$
\mathbf{P} \subseteq \mathbf{NP}
$$

### The Inclusion Chain

NP sits within the standard hierarchy:

$$
\mathbf{P} \subseteq \mathbf{NP} \subseteq \mathbf{PSPACE} \subseteq \mathbf{EXPTIME}
$$

The inclusion $\mathbf{NP} \subseteq \mathbf{PSPACE}$ holds because a deterministic machine can enumerate all possible certificates using polynomial space (though exponential time).

### Closure Properties

- **Union and Intersection**: if $L_1, L_2 \in \mathbf{NP}$, then $L_1 \cup L_2 \in \mathbf{NP}$ and $L_1 \cap L_2 \in \mathbf{NP}$.
- **Concatenation and Kleene star**: NP is closed under these operations.
- **Complement**: it is *unknown* whether NP is closed under complement.  This question is equivalent to asking whether $\mathbf{NP} = \mathbf{co\text{-}NP}$.

### Polynomial-Time Reductions

If $L_1 \leq_p L_2$ (there is a polynomial-time many-one reduction from $L_1$ to $L_2$) and $L_2 \in \mathbf{NP}$, then $L_1 \in \mathbf{NP}$.  This property is fundamental for the theory of NP-completeness.

## Canonical Examples

| Problem | Certificate | Verification |
|---------|------------|--------------|
| SAT | A satisfying assignment | Evaluate the formula: $O(m)$ |
| Hamiltonian Cycle | A permutation of vertices | Check edges exist: $O(n)$ |
| Graph Coloring ($k$-COLOR) | A color assignment | Check adjacencies: $O(m)$ |
| Clique ($k$-CLIQUE) | A set of $k$ vertices | Check all pairs adjacent: $O(k^2)$ |
| Subset Sum | A subset of numbers | Sum and compare: $O(n)$ |
| Integer Factoring | A nontrivial factor | Divide and check: $O(\log^2 n)$ |

!!! tip "Certificate perspective"
    When proving a problem is in NP, the standard approach is: (1) describe what the certificate looks like, (2) show it has polynomial length, and (3) describe a polynomial-time verification procedure.

## Problems Believed Not in NP

Not all natural problems are in NP.  Problems requiring a *universal* quantifier ("for all" rather than "there exists") may lie outside NP:

- **TAUTOLOGY**: Is a Boolean formula true under *every* assignment?  The natural certificate would need to certify that *no* falsifying assignment exists, which is not captured by the existential structure of NP.
- **Co-problems**: for every NP problem, the complementary problem (swapping yes/no) lies in co-NP, and it is unknown whether $\mathbf{NP} = \mathbf{co\text{-}NP}$.

## NP in Practice

Understanding NP has direct practical consequences:

- **Algorithm design**: when a problem is shown to be in NP but likely not in P (i.e., NP-complete), practitioners turn to approximation, heuristics, or parameterized algorithms rather than seeking exact polynomial-time solutions.
- **Cryptography**: many cryptographic schemes rely on the assumption that certain NP problems (like factoring or discrete logarithm) are *not* in P.
- **Verification**: the certificate structure of NP underpins proof systems, interactive proofs, and zero-knowledge proofs.

## Reference

- Sipser, M. *Introduction to the Theory of Computation*. Cengage Learning.
- Arora, S. and Barak, B. *Computational Complexity: A Modern Approach*. Cambridge University Press.
- Cook, S. "The Complexity of Theorem-Proving Procedures." *STOC*, 1971.
