# The Class co-NP

NP captures problems where "yes" answers have short, efficiently checkable proofs.  But what about problems where "no" answers are the ones with short proofs?  For example, proving a Boolean formula is *unsatisfiable* requires showing that *no* assignment works -- the natural certificate structure is reversed.  The class **co-NP** formalizes this complementary perspective, and the question of whether NP equals co-NP is one of the fundamental open problems in complexity theory.

## Intuition

Consider two related questions about a graph $G$:

- "Does $G$ have a Hamiltonian cycle?" -- A yes-answer is easy to verify: just exhibit the cycle.  This problem is in NP.
- "Does $G$ *not* have a Hamiltonian cycle?" -- A yes-answer to this complementary question means *no* Hamiltonian cycle exists, but how do you efficiently certify the absence of something?

Co-NP contains problems where the *no*-instances (equivalently, yes-instances of the complement) have efficient certificates.  If NP and co-NP are different, then some problems have an inherent asymmetry between proving existence and proving non-existence.

## Formal Definition

For a language $L \subseteq \Sigma^*$, the **complement** is $\overline{L} = \Sigma^* \setminus L$.

The class **co-NP** is defined as:

$$
\mathbf{co\text{-}NP} = \{ L \subseteq \Sigma^* \mid \overline{L} \in \mathbf{NP} \}
$$

Equivalently, $L \in \mathbf{co\text{-}NP}$ if and only if there exists a polynomial-time verifier $V$ and polynomial $p$ such that:

$$
x \notin L \iff \exists\, c,\; |c| \leq p(|x|),\; V(x, c) = \text{accept}
$$

In other words, the *no*-instances of $L$ have short certificates.  Rearranging, for yes-instances, *all* potential certificates must be rejected:

$$
x \in L \iff \forall\, c,\; |c| \leq p(|x|),\; V(x, c) = \text{reject}
$$

This universal quantifier ("for all") is what distinguishes co-NP from NP's existential quantifier ("there exists").

## Relationship to P and NP

### P is Contained in Both

Since P is closed under complement (just flip accept/reject), every language in P belongs to both NP and co-NP:

$$
\mathbf{P} \subseteq \mathbf{NP} \cap \mathbf{co\text{-}NP}
$$

### The Open Question

It is unknown whether $\mathbf{NP} = \mathbf{co\text{-}NP}$.  The prevailing conjecture is that they are different:

$$
\mathbf{NP} \neq \mathbf{co\text{-}NP} \quad \text{(conjectured)}
$$

**Consequence**: if $\mathbf{NP} \neq \mathbf{co\text{-}NP}$, then $\mathbf{P} \neq \mathbf{NP}$ (because P is contained in both).

??? info "NP = co-NP implies what?"
    If $\mathbf{NP} = \mathbf{co\text{-}NP}$, then for every NP-complete problem, the complement is also in NP.  This would mean UNSAT (unsatisfiability) has polynomial-length certificates, which would be a breakthrough result with major implications for proof complexity.

### The Hierarchy

$$
\mathbf{P} \subseteq \mathbf{NP} \cap \mathbf{co\text{-}NP} \subseteq \mathbf{NP} \cup \mathbf{co\text{-}NP} \subseteq \mathbf{PSPACE}
$$

It is unknown whether any of these inclusions is strict.

## Canonical Examples

| NP Problem | co-NP Complement | Certificate for "No" |
|-----------|------------------|---------------------|
| SAT | UNSAT (TAUTOLOGY) | No known short certificate |
| Hamiltonian Cycle | No Hamiltonian Cycle | No known short certificate |
| Composite Number | PRIMES | Primality certificate (Pratt) |
| Graph 3-Colorability | Non-3-Colorability | No known short certificate |

### Problems in NP and co-NP

Some problems are known to lie in $\mathbf{NP} \cap \mathbf{co\text{-}NP}$ without being known to be in P:

- **Factoring** (the decision version): given $n$ and $k$, does $n$ have a factor $\leq k$?  A factor serves as a yes-certificate; a complete factorization serves as a no-certificate.
- **Linear programming** (feasibility): known to be in P (via the ellipsoid method), hence in both.
- **Primality**: now known to be in P (AKS), but was historically the classic example of $\mathbf{NP} \cap \mathbf{co\text{-}NP}$ before an efficient algorithm was found.

??? example "Pratt certificates for primality"
    Before AKS (2002), primality was known to be in $\mathbf{NP} \cap \mathbf{co\text{-}NP}$.  A **Pratt certificate** proves a number is prime by exhibiting a primitive root and recursive primality proofs of the prime factors of $p - 1$.  This gives a polynomial-length certificate verifiable in polynomial time, placing PRIMES in NP (and thus COMPOSITES in co-NP).

## co-NP-Completeness

A language $L$ is **co-NP-complete** if:

1. $L \in \mathbf{co\text{-}NP}$, and
2. every language in co-NP reduces to $L$ in polynomial time.

Equivalently, $L$ is co-NP-complete if and only if $\overline{L}$ is NP-complete.

**Examples of co-NP-complete problems:**

- **TAUTOLOGY**: Is a Boolean formula true under *every* assignment?
- **UNSAT**: Is a Boolean formula unsatisfiable?
- **VALIDITY**: Is a first-order formula valid (over finite structures)?

**Theorem.** If any co-NP-complete problem is in NP, then $\mathbf{NP} = \mathbf{co\text{-}NP}$.

## Implications

The NP vs. co-NP question connects to several areas:

- **Proof complexity**: if $\mathbf{NP} \neq \mathbf{co\text{-}NP}$, then there exist tautologies that require super-polynomial-length proofs in any proof system.
- **Cryptography**: many cryptographic assumptions implicitly rely on the conjecture that $\mathbf{NP} \neq \mathbf{co\text{-}NP}$.
- **Program verification**: proving program correctness involves showing that *no* execution path leads to an error, which is a co-NP-type statement.

## Reference

- Sipser, M. *Introduction to the Theory of Computation*. Cengage Learning.
- Arora, S. and Barak, B. *Computational Complexity: A Modern Approach*. Cambridge University Press.
- Pratt, V. "Every Prime has a Succinct Certificate." *SIAM Journal on Computing*, 1975.
