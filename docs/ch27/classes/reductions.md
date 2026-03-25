# Polynomial Reductions

How do we compare the difficulty of computational problems?  If we can efficiently transform every instance of problem $A$ into an instance of problem $B$ while preserving yes/no answers, then $B$ is "at least as hard" as $A$.  This idea -- **polynomial-time reduction** -- is the primary tool for establishing relative difficulty in complexity theory and the foundation upon which NP-completeness proofs are built.

## Intuition

A reduction from $A$ to $B$ is a compiler that translates $A$-questions into $B$-questions.  If you can solve $B$, you can solve $A$ by first translating, then solving.  The translation must be efficient (polynomial time) and correctness-preserving (yes maps to yes, no maps to no).

The direction of reduction is crucial: $A \leq_p B$ means "A reduces to B," which implies *B is at least as hard as A*, not the other way around.

## Many-One Reductions (Karp Reductions)

### Formal Definition

A **polynomial-time many-one reduction** (or **Karp reduction**) from language $A$ to language $B$ is a function $f : \Sigma^* \to \Sigma^*$ such that:

1. $f$ is computable in polynomial time, and
2. for every $x \in \Sigma^*$:

$$
x \in A \iff f(x) \in B
$$

When such a reduction exists, we write $A \leq_p B$ (or $A \leq_m^p B$).

??? info "Why 'many-one'?"
    The name "many-one" reflects that multiple inputs $x$ may map to the same output $f(x)$.  The function $f$ need not be injective.  This distinguishes many-one reductions from one-one reductions, where $f$ is required to be injective.

### Properties

**Transitivity.** If $A \leq_p B$ and $B \leq_p C$, then $A \leq_p C$.

*Proof.* Let $f$ reduce $A$ to $B$ in time $p(n)$ and $g$ reduce $B$ to $C$ in time $q(n)$.  The composition $g \circ f$ maps $x$ to $g(f(x))$.  Since $|f(x)| \leq p(|x|)$, the computation of $g(f(x))$ takes time at most $q(p(|x|))$, which is polynomial.  Correctness follows from both equivalences:

$$
x \in A \iff f(x) \in B \iff g(f(x)) \in C
$$

$\square$

**Hardness transfer.** If $A \leq_p B$ and $B \in \mathbf{P}$, then $A \in \mathbf{P}$.  Equivalently, by contrapositive: if $A \notin \mathbf{P}$, then $B \notin \mathbf{P}$.

**Closure.** If $A \leq_p B$, then:

- $B \in \mathbf{NP} \implies A \in \mathbf{NP}$
- $B \in \mathbf{co\text{-}NP} \implies A \in \mathbf{co\text{-}NP}$
- $B \in \mathbf{P} \implies A \in \mathbf{P}$

## Turing Reductions (Cook Reductions)

A more general notion allows the reduction to make *multiple* queries to an oracle for $B$.

### Formal Definition

A **polynomial-time Turing reduction** (or **Cook reduction**) from $A$ to $B$ is an algorithm that decides $A$ in polynomial time given access to an oracle for $B$.  We write $A \leq_T^p B$.

Every many-one reduction is a special case of a Turing reduction (make one oracle query on $f(x)$ and return the answer), so:

$$
A \leq_p B \implies A \leq_T^p B
$$

The converse does not hold in general.  Turing reductions are strictly more powerful because they allow:

- Multiple queries to the oracle.
- Adaptive queries (later queries depend on earlier answers).
- Using the oracle answer as part of further computation.

??? example "Turing reduction: optimization from decision"
    Consider optimization TSP (find the shortest tour) and decision TSP (is there a tour of cost $\leq k$?).  The optimization version Turing-reduces to the decision version: binary search on $k$ using the decision oracle finds the optimal cost, then edge-by-edge queries reconstruct the optimal tour.  This is a Turing reduction but *not* a many-one reduction.

## Comparison of Reduction Types

| Property | Many-One ($\leq_p$) | Turing ($\leq_T^p$) |
|----------|---------------------|---------------------|
| Queries to oracle | Exactly 1 | Polynomially many |
| Query must be final step | Yes | No |
| Output must equal oracle's answer | Yes | No |
| Defines NP-completeness | Yes (standard) | Yes (Cook's definition) |
| Transitivity | Yes | Yes |
| Strength | Weaker | Stronger |

!!! tip "Which reduction to use?"
    In complexity theory, NP-completeness is typically defined via many-one (Karp) reductions because they preserve membership in NP.  Turing reductions are used when studying structural properties or when many-one reductions are too restrictive (e.g., relating optimization to decision problems).

## Anatomy of a Reduction Proof

To prove $A \leq_p B$ (and thereby show $B$ is at least as hard as $A$):

**Step 1: Define the mapping.** Given an arbitrary instance $x$ of $A$, construct an instance $f(x)$ of $B$.

**Step 2: Prove the forward direction.** If $x \in A$, then $f(x) \in B$.

**Step 3: Prove the reverse direction.** If $f(x) \in B$, then $x \in A$.

**Step 4: Prove efficiency.** The computation of $f(x)$ runs in time polynomial in $|x|$.

### Example: CLIQUE Reduces to INDEPENDENT SET

**Claim.** CLIQUE $\leq_p$ INDEPENDENT SET.

**Reduction.** Given $(G, k)$ where $G = (V, E)$, output $(\overline{G}, k)$ where $\overline{G} = (V, \overline{E})$ is the complement graph.

**Forward.** If $G$ has a clique $S$ of size $k$, then every pair in $S$ is adjacent in $G$, so no pair in $S$ is adjacent in $\overline{G}$, meaning $S$ is an independent set of size $k$ in $\overline{G}$.

**Reverse.** If $\overline{G}$ has an independent set $S$ of size $k$, then no pair in $S$ is adjacent in $\overline{G}$, so every pair in $S$ is adjacent in $G$, meaning $S$ is a clique of size $k$ in $G$.

**Efficiency.** Constructing $\overline{G}$ takes $O(|V|^2)$ time.

$\square$

## Common Mistakes

!!! warning "Direction errors"
    The most common mistake in reduction proofs is reducing in the wrong direction.  To show $B$ is NP-hard, you must reduce *from* a known NP-hard problem $A$ *to* $B$: show $A \leq_p B$.  Reducing $B$ to $A$ would only show that $A$ is at least as hard as $B$, proving nothing about $B$'s hardness.

## Reference

- Sipser, M. *Introduction to the Theory of Computation*. Cengage Learning.
- Arora, S. and Barak, B. *Computational Complexity: A Modern Approach*. Cambridge University Press.
- Karp, R. M. "Reducibility Among Combinatorial Problems." *Complexity of Computer Computations*, 1972.
