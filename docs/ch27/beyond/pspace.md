# PSPACE

While NP captures the difficulty of finding solutions, some computational problems require reasoning about **all possible futures** --- as in two-player games or quantified logical statements. The class **PSPACE** captures problems solvable with polynomial memory, regardless of time. It contains both NP and co-NP, and its complete problems are believed to be strictly harder than NP-complete ones.

## Definition

!!! tip "Definition: PSPACE"
    **PSPACE** is the class of decision problems solvable by a deterministic Turing machine using $O(n^k)$ space for some constant $k$, where $n$ is the input size.

Equivalently, PSPACE can be defined using nondeterministic polynomial-space machines, since Savitch's theorem shows the two are equivalent.

## Savitch's Theorem

!!! tip "Theorem (Savitch, 1970)"
    NSPACE$(s(n)) \subseteq$ DSPACE$(s(n)^2)$ for any $s(n) \geq \log n$.

**Proof sketch.** The key idea is to solve reachability on the configuration graph. To check whether configuration $c_1$ reaches $c_2$ in at most $t$ steps, recursively check whether there exists a midpoint configuration $c_m$ such that $c_1$ reaches $c_m$ in $t/2$ steps and $c_m$ reaches $c_2$ in $t/2$ steps. The recursion depth is $O(\log t) = O(s(n))$, and each level stores one configuration of size $O(s(n))$, giving total space $O(s(n)^2)$.

**Corollary:** NPSPACE = PSPACE, since polynomial squared is still polynomial.

## Class Containments

The following chain of containments is known:

$$
\text{P} \subseteq \text{NP} \subseteq \text{PSPACE} \subseteq \text{EXPTIME}
$$

- **P $\subseteq$ NP:** A deterministic machine is a special case of a nondeterministic one.
- **NP $\subseteq$ PSPACE:** Given an NP problem with polynomial-length witnesses, enumerate all possible witnesses (exponentially many) using only polynomial space, reusing space for each attempt.
- **co-NP $\subseteq$ PSPACE:** Similarly, check that no witness exists by exhaustive search in polynomial space.
- **PSPACE $\subseteq$ EXPTIME:** A machine using $O(n^k)$ space has at most $2^{O(n^k)}$ configurations. It must halt within this many steps (or loop), so it runs in exponential time.

By the space hierarchy theorem, P $\neq$ PSPACE. However, whether NP $\neq$ PSPACE remains open.

## PSPACE-Completeness

!!! tip "Definition: PSPACE-Complete"
    A problem $L$ is **PSPACE-complete** if:

    1. $L \in \text{PSPACE}$
    2. Every problem in PSPACE is polynomial-time reducible to $L$

## TQBF: The Canonical PSPACE-Complete Problem

**True Quantified Boolean Formula (TQBF):** Given a fully quantified Boolean formula:

$$
\exists x_1 \forall x_2 \exists x_3 \cdots Q_n x_n \; \phi(x_1, \ldots, x_n)
$$

determine whether it is true.

!!! tip "Theorem"
    TQBF is PSPACE-complete.

**Membership in PSPACE.** Evaluate recursively: for $\exists x_i$, try both $x_i = 0$ and $x_i = 1$ (reusing space); for $\forall x_i$, check both. The recursion depth is $n$ and each level uses $O(n)$ space, giving $O(n^2)$ total space.

**PSPACE-hardness.** Any PSPACE computation can be encoded as a TQBF: the quantifiers alternate to express reachability in the configuration graph, mirroring Savitch's recursive construction.

## Game Problems

PSPACE naturally captures **two-player games** played on polynomial-size boards:

| Problem | Description | Complexity |
|---------|-------------|-----------|
| Generalized Geography | Does first player have a winning strategy? | PSPACE-complete |
| Generalized Hex | First player winning strategy on $n \times n$ board | PSPACE-complete |
| Generalized Chess | Winning strategy on $n \times n$ board | EXPTIME-complete |
| QBF Game | Two-player Boolean formula game | PSPACE-complete |

The alternation of $\exists$ and $\forall$ quantifiers in TQBF mirrors the alternation of moves between two players: "there exists a move for player 1 such that for all responses by player 2..."

## PSPACE vs the Polynomial Hierarchy

The **polynomial hierarchy** PH is defined as:

$$
\text{PH} = \Sigma_0^p \cup \Sigma_1^p \cup \Sigma_2^p \cup \cdots
$$

where $\Sigma_0^p = \text{P}$, $\Sigma_1^p = \text{NP}$, $\Sigma_2^p = \text{NP}^{\text{NP}}$, etc. The entire polynomial hierarchy is contained in PSPACE:

$$
\text{PH} \subseteq \text{PSPACE}
$$

If PSPACE = NP, then PH collapses to NP (which is considered unlikely). This provides evidence that PSPACE-complete problems are genuinely harder than NP-complete ones.

## IP = PSPACE

One of the landmark results in complexity theory:

!!! tip "Theorem (Shamir, 1992)"
    IP = PSPACE, where IP is the class of problems with interactive proof systems.

This means any PSPACE problem can be verified by an all-powerful prover interacting with a polynomial-time verifier. The proof uses arithmetization to convert TQBF into a protocol over finite fields.

??? example "Example: Geography Game"
    **Generalized Geography** is played on a directed graph. Players alternate choosing edges; each edge must start where the previous edge ended, and no vertex may be revisited. The player who cannot move loses.

    **Instance:** A directed graph with designated start vertex $s$. Does the first player have a winning strategy?

    This is PSPACE-complete because the game tree has alternating existential/universal choices (player 1 chooses, player 2 responds), mirroring the quantifier structure of TQBF.

    **Small example:** Graph with vertices $\{A, B, C, D\}$ and edges $A \to B$, $A \to C$, $B \to D$, $C \to D$, $D \to A$. Starting at $A$, player 1 can choose $A \to B$ or $A \to C$. The game tree determines who wins.

## Reference

- Sipser, M. (2012). *Introduction to the Theory of Computation* (3rd ed.). Cengage Learning, Chapter 8.
- Arora, S., & Barak, B. (2009). *Computational Complexity: A Modern Approach*. Cambridge University Press, Chapter 4.
