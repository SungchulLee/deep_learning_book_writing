# Complexity Zoo

Complexity theory has identified hundreds of complexity classes, each capturing a distinct model of computation or resource constraint. This page provides a guided tour of the major classes beyond P and NP, their containment relationships, and the key open questions that connect them. Think of it as a map of the computational landscape.

## Time-Based Classes

### Deterministic Time

| Class | Definition | Key Problems |
|-------|-----------|-------------|
| **P** | DTIME($n^{O(1)}$) | Sorting, shortest paths, matching |
| **EXPTIME** | DTIME($2^{n^{O(1)}}$) | Generalized chess |
| **2-EXPTIME** | DTIME($2^{2^{n^{O(1)}}}$) | Certain equivalence problems |

Known separation: P $\neq$ EXPTIME (by the time hierarchy theorem).

### Nondeterministic Time

| Class | Definition | Key Problems |
|-------|-----------|-------------|
| **NP** | NTIME($n^{O(1)}$) | SAT, Clique, TSP decision |
| **co-NP** | Complements of NP problems | Tautology, unsatisfiability |
| **NEXPTIME** | NTIME($2^{n^{O(1)}}$) | Succinct circuit SAT |

Known: NP $\subseteq$ EXPTIME. Unknown: whether NP $\neq$ co-NP.

## Space-Based Classes

| Class | Definition | Key Problems |
|-------|-----------|-------------|
| **L** | DSPACE($O(\log n)$) | Undirected connectivity (Reingold) |
| **NL** | NSPACE($O(\log n)$) | Directed connectivity |
| **PSPACE** | DSPACE($n^{O(1)}$) | TQBF, generalized geography |

By Savitch's theorem, NSPACE($s$) $\subseteq$ DSPACE($s^2$), so NL $\subseteq$ L$^2$ $\subseteq$ P and NPSPACE = PSPACE.

## Randomized Classes

| Class | Error Type | Definition |
|-------|-----------|-----------|
| **ZPP** | Zero error | Expected polynomial time, always correct |
| **RP** | One-sided | No false positives, $\Pr[\text{accept} \mid x \in L] \geq 1/2$ |
| **co-RP** | One-sided | No false negatives |
| **BPP** | Two-sided | $\Pr[\text{correct}] \geq 2/3$ |

**Containments:** P $\subseteq$ ZPP = RP $\cap$ co-RP $\subseteq$ RP $\subseteq$ BPP.

**Conjecture:** BPP = P (randomness is not essential for polynomial-time computation).

## Circuit Classes

| Class | Definition |
|-------|-----------|
| **NC** | Polylog depth, polynomial size circuits (efficient parallel computation) |
| **AC$^0$** | Constant depth, polynomial size, unbounded fan-in |
| **TC$^0$** | AC$^0$ with threshold gates |
| **NC$^1$** | $O(\log n)$ depth, bounded fan-in |
| **P/poly** | Polynomial-size circuits (with advice) |

**Known:** AC$^0 \subsetneq$ TC$^0 \subseteq$ NC$^1 \subseteq$ L $\subseteq$ NL $\subseteq$ NC$^2 \subseteq$ P $\subseteq$ P/poly.

The class P/poly includes some undecidable problems (via advice), so P/poly $\not\subseteq$ NP is possible. However, if NP $\subseteq$ P/poly, the polynomial hierarchy collapses (Karp-Lipton theorem).

## Counting and Functional Classes

| Class | Definition |
|-------|-----------|
| **#P** | Count accepting paths of NTM |
| **FP** | Functions computable in polynomial time |
| **GapP** | Difference of two #P functions |
| **PP** | Majority of computation paths accept |

**Toda's theorem:** PH $\subseteq$ P$^{\text{#P}}$.

**PP vs BPP:** BPP $\subseteq$ PP, but PP is much more powerful (PP is PP-complete under polynomial-time reductions, and PH $\subseteq$ P$^{\text{PP}}$).

## Interactive Proof Classes

| Class | Definition |
|-------|-----------|
| **IP** | Interactive proofs with polynomial-time verifier |
| **AM** | Arthur-Merlin protocols (public coins) |
| **MA** | Merlin-Arthur (Merlin sends proof, Arthur verifies probabilistically) |
| **MIP** | Multiple independent provers |

**Landmark results:**

- IP = PSPACE (Shamir, 1992)
- AM $\subseteq$ PH (AM $\subseteq \Pi_2^p$)
- MIP = NEXPTIME (Babai, Fortnow, Lund, 1991)
- MIP* = RE (Ji et al., 2020) --- with entangled provers, the class equals the recursively enumerable languages

## The Grand Containment Picture

$$
\text{L} \subseteq \text{NL} \subseteq \text{P} \subseteq \text{NP} \cap \text{co-NP} \subseteq \text{NP} \cup \text{co-NP} \subseteq \text{PH} \subseteq \text{PSPACE} \subseteq \text{EXPTIME}
$$

Additionally:

$$
\text{P} \subseteq \text{BPP} \subseteq \text{PSPACE}
$$

$$
\text{NP} \subseteq \text{P}^{\text{\#P}} \subseteq \text{PSPACE}
$$

## Major Open Problems

| Question | Status | Implication |
|----------|--------|-------------|
| P $\stackrel{?}{=}$ NP | Open since 1971 | Most important open problem in CS |
| NP $\stackrel{?}{=}$ co-NP | Open | Would collapse PH if equal |
| P $\stackrel{?}{=}$ PSPACE | Open | Known: P $\neq$ PSPACE (space hierarchy) |
| BPP $\stackrel{?}{=}$ P | Conjectured equal | Would show randomness is inessential |
| L $\stackrel{?}{=}$ P | Open | Fundamental space vs time question |
| NP $\stackrel{?}{\subseteq}$ P/poly | Believed false | Would collapse PH (Karp-Lipton) |

!!! warning "Note on Barriers"
    Three barrier results limit proof techniques for P vs NP: **relativization** (Baker-Gill-Solovay), **natural proofs** (Razborov-Rudich), and **algebrization** (Aaronson-Wigderson). Any proof of P $\neq$ NP must circumvent all three.

??? example "Example: Navigating the Zoo"
    **Question:** Where does Graph Isomorphism (GI) sit?

    - GI $\in$ NP (a permutation serves as a witness).
    - GI is unlikely NP-complete: if it were, PH would collapse (Boppana-Hastad-Zachos).
    - GI $\in$ co-AM, so GI is "low" in the hierarchy.
    - Babai (2016) showed GI $\in$ quasi-polynomial time: $2^{O((\log n)^c)}$.

    GI sits between P and NP-complete, in a region believed to contain problems of intermediate difficulty (Ladner's theorem guarantees such problems exist if P $\neq$ NP).

## Reference

- Sipser, M. (2012). *Introduction to the Theory of Computation* (3rd ed.). Cengage Learning.
- Arora, S., & Barak, B. (2009). *Computational Complexity: A Modern Approach*. Cambridge University Press.
- Aaronson, S. (2016). P $\stackrel{?}{=}$ NP. In *Open Problems in Mathematics*, Springer.
