# Inapproximability

Approximation algorithms provide guaranteed near-optimal solutions for NP-hard problems. A natural question follows: can we always do better with a cleverer algorithm? **Inapproximability** theory answers "no" for many problems, proving that certain approximation ratios cannot be achieved in polynomial time unless P = NP. These lower bounds complement the upper bounds from algorithm design and tell us when to stop searching for improvements.

## Gap-Producing Reductions

The standard technique for proving inapproximability uses **gap-producing reductions**. Rather than reducing one decision problem to another, we reduce a decision problem to a gap between two objective values.

A gap-producing reduction from an NP-hard decision problem $L$ to an optimization problem $\Pi$ works as follows:

- **YES instances** of $L$ map to instances of $\Pi$ with optimal value at most $c$.
- **NO instances** of $L$ map to instances of $\Pi$ with optimal value at least $\alpha \cdot c$.

If a polynomial-time algorithm achieved approximation ratio better than $\alpha$ on $\Pi$, it could distinguish YES from NO instances, solving $L$ in polynomial time. This contradicts P $\neq$ NP.

## The PCP Theorem

The most powerful tool for inapproximability is the **PCP (Probabilistically Checkable Proofs) Theorem**, one of the deepest results in complexity theory.

!!! tip "Theorem (PCP Theorem, Arora-Safra, Arora-Lund-Motwani-Sudan-Szegedy, 1998)"
    Every language in NP has a probabilistically checkable proof that can be verified by reading only $O(\log n)$ random bits and $O(1)$ bits of the proof.

Formally, NP = PCP$[\log n, 1]$, where PCP$[r(n), q(n)]$ denotes the class of languages with proofs verifiable using $r(n)$ random bits and $q(n)$ query bits.

### Connection to Inapproximability

The PCP Theorem implies that MAX-3SAT is NP-hard to approximate within some constant factor. Specifically:

!!! tip "Corollary (Hastad, 2001)"
    It is NP-hard to distinguish between 3SAT instances where at least a $(1 - \epsilon)$ fraction of clauses are satisfiable and instances where at most a $(7/8 + \epsilon)$ fraction are satisfiable, for any $\epsilon > 0$.

Since a random assignment satisfies $7/8$ of clauses in expectation, this result shows that beating the trivial random algorithm is NP-hard.

## Key Inapproximability Results

The following table summarizes landmark results, assuming P $\neq$ NP:

| Problem | Best Ratio | Inapproximability | Source |
|---------|------------|-------------------|--------|
| MAX-3SAT | $7/8 + \epsilon$ | $< 7/8 + \epsilon$ | Hastad (2001) |
| Set Cover | $O(\log n)$ | $(1 - \epsilon) \ln n$ | Dinur-Steurer (2014) |
| Clique | $O(n)$ | $n^{1 - \epsilon}$ | Hastad (1996), Zuckerman (2007) |
| Vertex Cover | $2$ | $< 2 - \epsilon$ (UGC) | Khot-Regev (2008) |
| General TSP | --- | Any finite ratio | Sahni-Gonzalez (1976) |
| Chromatic Number | $O(n)$ | $n^{1 - \epsilon}$ | Zuckerman (2007) |

## Set Cover Inapproximability

The **Set Cover** problem asks for the fewest sets from a collection $\mathcal{S} = \{S_1, \ldots, S_m\}$ that cover a universe $U$ of $n$ elements. The greedy algorithm achieves ratio $H_n = \ln n + O(1)$, and this is essentially optimal.

!!! tip "Theorem (Dinur-Steurer, 2014)"
    Unless P = NP, no polynomial-time algorithm can approximate Set Cover within a factor of $(1 - \epsilon) \ln n$ for any constant $\epsilon > 0$.

This shows that the simple greedy algorithm is essentially the best possible.

## General TSP Is Inapproximable

Unlike metric TSP (which admits a $3/2$-approximation), the general TSP has no finite approximation ratio.

**Proof sketch.** Suppose an $\alpha$-approximation exists for general TSP. Given a graph $G$ on $n$ vertices, construct a complete weighted graph: set $w(u,v) = 1$ if $(u,v) \in E(G)$, and $w(u,v) = \alpha n + 1$ otherwise. If $G$ has a Hamiltonian cycle, OPT $= n$. If not, any tour uses at least one heavy edge, giving cost $> \alpha n$. The $\alpha$-approximation would distinguish these cases, solving the NP-complete Hamiltonian Cycle problem. $\square$

## The Unique Games Conjecture

Many tight inapproximability results rely on the **Unique Games Conjecture (UGC)** of Khot (2002), which remains unproven.

**Unique Games Problem.** Given a constraint satisfaction problem where each constraint is a bijection between two variable domains of size $k$, distinguish:

- Instances where at least $(1 - \epsilon)$ fraction of constraints are satisfiable
- Instances where at most $\delta$ fraction are satisfiable

The UGC asserts this is NP-hard for all constants $\epsilon, \delta > 0$ and sufficiently large $k$.

Assuming the UGC, optimal inapproximability results follow for:

- **Vertex Cover:** ratio $2 - \epsilon$ is NP-hard
- **MAX-CUT:** the Goemans-Williamson ratio $\approx 0.878$ is optimal
- **Unique Label Cover:** forms the basis for many further reductions

??? example "Example: Proving MAX-CUT Inapproximability via Gap Reduction"
    Consider reducing from Unique Games to MAX-CUT.

    **Setup.** A Unique Games instance with value $(1 - \epsilon)$ maps to a MAX-CUT instance with cut value at least $(1 - f(\epsilon)) \cdot |E|$, where $f(\epsilon) \to 0$ as $\epsilon \to 0$.

    A Unique Games instance with value $\delta$ maps to a MAX-CUT instance with cut value at most $(c_{\text{GW}} + g(\delta)) \cdot |E|$, where $c_{\text{GW}} \approx 0.878$.

    Any algorithm beating $c_{\text{GW}}$ would distinguish these cases, contradicting the UGC.

## Implications for Algorithm Design

Inapproximability results guide where to direct research effort:

1. **Tight ratios.** When the best algorithm matches the inapproximability bound (e.g., Set Cover), the problem is "solved" from an approximation standpoint.
2. **Open gaps.** When a gap exists between the best algorithm and the best lower bound (e.g., Vertex Cover has ratio 2 but only $1.3606$ unconditional hardness), improving either side remains open.
3. **Parameterized approaches.** When polynomial-time approximation is hopeless, consider exact algorithms parameterized by input structure (treewidth, planarity).

## Reference

- Arora, S., & Barak, B. (2009). *Computational Complexity: A Modern Approach*. Cambridge University Press.
- Vazirani, V. V. (2001). *Approximation Algorithms*. Springer.
- Williamson, D. P., & Shmoys, D. B. (2011). *The Design of Approximation Algorithms*. Cambridge University Press.
