# Clique

In a social network, a **clique** is a group of people who all know each other -- every pair is connected.  Finding the largest such group, or even determining whether one of a given size exists, turns out to be computationally hard.  The CLIQUE problem is one of Karp's original 21 NP-complete problems and serves as a key source for reductions to other graph problems like INDEPENDENT SET and VERTEX COVER.

## Problem Definition

Let $G = (V, E)$ be an undirected graph.  A **clique** in $G$ is a subset $S \subseteq V$ such that every pair of vertices in $S$ is connected by an edge:

$$
\forall\, u, v \in S,\; u \neq v \implies \{u, v\} \in E
$$

A clique of size $k$ is called a **$k$-clique**.

**CLIQUE (decision problem):**

- **Input:** An undirected graph $G = (V, E)$ and a positive integer $k$.
- **Question:** Does $G$ contain a clique of size $\geq k$?

**MAX-CLIQUE (optimization problem):**

- **Input:** An undirected graph $G = (V, E)$.
- **Output:** The size of the largest clique in $G$, denoted $\omega(G)$.

## CLIQUE is in NP

A certificate for a yes-instance is a set $S \subseteq V$ with $|S| \geq k$.  Verification checks:

1. $|S| \geq k$ -- $O(1)$.
2. Every pair in $S$ is an edge -- $O(k^2) \subseteq O(n^2)$.

Total verification time: $O(n^2)$, which is polynomial.

## NP-Completeness: 3-SAT Reduces to CLIQUE

**Theorem.** CLIQUE is NP-complete.

*Proof.* We reduce 3-SAT to CLIQUE.  Given a 3-CNF formula $\phi = C_1 \wedge C_2 \wedge \cdots \wedge C_m$ with $m$ clauses over $n$ variables, construct a graph $G$ as follows.

**Construction.** For each clause $C_j = (\ell_{j,1} \vee \ell_{j,2} \vee \ell_{j,3})$, create three vertices, one per literal.  Label vertex $(j, r)$ with literal $\ell_{j,r}$.

Add an edge between $(j, r)$ and $(j', r')$ if and only if:

1. $j \neq j'$ (vertices are from different clauses), and
2. $\ell_{j,r}$ and $\ell_{j',r'}$ are not complementary (i.e., $\ell_{j,r} \neq \neg \ell_{j',r'}$).

Set $k = m$ (the number of clauses).

**Forward direction ($\phi$ satisfiable $\Rightarrow$ $G$ has $m$-clique).** Given a satisfying assignment, pick one true literal from each clause.  The corresponding $m$ vertices form a clique: they come from different clauses (condition 1), and since the assignment is consistent, no two selected literals are complementary (condition 2).

**Reverse direction ($G$ has $m$-clique $\Rightarrow$ $\phi$ satisfiable).** An $m$-clique must contain exactly one vertex from each clause (since vertices within the same clause are not connected).  The selected literals are pairwise non-complementary, so they can be consistently assigned true.  This assignment satisfies at least one literal per clause, satisfying $\phi$.

**Efficiency.** The graph has $3m$ vertices and at most $O(m^2)$ edges.  The construction runs in polynomial time.

$\square$

??? example "Worked example"
    Consider $\phi = (x_1 \vee \neg x_2 \vee x_3) \wedge (\neg x_1 \vee x_2 \vee x_3) \wedge (x_1 \vee x_2 \vee \neg x_3)$.

    The graph has 9 vertices (3 per clause).  Edges connect vertices from different clauses whose literals are compatible.  For instance, $x_1$ from clause 1 connects to $x_2$ and $x_3$ from clause 2 (but not $\neg x_1$).

    With $k = 3$, we seek a 3-clique.  The assignment $x_1 = 1, x_2 = 1, x_3 = 1$ satisfies all clauses, and selecting $x_1$ from clause 1, $x_2$ from clause 2, and $x_1$ from clause 3 gives a 3-clique.

## Relationship to Other Problems

CLIQUE connects to several other NP-complete graph problems through simple reductions:

### CLIQUE and INDEPENDENT SET

$S$ is a clique in $G$ if and only if $S$ is an independent set in the complement graph $\overline{G}$.  Therefore:

$$
\text{CLIQUE} \leq_p \text{INDEPENDENT SET}
$$

The reduction simply computes the complement graph: $(G, k) \mapsto (\overline{G}, k)$.

### CLIQUE and VERTEX COVER

$S$ is an independent set in $G$ if and only if $V \setminus S$ is a vertex cover in $G$.  Combined with the above:

$$
\text{CLIQUE} \leq_p \text{INDEPENDENT SET} \leq_p \text{VERTEX COVER}
$$

## Inapproximability

CLIQUE is not only NP-hard to solve exactly but also extremely hard to approximate.

**Theorem (Hastad, 1999; Zuckerman, 2007).** Unless $\mathbf{P} = \mathbf{NP}$, there is no polynomial-time algorithm that approximates MAX-CLIQUE within a factor of $n^{1-\epsilon}$ for any $\epsilon > 0$.

This makes CLIQUE one of the hardest NP-hard problems to approximate -- in contrast to problems like VERTEX COVER, which admits a 2-approximation.

## Special Cases

While CLIQUE is NP-complete in general, it can be solved efficiently on restricted graph classes:

| Graph Class | Complexity | Algorithm |
|-------------|-----------|-----------|
| Perfect graphs | Polynomial | Semidefinite programming |
| Chordal graphs | Polynomial | Perfect elimination ordering |
| Interval graphs | Polynomial | Greedy on sorted intervals |
| Planar graphs | Polynomial | Clique size $\leq 4$ |
| General graphs | NP-complete | Exhaustive search |

## Reference

- Karp, R. M. "Reducibility Among Combinatorial Problems." 1972.
- Sipser, M. *Introduction to the Theory of Computation*. Cengage Learning.
- Arora, S. and Barak, B. *Computational Complexity: A Modern Approach*. Cambridge University Press.
