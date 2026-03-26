# Hamiltonian Cycle

A **Hamiltonian cycle** visits every vertex in a graph exactly once and returns to the starting vertex. Unlike Euler circuits (which visit every *edge* once and are decidable in polynomial time), determining whether a Hamiltonian cycle exists is NP-complete. This contrast highlights how similar-sounding graph problems can differ dramatically in computational difficulty.

## Problem Definition

!!! tip "Definition: Hamiltonian Path and Cycle"
    Given an undirected graph $G = (V, E)$:

    - A **Hamiltonian path** is a path that visits every vertex exactly once.
    - A **Hamiltonian cycle** is a cycle that visits every vertex exactly once (returning to the start).

    The **Hamiltonian Cycle (HAM-CYCLE)** problem asks: does $G$ contain a Hamiltonian cycle?

The directed versions are defined analogously on directed graphs.

## NP-Completeness

!!! tip "Theorem"
    HAM-CYCLE is NP-complete.

**Membership in NP.** A Hamiltonian cycle is a sequence of $n$ vertices. Verifying that it forms a valid cycle (all vertices appear exactly once, all consecutive pairs are edges) takes $O(n)$ time.

**NP-Hardness.** We sketch the reduction from **Vertex Cover** to HAM-CYCLE, following the approach in CLRS.

### Reduction from Vertex Cover

Given a Vertex Cover instance $(G, k)$ with graph $G = (V, E)$, $|V| = n$, $|E| = m$:

1. **Edge gadgets.** For each edge $e = (u, v) \in E$, create a gadget with 12 vertices arranged so that any Hamiltonian cycle must traverse the gadget in a way that "covers" $e$ from either $u$'s side or $v$'s side (or both).

2. **Vertex chains.** For each vertex $u$, link its edge gadgets into a chain. The chain can be traversed in one direction or the other.

3. **Selector vertices.** Add $k$ selector vertices $s_1, \ldots, s_k$. Each selector connects to the start and end of every vertex chain.

4. **Hamiltonian cycle exists $\Leftrightarrow$ vertex cover of size $k$ exists.** A Hamiltonian cycle chooses $k$ vertex chains (one per selector), covering all edge gadgets. This corresponds to a vertex cover.

The construction is polynomial in $|V|$ and $|E|$, completing the reduction. $\square$

## Hamiltonian Path

The **Hamiltonian Path (HAM-PATH)** problem is also NP-complete. It reduces from HAM-CYCLE: given graph $G$, pick any vertex $v$, split it into $v_{\text{in}}$ and $v_{\text{out}}$, and ask for a Hamiltonian path from $v_{\text{out}}$ to $v_{\text{in}}$.

## Relationship to TSP

HAM-CYCLE reduces to the decision version of TSP: given a graph $G$, create a complete weighted graph where $w(u,v) = 1$ if $(u,v) \in E$ and $w(u,v) = 2$ otherwise. A Hamiltonian cycle in $G$ exists if and only if the TSP tour of cost $n$ exists.

This reduction also proves that general TSP has no finite approximation ratio (unless P = NP).

## Sufficient Conditions

While deciding HAM-CYCLE is hard in general, several sufficient conditions guarantee existence:

!!! tip "Dirac's Theorem (1952)"
    If $G$ has $n \geq 3$ vertices and every vertex has degree at least $n/2$, then $G$ has a Hamiltonian cycle.

!!! tip "Ore's Theorem (1960)"
    If $G$ has $n \geq 3$ vertices and for every pair of non-adjacent vertices $u, v$: $\deg(u) + \deg(v) \geq n$, then $G$ has a Hamiltonian cycle.

Ore's theorem generalizes Dirac's theorem (Dirac is the special case where each individual degree is $\geq n/2$).

## Special Graph Classes

| Graph Class | HAM-CYCLE Status | Notes |
|-------------|-----------------|-------|
| Complete graph $K_n$ ($n \geq 3$) | Always exists | $(n-1)!/2$ distinct cycles |
| Complete bipartite $K_{n,n}$ | Always exists | Requires equal parts |
| Hypercube $Q_n$ | Always exists | Gray code traversal |
| Petersen graph | Does not exist | Classic counterexample |
| Planar graphs | NP-complete | Remains hard |
| Grid graphs | NP-complete | Even for subgrids |

## Algorithms

### Exact Algorithms

| Algorithm | Time | Space | Notes |
|-----------|------|-------|-------|
| Brute force | $O(n!)$ | $O(n)$ | Try all permutations |
| Held-Karp DP | $O(2^n \cdot n^2)$ | $O(2^n \cdot n)$ | Bitmask DP |
| Inclusion-exclusion | $O(2^n \cdot n^2)$ | $O(n^2)$ | Polynomial space |

### The Held-Karp Approach

Define $\text{dp}[S][v]$ = whether there exists a Hamiltonian path visiting exactly the vertices in $S$ and ending at $v$.

$$
\text{dp}[S][v] = \bigvee_{u \in S \setminus \{v\},\; (u,v) \in E} \text{dp}[S \setminus \{v\}][u]
$$

A Hamiltonian cycle exists if $\text{dp}[V][v] = \text{true}$ for some $v$ adjacent to the start vertex.

??? example "Example: Checking Hamiltonian Cycle"
    **Graph:** $V = \{1, 2, 3, 4\}$, edges $\{(1,2), (2,3), (3,4), (4,1), (1,3)\}$.

    **Candidate cycle:** $1 \to 2 \to 3 \to 4 \to 1$.

    - $(1,2) \in E$? Yes.
    - $(2,3) \in E$? Yes.
    - $(3,4) \in E$? Yes.
    - $(4,1) \in E$? Yes.
    - All 4 vertices visited exactly once? Yes.

    **Valid Hamiltonian cycle.** Another cycle: $1 \to 3 \to 2 \to 4 \to 1$ --- check $(2,4)$: not in $E$. Invalid.

    Note: the graph has edge $(1,3)$ creating a shortcut, but the only valid Hamiltonian cycle uses the 4-cycle $1 \to 2 \to 3 \to 4 \to 1$.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press, Chapter 34.
- Sipser, M. (2012). *Introduction to the Theory of Computation* (3rd ed.). Cengage Learning.
- Garey, M. R., & Johnson, D. S. (1979). *Computers and Intractability*. W. H. Freeman.
