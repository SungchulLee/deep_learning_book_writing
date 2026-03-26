# Vertex Cover

In many network problems we need to select a small set of nodes that
"monitors" every connection.  The Vertex Cover problem formalizes this idea
and stands as one of Karp's original 21 NP-complete problems, with direct
ties to Independent Set and graph matching.

## Problem Definition

Given an undirected graph $G = (V, E)$ and an integer $k$, the **Vertex
Cover** decision problem asks:

> Does there exist a subset $V' \subseteq V$ with $|V'| \le k$ such that
> every edge $(u, v) \in E$ has at least one endpoint in $V'$?

Such a set $V'$ is called a **vertex cover** of size $k$.

!!! example "Concrete Instance"
    Consider the graph with $V = \{a, b, c, d\}$ and
    $E = \{(a,b), (b,c), (c,d), (a,d)\}$ (a 4-cycle).  The set $\{b, d\}$
    covers every edge, so vertex cover of size $k = 2$ exists.

## Membership in NP

A certificate is the subset $V'$.  A polynomial-time verifier:

1. Checks $|V'| \le k$.
2. For every edge $(u, v) \in E$, checks that $u \in V'$ or $v \in V'$.

Both steps run in $O(|V| + |E|)$ time, so Vertex Cover $\in$ NP.

## NP-Completeness via Reduction from 3-SAT

We show **3-SAT** $\le_p$ **Vertex Cover**.

### Gadget Construction

Given a 3-SAT formula $\phi$ with $n$ variables and $m$ clauses:

1. **Variable gadgets.**  For each variable $x_i$, create two vertices
   $x_i$ and $\overline{x_i}$ connected by an edge.  At least one of these
   must be in any vertex cover (to cover the connecting edge).

2. **Clause gadgets.**  For each clause $C_j = (\ell_{j1} \vee \ell_{j2}
   \vee \ell_{j3})$, create a triangle (three vertices $c_{j1}, c_{j2},
   c_{j3}$ with edges forming a 3-clique).  Covering a triangle requires
   at least two of its three vertices.

3. **Connecting edges.**  Connect each clause vertex $c_{jk}$ to the
   literal vertex corresponding to $\ell_{jk}$.

Set

$$
k = n + 2m
$$

### Correctness

- **Satisfying assignment $\Rightarrow$ vertex cover of size $k$.**  For
  each variable, include the true literal's vertex (covers the variable
  edge).  For each clause, at least one literal is true, meaning the
  connecting edge from that clause vertex is already covered by the literal
  vertex.  Include the other two clause vertices (covering the triangle
  edges and remaining connecting edges).  Total: $n + 2m = k$.

- **Vertex cover of size $k$ $\Rightarrow$ satisfying assignment.**  Each
  variable edge forces at least one literal vertex into the cover ($\ge n$).
  Each triangle forces at least two clause vertices ($\ge 2m$).  With
  budget $k = n + 2m$, exactly one literal per variable and exactly two per
  clause are chosen.  The unchosen clause vertex must have its connecting
  edge covered by a literal vertex, meaning that literal is in the cover
  and thus set to true---satisfying the clause.

The construction uses $O(n + m)$ vertices and edges, so it runs in
polynomial time.

## Relationship to Independent Set

A set $V' \subseteq V$ is a vertex cover if and only if $V \setminus V'$ is
an **independent set** (no two vertices in $V \setminus V'$ share an edge).

$$
\text{Vertex Cover of size } k \iff \text{Independent Set of size } n - k
$$

This complementarity immediately proves Independent Set NP-complete given
Vertex Cover is NP-complete.

## Approximation Algorithm

A simple greedy algorithm achieves a **2-approximation**:

```python
"""
2-approximation algorithm for Minimum Vertex Cover.

Time : O(V + E)
Space: O(V)
"""


# === Approximate Vertex Cover ===
def approx_vertex_cover(
    n: int, edges: list[tuple[int, int]]
) -> set[int]:
    """Return a vertex cover at most twice the optimal size."""
    cover: set[int] = set()
    remaining = set(range(len(edges)))

    for idx in list(remaining):
        if idx not in remaining:
            continue
        u, v = edges[idx]
        if u in cover or v in cover:
            remaining.discard(idx)
            continue
        cover.add(u)
        cover.add(v)
        remaining.discard(idx)

    return cover


# === Example ===
if __name__ == "__main__":
    edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
    cover = approx_vertex_cover(4, edges)
    print(f"Vertex cover: {cover}")
    print(f"Size: {len(cover)}")
```

**Why factor 2?**  Each edge chosen adds two vertices, but the optimal
solution must include at least one endpoint of each chosen edge.  The number
of chosen edges is a lower bound on OPT, so the algorithm returns at most
$2 \cdot \text{OPT}$ vertices.

!!! tip "Inapproximability"
    Under the Unique Games Conjecture, Vertex Cover cannot be approximated
    within a factor better than $2 - \epsilon$ for any $\epsilon > 0$ in
    polynomial time.

## Parameterized Tractability

Vertex Cover is **fixed-parameter tractable** (FPT) with parameter $k$.
A bounded search tree algorithm runs in $O(2^k \cdot n)$ time: for each
uncovered edge, branch on which endpoint to include.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction
  to Algorithms* (CLRS), Chapter 34.
- Sipser, M. *Introduction to the Theory of Computation*. Cengage Learning.
