# Cycle Property

The cut property tells us which edges *must* appear in an MST. The cycle property provides the complementary perspective: it tells us which edges can *never* appear. Together, these two properties form the theoretical backbone of every MST algorithm. The cycle property justifies, for instance, why Kruskal's algorithm safely skips an edge that would create a cycle -- that edge is always the heaviest in the cycle and therefore dispensable.

## Intuition

Consider adding an edge $e$ to an MST $T$. This creates exactly one cycle (since $T$ is a tree). If $e$ is the heaviest edge in that cycle, replacing any other cycle edge with $e$ would only increase the total weight. Therefore $e$ has no reason to be in any MST. The cycle property formalizes this reasoning.

## Theorem (Cycle Property)

Let $G = (V, E)$ be a connected, undirected graph with weight function $w : E \to \mathbb{R}$. Let $C$ be any cycle in $G$, and let $e$ be the unique heaviest edge in $C$ (i.e., $w(e) > w(e')$ for all other edges $e' \in C$). Then $e$ does not belong to any MST of $G$.

??? warning "Strict inequality is essential"
    The uniqueness condition matters. If two edges in a cycle share the maximum weight, both *could* appear in different MSTs. The cycle property applies only when the heaviest edge is strictly heavier than every other edge in the cycle.

## Proof

Suppose for contradiction that $e = (u, v)$ is the unique heaviest edge in cycle $C$ and that $e$ belongs to some MST $T$.

Removing $e$ from $T$ splits $T$ into two connected components $T_u$ (containing $u$) and $T_v$ (containing $v$). Since $C$ is a cycle containing $e$, there exists a path from $u$ to $v$ in $C$ that does not use $e$. This path must cross from $T_u$ to $T_v$ at some edge $e' \ne e$, where $e' \in C$.

Define

$$
T' = T \setminus \{e\} \cup \{e'\}
$$

We verify that $T'$ is a spanning tree:

- Removing $e$ disconnects $T$ into $T_u$ and $T_v$.
- Adding $e'$ reconnects them because $e'$ has one endpoint in $T_u$ and the other in $T_v$.
- $T'$ has $|V| - 1$ edges, is connected, and is acyclic, so it is a spanning tree.

Since $e$ is the unique heaviest edge in $C$ and $e' \in C$ with $e' \ne e$, we have $w(e') < w(e)$. Therefore

$$
w(T') = w(T) - w(e) + w(e') < w(T)
$$

This contradicts the assumption that $T$ is a minimum spanning tree. $\square$

## Example

Consider a graph on vertices $\{A, B, C, D\}$ with edges:

| Edge | Weight |
|------|--------|
| (A, B) | 4 |
| (A, C) | 1 |
| (B, C) | 3 |
| (B, D) | 2 |
| (C, D) | 5 |

The cycle $A \to B \to C \to A$ contains edges $(A, B)$ with weight 4, $(B, C)$ with weight 3, and $(A, C)$ with weight 1. The unique heaviest edge is $(A, B)$ with weight 4, so by the cycle property, $(A, B)$ cannot belong to any MST.

Similarly, the cycle $B \to C \to D \to B$ contains edges $(B, C)$ with weight 3, $(C, D)$ with weight 5, and $(B, D)$ with weight 2. The unique heaviest edge is $(C, D)$ with weight 5, so $(C, D)$ cannot belong to any MST.

The remaining edges $\{(A, C), (B, C), (B, D)\}$ with total weight $1 + 3 + 2 = 6$ form the unique MST.

## Duality with the Cut Property

The cut and cycle properties are dual perspectives on the same underlying structure:

| Property | Statement | Algorithm action |
|----------|-----------|------------------|
| **Cut** | The lightest edge crossing a cut respecting $A$ is safe to **include** | Add the edge |
| **Cycle** | The unique heaviest edge in a cycle is safe to **exclude** | Skip the edge |

The cut property builds the MST by inclusion (adding safe edges), while the cycle property builds it by exclusion (removing unsafe edges). Kruskal's algorithm uses both: it includes a light edge when it connects two components (cut property) and skips an edge when both endpoints are in the same component (cycle property, since that edge would complete a cycle and be the heaviest edge considered so far).

## Corollary: Red-Blue Meta-Algorithm

The duality leads to a unified framework sometimes called the **red-blue algorithm**:

- **Blue rule** (cut property): find a cut with no blue edge crossing it; color the lightest crossing edge blue.
- **Red rule** (cycle property): find a cycle with no red edge; color the heaviest edge red.

Repeat until every edge is colored. The blue edges form an MST. This framework generalizes Kruskal's, Prim's, and Boruvka's algorithms as special cases of how cuts and cycles are selected.

## Reference

- [Introduction to Algorithms (CLRS), Chapter 23](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Tarjan, R. E. (1983). *Data Structures and Network Algorithms*, Chapter 6.
