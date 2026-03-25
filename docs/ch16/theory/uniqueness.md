# MST Uniqueness

Do different MST algorithms always produce the same tree? If a graph has multiple spanning trees of the same minimum weight, does it matter which one an algorithm returns? Understanding when the MST is unique -- and when it is not -- clarifies what we can expect from any MST algorithm and simplifies correctness arguments.

## Uniqueness Theorem

**Theorem.** If all edge weights in a connected, undirected graph $G = (V, E)$ are distinct, then $G$ has exactly one minimum spanning tree.

## Proof

Suppose for contradiction that $G$ has two distinct MSTs $T_1$ and $T_2$ with all edge weights distinct. Since $T_1 \ne T_2$, there exists at least one edge in $T_1 \setminus T_2$. Let $e = (u, v)$ be the edge of minimum weight in the symmetric difference $T_1 \triangle T_2 = (T_1 \setminus T_2) \cup (T_2 \setminus T_1)$.

Without loss of generality, assume $e \in T_1 \setminus T_2$. Adding $e$ to $T_2$ creates a unique cycle $C$ in $T_2 \cup \{e\}$. Since $T_1$ is a tree (acyclic), not all edges of $C$ belong to $T_1$, so there exists an edge $e' \in C$ with $e' \in T_2 \setminus T_1$.

Since $e' \in T_2 \setminus T_1 \subseteq T_1 \triangle T_2$ and $e$ has minimum weight in $T_1 \triangle T_2$, we have

$$
w(e) \le w(e')
$$

Because all edge weights are distinct, if $w(e) = w(e')$ then $e = e'$, which contradicts $e \in T_1 \setminus T_2$ and $e' \in T_2 \setminus T_1$. Therefore $w(e) < w(e')$.

Now consider

$$
T_2' = T_2 \setminus \{e'\} \cup \{e\}
$$

This is a spanning tree (removing $e'$ from the cycle $C$ in $T_2 \cup \{e\}$ keeps connectivity) with weight

$$
w(T_2') = w(T_2) - w(e') + w(e) < w(T_2)
$$

This contradicts $T_2$ being an MST. $\square$

## Alternative Proof via the Cut Property

A shorter proof follows directly from the cut property. In a graph with distinct edge weights, for every cut $(S, V \setminus S)$ there is a unique lightest crossing edge. The cut property forces this edge into every MST. Since each edge's inclusion or exclusion is uniquely determined, all MSTs must be identical.

## When Uniqueness Fails

If edge weights are not distinct, multiple MSTs can exist. They will always share the same total weight, but may differ in which edges they include.

**Example.** Consider a triangle with vertices $\{A, B, C\}$ and edges:

| Edge | Weight |
|------|--------|
| (A, B) | 1 |
| (B, C) | 1 |
| (A, C) | 1 |

Every pair of edges forms a spanning tree with total weight 2. There are three distinct MSTs:

- $\{(A, B), (B, C)\}$
- $\{(A, B), (A, C)\}$
- $\{(B, C), (A, C)\}$

All three have the same total weight, but different edge sets.

## MST Weight is Always Unique

Even when multiple MSTs exist, an important invariant holds.

**Theorem.** All MSTs of a graph $G$ have the same total weight.

This follows immediately from the definition: if two spanning trees $T_1$ and $T_2$ are both minimum, then $w(T_1) \le w(T_2)$ and $w(T_2) \le w(T_1)$, so $w(T_1) = w(T_2)$.

??? note "Stronger result: edge-weight multiset"
    In fact, all MSTs share the same multiset of edge weights -- not just the same total. This stronger result means that the sorted sequence of edge weights is identical across all MSTs. The proof uses the matroid intersection theory or can be shown by an exchange argument on weight classes.

## Practical Implications

1. **Algorithm independence**: when edge weights are distinct, Kruskal's, Prim's, and Boruvka's algorithms all produce the same tree, regardless of tie-breaking rules.
2. **Perturbation for uniqueness**: if uniqueness is desired but weights have ties, a standard technique adds infinitesimal perturbations (e.g., $w'(e_i) = w(e_i) + i \cdot \epsilon$ for a sufficiently small $\epsilon$) to break all ties without changing the relative order of edges with distinct weights.
3. **Verification**: to check whether a given spanning tree is the unique MST, verify that every non-tree edge is the unique heaviest edge in the cycle it creates with the tree. If any non-tree edge ties with a tree edge in its cycle, another MST exists.

## Reference

- [Introduction to Algorithms (CLRS), Chapter 23](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Whitney, H. (1935). On the abstract properties of linear dependence. *American Journal of Mathematics*, 57(3), 509--533.
