# Cut Property

Every MST algorithm -- Kruskal's, Prim's, and Boruvka's -- adds edges one at a time and claims the result is optimal. But why does greedily picking light edges produce a global minimum? The answer rests on the cut property, which guarantees that certain locally optimal choices remain globally safe. This page defines cuts formally, states the cut property as a theorem, and proves it using an exchange argument.

## Cuts and Crossing Edges

Let $G = (V, E)$ be a connected, undirected graph with weight function $w : E \to \mathbb{R}$.

A **cut** $(S, V \setminus S)$ is a partition of the vertex set $V$ into two nonempty subsets $S$ and $V \setminus S$.

An edge $(u, v) \in E$ **crosses** the cut $(S, V \setminus S)$ if one endpoint lies in $S$ and the other in $V \setminus S$. The set of all crossing edges is sometimes called the **cut-set**.

A cut **respects** a set of edges $A \subseteq E$ if no edge in $A$ crosses the cut.

A **light edge** crossing a cut is an edge of minimum weight among all edges crossing that cut. If weights are distinct, the light edge is unique; otherwise there may be ties.

## Safe Edges and the Generic MST Algorithm

Before stating the cut property, consider the generic strategy shared by all MST algorithms. The algorithm maintains a set $A$ of edges that is always a subset of some MST, and repeatedly adds a **safe edge** -- an edge $(u, v)$ such that $A \cup \{(u, v)\}$ is still a subset of some MST.

```
GENERIC-MST(G, w):
    A = {}
    while A does not form a spanning tree:
        find a safe edge (u, v) for A
        A = A ∪ {(u, v)}
    return A
```

The cut property tells us exactly how to identify safe edges.

## Theorem (Cut Property)

Let $G = (V, E)$ be a connected, undirected graph with weight function $w : E \to \mathbb{R}$. Let $A \subseteq E$ be a subset of some MST of $G$, and let $(S, V \setminus S)$ be any cut that respects $A$. If $(u, v)$ is a light edge crossing $(S, V \setminus S)$, then $(u, v)$ is safe for $A$.

In other words: the lightest edge crossing any cut that respects the current partial MST can always be added without violating optimality.

## Proof

Let $T$ be an MST such that $A \subseteq T$. There are two cases.

**Case 1**: $(u, v) \in T$. Then $A \cup \{(u, v)\} \subseteq T$, so $(u, v)$ is trivially safe.

**Case 2**: $(u, v) \notin T$. Adding $(u, v)$ to $T$ creates a unique cycle $C$ in $T \cup \{(u, v)\}$. Since $u \in S$ and $v \in V \setminus S$, the cycle $C$ must cross the cut $(S, V \setminus S)$ at least twice. Therefore, there exists another edge $(x, y) \in T$ on $C$ that also crosses the cut.

Since the cut respects $A$, the edge $(x, y) \notin A$. Now define

$$
T' = T \setminus \{(x, y)\} \cup \{(u, v)\}
$$

We verify that $T'$ is a spanning tree:

- Removing $(x, y)$ from $T$ disconnects $T$ into two components.
- Adding $(u, v)$ reconnects them because $(u, v)$ also crosses the same cut.
- $T'$ has $|V| - 1$ edges, is connected, and is therefore a spanning tree.

Since $(u, v)$ is a light edge crossing the cut, $w(u, v) \le w(x, y)$, so

$$
w(T') = w(T) - w(x, y) + w(u, v) \le w(T)
$$

Since $T$ is an MST, $w(T') \ge w(T)$, which forces $w(T') = w(T)$. Thus $T'$ is also an MST. Moreover, $A \cup \{(u, v)\} \subseteq T'$, so $(u, v)$ is safe for $A$. $\square$

## Example

Consider a graph on vertices $\{A, B, C, D\}$ with edges and weights:

| Edge | Weight |
|------|--------|
| (A, B) | 4 |
| (A, C) | 1 |
| (B, C) | 3 |
| (B, D) | 2 |
| (C, D) | 5 |

Suppose the current edge set is $A = \emptyset$. Choose the cut $S = \{A\}$, $V \setminus S = \{B, C, D\}$. This cut respects $A$ (vacuously, since $A$ is empty). The edges crossing this cut are $(A, B)$ with weight 4 and $(A, C)$ with weight 1. The light edge is $(A, C)$, so the cut property guarantees that $(A, C)$ is safe to add.

After adding $(A, C)$, set $A = \{(A, C)\}$. Now choose the cut $S = \{A, C\}$, $V \setminus S = \{B, D\}$. This cut respects $A$ because $(A, C)$ has both endpoints in $S$. The crossing edges are $(A, B)$ with weight 4, $(B, C)$ with weight 3, and $(C, D)$ with weight 5. The light edge is $(B, C)$ with weight 3.

Continuing this process builds up the MST edge by edge, each choice justified by the cut property.

## How Algorithms Use the Cut Property

Each MST algorithm instantiates the generic strategy by choosing cuts in a specific way:

- **Prim's algorithm**: maintains a growing tree rooted at a starting vertex. The cut $(S, V \setminus S)$ separates vertices already in the tree from those not yet included.
- **Kruskal's algorithm**: maintains a forest of components. For the lightest unprocessed edge $(u, v)$, if $u$ and $v$ are in different components, the cut separating those two components respects $A$, and $(u, v)$ is the light edge crossing it.
- **Boruvka's algorithm**: each component independently selects its lightest outgoing edge, applying the cut property in parallel.

## Reference

- [Introduction to Algorithms (CLRS), Chapter 23](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Cormen, T. H. et al. *Introduction to Algorithms*, 4th ed., Theorem 23.1.
