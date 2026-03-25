# Handshaking Lemma

The Handshaking Lemma is often the first theorem encountered in graph theory, and it remains one of the most frequently used. The name comes from a party analogy: if every person at a gathering shakes hands with some others, the total number of "handshake endpoints" is always even, because each handshake involves exactly two hands. This simple counting argument has surprisingly powerful consequences, from proving that certain graphs cannot exist to analyzing network [degree](degree.md) distributions.

## Statement and Proof

!!! tip "Theorem: Handshaking Lemma"
    For any undirected graph $G = (V, E)$,

$$
\sum_{v \in V} \deg(v) = 2|E|
$$

**Proof.** Consider the sum $\sum_{v \in V} \deg(v)$. Each edge $\{u, w\} \in E$ contributes exactly 1 to $\deg(u)$ and exactly 1 to $\deg(w)$. No other vertex has its degree affected by this edge. Therefore, each edge contributes exactly 2 to the total sum, and the sum over all edges gives $2|E|$. $\square$

!!! example "Verifying the Lemma"
    Consider a graph on $\{a, b, c, d\}$ with edges $\{a,b\}, \{b,c\}, \{c,d\}, \{a,d\}, \{a,c\}$. The degrees are $\deg(a)=3$, $\deg(b)=2$, $\deg(c)=3$, $\deg(d)=2$. The sum is $3+2+3+2=10=2 \times 5 = 2|E|$.

## Corollary: Even Number of Odd-Degree Vertices

!!! tip "Corollary"
    In any undirected graph, the number of vertices with odd degree is even.

**Proof.** Let $V_{\text{odd}} = \{v \in V : \deg(v) \text{ is odd}\}$ and $V_{\text{even}} = V \setminus V_{\text{odd}}$. Then

$$
\sum_{v \in V_{\text{odd}}} \deg(v) = 2|E| - \sum_{v \in V_{\text{even}}} \deg(v)
$$

The right side is even (difference of two even numbers). Since each term on the left is odd, the number of terms $|V_{\text{odd}}|$ must be even. $\square$

This corollary immediately tells us, for instance, that there is no graph where exactly 3 vertices have odd degree.

## Directed Graph Analog

For directed graphs, each directed edge $(u, v)$ contributes 1 to $\deg^+(u)$ and 1 to $\deg^-(v)$. Summing over all vertices:

$$
\sum_{v \in V} \deg^+(v) = \sum_{v \in V} \deg^-(v) = |E|
$$

This is the directed version of the handshaking lemma. The total out-degree equals the total in-degree, and both equal the number of edges.

## Applications

The Handshaking Lemma serves as a proof tool in many contexts.

### Existence Arguments

To show that a graph with certain degree constraints cannot exist, check whether $\sum \deg(v)$ would be odd:

- **Claim:** There is no graph on 5 vertices where every vertex has degree 3.
- **Check:** $\sum \deg(v) = 5 \times 3 = 15$, which is odd. By the lemma, $2|E|$ must be even. Contradiction.

### Counting Edges

Given only the degree sequence, the lemma immediately yields the edge count:

$$
|E| = \frac{1}{2}\sum_{v \in V} \deg(v)
$$

### Average Degree

The **average degree** of a graph is

$$
\bar{d} = \frac{1}{|V|}\sum_{v \in V} \deg(v) = \frac{2|E|}{|V|}
$$

This relationship is fundamental in the analysis of random graphs and network models.

## Verification Code

```python
"""
Verification of the Handshaking Lemma on undirected and directed graphs.

Computes degree sums and verifies they equal twice the edge count
(undirected) or the edge count (directed).
"""


# === Undirected Verification ===

def verify_handshaking_undirected(adj, n, num_edges):
    """Verify that sum of degrees equals 2 * number of edges."""
    degree_sum = sum(len(adj[v]) for v in range(n))
    holds = (degree_sum == 2 * num_edges)
    return degree_sum, holds


# === Directed Verification ===

def verify_handshaking_directed(adj, n, num_edges):
    """Verify that sum of out-degrees equals number of edges."""
    out_degree_sum = sum(len(adj[v]) for v in range(n))
    in_degrees = [0] * n
    for u in range(n):
        for v in adj[u]:
            in_degrees[v] += 1
    in_degree_sum = sum(in_degrees)
    holds = (out_degree_sum == num_edges == in_degree_sum)
    return out_degree_sum, in_degree_sum, holds


# === Odd-Degree Count ===

def count_odd_degree_vertices(adj, n):
    """Count vertices with odd degree and verify the count is even."""
    odd_count = sum(1 for v in range(n) if len(adj[v]) % 2 == 1)
    return odd_count, odd_count % 2 == 0


# === Main ===

if __name__ == "__main__":
    # Undirected graph: 5 edges
    adj_u = [[1, 2, 3], [0, 2], [0, 1, 3], [0, 2]]
    deg_sum, ok = verify_handshaking_undirected(adj_u, 4, 5)
    print(f"Undirected: sum(deg) = {deg_sum}, 2*|E| = 10, holds: {ok}")

    odd_count, even_check = count_odd_degree_vertices(adj_u, 4)
    print(f"Odd-degree vertices: {odd_count}, count is even: {even_check}")

    # Directed graph: 4 edges
    adj_d = [[1, 2], [2], [0], []]
    out_s, in_s, ok_d = verify_handshaking_directed(adj_d, 4, 3)
    print(f"\nDirected: sum(out-deg) = {out_s}, sum(in-deg) = {in_s}, "
          f"|E| = 3, holds: {ok_d}")
```

**Output:**
```
Undirected: sum(deg) = 10, 2*|E| = 10, holds: True
Odd-degree vertices: 2, count is even: True
Directed: sum(out-deg) = 3, sum(in-deg) = 3, |E| = 3, holds: True
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 22.
- West, D. B. (2001). *Introduction to Graph Theory* (2nd ed.). Prentice Hall. Proposition 1.3.3.
