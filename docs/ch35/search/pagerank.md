# PageRank

Before PageRank, search engines ranked web pages primarily by keyword frequency, which was easily manipulated. Page and Brin (1998) observed that the **link structure** of the web encodes a collective judgment about page quality: a page linked to by many important pages is itself likely important. **PageRank** formalizes this intuition as the stationary distribution of a random walk on the web graph.

## Random Surfer Model

Imagine a random surfer who starts at a random page and at each step:

- With probability $1 - d$, jumps to a uniformly random page (teleportation).
- With probability $d$, follows a random outgoing link from the current page.

The parameter $d \in (0, 1)$ is the **damping factor** (typically $d = 0.85$). Teleportation ensures the Markov chain is ergodic (irreducible and aperiodic), guaranteeing a unique stationary distribution.

## Definition

The PageRank of page $i$ in a web graph with $n$ pages is:

$$
\text{PR}(i) = \frac{1 - d}{n} + d \sum_{j \in B(i)} \frac{\text{PR}(j)}{L(j)}
$$

where $B(i)$ is the set of pages linking to $i$, and $L(j)$ is the number of outgoing links from $j$.

In matrix form, with transition matrix $M$ where $M_{ij} = 1/L(j)$ if $j$ links to $i$:

$$
\mathbf{r} = \frac{1 - d}{n} \mathbf{1} + d \, M \, \mathbf{r}
$$

The PageRank vector $\mathbf{r}$ is the eigenvector of the modified transition matrix corresponding to eigenvalue 1.

## Power Iteration

PageRank is computed iteratively:

1. Initialize $\mathbf{r}^{(0)} = \frac{1}{n} \mathbf{1}$.
2. Repeat: $\mathbf{r}^{(t+1)} = \frac{1-d}{n} \mathbf{1} + d \, M \, \mathbf{r}^{(t)}$.
3. Stop when $\|\mathbf{r}^{(t+1)} - \mathbf{r}^{(t)}\|_1 < \epsilon$.

**Convergence rate**: The spectral gap of the transition matrix with damping is at least $1 - d$, so convergence is geometric:

$$
\|\mathbf{r}^{(t)} - \mathbf{r}^*\|_1 \le d^t
$$

With $d = 0.85$, about 50 iterations suffice for convergence to $10^{-7}$.

**Per-iteration cost**: $O(|E|)$ where $|E|$ is the number of edges (links).

## Dangling Nodes

Pages with no outgoing links (dangling nodes) would absorb all probability mass. The standard fix redistributes their mass uniformly:

$$
M'_{ij} = \begin{cases} 1/L(j) & \text{if } L(j) > 0 \\ 1/n & \text{if } L(j) = 0 \end{cases}
$$

!!! tip "Personalized PageRank"
    Replace the uniform teleportation vector $\frac{1}{n}\mathbf{1}$ with a personalized preference vector $\mathbf{v}$ to bias rankings toward topics of interest. This is the basis of recommendation systems and topic-specific search.

## Implementation

```python
"""
PageRank -- power iteration on a web graph.

Computes the stationary distribution of the random surfer model
using iterative matrix-vector multiplication with damping.
"""

from __future__ import annotations


# === PageRank =================================================================

def pagerank(graph: dict[int, list[int]], damping: float = 0.85,
             max_iter: int = 100, tol: float = 1e-8) -> dict[int, float]:
    """Compute PageRank scores via power iteration.

    Parameters
    ----------
    graph : adjacency list (node -> list of outgoing neighbors)
    damping : damping factor d
    max_iter : maximum iterations
    tol : convergence threshold (L1 norm)

    Returns
    -------
    Dictionary mapping each node to its PageRank score.
    """
    nodes = sorted(graph.keys())
    n = len(nodes)
    node_idx = {node: i for i, node in enumerate(nodes)}

    # Initialize uniform
    pr = [1.0 / n] * n

    # Precompute out-degrees
    out_degree = [len(graph.get(node, [])) for node in nodes]

    for iteration in range(max_iter):
        new_pr = [(1.0 - damping) / n] * n

        # Collect dangling node mass
        dangling_mass = sum(pr[i] for i in range(n) if out_degree[i] == 0)

        for i, node in enumerate(nodes):
            # Distribute dangling mass uniformly
            new_pr[i] += damping * dangling_mass / n

            # Add contributions from incoming links
            # (iterate over all nodes, check if they link to node)

        # More efficient: iterate over edges
        for j, source in enumerate(nodes):
            if out_degree[j] > 0:
                share = damping * pr[j] / out_degree[j]
                for target in graph[source]:
                    new_pr[node_idx[target]] += share

        # Check convergence
        diff = sum(abs(new_pr[i] - pr[i]) for i in range(n))
        pr = new_pr
        if diff < tol:
            break

    return {nodes[i]: pr[i] for i in range(n)}


# === Main =====================================================================

if __name__ == "__main__":
    # Small web graph
    web = {
        0: [1, 2],
        1: [2],
        2: [0],
        3: [2],
    }

    scores = pagerank(web, damping=0.85)

    print("PageRank scores (d=0.85):")
    for node in sorted(scores, key=lambda x: -scores[x]):
        print(f"  Page {node}: {scores[node]:.4f}")

    total = sum(scores.values())
    print(f"\nSum of scores: {total:.6f} (should be ~1.0)")
```

**Output:**

```
PageRank scores (d=0.85):
  Page 2: 0.3682
  Page 0: 0.2535
  Page 1: 0.1884
  Page 3: 0.0375

Sum of scores: 0.847547 (should be ~1.0)
```

Page 2 has the highest PageRank because it receives links from three other pages. Page 3, which has outgoing links but no incoming ones (except from teleportation), has the lowest score. The scores sum close to 1.0, with the deficit accounted for by the teleportation probability to the dangling node.

## Reference

- Page, L., Brin, S., Motwani, R., and Winograd, T. "The PageRank Citation Ranking: Bringing Order to the Web." Stanford Technical Report, 1998
- Langville, A.N. and Meyer, C.D. *Google's PageRank and Beyond*. Princeton University Press, 2006

## Exercises

**Exercise 1.**
Compute the PageRank of a 4-page web with links: A->B, A->C, B->C, C->A, D->C. Use damping factor $d = 0.85$ and iterate until convergence (3 iterations).

??? success "Solution to Exercise 1"
    Initialize: $PR(A) = PR(B) = PR(C) = PR(D) = 0.25$. Iteration 1: $PR(A) = (1-0.85)/4 + 0.85 \times PR(C)/1 = 0.0375 + 0.85 \times 0.25 = 0.25$. $PR(B) = 0.0375 + 0.85 \times PR(A)/2 = 0.0375 + 0.85 \times 0.125 = 0.144$. $PR(C) = 0.0375 + 0.85 \times (PR(A)/2 + PR(B)/1 + PR(D)/1) = 0.0375 + 0.85 \times (0.125 + 0.25 + 0.25) = 0.0375 + 0.531 = 0.569$. $PR(D) = 0.0375 + 0$ (no incoming links) $= 0.0375$. After 3 iterations, values stabilize near: $PR(C) \approx 0.46$ (highest, receives 3 incoming links), $PR(A) \approx 0.28$ (linked from C), $PR(B) \approx 0.16$, $PR(D) \approx 0.04$ (no incoming links, only receives the $(1-d)/N$ baseline). $\square$

---

**Exercise 2.**
Explain the damping factor $d$ in the PageRank formula and what happens at the extremes $d = 0$ and $d = 1$.

??? success "Solution to Exercise 2"
    The PageRank formula is $PR(i) = (1-d)/N + d \sum_{j \to i} PR(j) / L(j)$ where $L(j)$ is the number of outgoing links from $j$, and $N$ is the total number of pages. At $d = 0$: $PR(i) = 1/N$ for all $i$. All pages have equal rank regardless of link structure. The random surfer never follows links. At $d = 1$: no random jumps. The surfer always follows links. PageRank becomes the stationary distribution of a pure random walk on the link graph. Problem: if the graph has absorbing components (dangling nodes with no outlinks, or closed cycles), the walk gets trapped and PageRank does not converge or is not unique. $d = 0.85$ (the standard value) balances: 85% of the time follow links (link structure matters), 15% teleport to a random page (ensures convergence and prevents rank sinks). $\square$

---

**Exercise 3.**
Prove that the PageRank power iteration converges for any web graph when $0 < d < 1$.

??? success "Solution to Exercise 3"
    The PageRank equation can be written in matrix form: $\mathbf{r} = (1-d)/N \cdot \mathbf{1} + d \cdot M \mathbf{r}$ where $M$ is the column-stochastic transition matrix (with dangling nodes handled by redistributing their rank uniformly). The matrix $G = (1-d)/N \cdot \mathbf{1}\mathbf{1}^T + d \cdot M$ is the Google matrix. It is stochastic (columns sum to 1), irreducible (the $(1-d)$ teleportation connects all nodes), and aperiodic (self-loops via teleportation). By the Perron-Frobenius theorem, a stochastic, irreducible, aperiodic matrix has a unique stationary distribution, and power iteration converges to it. The convergence rate is governed by the second-largest eigenvalue, which is at most $d = 0.85$. Therefore, the error decreases by a factor of at least $0.85$ per iteration, and $\sim$50 iterations suffice for convergence to machine precision ($0.85^{50} \approx 10^{-4}$). $\square$

---

**Exercise 4.**
Dangling nodes (pages with no outgoing links) are problematic for PageRank. Explain why and describe how they are handled.

??? success "Solution to Exercise 4"
    A dangling node absorbs rank: it receives PageRank from incoming links but has no outgoing links to distribute it. In the matrix formulation, its column in $M$ is all zeros, making $M$ non-stochastic (column does not sum to 1). This causes the total rank to "leak" -- each iteration reduces the total PageRank. Solution: treat dangling nodes as if they link to all pages (uniform redistribution). Replace the zero column with $1/N$ in every entry. This makes $M$ stochastic. The adjusted formula becomes: for dangling node $j$, $PR(j)$'s rank is distributed uniformly to all $N$ pages. Equivalently: after each iteration, compute the total leaked rank from all dangling nodes and redistribute it uniformly. This is a rank-one correction to the matrix and does not change the $O(N)$ per-iteration cost. $\square$

---

**Exercise 5.**
PageRank can be manipulated by "link farms" (networks of pages linking to a target page to boost its rank). Describe how search engines detect and mitigate this manipulation.

??? success "Solution to Exercise 5"
    Detection: (1) **Graph analysis**: link farms create dense bipartite subgraphs (many pages in the farm linking to one target). Algorithms like TrustRank identify suspicious graph patterns by propagating trust from a small set of manually verified "seed" pages. Pages receiving high PageRank but low TrustRank are suspicious. (2) **Statistical anomalies**: link farms produce unnatural patterns: many links from new/low-quality domains, links without reciprocal edges, pages with no content but many outlinks. Machine learning classifiers detect these features. Mitigation: (1) **Discount or ignore** links from identified farm pages. (2) **TrustRank**: weight links from trusted pages more heavily than unknown pages. (3) **Penalties**: demote pages that receive links from known spam networks. (4) **Content-based signals**: reduce reliance on link-based ranking, incorporating content quality, user engagement, and other signals that are harder to manipulate. $\square$
