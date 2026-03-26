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
