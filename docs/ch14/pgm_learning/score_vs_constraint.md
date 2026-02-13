# Score-Based vs Constraint-Based Structure Learning

## Overview

Structure learning discovers the graph structure of a PGM from data. Two main paradigms exist: score-based methods that search over graph structures to optimize a scoring function, and constraint-based methods that test for conditional independence.

## Score-Based Methods

### Approach
1. Define a scoring function $S(G, D)$ that measures how well graph $G$ fits data $D$
2. Search over the space of possible graphs to find $G^* = \arg\max_G S(G, D)$

### Scoring Functions

| Score | Formula | Properties |
|-------|---------|-----------|
| BIC/MDL | $\log P(D \mid \hat{\theta}, G) - \frac{d}{2} \log N$ | Penalizes complexity |
| AIC | $\log P(D \mid \hat{\theta}, G) - d$ | Less penalization |
| BGe | $\log P(D \mid G)$ (marginal likelihood) | Bayesian, integrates over parameters |

### Search Algorithms
- **Greedy hill climbing**: start from empty graph, add/remove/reverse edges
- **GES** (Greedy Equivalence Search): search over equivalence classes
- **MCMC over structures**: sample from posterior $P(G \mid D)$

## Constraint-Based Methods

### Approach
1. Test for conditional independence between all pairs of variables given various conditioning sets
2. Use the results to construct the graph skeleton and orient edges

### PC Algorithm
1. Start with a complete undirected graph
2. For each pair $(X, Y)$, test $X \perp Y \mid S$ for conditioning sets $S$ of increasing size
3. Remove the edge if independence is found
4. Orient edges using v-structure detection and orientation rules

### FCI Algorithm
Extension of PC that handles latent confounders by using partial ancestral graphs (PAGs).

## Comparison

| Aspect | Score-Based | Constraint-Based |
|--------|------------|-----------------|
| Output | Single best graph | Equivalence class |
| Handles latent variables | With modification | FCI handles natively |
| Computational cost | NP-hard search | Many CI tests |
| Sensitivity to sample size | Gradual degradation | Binary test failures |
| Software | bnlearn, pgmpy | pcalg, causal-learn |

## Hybrid Methods

Modern methods combine both approaches: use constraint-based tests to restrict the search space, then apply score-based optimization within the reduced space.
