# 31.6.3 Synthetic Market Networks

## Overview

Market networks capture the relationships between financial instruments—stocks, bonds, currencies, commodities—through their co-movement patterns. The most common construction is the **correlation network**, where nodes are assets and edge weights reflect pairwise return correlations. Generating synthetic market networks enables portfolio managers to stress-test allocation strategies under hypothetical market regimes, researchers to study network-based risk measures without data constraints, and regulators to model systemic risk in asset markets.

## Correlation Network Construction

Given $n$ assets with return time series $\mathbf{r}_i(t)$ over $T$ periods, the Pearson correlation matrix is:

$$\rho_{ij} = \frac{\text{Cov}(r_i, r_j)}{\sigma_i \sigma_j}$$

To construct a network, apply a threshold $\theta$ or use the **Minimum Spanning Tree (MST)** to extract the most informative structure:

**Threshold network**: Edge $(i,j)$ exists if $|\rho_{ij}| > \theta$. Simple but sensitive to the threshold choice. Too low → dense, uninformative; too high → sparse, disconnected.

**MST (Mantegna, 1999)**: Convert correlations to distances $d_{ij} = \sqrt{2(1 - \rho_{ij})}$ and compute the minimum spanning tree. The MST preserves the most important relationships with exactly $n-1$ edges, revealing hierarchical sector structure.

**PMFG (Planar Maximally Filtered Graph)**: Retains $3(n-2)$ edges while maintaining planarity, capturing more structure than the MST while still filtering noise.

## Stylized Facts of Market Networks

Empirical market networks exhibit consistent properties across markets and time periods:

**Sector clustering**: Assets from the same sector (technology, financials, energy) cluster together in the network, reflecting common factor exposures.

**Hierarchical structure**: The MST reveals a multi-scale hierarchy—individual stocks cluster into sub-industries, which group into sectors, which connect through a few hub assets.

**Scale-free properties**: Degree distributions in threshold networks are heavy-tailed: a few assets (typically large-cap, diversified companies or sector ETFs) serve as hubs with many connections.

**Regime dependence**: Network topology changes with market conditions. During crises, correlations increase universally ("correlation breakdown"), the network becomes denser, and the clustering structure collapses toward a single cluster.

**Temporal evolution**: The MST and correlation structure evolve over time, with some edges persistent (structural relationships) and others transient (temporary co-movement).

## Random Matrix Theory (RMT) Filtering

Raw correlation matrices contain substantial noise, especially when $T$ (time periods) is not much larger than $n$ (assets). RMT provides a principled way to separate signal from noise.

The **Marchenko-Pastur distribution** describes the eigenvalue spectrum of random correlation matrices. For a random matrix with $n$ assets and $T$ observations, eigenvalues fall in the range:

$$\lambda_{\pm} = \left(1 \pm \sqrt{n/T}\right)^2$$

Eigenvalues outside this range carry genuine information about market structure. The filtered correlation matrix retains only these signal eigenvalues:

$$\hat{\mathbf{C}} = \sum_{k: \lambda_k > \lambda_+} \lambda_k \mathbf{v}_k \mathbf{v}_k^\top + \text{diagonal correction}$$

## Generative Models for Market Networks

### Factor-Based Generation

The most interpretable approach generates returns from a factor model, then derives the correlation network:

$$r_i(t) = \alpha_i + \sum_{k=1}^{K} \beta_{ik} f_k(t) + \epsilon_i(t)$$

where $f_k(t)$ are factor returns (market, size, value, momentum) and $\beta_{ik}$ are factor loadings. The implied correlation is:

$$\rho_{ij} = \frac{\boldsymbol{\beta}_i^\top \boldsymbol{\Sigma}_f \boldsymbol{\beta}_j}{\sigma_i \sigma_j}$$

By varying the factor structure (number of factors, loading distributions, factor covariances), diverse market networks can be generated.

### Regime-Switching Networks

Model the network as switching between discrete regimes:

- **Normal regime**: Moderate correlations, clear sector structure, sparse network
- **Crisis regime**: High correlations, collapsed structure, dense network
- **Recovery regime**: Intermediate state with evolving structure

A hidden Markov model governs regime transitions, and each regime has its own network generation parameters.

### Deep Generative Models

**Graph VAE for market networks**: Encode correlation matrices into a latent space, conditioned on market regime indicators (VIX, yield curve slope, credit spreads). The decoder generates correlation matrices that preserve positive semi-definiteness.

**Diffusion models**: Generate the upper triangle of the correlation matrix using continuous diffusion, with a projection step to ensure the result is a valid correlation matrix (positive semi-definite, unit diagonal).

## Applications in Portfolio Management

**Stress testing**: Generate synthetic market networks under extreme scenarios (all correlations increase by 0.3, sector structure collapses) and evaluate portfolio performance. This goes beyond historical stress tests by exploring scenarios that haven't occurred yet.

**Regime-aware allocation**: Train portfolio optimization models on synthetic networks from different regimes. At inference time, estimate the current regime and apply the corresponding allocation strategy.

**Network-based risk measures**: Compute centrality measures (eigenvector centrality, betweenness) on the market network and use them as risk factors. Highly central assets are more exposed to systemic risk.

**Diversification analysis**: Use the network to identify truly diversifying assets—those in different network communities with weak connections to the rest of the portfolio.

## Temporal Network Generation

For backtesting dynamic strategies, generate sequences of market networks that evolve realistically over time:

$$\mathcal{G}_{t+1} = f_\theta(\mathcal{G}_t, \mathbf{x}_t)$$

where $\mathbf{x}_t$ represents macroeconomic conditions. The evolution must preserve:

- **Persistence**: Most edges are stable over short horizons
- **Smooth transitions**: Network statistics change gradually except during regime shifts
- **Regime shifts**: Abrupt structural changes during crises
- **Mean reversion**: Network properties tend to revert to long-term averages
