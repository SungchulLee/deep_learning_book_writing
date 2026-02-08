# 31.6.1 Financial Network Generation

## Overview

Financial systems are intrinsically networked: banks lend to each other in the interbank market, firms share supply chains, investors hold overlapping portfolios, and counterparties are connected through derivative contracts. These **financial networks** determine how risk propagates through the system—a single institution's failure can cascade through the network, amplifying losses far beyond the initial shock. Generating realistic synthetic financial networks is critical for stress testing, systemic risk assessment, regulatory compliance (e.g., Basel III/IV), and privacy-preserving data sharing.

## Financial Network Types

| Network | Nodes | Edges | Key Properties |
|---------|-------|-------|----------------|
| Interbank lending | Banks | Bilateral loans | Directed, weighted, sparse, core-periphery |
| Correlation network | Assets | Return correlations | Undirected, weighted, dense, hierarchical |
| Portfolio overlap | Funds/banks | Shared holdings | Bipartite, weighted |
| Derivative/CDS | Counterparties | Contracts | Directed, weighted, concentrated |
| Payment network | Institutions | Payments | Directed, weighted, temporal |
| Supply chain | Firms | Supply relationships | Directed, multi-layer |

## Stylized Facts of Financial Networks

Empirical financial networks exhibit distinctive statistical regularities that synthetic generators must reproduce:

**Heavy-tailed degree distributions**: A few hub institutions have many connections, while most have few. The degree distribution often follows a power law or log-normal:

$$P(k) \propto k^{-\gamma}, \quad \gamma \in [2, 3]$$

**Core-periphery structure**: A densely connected core of major institutions surrounded by a sparse periphery. The core handles most of the volume; periphery nodes connect primarily to core nodes.

**Disassortativity**: High-degree nodes tend to connect to low-degree nodes (unlike social networks where hubs connect to hubs). This reflects the intermediation role of large banks.

**Small-world property**: Short average path lengths combined with high clustering—information (and contagion) can traverse the network quickly.

**Temporal dynamics**: Network topology evolves over time—edges appear and disappear, weights change with market conditions, and new institutions enter or exit.

**Weighted structure**: Edge weights (loan amounts, correlation magnitudes) are as important as topology. Weight distributions are typically heavy-tailed and exhibit strong heterogeneity.

## Classical Generative Models for Financial Networks

**Erdős-Rényi (ER)**: Each edge exists independently with probability $p$. Too simple for financial networks—fails to capture heavy tails, core-periphery, and degree heterogeneity.

**Configuration Model**: Preserves the degree sequence of the observed network. Sample degrees from the empirical distribution, then connect stubs randomly. Captures degree heterogeneity but not higher-order structure (clustering, core-periphery).

**Stochastic Block Model (SBM)**: Assigns nodes to blocks (communities) and specifies edge probabilities between blocks. Can capture core-periphery structure with two blocks. The degree-corrected SBM additionally matches degree heterogeneity.

**Fitness Model (Caldarelli et al.)**: Each node $i$ has a fitness $x_i$ (e.g., bank size). The probability of edge $(i, j)$ depends on both fitnesses:

$$P(A_{ij} = 1) = f(x_i, x_j)$$

A common choice: $f(x_i, x_j) = \sigma(\alpha + \beta \log x_i + \gamma \log x_j)$ where $\sigma$ is the sigmoid. This naturally produces heavy-tailed degree distributions from heavy-tailed fitness distributions.

## Deep Generative Models for Financial Networks

The graph generation methods from Chapters 31.2–31.4 can be adapted to financial networks with domain-specific modifications:

**GraphVAE for interbank networks**: Encode the adjacency matrix and node attributes (bank size, capital ratio) into a latent space. The decoder reconstructs both topology and edge weights. The latent space enables interpolation between observed network snapshots and generation of counterfactual networks.

**GraphRNN for temporal networks**: Generate the network sequentially, node by node. For temporal financial networks, condition on the previous time step's network to generate the next, capturing network evolution.

**Diffusion models**: The discrete diffusion framework (DiGress) naturally handles the categorical nature of financial networks (directed/undirected edges, multiple relationship types). Condition on macroeconomic variables to generate networks consistent with different economic scenarios.

## Systemic Risk and Contagion Simulation

A primary use case for synthetic financial networks is simulating systemic risk:

**Credit contagion (Eisenberg-Noe model)**: Given an interbank network with bilateral obligations, determine the clearing payment vector when one or more banks default. The model solves a fixed-point problem:

$$p_i = \min\left(\bar{p}_i,\; e_i + \sum_j \frac{\Pi_{ji} p_j}{\bar{p}_j}\right)$$

where $p_i$ is bank $i$'s actual payment, $\bar{p}_i$ is its total obligations, $e_i$ is external assets, and $\Pi_{ji}$ is the obligation of $j$ to $i$.

**DebtRank**: Measures the fraction of total economic value lost when a set of nodes is shocked. Unlike simple cascade models, DebtRank accounts for continuous devaluation of assets as neighbors experience distress.

**Why synthetic networks matter**: Real interbank networks are confidential. Regulators and researchers need synthetic networks that preserve the statistical properties of real networks to test contagion models, stress-test regulations, and train ML models for systemic risk detection.

## Conditional Generation for Stress Testing

Generate networks conditioned on macroeconomic scenarios:

$$p(\mathcal{G} \mid \text{GDP growth}, \text{interest rate}, \text{volatility})$$

This enables regulators to ask: "What would the interbank network look like under a severe recession?" by conditioning the generator on stressed macroeconomic variables and examining the resulting network's contagion properties.
