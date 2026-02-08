# 29.7.6 Financial Networks

## Overview
Financial networks are among the most impactful GNN applications. Asset correlations, transaction flows, and institutional relationships form natural graph structures that GNNs can exploit for prediction, risk management, and anomaly detection.

## Key Applications

### Portfolio Optimization
- Model asset correlation network with GNN
- Learn graph-aware portfolio weights
- Capture non-linear dependencies beyond correlation

### Risk Management
- **Systemic risk**: GNN propagates risk through interbank lending networks
- **Contagion modeling**: Message passing simulates cascading defaults
- **Stress testing**: Evaluate network resilience under shock scenarios

### Fraud Detection
- Transaction graphs: accounts as nodes, transactions as edges
- GNN detects anomalous patterns (circular transactions, layering)
- Semi-supervised: few labeled fraud cases, many unlabeled

### Stock Prediction
- Build stock relation graph from correlations, supply chain, or sector membership
- GNN aggregates related stock signals for improved prediction
- Temporal GNN captures evolving market structure

### Credit Scoring
- Company relationship networks augment traditional credit features
- GNN propagates credit signals through supply chain
- Semi-supervised learning with limited labeled defaults

## Graph Construction for Finance

### Correlation Networks
$A_{ij} = \mathbb{1}[|\rho_{ij}| > \tau]$ with rolling correlations.

### Sector/Industry Graphs
Connect companies in the same sector or GICS sub-industry.

### Supply Chain Graphs
Directed edges from supplier to customer with revenue exposure weights.

### Transaction Graphs
Directed, temporal, weighted edges representing money flows.

## Challenges
- **Non-stationarity**: Financial graphs evolve rapidly
- **Noise**: Correlation-based edges are noisy
- **Interpretability**: Regulatory requirements demand explainable models
- **Data quality**: Missing edges, incomplete transaction records

## Summary
Financial networks present rich opportunities for GNN applications. The combination of graph structure with financial time series creates powerful representations for prediction, risk assessment, and anomaly detection.
