# 29.6.4 Community Detection

## Overview
Community detection identifies groups of densely connected nodes. GNN approaches learn node embeddings that cluster naturally, outperforming classical methods on complex structures.

## Classical Methods
- **Spectral clustering**: Cluster using Fiedler vector / Laplacian eigenvectors
- **Louvain**: Greedy modularity optimization
- **Label propagation**: Iterative label spreading

## GNN-based Methods
- **Unsupervised**: Train GNN with reconstruction loss, cluster embeddings
- **DMON**: Differentiable modularity maximization
- **MinCutPool**: Minimize normalized cut objective via GNN

## Evaluation
- **Modularity**: Quality of community partition
- **NMI (Normalized Mutual Information)**: Agreement with ground truth
- **Conductance**: Quality of individual communities

## Financial Applications
- **Sector discovery**: Unsupervised sector identification from correlation networks
- **Fraud rings**: Detect coordinated fraud groups in transaction networks
- **Market regimes**: Identify clusters of co-moving assets
