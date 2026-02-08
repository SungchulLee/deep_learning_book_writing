# 29.5.2 Graph Regression

## Overview
Graph regression predicts continuous values for entire graphs. Uses the same GNN+readout pipeline as classification but with MSE/MAE loss. Key benchmarks: QM9 (quantum chemistry), ZINC (molecular solubility).

## Approach
1. GNN layers compute node embeddings
2. Readout pools nodes into graph-level vector
3. MLP predicts continuous target
4. Train with MSE or Huber loss

## Normalization
Target normalization is critical: standardize targets to zero mean and unit variance for stable training.

## Financial Applications
Predicting portfolio risk scores, systemic risk indices, or credit portfolio loss from network topology and features.
