# 29.6.1 Node Classification

## Overview
Node classification assigns labels to nodes using both node features and graph structure. It is the most studied GNN task, with semi-supervised settings where only a fraction of nodes are labeled.

## Pipeline
1. Apply $K$ GNN layers: $H^{(K)} = \text{GNN}(X, A)$
2. Classify: $\hat{Y} = \text{softmax}(H^{(K)} W)$
3. Loss: Cross-entropy on labeled nodes only

## Key Settings
- **Transductive**: Full graph available at training; predict unlabeled nodes
- **Inductive**: Train on one graph, predict on unseen graphs

## Benchmarks
- **Cora, CiteSeer, PubMed**: Citation networks (2-7 classes, ~3K-20K nodes)
- **ogbn-arxiv, ogbn-products**: Large-scale OGB benchmarks

## Training Tips
- Use full-batch for small graphs, mini-batch (neighbor sampling) for large graphs
- Label smoothing and feature normalization improve results
- Early stopping on validation accuracy prevents overfitting

## Financial Applications
- **Fraud detection**: Classify accounts as fraudulent/legitimate
- **Sector classification**: Predict company sector from transaction network
- **Credit scoring**: Semi-supervised risk rating from financial graph
