# 29.5.1 Graph Classification

## Introduction

**Graph classification** assigns a label to an entire graph. Applications include molecular property prediction (toxic/non-toxic), protein function classification, and financial network regime classification.

## Pipeline

1. **Message Passing**: Apply $K$ GNN layers to compute node embeddings $\{h_v^{(K)}\}$
2. **Readout/Pooling**: Aggregate node embeddings into a graph-level vector $h_G$
3. **Classification**: Apply MLP classifier to $h_G$

## Readout Functions

**Sum Pooling**: $h_G = \sum_{v \in V} h_v$ — preserves graph size information.

**Mean Pooling**: $h_G = \frac{1}{|V|}\sum_{v \in V} h_v$ — size-invariant.

**Max Pooling**: $h_G = \max_{v \in V} h_v$ — captures extreme features.

**Virtual Node**: Add a node connected to all others; its final embedding serves as the graph representation.

## Training

Standard cross-entropy loss with mini-batch training. Graphs are batched by creating a single disconnected graph (PyG's `DataLoader`).

## Evaluation

- k-fold cross-validation (common for molecular datasets)
- ROC-AUC for imbalanced classes
- Accuracy for balanced datasets

## Summary

Graph classification combines GNN node embeddings with graph-level pooling, enabling end-to-end learning on entire graphs.
