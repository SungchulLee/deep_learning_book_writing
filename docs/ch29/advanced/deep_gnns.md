# 29.4.1 Deep GNNs

## Introduction

Deeper networks generally improve performance in CNNs and transformers, but naively stacking GNN layers degrades performance due to over-smoothing, gradient vanishing, and over-squashing. This section covers techniques for building deeper, more effective GNNs.

## Challenges

### Over-Smoothing
Node representations converge to indistinguishable states as layers increase. After $K$ layers, all nodes in a connected component approach a common representation.

### Over-Squashing
Information from exponentially growing receptive fields is compressed into fixed-size vectors, creating bottlenecks for long-range message passing.

### Gradient Issues
Deep GNNs suffer from vanishing/exploding gradients, analogous to deep RNNs.

## Solutions

**Residual Connections**: $\mathbf{h}_v^{(l)} = \mathbf{h}_v^{(l-1)} + \text{GNN}^{(l)}(\cdot)$. Preserves information from earlier layers.

**Dense Connections**: Concatenate outputs from all previous layers (DenseGCN).

**Pre-activation**: Apply normalization and activation before GNN operation.

**DropEdge**: Randomly remove edges during training to slow over-smoothing.

**PairNorm**: Normalize to maintain pairwise distances between representations.

**GraphNorm**: Graph-level normalization with learnable shift.

## Summary

Building deep GNNs requires residual connections, normalization, and regularization to combat over-smoothing while enabling multi-hop information aggregation.
