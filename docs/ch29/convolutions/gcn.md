# 29.3.4 Graph Convolutional Network (GCN)

## Introduction

The **Graph Convolutional Network (GCN)** by Kipf & Welling (2017) is one of the most influential GNN architectures. It derives a simple, efficient graph convolution by simplifying ChebNet to first-order Chebyshev polynomials, resulting in a layer that aggregates features from immediate neighbors with symmetric normalization.

## GCN Layer

The GCN propagation rule:

$$H^{(l+1)} = \sigma\left(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)}\right)$$

where:
- $\tilde{A} = A + I_n$ (adjacency with self-loops)
- $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$ (degree matrix of $\tilde{A}$)
- $H^{(l)}$ is the node feature matrix at layer $l$
- $W^{(l)}$ is the learnable weight matrix
- $\sigma$ is an activation function (e.g., ReLU)

Per-node formulation:

$$\mathbf{h}_v^{(l+1)} = \sigma\left(\sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{1}{\sqrt{\tilde{d}_u \tilde{d}_v}} W^{(l)} \mathbf{h}_u^{(l)}\right)$$

## Derivation from Spectral Theory

Starting from ChebNet with $K=1$ and $\lambda_{max} \approx 2$:

$$g_\theta * x \approx \theta_0 x + \theta_1 (L - I)x = \theta_0 x - \theta_1 D^{-1/2}AD^{-1/2} x$$

Setting $\theta = \theta_0 = -\theta_1$ (single parameter):

$$g_\theta * x = \theta(I + D^{-1/2}AD^{-1/2}) x$$

The **renormalization trick** replaces $I + D^{-1/2}AD^{-1/2}$ with $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ to improve numerical stability.

## Properties

- **1-hop neighborhood**: Each layer aggregates from immediate neighbors
- **Symmetric normalization**: $\frac{1}{\sqrt{d_u d_v}}$ prevents high-degree nodes from dominating
- **Linear complexity**: $O(|E| \cdot d)$ per layer
- **Semi-supervised**: Designed for node classification with few labels
- **Spectral motivation**: Principled derivation from spectral graph theory

## Limitations

- Fixed aggregation weights (no attention mechanism)
- Limited expressiveness (cannot distinguish certain graph structures)
- Over-smoothing with many layers
- Transductive: requires full graph at training time

## Quantitative Finance Applications

GCN is well-suited for financial networks where symmetric relationships dominate:
- **Asset correlation networks**: Symmetric correlations between assets
- **Interbank lending**: Risk propagation through banking networks
- **Credit risk**: Semi-supervised node classification for credit scoring

## Summary

GCN provides an elegant, efficient graph convolution with spectral foundations. Its simplicity and effectiveness make it a default baseline for graph learning tasks, particularly semi-supervised node classification.
