# Chapter Overview

This chapter covers **External Memory**.

<<<<<<< HEAD
# Reference

[Algorithms and Data Structures for External Memory (Vitter)](https://www.ittc.ku.edu/~jsv/Papers/Vit.IO_book.pdf)
=======
Graph generation learns distributions over graph-structured data and produces novel graphs that are statistically indistinguishable from training samples. This chapter covers the fundamental challenges of generating discrete, permutation-invariant structures with variable dimensionality, and explores autoregressive, one-shot, and diffusion-based approaches. Applications span molecular design, financial network synthesis, and transaction graph generation.

## Contents

### 31.1 Foundations

- [Introduction to Graph Generation](foundations/introduction.md) -- The generation problem, combinatorial output spaces, permutation symmetry, and formal problem statement
- [Graph Representations for Generation](foundations/representations.md) -- Adjacency matrix, edge list, and sequential representations with their implications for generative model design
- [Evaluation Metrics](foundations/metrics.md) -- Degree distribution, clustering coefficients, and distributional comparison metrics for assessing generated graph quality

### 31.2 Autoregressive Methods

- [Sequential Graph Generation](autoregressive/sequential.md) -- Autoregressive factorization of graph distributions using the chain rule of probability with canonical orderings
- [GraphRNN](autoregressive/graphrnn.md) -- Hierarchical RNN architecture with graph-level and edge-level recurrence for sequential node and edge generation
- [GRAN](autoregressive/gran.md) -- Block-wise graph generation using GNN-based attention for scalable autoregressive generation

### 31.3 One-Shot Methods

- [One-Shot Adjacency Generation](one_shot/adjacency.md) -- Producing entire adjacency matrices in a single forward pass with continuous relaxations and permutation invariance
- [GraphVAE](one_shot/graphvae.md) -- Variational autoencoder for graphs with GNN encoder, probabilistic decoder, and graph-matching loss
- [GraphGAN](one_shot/graphgan.md) -- Adversarial graph generation with implicit density modeling and discretization strategies

### 31.4 Diffusion-Based Methods

- [Graph Diffusion Models](diffusion/graph_diffusion.md) -- Extending denoising diffusion to graph-structured data with continuous relaxation and discrete noise formulations
- [DiGress](diffusion/digress.md) -- Discrete denoising diffusion for categorical node and edge types using graph transformer denoisers
- [GDSS](diffusion/gdss.md) -- Score-based graph generation via coupled stochastic differential equations for joint node and adjacency generation

### 31.5 Molecular Generation

- [Molecular Graphs](molecular/graphs.md) -- Representing molecules as attributed graphs with atom types, bond types, and chemical valency constraints
- [SMILES-Based Generation](molecular/smiles.md) -- Reducing molecular generation to sequence modeling using SMILES string representations and language model techniques
- [3D Molecule Generation](molecular/3d_generation.md) -- Joint generation of molecular graphs and atomic coordinates with SE(3)-equivariant models
- [Property Optimization](molecular/property_optimization.md) -- Generating molecules optimized for target properties using RL, Bayesian optimization, and latent space search

### 31.6 Financial Applications

- [Financial Network Generation](finance/networks.md) -- Synthesizing realistic interbank, correlation, and derivative networks for stress testing and systemic risk assessment
- [Transaction Graph Generation](finance/transactions.md) -- Generating temporal transaction graphs for AML model development and fraud detection training
- [Synthetic Market Networks](finance/market_networks.md) -- Generating asset correlation and market structure networks for portfolio stress testing and regime analysis
>>>>>>> 96f31bd (...)
