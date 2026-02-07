# Chapter 20: Dimensionality Reduction

Transforming high-dimensional data into compact, informative representations while preserving the structure that matters.

---

## Overview

Real-world data in quantitative finance — tick-level features, factor exposures, order book snapshots, macroeconomic indicators — routinely lives in spaces with hundreds or thousands of dimensions. Most of this apparent complexity is redundant: asset returns cluster along a handful of systematic factors, yield curves move through a low-dimensional manifold, and regime shifts reveal themselves in two or three latent coordinates. Dimensionality reduction makes this hidden structure explicit.

This chapter develops two complementary families of methods. **Section 20.1** covers **Principal Component Analysis** and its extensions — the workhorse linear methods that project data onto directions of maximum variance. Starting from geometric intuition, we derive PCA from three equivalent perspectives (variance maximization, minimum reconstruction error, and linear autoencoder equivalence), then build computational machinery through eigendecomposition and SVD. The section continues with two powerful generalizations — **Probabilistic PCA**, which casts dimensionality reduction as latent variable inference and provides a direct bridge to variational autoencoders, and **Kernel PCA**, which lifts PCA into nonlinear feature spaces via the kernel trick — before closing with a comprehensive applications gallery and a dedicated treatment of the PCA–autoencoder equivalence that motivates the transition to deep generative models.

**Section 20.2** turns to **manifold learning** — nonlinear methods designed for data that lies on curved, low-dimensional surfaces embedded in high-dimensional space. Where PCA finds the best hyperplane, manifold methods recover the intrinsic geometry. We progress from classical **Multidimensional Scaling** (distance preservation) through **Isomap** (geodesic distances) and **Locally Linear Embedding** (local reconstruction weights) to the modern visualization workhorses **t-SNE** and **UMAP**, examining their mathematical foundations, computational trade-offs, and the subtle art of interpreting their output.

Throughout both sections, every method is implemented in PyTorch with complete, runnable code and applied to quantitative finance problems: yield curve decomposition, asset correlation mapping, market regime visualization, and factor model analysis.


## Learning Objectives

After completing this chapter, you will be able to:

1. **Derive PCA** from variance maximization, minimum reconstruction error, and linear autoencoder equivalence, and explain why all three yield the same solution
2. **Implement PCA** via both eigendecomposition and SVD, understanding when each approach is numerically preferable
3. **Select the number of components** using explained variance ratios, scree plots, and cross-validation
4. **Formulate Probabilistic PCA** as a generative model, fit it with EM, handle missing data, and connect it to the VAE framework
5. **Apply the kernel trick** to perform nonlinear PCA and understand the centering and pre-image problems
6. **Apply PCA** to practical problems including image compression, denoising, eigenfaces, and yield curve decomposition
7. **Prove the PCA–autoencoder equivalence** and explain the architectural progression from PCA to VAEs
8. **Articulate the manifold hypothesis** and explain why linear methods fail on curved data
9. **Implement MDS, Isomap, and LLE** from scratch, understanding the spectral embedding machinery they share
10. **Use t-SNE and UMAP** effectively — choosing hyperparameters, avoiding common misinterpretations, and distinguishing visualization artifacts from genuine structure
11. **Apply dimensionality reduction** to financial data for factor extraction, regime detection, and portfolio analysis


## Prerequisites

| Prerequisite | Where Covered | What You Need |
|:---|:---|:---|
| Linear algebra | Ch 1 | Eigenvalues, eigenvectors, matrix decompositions, positive semi-definiteness |
| Probability basics | Ch 18 | Gaussian distributions, marginal/conditional distributions, maximum likelihood |
| PyTorch tensors | Ch 1 | Tensor operations, `torch.linalg`, autograd basics |
| Kernel methods | Ch 3 | Kernel functions, Mercer's theorem (helpful but not required for §20.1.6) |
| Graph concepts | — | Nearest-neighbor graphs, shortest paths (introduced in §20.2 as needed) |


## Chapter Structure

### 20.1 PCA

The linear foundation — eight pages that build from intuition through theory and computation to applications and connections:

| Section | Focus | Key Idea |
|:---|:---|:---|
| [PCA Fundamentals](pca/pca_fundamentals.md) | Geometric intuition, preprocessing, loadings, reconstruction | The projection that captures maximum variance |
| [PCA Derivation](pca/pca_derivation.md) | Three equivalent derivations, explained variance analysis | Why eigenvectors of the covariance matrix are optimal |
| [Eigendecomposition](pca/eigendecomposition.md) | Spectral decomposition of $\mathbf{S}$, numerical considerations | The direct route: diagonalize the covariance matrix |
| [SVD](pca/svd.md) | Singular value decomposition, truncated SVD, low-rank approximation | The numerically stable route: decompose the data matrix directly |
| [Probabilistic PCA](pca/probabilistic_pca.md) | Generative model, EM algorithm, missing data, connection to VAEs | PCA as latent variable inference |
| [Kernel PCA](pca/kernel_pca.md) | Kernel trick, centering in feature space, pre-image problem | Nonlinear PCA without explicit feature maps |
| [PCA Applications](pca/pca_applications.md) | MNIST, denoising, eigenfaces, yield curves, PyTorch autoencoder | Complete implementations from toy examples to production pipelines |
| [PCA as Autoencoder](pca/pca_autoencoder.md) | Equivalence proof, tied weights, bridge to nonlinear models | Why a linear autoencoder with MSE loss *is* PCA |

### 20.2 Manifold Learning

Nonlinear methods for curved, low-dimensional structure:

| Section | Focus | Key Idea |
|:---|:---|:---|
| [Introduction](manifold/introduction.md) | Manifold hypothesis, taxonomy, method comparison | Why nonlinear methods exist and when you need them |
| [MDS](manifold/mds.md) | Classical, metric, and non-metric MDS; stress minimization | Preserve pairwise distances in the embedding |
| [Isomap](manifold/isomap.md) | Geodesic distances, nearest-neighbor graphs | Replace Euclidean distances with manifold distances, then apply MDS |
| [LLE](manifold/lle.md) | Local reconstruction weights, embedding via eigenvectors | Preserve local linear relationships |
| [t-SNE](manifold/tsne.md) | Probability-based similarity, Student-t kernel, crowding problem | Convert distances to probabilities, match distributions |
| [UMAP](manifold/umap.md) | Fuzzy simplicial sets, cross-entropy optimization, parametric extension | Topological data analysis meets stochastic gradient descent |


## Conceptual Roadmap

```
High-Dimensional Data
         │
         ├── Is the structure approximately linear?
         │       │
         │       ├── Yes ──► PCA / SVD
         │       │            ├── Need a generative model? ──► Probabilistic PCA
         │       │            ├── Need nonlinear features?  ──► Kernel PCA
         │       │            └── Need a learnable encoder? ──► PCA as Autoencoder → Ch 21
         │       │
         │       └── No  ──► Manifold Learning
         │                    ├── Global distances matter?   ──► MDS / Isomap
         │                    ├── Local geometry matters?    ──► LLE
         │                    └── Visualization / clustering?──► t-SNE / UMAP
         │
         └── Want deep generative models? ──► Autoencoders (Ch 21) / VAEs (Ch 22)
```


## Connections to Other Chapters

Dimensionality reduction sits at a crossroads in the curriculum:

- **Upstream**: Linear algebra foundations (Ch 1) provide the eigendecomposition and SVD machinery. Statistical learning concepts (Ch 2) supply the bias-variance lens for choosing $k$. Bayesian inference (Ch 18) underpins Probabilistic PCA.
- **Downstream**: The PCA–autoencoder equivalence (§20.1.8) leads directly into **autoencoders** (Ch 21) and **variational autoencoders** (Ch 22). Understanding why PCA finds flat subspaces motivates the nonlinear encoders in generative models. The architectural progression PCA → Linear AE → Nonlinear AE → VAE provides a concrete thread through several chapters. UMAP embeddings frequently serve as inputs to clustering and regime detection in applied finance workflows.
- **Parallel**: Kernel PCA connects to the kernel methods in Ch 3. Graph-based manifold learning (Isomap, LLE) shares structure with graph neural networks (Ch 8).


## Key Notation

| Symbol | Meaning |
|:---|:---|
| $\mathbf{X} \in \mathbb{R}^{n \times d}$ | Data matrix — $n$ observations, $d$ features |
| $\mathbf{S} = \frac{1}{n-1}\mathbf{X}^\top\mathbf{X}$ | Sample covariance matrix (centered data) |
| $\mathbf{W} \in \mathbb{R}^{d \times k}$ | Projection matrix — columns are principal directions |
| $\mathbf{Z} = \mathbf{X}\mathbf{W} \in \mathbb{R}^{n \times k}$ | Scores (projected data) in the reduced space |
| $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d$ | Eigenvalues of $\mathbf{S}$, sorted descending |
| $\sigma_1 \geq \sigma_2 \geq \cdots$ | Singular values of $\mathbf{X}$ |
| $k(\mathbf{x}, \mathbf{x}')$ | Kernel function |
| $\mathbf{K} \in \mathbb{R}^{n \times n}$ | Kernel (Gram) matrix |
| $\mathbf{W}_e, \mathbf{W}_d$ | Encoder and decoder weight matrices (autoencoder) |
