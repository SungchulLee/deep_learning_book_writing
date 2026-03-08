# Chapter Overview

<<<<<<< HEAD
This chapter covers **Dynamic Programming**.
=======
This chapter covers linear and nonlinear dimensionality reduction techniques, from Principal Component Analysis and its variants to manifold learning methods. Each section develops the mathematical foundations rigorously and demonstrates applications to high-dimensional data analysis, with particular emphasis on financial applications including yield curve decomposition, portfolio optimization, and regime visualization.
>>>>>>> 96f31bd (...)

# Reference

<<<<<<< HEAD
[Introduction to Algorithms (CLRS), Chapter 15](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
=======
## PCA

Principal Component Analysis and its extensions for linear dimensionality reduction.

- [PCA Fundamentals](pca/pca_fundamentals.md) -- Variance maximization, orthogonal projections, and optimal linear compression
- [PCA Derivation](pca/pca_derivation.md) -- Rigorous derivation from variance-maximization and minimum-error perspectives
- [Eigendecomposition](pca/eigendecomposition.md) -- Computing principal components via spectral decomposition of the covariance matrix
- [SVD for PCA](pca/svd.md) -- Efficient and numerically stable PCA computation via Singular Value Decomposition
- [Kernel PCA](pca/kernel_pca.md) -- Nonlinear dimensionality reduction via the kernel trick
- [Probabilistic PCA](pca/probabilistic_pca.md) -- Generative latent variable formulation connecting PCA to VAEs
- [PCA Applications](pca/pca_applications.md) -- Practical implementations for projection, compression, denoising, and feature extraction
- [PCA as Linear Autoencoder](pca/pca_autoencoder.md) -- Mathematical equivalence between PCA and linear autoencoders

---

## Manifold Learning

Nonlinear dimensionality reduction methods that recover intrinsic low-dimensional structure.

- [Manifold Learning Introduction](manifold/introduction.md) -- The manifold hypothesis, neighborhood preservation, and method taxonomy
- [t-SNE](manifold/tsne.md) -- Probabilistic neighbor embedding with heavy-tailed distributions for visualization
- [UMAP](manifold/umap.md) -- Topological data analysis and SGD for scalable dimensionality reduction
- [Isomap](manifold/isomap.md) -- Geodesic distance preservation for nonlinear manifold unfolding
- [LLE](manifold/lle.md) -- Locally Linear Embedding preserving local reconstruction weights
- [MDS](manifold/mds.md) -- Multidimensional Scaling by preserving pairwise distances

---

## Finance

Applications of dimensionality reduction to quantitative finance problems.

- [PCA for Factor Models](finance/pca_factors.md) -- Extracting systematic risk factors from high-dimensional financial data
- [Yield Curve Decomposition](finance/yield_curve.md) -- PCA decomposition into level, slope, and curvature components
- [Portfolio Dimensionality Reduction](finance/portfolio_dimensionality.md) -- Improving portfolio optimization through factor-based compression
- [Regime Visualization](finance/regime_visualization.md) -- Visualizing market regimes using PCA, t-SNE, and manifold learning
>>>>>>> 96f31bd (...)
