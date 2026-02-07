# Chapter 29: Recommender Systems

## Overview

Recommender systems predict user preferences for items, forming the backbone of personalization in modern applications — from movie suggestions and e-commerce to financial product recommendations and portfolio allocation. This chapter develops the mathematical foundations and PyTorch implementations of recommender system architectures, progressing from classical collaborative filtering through neural methods to modern embedding-based and sequential approaches.

## Why Recommender Systems in a Deep Learning Curriculum?

At their core, recommender systems solve a **representation learning** problem: given sparse user–item interactions, learn dense vector representations (embeddings) that capture latent preferences and attributes. This connects directly to several themes developed throughout this book:

- **Embedding layers (Chapter 13)**: `nn.Embedding` maps discrete IDs to dense vectors — the same mechanism underlies both Word2Vec and collaborative filtering.
- **Attention mechanisms (Chapter 14)**: Modern recommender architectures increasingly leverage attention to weigh item features and sequential context.
- **Sequence models (Chapters 17–19)**: Session-based recommendations model user behavior as temporal sequences, applying RNN and Transformer architectures.
- **Autoencoders (Chapter 24)**: Variational autoencoders provide a generative approach to collaborative filtering.
- **Factor models (Chapter 20)**: Matrix factorization for recommendations is closely related to latent factor models in asset pricing.

## The Recommendation Problem

We observe a sparse **rating matrix** $R \in \mathbb{R}^{m \times n}$ where $R_{ui}$ is the rating user $u$ gave item $i$. Most entries are missing. The goal is to predict unobserved entries:

$$\hat{R}_{ui} = f(u, i; \theta)$$

where $f$ is a learned scoring function parameterized by $\theta$. The choice of $f$ defines the model family — from simple dot products (matrix factorization) to deep neural networks (neural collaborative filtering) to sequential architectures (Transformer-based recommendations).

## Feedback Types

Recommender systems operate on two fundamentally different types of user feedback:

| Feedback Type | Examples | Signal | Challenge |
|---------------|----------|--------|-----------|
| **Explicit** | Star ratings, thumbs up/down, review scores | Direct preference indication | Sparse — users rate few items |
| **Implicit** | Clicks, purchases, watch time, page views | Indirect behavioral signal | No negative signal — absence ≠ dislike |

Most real-world systems rely on implicit feedback, which is abundant but noisy. The models in this chapter are developed primarily for explicit feedback (rating prediction) but the architectures extend naturally to implicit settings through appropriate loss functions (e.g., BPR loss, binary cross-entropy).

## Taxonomy of Approaches

| Approach | Key Idea | Strengths | Limitations |
|----------|----------|-----------|-------------|
| **Collaborative Filtering** | Users who agreed in the past will agree in the future | No item features needed | Cold-start problem |
| **Content-Based** | Recommend items similar to what user liked | Handles cold items | Limited diversity |
| **Hybrid Methods** | Combine collaborative and content signals | Best of both worlds | Increased complexity |
| **Sequential** | Model temporal dynamics of user behavior | Captures evolving preferences | Requires interaction sequences |
| **Embedding-Based** | Learn general-purpose representations | Scalable, transferable | Requires large data |

## Mathematical Notation

The following notation is used throughout this chapter:

| Symbol | Meaning |
|--------|---------|
| $m$ | Number of users |
| $n$ | Number of items |
| $R \in \mathbb{R}^{m \times n}$ | Rating matrix (sparse) |
| $\Omega$ | Set of observed (user, item) pairs |
| $\mathbf{p}_u \in \mathbb{R}^d$ | Latent factor vector for user $u$ |
| $\mathbf{q}_i \in \mathbb{R}^d$ | Latent factor vector for item $i$ |
| $d$ | Embedding dimension |
| $b_u, b_i$ | User and item bias terms |
| $\mu$ | Global mean rating |
| $\mathbf{x}_i \in \mathbb{R}^p$ | Content feature vector for item $i$ |

## Learning Objectives

After completing this chapter, you will be able to:

1. Derive the matrix factorization objective from the perspective of low-rank approximation and explain why direct SVD fails on sparse matrices
2. Explain why bias terms are essential and derive their optimal closed-form values
3. Implement MF, MF with bias, and neural collaborative filtering in PyTorch
4. Understand the connection between `nn.Embedding` lookup and matrix multiplication
5. Build content-based and hybrid recommender systems that combine collaborative and feature-based signals
6. Apply sequential models to capture temporal dynamics in user behavior
7. Design embedding-based retrieval systems for large-scale recommendation
8. Evaluate recommender systems using appropriate metrics for both rating prediction and ranking tasks

## Prerequisites

- PyTorch fundamentals (Chapter 1): tensor operations, `nn.Module`, training loops
- Embedding layers (Chapter 13): `nn.Embedding` mechanics
- Feedforward networks (Chapter 7): MLP architecture, backpropagation
- Linear algebra: matrix factorization, dot products, rank
- Sequence models (Chapters 17–19): helpful for sequential recommendations

## Chapter Contents

| Section | Topic | Key Concepts |
|---------|-------|-------------|
| **29.1** | **Foundations** | |
| 29.1.1 | [RecSys Overview](recsys_overview.md) | Problem formulation, feedback types, taxonomy, design considerations |
| 29.1.2 | [Collaborative Filtering](collaborative_filtering.md) | User-based vs item-based CF, neighborhood methods, sparsity |
| 29.1.3 | [Matrix Factorization](matrix_factorization.md) | Low-rank approximation, SVD connection, bias decomposition, embeddings |
| **29.2** | **Neural Methods** | |
| 29.2.1 | [Neural Collaborative Filtering](ncf.md) | MLP over embeddings, nonlinear interactions, NeuMF |
| 29.2.2 | [Content-Based Filtering](content_based.md) | Item features, user profiles, TF-IDF, neural feature extraction |
| 29.2.3 | [Hybrid Methods](hybrid.md) | Combining CF and content, weighted hybrids, feature augmentation |
| 29.2.4 | [Sequential Recommendations](sequential.md) | Session-based models, GRU4Rec, SASRec, Transformer recommenders |
| 29.2.5 | [Embedding-Based RecSys](embedding_recsys.md) | Two-tower models, ANN retrieval, contrastive learning |
| **29.3** | **Evaluation** | |
| 29.3.1 | [Evaluation Metrics](evaluation.md) | MSE, RMSE, ranking metrics, offline evaluation protocols |

## Dataset

All implementations use the [MovieLens Small](https://grouplens.org/datasets/movielens/) dataset (100K ratings, 600 users, 9,000 movies), a standard benchmark for recommender system research. The dataset provides explicit ratings on a 0.5–5.0 scale with timestamps, enabling both rating prediction and temporal evaluation.

---

!!! info "Connection to Quantitative Finance"
    Recommender systems have direct applications in finance: recommending financial products to clients, suggesting portfolio allocations based on investor profiles, or identifying similar assets based on co-movement patterns. The matrix factorization techniques in this chapter are closely related to factor models in asset pricing (Chapter 20), and the embedding-based retrieval methods parallel techniques used in quantitative research for finding similar securities or matching trades.
