# Collaborative Filtering Fundamentals

## Learning Objectives

- Understand the collaborative filtering hypothesis and when it applies
- Distinguish between memory-based (neighborhood) and model-based approaches
- Derive user-based and item-based similarity measures
- Recognize the sparsity challenge and its implications for scalability

## The Collaborative Filtering Hypothesis

**Core Idea**: Users who agreed in the past will agree in the future. If users $u$ and $v$ rated items similarly, then $u$'s rating for an unseen item can be predicted from $v$'s rating of that item.

Formally, given the observed rating matrix $R \in \mathbb{R}^{m \times n}$ with observed entries indexed by $\Omega \subseteq \{1, \ldots, m\} \times \{1, \ldots, n\}$, for any unobserved pair $(u, i) \notin \Omega$, we seek:

$$\hat{R}_{ui} = g\bigl(\{R_{vj} : (v, j) \in \Omega\}\bigr)$$

where $g$ aggregates information from the observed ratings.

!!! note "Key Distinction"
    Collaborative filtering uses **only** the interaction data (who rated what and how). It does not use item features (genre, description) or user features (age, location) — that is the domain of content-based methods (Section 29.2.2).

## Memory-Based Methods

Memory-based (or neighborhood-based) methods directly use the rating matrix to find similar users or items.

### User-Based Collaborative Filtering

**Intuition**: To predict user $u$'s rating for item $i$, find users similar to $u$ who rated item $i$, and take a weighted average.

**Step 1: Compute user similarity.** Let $I_u$ denote the set of items rated by user $u$. The cosine similarity between users $u$ and $v$ is:

$$\text{sim}(u, v) = \frac{\sum_{i \in I_u \cap I_v} R_{ui} \cdot R_{vi}}{\sqrt{\sum_{i \in I_u \cap I_v} R_{ui}^2} \cdot \sqrt{\sum_{i \in I_u \cap I_v} R_{vi}^2}}$$

Alternatively, **Pearson correlation** adjusts for rating scale differences:

$$\text{sim}(u, v) = \frac{\sum_{i \in I_u \cap I_v} (R_{ui} - \bar{R}_u)(R_{vi} - \bar{R}_v)}{\sqrt{\sum_{i \in I_u \cap I_v} (R_{ui} - \bar{R}_u)^2} \cdot \sqrt{\sum_{i \in I_u \cap I_v} (R_{vi} - \bar{R}_v)^2}}$$

where $\bar{R}_u = \frac{1}{|I_u|}\sum_{i \in I_u} R_{ui}$ is the mean rating of user $u$.

**Step 2: Predict ratings.** Given a neighborhood $\mathcal{N}_k(u)$ of the $k$ most similar users to $u$ who rated item $i$:

$$\hat{R}_{ui} = \bar{R}_u + \frac{\sum_{v \in \mathcal{N}_k(u)} \text{sim}(u, v) \cdot (R_{vi} - \bar{R}_v)}{\sum_{v \in \mathcal{N}_k(u)} |\text{sim}(u, v)|}$$

The subtraction and re-addition of means normalizes for different rating scales across users.

### Item-Based Collaborative Filtering

**Intuition**: Instead of finding similar users, find items similar to item $i$ that user $u$ has rated.

The item similarity between items $i$ and $j$ (using Pearson correlation) is:

$$\text{sim}(i, j) = \frac{\sum_{u \in U_i \cap U_j} (R_{ui} - \bar{R}_i)(R_{uj} - \bar{R}_j)}{\sqrt{\sum_{u \in U_i \cap U_j} (R_{ui} - \bar{R}_i)^2} \cdot \sqrt{\sum_{u \in U_i \cap U_j} (R_{uj} - \bar{R}_j)^2}}$$

where $U_i$ is the set of users who rated item $i$ and $\bar{R}_i$ is the mean rating of item $i$.

The prediction becomes:

$$\hat{R}_{ui} = \frac{\sum_{j \in \mathcal{N}_k(i) \cap I_u} \text{sim}(i, j) \cdot R_{uj}}{\sum_{j \in \mathcal{N}_k(i) \cap I_u} |\text{sim}(i, j)|}$$

### User-Based vs Item-Based: When to Use Which

| Criterion | User-Based | Item-Based |
|-----------|-----------|-----------|
| **Number of users vs items** | Better when $m \ll n$ | Better when $n \ll m$ |
| **Stability** | Similarities change as users add ratings | Item similarities are more stable |
| **Scalability** | $O(m^2)$ similarity computation | $O(n^2)$ similarity computation |
| **Interpretability** | "Users like you also liked..." | "Because you liked X, you'll like Y..." |
| **Industry adoption** | Less common | More common (Amazon's original system) |

## The Sparsity Problem

In practice, the rating matrix is extremely sparse. For MovieLens-100K with 600 users and 9,000 movies:

$$\text{density} = \frac{|\Omega|}{m \times n} = \frac{100{,}000}{600 \times 9{,}000} \approx 1.85\%$$

This means over 98% of the matrix is unobserved, which causes several problems:

1. **Insufficient overlap**: Two users may share very few co-rated items, making similarity estimates unreliable.
2. **Scalability**: Computing all pairwise similarities is $O(m^2 n)$ or $O(n^2 m)$.
3. **Cold-start**: New users or items have no ratings, so neighborhood methods fail entirely.

These limitations motivate **model-based** approaches, particularly matrix factorization.

## Model-Based Collaborative Filtering

Instead of directly computing similarities, model-based methods learn a parametric model from the data. The key idea: approximate the sparse rating matrix as a product of low-dimensional factor matrices.

$$R \approx P Q^\top$$

where $P \in \mathbb{R}^{m \times d}$ and $Q \in \mathbb{R}^{n \times d}$, with $d \ll \min(m, n)$.

Each row $\mathbf{p}_u$ of $P$ represents user $u$ in a $d$-dimensional latent space, and each row $\mathbf{q}_i$ of $Q$ represents item $i$ in the same space. A rating is predicted as:

$$\hat{R}_{ui} = \mathbf{p}_u^\top \mathbf{q}_i$$

This is the foundation of **matrix factorization**, covered in the next section.

!!! info "Why Latent Factors?"
    The latent dimensions can be interpreted as abstract concepts. For movies, one dimension might capture "action vs drama," another "mainstream vs indie." Neither users nor items are described by these features explicitly — they are **learned** from the interaction data.

## From Neighborhood to Embedding

The progression from memory-based to model-based CF mirrors a broader pattern in machine learning:

| Memory-Based CF | Model-Based CF |
|----------------|---------------|
| Store all ratings, compute at query time | Learn parameters offline, predict at query time |
| Non-parametric | Parametric ($2d(m + n)$ parameters) |
| Exact neighbors in rating space | Approximate neighbors in latent space |
| No generalization beyond observed data | Generalization through low-rank structure |
| Similar to $k$-NN | Similar to learned embeddings |

The embedding-based view is central: `nn.Embedding(num_users, d)` in PyTorch is precisely the matrix $P$, and `nn.Embedding(num_items, d)` is $Q$. A forward pass computes $\mathbf{p}_u^\top \mathbf{q}_i$ — the predicted rating.

## Summary

Collaborative filtering leverages collective user behavior to make predictions. Memory-based methods are intuitive but struggle with sparsity and scale. Model-based methods — particularly matrix factorization — address these limitations by learning compact representations. The next section develops the MF framework mathematically and implements it in PyTorch.

---

## Exercises

1. **Similarity computation**: Given the following rating matrix, compute the cosine similarity and Pearson correlation between users 1 and 2:

    | | Item A | Item B | Item C | Item D |
    |---|---|---|---|---|
    | User 1 | 5 | 3 | — | 1 |
    | User 2 | 4 | — | 2 | 1 |
    | User 3 | — | 1 | 5 | 4 |

2. **Cold-start analysis**: Suppose a new user joins with zero ratings. Explain why memory-based CF produces no predictions. How could you provide initial recommendations?

3. **Sparsity bound**: If a rating matrix has density $\rho$, what is the expected number of co-rated items between two random users? Under what density does user-based CF become impractical (say, expected overlap < 5)?

4. **Computational complexity**: Derive the time complexity of computing all pairwise user similarities for a matrix with $m$ users, $n$ items, and average $k$ ratings per user.
