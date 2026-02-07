# RecSys Overview

## Learning Objectives

- Formalize the recommendation problem as a function approximation task over sparse interaction data
- Distinguish between explicit and implicit feedback settings and their implications for model design
- Classify recommender system architectures by their information requirements and inductive biases
- Understand the cold-start problem and its role in motivating different model families
- Identify real-world design considerations for production recommender systems

## The Recommendation Problem

### Formal Definition

A recommender system operates on a set of $m$ users $\mathcal{U} = \{u_1, \ldots, u_m\}$ and $n$ items $\mathcal{I} = \{i_1, \ldots, i_n\}$. The system observes partial interactions between users and items, stored in a **rating matrix** $R \in \mathbb{R}^{m \times n}$ where entry $R_{ui}$ records the interaction of user $u$ with item $i$. The vast majority of entries are unobserved.

Formally, let $\Omega \subseteq \{1, \ldots, m\} \times \{1, \ldots, n\}$ denote the set of observed (user, item) pairs. The recommendation task is to learn a scoring function:

$$f: \mathcal{U} \times \mathcal{I} \to \mathbb{R}, \quad \hat{R}_{ui} = f(u, i; \theta)$$

that generalizes from the observed entries $\{R_{ui} : (u, i) \in \Omega\}$ to predict unobserved entries.

### Two Formulations

The recommendation problem admits two complementary formulations depending on the deployment context:

**Rating prediction** aims to estimate the exact value of unobserved ratings. This is a regression problem:

$$\min_\theta \sum_{(u,i) \in \Omega} \ell\bigl(R_{ui}, f(u, i; \theta)\bigr)$$

where $\ell$ is typically the squared loss. This formulation is natural for explicit feedback (star ratings) and was the focus of the Netflix Prize competition.

**Top-$K$ recommendation** aims to produce a ranked list of $K$ items most likely to be relevant to each user. This is a ranking problem — the model need not predict exact ratings, only the correct ordering. This formulation dominates in production systems where the goal is to surface the best items from a large catalog.

The distinction matters because models optimized for rating prediction (low RMSE) do not necessarily produce the best rankings, and vice versa.

## Explicit vs Implicit Feedback

### Explicit Feedback

Users directly express preferences through ratings, reviews, or thumbs up/down. The signal is clear but sparse — users rate a tiny fraction of available items.

For MovieLens-100K with 600 users and 9,000 movies:

$$\text{density} = \frac{|\Omega|}{m \times n} = \frac{100{,}000}{600 \times 9{,}000} \approx 1.85\%$$

Over 98% of the rating matrix is unobserved. This extreme sparsity is the central challenge of collaborative filtering.

### Implicit Feedback

Users express preferences indirectly through behavioral signals: clicks, purchases, watch time, scroll depth, search queries. Implicit feedback is far more abundant than explicit feedback but introduces two key challenges:

1. **No negative signal**: A user not clicking an item could mean dislike, unawareness, or irrelevance. The absence of interaction is not equivalent to a negative rating.
2. **Varying confidence**: A user who watched a movie for 2 hours provides stronger signal than one who watched for 30 seconds.

Implicit feedback is typically modeled as a binary matrix $Y \in \{0, 1\}^{m \times n}$ where $Y_{ui} = 1$ if user $u$ interacted with item $i$, with an associated confidence matrix $C_{ui}$ reflecting interaction strength. The weighted loss becomes:

$$\mathcal{L} = \sum_{u,i} C_{ui} \bigl(Y_{ui} - f(u, i; \theta)\bigr)^2$$

This is the formulation used by Hu, Koren, and Volinsky (2008) in their influential work on implicit feedback.

### Comparison

| Aspect | Explicit | Implicit |
|--------|----------|----------|
| **Volume** | Sparse (1–5% density) | Dense (many interactions) |
| **Signal quality** | High (direct preference) | Noisy (indirect inference) |
| **Negative signal** | Present (low ratings) | Absent (must be inferred) |
| **Bias** | Selection bias (users rate what they chose) | Position bias, popularity bias |
| **Examples** | Star ratings, likes/dislikes | Clicks, purchases, dwell time |
| **Loss functions** | MSE, MAE | BPR, binary cross-entropy, weighted MSE |

## Architecture Taxonomy

### By Information Source

Recommender systems can be classified by what information they use:

**Collaborative filtering (CF)** uses only the interaction matrix $R$. It assumes that users who agreed in the past will agree in the future. CF methods include neighborhood-based approaches (user-based and item-based) and model-based approaches (matrix factorization, neural CF).

**Content-based filtering** uses item features $\mathbf{x}_i$ (and optionally user features $\mathbf{z}_u$) to recommend items similar to what the user has previously liked. It constructs a user profile from the features of items the user has interacted with.

**Hybrid methods** combine collaborative and content-based signals to mitigate the weaknesses of each. Modern deep learning systems are almost always hybrid, jointly learning from interactions and features.

**Knowledge-based systems** use explicit domain knowledge (ontologies, constraint rules) to match user requirements to item properties. These are common in high-stakes domains like financial advisory where users can articulate specific needs.

### By Model Family

| Family | Scoring Function $f(u, i; \theta)$ | Key Property |
|--------|-------------------------------------|--------------|
| Neighborhood | $\sum_{v \in \mathcal{N}(u)} \text{sim}(u,v) \cdot R_{vi}$ | Non-parametric, interpretable |
| Matrix Factorization | $\mathbf{p}_u^\top \mathbf{q}_i + b_u + b_i$ | Bilinear, efficient |
| Neural CF | $\text{MLP}([\mathbf{p}_u ; \mathbf{q}_i])$ | Universal approximator |
| Sequential | $g(\mathbf{p}_u, \mathbf{q}_i, \mathbf{h}_t)$ | Captures temporal dynamics |
| Two-Tower | $\phi_u(u)^\top \phi_i(i)$ | Scalable retrieval |
| Graph-Based | Message passing on user-item graph | Captures higher-order structure |

### The Expressiveness–Efficiency Tradeoff

There is a fundamental tension between model expressiveness and computational efficiency:

- **Dot-product models** (MF, two-tower) support efficient approximate nearest neighbor (ANN) retrieval over millions of items, enabling sub-millisecond inference. But they can only model interactions that decompose into inner products.
- **Cross-feature models** (NCF, deep hybrids) can model arbitrary nonlinear interactions but require scoring every candidate item, making them expensive for large catalogs.

Production systems typically use a **two-stage architecture**: a fast retrieval model (dot-product based) generates a candidate set of hundreds of items, followed by a more expressive ranking model that re-scores the candidates.

## The Cold-Start Problem

The cold-start problem arises when the system must make recommendations for users or items with no (or very few) interaction history:

**New user cold-start**: A user who just signed up has no rating history. Collaborative filtering cannot generate predictions because there are no observed entries in the user's row of $R$.

**New item cold-start**: A newly added item has no ratings. CF methods cannot recommend it because its column in $R$ is entirely empty.

**System cold-start**: When the entire system is new, there is no interaction data at all.

Each model family addresses cold-start differently:

| Method | New Users | New Items |
|--------|-----------|-----------|
| Collaborative Filtering | ✗ Cannot predict | ✗ Cannot predict |
| Content-Based | Partial (needs some history) | ✓ Uses item features |
| Hybrid | ✓ Falls back to content | ✓ Falls back to content |
| Sequential | ✗ Needs interaction sequence | ✗ Cannot predict |
| Two-Tower with features | ✓ Uses user features | ✓ Uses item features |

The cold-start problem is one of the primary motivations for moving beyond pure collaborative filtering to content-aware and hybrid architectures.

## Design Considerations for Production Systems

### Scale

Production recommender systems operate at massive scale. Netflix serves 200M+ subscribers with a catalog of thousands of titles. YouTube recommends from a corpus of billions of videos. This scale imposes architectural constraints:

- **Retrieval + ranking**: Two-stage (or multi-stage) architecture separating fast candidate generation from expensive re-ranking.
- **Approximate nearest neighbors**: FAISS, ScaNN, or similar libraries for sub-linear retrieval from embedding spaces.
- **Feature stores**: Precomputed user and item features served with low latency.

### Bias and Fairness

Recommender systems amplify existing biases in the interaction data:

- **Popularity bias**: Popular items receive more exposure, generating more interactions, which makes them even more popular — a feedback loop.
- **Position bias**: Items shown at the top of a list receive more clicks regardless of relevance.
- **Demographic bias**: Underrepresented user groups may receive lower-quality recommendations due to sparse interaction data.

Addressing these biases requires careful evaluation (disaggregated metrics), debiasing techniques (inverse propensity scoring, causal inference), and diversity-aware re-ranking.

### Exploration vs Exploitation

A pure exploitation strategy recommends items the model is most confident the user will like, based on past behavior. This can lead to narrowing recommendations ("filter bubble"). Exploration strategies deliberately recommend uncertain or novel items to gather new information and broaden user experience. Balancing exploration and exploitation is critical for long-term recommendation quality.

## The Progression of This Chapter

This chapter follows a natural progression of increasing model complexity:

1. **Collaborative Filtering Fundamentals** (Section 29.1.2): Memory-based methods that directly use the rating matrix — intuitive but limited by sparsity.

2. **Matrix Factorization** (Section 29.1.3): The workhorse of recommendation — learns low-dimensional embeddings via `nn.Embedding`, introducing bias decomposition for systematic effects.

3. **Neural Collaborative Filtering** (Section 29.2.1): Replaces the dot product with an MLP, enabling nonlinear interactions and cross-dimensional feature crossing.

4. **Content-Based and Hybrid Methods** (Sections 29.2.2–29.2.3): Incorporates item and user features to address cold-start and enrich representations.

5. **Sequential Recommendations** (Section 29.2.4): Models temporal dynamics in user behavior using RNNs and Transformers.

6. **Embedding-Based RecSys** (Section 29.2.5): Scales to production with two-tower architectures and approximate nearest neighbor retrieval.

7. **Evaluation** (Section 29.3.1): Covers both rating prediction and ranking metrics with proper offline evaluation protocols.

Each section builds on the previous, and the PyTorch implementations share a common data pipeline using the MovieLens dataset.

## Summary

Recommender systems are a rich application of representation learning, connecting embedding methods, sequence modeling, and information retrieval. The field has evolved from simple neighborhood methods through matrix factorization to deep neural architectures, driven by the need to handle sparsity, cold-start, and scale. Understanding the tradeoffs between different model families — their inductive biases, computational costs, and information requirements — is essential for both research and deployment.

---

## Exercises

1. **Sparsity analysis**: For a rating matrix with $m$ users, $n$ items, and average $k$ ratings per user, derive the density $\rho = k / n$. For MovieLens-100K ($m = 600$, $n = 9{,}000$, $|\Omega| = 100{,}000$), compute the expected number of co-rated items between two random users. Under what density does user-based CF become impractical (expected overlap < 5)?

2. **Feedback comparison**: Consider a music streaming service. List three types of implicit feedback available (beyond play counts). For each, discuss what preference signal it carries and potential biases.

3. **Cold-start strategies**: A new e-commerce platform launches with a product catalog of 10,000 items and zero user interaction data. Design a recommendation strategy that provides reasonable suggestions from day one and transitions to collaborative filtering as data accumulates.

4. **Two-stage architecture**: Explain why a production recommender system cannot simply score all items with a neural CF model for each request. Sketch a two-stage retrieval + ranking architecture and identify which model families are appropriate for each stage.

5. **Bias amplification**: Suppose 10% of items receive 90% of all interactions (a power-law distribution). Explain how training a collaborative filtering model on this data amplifies popularity bias. Propose two techniques to mitigate this effect.
