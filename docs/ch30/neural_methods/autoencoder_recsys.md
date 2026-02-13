# Autoencoders for Recommender Systems

## Introduction

Autoencoders provide an elegant neural architecture for recommendation through unsupervised learning of latent user/item representations. Unlike collaborative filtering that explicitly models user-item interactions, autoencoders learn compressed representations by reconstructing user preference vectors (e.g., rating profiles across items). The bottleneck layer naturally captures latent factors driving preferences, enabling dimensionality reduction while preserving recommendation-relevant structure.

Variational Autoencoders (VAEs) extend basic autoencoders by imposing probabilistic structure—learning distributions over latent factors rather than point estimates—enabling uncertainty quantification, principled recommendation sampling, and improved generalization. In financial recommendation systems, VAE uncertainty estimates enable portfolio risk assessment and regulatory-compliant confidence-qualified recommendations.

This section develops autoencoder approaches for recommendations, explores architectural variants, and demonstrates financial applications.

## Key Concepts

### Autoencoder Recommendation Framework
- **Input**: User's item rating profile (implicit or explicit)
- **Encoder**: Maps ratings to latent user factors
- **Bottleneck**: Compressed representation of user preferences
- **Decoder**: Reconstructs predicted ratings from factors
- **Loss**: Reconstruction error on observed ratings

### Variational Autoencoders (VAE)
- **Probabilistic Latent**: Latent distribution q(z|x) instead of point estimates
- **KL Regularization**: Prior p(z) encourages well-behaved latent space
- **Sampling**: Generate recommendations by sampling from learned distribution
- **Uncertainty**: Posterior variance quantifies preference uncertainty

## Mathematical Framework

### Autoencoder Reconstruction

For user u with rating profile r_u ∈ ℝ^n:

**Encoder**:
$$z_u = \text{Encoder}(r_u) \in \mathbb{R}^k, \quad k \ll n$$

**Decoder**:
$$\hat{r}_u = \text{Decoder}(z_u) \in \mathbb{R}^n$$

**Loss**:
$$\mathcal{L}_{\text{AE}} = \sum_u \|r_u - \hat{r}_u\|^2$$

### Variational Autoencoder

Probabilistic model with latent distribution:

**Encoder (Variational Inference)**:
$$q_\phi(z_u | r_u) = \mathcal{N}(\mu_\phi(r_u), \sigma_\phi^2(r_u) I)$$

**Decoder (Generative Model)**:
$$p_\theta(r_u | z_u)$$

**Loss (ELBO)**:
$$\mathcal{L}_{\text{VAE}} = -\mathbb{E}_{q_\phi}[\log p_\theta(r_u | z_u)] + \text{KL}(q_\phi(z_u | r_u) \| p(z_u))$$

First term: reconstruction; second term: regularization toward prior N(0,I).

### Conditional Generation

Given latent z, probability of rating item i:

$$p(r_{ui} | z_u) = \text{Softmax}_i(\text{Decoder}(z_u))$$

For explicit ratings, typical choice: categorical distribution over {1, 2, 3, 4, 5}.

## Training Procedures

### Implicit Feedback Autoencoders

For binary feedback (click/no-click):

$$r_{ui} \in \{0, 1\}, \quad p(r_{ui} = 1 | z_u) = \sigma(\text{Decoder}_i(z_u))$$

**Loss**: Binary cross-entropy

$$\mathcal{L} = -\sum_{(u,i) \in E} \log(\sigma(\hat{r}_{ui})) - \sum_{(u,i) \notin E} \log(1 - \sigma(\hat{r}_{ui}))$$

where E is observed interaction set.

### Weighted Reconstruction

Down-weight negative samples (missing interactions harder to interpret):

$$\mathcal{L} = \sum_{(u,i) \in E} \log(\sigma(\hat{r}_{ui})) + w \sum_{(u,i) \notin E} \log(1 - \sigma(\hat{r}_{ui}))$$

Typical w = 0.01 to 0.1 (rare classes weighted less).

### Mini-Batch Training

Sample users and items efficiently:

1. Sample mini-batch of users
2. For each user, sample k positive items and k negative items
3. Compute loss on batch
4. Backpropagate, update parameters

## Autoencoder Architectures for Recommendation

### Dense Bottleneck VAE

Standard fully-connected VAE:

```
Input (1000): Full item rating vector
Hidden 1 (500): ReLU
Hidden 2 (200): ReLU
Bottleneck (20): Latent (μ, σ)
Hidden 2 (200): ReLU
Hidden 1 (500): ReLU
Output (1000): Sigmoid (predicted ratings)
```

Simple, interpretable, suitable for ~1000 items.

### Convolutional Bottleneck

For implicit feedback viewed as image, use CNN:

```
Input: Item rating heatmap
Conv1: 32 filters, 3×3 kernel
Conv2: 64 filters, 3×3 kernel
Flatten → Bottleneck (k dims) → Upsample
...
Output: Rating heatmap
```

Captures local item correlation structure.

### Recurrent Bottleneck

For sequential recommendations (time-aware):

```
Input sequence: [item_1, item_2, ..., item_T]
LSTM encode: Forward + backward passes
Bottleneck: LSTM hidden state (k dims)
LSTM decode: Reconstruct sequence
Output: Predicted next item
```

Captures temporal dynamics of preferences.

## Recommendation via Autoencoder

### Rating Prediction

After training, predict ratings for unobserved items:

$$\hat{r}_{ui} = \text{Decoder}_i(\text{Encoder}(r_u))$$

Recommend items with highest predicted ratings.

### Sampling-Based Recommendation

VAE enables sampling alternative preference profiles:

1. Encode user: $\mu_u, \sigma_u^2 = \text{Encoder}(r_u)$
2. Sample latent: $z \sim \mathcal{N}(\mu_u, \sigma_u^2)$
3. Decode: $\hat{r} = \text{Decoder}(z)$
4. Recommend high-rated items

Multiple samples reveal preference uncertainty.

### Nearest Neighbor in Latent Space

Recommend items liked by latent-similar users:

1. Encode all users: $z_u = \text{Encoder}(r_u)$ for all u
2. Find K nearest neighbors to user u in latent space
3. Recommend items those neighbors like

Implicit collaborative filtering via learned latent representations.

## Variational Autoencoder Applications

### Uncertainty Quantification

Posterior variance on latent factors quantifies preference uncertainty:

$$\text{Uncertainty}(u, i) = \text{Var}_{q(z|r_u)}[\text{Decoder}_i(z)]$$

High uncertainty indicates low confidence in recommendation.

### Risk-Aware Portfolio Recommendation

Use VAE uncertainty for portfolio recommendations:

$$\text{Score}(u, p) = \text{Expected Return} - \lambda \times \text{Uncertainty}(u, p)$$

Trade off expected return against preference uncertainty.

### Confidence-Qualified Recommendations

Report recommendations with confidence intervals:

```
Recommended: Tech ETF
Expected Rating: 4.2/5
Confidence Interval: [3.1, 5.0]  (high uncertainty)

Recommended: Dividend ETF
Expected Rating: 4.1/5
Confidence Interval: [3.9, 4.3]  (low uncertainty)
```

### Anomaly Detection

Reconstruction error indicates unusual preference profiles:

$$\text{Anomaly}_u = \|r_u - \text{Decoder}(\text{Encoder}(r_u))\|$$

High error suggests outlier preferences or data quality issues.

## Comparison with Alternatives

| Method | Accuracy | Interpretability | Speed | Scalability |
|--------|----------|-----------------|-------|------------|
| Matrix Factorization | 0.60 | Medium | High | Very High |
| VAE | 0.65 | Low | Medium | High |
| Attention | 0.68 | Medium | Low | Medium |
| Hybrid (VAE + CF) | 0.70 | Medium | Medium | High |

Autoencoders balance accuracy and interpretability; VAE uncertainty valuable for risk-aware recommendations.

## Financial Application: Mutual Fund Recommendation

### Problem

Recommend mutual funds to investors based on past fund holdings.

### Solution

**VAE Autoencoder**:

1. **Input**: Vector of 500 mutual funds, rating = # shares held (normalized)
2. **Encoder**: 3 hidden layers (500→200→50→20 latent)
3. **Bottleneck**: 20-dimensional latent factor
4. **Decoder**: 3 hidden layers reconstructing fund ratings
5. **Loss**: Weighted reconstruction (0.1 weight on unobserved funds)

### Training

- Data: 100,000 investor portfolios
- Training: 1 epoch = ~10 minutes on GPU
- Validation: 20,000 hold-out portfolios

### Recommendation Generation

For investor u:

1. Encode portfolio: $\mu_u, \sigma_u^2 = \text{Encoder}(r_u)$
2. Predict unobserved funds: $\hat{r}_{u,\text{unobserved}} = \text{Decoder}(\mu_u)$
3. Rank by predicted rating
4. Recommend top-k with highest ratings

### Results

- NDCG@10: 0.72 (vs 0.65 baseline matrix factorization)
- Correlation with acceptance: 0.68 (high confidence in predictions)
- Execution time: <100ms per recommendation (real-time capable)

### Confidence Quantification

Use VAE variance for confidence:

```python
# High confidence (low variance)
Low variance latent → Low reconstruction variance → 
High confidence recommendations

# Low confidence (high variance)
High variance latent → High reconstruction variance → 
Low confidence recommendations
```

Report uncertainty to users; caveat low-confidence recommendations.

## Implementation Considerations

### Handling Sparse Data

User-item matrices often sparse (user rated only tiny fraction of items).

Approaches:

1. **Selective Reconstruction**: Only reconstruct observed items + sample negatives
2. **Confidence Weighting**: Weight observed items higher in loss
3. **Implicit Feedback**: Treat unobserved as weak negative signal

### Cold Start Handling

New users without history: supplement with content features:

$$z_{\text{new}} = \text{Encoder}_{\text{hybrid}}(r_{\text{demographics}})$$

Or use prior: $z_{\text{new}} \sim \mathcal{N}(0, I)$ (sampled from prior).

### Computational Efficiency

For millions of items, full reconstruction infeasible.

Solutions:

1. **Sampling**: Recommend from sample of items, not full catalog
2. **Hierarchical**: Hierarchy of autoencoders (categories → items)
3. **Approximate**: Factorization of decoder for fast inference

## Practical Guidelines

### When to Use Autoencoders

Good for recommendation when:
- Implicit feedback primary data source
- Uncertainty quantification important
- Interpretability (via latent factors) desired
- Model size/complexity not bottleneck

Avoid when:
- Extreme sparsity (< 0.001% observed)
- Real-time constraints tight
- Cold start users dominant

### Architecture Selection

- **Small datasets**: Shallow network (1-2 hidden layers)
- **Medium datasets**: 2-3 hidden layers, bottleneck k=10-50
- **Large datasets**: Deeper networks, bottleneck k=50-200

Monitor validation loss; stop when plateaus.

!!! note "Autoencoders for Recommendations"
    Autoencoders provide flexible neural recommendation framework with built-in uncertainty via VAEs. Particularly suited for financial applications requiring confidence quantification and interpretable latent factors. Main limitation: cold start for new users; mitigate via hybrid approaches combining collaborative and content-based signals.

