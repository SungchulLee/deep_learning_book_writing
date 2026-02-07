# Hybrid Methods

## Learning Objectives

- Understand the motivation for combining collaborative and content-based signals
- Classify hybrid recommendation strategies (weighted, switching, feature augmentation, etc.)
- Implement feature-augmented collaborative filtering in PyTorch
- Analyze when hybrid methods provide the largest gains

## Why Hybrid?

Collaborative filtering and content-based methods have **complementary** strengths and weaknesses:

| Challenge | CF | Content-Based | Hybrid |
|-----------|----|--------------:|--------|
| Item cold-start | ✗ | ✓ | ✓ |
| User cold-start | ✗ | Partial | Partial |
| Sparsity | Struggles | Less affected | Robust |
| Serendipity | ✓ | ✗ | ✓ |
| Explainability | Limited | ✓ | Moderate |

Hybrid methods aim to mitigate the weaknesses of each approach while retaining their strengths.

## Taxonomy of Hybrid Strategies

Burke (2002) categorizes hybrid recommender systems into several design patterns:

### 1. Weighted Hybrid

Combine scores from CF and content-based models:

$$\hat{R}_{ui}^{\text{hybrid}} = \alpha \cdot \hat{R}_{ui}^{\text{CF}} + (1 - \alpha) \cdot \hat{R}_{ui}^{\text{CB}}$$

The weight $\alpha$ can be fixed, tuned on validation data, or adapted per user (e.g., lower $\alpha$ for new users with few ratings).

### 2. Switching Hybrid

Use one model or the other based on a condition:

$$\hat{R}_{ui}^{\text{hybrid}} = \begin{cases} \hat{R}_{ui}^{\text{CF}} & \text{if user } u \text{ has } \geq k \text{ ratings} \\ \hat{R}_{ui}^{\text{CB}} & \text{otherwise} \end{cases}$$

This directly addresses the cold-start problem: fall back to content-based when CF data is insufficient.

### 3. Feature Augmentation

Use the output of one model as input features for another. For example, use item embeddings learned by MF as additional features for a content-based model:

$$\hat{R}_{ui} = f\bigl(\mathbf{x}_i, \mathbf{q}_i^{\text{MF}}, \mathbf{w}_u\bigr)$$

### 4. Unified Model (Feature-Augmented CF)

The most common modern approach: augment the CF model with content features directly. Extend the NCF architecture to accept both ID embeddings and content features:

$$\hat{R}_{ui} = \text{MLP}\bigl([\mathbf{p}_u ; \mathbf{q}_i ; \mathbf{x}_i ; \mathbf{z}_u]\bigr)$$

where $\mathbf{p}_u, \mathbf{q}_i$ are learned embeddings and $\mathbf{x}_i, \mathbf{z}_u$ are content features.

### 5. Meta-Level Hybrid

One model's entire learned representation becomes input to another. For example, the content-based model learns a user profile, which replaces (or augments) the user embedding in a CF model.

## Implementation: Feature-Augmented NCF

```python
class HybridCollabFNet(nn.Module):
    """
    Hybrid model combining learned embeddings with content features.
    
    Concatenates:
    - Learned user embedding (from interaction data)
    - Learned item embedding (from interaction data)  
    - Item content features (genre, year, etc.)
    - Optional: user demographic features
    """
    def __init__(self, num_users, num_items, emb_size=100, 
                 item_feat_dim=20, n_hidden=64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        
        # Total input: user_emb + item_emb + item_features
        input_dim = emb_size * 2 + item_feat_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, n_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(n_hidden, n_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(n_hidden // 2, 1)
        )
    
    def forward(self, user_ids, item_ids, item_features):
        u = self.user_emb(user_ids)
        v = self.item_emb(item_ids)
        x = torch.cat([u, v, item_features], dim=1)
        return self.mlp(x)
```

### Design Considerations

**Feature normalization**: Content features should be normalized (e.g., standardized to zero mean and unit variance) before concatenation with embeddings, since embeddings are initialized with small values.

**Feature dimensionality balance**: If content features are high-dimensional (e.g., TF-IDF with $V = 10{,}000$), project them to a lower dimension first to avoid drowning out the learned embeddings:

```python
self.item_feat_proj = nn.Linear(raw_feat_dim, projected_dim)
```

**Embedding freezing**: For a two-stage approach, pretrain the CF embeddings on interaction data, then freeze them and train only the MLP with content features. This can improve stability when content features are noisy.

## When Hybrid Methods Help Most

Hybrid methods provide the largest gains when:

1. **Data is sparse**: Content features compensate for insufficient interaction data.
2. **Cold-start is frequent**: Many new users or items enter the system regularly.
3. **Item features are informative**: Rich metadata (text descriptions, images, structured attributes) is available.
4. **User segments differ**: Some users have rich interaction history (CF works well), others are new (content-based helps).

Conversely, if interaction data is abundant and item features are uninformative, pure CF may be sufficient.

## Hybrid Methods in Finance

Hybrid recommender systems are particularly valuable in financial applications where both interaction data and rich item features are available:

- **Investment product recommendation**: Combine collaborative signals (what similar investors hold) with content features (risk metrics, sector exposure, ESG scores) to recommend mutual funds or ETFs.
- **News and research**: Use collaborative filtering on reading patterns alongside content-based filtering on article text to recommend relevant financial news and analyst reports.
- **Client advisory**: Switch between CF (for clients with long history) and content-based (for new clients with a stated risk profile) in a wealth management context.

## Summary

Hybrid recommender systems combine collaborative filtering and content-based approaches to mitigate the weaknesses of each. The most practical modern approach is feature augmentation — extending a neural CF model with content features. The choice of hybridization strategy depends on the availability of features, the severity of cold-start, and computational constraints.

---

## Exercises

1. **Weighted hybrid**: Train separate MF and content-based models on MovieLens. Combine their predictions with weight $\alpha \in \{0.0, 0.2, 0.4, 0.6, 0.8, 1.0\}$. Plot validation MSE vs $\alpha$. Is the optimal $\alpha$ at an extreme?

2. **Cold-start experiment**: Partition users into "warm" (≥20 ratings) and "cold" (<5 ratings). Compare CF, content-based, and hybrid model performance on each group separately.

3. **Feature ablation**: Train the `HybridCollabFNet` with and without item content features. How much does adding genre information improve predictions?

4. **Two-stage training**: Pretrain MF embeddings, freeze them, then train a hybrid model that adds content features. Compare against end-to-end training. Which converges faster? Which achieves lower validation loss?
