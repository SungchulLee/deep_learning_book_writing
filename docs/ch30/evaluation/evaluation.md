# Evaluating Recommender Systems

## Learning Objectives

- Distinguish between rating prediction and ranking evaluation paradigms
- Compute MSE, RMSE, MAE for rating prediction tasks
- Understand ranking metrics: Precision@K, Recall@K, NDCG, MAP
- Design proper offline evaluation protocols with temporal splits

## Two Evaluation Paradigms

Recommender system evaluation falls into two categories depending on the task:

| Paradigm | Question | Metrics | Use Case |
|----------|----------|---------|----------|
| **Rating Prediction** | How accurately can we predict the rating? | MSE, RMSE, MAE | Explicit feedback (1–5 stars) |
| **Ranking** | Are the top-$K$ recommendations relevant? | Precision@K, Recall@K, NDCG, MAP | Implicit feedback (clicks, purchases) |

The implementations in this chapter evaluate using MSE (rating prediction). We cover both paradigms.

## Rating Prediction Metrics

### Mean Squared Error (MSE)

$$\text{MSE} = \frac{1}{|\Omega_{\text{test}}|} \sum_{(u,i) \in \Omega_{\text{test}}} \bigl(R_{ui} - \hat{R}_{ui}\bigr)^2$$

This is the loss function used during training (`F.mse_loss`). Lower is better.

### Root Mean Squared Error (RMSE)

$$\text{RMSE} = \sqrt{\text{MSE}}$$

RMSE has the same units as the ratings (e.g., stars), making it more interpretable. A RMSE of 0.9 on a 1–5 scale means predictions are off by about 1 star on average.

### Mean Absolute Error (MAE)

$$\text{MAE} = \frac{1}{|\Omega_{\text{test}}|} \sum_{(u,i) \in \Omega_{\text{test}}} \bigl|R_{ui} - \hat{R}_{ui}\bigr|$$

MAE is less sensitive to outliers than MSE/RMSE. A rating prediction that is off by 4 stars contributes $16$ to MSE but only $4$ to MAE.

### Implementation

```python
def evaluate_rating_prediction(model, df_test, unsqueeze=False):
    """Compute MSE, RMSE, and MAE on test data."""
    model.eval()
    with torch.no_grad():
        users = torch.LongTensor(df_test.userId.values)
        items = torch.LongTensor(df_test.movieId.values)
        ratings = torch.FloatTensor(df_test.rating.values)
        if unsqueeze:
            ratings = ratings.unsqueeze(1)
        
        preds = model(users, items)
        if unsqueeze:
            preds = preds.squeeze()
            ratings = ratings.squeeze()
        
        mse = F.mse_loss(preds, ratings).item()
        rmse = mse ** 0.5
        mae = F.l1_loss(preds, ratings).item()
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae}
```

## Ranking Metrics

When the goal is to produce a **ranked list** of recommendations (not predict exact ratings), we use information retrieval metrics.

### Setup

For each user $u$, let:

- $\text{Rec}_K(u)$: the top-$K$ recommended items
- $\text{Rel}(u)$: the set of relevant items (e.g., items rated $\geq 4$)

### Precision@K

$$\text{Precision@K}(u) = \frac{|\text{Rec}_K(u) \cap \text{Rel}(u)|}{K}$$

The fraction of recommended items that are relevant. Averaged over all users:

$$\text{Precision@K} = \frac{1}{m} \sum_{u=1}^m \text{Precision@K}(u)$$

### Recall@K

$$\text{Recall@K}(u) = \frac{|\text{Rec}_K(u) \cap \text{Rel}(u)|}{|\text{Rel}(u)|}$$

The fraction of relevant items that appear in the top-$K$ list.

### Normalized Discounted Cumulative Gain (NDCG@K)

NDCG accounts for the **position** of relevant items — a relevant item at rank 1 is more valuable than at rank 10.

$$\text{DCG@K}(u) = \sum_{k=1}^K \frac{2^{\text{rel}(u, \pi(k))} - 1}{\log_2(k + 1)}$$

where $\pi(k)$ is the item at rank $k$ and $\text{rel}(u, i)$ is the relevance of item $i$ to user $u$ (e.g., the rating, or a binary relevant/irrelevant indicator).

$$\text{NDCG@K}(u) = \frac{\text{DCG@K}(u)}{\text{IDCG@K}(u)}$$

where IDCG@K is the DCG of the ideal (perfectly sorted) ranking.

### Mean Average Precision (MAP)

$$\text{AP@K}(u) = \frac{1}{\min(K, |\text{Rel}(u)|)} \sum_{k=1}^K \text{Precision@k}(u) \cdot \mathbf{1}[\pi(k) \in \text{Rel}(u)]$$

$$\text{MAP@K} = \frac{1}{m} \sum_{u=1}^m \text{AP@K}(u)$$

MAP rewards models that place relevant items at the top of the list.

### Implementation

```python
def precision_recall_at_k(model, df_test, df_train, k=10, threshold=4.0):
    """
    Compute Precision@K and Recall@K.
    
    Args:
        threshold: minimum rating to consider an item "relevant"
    """
    model.eval()
    precisions, recalls = [], []
    
    users = df_test.userId.unique()
    for u in users:
        # Items the user actually liked in the test set
        relevant = set(
            df_test[(df_test.userId == u) & 
                     (df_test.rating >= threshold)].movieId.values
        )
        if len(relevant) == 0:
            continue
        
        # Items already seen in training
        seen = set(df_train[df_train.userId == u].movieId.values)
        
        # Score all unseen items
        all_items = set(df_test.movieId.unique()) - seen
        item_ids = torch.LongTensor(list(all_items))
        user_ids = torch.LongTensor([u] * len(all_items))
        
        with torch.no_grad():
            scores = model(user_ids, item_ids)
        
        # Top-K recommendations
        top_k_idx = scores.topk(min(k, len(all_items))).indices
        recommended = set(item_ids[top_k_idx].numpy())
        
        # Metrics
        hits = len(recommended & relevant)
        precisions.append(hits / k)
        recalls.append(hits / len(relevant))
    
    return np.mean(precisions), np.mean(recalls)
```

## Evaluation Protocols

### Random Split (Simple but Flawed)

The source code uses a random 80/20 split:

```python
np.random.seed(3)
msk = np.random.rand(len(data)) < 0.8
train = data[msk].copy()
val = data[~msk].copy()
```

This is simple and reproducible but ignores temporal ordering — the model might train on future ratings to predict past ones.

### Temporal Split (Recommended)

In production, recommendations must predict future behavior from past data. A temporal split respects this:

```python
# Sort by timestamp, use first 80% for training
data_sorted = data.sort_values('timestamp')
split_idx = int(len(data_sorted) * 0.8)
train = data_sorted.iloc[:split_idx]
val = data_sorted.iloc[split_idx:]
```

### Leave-One-Out

For each user, hold out their most recent interaction for testing:

```python
test = data.groupby('userId').apply(
    lambda x: x.nlargest(1, 'timestamp')
)
train = data.drop(test.index.get_level_values(1))
```

This is common for implicit feedback evaluation.

### Cross-Validation

$k$-fold cross-validation provides more robust estimates but is expensive for large datasets. It is most useful during model selection on smaller benchmarks.

## Pitfalls in Recommender Evaluation

### 1. Popularity Bias

Popular items dominate random test sets. A model that simply recommends the most popular items can achieve deceptively high metrics. Evaluate separately on popular vs long-tail items.

### 2. Missing-Not-At-Random (MNAR)

Users don't rate items at random — they rate items they chose to watch. The test set is biased toward items users were predisposed to like. This makes all offline metrics optimistic.

### 3. Metric Disagreement

A model can have low RMSE (good rating prediction) but poor NDCG (bad ranking). This happens when the model is accurate for middling ratings but poor at distinguishing top-rated items from merely good ones. Choose metrics that align with the deployment objective.

### 4. Online vs Offline

Offline metrics (computed on held-out data) are necessary but not sufficient. The ultimate evaluation is an **A/B test** in production, measuring engagement, click-through rate, conversion, or revenue.

## Summary of Metrics

| Metric | Task | Sensitive to | Range |
|--------|------|-------------|-------|
| MSE | Rating prediction | Outliers | $[0, \infty)$ |
| RMSE | Rating prediction | Outliers | $[0, \infty)$ |
| MAE | Rating prediction | Uniform errors | $[0, \infty)$ |
| Precision@K | Ranking | False positives | $[0, 1]$ |
| Recall@K | Ranking | False negatives | $[0, 1]$ |
| NDCG@K | Ranking | Position + relevance | $[0, 1]$ |
| MAP@K | Ranking | Position | $[0, 1]$ |

## Summary

Evaluating recommender systems requires choosing metrics aligned with the deployment objective. Rating prediction uses MSE/RMSE/MAE; ranking uses Precision@K, NDCG, and MAP. Temporal splits are preferred over random splits for realistic evaluation. Offline metrics provide necessary but not sufficient evidence — production A/B tests are the gold standard.

---

## Exercises

1. **Metric comparison**: Train MF, MF-bias, and NCF on MovieLens. Report MSE, RMSE, MAE, Precision@10, and NDCG@10. Do the models rank the same across all metrics?

2. **Temporal vs random split**: Compare validation MSE using random split vs temporal split. Which gives a lower (more optimistic) estimate? Why?

3. **Popularity baseline**: Implement a "most popular" recommender that always recommends the items with the highest average rating. Compute Precision@10 and NDCG@10. How close does this simple baseline come to learned models?

4. **Per-user analysis**: Compute RMSE separately for users with few ratings (<10) and many ratings (>50). Which group does each model serve better?

5. **Metric sensitivity**: Add Gaussian noise $\mathcal{N}(0, \sigma^2)$ to model predictions with increasing $\sigma$. Plot each metric vs $\sigma$. Which metrics degrade fastest?
