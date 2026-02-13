# Ranking Metrics

## Overview

Recommender systems are fundamentally ranking problems: the goal is to rank items such that relevant items appear at the top of the list. Ranking metrics evaluate this ordering quality.

## Precision@K and Recall@K

$$\text{Precision@K} = \frac{|\text{relevant items in top K}|}{K}$$

$$\text{Recall@K} = \frac{|\text{relevant items in top K}|}{|\text{all relevant items}|}$$

## Normalized Discounted Cumulative Gain (nDCG)

Accounts for the position of relevant items, giving higher credit to items ranked higher:

$$\text{DCG@K} = \sum_{i=1}^{K} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}$$

$$\text{nDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

where IDCG is the DCG of the ideal ranking.

## Mean Average Precision (MAP)

$$\text{AP} = \frac{1}{|\text{rel}|} \sum_{k=1}^{K} \text{Precision@k} \cdot \text{rel}(k)$$

$$\text{MAP} = \frac{1}{|U|} \sum_{u \in U} \text{AP}_u$$

## Hit Rate (HR@K)

$$\text{HR@K} = \frac{1}{|U|} \sum_{u} \mathbb{1}[\text{relevant item in top K}]$$

## Mean Reciprocal Rank (MRR)

$$\text{MRR} = \frac{1}{|U|} \sum_{u} \frac{1}{\text{rank}_u}$$

where $\text{rank}_u$ is the position of the first relevant item for user $u$.

## Implementation

```python
def ndcg_at_k(predictions, relevances, k):
    _, indices = predictions.topk(k)
    dcg = (relevances[indices] / torch.log2(torch.arange(2, k+2).float())).sum()
    
    ideal_rel, _ = relevances.sort(descending=True)
    idcg = (ideal_rel[:k] / torch.log2(torch.arange(2, k+2).float())).sum()
    
    return (dcg / idcg).item() if idcg > 0 else 0.0
```
