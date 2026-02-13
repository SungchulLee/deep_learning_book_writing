# Beyond Accuracy Metrics for Recommender Systems

## Introduction

Traditional recommender system evaluation focuses on accuracy metrics—how often recommended items match user preferences (RMSE, MAE, NDCG, AUC). However, accuracy alone fails to capture the diverse objectives underlying recommendation problems. A highly accurate recommender that recommends the same popular items to everyone provides poor user experience and little value; a recommender that suggests items with mediocre match but high novelty might drive discovery and engagement.

Modern recommendation systems require multi-dimensional evaluation accounting for accuracy, novelty, diversity, coverage, serendipity, and fairness. Each dimension captures important aspects: accuracy measures prediction fidelity; novelty measures degree of surprise; diversity measures variety within recommendations; coverage measures percentage of catalog discovered; serendipity measures value of unexpected discoveries. This section develops comprehensive evaluation frameworks extending beyond accuracy, addresses trade-offs between objectives, and explores practical metric implementations.

## Key Concepts

### Multidimensional Quality
- **Accuracy**: Correctness of predicted ratings/relevance
- **Novelty**: Degree to which recommendations differ from obvious/popular
- **Diversity**: Variety of recommended items (dissimilarity)
- **Coverage**: Percentage of catalog included in recommendations
- **Serendipity**: Unexpectedness combined with satisfaction

### Business Metrics
- **Engagement**: Click-through rate, session length, return visits
- **Conversion**: Purchase rate, add-to-cart rate
- **Retention**: User retention, churn rate
- **Revenue**: Revenue per user, profit metrics

## Mathematical Framework

### Accuracy Metrics (Baseline)

For predicted rating $\hat{r}_{ui}$ vs true rating r_{ui}:

$$\text{RMSE} = \sqrt{\frac{1}{|T|}\sum_{(u,i) \in T} (\hat{r}_{ui} - r_{ui})^2}$$

For ranked recommendations with relevance labels y_{ui} ∈ {0,1}:

$$\text{NDCG@k} = \frac{1}{|Z|}\sum_u \frac{\text{DCG}_k(u)}{\text{IDCG}_k(u)}$$

where $\text{DCG}_k(u) = \sum_{i=1}^k \frac{y_{u,\pi(i)}}{log_2(i+1)}$.

### Novelty Metric

Novelty of item i: inverse of its popularity in training set:

$$\text{Novelty}(i) = -\log_2 \frac{\text{popularity}(i)}{|U|}$$

where popularity(i) = number of users who rated/interacted with i. Average novelty of recommendation set:

$$\text{Novelty}(R_u) = \frac{1}{|R_u|}\sum_{i \in R_u} \text{Novelty}(i)$$

### Diversity Metric

Pairwise similarity between recommended items:

$$\text{Diversity}(R_u) = 1 - \frac{1}{\binom{|R_u|}{2}}\sum_{i,j \in R_u, i \neq j} \text{sim}(i, j)$$

where sim(i,j) ∈ [0,1] is content similarity. Diversity = 1 indicates completely dissimilar recommendations; 0 indicates identical items.

### Coverage Metric

**Catalog Coverage**: Percentage of items ever recommended:

$$\text{Coverage} = \frac{|\{i : \exists u, i \in R_u\}|}{|I|}$$

**Inter-list Diversity**: How much recommendations vary across users:

$$\text{ILD} = \frac{2}{|U|(|U|-1)}\sum_{u_1 < u_2} \text{Dissimilarity}(R_{u_1}, R_{u_2})$$

### Serendipity Metric

Serendipity: unexpectedness + satisfaction. For user u and item i:

$$\text{Serendipity}(u, i) = \mathbb{1}[\text{rated}(u,i) = 1] \times (1 - P(i | \text{history}_u))$$

where P(i|history_u) is probability of user liking i based on history. Average serendipity:

$$\text{Serendipity} = \frac{1}{|R|}\sum_{(u,i) \in R} \text{Serendipity}(u, i)$$

## Trade-offs Between Objectives

### Accuracy vs Novelty

Improving accuracy typically decreases novelty (novel items harder to predict):

$$\rho(\text{Accuracy}, \text{Novelty}) \approx -0.3 \text{ to } -0.7$$

**Trade-off Curve**: Explicit ranking solution adjusts weights:

$$\max_{\sigma} w_{\text{acc}} \cdot \text{NDCG} + (1-w_{\text{acc}}) \cdot \text{Novelty}$$

where w_acc ∈ [0,1] controls priority. Higher w_acc → more accurate but less novel.

### Coverage vs Accuracy

Recommending long-tail items improves coverage but hurts accuracy (tail items less predictable):

$$\text{Coverage} = 1 - \text{Gini}(\text{recommendation frequency})$$

where Gini ∈ [0,1] measures inequality. Perfect coverage (Gini=0) typically achieves 5-10% accuracy loss.

### Diversity vs Accuracy

Encouraging diversity forces inclusion of dissimilar items, reducing overall relevance:

$$\text{Correlation}(\text{Diversity}, \text{Accuracy}) \approx -0.4$$

**Mitigation**: Post-processing diversification applied to top-k candidates minimizes accuracy loss.

## Fairness Metrics

### Provider Fairness (Supplier/Item Fairness)

Ensure long-tail items receive fair exposure:

$$\text{Exposure}(i) = \frac{\sum_u \mathbb{1}[i \in R_u]}{|U|}$$

Ideal: Exposure proportional to user interest. Reality: Popular items overexposed.

**Gini Coefficient** measures inequality:

$$\text{Gini} = \frac{1}{n-1}\sum_{i=1}^n (2i - n - 1) \frac{x_i}{X}$$

where x_i are exposures sorted. Gini ∈ [0,1]; 0=perfect equality.

### User Fairness

Ensure recommendations equally helpful across user groups:

$$\text{Fairness} = 1 - \frac{\max_g \text{NDCG}_g - \min_g \text{NDCG}_g}{\text{avg NDCG}}$$

where g indexes user groups (demographics, activity level). Value 1 = fair; 0 = unfair.

## Practical Metric Implementation

### Offline Evaluation

Compute metrics using held-out test set:

1. **Train/Test Split**: 80/20 temporal split (earlier data trains; later tests)
2. **Generate Recommendations**: Recommend k items per user
3. **Compute Metrics**: Calculate accuracy, novelty, diversity, coverage
4. **Aggregate**: Report mean ± std across users

### Online Evaluation (A/B Testing)

Deploy recommender in production; measure business metrics:

1. **Control Group**: Current recommender
2. **Treatment Group**: New recommender
3. **Metrics**: Click-through rate, session length, conversion rate, revenue

Requires thousands of users for statistical significance.

### Metric Weights and Composite Scores

Combine multiple metrics into single score:

$$\text{Score} = w_1 \cdot \text{NDCG} + w_2 \cdot \text{Diversity} + w_3 \cdot \text{Novelty} + w_4 \cdot \text{Fairness}$$

Weights chosen based on business objectives. Example:
- E-commerce (maximize sales): w_NDCG=0.4, w_Diversity=0.2, w_Novelty=0.2, w_Fairness=0.2
- Discovery platform (maximize engagement): w_Novelty=0.4, w_Diversity=0.3, w_NDCG=0.2, w_Fairness=0.1

## Financial Recommendations Specific Metrics

### Recommendation Correctness

For financial products (funds, stocks, services):

$$\text{Suitability}(u, i) = \mathbb{1}[\text{riskProfile}(u) \approx \text{riskProfile}(i)]$$

Unsuitable recommendations carry regulatory risk; metric should heavily weighted.

### Risk-Adjusted Returns

If recommendations used for investing:

$$\text{Sharpe Ratio} = \frac{\text{Return} - r_f}{\text{Volatility}}$$

Measure performance of recommended portfolio vs benchmark.

### Information Ratio

For factor-based recommendations:

$$\text{IR} = \frac{\text{Portfolio Return} - \text{Benchmark Return}}{\text{Tracking Error}}$$

Quantifies value-add of recommendations.

## Evaluation Framework Summary

| Metric | Range | Interpretation | Finance Priority |
|--------|-------|-----------------|-------------------|
| NDCG@10 | [0, 1] | Higher = better match to preference | High |
| Novelty | [0, ∞] | Higher = more novel | Medium |
| Diversity | [0, 1] | Higher = more variety | Low-Medium |
| Coverage | [0, 1] | Higher = more items recommended | Low |
| Suitability | [0, 1] | Higher = appropriate risk level | **Critical** |
| Fairness | [0, 1] | Higher = equitable across groups | Medium |

## Practical Recommendations

### Evaluation Procedure

1. **Select Primary Metric**: Business-driven (e.g., accuracy for accuracy-focused, Sharpe for trading)
2. **Select Secondary Metrics**: Address other objectives (novelty, diversity, fairness)
3. **Set Thresholds**: Establish minimum acceptable values
4. **Offline Test**: Validate offline before deployment
5. **Online Test**: A/B test to confirm online performance
6. **Monitor Continuously**: Track metrics post-deployment

### Reporting Standards

Always report:
- Multiple metrics (not just accuracy)
- Aggregation method (mean, median, percentiles)
- Statistical significance (confidence intervals)
- Trade-offs explicitly (e.g., "Novelty improved 20%, NDCG decreased 5%")

!!! warning "Metric Selection"
    Choosing evaluation metrics implicitly optimizes the recommender toward those metrics. Select metrics aligned with business objectives. Over-optimizing single metric (e.g., accuracy) can harm user experience on other dimensions. Regular metric reviews ensure continued alignment with business goals.

