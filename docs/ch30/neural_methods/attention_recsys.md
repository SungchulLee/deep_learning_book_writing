# Attention-Based Recommender Systems

## Introduction

Attention mechanisms enable neural recommender systems to focus on the most relevant aspects of user history, item features, and contextual information when generating recommendations. Rather than treating all historical interactions equally, attention-based systems learn dynamic weights indicating which past items most influence current recommendations. This enables capturing long-range dependencies in sequential user behavior and handling variable-length sequences efficiently.

In quantitative finance, attention mechanisms prove particularly valuable for capturing non-stationary user preferences and market regimes: an investor's preference for growth stocks may intensify during bull markets while losing salience during downturns. Attention over historical holdings, recent trades, and current market conditions allows recommendation systems to adapt dynamically to changing contexts.

This section develops attention mechanisms for recommendation, explores architectures combining attention with collaborative filtering, and demonstrates financial applications.

## Key Concepts

### Attention Fundamentals
- **Query**: Current recommendation context (who to recommend to, what context)
- **Key/Value**: Historical information (past purchases, item features)
- **Attention Weights**: Learned scores indicating relevance of each key
- **Output**: Weighted aggregation of values guided by attention

### Attention Types for Recommendations
- **Item Attention**: Which historical items influence current preference
- **Sequential Attention**: Which position in sequence most important
- **Feature Attention**: Which item features most relevant
- **Cross-Attention**: Interaction between users and items

## Mathematical Framework

### Basic Attention Mechanism

For query q, keys K, and values V:

$$\text{Attention}(q, K, V) = \text{softmax}\left(\frac{q K^T}{\sqrt{d}}\right) V$$

Attention weights:

$$\alpha_i = \frac{\exp(q \cdot k_i / \sqrt{d})}{\sum_j \exp(q \cdot k_j / \sqrt{d})}$$

Output: Weighted combination of values:

$$\text{Output} = \sum_i \alpha_i v_i$$

### Multi-Head Attention

Attend to multiple aspects simultaneously:

$$\text{MultiHead}(q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

where:

$$\text{head}_i = \text{Attention}(q W_i^Q, K W_i^K, V W_i^V)$$

Each head learns different aspects (e.g., short-term vs long-term preferences).

### Self-Attention for Sequences

Attend within sequence itself:

$$q = K = V = \text{Item embeddings}$$

Enables item-to-item attention, modeling how past items influence each other.

### Cross-Attention for User-Item Matching

User (query) attends to items (keys/values):

$$q = \text{user embedding}, K = V = \text{item embeddings}$$

Computes relevance of each item to user.

## Attention-Based Recommender Architectures

### Sequential Attention for Purchase History

Learn temporal dynamics of user preferences via attention over purchase sequence:

1. **Embed historical items**: $h_t = \text{embed}(i_t)$ for purchases $i_1, \ldots, i_T$
2. **Compute attention**: Query (current context) attends to historical embeddings
3. **Output**: Weighted historical context, combined with item embeddings

Prediction:

$$\hat{r}_{u,i} = \text{MLP}(\text{attention context}, \text{item embedding})$$

### Transformer-Based Recommendations

Stack multi-head self-attention layers:

```
Input: Item embeddings h₀
Layer 1: MultiHeadAttention(h₀) → h₁
Layer 2: MultiHeadAttention(h₁) → h₂
...
Layer L: MultiHeadAttention(h_{L-1}) → h_L
Output: Predict user preference from h_L
```

Enables capturing complex item dependencies.

### Attention-Based Collaborative Filtering

Combine user and item attention:

1. **Item Attention**: User attention over items (what items matter to user)
2. **User Attention**: Item attention over similar users (what users matter for item)
3. **Joint**: Both attentions interact in prediction

$$\hat{r}_{u,i} = f(\text{ItemAttention}(u) + \text{UserAttention}(i))$$

## Financial Applications of Attention

### Portfolio Recommendation with Attention

For investor with 10-stock portfolio, recommend next addition:

1. **Historical Items**: Embed owned stocks
2. **Item Features**: Sector, dividend yield, risk metrics
3. **Attention**: Determine which owned stocks most similar to candidate stock
4. **Prediction**: Recommend stocks similar to heavily-attended portfolio positions

**Interpretation**: "Recommend Tech Dividend ETF because your portfolio heavily weights dividend tech stocks (Apple, Microsoft)."

### Regime-Aware Attention

Modulate attention based on market regime:

$$\alpha_i^{\text{regime}} = \text{softmax}(\alpha_i \cdot \text{regime factor})$$

Bullish regime: Upweight stocks performing well recently
Bear regime: Upweight defensive stocks

### Temporal Attention

Recent trades more relevant than distant past:

$$\alpha_t = \text{softmax}(\alpha_t^{\text{base}} \times \exp(-\lambda(T - t)))$$

Exponential decay ensures recent items weighted more.

## Implementation Details

### Positional Encoding

Transformers need position information (unlike RNNs with inherent sequence order):

$$\text{pos\_encoding}(t, 2d) = \sin(t / 10000^{2d/d_{model}})$$
$$\text{pos\_encoding}(t, 2d+1) = \cos(t / 10000^{2d/d_{model}})$$

Add to embeddings before attention.

### Layer Normalization and Residuals

Stabilize training:

$$\text{Output} = \text{LayerNorm}(x + \text{Attention}(x))$$

Residual connection and normalization improve gradient flow.

### Masking for Causal Attention

For sequential prediction, mask future items (information leakage):

$$\text{Attention}(q, K, V) = \text{softmax}\left(\frac{qK^T}{\sqrt{d}} + M\right) V$$

where M masks future positions with -∞.

## Attention Interpretability

### Visualization

Plot attention weights as heatmap:

```
              Item 1  Item 2  Item 3  Item 4
User A        0.5     0.3     0.1     0.1
User B        0.1     0.1     0.6     0.2
User C        0.2     0.2     0.2     0.4
```

Shows which items matter to each user.

### Attention Explanations

Generate explanations from attention weights:

```
Recommend: Apple Stock
Attention scores:
- AAPL (currently hold): 0.6 ← highest
- MSFT (currently hold): 0.3
- IBM (previously sold): 0.05

Explanation: "You're highly engaged with Apple stock 
in your portfolio. Similar tech companies recommended."
```

## Training Considerations

### Data Requirements

Attention-based models have many parameters; require substantial training data:

- Minimum: 50-100 interactions per user
- Preferred: 500+ interactions per user
- Optimal: 5000+ interactions

### Computational Cost

Multi-head attention scales as O(L² × d) where L = sequence length, d = embedding dim.

For long sequences (L > 1000), becomes expensive. Remedies:

1. **Truncation**: Only attend to recent 100-200 items
2. **Sparse Attention**: Attend to only k nearest items
3. **Linear Attention**: Approximate softmax with linear operations

### Optimization

Train with Adam optimizer, learning rate scheduling:

- Initial learning rate: 1e-3
- Decay: Linear or exponential
- Warmup: Gradual increase first 10% of training

## Evaluation

### Accuracy Metrics

Standard NDCG, AUC on hold-out test set. Attention models typically match or exceed RNN baselines.

### Attention Quality Evaluation

**Ablation Study**: Remove attention mechanism; measure accuracy drop.

Significant drop (>5%) indicates attention genuinely helps.

**Attention Consistency**: Do similar users have similar attention patterns?

High consistency suggests learned patterns meaningful.

### Online Evaluation

Deploy attention model with A/B test:
- Control: Baseline recommender
- Treatment: Attention-based recommender
- Metrics: CTR, engagement, conversion

## Case Study: Analyst Research Attention

### Problem

Recommend research papers to equity analyst with large read history.

### Solution

**Attention Model**:

1. **Embed** papers analyst read (title, abstract, sector, author)
2. **Sequence**: Organize reads chronologically
3. **Self-Attention**: Learn which papers influence each other
4. **Cross-Attention**: Query (analyst profile) attends to papers
5. **Predict**: Candidate papers scored via cross-attention

### Results

- NDCG@5: 0.68 (vs 0.60 baseline RNN)
- 12% improvement in engagement (clicks on recommendations)
- Attention shows analyst focused on tech/healthcare sectors (matches coverage)

### Explanation Example

```
Recommended: "Cloud Database Architecture"
Attention insights:
- Analyst recently read (0.7 attention): "Serverless Computing Trends"
- Analyst read frequently (0.5 attention): "Database Benchmarks"
- Recommendation: Similar technical papers from past interests
```

## Advanced Topics

### Hierarchical Attention

Multiple levels of attention:
- **Document Level**: Attend to chapters within papers
- **Sentence Level**: Attend to important sentences within chapters
- **Word Level**: Attend to key terms

Enables hierarchical representation learning.

### Cross-Modal Attention

Attend across different modalities (text, price data, sentiment):

$$\text{Attention}_{\text{cross-modal}}(q_{\text{text}}, K_{\text{price}}, V_{\text{sentiment}})$$

Enables reasoning across data types.

!!! note "Attention for Recommendations"
    Attention mechanisms excel at capturing user preference dynamics and providing interpretable recommendations. Particularly valuable when:
    - Sequential patterns important (user preference evolution)
    - Explainability critical (finance, healthcare)
    - Item features complex (multiple attributes)
    
    Requires substantial data and computational resources; typically worthwhile for large-scale systems.

