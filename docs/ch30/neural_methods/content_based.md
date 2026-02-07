# Content-Based Filtering

## Learning Objectives

- Understand the content-based filtering paradigm and how it differs from collaborative filtering
- Build user profiles from item features and interaction history
- Implement content-based models using TF-IDF and neural feature extractors
- Identify the strengths and limitations of content-based approaches

## The Content-Based Paradigm

Content-based filtering recommends items **similar to what the user has previously liked**, using item features rather than other users' behavior.

### Formal Setup

Each item $i$ is described by a feature vector $\mathbf{x}_i \in \mathbb{R}^p$, which may include:

- **Categorical features**: genre, director, language (one-hot or multi-hot encoded)
- **Text features**: description, reviews (TF-IDF or neural embeddings)
- **Numerical features**: year, duration, budget
- **Derived features**: tags, keywords, sentiment scores

The user profile $\mathbf{w}_u \in \mathbb{R}^p$ is constructed from the features of items the user has interacted with. The predicted rating is:

$$\hat{R}_{ui} = f(\mathbf{w}_u, \mathbf{x}_i)$$

where $f$ is typically a similarity function or a learned model.

### Comparison with Collaborative Filtering

| Aspect | Collaborative Filtering | Content-Based |
|--------|------------------------|---------------|
| **Input** | User–item interaction matrix only | Item features + user history |
| **Cold-start (new items)** | Cannot recommend | Can recommend from features |
| **Cold-start (new users)** | Cannot recommend | Needs some history |
| **Diversity** | Can suggest unexpected items | Limited to similar items ("filter bubble") |
| **Domain knowledge** | None required | Requires feature engineering |
| **Cross-domain** | Limited to one interaction type | Features can span domains |

## Building User Profiles

### Explicit Profile Construction

Given user $u$'s rated items $I_u$ with ratings $R_{ui}$, construct a weighted user profile:

$$\mathbf{w}_u = \frac{\sum_{i \in I_u} R_{ui} \cdot \mathbf{x}_i}{\sum_{i \in I_u} R_{ui}}$$

This is a rating-weighted average of item feature vectors. Items rated highly contribute more to the profile.

### TF-IDF for Text Features

For item descriptions or metadata, TF-IDF (Term Frequency–Inverse Document Frequency) provides a principled weighting:

$$\text{TF-IDF}(t, i) = \text{TF}(t, i) \times \log\frac{N}{\text{DF}(t)}$$

where $\text{TF}(t, i)$ is the frequency of term $t$ in item $i$'s description, $\text{DF}(t)$ is the number of items containing term $t$, and $N$ is the total number of items.

Each item is represented as a sparse TF-IDF vector $\mathbf{x}_i \in \mathbb{R}^V$ where $V$ is the vocabulary size.

### Neural Feature Extraction

For richer representations, use pretrained models:

- **Text**: Sentence-BERT or similar encoders produce dense embeddings from item descriptions.
- **Images**: ResNet or ViT features from item images (e.g., movie posters).
- **Tabular**: Learned embeddings for categorical features combined with numerical features.

```python
# Example: Content-based model with pretrained text features
class ContentBasedModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        self.user_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.item_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, user_profile, item_features):
        u = self.user_net(user_profile)
        v = self.item_net(item_features)
        return (u * v).sum(1)
```

## Similarity Functions

### Cosine Similarity

The most common choice for content-based recommendations:

$$\text{sim}(\mathbf{w}_u, \mathbf{x}_i) = \frac{\mathbf{w}_u^\top \mathbf{x}_i}{\|\mathbf{w}_u\| \cdot \|\mathbf{x}_i\|}$$

Cosine similarity is invariant to vector magnitude, focusing only on the **direction** of the feature vectors.

### Learned Similarity

Instead of a fixed similarity function, learn one:

$$\hat{R}_{ui} = \sigma\bigl(\text{MLP}([\mathbf{w}_u ; \mathbf{x}_i])\bigr)$$

This is analogous to the NCF architecture but uses content features instead of learned embeddings.

## Advantages of Content-Based Methods

1. **No cold-start for items**: A new movie can be recommended immediately from its genre, cast, and description — no ratings needed.
2. **Transparency**: Recommendations are explainable: "Because you liked sci-fi movies with strong female leads..."
3. **User independence**: Each user's recommendations depend only on their own profile, not on other users.
4. **Domain adaptation**: Features can incorporate domain expertise (e.g., financial risk factors for investment recommendations).

## Limitations

1. **Filter bubble**: The model only recommends items similar to what the user already likes, limiting serendipitous discovery.
2. **Feature engineering**: Requires meaningful item features, which may be expensive to obtain or maintain.
3. **Overspecialization**: Cannot capture that "users who like A also tend to like B" when A and B have dissimilar features.
4. **New user cold-start**: Still requires some interaction history to build a user profile.

## Content-Based Filtering in Finance

Content-based methods are particularly natural in financial applications:

- **Fund recommendation**: Match investor risk profiles (features: risk tolerance, time horizon, sector preferences) to fund characteristics (features: volatility, sector exposure, expense ratio).
- **Research report recommendation**: Match analyst interests to reports using text similarity (TF-IDF or neural embeddings on report abstracts).
- **Bond recommendation**: Match portfolio needs to bond characteristics (duration, credit rating, yield, sector).

!!! info "Connection to Factor Models"
    Content-based filtering with linear similarity is closely related to **factor models** in finance. If item features are risk factors (market, size, value, momentum), then the user profile becomes a factor loading vector, and the recommendation score is a factor model prediction.

## Summary

Content-based filtering leverages item features to build user profiles and recommend similar items. It solves the item cold-start problem and provides transparent recommendations, but risks creating filter bubbles. The approach requires meaningful feature engineering and cannot capture collaborative signals. In practice, content-based methods are most powerful when combined with collaborative filtering in hybrid systems (Section 29.2.3).

---

## Exercises

1. **TF-IDF profiles**: Using the MovieLens dataset (which includes movie genres), construct TF-IDF feature vectors from genre tags. Build user profiles as weighted averages of movie genre vectors. Compute cosine similarity between a user profile and all unrated movies to generate recommendations.

2. **Feature comparison**: For the same user, compare recommendations from (a) collaborative filtering (MF) and (b) content-based (genre similarity). How much overlap is there? Which system suggests more diverse movies?

3. **Cold-start simulation**: Remove all ratings for 10 movies from the training set. Can MF predict ratings for these movies? Can a content-based model? Quantify the difference.

4. **Filter bubble measurement**: Define a diversity metric (e.g., average pairwise cosine distance among top-$k$ recommendations). Compare diversity between CF and content-based methods.
