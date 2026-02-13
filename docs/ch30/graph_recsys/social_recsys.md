# Social Graph-Based Recommender Systems

## Introduction

Social networks encode valuable information about user preferences and behavior: friends often share interests, users influence each other's decisions, trusted advisors guide recommendations. Social graph-based recommender systems leverage these relationships to generate recommendations assuming that users will prefer items liked by friends or influencers. Unlike collaborative filtering treating all users as independent, social recommenders exploit trust relationships and influence networks to improve accuracy, especially for cold-start users with few direct interactions.

In financial contexts, social signals prove particularly valuable: retail investors follow successful traders and analysts; institutional investors track peer portfolio behavior; financial advisors' recommendations depend on peer networks. Trust-based recommendations provide a principled way to incorporate social influence into financial product suggestions while maintaining regulatory compliance through documented reasoning.

This section develops social graph theory for recommendations, explores trust-aware algorithms, and demonstrates financial applications.

## Key Concepts

### Social Network Structure
- **Followers**: Users who trust/follow another user
- **Friends**: Mutual social connections
- **Influencers**: High-impact users shaping community opinions
- **Communities**: Clusters of densely-connected users

### Social Signals for Recommendation
- **Explicit Trust**: User explicitly rates others (e.g., 1-5 star trust ratings)
- **Implicit Trust**: Inferred from behavior (follows, shares, comments)
- **Co-engagement**: Users engaging with same content
- **Influence Scores**: Measures of user impact in network

### Trust Propagation
- **Direct Trust**: A trusts B explicitly
- **Transitive Trust**: A trusts B, B trusts C, infer A somewhat trusts C
- **Weighted Trust**: Trust strength varies with relationship type
- **Directed Trust**: Trust is directional (A→B ≠ B→A)

## Mathematical Framework

### Trust-Based Similarity

For users u and v with overlapping interests:

$$\text{Trust}(u, v) = f(\text{direct connections}, \text{shared interests}, \text{ratings agreement})$$

Users with high trust score likely have aligned preferences.

### Weighted Trust Graph

Direct trust edges with weights w(u,v) ∈ [0,1]:

$$w(u, v) = \frac{\# \text{mutual likes} + \lambda \cdot \text{rating correlation}}{Z}$$

Higher overlap → stronger trust; normalization ensures weights ∈ [0,1].

### Trust Propagation via Random Walk

Model trust spreading through network:

$$\text{Trust}^{(k)}(u, v) = \sum_w w(u, w) \cdot \text{Trust}^{(k-1)}(w, v)$$

k-hop trust computed via matrix multiplication:

$$T^{(k)} = (T^{(1)})^k$$

where T^{(1)} is direct trust adjacency matrix.

### Social Recommendation Scoring

Recommend items liked by trusted users:

$$\text{Score}(u, i) = \sum_{v: \text{trust}(u,v) > 0} \text{Trust}(u, v) \times \text{Rating}(v, i)$$

Weighted average of trusted users' ratings.

## Trust Models

### Explicit Trust Ratings

Users rate trust in others (1-5 scale):

$$\text{Trust}(u, v) \in \{1, 2, 3, 4, 5\}$$

Direct measurement but requires user effort.

### Implicit Trust Inference

Infer trust from behavior:

$$\text{Trust}(u, v) = \begin{cases}
1.0 & \text{if } v \text{ is friend} \\
0.7 & \text{if } v \text{ frequently co-engages} \\
0.5 & \text{if } v \text{ has similar interests} \\
0.0 & \text{otherwise}
\end{cases}$$

Automatic but less reliable than explicit ratings.

### Distrust Modeling

Users can also distrust others:

$$\text{Trust}(u, v) \in [-1, 1]$$

Negative trust indicates distrust; algorithms avoid recommendations from distrusted users.

$$\text{Score}(u, i) = \sum_{v: \text{Trust}(u,v) > 0} \text{Trust}(u, v) \times \text{Rating}(v, i) - \sum_{v: \text{Trust}(u,v) < 0} |\text{Trust}(u,v)| \times \text{Rating}(v, i)$$

## Trust-Aware Recommendation Algorithms

### TidalTrust

Compute trust only through paths of sufficient quality:

1. Limit path length (e.g., k ≤ 3)
2. Threshold minimum trust on each hop (e.g., trust > 0.3)
3. Aggregate trust from all valid paths
4. Normalize by path count

$$\text{Trust}_{\text{TidalTrust}}(u, v) = \frac{\sum_{\text{paths}} \text{TrustProduct}(path)}{\# \text{valid paths}}$$

### Probabilistic Trust Propagation

Model trust as probability of agreement:

$$P(\text{rate same}) = \prod_{e \in \text{path}} p(\text{edge})$$

Trust decreases exponentially with path length, preventing long-range trust.

### Matrix Factorization with Social Regularization

Learn user embeddings regularized by social trust:

$$\mathcal{L} = \sum_{(u,i)} (r_{ui} - \langle u, i \rangle)^2 + \lambda \sum_{u,v: \text{trust}(u,v) > 0} \text{Trust}(u,v) \|u - v\|^2$$

Second term: embeddings of trusted users should be similar.

## Financial Social Networks

### Network Types

**Trader Social Networks**:
- Nodes: Retail investors
- Edges: Follow on trading platforms (Robinhood, E*TRADE)
- Signal: Following indicates interest in trader's strategies

**Institutional Networks**:
- Nodes: Hedge funds, asset managers
- Edges: Competitive relationship, co-investments
- Signal: Portfolio similarity indicates correlated strategies

**Analyst Networks**:
- Nodes: Equity research analysts
- Edges: Work at same institution, cover same sectors
- Signal: Co-analysts likely similar perspectives

**Advisor Networks**:
- Nodes: Financial advisors
- Edges: Professional associations, geographic proximity
- Signal: Nearby advisors share local market knowledge

### Trust Signals in Finance

**Explicit**:
- Advisor/analyst ratings from platforms
- User testimonials and reviews
- Regulatory reputation/disciplinary records

**Implicit**:
- Portfolio copying: User adopts same allocations as peer
- Trade following: Retail investors mimic analyst recommendations
- Fund flows: Capital allocation to similar managers
- Search behavior: Users seeking similar information

## Influence Detection

### Influence Scoring

Quantify user impact on network:

$$\text{Influence}(u) = \frac{\sum_{v: u \text{ influenced}} \|v_{\text{after}} - v_{\text{before}}\|}{\sum_v |v_{\text{activity}}|}$$

Users whose decisions correlate with others' changes are influential.

### Pagerank for Influence

Apply PageRank to social graph:

$$\text{PageRank}(u) = \frac{1-\alpha}{n} + \alpha \sum_v \frac{\text{PageRank}(v)}{d_v}$$

Influential users accumulate high PageRank through many followers.

### Temporal Influence

Measure influence on specific time scales:

$$\text{Influence}_{\text{1week}}(u) = P(\text{follower changes portfolio within 1 week of } u)$$

Separates truly influential users from popular-but-not-influential users.

## Cold Start and Trust

### Trust for New User Cold Start

For new user u_new with no direct ratings:

1. Identify trusted similar users from social network
2. Recommend items those users prefer
3. Gradually gather u_new's own preferences

Works well when user can rapidly identify trusted peers.

### Trust Bootstrapping

New users lacking full trust profile:

$$\text{Score}(u_{\text{new}}, i) = w_1 \cdot \text{Similarity}(u_{\text{new}}) + w_2 \cdot \text{Trust_proxy}(u_{\text{new}})$$

Similarity-based component helps until trust profile builds.

## Practical Implementation

### Social Graph Construction

```python
import networkx as nx

# Create directed graph
G = nx.DiGraph()

# Add edges from social connections
for (user_a, user_b, trust_strength) in follows:
    G.add_edge(user_a, user_b, weight=trust_strength)

# Add edges from co-engagement
for (user_a, user_b) in co_engaged:
    if G.has_edge(user_a, user_b):
        G[user_a][user_b]['weight'] += 0.1
    else:
        G.add_edge(user_a, user_b, weight=0.1)
```

### Trust Propagation Algorithm

```python
def compute_trust(graph, source, targets, k=3):
    """Compute k-hop trust from source to targets"""
    trust = {}
    visited = {source}
    current_level = {source: 1.0}
    
    for hop in range(k):
        next_level = {}
        for node, confidence in current_level.items():
            for neighbor in graph.neighbors(node):
                edge_weight = graph[node][neighbor]['weight']
                new_conf = confidence * edge_weight
                
                if neighbor not in visited:
                    next_level[neighbor] = \
                        next_level.get(neighbor, 0) + new_conf
                    visited.add(neighbor)
        
        for target in targets:
            if target in next_level:
                trust[target] = next_level[target]
        
        current_level = next_level
    
    return trust
```

### Social Recommendation

```python
def recommend_social(user, graph, ratings, k=10):
    """Recommend items based on social trust"""
    
    # Compute trust to all other users
    all_users = [u for u in graph.nodes() if u != user]
    trust = compute_trust(graph, user, all_users)
    
    # Score items by trusted users' ratings
    item_scores = {}
    for other_user, trust_val in trust.items():
        for item, rating in ratings[other_user].items():
            if item not in item_scores:
                item_scores[item] = []
            item_scores[item].append((rating, trust_val))
    
    # Aggregate weighted ratings
    final_scores = {}
    for item, ratings_and_trusts in item_scores.items():
        ratings, trusts = zip(*ratings_and_trusts)
        weighted_rating = np.average(ratings, weights=trusts)
        final_scores[item] = weighted_rating
    
    # Return top-k
    return sorted(final_scores.items(), key=lambda x: x[1], 
                 reverse=True)[:k]
```

## Evaluation Metrics

### Recommendation Accuracy

Standard NDCG, AUC metrics apply. Trust-based recommendations should significantly outperform random baseline.

### Influencer Quality

Measure correlation between influencer recommendations and follower actions:

$$\text{Influence}_{\text{quality}} = \text{Corr}(\text{influencer recommendation}, \text{follower action})$$

High correlation indicates meaningful influence.

### Trust Network Coverage

What % of users have at least k trusted advisors?

$$\text{Coverage}_k = \frac{\# \text{users with } \geq k \text{ trusted connections}}{N_{\text{users}}}$$

Coverage > 80% indicates healthy trust network.

## Regulatory and Ethical Considerations

### Disclosure Requirements

When recommendations based on influencer opinions, disclose:
- Who the influencer is
- Nature of relationship (friend, analyst, platform)
- Whether influencer has financial interest

### Herding Risk

Following crowds can amplify market moves:

$$\text{Herding} = P(\text{user follows recommendation | others following})$$

Monitor for dangerous herd behavior; alert when concentration excessive.

### Diversification

Ensure social recommendations maintain portfolio diversity:

$$\text{Correlation}_{\text{followers}} = \text{mean pairwise correlation of follower portfolios}$$

High correlation indicates excessive herding; recommend diversification.

!!! warning "Social Influence Risks"
    Social-based recommendations can amplify groupthink and herding behavior. Financial regulators scrutinize influence-driven trading for market manipulation concerns. Implementations should include guardrails: position limits, herding alerts, and diversification enforcement.

