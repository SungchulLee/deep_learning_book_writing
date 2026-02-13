# Knowledge Graph-Based Recommender Systems

## Introduction

Knowledge graphs encode structured semantic information about entities and their relationships—assets and sectors, investors and investment styles, stocks and market events. By representing recommendation problems as knowledge graphs, systems can reason about recommendation justifications, incorporate external knowledge, and provide explainable recommendations. A knowledge graph-based recommender leverages both user-item interactions and semantic relationships to suggest items with clear, understandable justifications.

In finance, knowledge graphs naturally capture domain structure: stocks belong to sectors, sectors to industries; investors have preferences for asset classes; assets exhibit attributes (risk rating, dividend yield, ESG score). Unlike pure collaborative filtering treating assets as feature-less items, knowledge graphs enable reasoning: "Recommend energy stocks because user invested in renewable energy infrastructure" or "Suggest dividend funds because user profile indicates income focus."

This section develops knowledge graph representations for finance, explores reasoning engines for recommendation, and demonstrates implementations combining graph structure with neural methods.

## Key Concepts

### Knowledge Graph Components
- **Entities**: Users, assets, sectors, attributes, events
- **Relations**: owns, belongs_to, has_property, similar_to
- **Triples**: (subject, relation, object) representing facts
- **Weights**: Confidence or strength of relation

### Reasoning for Recommendations
- **Path-Based**: Multi-hop paths explain recommendations
- **Relation-Based**: Specific relations drive recommendations
- **Embedding-Based**: Learned representations of entities/relations
- **Rule-Based**: Logical rules encode domain knowledge

## Mathematical Framework

### Knowledge Graph Representation

Triple store: T = {(h, r, t) : h, t ∈ E, r ∈ R}

where:
- h = head entity
- r = relation type
- t = tail entity

Examples:
```
(Apple, traded_in, Healthcare_Sector)
(Apple, has_dividend, Yes)
(Healthcare_Sector, popular_with, Conservative_Investor)
```

### Knowledge Graph Embedding

Embed entities and relations in vector space for reasoning:

$$\text{Score}(h, r, t) = f(e_h, e_r, e_t)$$

**DistMult**: Bilinear scoring

$$\text{Score} = e_h^T \text{diag}(e_r) e_t$$

**ComplEx**: Complex-valued embeddings capturing asymmetry

$$\text{Score} = \Re(e_h^T \text{diag}(e_r) \bar{e}_t)$$

### Recommendation via Relation Traversal

For user u, find items via knowledge graph paths:

$$\text{Score}(u, i) = \sum_{\text{paths } p: u \rightsquigarrow i} w(p)$$

Path weight w(p) based on:
- Path length (shorter = stronger)
- Relation types (some relations more important)
- Frequency (common paths more reliable)

## Knowledge Graph Construction for Finance

### Entity Types

**Primary Entities**:
- Users: Individual investors, investment advisors, hedge funds
- Assets: Stocks, bonds, ETFs, mutual funds, crypto
- Sectors: Technology, Healthcare, Energy, Financials
- Attributes: Risk level, dividend yield, ESG score, market cap

**Event Entities**:
- Earnings announcements
- Analyst upgrades/downgrades
- Regulatory changes
- Market events (crashes, rallies)

### Relation Types

**User-Asset Relations**:
- owns: User owns asset
- traded: User traded asset
- rated: User gave rating to asset
- follows: User follows asset/analyst

**Asset-Asset Relations**:
- correlated_with: Assets move together
- substitutes: Similar assets
- complements: Assets often held together
- sector_peer: Assets in same sector

**Asset-Attribute Relations**:
- has_risk_level: Asset risk rating
- has_sector: Asset sector membership
- has_esg_score: Asset ESG rating
- tracks_index: Index fund tracks index

**Event Relations**:
- affects_asset: Event impacts asset
- affects_sector: Event impacts sector
- mentioned_in_report: Asset mentioned in analyst report

### Knowledge Graph Construction

```
User: John (Conservative, Income-focused)
├─ owns ──→ Bond ETF X
├─ owns ──→ Dividend Stock Y
├─ interested_in ──→ Income Strategy
└─ risk_profile ──→ Conservative

Asset: Dividend Stock Y
├─ sector ──→ Energy
├─ dividend_yield ──→ 4.5%
├─ has_risk ──→ Moderate
├─ correlated_with ──→ Energy ETF Z
└─ affected_by ──→ Oil Price Rise Event
```

## Recommendation Reasoning with Knowledge Graphs

### Path-Based Explanation

Recommend item i to user u via meta-path:

$$\text{User} \xrightarrow[\text{has preference}]{} \text{Attribute} \xrightarrow[\text{relevant to}]{} \text{Asset}$$

Generates explanation: "We recommend Stock X because you prefer income-generating assets and X yields 4%."

### Relation-Based Weighting

Weight relations by importance:

| Relation | Weight |
|----------|--------|
| owns | 10.0 |
| traded | 5.0 |
| rated | 3.0 |
| sector | 2.0 |
| correlated | 1.0 |

Stronger relations dominate recommendation signals.

### Multi-Hop Reasoning

Longer paths contribute less but enable discovery:

$$\text{Score} = \sum_{k=1}^{K} \gamma^{k-1} \cdot \text{PathScore}_k$$

where γ < 1 is discount factor, K is max path length.

## Embedding-Based Knowledge Graph Recommendations

### Joint User-Item-Relation Embedding

Learn embeddings for users, items, relations:

$$\mathcal{L} = \sum_{(u,r,i) \in \text{interactions}} [1 + \text{Score}_\theta(u, r, i)]_+ + \lambda \sum_{\phi} \|\phi\|^2$$

Training pulls user and item embeddings together via relation r; pushes apart unobserved (u,i) pairs.

### Translational Models

Interpret relations as translations in embedding space:

$$e_u + e_r \approx e_i \quad \text{for } (u, r, i) \in KG$$

TransE: $\mathcal{L} = \sum_{(u,r,i)} \|e_u + e_r - e_i\|^2$

User embeddings for liked items translate through relation to item embeddings.

### Combining Graph and Content

Learn embeddings from both knowledge graph and item features:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{KG}} + \mathcal{L}_{\text{content}} + \mathcal{L}_{\text{interaction}}$$

Multi-task learning exploits both structural and feature information.

## Knowledge Graph Completion and Expansion

### Link Prediction

Predict missing relations in knowledge graph:

$$p(r_{new} | (h, t)) = \text{Sigmoid}(\text{Score}_\theta(h, r_{\text{new}}, t))$$

Identify likely new relations (e.g., which assets should user own given profile).

### Relation Discovery

Find new relation types between entities:

1. Compute similarity of entity pairs
2. Cluster similar pairs
3. Assign relation type to cluster
4. Infer missing relations within cluster

Enables automated knowledge graph expansion.

## Practical Implementation

### Knowledge Graph Storage

Options:
- **RDF Store**: Semantic web standard (SPARQL queries)
- **Graph Database**: Neo4j, ArangoDB (efficient traversal)
- **Triple Store**: MongoDB with triple indexing
- **In-Memory**: Entire graph in memory for speed (requires sufficient RAM)

### Recommendation Algorithm

```python
class KGRecommender:
    def __init__(self, kg, embeddings):
        self.kg = kg  # Knowledge graph
        self.embeddings = embeddings  # Learned embeddings
    
    def recommend(self, user, k=10):
        # Find all items reachable from user
        reachable_items = self.bfs(user, max_depth=3)
        
        # Score via path weights + embeddings
        scores = {}
        for item in reachable_items:
            path_score = self.compute_path_weight(user, item)
            embedding_score = 
                np.dot(self.embeddings[user], 
                       self.embeddings[item])
            scores[item] = 0.5 * path_score + 0.5 * embedding_score
        
        # Return top-k
        return sorted(scores.items(), key=lambda x: x[1], 
                     reverse=True)[:k]
    
    def bfs(self, start, max_depth):
        """BFS to find reachable items"""
        items = []
        visited = {start}
        queue = [(start, 0)]
        
        while queue:
            node, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            
            for neighbor, relation in self.kg.neighbors(node):
                items.append(neighbor)
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
        
        return items
```

### Explanation Generation

For each recommendation, generate human-readable explanation:

```python
def explain(self, user, item):
    # Find shortest path user → item
    path = self.kg.shortest_path(user, item)
    
    # Convert to natural language
    explanation = "We recommend " + item
    for i, (node, relation) in enumerate(path[:-1]):
        explanation += f" because you {relation.verb} {node}"
    
    return explanation

# Example: explain(john, "Energy_ETF")
# "We recommend Energy_ETF because you own 
#  dividend stocks, which are related to 
#  energy sector, which includes Energy_ETF"
```

## Evaluation

### Ranking Accuracy

Standard metrics (NDCG, AUC) apply:

$$\text{NDCG@10} = \frac{\text{DCG@10}}{\text{IDCG@10}}$$

### Explainability Metrics

**Path Validity**: Are explanations factually correct?

**Path Meaningfulness**: Do explanations align with user understanding?

**Diversity**: Do recommendations explore multiple relations?

### Knowledge Graph Quality

**Completeness**: What % of true relations in KG?

**Correctness**: Are relations in KG accurate?

**Coverage**: Do recommendations use all available relations?

## Case Study: Financial Product Recommendation with KG

### Knowledge Graph

```
Entities: Users, Stocks, Bonds, Sectors, Risk Levels, ESG Scores
Relations: owns, rated, sector, risk_level, esg_rating, 
           correlates_with, substitute_for, mentioned_in
```

### Recommendation for Conservative Income-Seeking Investor

**Path 1** (length 2):
Conservative → income_focus → Dividend_Stocks
→ Recommend: Dividend ETF

**Path 2** (length 3):
Conservative → seeks_stability → Bond_Sector → 
Treasury_Bonds → Recommend: Short-term Treasury Fund

**Path 3** (length 4):
Conservative → likes_ESG → ESG_Companies → Technology → 
Dividend_Tech_ETF → Recommend: Tech Dividend Fund

### Results

- Accuracy: NDCG@10 = 0.72
- Explainability: All recommendations have 2-4 hop explanations
- Novelty: 30% of recommendations non-obvious from user profile
- User satisfaction: 80% of recommended items reasonable according to survey

!!! note "Knowledge Graphs"
    Knowledge graphs excel when domain structure well-understood and explainability critical. Particularly valuable in finance where regulatory requirements demand documented recommendation rationale. Hybrid approaches combining KG reasoning with learned embeddings provide best balance of interpretability and accuracy.

