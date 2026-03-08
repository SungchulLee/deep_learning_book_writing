<<<<<<< HEAD
# Chapter Overview
=======
# Chapter 30: Recommender Systems
>>>>>>> 96f31bd (...)

This chapter covers **Parallel and Distributed**.

<<<<<<< HEAD
# Reference

[Introduction to Parallel Computing (Grama et al.)](https://www.amazon.com/Introduction-Parallel-Computing-Ananth-Grama/dp/0201648652)
=======
Recommender systems predict user preferences and surface relevant items from large catalogs. This chapter covers the full spectrum of recommendation approaches, from classical collaborative filtering and matrix factorization to modern neural methods, graph-based recommenders, and rigorous evaluation frameworks. Financial applications including portfolio suggestion, product recommendation, and risk-aware systems are explored throughout.

## Contents

### 30.1 Foundations

- [RecSys Overview](foundations/recsys_overview.md) -- Formal definition of the recommendation problem, rating matrices, and taxonomy of recommender system architectures
- [Collaborative Filtering](foundations/collaborative_filtering.md) -- Memory-based and model-based collaborative filtering using user and item similarity measures
- [Content-Based Filtering](foundations/content_based.md) -- Recommending items based on feature similarity to previously liked items using TF-IDF and neural extractors
- [Matrix Factorization](foundations/matrix_factorization.md) -- Low-rank approximation of the rating matrix with SVD-based embeddings and biased MF in PyTorch
- [Implicit vs Explicit Feedback](foundations/implicit_explicit.md) -- Frameworks for explicit ratings and implicit behavioral signals with approaches to combine both modalities
- [Cold Start Problem](foundations/cold_start.md) -- Strategies for handling new users, new items, and new systems with limited interaction data

### 30.2 Neural Methods

- [Neural Collaborative Filtering](neural_methods/ncf.md) -- Replacing the dot product with neural networks for more expressive user-item interaction modeling
- [Autoencoder-Based RecSys](neural_methods/autoencoder_recsys.md) -- Variational and standard autoencoders for learning latent user representations from rating profiles
- [Sequential Recommendations](neural_methods/sequential.md) -- Temporal-aware recommendations using GRU4Rec and self-attentive models (SASRec)
- [Neural Content-Based Filtering](neural_methods/content_based.md) -- Deep learning approaches to content-based recommendation with neural feature extraction
- [Embedding-Based RecSys](neural_methods/embedding_recsys.md) -- Two-tower architectures and approximate nearest neighbor retrieval for scalable recommendation
- [Attention-Based RecSys](neural_methods/attention_recsys.md) -- Attention mechanisms for dynamically weighting user history and contextual information
- [Hybrid Methods](neural_methods/hybrid.md) -- Combining collaborative and content-based signals to overcome individual method limitations

### 30.3 Graph-Based Recommender Systems

- [Bipartite Graph Modeling](graph_recsys/bipartite.md) -- Representing user-item interactions as bipartite graphs with random walks and spectral methods
- [GNN-Based RecSys](graph_recsys/gnn_recsys.md) -- End-to-end graph neural networks for recommendation with message passing on interaction graphs
- [Knowledge Graph RecSys](graph_recsys/kg_recsys.md) -- Leveraging structured semantic knowledge for explainable, reasoning-based recommendations
- [Social Graph RecSys](graph_recsys/social_recsys.md) -- Trust-aware and influence-based recommendations using social network structure

### 30.4 Evaluation

- [Evaluating Recommender Systems](evaluation/evaluation.md) -- Rating prediction vs ranking evaluation paradigms with MSE, RMSE, and MAE metrics
- [Ranking Metrics](evaluation/ranking_metrics.md) -- Precision@K, Recall@K, NDCG, and MAP for evaluating recommendation list quality
- [Online vs Offline Evaluation](evaluation/online_offline.md) -- Temporal splits, leave-one-out protocols, and the gap between offline metrics and live performance
- [Coverage and Diversity](evaluation/coverage_diversity.md) -- Measuring catalog coverage, intra-list diversity, and addressing popularity bias
- [A/B Testing for RecSys](evaluation/ab_testing.md) -- Experimental design, randomization, sample size calculation, and statistical significance for production tests
- [Beyond Accuracy Metrics](evaluation/beyond_accuracy.md) -- Novelty, serendipity, fairness, and multi-dimensional evaluation frameworks for comprehensive assessment

### 30.5 Financial Applications

- [Portfolio Suggestion](finance/portfolio_suggestion.md) -- Personalized portfolio recommendations balancing returns, risk suitability, diversification, and regulatory compliance
- [Financial Product Recommendation](finance/product_recommendation.md) -- Recommending funds, ETFs, and banking products with suitability requirements and fiduciary constraints
- [Research Report Recommendation](finance/research_recommendation.md) -- Surfacing relevant equity research and macro commentary using NLP content understanding and collaborative filtering
- [Risk-Aware RecSys](finance/risk_aware.md) -- Incorporating downside risk, portfolio concentration, and stress scenarios into recommendation algorithms
>>>>>>> 96f31bd (...)
