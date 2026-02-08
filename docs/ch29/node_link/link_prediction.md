# 29.6.2 Link Prediction

## Overview
Link prediction predicts whether an edge should exist between two nodes. Essential for recommendation, knowledge graph completion, and financial network analysis.

## Approaches

### Score-based
Compute a score for each node pair using embeddings:
$$\text{score}(u, v) = f(\mathbf{h}_u, \mathbf{h}_v)$$

Common scoring functions: dot product, concatenation + MLP, distance-based.

### Training
- **Positive edges**: Existing edges in the graph
- **Negative edges**: Random non-existing edges (negative sampling)
- **Loss**: Binary cross-entropy or margin-based

## Evaluation Metrics
- **AUC-ROC**: Area under ROC curve
- **Average Precision (AP)**: Area under precision-recall curve
- **Hits@K**: Fraction of true links in top-K predictions
- **MRR**: Mean reciprocal rank

## Financial Applications
- **Predict future correlations**: Will two assets become correlated?
- **Transaction prediction**: Predict future transactions between accounts
- **Supply chain**: Predict new supplier-customer relationships
