# Text Classification Fundamentals

## Overview

Text classification assigns predefined categories to documents. Given document $x$ and label set $\mathcal{Y}$:

$$y^* = \arg\max_{y \in \mathcal{Y}} P(y | x)$$

## Pipeline

Raw Text → Preprocessing → Feature Extraction → Classification → Label

## Feature Representations

| Method | Representation | Dimensionality |
|--------|---------------|----------------|
| Bag-of-Words | Word counts | $|V|$ |
| TF-IDF | Weighted frequencies | $|V|$ |
| Word Embeddings | Dense vectors | $d$ |
| Contextual | Transformer states | $d_{model}$ |

## Classical Classifiers

- **Naive Bayes**: $P(y|x) \propto P(y) \prod_i P(x_i|y)$
- **Logistic Regression**: $P(y|x) = \sigma(\mathbf{w}^T \mathbf{x} + b)$
- **SVM**: Maximum-margin hyperplane

## Evaluation

Standard metrics: accuracy, precision, recall, F1, AUC-ROC. Use macro-F1 for imbalanced datasets.

## References

1. Joachims, T. (1998). Text Categorization with Support Vector Machines. *ECML*.
2. Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level CNNs for Text Classification. *NeurIPS*.
