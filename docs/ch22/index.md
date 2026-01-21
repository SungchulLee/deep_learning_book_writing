# Chapter 22: Classical Machine Learning with Scikit-learn

This chapter provides comprehensive coverage of scikit-learn fundamentals, serving as a foundation before transitioning to PyTorch implementations. Each topic follows the pedagogical approach of starting with sklearn (high-level, familiar API) and comparing to PyTorch equivalents.

---

## Chapter Structure

### 22.1 Preprocessing
- [Scaling and Normalization](preprocessing/scaling.md) - StandardScaler, MinMaxScaler, RobustScaler
- [Encoding Categorical Variables](preprocessing/encoding.md) - OneHotEncoder, LabelEncoder, OrdinalEncoder
- [Handling Missing Data](preprocessing/missing_data.md) - SimpleImputer, KNNImputer, IterativeImputer

### 22.2 Model Selection
- [Train-Test Split](model_selection/train_test_split.md) - Data splitting strategies
- [Cross-Validation](model_selection/cross_validation.md) - K-Fold, Stratified, Time Series CV
- [Hyperparameter Tuning](model_selection/hyperparameters.md) - GridSearchCV, RandomizedSearchCV

### 22.3 Supervised Learning
- [Linear Models](supervised/linear_models.md) - LinearRegression, Ridge, Lasso, LogisticRegression
- [Tree-Based Models](supervised/tree_models.md) - DecisionTree, feature importance, pruning
- [Support Vector Machines](supervised/svm.md) - SVC, SVR, kernels, regularization
- [Ensemble Methods](supervised/ensemble.md) - RandomForest, GradientBoosting, Stacking

### 22.4 Unsupervised Learning
- [Clustering](unsupervised/clustering.md) - K-Means, DBSCAN, Hierarchical, GMM
- [Dimensionality Reduction](unsupervised/dimensionality.md) - PCA, t-SNE, UMAP, LDA

### 22.5 Evaluation
- [Regression Metrics](evaluation/regression_metrics.md) - MSE, RMSE, MAE, R², MAPE
- [Classification Metrics](evaluation/classification_metrics.md) - Accuracy, Precision, Recall, F1, ROC-AUC
- [Confusion Matrix](evaluation/confusion_matrix.md) - Visualization, error analysis, thresholds

### 22.6 Pipelines
- [Pipeline Basics](pipelines/pipeline_basics.md) - Building reproducible ML workflows
- [Feature Engineering](pipelines/feature_engineering.md) - Polynomial, binning, transforms

---

## Key Themes

### sklearn → PyTorch Transition

Each section includes PyTorch equivalents to help understand:
- How sklearn abstracts away implementation details
- When to use sklearn vs PyTorch
- Converting sklearn workflows to PyTorch

### Quantitative Finance Focus

Examples throughout include financial applications:
- Feature scaling for price data
- Time series cross-validation
- Portfolio optimization with clustering
- Risk metrics evaluation

---

## File Summary

| Section | Files | Total Lines |
|---------|-------|-------------|
| Preprocessing | 3 | ~1,565 |
| Model Selection | 3 | ~540 |
| Supervised | 4 | ~2,646 |
| Unsupervised | 2 | ~985 |
| Evaluation | 3 | ~1,381 |
| Pipelines | 2 | ~769 |
| **Total** | **17** | **~7,886** |

---

## Prerequisites

- Python 3.8+
- NumPy, Pandas, Matplotlib
- scikit-learn 1.0+
- PyTorch (for comparison examples)

## Next Steps

After completing this chapter, proceed to:
- Chapter 1: Deep Learning Foundations (neural network basics)
- Chapter 2: Optimization (gradient descent, optimizers)
