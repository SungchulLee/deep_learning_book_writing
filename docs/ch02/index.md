# Chapter 2: Scikit-learn

Scikit-learn provides the standard Python interface for classical machine learning. This chapter covers its API design philosophy, preprocessing utilities, model families, evaluation methodology, and integration patterns with PyTorch—all grounded in quantitative finance applications.

## Why Scikit-learn Before Deep Learning?

Classical ML models remain the **production workhorse** for tabular financial data. Understanding scikit-learn first provides three advantages for the deep learning practitioner:

1. **Baseline models** — Tree ensembles and linear models often outperform neural networks on structured data with $n < 10{,}000$ features. Every deep learning project needs a classical baseline.
2. **Pipeline discipline** — Scikit-learn's `fit`/`transform`/`predict` contract and `Pipeline` abstraction enforce the train-test separation and reproducibility that carry directly into PyTorch workflows.
3. **Evaluation literacy** — Metrics, cross-validation, and hyperparameter search transfer unchanged to neural network evaluation.

## Chapter Structure

### 2.1 Foundations

The API conventions, estimator interface, and pipeline design that unify all of scikit-learn.

- [API Overview](foundations/api.md) — `fit`/`predict`/`transform`, parameter conventions, `get_params`/`set_params`
- [Estimator Interface](foundations/estimator.md) — `BaseEstimator`, `TransformerMixin`, `ClassifierMixin`, writing custom estimators
- [Pipeline Design](foundations/pipeline.md) — `Pipeline`, `ColumnTransformer`, `FeatureUnion`, caching, preventing data leakage

### 2.2 Preprocessing

Transforming raw features into model-ready representations.

- [Scalers](preprocessing/scalers.md) — `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler`, power and quantile transforms
- [Encoders](preprocessing/encoders.md) — `OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder`, target encoding, hashing
- [Imputers](preprocessing/imputers.md) — `SimpleImputer`, `KNNImputer`, `IterativeImputer`, missing indicators
- [Feature Selection](preprocessing/feature_selection.md) — Filter, wrapper, and embedded methods; `SelectKBest`, `RFE`, `SelectFromModel`
- [Transformers](preprocessing/transformers.md) — `PolynomialFeatures`, `KBinsDiscretizer`, `FunctionTransformer`, date/time and text features

### 2.3 Classical Models

Supervised learning algorithms from linear models through ensembles.

- [Linear Models](models/linear.md) — `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `LogisticRegression`
- [Tree Models](models/trees.md) — `DecisionTreeClassifier`/`Regressor`, splitting criteria, pruning, visualisation
- [Ensemble Methods](models/ensemble.md) — `RandomForest`, `GradientBoosting`, `AdaBoost`, stacking, voting
- [SVM](models/svm.md) — `SVC`, `SVR`, kernel trick, regularisation, scaling requirements
- [Neighbors](models/neighbors.md) — `KNeighborsClassifier`/`Regressor`, distance metrics, `BallTree`, `KDTree`
- [Naive Bayes](models/naive_bayes.md) — `GaussianNB`, `MultinomialNB`, `BernoulliNB`, conditional independence

### 2.4 Model Selection

Principled approaches to splitting, validation, and hyperparameter search.

- [Cross-Validation](selection/cross_validation.md) — K-Fold, Stratified, LOOCV, `TimeSeriesSplit`, `GroupKFold`, nested CV
- [Grid Search](selection/grid_search.md) — `GridSearchCV`, parameter grids, multi-metric evaluation
- [Randomized Search](selection/random_search.md) — `RandomizedSearchCV`, distribution specification, efficiency vs. grid
- [Bayesian Optimization](selection/bayesian.md) — Surrogate models, acquisition functions, `scikit-optimize`, `Optuna`

### 2.5 Metrics

Quantifying model performance for classification, regression, and clustering.

- [Classification Metrics](metrics/classification.md) — Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix
- [Regression Metrics](metrics/regression.md) — MSE, RMSE, MAE, $R^2$, MAPE, explained variance
- [Clustering Metrics](metrics/clustering.md) — Silhouette, Calinski–Harabasz, Davies–Bouldin, adjusted Rand index
- [Custom Scorers](metrics/custom.md) — `make_scorer`, business-specific loss functions, asymmetric costs

### 2.6 PyTorch Integration

Bridging scikit-learn workflows with deep learning.

- [Skorch](pytorch/skorch.md) — Wrapping PyTorch modules as sklearn estimators, using `NeuralNetClassifier`/`NeuralNetRegressor`
- [Custom Estimators](pytorch/custom_estimator.md) — Implementing `fit`/`predict`/`score` for PyTorch models
- [Hybrid Pipelines](pytorch/hybrid.md) — sklearn preprocessing → PyTorch model → sklearn evaluation

### 2.7 Finance Applications

Domain-specific patterns for quantitative finance.

- [Factor Models](finance/factor_models.md) — Cross-sectional regression, Fama–French factors, feature importance as factor loading
- [Credit Scoring](finance/credit.md) — Imbalanced classification, scorecard development, regulatory constraints
- [Time Series CV](finance/time_series_cv.md) — Walk-forward validation, purging, embargo, combinatorial purged CV

## Source Material Mapping

This chapter consolidates and restructures content from the original Chapter 22 source files:

| Original Source | Target Section(s) |
|---|---|
| `ch22/index.md` | Chapter overview (this page) |
| `ch22/preprocessing/scaling.md` | §2.2 Scalers |
| `ch22/preprocessing/encoding.md` | §2.2 Encoders |
| `ch22/preprocessing/missing_data.md` | §2.2 Imputers |
| `ch22/pipelines/pipeline_basics.md` | §2.1 Foundations (API, Estimator, Pipeline) |
| `ch22/pipelines/feature_engineering.md` | §2.2 Feature Selection + Transformers |
| `ch22/supervised/linear_models.md` | §2.3 Linear Models |
| `ch22/supervised/tree_models.md` | §2.3 Tree Models |
| `ch22/supervised/ensemble.md` | §2.3 Ensemble Methods |
| `ch22/supervised/svm.md` | §2.3 SVM |
| `ch22/unsupervised/clustering.md` | §2.5 Clustering Metrics |
| `ch22/unsupervised/dimensionality.md` | §2.2 Feature Selection + Transformers |
| `ch22/model_selection/cross_validation.md` | §2.4 Cross-Validation |
| `ch22/model_selection/train_test_split.md` | §2.4 Cross-Validation |
| `ch22/model_selection/hyperparameters.md` | §2.4 Grid Search + Randomized Search + Bayesian |
| `ch22/evaluation/classification_metrics.md` | §2.5 Classification Metrics |
| `ch22/evaluation/regression_metrics.md` | §2.5 Regression Metrics |
| `ch22/evaluation/confusion_matrix.md` | §2.5 Classification Metrics |

## Prerequisites

- Python 3.9+, NumPy, Pandas, Matplotlib
- scikit-learn ≥ 1.3
- PyTorch ≥ 2.0 (for §2.6 integration examples)
- Chapter 1 (deep learning foundations) is helpful but not required—this chapter is self-contained
