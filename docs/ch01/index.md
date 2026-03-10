# Chapter 1: Scikit-learn


!!! warning "Incomplete page"
    This page is missing the required five-section structure (Concept Definition, Explanation, Diagram / Example). Content needs to be reorganized and expanded.

Scikit-learn provides the standard Python interface for classical machine learning. This chapter covers environment setup, API design philosophy, preprocessing utilities, model families, evaluation methodology, and integration patterns with PyTorch -- all grounded in quantitative finance applications. Understanding scikit-learn first establishes the baseline discipline and pipeline thinking that carries directly into deep learning workflows.

## Setup

Configuring your development environment for Python-based machine learning and deep learning.

- Environment Setup -- Install and configure system tools, Miniforge, VS Code, and isolated Python environments
- Basic Configuration -- Project directory structure, essential libraries, Jupyter customization, and Git setup
- Package Management -- Conda vs pip, channels, dependency conflicts, and reproducible environment specifications
- Virtual Environments -- Environment isolation with conda and venv, exporting and reproducing environments
- IDEs and Jupyter -- Jupyter Notebook, JupyterLab, Spyder, PyCharm, VS Code, and Google Colab

## Foundations

The API conventions, estimator interface, and pipeline design that unify all of scikit-learn.

- API Overview -- The uniform `fit`/`predict`/`transform` interface and parameter conventions
- Estimator Interface -- `BaseEstimator`, `TransformerMixin`, `ClassifierMixin`, and writing custom estimators
- Pipeline Design -- `Pipeline`, `ColumnTransformer`, `FeatureUnion`, caching, and preventing data leakage

## Preprocessing

Transforming raw features into model-ready representations.

- Scalers -- `StandardScaler`, `MinMaxScaler`, `RobustScaler`, and power transforms
- Encoders -- `OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder`, target encoding, and hashing
- Imputers -- `SimpleImputer`, `KNNImputer`, `IterativeImputer`, and missing indicators
- Feature Selection -- Filter, wrapper, and embedded methods for dimensionality reduction
- Transformers -- Polynomial expansion, discretization, PCA, t-SNE, and text feature engineering

## Models

Supervised learning algorithms from linear models through ensembles.

- [Linear Models](models/linear.md) -- `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, and `LogisticRegression`
- [Tree Models](models/trees.md) -- `DecisionTreeClassifier`/`Regressor`, splitting criteria, pruning, and visualization
- Ensemble Methods -- `RandomForest`, `GradientBoosting`, `AdaBoost`, stacking, and voting
- [SVM](models/svm.md) -- `SVC`, `SVR`, kernel trick, regularization, and scaling requirements
- [Neighbors](models/neighbors.md) -- `KNeighborsClassifier`/`Regressor`, distance metrics, `BallTree`, and `KDTree`
- Naive Bayes -- `GaussianNB`, `MultinomialNB`, `BernoulliNB`, and conditional independence

## Selection

Principled approaches to splitting, validation, and hyperparameter search.

- Cross-Validation -- K-Fold, Stratified, LOOCV, `TimeSeriesSplit`, `GroupKFold`, and nested CV
- Grid Search -- `GridSearchCV`, parameter grids, and multi-metric evaluation
- Randomized Search -- `RandomizedSearchCV`, distribution specification, and efficiency vs grid
- Bayesian Optimization -- Surrogate models, acquisition functions, `scikit-optimize`, and `Optuna`

## Metrics

Quantifying model performance for classification, regression, and clustering.

- Classification Metrics -- Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and confusion matrix
- Regression Metrics -- MSE, RMSE, MAE, R-squared, and MAPE
- [Clustering Metrics](metrics/clustering.md) -- Silhouette, Calinski-Harabasz, Davies-Bouldin, and adjusted Rand index
- Custom Scorers -- `make_scorer`, business-specific loss functions, and asymmetric costs

## PyTorch Integration

Bridging scikit-learn workflows with deep learning.

- Skorch -- Wrapping PyTorch modules as sklearn estimators with `NeuralNetClassifier`/`NeuralNetRegressor`
- Custom Estimators -- Implementing `fit`/`predict`/`score` for PyTorch models directly
- Hybrid Pipelines -- Combining sklearn preprocessing with PyTorch models and sklearn evaluation

## Finance Applications

Domain-specific patterns for quantitative finance.

- Factor Models -- Cross-sectional regression, Fama-MacBeth, and feature importance as factor loading
- Credit Scoring -- Imbalanced classification, scorecard development, and regulatory constraints
- Time Series CV -- Walk-forward validation, purging, embargo, and combinatorial purged CV

## Installation Guides

Platform-specific guides for setting up a Python development environment.

- Installation Overview -- Overview and quick-start links for all platforms
- Mac Installation Guide -- Homebrew, Miniforge, and VS Code setup on macOS
- Windows Installation Guide -- Chocolatey, Miniconda, and VS Code setup on Windows
- Linux Installation Guide -- Miniforge and VS Code setup on Ubuntu, Fedora, and Arch
- Mac Quick Reference -- One-line install commands for macOS
- Windows Quick Reference -- One-line install commands for Windows
- Linux Quick Reference -- One-line install commands for Linux
