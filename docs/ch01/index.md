<<<<<<< HEAD
# Chapter 1: Introduction to Algorithms

This chapter introduces the fundamental concepts of algorithms: what they are, why they matter, and the mathematical tools needed to study them.

$$

\text{Algorithm} = \text{Well-defined computational procedure that transforms input to output}

$$

## Topics

- **What is an Algorithm?** — definitions, properties, correctness, efficiency
- **Problem Solving Strategies** — brute force, incremental improvement, divide and conquer
- **Mathematical Background** — summations, logarithms, proofs
- **Programming Languages** — Python and C++ for algorithm implementation
=======
# Chapter 1: Scikit-learn

Scikit-learn provides the standard Python interface for classical machine learning. This chapter covers environment setup, API design philosophy, preprocessing utilities, model families, evaluation methodology, and integration patterns with PyTorch -- all grounded in quantitative finance applications. Understanding scikit-learn first establishes the baseline discipline and pipeline thinking that carries directly into deep learning workflows.

## Setup

Configuring your development environment for Python-based machine learning and deep learning.

- [Environment Setup](setup/environment_setup.md) -- Install and configure system tools, Miniforge, VS Code, and isolated Python environments
- [Basic Configuration](setup/basic_configuration.md) -- Project directory structure, essential libraries, Jupyter customization, and Git setup
- [Package Management](setup/package_management.md) -- Conda vs pip, channels, dependency conflicts, and reproducible environment specifications
- [Virtual Environments](setup/virtual_environments.md) -- Environment isolation with conda and venv, exporting and reproducing environments
- [IDEs and Jupyter](setup/ides_and_jupyter.md) -- Jupyter Notebook, JupyterLab, Spyder, PyCharm, VS Code, and Google Colab

## Foundations

The API conventions, estimator interface, and pipeline design that unify all of scikit-learn.

- [API Overview](foundations/api.md) -- The uniform `fit`/`predict`/`transform` interface and parameter conventions
- [Estimator Interface](foundations/estimator.md) -- `BaseEstimator`, `TransformerMixin`, `ClassifierMixin`, and writing custom estimators
- [Pipeline Design](foundations/pipeline.md) -- `Pipeline`, `ColumnTransformer`, `FeatureUnion`, caching, and preventing data leakage

## Preprocessing

Transforming raw features into model-ready representations.

- [Scalers](preprocessing/scalers.md) -- `StandardScaler`, `MinMaxScaler`, `RobustScaler`, and power transforms
- [Encoders](preprocessing/encoders.md) -- `OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder`, target encoding, and hashing
- [Imputers](preprocessing/imputers.md) -- `SimpleImputer`, `KNNImputer`, `IterativeImputer`, and missing indicators
- [Feature Selection](preprocessing/feature_selection.md) -- Filter, wrapper, and embedded methods for dimensionality reduction
- [Transformers](preprocessing/transformers.md) -- Polynomial expansion, discretization, PCA, t-SNE, and text feature engineering

## Models

Supervised learning algorithms from linear models through ensembles.

- [Linear Models](models/linear.md) -- `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, and `LogisticRegression`
- [Tree Models](models/trees.md) -- `DecisionTreeClassifier`/`Regressor`, splitting criteria, pruning, and visualization
- [Ensemble Methods](models/ensemble.md) -- `RandomForest`, `GradientBoosting`, `AdaBoost`, stacking, and voting
- [SVM](models/svm.md) -- `SVC`, `SVR`, kernel trick, regularization, and scaling requirements
- [Neighbors](models/neighbors.md) -- `KNeighborsClassifier`/`Regressor`, distance metrics, `BallTree`, and `KDTree`
- [Naive Bayes](models/naive_bayes.md) -- `GaussianNB`, `MultinomialNB`, `BernoulliNB`, and conditional independence

## Selection

Principled approaches to splitting, validation, and hyperparameter search.

- [Cross-Validation](selection/cross_validation.md) -- K-Fold, Stratified, LOOCV, `TimeSeriesSplit`, `GroupKFold`, and nested CV
- [Grid Search](selection/grid_search.md) -- `GridSearchCV`, parameter grids, and multi-metric evaluation
- [Randomized Search](selection/random_search.md) -- `RandomizedSearchCV`, distribution specification, and efficiency vs grid
- [Bayesian Optimization](selection/bayesian.md) -- Surrogate models, acquisition functions, `scikit-optimize`, and `Optuna`

## Metrics

Quantifying model performance for classification, regression, and clustering.

- [Classification Metrics](metrics/classification.md) -- Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and confusion matrix
- [Regression Metrics](metrics/regression.md) -- MSE, RMSE, MAE, R-squared, and MAPE
- [Clustering Metrics](metrics/clustering.md) -- Silhouette, Calinski-Harabasz, Davies-Bouldin, and adjusted Rand index
- [Custom Scorers](metrics/custom.md) -- `make_scorer`, business-specific loss functions, and asymmetric costs

## PyTorch Integration

Bridging scikit-learn workflows with deep learning.

- [Skorch](pytorch/skorch.md) -- Wrapping PyTorch modules as sklearn estimators with `NeuralNetClassifier`/`NeuralNetRegressor`
- [Custom Estimators](pytorch/custom_estimator.md) -- Implementing `fit`/`predict`/`score` for PyTorch models directly
- [Hybrid Pipelines](pytorch/hybrid.md) -- Combining sklearn preprocessing with PyTorch models and sklearn evaluation

## Finance Applications

Domain-specific patterns for quantitative finance.

- [Factor Models](finance/factor_models.md) -- Cross-sectional regression, Fama-MacBeth, and feature importance as factor loading
- [Credit Scoring](finance/credit.md) -- Imbalanced classification, scorecard development, and regulatory constraints
- [Time Series CV](finance/time_series_cv.md) -- Walk-forward validation, purging, embargo, and combinatorial purged CV

## Installation Guides

Platform-specific guides for setting up a Python development environment.

- [Installation Overview](installation/installation_overview.md) -- Overview and quick-start links for all platforms
- [Mac Installation Guide](installation/mac_installation_guide.md) -- Homebrew, Miniforge, and VS Code setup on macOS
- [Windows Installation Guide](installation/windows_installation_guide.md) -- Chocolatey, Miniconda, and VS Code setup on Windows
- [Linux Installation Guide](installation/linux_installation_guide.md) -- Miniforge and VS Code setup on Ubuntu, Fedora, and Arch
- [Mac Quick Reference](installation/quick_ref_mac.md) -- One-line install commands for macOS
- [Windows Quick Reference](installation/quick_ref_windows.md) -- One-line install commands for Windows
- [Linux Quick Reference](installation/quick_ref_linux.md) -- One-line install commands for Linux
>>>>>>> 96f31bd (...)
