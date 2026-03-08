# Chapter Overview

<<<<<<< HEAD
This chapter covers **Recursion**.

# Reference

[Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
=======
This chapter develops the statistical foundations that underpin deep learning: maximum likelihood estimation, linear regression, logistic regression, softmax regression, and loss functions. Each topic is presented from first principles with rigorous mathematical derivations, probabilistic interpretations, and accompanying PyTorch implementations. Understanding these foundations is essential because virtually every loss function in deep learning can be derived from MLE principles.

## Maximum Likelihood Estimation

The foundational parameter estimation framework that connects loss functions to probabilistic models.

- [MLE Overview](mle/mle_overview.md) -- Tutorial overview for MLE with PyTorch, from basic probability to neural network applications
- [Quick Start](mle/quickstart.md) -- Five-minute installation and first example guide
- [Maximum Likelihood Estimation](mle/mle.md) -- Core MLE theory: likelihood functions, log-likelihood, and the coin-flip intuition
- [Probabilistic Interpretation](mle/probabilistic_interpretation.md) -- The unified view: every loss is a negative log-likelihood, every regularizer is a prior
- [MLE for Regression](mle/mle_regression.md) -- MSE as Gaussian NLL, MAE as Laplace NLL, and heteroscedastic models
- [MLE for Classification](mle/mle_classification.md) -- BCE as Bernoulli NLL, cross-entropy as categorical NLL, and softmax from MLE

## Linear Regression

The foundational supervised learning algorithm with both closed-form and iterative solutions.

- [Linear Regression Overview](linear_regression/linear_regression_overview.md) -- Tutorial series guide from basics to advanced concepts
- [Quick Start](linear_regression/quickstart.md) -- Installation and first tutorial guide
- [Linear Regression](linear_regression/linear_regression.md) -- Model specification, univariate and multivariate formulations, and probabilistic interpretation
- [Closed-Form Solution](linear_regression/closed_form.md) -- Normal equations, vector calculus derivation, geometric interpretation, and NumPy/PyTorch implementation
- [Gradient Descent Solution](linear_regression/gd_solution.md) -- MSE gradient derivation and four levels of PyTorch implementation
- [Polynomial Features](linear_regression/polynomial_features.md) -- Nonlinear feature maps, bias-variance tradeoff, and cross-validated model selection
- [Ridge Regression](linear_regression/ridge_regression.md) -- L2 penalty, closed-form solution, geometric and Bayesian interpretations
- [Lasso Regression](linear_regression/lasso_regression.md) -- L1 penalty, sparsity and feature selection, coordinate descent, and Elastic Net

## Logistic Regression

Binary and multiclass classification through the lens of probabilistic modeling.

- [Logistic Regression Overview](logistic_regression/logistic_regression_overview.md) -- Tutorial series covering four progressive levels
- [Getting Started](logistic_regression/getting_started.md) -- Quick-start guide and recommended learning sequence
- [Basics Overview](logistic_regression/basics_overview.md) -- Level 1 tutorials: PyTorch tensors, sigmoid, and binary classification
- [Intermediate Overview](logistic_regression/intermediate_overview.md) -- Level 2 tutorials: proper training loops, DataLoader, and batching
- [Advanced Overview](logistic_regression/advanced_overview.md) -- Level 3 tutorials: custom datasets, multiclass, and advanced techniques
- [Applications Overview](logistic_regression/applications_overview.md) -- Level 4 tutorials: real-world pipelines for medical diagnosis and beyond
- [Sigmoid Function](logistic_regression/sigmoid.md) -- Derivation from log-odds, properties, odds ratios, and PyTorch visualization
- [Binary Classification](logistic_regression/binary_classification.md) -- Bernoulli model, GLM framework, and the connection between BCE and MLE
- [Decision Boundary](logistic_regression/decision_boundary.md) -- Boundary equation, geometry of weight vectors, and `BCEWithLogitsLoss`
- [Gradient Computation](logistic_regression/gradient.md) -- BCE gradient derivation, Hessian, convexity proof, and Newton's method
- [Regularized Logistic Regression](logistic_regression/regularized.md) -- L2, L1, Elastic Net penalties, and a complete PyTorch pipeline

## Softmax Regression

Extending logistic regression to multiclass classification with the softmax function.

- [Softmax Regression Overview](softmax_regression/softmax_regression_overview.md) -- Tutorial series for multiclass classification with PyTorch
- [Quick Start](softmax_regression/quickstart.md) -- Installation and starting-point guide
- [Multiclass Classification](softmax_regression/multiclass.md) -- Categorical distribution, one-hot encoding, and the generalization from Bernoulli
- [Softmax Function](softmax_regression/softmax_function.md) -- Derivation, Jacobian matrix, temperature scaling, and numerically stable implementation
- [Numerical Stability](softmax_regression/numerical_stability.md) -- Log-sum-exp trick, `nn.Module` classifiers, and complete training loops
- [Cross-Entropy Loss](softmax_regression/cross_entropy.md) -- MLE derivation, PyTorch interfaces, and N-gram language model demonstration

## Loss Functions

Mathematical foundations of loss functions from information theory through practical PyTorch usage.

- [Loss Functions Overview](loss/loss_overview.md) -- What loss functions are, their role in optimization, and PyTorch computation approaches
- [Information Theory](loss/information_theory.md) -- Self-information, entropy, cross-entropy, and mutual information for deep learning
- [MSE and MAE](loss/mse_mae.md) -- Gaussian and Laplace noise models, MLE derivations, and PyTorch training pipelines
- [Binary Cross-Entropy](loss/bce.md) -- Bernoulli likelihood, `BCELoss` vs `BCEWithLogitsLoss`, and practical usage
- [Cross-Entropy Loss](loss/cross_entropy.md) -- Information-theoretic derivation, full gradient derivation, and connection to KL divergence
- [KL Divergence](loss/kl_divergence.md) -- Definition, properties, PyTorch interfaces, and applications in VAEs and distillation
- [KL Divergence and Distance Axioms](loss/kl_distance_axioms.md) -- Metric axiom analysis and symmetrized alternatives
- [KL Divergence for Gaussians](loss/kl_gaussian.md) -- Closed-form derivations for univariate, multivariate, and diagonal-covariance cases
- [KL Divergence and Fisher Information](loss/kl_fisher_information.md) -- Local quadratic approximation, natural gradient, and information geometry
>>>>>>> 96f31bd (...)
