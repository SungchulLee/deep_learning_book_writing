<<<<<<< HEAD
# Chapter Overview

This chapter covers **Linear-Time Sorting and Selection**.

# Reference

[Introduction to Algorithms (CLRS), Chapters 8-9](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
=======
# Chapter 13: Bayesian Foundations

This chapter establishes the theoretical foundations of Bayesian inference, from Bayes' theorem and conjugate priors through to hierarchical models, model comparison, and applications in finance. These concepts form the mathematical backbone for the approximate inference methods, sampling algorithms, and Bayesian neural networks covered in subsequent chapters.

---

## Bayesian Foundations

- [Course Overview](bayesian_foundations/readme_01_readme.md) -- Comprehensive curriculum overview spanning classical Bayesian inference to modern Bayesian neural networks
- [Learning Roadmap](bayesian_foundations/readme_02_roadmap.md) -- Conceptual roadmap explaining the natural progression from Bayesian inference to advanced computation
- [Bayes' Theorem](bayesian_foundations/bayes_theorem.md) -- Derivation from conditional probability, the Bayesian formulation, and discrete examples
- [Prior, Likelihood, and Posterior](bayesian_foundations/prior_likelihood_posterior.md) -- Rigorous treatment of the three fundamental quantities and their interplay through Bayes' theorem
- [Conjugate Priors](bayesian_foundations/conjugate_priors.md) -- Theory of conjugacy with Beta-Binomial, Gamma-Poisson, and Normal-Normal families for analytical solutions
- [MAP Estimation](bayesian_foundations/map_estimation.md) -- Maximum a posteriori point estimates, comparison with MLE and posterior mean, and connection to regularization
- [Credible Intervals](bayesian_foundations/credible_intervals.md) -- Bayesian uncertainty quantification through equal-tailed and Highest Posterior Density intervals

## Bayesian Distributions

- [Bayesian Linear Regression](bayesian_distributions/bayesian_linear_regression.md) -- Full posterior distributions over parameters and predictions with the conjugate Normal-Normal model
- [Bayesian Logistic Regression](bayesian_distributions/bayesian_logistic_regression.md) -- Non-conjugate posterior requiring approximate inference, bridging analytical and computational methods
- [Gaussian Processes](bayesian_distributions/gaussian_processes.md) -- Nonparametric Bayesian regression defining priors directly over functions with kernel-based assumptions

## Hierarchical Models

- [Hierarchical Bayesian Models](hierarchical/hierarchical_bayes.md) -- Multi-level inference with partial pooling, group-level variation, and the shrinkage phenomenon
- [Multilevel Models](hierarchical/multilevel.md) -- Mixed-effects models with structured random effects for nested data analysis
- [Empirical Bayes](hierarchical/empirical_bayes.md) -- Estimating hyperparameters from data as a practical middle ground between fully Bayesian and frequentist approaches

## Model Comparison

- [Model Evidence (Marginal Likelihood)](model_comparison/selection.md) -- Computing the probability of data under a model, implementing Bayesian Occam's razor
- [Bayes Factors](model_comparison/bayes_factors.md) -- Ratio of model evidences for principled comparison of competing models
- [Bayesian Hypothesis Testing](model_comparison/hypothesis_testing.md) -- Using Bayes factors and posterior odds for quantifying evidence between competing hypotheses
- [Information Criteria](model_comparison/information_criteria.md) -- Computationally tractable approximations (AIC, BIC, WAIC) connecting frequentist and Bayesian model selection

## Finance Applications

- [Bayesian Portfolio Optimization](finance/portfolio.md) -- Incorporating parameter uncertainty into mean-variance optimization for more robust portfolios
- [Parameter Uncertainty in Finance](finance/parameter_uncertainty.md) -- Quantifying and propagating estimation risk through portfolio construction and risk management
- [Regime Detection and Strategy Evaluation](finance/regime.md) -- Online Bayesian updating for market regime detection and Bayesian A/B testing for strategy comparison
>>>>>>> 96f31bd (...)
