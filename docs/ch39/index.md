# Chapter 39: Model Uncertainty

This chapter provides a comprehensive treatment of uncertainty quantification in deep learning, covering the theoretical foundations of Bayesian inference, practical methods including MC Dropout and deep ensembles, calibration techniques, out-of-distribution detection, and evaluation frameworks. Understanding and measuring model uncertainty is critical for deploying neural networks in high-stakes domains such as quantitative finance, where overconfident predictions can lead to catastrophic losses.

---

## Fundamentals

Theoretical foundations and taxonomy of uncertainty in deep learning.

- [Introduction to Model Uncertainty](fundamentals/introduction.md) -- The overconfidence problem in neural networks and why uncertainty quantification matters
- [Types of Uncertainty](fundamentals/types.md) -- Taxonomy of parameter, structural, and distributional uncertainty sources
- [Epistemic vs Aleatoric Uncertainty](fundamentals/epistemic_aleatoric.md) -- Distinguishing model uncertainty (reducible) from data noise (irreducible)
- [Mathematical Framework](fundamentals/framework.md) -- Bayesian posterior predictive distribution and Monte Carlo approximation methods
- [Model Uncertainty Overview](fundamentals/model_uncertainty_overview.md) -- Module overview with learning objectives and prerequisites

## MC Dropout

Monte Carlo Dropout as approximate Bayesian inference for uncertainty estimation.

- [Monte Carlo Dropout](mc_dropout/mc_dropout.md) -- Core method: keeping dropout active at inference for posterior predictive estimation
- [Theoretical Foundation](mc_dropout/theory.md) -- Rigorous development of dropout as variational inference over neural network weights
- [Implementation Details](mc_dropout/implementation.md) -- Architecture patterns, inference procedures, and production considerations
- [Sample Convergence](mc_dropout/convergence.md) -- Convergence analysis and guidelines for selecting the number of forward passes
- [Dropout Rate Selection](mc_dropout/dropout_rate.md) -- Principled approaches to choosing dropout rates for uncertainty quality

## Deep Ensembles

Uncertainty estimation through training multiple independent networks.

- [Deep Ensemble Fundamentals](ensembles/fundamentals.md) -- Why ensembles work: multiple loss landscape modes and the predictive distribution
- [Training Strategies](ensembles/training.md) -- Independent training with diverse initialization for effective ensemble members
- [Uncertainty Decomposition](ensembles/decomposition.md) -- Separating epistemic and aleatoric components from ensemble predictions
- [Diversity Methods](ensembles/diversity.md) -- Random initialization, bootstrapping, and architecture diversity for ensemble quality
- [Efficient Ensembles](ensembles/efficient.md) -- BatchEnsemble and other techniques reducing compute while retaining uncertainty quality

## Bayesian Methods

Principled Bayesian approaches to neural network uncertainty quantification.

- [BNN Fundamentals](bayesian_methods/fundamentals.md) -- Placing probability distributions over weights for principled uncertainty estimation
- [Variational Bayesian Neural Networks](bayesian_methods/variational_bnn.md) -- Scalable approximate posterior inference via ELBO maximization
- [MCMC Methods for BNNs](bayesian_methods/mcmc_bnn.md) -- Hamiltonian Monte Carlo and stochastic gradient variants as gold-standard inference
- [Laplace Approximation](bayesian_methods/laplace.md) -- Post-hoc Gaussian approximation around trained weights for uncertainty without retraining
- [SWAG](bayesian_methods/swag.md) -- Stochastic Weight Averaging Gaussian fitting a posterior from SGD trajectory statistics
- [Posterior Inference](bayesian_methods/posterior.md) -- Central computational challenge of Bayesian neural networks and approximation methods
- [Prior Distributions](bayesian_methods/priors.md) -- Principled approaches to specifying weight priors and their effect on uncertainty
- [Method Comparison](bayesian_methods/comparison.md) -- Comprehensive comparison guide for selecting the right BNN method
- [MC Dropout as Bayesian Inference](bayesian_methods/mc_dropout_connection.md) -- The formal connection between dropout regularization and approximate Bayesian inference

## Calibration

Ensuring predicted probabilities reflect true outcome frequencies.

- [Calibration Fundamentals](calibration/fundamentals.md) -- Mathematical definition and importance of calibration in financial applications
- [Temperature Scaling](calibration/temperature.md) -- Single-parameter post-hoc calibration that preserves model accuracy
- [Platt Scaling](calibration/platt.md) -- Logistic regression mapping from uncalibrated scores to calibrated probabilities
- [Isotonic Regression](calibration/isotonic.md) -- Non-parametric monotonic calibration for flexible probability mapping
- [Focal Loss](calibration/focal_loss.md) -- Training-time calibration improvement by down-weighting well-classified examples
- [Calibration Metrics](calibration/metrics.md) -- Expected Calibration Error (ECE) and related quantitative calibration measures

## Evaluation

Assessing the quality of uncertainty estimates.

- [Proper Scoring Rules](evaluation/scoring_rules.md) -- Log score, Brier score, and CRPS for evaluating probabilistic predictions
- [Reliability Diagrams](evaluation/reliability.md) -- Visualizing calibration by plotting observed frequency against predicted probability
- [Sharpness](evaluation/sharpness.md) -- Measuring predictive distribution concentration while maintaining calibration
- [Uncertainty Quantification](evaluation/uncertainty_quantification.md) -- Comprehensive framework for evaluating epistemic and aleatoric uncertainty estimates
- [Coverage Analysis](evaluation/coverage.md) -- Measuring whether prediction intervals achieve their nominal confidence levels

## Out-of-Distribution Detection

Identifying inputs that differ significantly from the training distribution.

- [OOD Detection Fundamentals](ood/fundamentals.md) -- Problem formulation, types of distribution shift, and detection scoring
- [Softmax Baseline (MSP)](ood/softmax.md) -- Maximum softmax probability as a simple but effective OOD detection baseline
- [ODIN](ood/odin.md) -- Temperature scaling with input perturbation for improved OOD detection
- [Energy-Based Detection](ood/energy.md) -- Log-sum-exp of logits as a theoretically motivated OOD score
- [Mahalanobis Distance](ood/mahalanobis.md) -- Class-conditional Gaussian distance in feature space for OOD detection
- [OOD Detection Metrics](ood/metrics.md) -- AUROC, FPR@95TPR, and AUPR for evaluating detection performance

## Finance Applications

Applying uncertainty quantification to quantitative finance decision-making.

- [Practical Applications](finance/practical_applications.md) -- Active learning, selective prediction, and uncertainty-driven decision systems
- [Portfolio Uncertainty](finance/portfolio.md) -- Uncertainty-aware portfolio optimization with conservative allocation and Black-Litterman integration
- [Risk Estimation](finance/risk_estimation.md) -- Uncertainty-aware Value-at-Risk and tail risk assessment
- [Regime Detection](finance/regime.md) -- Using rising epistemic uncertainty as an early warning signal for regime changes
- [Model Confidence Scoring](finance/confidence.md) -- Selective trading based on uncertainty thresholds and confidence-based position sizing
- [Prediction Intervals](finance/intervals.md) -- Well-calibrated prediction intervals for financial time series incorporating model and market noise
