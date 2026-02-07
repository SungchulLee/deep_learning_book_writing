# Chapter 38: Model Uncertainty

## Overview

Uncertainty quantification is essential for deploying deep learning models in high-stakes applications. A model that cannot express "I don't know" may confidently make catastrophic predictions on out-of-distribution inputs. This chapter provides a comprehensive treatment of uncertainty estimation methods, from simple ensemble approaches to principled Bayesian techniques, with emphasis on applications in quantitative finance.

## Why Uncertainty Matters

Consider these scenarios where uncertainty is critical:

| Domain | Failure Mode | Consequence |
|--------|--------------|-------------|
| Medical Diagnosis | Confident wrong diagnosis | Patient harm |
| Autonomous Driving | Confident in unusual scenario | Accidents |
| Financial Trading | Overconfident position sizing | Large losses |
| Credit Scoring | Confident on edge cases | Regulatory violations |
| Portfolio Construction | Ignoring model uncertainty | Concentrated risk |

Standard neural networks produce point predictions without any measure of reliability. Uncertainty quantification transforms predictions into probability distributions, enabling principled decision-making under uncertainty.

## Mathematical Framework

### Bayesian Predictive Distribution

The gold standard for uncertainty is the Bayesian predictive distribution:

$$p(y^*|\mathbf{x}^*, \mathcal{D}) = \int p(y^*|\mathbf{x}^*, \mathbf{w}) p(\mathbf{w}|\mathcal{D}) d\mathbf{w}$$

This integral marginalizes over all possible model parameters, weighted by their posterior probability. In practice, this integral is intractable for neural networks, motivating the approximate methods covered in this chapter.

### Total Predictive Uncertainty

The law of total variance decomposes predictive uncertainty:

$$\text{Var}[y|\mathbf{x}, \mathcal{D}] = \underbrace{\mathbb{E}_{\mathbf{w}}[\text{Var}[y|\mathbf{x}, \mathbf{w}]]}_{\text{Aleatoric}} + \underbrace{\text{Var}_{\mathbf{w}}[\mathbb{E}[y|\mathbf{x}, \mathbf{w}]]}_{\text{Epistemic}}$$

This decomposition is fundamental to understanding what type of uncertainty your model is capturing—and what actions are appropriate in response.

## Chapter Structure

### 38.1 Fundamentals
Epistemic vs aleatoric uncertainty, prediction intervals vs confidence intervals, softmax temperature and overconfidence, and why neural networks are poorly calibrated.

### 38.2 Monte Carlo Dropout
Dropout as approximate Bayesian inference, the MC Dropout algorithm, uncertainty estimation from multiple forward passes, sample convergence analysis, and dropout rate selection.

### 38.3 Deep Ensembles
Ensemble construction and training, uncertainty from model disagreement, diversity methods, uncertainty decomposition, and efficient ensemble techniques.

### 38.4 Bayesian Neural Networks
Weight distributions vs point estimates, Bayes by Backprop, variational inference for BNNs, MCMC methods, SWAG, and Laplace approximation.

### 38.5 Calibration
Temperature scaling, isotonic regression, Platt scaling, focal loss for calibration, and calibration metrics.

### 38.6 Evaluation
Proper scoring rules, reliability diagrams, sharpness, and coverage analysis.

### 38.7 OOD Detection
Out-of-distribution detection fundamentals, softmax baseline, energy-based methods, Mahalanobis distance, ODIN, and evaluation metrics.

### 38.8 Finance Applications
Risk estimation with uncertainty, portfolio uncertainty quantification, prediction intervals for financial time series, model confidence scoring, and uncertainty-based regime detection.

## Method Comparison

| Method | Epistemic | Aleatoric | Compute Cost | Implementation |
|--------|-----------|-----------|--------------|----------------|
| Single Model | ✗ | ✗ | Low | Trivial |
| MC Dropout | ✓ | ✗ | Medium | Easy |
| Deep Ensemble | ✓ | ✓ | High | Easy |
| Bayesian NN | ✓ | ✓ | High | Hard |
| SWAG | ✓ | ✗ | Medium | Medium |
| Laplace | ✓ | ✗ | Low | Medium |

## Practical Recommendations

**Resource-constrained, fast inference**: MC Dropout (single model, multiple passes)

**Best uncertainty quality**: Deep Ensemble (5 models typically sufficient)

**Principled Bayesian treatment**: Bayes by Backprop or SWAG

**Post-hoc calibration**: Temperature scaling (simple, effective)

**Financial applications**: Deep Ensembles + temperature scaling for well-calibrated prediction intervals

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Distinguish** between epistemic and aleatoric uncertainty and their implications for decision-making
2. **Implement** MC Dropout for uncertainty estimation with convergence diagnostics
3. **Build** and train deep ensembles with diversity-promoting techniques
4. **Apply** Bayesian Neural Networks for principled uncertainty quantification
5. **Calibrate** model predictions using temperature scaling, Platt scaling, and isotonic regression
6. **Evaluate** uncertainty quality with proper scoring rules and reliability diagrams
7. **Detect** out-of-distribution inputs using energy-based and distance-based methods
8. **Deploy** uncertainty-aware models for financial risk estimation and portfolio construction

## Prerequisites

| Topic | Chapter | Key Concepts |
|-------|---------|--------------|
| Feedforward Networks | Ch 5 | Architecture, training loops |
| Regularization | Ch 9 | Dropout mechanism |
| Loss Functions | Ch 8 | Cross-entropy, NLL |
| Bayesian Inference | Ch 12-13 | Posterior, prior, likelihood |
| Probability Theory | — | Variance, entropy, KL divergence |

## Key Papers

1. **Gal & Ghahramani (2016)** — "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning"
2. **Lakshminarayanan et al. (2017)** — "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"
3. **Blundell et al. (2015)** — "Weight Uncertainty in Neural Networks"
4. **Guo et al. (2017)** — "On Calibration of Modern Neural Networks"
5. **Maddox et al. (2019)** — "A Simple Baseline for Bayesian Inference in Deep Learning" (SWAG)
6. **Liu et al. (2020)** — "Energy-based Out-of-distribution Detection"
7. **Kendall & Gal (2017)** — "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"

## Connections to Other Chapters

| Chapter | Connection |
|---------|------------|
| Ch 13 Bayesian Methods | MCMC, variational inference foundations |
| Ch 30 Bias and Fairness | Uncertainty in fairness metrics |
| Ch 29 Interpretability | Uncertainty as explanation |
| Ch 21 Deployment | Production uncertainty monitoring |
| Ch 37 Ensemble Methods | Ensemble construction foundations |
