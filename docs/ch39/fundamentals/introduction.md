# Introduction to Model Uncertainty

## Why Uncertainty Quantification Matters

Deep learning models deployed in production make consequential decisions—from medical diagnoses and autonomous driving to financial trading and credit scoring. Yet standard neural networks produce point predictions with no indication of reliability. A model that outputs a confident prediction on an input far from its training distribution is not just unhelpful—it is actively dangerous.

Uncertainty quantification (UQ) addresses this fundamental limitation by transforming point predictions into probability distributions. Instead of asking "What is the model's prediction?", we ask "What is the distribution of plausible outcomes, and how confident should we be?"

## The Overconfidence Problem

Modern neural networks are notoriously overconfident. A classifier might predict a class with 99% softmax probability while being wrong 20% of the time. This miscalibration arises from several sources:

**Cross-entropy training** encourages pushing softmax probabilities toward 0 or 1, because the loss $\mathcal{L} = -\log(p_y)$ drives the correct-class logit arbitrarily high.

**High model capacity** allows deep networks to memorize training data, producing sharp decision boundaries even in regions of genuine ambiguity.

**Batch normalization** can amplify logit magnitudes, and **ReLU activations** produce unbounded outputs.

The result is that neural network confidence scores are unreliable for decision-making without additional calibration or uncertainty estimation.

## The Bayesian Framework for Uncertainty

The principled approach to uncertainty is Bayesian: instead of learning a single set of weights $\hat{\mathbf{w}}$, maintain a posterior distribution over weights given data:

$$p(\mathbf{w}|\mathcal{D}) = \frac{p(\mathcal{D}|\mathbf{w}) p(\mathbf{w})}{p(\mathcal{D})}$$

The posterior predictive distribution then integrates over all plausible weight configurations:

$$p(y^*|\mathbf{x}^*, \mathcal{D}) = \int p(y^*|\mathbf{x}^*, \mathbf{w}) p(\mathbf{w}|\mathcal{D}) d\mathbf{w}$$

This integral is intractable for neural networks, motivating the family of approximate methods that form the core of this chapter: MC Dropout, deep ensembles, variational Bayesian neural networks, SWAG, and Laplace approximation.

## Uncertainty in Quantitative Finance

Financial applications present particularly compelling use cases for uncertainty quantification:

**Position sizing** — Overconfident return predictions lead to excessive position sizes and blow-up risk. Uncertainty-aware models naturally scale positions inversely with prediction uncertainty.

**Risk management** — Value-at-Risk and Expected Shortfall estimates require well-calibrated tail probabilities, not just point forecasts.

**Regime detection** — Rising epistemic uncertainty can signal distributional shift—a regime change where historical patterns may no longer apply.

**Model selection** — Bayesian model comparison via marginal likelihoods provides a principled framework for choosing among competing strategies.

**Regulatory compliance** — Financial regulators increasingly require quantified model risk, making uncertainty estimation a compliance necessity.

## Chapter Roadmap

This chapter progresses from foundational concepts to advanced methods and practical applications:

1. **Fundamentals** establish the mathematical framework: uncertainty types, decomposition, and the calibration problem
2. **MC Dropout** provides the simplest path from existing models to uncertainty estimates
3. **Deep Ensembles** offer state-of-the-art uncertainty quality with straightforward implementation
4. **Bayesian Neural Networks** give the most principled treatment through variational inference, MCMC, and approximation methods
5. **Calibration** methods ensure that stated confidences match empirical frequencies
6. **Evaluation** covers proper scoring rules and diagnostic tools for assessing uncertainty quality
7. **OOD Detection** addresses the critical problem of identifying inputs outside the training distribution
8. **Finance Applications** demonstrates end-to-end uncertainty-aware systems for risk estimation, portfolio construction, and regime detection

Throughout, we emphasize both mathematical rigor and practical PyTorch implementations, with particular attention to the unique challenges of financial data.
