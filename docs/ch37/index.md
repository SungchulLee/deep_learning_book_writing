# Chapter 37: Bias and Fairness

This chapter addresses the ethical, legal, and technical challenges of ensuring machine learning systems do not discriminate against individuals or groups based on protected characteristics. We cover the mathematical foundations of fairness definitions, impossibility theorems that constrain what fairness criteria can be simultaneously achieved, and practical mitigation techniques spanning pre-processing, in-processing, and post-processing approaches. Special attention is given to fairness in financial applications where regulatory requirements are particularly stringent.

---

## Foundations

Core concepts, historical context, and sources of bias in ML systems.

- [Introduction to Algorithmic Fairness](foundations/introduction.md) -- The fairness problem, motivating examples, and why fairness matters in ML
- [Sources of Bias in ML Systems](foundations/sources.md) -- Taxonomy of bias from historical data, representation, measurement, and deployment
- [Historical Context and Landmark Cases](foundations/history.md) -- Legal foundations, disparate treatment vs. disparate impact, and the 80% rule

## Fairness Definitions

Mathematical formalizations of what it means for a model to be fair.

- [Demographic Parity](definitions/demographic_parity.md) -- Equal positive prediction rates across protected groups (statistical parity)
- [Equal Opportunity](definitions/equal_opportunity.md) -- Equal true positive rates ensuring qualified individuals receive equal treatment
- [Equalized Odds](definitions/equalized_odds.md) -- Equal TPR and FPR across groups via conditional independence of predictions and attributes
- [Calibration](definitions/calibration.md) -- Predicted probabilities accurately reflecting true outcome rates within each group
- [Individual Fairness](definitions/individual_fairness.md) -- Similar individuals receiving similar predictions via Lipschitz continuity constraints
- [Counterfactual Fairness](definitions/counterfactual.md) -- Causal reasoning ensuring predictions are invariant to counterfactual group membership changes

## Impossibility Theorems

Mathematical results showing fundamental tradeoffs between fairness criteria.

- [Chouldechova's Theorem](impossibility/chouldechova.md) -- Proving calibration and equal error rates cannot coexist when base rates differ
- [KMR Impossibility Theorem](impossibility/kmr.md) -- Kleinberg-Mullainathan-Raghavan result on calibration vs. balance incompatibility
- [Tradeoff Analysis](impossibility/tradeoffs.md) -- Quantifying the fairness-accuracy Pareto frontier and finding optimal solutions

## Fairness Metrics

Quantitative measurement of bias across protected groups.

- [Statistical Fairness Metrics](metrics/statistical.md) -- Unified PyTorch framework for computing demographic parity, equal opportunity, and equalized odds
- [Causal Fairness Metrics](metrics/causal.md) -- Total, direct, and indirect causal effects of protected attributes on predictions
- [Multi-Group Fairness Metrics](metrics/multi_group.md) -- Extending binary fairness measures to handle multiple protected groups
- [Intersectionality in Fairness](metrics/intersectionality.md) -- Detecting bias in intersectional subgroups that may be hidden in marginal analysis

## Pre-processing Methods

Mitigating bias by transforming training data before model training.

- [Reweighing](preprocessing/reweighing.md) -- Assigning sample weights to remove correlation between protected attributes and labels
- [Disparate Impact Remover](preprocessing/disparate_impact.md) -- Transforming feature distributions to be identical across protected groups
- [Fair Representation Learning](preprocessing/representation.md) -- Learning latent representations that retain predictive signal while removing group information
- [Data Augmentation for Fairness](preprocessing/augmentation.md) -- Creating synthetic samples to balance the joint distribution of attributes and labels

## In-processing Methods

Incorporating fairness objectives directly into the model training procedure.

- [Adversarial Debiasing](inprocessing/adversarial.md) -- Training a predictor to fool an adversary that tries to infer protected attributes
- [Fairness Constraints](inprocessing/constraints.md) -- Constrained optimization with explicit fairness bounds via Lagrangian relaxation
- [Fairness Regularization](inprocessing/regularization.md) -- Adding fairness penalty terms to the training loss for soft bias mitigation
- [Multi-Objective Fairness Optimization](inprocessing/multi_objective.md) -- Pareto-optimal solutions balancing accuracy with multiple fairness criteria

## Post-processing Methods

Adjusting model outputs after training to satisfy fairness criteria.

- [Threshold Optimization](postprocessing/thresholds.md) -- Group-specific classification thresholds to achieve fairness while maximizing accuracy
- [Calibrated Equalized Odds](postprocessing/calibrated_eo.md) -- Achieving approximate equalized odds while preserving calibration via randomized thresholds
- [Reject Option Classification](postprocessing/reject_option.md) -- Deferring uncertain predictions to improve fairness near the decision boundary

## Evaluation

Systematic assessment of fairness in deployed models.

- [Fairness Audits](evaluation/audits.md) -- Comprehensive five-stage audit framework combining quantitative metrics and qualitative analysis
- [Disparate Impact Testing](evaluation/testing.md) -- Statistical hypothesis tests for determining whether fairness violations are significant
- [Longitudinal Fairness Analysis](evaluation/longitudinal.md) -- Monitoring fairness metrics over time to detect drift and emerging biases

## Finance Applications

Fairness considerations specific to financial services and regulatory compliance.

- [Fair Credit Scoring](finance/credit_scoring.md) -- ECOA-compliant credit models with adverse action explanations and disparate impact testing
- [Fair Insurance Pricing](finance/insurance.md) -- Actuarial fairness under evolving regulations restricting demographic pricing variables
- [Fairness in Algorithmic Trading](finance/trading.md) -- Equitable execution quality, market access, and information asymmetry across participant types
- [Regulatory Framework](finance/regulatory.md) -- Mapping ECOA, Fair Housing Act, GDPR, and SR 11-7 requirements to technical fairness implementations

## Code Examples

Practical implementations and toolkit overviews.

- [Bias and Fairness Toolkit Overview](fairness_code/bias_and_fairness_overview.md) -- Python implementations of bias detection, fairness metrics, and mitigation techniques
