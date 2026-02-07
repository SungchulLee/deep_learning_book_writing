# Chapter 36: Bias and Fairness in Deep Learning

## Chapter Overview

Machine learning models can inadvertently learn, perpetuate, and amplify biases present in training data, leading to unfair outcomes for certain demographic groups. As deep learning systems increasingly influence consequential decisions in domains such as hiring, lending, criminal justice, and healthcare, understanding and mitigating algorithmic bias has become both a technical necessity and an ethical imperative.

This chapter provides a comprehensive treatment of bias and fairness in deep learning, covering mathematical foundations, formal fairness definitions, impossibility theorems, detection metrics, and mitigation techniques at every stage of the ML pipeline—all with complete PyTorch implementations and applications to quantitative finance.

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Understand the mathematical foundations** of algorithmic fairness including formal fairness definitions and their statistical properties
2. **Detect and measure bias** in datasets and model predictions using established metrics
3. **Implement fairness metrics** including demographic parity, equal opportunity, equalized odds, calibration, and individual fairness measures
4. **Prove and apply impossibility theorems** showing that certain fairness criteria cannot be simultaneously satisfied
5. **Apply bias mitigation techniques** at the pre-processing, in-processing, and post-processing stages
6. **Navigate trade-offs** between different fairness definitions and between fairness and accuracy
7. **Conduct fairness audits** and longitudinal monitoring of deployed systems
8. **Design fair ML systems** for real-world applications in credit scoring, insurance pricing, algorithmic trading, and regulatory compliance

## Chapter Structure

### 36.1 Foundations
- [Introduction to Algorithmic Fairness](foundations/introduction.md)
- [Sources of Bias in ML Systems](foundations/sources.md)
- [Historical Context and Landmark Cases](foundations/history.md)

### 36.2 Fairness Definitions
- [Demographic Parity](definitions/demographic_parity.md)
- [Equal Opportunity](definitions/equal_opportunity.md)
- [Equalized Odds](definitions/equalized_odds.md)
- [Calibration](definitions/calibration.md)
- [Individual Fairness](definitions/individual_fairness.md)
- [Counterfactual Fairness](definitions/counterfactual.md)

### 36.3 Impossibility Theorems
- [Chouldechova's Theorem](impossibility/chouldechova.md)
- [KMR Impossibility](impossibility/kmr.md)
- [Tradeoff Analysis](impossibility/tradeoffs.md)

### 36.4 Fairness Metrics
- [Statistical Metrics](metrics/statistical.md)
- [Causal Metrics](metrics/causal.md)
- [Multi-Group Metrics](metrics/multi_group.md)
- [Intersectionality](metrics/intersectionality.md)

### 36.5 Pre-processing Mitigation
- [Reweighing](preprocessing/reweighing.md)
- [Disparate Impact Remover](preprocessing/disparate_impact.md)
- [Fair Representation Learning](preprocessing/representation.md)
- [Data Augmentation for Fairness](preprocessing/augmentation.md)

### 36.6 In-processing Mitigation
- [Adversarial Debiasing](inprocessing/adversarial.md)
- [Fairness Constraints](inprocessing/constraints.md)
- [Fairness Regularization](inprocessing/regularization.md)
- [Multi-Objective Optimization](inprocessing/multi_objective.md)

### 36.7 Post-processing Mitigation
- [Threshold Optimization](postprocessing/thresholds.md)
- [Calibrated Equalized Odds](postprocessing/calibrated_eo.md)
- [Reject Option Classification](postprocessing/reject_option.md)

### 36.8 Evaluation and Monitoring
- [Fairness Audits](evaluation/audits.md)
- [Disparate Impact Testing](evaluation/testing.md)
- [Longitudinal Analysis](evaluation/longitudinal.md)

### 36.9 Applications in Finance
- [Credit Scoring](finance/credit_scoring.md)
- [Insurance Pricing](finance/insurance.md)
- [Algorithmic Trading](finance/trading.md)
- [Regulatory Framework](finance/regulatory.md)

## Prerequisites

This chapter assumes familiarity with:

- Basic probability and statistics (conditional probability, Bayes' theorem)
- PyTorch fundamentals (tensors, autograd, `nn.Module`)
- Classification metrics (accuracy, precision, recall, F1-score)
- Binary classification and logistic regression
- Basic neural network training and optimization
- Causal inference concepts (helpful but not required for §36.2.6 and §36.4.2)

## Key Mathematical Notation

| Symbol | Description |
|--------|-------------|
| $Y$ | True label (ground truth) |
| $\hat{Y}$ | Predicted label |
| $A$ | Protected/sensitive attribute |
| $X$ | Input features |
| $S$ | Predicted score/probability |
| $\text{TPR}$ | True Positive Rate $= P(\hat{Y}=1 \mid Y=1)$ |
| $\text{FPR}$ | False Positive Rate $= P(\hat{Y}=1 \mid Y=0)$ |
| $\text{FNR}$ | False Negative Rate $= P(\hat{Y}=0 \mid Y=1)$ |
| $\text{PPV}$ | Positive Predictive Value (Precision) $= P(Y=1 \mid \hat{Y}=1)$ |
| $\text{FDR}$ | False Discovery Rate $= P(Y=0 \mid \hat{Y}=1)$ |

## References

1. Barocas, S., Hardt, M., & Narayanan, A. (2019). *Fairness and Machine Learning*. fairmlbook.org
2. Mehrabi, N., et al. (2021). "A Survey on Bias and Fairness in Machine Learning." *ACM Computing Surveys*, 54(6), 1–35
3. Chouldechova, A. (2017). "Fair Prediction with Disparate Impact: A Study of Bias in Recidivism Prediction Instruments." *Big Data*, 5(2), 153–163
4. Hardt, M., Price, E., & Srebro, N. (2016). "Equality of Opportunity in Supervised Learning." *NeurIPS*
5. Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R. (2012). "Fairness Through Awareness." *ITCS*
6. Kleinberg, J., Mullainathan, S., & Raghavan, M. (2016). "Inherent Trade-Offs in the Fair Determination of Risk Scores." *ITCS*
7. Kusner, M. J., Loftus, J., Russell, C., & Silva, R. (2017). "Counterfactual Fairness." *NeurIPS*
8. Calmon, F. P., et al. (2017). "Optimized Pre-Processing for Discrimination Prevention." *NeurIPS*
9. Zhang, B. H., Lemoine, B., & Mitchell, M. (2018). "Mitigating Unwanted Biases with Adversarial Learning." *AIES*
10. Pleiss, G., Raghavan, M., Wu, F., Kleinberg, J., & Weinberger, K. Q. (2017). "On Fairness and Calibration." *NeurIPS*
