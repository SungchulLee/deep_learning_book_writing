# Chapter 37: Adversarial Robustness

## Overview

Adversarial robustness is the study of neural network vulnerabilities to deliberately crafted input perturbations and the development of defenses against such attacks. This chapter provides a comprehensive treatment of adversarial machine learning, from theoretical foundations through practical implementations and applications in quantitative finance.

## Motivation

Neural networks achieve remarkable performance across diverse tasks, yet they exhibit a surprising fragility: small, carefully designed perturbations to inputs can cause dramatic misclassifications. These **adversarial examples** pose fundamental questions about the reliability and security of deep learning systems.

### The Adversarial Example Phenomenon

Consider a well-trained image classifier that correctly identifies an image of a panda with 99% confidence. Adding a small perturbation—imperceptible to human observers—can cause the same classifier to confidently predict "gibbon" with even higher probability:

$$
\begin{aligned}
f(\mathbf{x}) &= \text{panda} \quad \text{(confidence: 99\%)} \\
f(\mathbf{x} + \boldsymbol{\delta}) &= \text{gibbon} \quad \text{(confidence: 99.9\%)}
\end{aligned}
$$

where $\|\boldsymbol{\delta}\|_\infty < 0.01$ (each pixel changes by less than 1%).

### Implications for Quantitative Finance

In financial applications, adversarial vulnerabilities have serious implications:

1. **Trading Systems**: An adversarial agent could manipulate market data to trigger erroneous trading signals
2. **Risk Models**: Adversarial perturbations to portfolio positions could mask true risk exposures
3. **Credit Scoring**: Malicious actors could craft applications that game ML-based approval systems
4. **Fraud Detection**: Fraudsters could design transactions that evade detection while maintaining fraudulent intent

## Formal Problem Statement

### Definition of Adversarial Examples

Given a classifier $f: \mathcal{X} \to \mathcal{Y}$ mapping inputs to labels, an **adversarial example** $\mathbf{x}'$ for input $\mathbf{x}$ with true label $y$ satisfies:

$$
\begin{aligned}
&f(\mathbf{x}') \neq y \quad \text{(misclassification)} \\
&\|\mathbf{x}' - \mathbf{x}\|_p \leq \varepsilon \quad \text{(bounded perturbation)}
\end{aligned}
$$

The perturbation constraint uses an $\ell_p$ norm, most commonly:

| Norm | Definition | Interpretation |
|------|------------|----------------|
| $\ell_\infty$ | $\|\boldsymbol{\delta}\|_\infty = \max_i \|\delta_i\|$ | Maximum per-coordinate change |
| $\ell_2$ | $\|\boldsymbol{\delta}\|_2 = \sqrt{\sum_i \delta_i^2}$ | Euclidean distance |
| $\ell_0$ | $\|\boldsymbol{\delta}\|_0 = \|\{i : \delta_i \neq 0\}\|$ | Number of modified features |

### Attack Formulation as Optimization

Most adversarial attacks solve an optimization problem. The **untargeted attack** finds perturbations that maximize loss:

$$
\boldsymbol{\delta}^* = \arg\max_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y)
$$

The **targeted attack** forces prediction toward a specific class $y_{\text{target}}$:

$$
\boldsymbol{\delta}^* = \arg\min_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y_{\text{target}})
$$

## Theoretical Foundations

### Why Do Adversarial Examples Exist?

Several complementary hypotheses explain this phenomenon:

#### 1. Linear Hypothesis (Goodfellow et al., 2015)

Neural networks behave approximately linearly in high-dimensional space. For a linear model $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$, the change in output from perturbation $\boldsymbol{\delta}$ is:

$$
f(\mathbf{x} + \boldsymbol{\delta}) - f(\mathbf{x}) = \mathbf{w}^\top \boldsymbol{\delta}
$$

To maximize this under $\ell_\infty$ constraint, set $\delta_i = \varepsilon \cdot \text{sign}(w_i)$:

$$
\mathbf{w}^\top \boldsymbol{\delta} = \varepsilon \sum_i |w_i| = \varepsilon \|\mathbf{w}\|_1
$$

In high dimensions (e.g., $d = 3 \times 224 \times 224 \approx 150,000$ for ImageNet), even small $\varepsilon$ yields large $\varepsilon \|\mathbf{w}\|_1$.

#### 2. High-Dimensional Geometry

The geometry of high-dimensional space is counterintuitive:
- Decision boundaries have enormous surface area
- Most of the volume of a high-dimensional ball concentrates near its surface
- Small perturbations can traverse significant distances in feature space

#### 3. Non-Robust Features (Ilyas et al., 2019)

Data contains both:
- **Robust features**: Correlate with labels and remain predictive under perturbation
- **Non-robust features**: Correlate with labels but are highly sensitive to perturbation

Standard training learns to exploit both, but only robust features generalize under adversarial conditions.

### The Robustness-Accuracy Tradeoff

A fundamental tension exists between standard and robust accuracy. Define:

$$
\begin{aligned}
\text{Standard Risk: } R_{\text{std}}(f) &= \mathbb{E}_{(\mathbf{x},y) \sim \mathcal{D}}[\mathbf{1}[f(\mathbf{x}) \neq y]] \\
\text{Robust Risk: } R_{\text{rob}}(f) &= \mathbb{E}_{(\mathbf{x},y) \sim \mathcal{D}}\left[\max_{\|\boldsymbol{\delta}\| \leq \varepsilon} \mathbf{1}[f(\mathbf{x} + \boldsymbol{\delta}) \neq y]\right]
\end{aligned}
$$

Theoretical results (Tsipras et al., 2019; Zhang et al., 2019) establish:

> **Theorem (Robustness-Accuracy Tradeoff):** For certain data distributions, any classifier achieving optimal robust accuracy must have strictly suboptimal standard accuracy.

Empirically, this manifests as a typical 5-15% accuracy drop when training for robustness.

## Chapter Structure

This chapter progresses from foundational concepts through advanced applications:

### 37.1 Foundations
- Introduction to adversarial robustness
- Threat models and attack taxonomies
- Perturbation types and norm constraints
- Formal robustness definitions and metrics

### 37.2 White-Box Attacks
- Fast Gradient Sign Method (FGSM)
- Projected Gradient Descent (PGD)
- Carlini-Wagner (C&W) attack
- DeepFool
- AutoAttack

### 37.3 Black-Box Attacks
- Transfer-based attacks
- Query-based methods
- Score-based gradient estimation
- Decision-based boundary attacks

### 37.4 Physical Attacks
- Adversarial patches
- 3D adversarial examples
- Real-world robustness considerations

### 37.5 Adversarial Training
- Standard PGD-based adversarial training
- TRADES: principled accuracy-robustness trade-off
- MART: misclassification-aware robust training
- Free adversarial training
- Fast adversarial training

### 37.6 Certified Defenses
- Randomized smoothing
- Interval Bound Propagation (IBP)
- CROWN
- Lipschitz-constrained networks

### 37.7 Detection Methods
- Statistical detection of adversarial examples
- Feature squeezing
- Input transformation defenses

### 37.8 Evaluation
- Robustness benchmarks and RobustBench
- Adaptive attacks and gradient masking
- Certified accuracy metrics

### 37.9 Finance Applications
- Fraud detection robustness
- Market manipulation detection
- Model security in production

## Prerequisites

This chapter builds upon:

- **Chapter 1-4**: PyTorch fundamentals, tensors, autograd, and training loops
- **Chapter 10-12**: Convolutional neural networks and architectures
- **Chapter 8**: Loss functions and optimization
- **Chapter 28-30**: Financial time series and applications

## Learning Objectives

Upon completing this chapter, you will be able to:

1. Explain why neural networks are vulnerable to adversarial perturbations
2. Implement gradient-based and optimization-based white-box attacks
3. Understand black-box attack strategies including transfer and query-based methods
4. Train robust models using adversarial training techniques (AT, TRADES, MART)
5. Apply certified defense methods with provable guarantees
6. Detect adversarial examples using statistical and feature-based methods
7. Evaluate model robustness using standardized protocols and AutoAttack
8. Design robust machine learning systems for financial applications

## Key Notation

| Symbol | Description |
|--------|-------------|
| $\mathbf{x} \in \mathcal{X}$ | Input (e.g., image, feature vector) |
| $y \in \mathcal{Y}$ | True label |
| $f_\theta: \mathcal{X} \to \mathcal{Y}$ | Classifier with parameters $\theta$ |
| $\mathcal{L}$ | Loss function (e.g., cross-entropy) |
| $\boldsymbol{\delta}$ | Adversarial perturbation |
| $\varepsilon$ | Perturbation budget |
| $\|\cdot\|_p$ | $\ell_p$ norm |
| $\Pi_\varepsilon$ | Projection onto $\varepsilon$-ball |
| $R$ | Certified robustness radius |

## References

1. Szegedy, C., et al. (2014). "Intriguing Properties of Neural Networks." ICLR.
2. Goodfellow, I., Shlens, J., & Szegedy, C. (2015). "Explaining and Harnessing Adversarial Examples." ICLR.
3. Madry, A., et al. (2018). "Towards Deep Learning Models Resistant to Adversarial Attacks." ICLR.
4. Carlini, N., & Wagner, D. (2017). "Towards Evaluating the Robustness of Neural Networks." IEEE S&P.
5. Tsipras, D., et al. (2019). "Robustness May Be at Odds with Accuracy." ICLR.
6. Zhang, H., et al. (2019). "Theoretically Principled Trade-off between Robustness and Accuracy." ICML.
7. Croce, F., & Hein, M. (2020). "Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-Free Attacks." ICML.
8. Ilyas, A., et al. (2019). "Adversarial Examples Are Not Bugs, They Are Features." NeurIPS.
9. Cohen, J., Rosenfeld, E., & Kolter, Z. (2019). "Certified Adversarial Robustness via Randomized Smoothing." ICML.
