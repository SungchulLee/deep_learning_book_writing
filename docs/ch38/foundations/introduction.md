# Introduction to Adversarial Robustness

## The Discovery of Adversarial Examples

The discovery that neural networks are vulnerable to carefully crafted input perturbations has been one of the most consequential findings in modern deep learning. Szegedy et al. (2014) first demonstrated that imperceptible perturbations could cause state-of-the-art image classifiers to fail catastrophically, launching an entire subfield of adversarial machine learning.

An **adversarial example** is an input modified by a small, often imperceptible perturbation that causes a trained model to produce an incorrect output with high confidence. Formally, given a classifier $f: \mathcal{X} \to \mathcal{Y}$, an adversarial example $\mathbf{x}'$ for input $\mathbf{x}$ with true label $y$ satisfies:

$$
f(\mathbf{x}') \neq y \quad \text{and} \quad d(\mathbf{x}', \mathbf{x}) \leq \varepsilon
$$

where $d(\cdot, \cdot)$ is a distance metric and $\varepsilon$ is a small perturbation budget.

## Why Do Adversarial Examples Exist?

Several complementary hypotheses explain this phenomenon, each illuminating different aspects of neural network geometry and learning dynamics.

### The Linear Hypothesis

Goodfellow et al. (2015) proposed that adversarial vulnerability arises from the **locally linear** behavior of neural networks in high-dimensional space. Consider a linear model $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$. The change in output from a perturbation $\boldsymbol{\delta}$ is:

$$
f(\mathbf{x} + \boldsymbol{\delta}) - f(\mathbf{x}) = \mathbf{w}^\top \boldsymbol{\delta}
$$

To maximize this change under an $\ell_\infty$ constraint $\|\boldsymbol{\delta}\|_\infty \leq \varepsilon$, the optimal perturbation sets each coordinate to:

$$
\delta_i^* = \varepsilon \cdot \text{sign}(w_i)
$$

yielding maximum change:

$$
\mathbf{w}^\top \boldsymbol{\delta}^* = \varepsilon \|\mathbf{w}\|_1
$$

In high dimensions (e.g., $d = 3 \times 224 \times 224 \approx 150{,}000$ for ImageNet), even a tiny $\varepsilon$ produces large $\varepsilon \|\mathbf{w}\|_1$. Neural networks, which behave approximately linearly in local neighborhoods, inherit this vulnerability. The first-order Taylor expansion makes this precise:

$$
\mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y) \approx \mathcal{L}(f_\theta(\mathbf{x}), y) + \boldsymbol{\delta}^\top \nabla_\mathbf{x} \mathcal{L}(f_\theta(\mathbf{x}), y)
$$

### High-Dimensional Geometry

The geometry of high-dimensional space produces several counterintuitive properties that facilitate adversarial examples:

- **Surface concentration**: Most volume of a high-dimensional ball concentrates near its surface, meaning random points are almost always near the boundary of any enclosing region
- **Decision boundary proximity**: In high dimensions, decision boundaries have enormous surface area relative to the volume they enclose, so most correctly classified points are close to a boundary
- **Distance concentration**: Pairwise distances between random points concentrate around their mean, making the distinction between "nearby" and "far" less meaningful

These geometric facts imply that small perturbations—which seem negligible in low dimensions—can traverse significant distances in feature space and cross decision boundaries.

### Non-Robust Features

Ilyas et al. (2019) offered a complementary perspective by demonstrating that adversarial vulnerability is a consequence of how models use input features. Data contains two types of features:

- **Robust features** ($\rho$-robust): Features that correlate with labels and remain predictive under bounded perturbation. Formally, feature $\phi$ is $(\gamma, \rho)$-robust if $\mathbb{E}[y \cdot \phi(\mathbf{x})] \geq \gamma$ and $\mathbb{E}[\inf_{\|\boldsymbol{\delta}\| \leq \rho} y \cdot \phi(\mathbf{x} + \boldsymbol{\delta})] \geq \gamma$.

- **Non-robust features**: Features that correlate with labels under the natural distribution but are highly sensitive to adversarial perturbation.

Standard training maximizes accuracy by learning **both** robust and non-robust features. Since non-robust features are genuinely predictive (not noise), removing them reduces standard accuracy—explaining the robustness-accuracy tradeoff.

## The Robustness-Accuracy Tradeoff

A fundamental tension exists between standard and robust performance. Define the two risk measures:

$$
\begin{aligned}
\text{Standard Risk: } R_{\text{std}}(f) &= \mathbb{E}_{(\mathbf{x},y) \sim \mathcal{D}}[\mathbf{1}[f(\mathbf{x}) \neq y]] \\
\text{Robust Risk: } R_{\text{rob}}(f) &= \mathbb{E}_{(\mathbf{x},y) \sim \mathcal{D}}\left[\max_{\|\boldsymbol{\delta}\| \leq \varepsilon} \mathbf{1}[f(\mathbf{x} + \boldsymbol{\delta}) \neq y]\right]
\end{aligned}
$$

> **Theorem (Tsipras et al., 2019):** For certain data distributions, any classifier achieving optimal robust accuracy must have strictly suboptimal standard accuracy.

The intuitive explanation is that robust classifiers must ignore non-robust features, which carry genuine predictive signal. Empirically, this manifests as a 5-15% clean accuracy drop when training for adversarial robustness on standard benchmarks like CIFAR-10.

## Attack Formulation as Optimization

Most adversarial attacks are formulated as constrained optimization problems.

**Untargeted attack** (cause any misclassification):

$$
\boldsymbol{\delta}^* = \arg\max_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y)
$$

**Targeted attack** (force a specific prediction $y_{\text{target}}$):

$$
\boldsymbol{\delta}^* = \arg\min_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y_{\text{target}})
$$

The defense problem is the dual: find parameters $\theta$ that minimize the worst-case loss:

$$
\min_\theta \mathbb{E}_{(\mathbf{x},y) \sim \mathcal{D}} \left[ \max_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y) \right]
$$

This min-max formulation underpins adversarial training and motivates the entire taxonomy of attacks and defenses covered in subsequent sections.

## Relevance to Quantitative Finance

Adversarial robustness is not merely an academic concern for financial applications:

- **Model integrity**: Trading systems using ML-based signals must be robust to adversarial manipulation of input data (e.g., spoofed market data feeds)
- **Regulatory compliance**: Financial models deployed in production require demonstrable robustness guarantees, particularly for credit scoring and risk assessment
- **Fraud detection**: Adversaries actively attempt to craft inputs that evade detection systems while maintaining fraudulent intent
- **Market manipulation**: Understanding adversarial attacks informs the design of manipulation-resistant surveillance systems

Throughout this chapter, we connect each theoretical concept and implementation to concrete financial applications.

## References

1. Szegedy, C., et al. (2014). "Intriguing Properties of Neural Networks." ICLR.
2. Goodfellow, I., Shlens, J., & Szegedy, C. (2015). "Explaining and Harnessing Adversarial Examples." ICLR.
3. Ilyas, A., et al. (2019). "Adversarial Examples Are Not Bugs, They Are Features." NeurIPS.
4. Tsipras, D., et al. (2019). "Robustness May Be at Odds with Accuracy." ICLR.
5. Gilmer, J., et al. (2018). "Adversarial Spheres." ICLR Workshop.
