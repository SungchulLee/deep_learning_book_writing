# Self-Training for Domain Adaptation

## Introduction

Self-training is a semi-supervised learning approach where models iteratively label unlabeled target domain data and retrain using these pseudo-labels. In domain adaptation contexts, self-training enables leveraging abundant unlabeled target data to improve model performance when labeled target examples are unavailable. The method is intuitive, generally applicable across architectures, and often surprisingly effective despite its simplicity.

For quantitative finance, self-training proves particularly valuable: historical market data is abundant and naturally unlabeled, while obtaining high-quality labels requires expert annotation. By iteratively refining predictions on recent market data, self-training enables models to adapt to changing market regimes without requiring costly manual labeling.

## Key Concepts

- **Pseudo-Labels**: Model predictions used as training labels for unlabeled target data
- **Confidence Thresholding**: Using only high-confidence predictions as pseudo-labels
- **Iterative Refinement**: Repeat training, prediction, and retraining cycles
- **Label Noise**: Risk of error propagation when pseudo-labels are incorrect
- **Curriculum Learning**: Start with easiest samples, gradually include harder ones

## Self-Training Algorithm

### Basic Iterative Procedure

```
1. Train model on source domain labeled data
2. Repeat:
   a. Predict on target domain unlabeled data
   b. Select high-confidence predictions as pseudo-labels
   c. Retrain model on source + pseudo-labeled target data
   d. Evaluate convergence or stopping criteria
```

### Formal Algorithm

```
Input: Source labeled data D_s, Target unlabeled data D_t
       Initial model θ₀, confidence threshold τ, max iterations T

θ ← θ₀
for t = 1 to T:
    Predict on D_t: {ŷ, p(ŷ)} = model(D_t; θ)
    Select confident samples: M_t = {i : max p_i(ŷ_i) ≥ τ}
    Pseudo-label: D_pseudo = {(x_i, ŷ_i) : i ∈ M_t}
    Combine: D_combined = D_s ∪ D_pseudo
    Retrain: θ ← train(D_combined)
    if converged: break
```

## Mathematical Framework

### Confidence-Based Selection

Select pseudo-labels based on maximum softmax probability:

$$M_t = \{i : \max_y p(y | \mathbf{x}_i; \theta_t) \geq \tau\}$$

**Confidence Threshold $\tau$**: Typically 0.95-0.99 for classification

**Proportion Selected**: Usually 30-50% of target data per iteration

### Loss Function with Pseudo-Labels

Combine supervised and unsupervised components:

$$\mathcal{L} = \mathcal{L}_s(\mathbf{x}_s, \mathbf{y}_s) + \lambda \mathcal{L}_u(\mathbf{x}_t, \hat{\mathbf{y}}_t)$$

where:
- $\mathcal{L}_s$ is supervised loss on source data
- $\mathcal{L}_u$ is loss on pseudo-labeled target data
- $\lambda$ balances contributions

## Advanced Self-Training Variants

### Weighted Self-Training

Weight pseudo-labeled samples by confidence:

$$\mathcal{L}_u = \sum_i \max_y p(y | \mathbf{x}_i) \cdot \ell(\hat{y}_i, y_i)$$

High-confidence predictions contribute more to gradient updates.

### Temporal Self-Training

In sequential settings, use recent successful predictions:

$$\text{confidence}_t = \text{accuracy}(\hat{y}_{t-1}, y_t^{\text{true}})$$

!!! tip "Market Adaptation"
    In financial applications, track self-training accuracy on recent held-out sets to gauge adaptation quality.

### Co-Training

Multiple models generate pseudo-labels for each other:

1. **Model A** predicts on target, high-confidence samples selected
2. **Model B** trained on source + Model A pseudo-labels
3. **Model B** predicts on target, high-confidence samples selected
4. **Model A** retrained on source + Model B pseudo-labels
5. **Repeat**

Diversity between models prevents error accumulation.

## Challenges and Solutions

### Label Noise and Error Propagation

!!! warning "Pseudo-Label Quality"
    Incorrect pseudo-labels accumulate over iterations, degrading performance. This is self-training's primary limitation.

#### Strategies to Mitigate

**Confidence Thresholding**: Only use high-confidence predictions (reduces coverage but increases accuracy)

**Ensemble Voting**: Use multiple models, only pseudo-label when agreeing

**Clean Label Selection**: Use validation set performance to detect when pseudo-labels degrade

**Mixup-Based Smoothing**: Blend pseudo-labels with uniform distribution

$$\tilde{y}_i = (1 - \alpha) \hat{y}_i + \alpha \cdot \mathbf{1}/C$$

where $\alpha$ is small mixing coefficient.

### Convergence to Poor Solutions

Models may converge to incorrect local optima by over-relying on noisy pseudo-labels.

**Progressive Expansion**: Start with easiest samples (highest confidence), expand gradually:

$$\tau(t) = \tau_{\text{init}} \cdot (1 - e^{-t/\sigma})$$

Confidence threshold decreases (more samples included) as training progresses.

**Auxiliary Tasks**: Combine with auxiliary objectives (reconstruction, consistency) to regularize learning.

## Confidence Estimation

### Calibration for Better Selection

Model confidence may not reflect true accuracy. Calibrate using validation data:

$$\text{Calibrated\_Confidence} = \text{Platt Scaling}(\text{Raw\_Confidence})$$

or temperature scaling:

$$p_{\text{calibrated}}(y|\mathbf{x}) = \text{softmax}(\text{logits}/T)$$

### Entropy-Based Confidence

Use prediction entropy instead of maximum probability:

$$H = -\sum_y p(y|\mathbf{x}) \log p(y|\mathbf{x})$$

Low entropy indicates high confidence. Select samples with $H < H_{\text{threshold}}$.

## Self-Training vs. Other Adaptation Methods

| Method | Simplicity | Performance | Theory | Requirements |
|--------|-----------|-------------|--------|--------------|
| **Self-Training** | Very High | Good | Weak | Unlabeled target |
| **MMD-Based** | Medium | Very Good | Strong | Unlabeled target |
| **Adversarial DA** | Medium | Excellent | Strong | Unlabeled target |
| **Self-Distillation** | High | Good | Moderate | Unlabeled target |

## Applications in Quantitative Finance

!!! warning "Market Regime Self-Training"
    
    1. **Train** on historical labeled data from similar market regime
    2. **Apply** to current unlabeled market data
    3. **Threshold** predictions by confidence (e.g., ≥ 0.95)
    4. **Pseudo-Label** high-confidence predictions as market regime
    5. **Retrain** incorporating new pseudo-labeled data
    6. **Monitor** validation set to detect degradation
    7. **Reset** if validation performance drops (regime change detected)

### Risk Model Adaptation

Use self-training to adapt risk models across markets:

**Source**: Historical risk estimates with known outcomes
**Target**: Current market with uncertain true risks
**Process**: Self-training continuously adapts to recent market behavior

## Theoretical Properties

### Convergence Analysis

Under assumptions of feature separability and low label noise:

$$\mathbb{E}[\text{Error}_t] \leq \mathbb{E}[\text{Error}_{t-1}] - \delta + \rho \cdot \text{LabelNoise}_t$$

where $\delta > 0$ is convergence rate and $\rho$ controls noise sensitivity.

### Label Noise Tolerance

Self-training tolerates pseudo-label noise up to threshold before diverging.

**Safe Noise Level**: ~5-10% incorrect labels

**Danger Zone**: > 15% incorrect labels may lead to performance collapse

## Research Directions

- Theoretically-sound confidence thresholding strategies
- Automatic detection of harmful pseudo-labels
- Integration with modern semi-supervised techniques (FixMatch, ReMixMatch)
- Active learning for efficient pseudo-label selection

## Related Topics

- Domain Adaptation Overview (Chapter 10.2)
- Maximum Mean Discrepancy (Chapter 10.2.1)
- Semi-Supervised Learning Techniques
- Adversarial Domain Adaptation
