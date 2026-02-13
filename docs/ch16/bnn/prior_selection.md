# Prior Selection for Bayesian Neural Networks

## Introduction

Prior selection is a critical design decision in Bayesian Neural Networks that profoundly influences posterior inference, regularization strength, and ultimately model performance. The prior encodes domain knowledge and structural assumptions about the weight distribution, serving simultaneously as regularization and as a mechanism for incorporating expert knowledge into the learning process.

In quantitative finance, prior selection becomes particularly important given the presence of multiple distributional regimes, fat tails, and the need to incorporate risk constraints into the learning framework. Improper prior specification can lead to posterior inference that poorly captures market microstructure and systemic risks, while well-designed priors can substantially improve model robustness to out-of-distribution scenarios and enhance generalization to unseen market conditions.

This section examines practical and theoretical aspects of prior selection, including standard choices, data-dependent priors, hierarchical structures, and empirical Bayes approaches that enable automatic prior specification from data.

## Key Concepts

### Standard Prior Choices
- **Gaussian Priors**: $w \sim \mathcal{N}(0, \sigma_w^2)$ - most common in practice
- **Laplace Priors**: $w \sim \text{Laplace}(0, b)$ - promotes sparsity
- **Horseshoe Priors**: Heavy-tailed priors for sparse weight selection
- **Spike-and-Slab**: Mixture distributions for explicit sparsity

### Prior Hyperparameters
- **Scale Parameters**: $\sigma_w$ determines regularization strength
- **Hierarchy**: Hyperpriors on hyperparameters for adaptive regularization
- **Correlation Structure**: Dependencies between weight groups

## Mathematical Framework

### Standard Gaussian Prior

The most commonly used prior is a factorized Gaussian:

$$p(w) = \prod_{i=1}^{D} \mathcal{N}(w_i | 0, \sigma_w^2)$$

This corresponds to L2 regularization (weight decay) in the maximum a posteriori (MAP) estimate:

$$\mathcal{L}_{\text{regularized}} = \mathcal{L}(w) + \frac{\lambda}{2}\|w\|_2^2$$

with $\lambda = 1/\sigma_w^2$.

### Hierarchical Priors

For improved flexibility, hierarchical priors introduce level-specific hyperpriors:

$$p(w|\tau) = \prod_{i=1}^{D} \mathcal{N}(w_i | 0, \tau_i)$$

$$p(\tau) = \prod_{i=1}^{D} \text{Gamma}(\tau_i | a, b)$$

This allows the posterior to automatically determine appropriate regularization strength for different parameters.

### Weight-Decaying Priors for Financial Networks

For neural networks applied to financial time series, a common choice is:

$$w \sim \mathcal{N}(0, \sigma_w^2 \mathbf{I})$$

where $\sigma_w$ is typically set based on layer width to maintain gradient flow:

$$\sigma_w^2 \approx \frac{2}{n_{\text{in}} + n_{\text{out}}}$$

This initialization-based prior helps preserve gradient information through deep networks.

## Prior Design Strategies

### Data-Dependent Priors

Empirical Bayes methods determine hyperprior parameters from marginal likelihood optimization:

$$\hat{\sigma}_w = \arg\max_{\sigma_w} p(\mathcal{D}|\sigma_w) = \int p(\mathcal{D}|w)p(w|\sigma_w) dw$$

This provides automatic prior tuning without manual hyperparameter selection.

### Domain-Informed Priors

In quantitative finance, domain knowledge can guide prior specification:

1. **Volatility Forecasting**: Prior on networks handling normalized returns should reflect stable dynamics
2. **Portfolio Optimization**: Constraints on weight magnitudes encode risk limits
3. **Market Microstructure**: Priors on high-frequency components enforce stability

### Layer-Wise Prior Heterogeneity

Different layers may benefit from different priors:

$$p(w^{(l)}) = \mathcal{N}(0, \sigma_l^2 \mathbf{I})$$

where $\sigma_l^2$ is optimized per layer through empirical Bayes, allowing deeper layers to be more constrained than input layers.

## Sparsity-Inducing Priors

### Laplace Prior

The Laplace distribution:

$$p(w_i) = \frac{1}{2b}\exp\left(-\frac{|w_i|}{b}\right)$$

induces sparsity through its sharp peak at zero, corresponding to L1 regularization.

### Horseshoe Prior

For selective sparsity with heavy tails:

$$p(w_i|\lambda_i) = \mathcal{N}(0, \lambda_i^2)$$
$$p(\lambda_i) = \text{Exponential}(\nu)$$

This allows small weights to remain zero while large weights retain full posterior uncertainty.

## Empirical Bayes Approaches

### Automatic Relevance Determination (ARD)

ARD priors automatically determine the importance of different input features:

$$p(w_j|\alpha_j) = \mathcal{N}(0, \alpha_j^{-1})$$

Maximizing the marginal likelihood concentrates posterior mass on features with strong data support while suppressing irrelevant features.

## Practical Guidelines

For financial applications, we recommend:

1. **Start with Gaussian Priors**: $\mathcal{N}(0, 0.1^2)$ as default, justified by neural network initialization literature
2. **Use Layer-Wise Hyperpriors**: Allow different layers separate precision parameters via empirical Bayes
3. **Validate with Cross-Validation**: Test sensitivity to prior specification across train/test splits
4. **Consider Domain Constraints**: Incorporate financial constraints (e.g., weight bounds) through modified priors
5. **Monitor KL Divergence**: Ensure priors are informative but not overly restrictive

!!! warning "Prior Sensitivity"
    Posterior inference can be sensitive to prior specification, particularly with limited data. Always conduct sensitivity analysis by varying prior scale parameters and comparing predictive performance across specifications.

