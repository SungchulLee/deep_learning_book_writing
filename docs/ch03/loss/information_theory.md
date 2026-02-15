# Information Theory for Deep Learning

## Introduction

Information theory provides a mathematical framework for understanding uncertainty, surprise, and the flow of information through systems. For deep learning practitioners in quantitative finance, this foundation is essential because:

- **Loss Functions**: Classification and regression losses are fundamentally rooted in information-theoretic principles
- **Model Compression**: Entropy and mutual information guide neural network pruning and distillation
- **Representation Learning**: Information bottleneck principle explains why deep networks learn hierarchical features
- **Risk Quantification**: Entropy measures capture portfolio concentration and model uncertainty

This chapter bridges classical information theory and modern deep learning, with applications to quantitative trading and portfolio optimization.

## Self-Information and Surprise

The **self-information** (or *surprise*) of an event is defined as:

$$I(x) = -\log p(x)$$

where $p(x)$ is the probability of event $x$.

### Intuition

- **High probability events** (close to 1) → Low information: We expect them, so they're unsurprising
- **Low probability events** (close to 0) → High information: Rare events carry more "news"
- Logarithm base determines units: base-2 gives bits, base-e gives nats

**Finance Example**: A 0.1% tail event in market returns contains more information (surprise) than a ±2% daily move, reflecting its rarity.

## Entropy: Measuring Uncertainty

**Shannon entropy** quantifies the average uncertainty in a probability distribution:

$$H(X) = -\sum_{x} p(x) \log p(x)$$

### Properties

- $H(X) \geq 0$, with equality only if one outcome has probability 1
- Maximum entropy: Uniform distribution (maximum uncertainty)
- Minimum entropy: Deterministic distribution (zero uncertainty)

### Portfolio Application

For a portfolio's return distribution, entropy measures:
- **Low entropy**: Concentrated returns (e.g., $P(\text{return} = 0.05) = 1.0$)
- **High entropy**: Dispersed returns (many possible outcomes equally likely)

Higher entropy return distributions require more "information" to describe, suggesting greater model uncertainty.

## Cross-Entropy: Matching Probability Distributions

**Cross-entropy** measures the average number of bits needed to encode events from distribution $p$ using a code optimized for distribution $q$:

$$H(p, q) = -\sum_{x} p(x) \log q(x)$$

This is fundamental in classification:
- $p(x)$ = true label distribution (often one-hot)
- $q(x)$ = predicted probability distribution

### Key Relationship

$$H(p, q) = H(p) + D_{\text{KL}}(p \| q)$$

When $p$ is fixed (one-hot encoding), minimizing cross-entropy is equivalent to minimizing KL divergence.

## KL Divergence: Measuring Distribution Distance

**Kullback-Leibler divergence** quantifies how much distribution $q$ diverges from distribution $p$:

$$D_{\text{KL}}(p \| q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)}$$

### Properties

- $D_{\text{KL}}(p \| q) \geq 0$, with equality iff $p = q$ almost everywhere
- **Asymmetric**: $D_{\text{KL}}(p \| q) \neq D_{\text{KL}}(q \| p)$
  - Forward KL: Penalizes placing mass where $p$ is zero (mode-seeking)
  - Reverse KL: Penalizes missing modes of $p$ (mass-covering)

### Connection in Deep Learning

For classification, the true distribution $p$ is often one-hot. Minimizing cross-entropy loss:

$$\mathcal{L} = H(p, q_{\theta}) = -\log q_{\theta}(y_{\text{true}})$$

directly minimizes how far $q_{\theta}$ (network predictions) deviate from ground truth.

!!! note "KL Divergence in the Course"
    This course covers KL divergence extensively in variational autoencoders and probabilistic models. Here we emphasize its role as the mathematical basis for classification loss.

## Mutual Information: Measuring Dependencies

**Mutual information** quantifies dependence between two variables:

$$I(X; Y) = H(X) - H(X | Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$

### Interpretation

- $I(X; Y) = 0$: $X$ and $Y$ are independent
- $I(X; Y) > 0$: Knowing $Y$ reduces uncertainty in $X$
- Symmetric: $I(X; Y) = I(Y; X)$

### Feature Selection

In high-dimensional quant problems, select features $X$ that maximize mutual information with target $Y$:

$$\text{Features} = \arg\max_{\mathcal{S}} \sum_{X \in \mathcal{S}} I(X; Y)$$

Higher MI features contain more predictive information about returns.

## Connection to Loss Functions

Why is **cross-entropy loss the natural choice** for classification?

From information theory:

1. One-hot labels define a deterministic distribution with $H(p) = 0$
2. Minimizing $H(p, q_{\theta})$ forces $q_{\theta}$ closer to this deterministic distribution
3. This is equivalent to maximizing the likelihood $p(y_{\text{true}} | x)$
4. By the coding theorem, it minimizes expected message length for labels

$$\text{Minimize} \ \mathcal{L}_{\text{CE}} = H(p, q_{\theta}) \Leftrightarrow \text{Minimize} \ D_{\text{KL}}(p \| q_{\theta})$$

## Applications in Quantitative Finance

### Information Ratio

The information ratio extends Shannon entropy to portfolio construction:

$$\text{IR} = \frac{E[R_p - R_b]}{\sigma(R_p - R_b)}$$

where $R_p$ is strategy return and $R_b$ is benchmark. High IR signals that the strategy compresses much return in low entropy (low tracking error).

### Entropy-Based Diversification

Portfolio entropy:

$$H(w) = -\sum_{i} w_i \log w_i$$

Maximum entropy portfolios (equal weighting) are most diversified; concentrated portfolios have lower entropy. Constraints on $H(w)$ enforce position limits.

### Information Coefficient

The IC between predicted and realized returns is bounded by mutual information:

$$\text{IC} \leq \sqrt{I(\text{prediction}; \text{returns})}$$

Higher mutual information between model predictions and true returns indicates more exploitable alpha.

## Summary

Information theory provides rigorous mathematical language for:

- **Loss Function Design**: Cross-entropy emerges naturally for classification
- **Model Evaluation**: Entropy and mutual information quantify information quality
- **Portfolio Optimization**: Information-theoretic measures guide diversification
- **Feature Selection**: Mutual information ranks predictive power

The deep learning practitioner should view information theory not as abstract mathematics, but as a practical toolkit for understanding what neural networks learn and how to measure learning progress.

## Further Reading

- MacKay, D.J.C. (2003). *Information Theory, Inference, and Learning Algorithms*
- Cover, T.M. & Thomas, J.A. (2006). *Elements of Information Theory*
- Tishby & Schwartz-Ziv (2015). "Opening the Black Box of Deep Neural Networks via Information"
