# Generative Adversarial Networks - Fundamentals

## Introduction to Generative vs Discriminative Models

In machine learning, we distinguish between two major classes of probabilistic models:

**Discriminative Models** learn the conditional probability $P(y|x)$ - the relationship between features $x$ and labels $y$. They focus on decision boundaries and are typically used for classification and regression tasks. Examples include logistic regression, SVMs, and most neural networks trained with supervised learning.

**Generative Models** learn the joint distribution $P(x, y)$ or, for unsupervised learning, the data distribution $P(x)$. They can generate new samples from the learned distribution and provide insights into the underlying data structure. Examples include Gaussian Mixture Models, Variational Autoencoders (VAEs), and Generative Adversarial Networks (GANs).

Generative models are particularly valuable when we need to:
- Synthesize realistic new data samples
- Perform data augmentation for limited datasets
- Understand data manifolds
- Generate conditional samples for specific scenarios
- Estimate likelihood of observed data

## GAN Architecture: Generator and Discriminator

A Generative Adversarial Network consists of two neural networks that compete in a game-theoretic framework:

### Generator (G)

The generator $G$ is a differentiable function (neural network) that maps random noise $z \sim p_z(z)$ from a latent distribution to the data space:

$$G: z \mapsto \hat{x} = G(z)$$

The generator learns to transform simple random noise into samples that resemble the true data distribution. It has no access to real data; instead, it receives gradient feedback from the discriminator about whether its outputs are convincing.

**Typical architecture:**
- Input: latent vector $z$ of dimension $d_z$ (often 50-1000)
- Series of fully-connected or transposed convolutional layers
- Output: synthetic sample $\hat{x}$ matching the shape of real data

### Discriminator (D)

The discriminator $D$ is a classifier that attempts to distinguish real data from generator-produced fakes:

$$D(x) \in [0, 1] \text{ (probability that } x \text{ is real)}$$

The discriminator receives both real samples from the data distribution and fake samples from the generator, learning to output high values for real data and low values for fake data.

**Typical architecture:**
- Input: data sample $x$ (real or fake)
- Series of convolutional or fully-connected layers
- Output: scalar probability (sigmoid activation) indicating "realness"

## The Minimax Game and Value Function

GANs employ a minimax game-theoretic formulation. The discriminator and generator have opposing objectives, formalized by the value function:

$$V(G, D) = \mathbb{E}_{x \sim p_{data}} [\log D(x)] + \mathbb{E}_{z \sim p_z} [\log(1 - D(G(z)))]$$

The training objective is:

$$\min_G \max_D V(G, D)$$

### Game Dynamics

**Discriminator's Objective:** Maximize $V(G, D)$
- Maximize $\log D(x)$ for real data (output close to 1)
- Maximize $\log(1 - D(G(z)))$ for fake data (output close to 0)
- Essentially minimizes binary cross-entropy loss for a classification task

**Generator's Objective:** Minimize $V(G, D)$
- Minimize the discriminator's ability to distinguish fakes
- Equivalently, maximize $\log D(G(z))$ (fool the discriminator into thinking fakes are real)
- Receives no explicit supervision about the data, only adversarial feedback

!!! note "The Non-Saturating Trick"
    In practice, minimizing $\log(1 - D(G(z)))$ can lead to vanishing gradients when the discriminator is confident. Instead, generators typically maximize $\log D(G(z))$, which provides stronger gradients early in training while maintaining the same equilibrium point.

## Training Dynamics and Nash Equilibrium

In game theory, a Nash equilibrium occurs when neither player can improve their objective by unilaterally changing their strategy.

### Theoretical Equilibrium

At Nash equilibrium with infinite capacity networks:
- The generator learns the true data distribution: $p_g = p_{data}$
- The discriminator cannot distinguish real from fake: $D^*(x) = 1/2$ for all $x$
- Both players' objectives are stable

This can be proven by analyzing when $\max_D V(G, D) = 2 \log 2 - 2JS(p_{data} || p_g)$, where $JS$ is the Jensen-Shannon divergence. The minimum occurs at $p_g = p_{data}$.

### Practical Training Reality

In practice, achieving equilibrium is challenging due to:

1. **Limited capacity** - Networks have finite parameters
2. **Non-convex optimization** - Multiple local minima exist
3. **Alternating updates** - Players don't update simultaneously
4. **Imbalanced learning rates** - One player may dominate the other
5. **Non-stationary objectives** - Each player changes as the other adapts

Training typically involves alternating updates:
- Update discriminator for $k$ steps: $D \leftarrow \arg\max_D V(G, D)$
- Update generator for 1 step: $G \leftarrow \arg\min_G V(G, D)$

## Mode Collapse and Training Instabilities

### Mode Collapse

**Definition:** The generator learns to produce only a limited subset of the data distribution's modes, ignoring diversity in real data.

For example, a GAN trained on MNIST might only generate images of the digit "3", or a financial GAN might only generate certain market regimes.

**Causes:**
- Generator finds a shortcut that fools the current discriminator without capturing full diversity
- Insufficient discriminator capacity to distinguish all modes
- Gradient flow issues preventing exploration

**Consequences:**
- Lacks generative diversity
- Poor coverage of the data distribution
- Unrealistic and limited synthetic samples

### Other Training Instabilities

1. **Vanishing Gradients** - Discriminator becomes too confident, providing near-zero gradients to generator
2. **Exploding Gradients** - Rapid changes in loss landscapes
3. **Oscillation** - Loss values fluctuate rather than converge
4. **Discriminator Overfitting** - D learns spurious patterns not representative of true distribution vs generator data

!!! warning "Practical Training Challenges"
    GAN training is notoriously unstable. Practitioners typically employ:
    - Batch normalization in both networks
    - Spectral normalization in discriminator
    - Gradient penalties (Wasserstein distance)
    - Careful learning rate scheduling
    - Multiple discriminator updates per generator update
    - Monitoring of evaluation metrics (Inception Score, FID)

## Key Variants and Improvements

### DCGAN (Deep Convolutional GAN)

Introduced stable architectural guidelines for convolutional GANs:
- Replace pooling with strided convolutions
- Use batch normalization (except discriminator output)
- Use ReLU in generator (except output, which uses tanh)
- Use LeakyReLU in discriminator
- Systematic approach to hyperparameter selection

**Impact:** Enabled high-quality image generation and became de facto standard architecture.

### Wasserstein GAN (WGAN)

Replaces binary cross-entropy loss with Wasserstein distance (Earth-Mover distance):

$$V(G, D) = \mathbb{E}_{x \sim p_{data}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]$$

**Advantages:**
- Smoother loss landscape
- Meaningful loss value even when discriminator is perfect
- Better training stability
- Reduces mode collapse

**Constraint:** Discriminator weights must satisfy Lipschitz constraint (enforced via weight clipping or gradient penalty)

### StyleGAN

Combines adaptive instance normalization with style-based architecture:
- Separates low-level features (style) from high-level structure
- Progressive training from low to high resolution
- Coarse-to-fine generation process

**Key innovations:**
- Style modulation in generator
- Mixing regularization to enforce disentanglement
- Achieves unprecedented visual quality in image synthesis

## Applications in Finance

GANs have promising applications in quantitative finance:

### Synthetic Data Generation for Backtesting

Generate realistic historical price paths and market scenarios:
- Create synthetic OHLC (Open-High-Low-Close) data maintaining statistical properties
- Extend backtesting datasets without overfitting to limited history
- Evaluate trading strategies across diverse market conditions

**Benefits:** More robust strategy evaluation with larger effective dataset sizes

### Data Augmentation for Rare Events

Market crises, flash crashes, and extreme events are underrepresented in data:
- Train GANs on historical normal and rare regime data
- Generate additional examples of rare market events
- Augment training data for risk models and anomaly detectors

**Example:** Generate synthetic stress testing scenarios for portfolio VaR models

### Scenario Generation

Financial institutions need diverse plausible future scenarios:
- Generate multivariate return scenarios respecting correlations
- Create interest rate curve evolutions for derivative pricing
- Model order book dynamics and microstructure

**Advantages:**
- Captures complex dependencies without explicit parametric assumptions
- Produces realistic scenarios within the learned distribution
- Enables stress testing and scenario analysis

!!! info "Practical Considerations"
    When applying GANs to finance:
    - Ensure synthetic data maintains statistical properties (moments, autocorrelation, correlations)
    - Validate that generated scenarios respect market conventions and constraints
    - Use evaluation metrics suited to financial data (e.g., Kolmogorov-Smirnov test, distributional distance)
    - Be cautious about extrapolation beyond training regimes
    - Combine with domain expertise rather than relying solely on generative capability

## Further Exploration

The GAN literature is extensive and rapidly evolving. Key directions include:
- Conditional GANs (cGAN) for controlled generation
- Class-conditional variants for regime-specific synthesis
- Information-theoretic improvements reducing mode collapse
- Integration with reinforcement learning for policy optimization
- Applications to multi-asset and multi-timeframe scenarios
