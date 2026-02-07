# Introduction to Normalizing Flows

## The Generative Modelling Problem

Given samples $\{x_1, x_2, \ldots, x_N\}$ drawn from an unknown distribution $p_{\text{data}}(x)$, a generative model pursues two intertwined goals:

1. **Density estimation** — learn $p_\theta(x) \approx p_{\text{data}}(x)$ so that the likelihood of any point can be evaluated.
2. **Sampling** — draw new points $x_{\text{new}} \sim p_\theta(x)$ that are statistically indistinguishable from the training data.

Different generative model families approach these goals with different trade-offs.  Autoregressive models provide exact likelihoods but sample sequentially.  GANs sample efficiently but offer no density.  VAEs optimise a lower bound rather than the exact likelihood.  Normalizing flows are unique in satisfying both requirements simultaneously: they yield **exact, tractable log-likelihoods** and generate samples in a **single forward pass**.

## Taxonomy of Generative Models

```
Generative Models
├── Explicit Density
│   ├── Tractable Density
│   │   ├── Autoregressive Models (PixelCNN, WaveNet)
│   │   └── Normalizing Flows  ← this chapter
│   └── Approximate Density
│       └── Variational Autoencoders (VAE)
└── Implicit Density
    └── Generative Adversarial Networks (GAN)
```

Normalizing flows sit in the *tractable explicit density* branch.  Every other family either approximates the likelihood or avoids computing it altogether.

## The Flow Idea

A normalizing flow transforms a **simple base distribution** $p_Z(z)$ (typically a standard Gaussian) into a **complex target distribution** $p_X(x)$ through a composition of invertible, differentiable maps:

$$z \sim p_Z(z) \quad\xrightarrow{\;f\;}\quad x = f(z) \sim p_X(x)$$

Because $f$ is a bijection, every data point $x$ maps to a unique latent code $z = f^{-1}(x)$, and the density of $x$ is given exactly by the change-of-variables formula:

$$p_X(x) = p_Z\!\bigl(f^{-1}(x)\bigr)\;\left|\det \frac{\partial f^{-1}}{\partial x}\right|$$

### Why "Normalizing Flow"?

The term *normalizing* reflects the fact that the inverse $f^{-1}$ maps complex data back to a "normal" (Gaussian) distribution.  The term *flow* comes from the interpretation of the composed transformations as a discrete-time flow of probability mass through the space.

## Advantages

### Exact Likelihood

Unlike VAEs (which maximise a lower bound) or GANs (which have no likelihood), flows compute the exact log-likelihood:

$$\log p(x) = \log p_Z(z) + \log\left|\det J\right|$$

This enables rigorous model comparison, proper scoring rules, and principled uncertainty quantification—all critical in quantitative finance.

### Efficient Sampling

Generating a new data point requires only a single forward pass through the network:

```python
def sample(self, n_samples):
    z = self.base_dist.sample(n_samples)   # sample N(0, I)
    x = self.forward(z)                     # one forward pass
    return x
```

Contrast this with diffusion models (hundreds of iterative denoising steps) or autoregressive models (one dimension at a time).

### Bijective Latent Space

Every data point has a unique, deterministic latent representation.  There is no stochastic encoder (as in VAEs) and no information loss:

- **Encoding**: $z = f^{-1}(x)$ — deterministic
- **Decoding**: $x = f(z)$ — deterministic
- **Exact reconstruction**: $f(f^{-1}(x)) = x$

This determinism enables meaningful interpolation and latent-space manipulation.

### Stable Training

Training optimises the log-likelihood directly by gradient descent—no adversarial dynamics (GAN), no KL-divergence balancing (VAE), no score-matching approximation (diffusion).

## Challenges and Constraints

### Architectural Restrictions

Transformations must be invertible with an efficiently computable Jacobian determinant.  Standard neural-network layers (ReLU, max-pooling, fully connected with different input/output dimensions) do not satisfy these constraints.  The entire field of flow architecture design is, at its core, the search for transformations that are expressive yet computationally tractable.

### Dimensionality Preservation

Because the map is a bijection $f: \mathbb{R}^d \to \mathbb{R}^d$, the input and output dimensions must match:

$$\dim(x) = \dim(z)$$

Unlike a VAE, whose latent space can be much lower-dimensional than the data, a flow cannot learn a compressed representation.  Multi-scale architectures partially address this by factoring out dimensions at intermediate scales, but the fundamental constraint remains.

### Expressiveness vs. Efficiency

More expressive transformations typically incur higher computational costs:

| Architecture | Expressiveness | Jacobian Cost |
|---|---|---|
| Planar / Radial | Low | $O(d)$ |
| Coupling (RealNVP, Glow) | Medium | $O(d)$ |
| Autoregressive (MAF) | High | $O(d)$ but sequential |
| Continuous (FFJORD) | Very High | $O(d)$ per ODE step |
| Neural Spline | High | $O(d)$ |

Choosing an architecture means navigating this trade-off in the context of the problem at hand.

## Finance Applications

Normalizing flows are particularly well-suited to quantitative finance:

- **Return distribution modelling** — learn non-Gaussian, heavy-tailed return distributions from data.
- **Risk measurement** — compute VaR and CVaR from the learned density, capturing tail risk that parametric models miss.
- **Option pricing** — learn risk-neutral densities directly from market prices, without assuming a specific stochastic model for the underlying.
- **Scenario generation** — generate realistic, multi-asset scenarios for stress testing that respect observed dependence structures.
- **Portfolio optimisation** — optimise allocations under the true learned return distribution rather than a Gaussian approximation.

These applications are developed in detail in Section 23.5.

## Key References

1. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.
2. Kobyzev, I., et al. (2020). Normalizing Flows: An Introduction and Review of Current Methods. *TPAMI*.
3. Rezende, D. J. & Mohamed, S. (2015). Variational Inference with Normalizing Flows. *ICML*.
