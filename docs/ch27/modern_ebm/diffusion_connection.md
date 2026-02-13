# Connection Between EBMs and Diffusion Models

## Introduction

Recent advances in generative modeling have established deep connections between Energy-Based Models and diffusion models, revealing that seemingly different approaches share fundamental mathematical structures. Diffusion models generate data by reversing a stochastic process that gradually adds noise, while EBMs model energy landscapes directly. However, both frameworks can be unified through the lens of score-based generative models, where the score function (gradient of log density) is the central object.

Understanding these connections illuminates why diffusion models succeed where older EBM methods struggled—diffusion processes provide efficient algorithms for learning and sampling from EBMs. Conversely, EBM theory provides principled interpretation of diffusion model objectives, revealing that diffusion training can be viewed as implicit EBM learning. This symbiotic relationship enables hybrid approaches combining benefits of both frameworks.

This section develops the mathematical foundations connecting EBMs and diffusion models, explores score-based unification, and demonstrates practical implications for financial applications.

## Key Concepts

### Score Functions
- **Score**: $\nabla_x \log p(x)$ - gradient of log probability density
- **Score Matching**: Learning score function from data without explicit density
- **Denoising**: Predicting clean data from noisy version (equivalent to score estimation)
- **Unification**: EBMs and diffusion models both learn implicit score functions

### Noise Addition as Path
- **Forward Process**: Gradually add noise: $q(x_t | x_0)$ for t ∈ [0, T]
- **Reverse Process**: Gradually remove noise via learned score: $p(x_t | x_{t+1})$
- **Continuous Limit**: Forward diffusion and reverse can be expressed as SDEs
- **EBM Connection**: Reverse process is Langevin dynamics guided by EBM energy

## Mathematical Framework

### Score Function and Density

For probability density p(x), the score function:

$$s(x) = \nabla_x \log p(x) = \frac{\nabla_x p(x)}{p(x)}$$

Score function provides gradient information for Langevin sampling:

$$x^{(t+1)} = x^{(t)} + \frac{\eta}{2}\nabla_x \log p(x^{(t)}) + \sqrt{\eta} \xi_t = x^{(t)} + \frac{\eta}{2}s(x^{(t)}) + \sqrt{\eta} \xi_t$$

### Energy Function and Score

EBM energy E(x) relates to score:

$$p(x) = \frac{\exp(-E(x))}{Z}$$

$$s(x) = \nabla_x \log p(x) = -\nabla_x E(x)$$

**Key insight**: EBM score function is negative gradient of energy.

### Score Matching Objective

Learning score directly from data without computing partition function:

$$\mathcal{L}_{\text{SM}} = \int p(x) \|s_\theta(x) - \nabla_x \log p(x)\|^2 dx$$

Equivalent to:

$$\mathcal{L}_{\text{SM}} = \mathbb{E}_{x \sim p}[\|\nabla_x s_\theta(x)\|_F^2 + 2 \text{tr}(\nabla_x s_\theta(x))]$$

Avoids partition function computation; trains score without explicit density.

## Diffusion Models and Score Functions

### Forward Diffusion Process (Variance Exploding)

Gradually add noise to data through stochastic differential equation:

$$dx = f(x, t) dt + g(t) dw$$

With proper scaling, marginal density $q(x_t)$ at time t. For Ornstein-Uhlenbeck process:

$$q(x_t | x_0) = \mathcal{N}(x_t | \sqrt{1-\beta_t} x_0, \beta_t I)$$

### Reverse Diffusion (Score-Based Generation)

Reverse process learned via score:

$$dx = [f(x,t) - g(t)^2 \nabla_x \log q(x_t)] dt + g(t) d\bar{w}$$

The term $\nabla_x \log q(x_t)$ is the score; in practice, replaced with learned approximation $s_\theta(x_t, t)$.

### Connection to Denoising

Denoising objective equivalent to score matching:

$$\mathcal{L}_{\text{denoise}} = \mathbb{E}_{x_0, \epsilon}[\|\epsilon_\theta(x_t, t) - \epsilon\|^2]$$

where $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$ and $\epsilon_\theta$ predicts noise.

Denoising score $s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar{\alpha}_t}}$ produces same training signal as score matching.

## Unified Score-Based Framework

### General Score-Based Generative Model

Both EBMs and diffusion models can be expressed as:

1. **Learn Score**: Train network to approximate $\nabla_x \log p(x)$
2. **Sample via Langevin**: Use learned score for gradient-based sampling
3. **Generation**: Initialize from noise; follow score toward data manifold

### EBM as Implicit Diffusion

EBM in Langevin sampling context:

$$x^{(t+1)} = x^{(t)} - \frac{\eta}{2}\nabla_x E(x^{(t)}) + \sqrt{\eta} \xi_t$$

With learned energy E(x; θ), this is Langevin dynamics with score $s(x) = -\nabla_x E(x)$.

### Diffusion as Score-Based EBM

Diffusion model in reverse process context:

$$x_t = x_{t+1} + \frac{\eta}{2}[f(x_{t+1}, t) - g(t)^2 s_\theta(x_{t+1}, t)] dt + \sqrt{\eta} d\bar{w}$$

This resembles Langevin dynamics with implicit energy governing reverse process.

## Training Connections

### Contrastive Divergence vs Score Matching

Contrastive Divergence (EBM training):

$$\nabla_\theta \mathcal{L} \propto \mathbb{E}_{\text{data}}[\nabla_\theta E] - \mathbb{E}_{\text{model}}[\nabla_\theta E]$$

Score Matching (Diffusion training):

$$\nabla_\theta \mathcal{L} \propto \mathbb{E}_{\text{data}}[\nabla_\theta s_\theta - \nabla_x^2 s_\theta]$$

Different objectives, but both implicitly learn to increase likelihood; CD is contrastive while SM is regression-based.

### Advantage of Score Matching

Score matching advantages over CD:

1. **No Sampling Required**: Doesn't need negative phase sampling (expensive for EBMs)
2. **Stability**: Regression to explicit denoising target (not contrastive)
3. **Scalability**: Scales better to high dimensions than Langevin sampling in CD

### Why Diffusion Succeeds

Diffusion models implicitly combine advantages:

$$\mathcal{L}_{\text{diffusion}} = \sum_{t=1}^{T} \mathbb{E}_{x_0, \epsilon}[\|\epsilon_\theta(x_t, t) - \epsilon\|^2]$$

By adding noise during training:

- Makes targets easier (denoising is easier than density estimation)
- Avoids high-probability regions where score estimation hard
- Creates curriculum of noise levels aiding optimization
- Enables efficient score estimation across full input space

## Practical Implications

### EBM Improvements via Score Matching

Train EBM using score matching instead of CD:

1. Add noise during training: $x' = x + \sigma \xi$
2. Train network to predict $\nabla_x \log p_\sigma(x)$
3. Use trained score for Langevin sampling at inference

Dramatically improves sample quality while reducing training cost.

### Diffusion-Guided EBM Sampling

Improve EBM sampling by combining with diffusion:

1. Train diffusion model on data
2. Warm-start Langevin sampling with diffusion trajectory
3. Refine with EBM energy for additional guidance

Hybrid approach: fast diffusion initialization + precise EBM refinement.

### Joint EBM-Diffusion Learning

Train both objectives jointly:

$$\mathcal{L} = \mathcal{L}_{\text{EBM}}(\nabla_x E) + \mathcal{L}_{\text{diffusion}}(s_\theta)$$

with constraint that $s_\theta \approx -\nabla_x E$. Both objectives guide learning; EBM provides theoretical grounding while diffusion provides computational efficiency.

## Application to Financial Modeling

### Score-Based Financial Density Estimation

Learn score of return distribution:

$$s(r) = \nabla_r \log p(r)$$

via denoising regression on noisy returns. Score enables:

- Langevin sampling for return forecasts
- Gradient flow toward likely future states
- Conditional generation given constraints (e.g., VaR limits)

### Risk Factor Dynamics

Model factor dynamics as reverse diffusion:

$$df = \mu(f, t) dt + \sigma(t) dw$$

where $\mu$ estimated from learned score of factor distribution. Enables forecasting under changing market regimes.

### Market Microstructure Generation

Generate synthetic order books via:

1. Train diffusion on historical order book snapshots
2. Learn score of microstructure distribution
3. Sample via reverse diffusion with EBM energy guiding toward realistic configurations

## Theoretical Insights

### Information-Theoretic Connection

Both approaches minimize information distance between data and model:

**EBM (via CD)**: Minimizes $\text{KL}(p_{\text{data}} \| p_\theta)$ (contrastively)

**Diffusion (via Score)**: Minimizes Fisher divergence $\mathbb{E}_{p}[\|s_\theta - s_*\|^2]$

Fisher divergence related to KL via Itô calculus; both optimize divergences between distributions.

### Sample Complexity

Diffusion models empirically require fewer samples for good density estimation than explicit EBM training. Score matching more sample-efficient than CD due to regression target being explicit rather than implicit.

!!! note "EBM-Diffusion Synergy"
    EBMs and diffusion models represent complementary approaches to generative modeling. Diffusion provides computational efficiency and scalability; EBMs provide principled density framework. Modern practice combines both: use diffusion training (score matching) with EBM-based refinement or guidance, achieving best of both worlds.

