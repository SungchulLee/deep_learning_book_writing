# Energy-Based Models for Generation Tasks

## Introduction

While traditionally used for density estimation and anomaly detection, Energy-Based Models can be adapted for generation—producing new samples from learned distributions. Unlike explicit generative models (VAEs, GANs) that directly map from noise to data, EBMs generate by learning energy landscapes and sampling via dynamics (Langevin, Hamiltonian). Though computationally more expensive than alternatives, EBM generation provides unique advantages: principled probabilistic framework, flexible architecture, and ability to combine with discriminative losses for improved mode coverage.

In quantitative finance, EBM generation enables synthetic data augmentation for training robust models, stress-testing portfolios through generated market scenarios, and creating realistic market microstructure data for algorithm backtesting. The flexibility of energy functions allows incorporation of financial constraints (no negative prices, bid-ask spread bounds) directly into generation process.

This section develops EBM-based generation methods, explores sampling algorithms, addresses computational challenges, and demonstrates financial applications.

## Key Concepts

### Generation via Sampling
- **Energy Landscape**: E(x) defines probability p(x) ∝ exp(-E(x))
- **Sampling Algorithms**: Langevin, Hamiltonian, tempering methods
- **Mode Coverage**: Ability to generate from all probability modes
- **Computational Cost**: Expensive compared to feedforward generation

### Generation Objectives
- **Data Augmentation**: Generate realistic synthetic samples for training
- **Scenario Analysis**: Generate plausible market scenarios for risk testing
- **Missing Data**: Generate conditional samples given partial observations
- **Controlled Generation**: Generate samples satisfying specified constraints

## Mathematical Framework

### Langevin Sampling Algorithm

Generate samples from energy landscape via stochastic gradient descent:

$$x^{(t+1)} = x^{(t)} - \frac{\eta}{2}\nabla_x E(x^{(t)}; \theta) + \sqrt{\eta} \xi_t$$

where ξ_t ~ N(0, I) is injected noise. As t→∞, x^{(t)} converges to sample from:

$$p(x) = \frac{\exp(-E(x; \theta))}{Z}$$

### Hamiltonian Monte Carlo (HMC)

For faster mixing, use auxiliary momentum variable p:

$$x^{(t+1)} = x^{(t)} + \eta p^{(t)}$$

$$p^{(t+1)} = p^{(t)} - \frac{\eta}{2}\nabla_x E(x^{(t)}) - \frac{\eta}{2}\nabla_x E(x^{(t+1)}) + \eta \xi_t$$

HMC provides better exploration of energy landscape through momentum, reducing random walk behavior.

### Parallel Tempering

Improve mode coverage via ensemble of chains at different temperatures:

$$p_\beta(x) = \frac{\exp(-\beta E(x))}{Z_\beta}$$

with β ∈ [β_min, β_max]. High temperature (low β) chains explore freely; cold chains (high β) concentrate near modes. Periodically swap between chains to propagate information.

## Conditional Generation

### Conditional Energy Function

Generate samples x satisfying condition c via conditional energy:

$$E(x | c) = E(x) + E_{\text{constraint}}(x | c)$$

### Examples for Finance

**VaR-Constrained Generation**: Generate returns within VaR bounds:

$$E_{\text{constraint}}(r) = \lambda \cdot \text{Indicator}(r < -\text{VaR}_{95\%})$$

**Positive Price Generation**: Ensure generated prices positive:

$$E_{\text{constraint}}(p) = \lambda \cdot (p - p_{\text{min}})_-^2$$

where $(·)_-$ denotes negative part.

**Correlation-Matching Generation**: Generate portfolio returns preserving historical correlations:

$$E_{\text{constraint}}(r) = \gamma \|\text{Corr}(r_{\text{gen}}) - \text{Corr}(r_{\text{hist}})\|_F^2$$

### Guidance Through Score

Incorporate constraints via score function:

$$s_{\text{guided}}(x|c) = s(x) + \nabla_x E_{\text{constraint}}(x|c)$$

Langevin with guided score automatically moves toward constraint satisfaction.

## Comparison with Alternative Generation Methods

### EBM vs VAE Generation

| Aspect | EBM | VAE |
|--------|-----|-----|
| **Framework** | Unnormalized density | Normalized distribution |
| **Generation** | Iterative sampling | Single feedforward pass |
| **Speed** | Slow (100s-1000s steps) | Fast (one pass) |
| **Flexibility** | Very flexible (any energy) | Limited by architecture |
| **Constraints** | Easy to add | Difficult to incorporate |
| **Mode Coverage** | Good with tempering | May miss modes (posterior collapse) |

EBMs excel at constrained generation; VAEs excel at speed.

### EBM vs GAN Generation

| Aspect | EBM | GAN |
|--------|-----|-----|
| **Objective** | Density matching | Adversarial game |
| **Stability** | Stable training | Mode collapse risk |
| **Theory** | Principled | Less rigorous |
| **Sampling** | Iterative | Single pass |
| **Metrics** | Likelihood available | No likelihood |

EBMs more theoretically justified; GANs produce samples faster.

## Financial Applications

### Synthetic Market Data Generation

For testing trading algorithms without risking real capital:

1. **Train EBM** on historical OHLCV (Open, High, Low, Close, Volume) data
2. **Sample Paths** using Langevin dynamics
3. **Validation**: Ensure generated data preserves volatility clustering, skewness, kurtosis

Advantages:
- Realistic price dynamics
- Easy to add constraints (no negative prices)
- Conditional generation: "Generate paths given market regime X"

### Scenario-Based Risk Analysis

Generate plausible stress scenarios for portfolio testing:

$$E(r_{\text{portfolio}}) = E_{\text{normal}}(r) + E_{\text{regime}}(\text{high volatility})$$

Sample portfolio returns in stress regime; evaluate losses.

### Order Book Microstructure Synthesis

Generate synthetic order book snapshots preserving empirical characteristics:

1. **Features**: Spreads, depths, mid-price, volatility
2. **Energy**: Learned from real order book data
3. **Constraints**: No negative spreads, depth/price relationships
4. **Output**: Realistic synthetic order books for algorithm testing

### Missing Data Imputation

For portfolios with missing prices (infrequently traded assets):

$$p(x_{\text{missing}} | x_{\text{observed}}) = \frac{\exp(-E(x_{\text{missing}}, x_{\text{observed}}))}{\int \exp(-E(x', x_{\text{observed}})) dx'}$$

Langevin sampling fills missing values while preserving correlation structure.

## Practical Implementation

### Sampling Hyperparameters

Critical choices for Langevin dynamics:

1. **Step Size η**: Larger steps explore faster but lower acceptance. Typical: η ∈ [0.001, 0.01]
2. **Burn-In Iterations**: Discard early samples before mixing. Typical: 100-500 steps
3. **Sampling Iterations**: Number of samples needed for convergence. Typical: 500-5000
4. **Noise Schedule**: Start high noise → decrease for refinement

### Convergence Diagnostics

Assess sampling convergence via:

$$\text{Variance Ratio} = \frac{\text{Var}_{\text{between-chain}}(x^{(t)})}{\text{Var}_{\text{within-chain}}(x^{(t)})}$$

Ratio < 1.1 indicates convergence (Gelman-Rubin statistic).

### Computational Acceleration

Reduce sampling cost through:

1. **GPU Parallelization**: Run multiple chains in parallel
2. **Low-Rank Energy**: Approximate energy with smaller network
3. **Score-Based Initialization**: Use diffusion model to initialize chains
4. **Importance Weighting**: Weight samples; discard low-likelihood samples

## Assessing Generated Sample Quality

### Distributional Matching

Compare generated samples to real data on summary statistics:

$$\text{Error}_\mu = |\mu_{\text{gen}} - \mu_{\text{real}}|$$
$$\text{Error}_\sigma = |\sigma_{\text{gen}} - \sigma_{\text{real}}|$$
$$\text{Error}_\rho = \|\text{Corr}_{\text{gen}} - \text{Corr}_{\text{real}}\|_F$$

Generation successful if errors within acceptable bounds (typically < 5%).

### Likelihood Evaluation

If exact likelihood computable (rare), evaluate on generated samples:

$$\text{Likelihood} = \frac{1}{N_{\text{gen}}}\sum_n \log p(x_n^{\text{gen}})$$

Should be close to data likelihood, indicating faithful distribution.

### Domain-Specific Validation

For financial applications:

1. **Volatility Clustering**: Generated returns exhibit GARCH-like clustering
2. **Drawdown Statistics**: Drawdown distribution matches historical
3. **Trading Metrics**: Strategy performance on generated data similar to backtests
4. **Correlation Stability**: Portfolio-level correlations preserved

## Limitations and Mitigation

### Computational Cost

Langevin sampling expensive (requires 1000s of energy evaluations per sample).

**Mitigations**:
- Use tempering for better exploration
- Parallelize across GPUs
- Combine with fast generation methods (distillation into VAE)

### Mode Coverage

May miss low-probability modes not visited during sampling.

**Mitigations**:
- Parallel tempering to explore high-temperature modes
- Combination with explicit mode specification
- Hybrid: VAE for mode identification + EBM for refinement

### Non-Stationarity

Financial distributions change over time; EBM trained on historical data may fail on future data.

**Mitigations**:
- Regular retraining on recent data
- Regime-conditioned energy functions
- Ensemble of EBMs from different time periods

!!! warning "Generation Practicality"
    EBM generation powerful but computationally expensive. For most applications, hybrid approaches work best: use fast methods (VAE, GAN) for bulk generation; use EBM for controlled refinement or constrained generation. Pure EBM generation most appropriate when constraints essential or modes must be exhaustively explored.

