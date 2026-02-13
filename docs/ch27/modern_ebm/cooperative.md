# Cooperative Learning Between EBMs and Other Models

## Introduction

Energy-Based Models, while powerful for density estimation and anomaly detection, have limitations including computational cost of sampling and difficulty training on high-dimensional data. Modern deep learning advances suggest that EBMs should not be viewed as standalone models but rather as components within larger heterogeneous systems that leverage complementary strengths of other architectures.

Cooperative learning frameworks combine EBMs with autoencoders, VAEs, generative adversarial networks, and discriminative models, enabling each component to address the other's limitations. EBMs can provide principled density modeling to regularize generative models; autoencoders can provide efficient dimensionality reduction before EBM training; VAEs can provide fast approximate sampling to initialize expensive EBM sampling procedures; discriminative models can focus EBM training on decision boundaries.

This section explores cooperative learning paradigms, develops joint training procedures, and demonstrates how hybrid systems achieve improved performance compared to individual methods.

## Key Concepts

### Cooperative Learning Paradigm
- **Complementary Objectives**: Different models optimize different loss functions
- **Knowledge Transfer**: Information flows between models through shared representations
- **Ensemble Robustness**: Hybrid systems more robust than individual components
- **Improved Efficiency**: Each model compensates for others' computational limitations

### Integration Patterns
- **Sequential**: Train models sequentially; output of one feeds into next
- **Parallel**: Train models simultaneously on shared data with interaction loss
- **Hierarchical**: Cascade of models from high-dimensional to low-dimensional space

## Mathematical Framework

### Joint Training Objective

Train EBM with auxiliary model jointly on data D:

$$\mathcal{L}_{\text{joint}} = \mathcal{L}_{\text{EBM}} + \mathcal{L}_{\text{aux}} + \lambda \mathcal{L}_{\text{interaction}}$$

where:

$$\mathcal{L}_{\text{EBM}} = -\mathbb{E}_{r \sim p_{\text{data}}}[\log p(r; \theta_e)]$$

$$\mathcal{L}_{\text{aux}} = \text{Task-specific loss for auxiliary model}$$

$$\mathcal{L}_{\text{interaction}} = \text{Coupling loss encouraging coherence}$$

### Shared Representation Learning

Both models operate on shared latent representation z:

$$z = f_{\text{enc}}(x)$$

$$\mathcal{L}_{\text{EBM}} = -\log p(z; \theta_e)$$

$$\mathcal{L}_{\text{aux}} = \text{Reconstruction or classification on } z$$

Shared representation enables efficient joint training and prevents divergence.

### Interaction Loss for Coherence

Encourage models to agree on important aspects:

$$\mathcal{L}_{\text{interaction}} = \text{KL}(p_{\text{EBM}}(z) \| p_{\text{aux}}(z))$$

forces EBM density to match auxiliary model's implicit density estimate, ensuring consistency.

## EBM + Autoencoder Cooperation

### Architecture

1. **Autoencoder**: Compresses high-dimensional observations to lower-dimensional latent space
2. **EBM**: Trained on latent representation; models p(z; θ_e)
3. **Joint Training**: AE reconstruction loss + EBM likelihood loss + KL regularization

### Joint Loss Function

$$\mathcal{L} = \|x - \text{AE}_{\text{dec}}(\text{AE}_{\text{enc}}(x))\|^2 + \lambda_1 (-\log p(z; \theta_e)) + \lambda_2 \text{KL}(q_\phi(z|x) \| p(z; \theta_e))$$

### Benefits

- **Scalability**: AE reduces dimensionality before EBM training (avoids curse of dimensionality)
- **Efficiency**: AE provides fast reconstruction; EBM provides density modeling
- **Regularization**: EBM density acts as prior on AE latent space
- **Generation**: Sample from EBM in latent space; decode via AE

### Application: Financial Data

For high-dimensional market microstructure data:

1. **Train AE**: Compress 100+ order book features to 10-20 latent dimensions
2. **Train EBM**: Learn density of latent representation
3. **Hybrid Generation**: Sample from EBM; decode to synthetic order books
4. **Anomaly Detection**: High latent energy indicates unusual microstructure

## EBM + VAE Cooperation

### Cooperative Architecture

1. **VAE**: Provides probabilistic latent space with tractable p(z)
2. **EBM**: Refines latent density; models p(z; θ_e) to capture non-Gaussian aspects
3. **Shared Decoder**: Both use same decoder; compete/cooperate on latent representation

### Joint Training Objective

$$\mathcal{L} = \mathbb{E}_{q_\phi}[\log p_\psi(x|z)] - \beta \text{KL}(q_\phi(z|x) \| p(z))$$
$$+ \lambda_1 \log p_{\text{EBM}}(z; \theta_e) + \lambda_2 \text{correlation\_regularization}$$

### Benefits

- **Flexible Latent Density**: VAE provides baseline; EBM refines for non-Gaussianity
- **Stable Training**: VAE provides stable gradient flow; EBM focuses on tail behavior
- **Improved Sampling**: Combine fast VAE sampling with EBM refinement
- **Better Likelihood**: Joint model achieves higher likelihood than individual components

### Information Flow

- **VAE → EBM**: VAE learns efficient latent representation; EBM models refined density
- **EBM → VAE**: EBM gradients flow back to improve VAE latent space
- **Mutual Benefit**: Neither component dominates; both contribute to joint objective

## EBM + GAN Cooperation

### Adversarial Cooperation

Traditional setup: Discriminator (D) learns p(real vs generated). Cooperative setup:

1. **Generator (G)**: Produces samples
2. **EBM**: Models p_data; provides density estimation
3. **Discriminator**: Assists both G and EBM through adversarial training

### Joint Training

$$\mathcal{L}_G = -\mathbb{E}_{z \sim p_z}[\log p(E(G(z)))]$$
$$\mathcal{L}_D = -\mathbb{E}_{x \sim p_{\text{data}}}[\log p(E(x))] - \mathbb{E}_{x \sim p_G}[\log(1 - p(E(x)))]$$
$$\mathcal{L}_E = \mathcal{L}_{\text{CD}}(E)$$

where E is EBM energy, D is discriminator.

### Benefits

- **Implicit and Explicit Models**: GAN provides implicit generation; EBM provides explicit density
- **Stabilized Training**: EBM provides likelihood signal to guide generation
- **Improved Diversity**: EBM prevents mode collapse through density modeling
- **Multiple Objectives**: GAN loss and likelihood loss prevent overfit to single objective

## EBM + Discriminative Model Cooperation

### Decision Boundary Focusing

Train discriminative classifier (e.g., neural network) to distinguish normal vs anomalous:

1. **Discriminative Model**: Learns p(y|x) via supervised learning on labeled data
2. **EBM**: Models p(x | y=normal) unsupervised on normal data
3. **Cooperation**: EBM focuses on regions near decision boundary; discriminator provides labels

### Joint Objective

$$\mathcal{L} = -\sum_n \log p(y_n | x_n; \theta_d) + \lambda_1 \log p(x_n | y_n=\text{normal}; \theta_e)$$

where first term is supervised classification loss; second term regularizes with EBM density.

### Benefits

- **Semi-Supervised Learning**: Leverage both labeled (discriminative) and unlabeled (EBM) data
- **Interpretability**: Discriminative decision boundary + EBM density reveals normal manifold
- **Robustness**: Classification regularized by density model; misclassified points have high EBM energy
- **Few-Shot Learning**: EBM density enables reliable predictions with few labels

## Practical Cooperative Implementations

### Training Procedure

1. **Initialize Models**: Train individual models separately (faster convergence)
2. **Joint Fine-Tuning**: Lock component weights; train coupling loss
3. **Simultaneous Training**: Enable all parameters; continue joint training
4. **Convergence Check**: Monitor separate and joint objectives; stop when both plateau

### Hyperparameter Coordination

Balance component losses through weighting:

$$\mathcal{L}_{\text{total}} = w_1 \mathcal{L}_1 + w_2 \mathcal{L}_2 + w_3 \mathcal{L}_3$$

Typical weights:
- $w_1 = 1.0$ (primary loss)
- $w_2 = 0.5-1.0$ (auxiliary loss)
- $w_3 = 0.1-0.5$ (interaction loss)

### Computational Efficiency

Cooperative systems balance computational cost:

| Component | Time | Output |
|-----------|------|--------|
| AE Encode | Fast | Latent z |
| EBM Forward | Slow | Energy E(z) |
| AE Decode | Fast | Reconstruction |

Total: ~2x single model (dominated by EBM sampling).

## Hybrid Model Evaluation

### Unified Evaluation Metrics

Compare cooperative system to individual components:

1. **Likelihood on Test Data**: EBM log-likelihood
2. **Reconstruction Error**: AE reconstruction MSE
3. **Classification Accuracy**: Anomaly detection performance
4. **Computational Cost**: Wallclock time for inference

### Component Contribution Analysis

Ablate individual components to measure contribution:

$$\Delta_{\text{performance}} = \text{Performance}_{\text{full}} - \text{Performance}_{\text{without component}}$$

Positive Δ indicates component improves performance.

## Applications to Finance

### Cooperative Anomaly Detection System

Combine:
1. **AE**: Compress market microstructure to latent factors
2. **EBM**: Model normal latent distribution
3. **LSTM**: Track temporal dependencies

Anomaly = high EBM energy OR high LSTM error OR AE reconstruction error.

### Cooperative Risk Forecasting

Combine:
1. **VAE**: Learn return distribution
2. **EBM**: Refine tail behavior
3. **GARCH**: Model conditional volatility

Predictive density: $p(r_{t+1} | r_t) = w_1 p_{\text{VAE}} + w_2 p_{\text{EBM}} + w_3 p_{\text{GARCH}}$

!!! note "Cooperative Design Principles"
    Successful cooperative systems pair models that excel at different aspects: one efficient but approximate (VAE, GAN); one principled but expensive (EBM). Coupling losses encourage coherent representations. Start with sequential training; upgrade to joint training if needed.

