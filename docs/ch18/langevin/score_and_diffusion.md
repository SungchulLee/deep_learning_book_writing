# Score Matching, Langevin Dynamics, and Diffusion Models

## The Score Function: A Unifying Concept

The **score function** is the gradient of the log-density:
$$
s(x) = \nabla_x \log p(x)
$$

This seemingly simple quantity connects:
- Langevin dynamics (MCMC sampling)
- Score matching (density estimation)
- Diffusion models (generative modeling)

## Why the Score?

### Property 1: Score Determines the Distribution (Up to Constant)

If we know $s(x) = \nabla \log p(x)$ everywhere, we can recover $p$ up to normalization:

$$
p(x) \propto \exp\left(\int_0^x s(u) \cdot du\right)
$$

**Implication**: Learning the score is equivalent to learning the distribution!

### Property 2: Score Doesn't Require Normalization

Computing $\nabla \log p(x)$ doesn't need the partition function $Z$:

$$
\nabla \log p(x) = \nabla \log \frac{\tilde{p}(x)}{Z} = \nabla \log \tilde{p}(x) - \nabla \log Z = \nabla \log \tilde{p}(x)
$$

Since $Z$ is a constant, $\nabla \log Z = 0$.

**Implication**: Can work with unnormalized densities directly!

### Property 3: Score Determines Langevin Dynamics

The Langevin SDE is driven by the score:
$$
dx_t = s(x_t)\,dt + \sqrt{2}\,dW_t
$$

Sampling from $p$ reduces to following the score function with noise.

## Score Matching: Learning the Score from Data

**Problem**: Given samples $\{x_i\}_{i=1}^N \sim p_{\text{data}}$, learn $s(x) = \nabla \log p_{\text{data}}(x)$.

**Approach**: Train a neural network $s_\theta(x)$ to approximate the true score.

### Naive Approach: Minimize Score Difference

$$
\mathcal{L}(\theta) = \frac{1}{2}\mathbb{E}_{p_{\text{data}}}[\|s_\theta(x) - \nabla \log p_{\text{data}}(x)\|^2]
$$

**Problem**: We don't know $\nabla \log p_{\text{data}}(x)$ (that's what we're trying to learn)!

### Score Matching Objective (Hyvärinen 2005)

**Key identity**: For any smooth vector field $s_\theta(x)$:

$$
\frac{1}{2}\mathbb{E}_{p}[\|s_\theta(x) - \nabla \log p(x)\|^2] = \mathbb{E}_p[\text{tr}(\nabla s_\theta(x)) + \frac{1}{2}\|s_\theta(x)\|^2] + \text{const}
$$

The right side doesn't require knowing $p$ — only samples from it!

**Score matching objective**:
$$
\mathcal{L}_{\text{SM}}(\theta) = \mathbb{E}_{x \sim p_{\text{data}}}\left[\text{tr}(\nabla s_\theta(x)) + \frac{1}{2}\|s_\theta(x)\|^2\right]
$$

**Implementation**:
```python
def score_matching_loss(score_network, x):
    """
    x: batch of data samples, shape (batch, dim)
    score_network: neural net outputting score s_θ(x)
    """
    # Compute score
    s = score_network(x)  # shape: (batch, dim)
    
    # Compute divergence: tr(∇s_θ)
    # This requires computing Jacobian
    div_s = 0
    for i in range(x.shape[1]):
        div_s += torch.autograd.grad(
            s[:, i].sum(), x, create_graph=True
        )[0][:, i]
    
    # Score matching loss
    loss = (div_s + 0.5 * (s ** 2).sum(dim=1)).mean()
    return loss
```

### Denoising Score Matching (Vincent 2011)

**Idea**: Instead of matching score of $p_{\text{data}}$, match score of noised data.

**Noise model**: $q_\sigma(x|x_0) = \mathcal{N}(x | x_0, \sigma^2 I)$

**Noised distribution**: $q_\sigma(x) = \int p_{\text{data}}(x_0)q_\sigma(x|x_0)\,dx_0$

**Denoising score matching objective**:
$$
\mathcal{L}_{\text{DSM}}(\theta) = \mathbb{E}_{x_0 \sim p_{\text{data}}, x \sim q_\sigma(\cdot|x_0)}\left[\frac{1}{2}\|s_\theta(x, \sigma) - \nabla \log q_\sigma(x|x_0)\|^2\right]
$$

**Key insight**: We can compute $\nabla \log q_\sigma(x|x_0)$ exactly:
$$
\nabla \log q_\sigma(x|x_0) = -\frac{x - x_0}{\sigma^2}
$$

**Implementation**:
```python
def denoising_score_matching_loss(score_network, x_data, sigma):
    """
    x_data: clean data samples
    sigma: noise level
    """
    # Add noise
    noise = torch.randn_like(x_data) * sigma
    x_noisy = x_data + noise
    
    # Predicted score
    s_pred = score_network(x_noisy, sigma)
    
    # True score of noised data
    s_true = -noise / (sigma ** 2)
    
    # Denoising score matching loss
    loss = 0.5 * ((s_pred - s_true) ** 2).sum(dim=1).mean()
    return loss
```

**Advantages**:
- Simpler to implement (no divergence computation)
- More stable training
- Connects directly to diffusion models

## From Score Matching to Sampling: Langevin MCMC

Once we've learned $s_\theta(x) \approx \nabla \log p_{\text{data}}(x)$, we can sample:

**Langevin dynamics**:
$$
x_{t+1} = x_t + \epsilon s_\theta(x_t) + \sqrt{2\epsilon}\,\eta_t, \quad \eta_t \sim \mathcal{N}(0, I)
$$

Start from random noise, run this process → samples from $p_{\text{data}}$!

This is **score-based generative modeling**.

## Multi-Scale Score Matching

**Problem**: Score is poorly defined in low-density regions (where data is sparse).

**Solution**: Learn scores at multiple noise levels.

**Noised distributions at different scales**:
$$
q_{\sigma_i}(x) = \int p_{\text{data}}(x_0)\mathcal{N}(x|x_0, \sigma_i^2 I)\,dx_0
$$

with $\sigma_1 < \sigma_2 < \cdots < \sigma_L$.

**Train score network** $s_\theta(x, \sigma)$ to match scores at all noise levels:
$$
\mathcal{L}(\theta) = \sum_{i=1}^L \lambda_i \mathbb{E}_{x \sim q_{\sigma_i}}\left[\frac{1}{2}\|s_\theta(x, \sigma_i) - \nabla \log q_{\sigma_i}(x)\|^2\right]
$$

**Annealed Langevin dynamics** for sampling:
1. Start at highest noise level $\sigma_L$ (nearly uniform)
2. Run Langevin at $\sigma_L$ → sample from $q_{\sigma_L}$
3. Decrease noise: $\sigma_L \to \sigma_{L-1} \to \cdots \to \sigma_1$
4. End at lowest noise → sample from $p_{\text{data}}$

**This is exactly simulated annealing applied to sampling!**

## Connection to Diffusion Models

Diffusion models are multi-scale score matching with continuous noise schedule.

### Forward Process (Add Noise)

$$
x_t = \sqrt{\alpha_t}x_0 + \sqrt{1-\alpha_t}\,\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

where $\alpha_t$ decreases from 1 to 0 as $t$ goes from 0 to $T$.

**This is the same as the noise model** $q_\sigma(x|x_0)$ with $\sigma^2 = 1 - \alpha_t$.

### Reverse Process (Denoise)

The reverse process removes noise guided by the score:
$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t + (1-\alpha_t)s_\theta(x_t, t)\right) + \sqrt{1-\alpha_t}\,\eta
$$

**This is Langevin dynamics** with time-dependent noise!

### The Score Function in Diffusion Models

The "noise prediction network" $\epsilon_\theta(x_t, t)$ in diffusion models is related to the score:

$$
s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\alpha_t}}
$$

**Training objective** (simplified):
$$
\mathcal{L}(\theta) = \mathbb{E}_{x_0, t, \epsilon}\left[\|\epsilon_\theta(x_t, t) - \epsilon\|^2\right]
$$

where $x_t = \sqrt{\alpha_t}x_0 + \sqrt{1-\alpha_t}\epsilon$.

**This is denoising score matching** with continuous time!

## The Complete Picture

```
Score Function s(x) = ∇log p(x)
           |
           |--- [Learning] ---> Score Matching
           |                     |
           |                     |--- Denoising Score Matching
           |                     |       |
           |                     |       |--- Multi-Scale
           |                     |               |
           |                     |               |--- Diffusion Models
           |
           |--- [Sampling] ---> Langevin Dynamics
                                 |
                                 |--- Annealed Langevin
                                 |       |
                                 |       |--- Reverse Diffusion
                                 |
                                 |--- SGLD (Stochastic Gradient LD)
```

## Detailed Comparison

| Concept | MCMC View | Generative Modeling View |
|---------|-----------|-------------------------|
| **Goal** | Sample from $\pi(x)$ | Generate samples from $p_{\text{data}}$ |
| **Score** | $\nabla \log \pi(x)$ | $\nabla \log p_{\text{data}}(x)$ |
| **Known?** | Yes (posterior from Bayes) | No (learned from data) |
| **Method** | Langevin dynamics | Score-based generative model |
| **Annealing** | Simulated annealing | Multi-scale / diffusion |
| **Temperature** | Controls exploration | Noise level $\sigma$ |

**Key insight**: Same mathematics, different perspective!

## Score-Based Models vs Other Generative Models

### GANs (Generative Adversarial Networks)

**Pros**:
- Fast sampling (one forward pass)
- High-quality samples

**Cons**:
- Unstable training (adversarial)
- Mode collapse
- No density estimation

**Score-based**:
- Stable training (regression, not adversarial)
- Covers all modes (if training is thorough)
- Slower sampling (iterative process)

### VAEs (Variational Autoencoders)

**Pros**:
- Fast sampling
- Encoder available (for inference)

**Cons**:
- Blurry samples (compared to GANs/diffusion)
- Approximate inference (variational)

**Score-based**:
- Higher quality samples
- No encoder (pure generative)
- Exact sampling (in limit)

### Normalizing Flows

**Pros**:
- Exact likelihood
- Exact sampling (invertible)

**Cons**:
- Architecture constraints (invertibility)
- Difficult to scale

**Score-based**:
- Flexible architectures
- Approximate but high-quality
- Scales to high dimensions

## Practical Considerations

### Noise Schedule Design

**For score matching**, choose $\{\sigma_i\}$:
- **Geometric progression**: $\sigma_i = \sigma_{\max} \cdot r^i$, $r < 1$
- Typical: $\sigma_{\max} \approx \text{std}(\text{data})$, $\sigma_{\min} \approx 0.01$

**For diffusion models**, choose $\{\alpha_t\}$:
- **Linear schedule**: $\beta_t = \text{const}$
- **Cosine schedule**: $\alpha_t = \cos^2(\pi t / 2)$
- Modern: learned schedules

### Number of Steps

**Langevin MCMC**: Typically 100-1000 steps per noise level

**Annealed Langevin**: 10-50 noise levels × 100-1000 steps = 1000-50,000 total

**Diffusion models**: 
- Training: 1000 time steps
- Sampling: Can use 50-100 steps (with DDIM or DDPM)

### Computational Cost

**Score network evaluation**:
- Typically a U-Net or Transformer
- GPU required for large models
- Bottleneck for sampling

**Sampling cost**:
- Score-based: $N_{\text{steps}}$ × (score network eval)
- Much slower than GANs or VAEs
- But improving (fast samplers, distillation)

## Advanced Topics

### Stochastic Gradient Langevin Dynamics (SGLD)

Apply Langevin to Bayesian neural networks:
$$
\theta_{t+1} = \theta_t + \frac{\epsilon}{2}\nabla \log p(\theta | \mathcal{D}) + \sqrt{\epsilon}\,\eta_t
$$

Using minibatch gradients (stochastic).

**Connection**: Score-based sampling + mini-batch = SGLD.

### Score-Based SDEs

Generalize to continuous time:
$$
dx = f(x,t)\,dt + g(t)\,dW
$$

where $f$ is learned from data.

**Probability flow ODE**: Deterministic version
$$
dx = \left[f(x,t) - \frac{1}{2}g(t)^2 s(x,t)\right]dt
$$

Allows exact likelihood computation!

### Conditional Generation

Learn conditional score: $s_\theta(x, y, t)$

Sample $x | y$ via:
$$
x_{t-1} = x_t + \epsilon s_\theta(x_t, y, t) + \sqrt{2\epsilon}\,\eta
$$

Applications:
- Image inpainting: $y$ = masked image
- Super-resolution: $y$ = low-res image
- Text-to-image: $y$ = text embedding

### Classifier Guidance

Combine unconditional score with classifier gradient:
$$
s_\theta^{\text{guided}}(x, y, t) = s_\theta(x, t) + w \cdot \nabla_x \log p(y|x_t)
$$

where $w$ is guidance strength.

**Effect**: Better quality at cost of diversity.

## Code Example: Complete Pipeline

```python
import torch
import torch.nn as nn

class ScoreNetwork(nn.Module):
    """Simple MLP score network"""
    def __init__(self, data_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1, hidden_dim),  # +1 for sigma
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim)
        )
    
    def forward(self, x, sigma):
        # Concatenate data and noise level
        sigma_vec = sigma * torch.ones(x.shape[0], 1, device=x.device)
        inp = torch.cat([x, sigma_vec], dim=1)
        return self.net(inp)

def train_score_model(data, sigmas, epochs=1000):
    """Train score network via denoising score matching"""
    model = ScoreNetwork(data.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        # Sample noise level
        sigma = sigmas[torch.randint(len(sigmas), (1,))]
        
        # Add noise to data
        noise = torch.randn_like(data) * sigma
        x_noisy = data + noise
        
        # Predict score
        score_pred = model(x_noisy, sigma)
        
        # True score
        score_true = -noise / (sigma ** 2)
        
        # Loss
        loss = 0.5 * ((score_pred - score_true) ** 2).sum(dim=1).mean()
        
        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    return model

def sample_annealed_langevin(model, sigmas, n_steps=100):
    """Sample via annealed Langevin dynamics"""
    x = torch.randn(100, data_dim) * sigmas[0]  # Start from noise
    
    for sigma in sigmas:
        epsilon = 0.1 * (sigma / sigmas[-1]) ** 2
        
        for _ in range(n_steps):
            # Langevin step
            score = model(x, torch.tensor(sigma))
            x = x + epsilon * score + torch.sqrt(2 * epsilon) * torch.randn_like(x)
    
    return x

# Usage
data = load_data()  # Your data
sigmas = torch.logspace(-2, 1, 10)  # Geometric sequence

model = train_score_model(data, sigmas)
samples = sample_annealed_langevin(model, sigmas)
```

## Summary

**The score function $\nabla \log p(x)$ is central** to:

1. **Langevin MCMC**: Sampling from known distributions
2. **Score matching**: Learning distributions from data
3. **Diffusion models**: Generative modeling

**Key connections**:
- **Denoising score matching** = training objective for diffusion
- **Annealed Langevin** = reverse diffusion process
- **Temperature/noise** = same concept in both frameworks

**Practical impact**:
- MCMC methods inform generative modeling
- Generative modeling insights improve MCMC
- Cross-pollination between fields

**The big picture**: 
These aren't separate techniques — they're different applications of the same fundamental mathematics of following the score function through probability space!

**Modern state-of-the-art**:
- Stable Diffusion, DALL-E 2: Image generation
- WaveGrad: Audio generation  
- AlphaFold: Protein structure (uses score-based ideas)

The score function has become one of the most important concepts in modern machine learning, unifying classical MCMC with cutting-edge generative models.
