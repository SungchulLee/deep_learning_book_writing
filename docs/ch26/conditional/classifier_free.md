# Classifier-Free Guidance

**Classifier-Free Guidance (CFG)** is a technique for improving conditional generation quality in diffusion models without requiring a separate classifier.

## Motivation

### The Conditioning Problem

Conditional diffusion models $p_\theta(x|c)$ often produce samples that:
- Don't strongly reflect the condition $c$
- Lack the sharpness of unconditional samples
- Fail to capture fine-grained conditioning details

### Classifier Guidance (Predecessor)

Dhariwal & Nichol (2021) proposed using a classifier $p_\phi(c|x_t)$ to guide sampling:

$$
\tilde{\epsilon}(x_t, t, c) = \epsilon_\theta(x_t, t) - \sqrt{1-\bar{\alpha}_t} \nabla_{x_t} \log p_\phi(c|x_t)
$$

**Limitations**: Requires training a noisy classifier and computing gradients at each step.

## Classifier-Free Guidance

### Key Idea (Ho & Salimans, 2022)

Train a **single model** that can be both conditional and unconditional, then combine predictions:

$$
\tilde{\epsilon}_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, \varnothing) + w \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing))
$$

where:
- $\epsilon_\theta(x_t, t, c)$: conditional prediction
- $\epsilon_\theta(x_t, t, \varnothing)$: unconditional prediction
- $w \geq 1$: guidance scale

### Simplified Form

$$
\tilde{\epsilon}_\theta = (1 + w) \epsilon_\theta(x_t, t, c) - w \cdot \epsilon_\theta(x_t, t, \varnothing)
$$

Or equivalently:

$$
\tilde{\epsilon}_\theta = \epsilon_\theta(x_t, t, \varnothing) + w \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing))
$$

## Interpretation

### Score Function View

The guided score is:

$$
\tilde{s}(x_t, t, c) = s(x_t, t) + w \cdot \nabla_{x_t} \log p(c|x_t)
$$

CFG implicitly computes the classifier gradient as:

$$
\nabla_{x_t} \log p(c|x_t) \propto \epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing)
$$

### Effect of Guidance Scale

| $w$ | Behavior |
|-----|----------|
| 0 | Unconditional sampling |
| 1 | Standard conditional sampling |
| 1-5 | Mild guidance, balanced diversity/fidelity |
| 5-15 | Strong guidance, high fidelity |
| >15 | Over-saturated, reduced diversity |

## Training with Classifier-Free Guidance

### Dropout Conditioning

During training, randomly drop the condition with probability $p_{\text{uncond}}$:

```python
def training_step(model, x_0, condition, p_uncond=0.1):
    # Random dropout of condition
    mask = torch.rand(x_0.shape[0]) < p_uncond
    condition_input = condition.clone()
    condition_input[mask] = null_condition  # e.g., empty embedding
    
    # Normal DDPM training
    t = sample_timesteps(x_0.shape[0])
    noise = torch.randn_like(x_0)
    x_t = forward_diffusion(x_0, t, noise)
    
    noise_pred = model(x_t, t, condition_input)
    loss = mse_loss(noise_pred, noise)
    
    return loss
```

### Null Condition Choices

| Condition Type | Null Representation |
|----------------|---------------------|
| Class labels | Special "no class" token |
| Text embeddings | Empty string embedding |
| Images | Zero tensor or learned embedding |

## Sampling with CFG

### Algorithm

```
Algorithm: CFG Sampling
───────────────────────
Input: Model ε_θ, condition c, guidance scale w

x_T ~ N(0, I)

for t = T, T-1, ..., 1:
    # Unconditional prediction
    ε_uncond = ε_θ(x_t, t, ∅)
    
    # Conditional prediction  
    ε_cond = ε_θ(x_t, t, c)
    
    # Guided prediction
    ε̃ = ε_uncond + w * (ε_cond - ε_uncond)
    
    # DDPM/DDIM update using ε̃
    x_{t-1} = sample_step(x_t, ε̃, t)

return x_0
```

### Implementation

```python
import torch

class CFGSampler:
    def __init__(self, model, schedule, guidance_scale=7.5):
        self.model = model
        self.schedule = schedule
        self.w = guidance_scale
    
    @torch.no_grad()
    def sample(self, shape, condition, null_condition, device, num_steps=50):
        """Sample with classifier-free guidance."""
        x = torch.randn(shape, device=device)
        
        timesteps = self.get_timesteps(num_steps)
        
        for t in timesteps:
            t_batch = torch.full((shape[0],), t, device=device)
            
            # Compute both predictions
            # Option 1: Two forward passes
            eps_uncond = self.model(x, t_batch, null_condition)
            eps_cond = self.model(x, t_batch, condition)
            
            # Option 2: Batched (more efficient)
            # x_double = torch.cat([x, x])
            # t_double = torch.cat([t_batch, t_batch])
            # c_double = torch.cat([null_condition, condition])
            # eps_both = self.model(x_double, t_double, c_double)
            # eps_uncond, eps_cond = eps_both.chunk(2)
            
            # Apply guidance
            eps_guided = eps_uncond + self.w * (eps_cond - eps_uncond)
            
            # DDIM step
            x = self.ddim_step(x, eps_guided, t)
        
        return x
    
    def ddim_step(self, x, eps, t):
        """Single DDIM update step."""
        alpha_bar_t = self.schedule.alpha_bars[t]
        alpha_bar_prev = self.schedule.alpha_bars[t-1] if t > 0 else 1.0
        
        # Predict x_0
        x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
        x0_pred = torch.clamp(x0_pred, -1, 1)
        
        # Direction to x_t
        dir_xt = torch.sqrt(1 - alpha_bar_prev) * eps
        
        # Next sample
        x_prev = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt
        
        return x_prev
```

## Effect on Generation

### Quality vs Diversity Trade-off

CFG improves **fidelity** (how well samples match the condition) at the cost of **diversity** (variety in outputs).

### FID vs CLIP Score

| Guidance Scale | FID (↓) | CLIP Score (↑) |
|----------------|---------|----------------|
| 1.0 | Good | Moderate |
| 3.0 | Better | Good |
| 7.5 | Best | Best |
| 15.0 | Worse | Saturated |

Optimal $w$ depends on the task (typically 3-15 for images).

## Dynamic Guidance

### Per-Timestep Scaling

Some methods vary $w$ across timesteps:

$$
w(t) = w_{\min} + (w_{\max} - w_{\min}) \cdot f(t)
$$

Common choices for $f(t)$:
- Constant: $f(t) = 1$
- Linear: $f(t) = t/T$
- Cosine: $f(t) = \cos(\pi t / 2T)$

### Negative Prompts

For text-to-image, use a "negative prompt" instead of null:

$$
\tilde{\epsilon} = \epsilon_\theta(x_t, t, c_{\text{neg}}) + w \cdot (\epsilon_\theta(x_t, t, c_{\text{pos}}) - \epsilon_\theta(x_t, t, c_{\text{neg}}))
$$

This steers away from undesired attributes.

## Computational Considerations

### Cost

CFG requires **two forward passes** per denoising step (conditional + unconditional).

### Optimizations

1. **Batching**: Compute both in a single forward pass with batch size 2
2. **Caching**: For fixed conditions, cache unconditional predictions
3. **Distillation**: Train a guided model to match CFG outputs

## Applications

### Text-to-Image

Stable Diffusion uses CFG with $w \approx 7.5$:
- Without CFG: Blurry, weakly conditioned
- With CFG: Sharp, strongly conditioned on text

### Class-Conditional Generation

ImageNet models use CFG with $w \approx 2-4$:
- Improves class fidelity
- Maintains diversity within class

### Image Editing

Combine CFG with inversion for controlled editing:
1. Invert image to latent
2. Resample with modified condition + CFG

## Summary

Classifier-free guidance improves conditional generation by combining unconditional and conditional predictions. It requires training with condition dropout and doubles inference cost, but dramatically improves sample quality and conditioning strength. The guidance scale $w$ controls the fidelity-diversity trade-off, with values around 7.5 commonly used for text-to-image generation.
