# Reverse Mean Derivation

This document explains **why** the reverse mean formula in DDPM has its specific form, starting from the noise prediction view.

## The Formula

The reverse mean in DDPM is:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right)$$

This formula might look like "magic" at first glance. Let's unpack where it comes from.

## Step 1: Start from the Noise Parameterization

From the forward process we know:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

If the model predicts noise $\epsilon_\theta(x_t, t)$, we can estimate $x_0$ as:

$$\hat{x}_0(x_t, t) = \frac{1}{\sqrt{\bar{\alpha}_t}}\left(x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta(x_t, t)\right)$$

This is just rearranging the forward equation to solve for $x_0$.

## Step 2: Posterior Mean Using $x_0$ (DDPM Formula)

In DDPM, the **true** Gaussian posterior $q(x_{t-1} | x_t, x_0)$ has mean:

$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t$$

where $\beta_t = 1 - \alpha_t$ and $\bar{\alpha}_t = \alpha_t \bar{\alpha}_{t-1}$.

**Where does this come from?** It comes from multiplying two Gaussians:
- $q(x_{t-1} | x_0)$ — the prior
- $q(x_t | x_{t-1})$ — the likelihood

and using the closed-form formula for the product of Gaussians.

In practice, we don't know $x_0$, so we plug in the estimate $\hat{x}_0$:

$$\mu_\theta(x_t, t) := \tilde{\mu}_t(x_t, \hat{x}_0(x_t, t))$$

## Step 3: Plug $\hat{x}_0$ into the Posterior Mean

Substitute the $\hat{x}_0$ formula into the posterior mean:

$$\mu_\theta(x_t, t) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} \cdot \frac{1}{\sqrt{\bar{\alpha}_t}}\left(x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta(x_t, t)\right) + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t$$

Use the relation $\bar{\alpha}_t = \alpha_t \bar{\alpha}_{t-1}$, so:

$$\frac{\sqrt{\bar{\alpha}_{t-1}}}{\sqrt{\bar{\alpha}_t}} = \frac{1}{\sqrt{\alpha_t}}$$

Then:

$$\mu_\theta(x_t, t) = \frac{\beta_t}{(1-\bar{\alpha}_t)\sqrt{\alpha_t}}\left(x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta\right) + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t$$

## Step 4: Collect Terms

Now collect the $x_t$ terms and the $\epsilon_\theta$ term separately.

**$x_t$ coefficient:**
$$\frac{\beta_t}{(1-\bar{\alpha}_t)\sqrt{\alpha_t}} + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}$$

Using $\beta_t = 1 - \alpha_t$ and $1 - \bar{\alpha}_t = 1 - \alpha_t \bar{\alpha}_{t-1}$:
$$= \frac{1-\alpha_t + \alpha_t(1-\bar{\alpha}_{t-1})}{(1-\bar{\alpha}_t)\sqrt{\alpha_t}} = \frac{1-\bar{\alpha}_t}{(1-\bar{\alpha}_t)\sqrt{\alpha_t}} = \frac{1}{\sqrt{\alpha_t}}$$

**$\epsilon_\theta$ coefficient:**
$$-\frac{\beta_t \sqrt{1-\bar{\alpha}_t}}{(1-\bar{\alpha}_t)\sqrt{\alpha_t}} = -\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}\sqrt{\alpha_t}}$$

## Step 5: Final Result

Combining everything:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right)$$

**That's exactly the DDPM reverse mean formula!**

## Intuition

The formula says:
1. **Start with $x_t$** (the current noisy sample)
2. **Subtract the predicted noise** scaled appropriately
3. **Rescale** by $1/\sqrt{\alpha_t}$

This makes sense: we're removing the noise we think was added and scaling to account for the signal attenuation in the forward process.

## Alternative Parameterizations

The same idea works with different predictions:

| Prediction | Formula for $\mu_\theta$ |
|------------|--------------------------|
| $\epsilon_\theta$ (noise) | $\frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta\right)$ |
| $\hat{x}_0$ (clean data) | $\frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} \hat{x}_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t$ |
| $v$ (velocity) | Interpolation between $\epsilon$ and $x_0$ parameterizations |

## PyTorch Implementation

```python
def compute_reverse_mean(x_t, t, eps_pred, alphas, alphas_cumprod):
    """
    Compute reverse mean from noise prediction.
    
    μ_θ(x_t, t) = (1/√α_t) * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ)
    """
    alpha_t = alphas[t]
    alpha_bar_t = alphas_cumprod[t]
    
    # Reshape for broadcasting
    alpha_t = alpha_t.view(-1, 1, 1, 1)
    alpha_bar_t = alpha_bar_t.view(-1, 1, 1, 1)
    
    # Compute mean
    coef1 = 1 / torch.sqrt(alpha_t)
    coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
    
    mu = coef1 * (x_t - coef2 * eps_pred)
    
    return mu
```

## Summary

The reverse mean formula arises naturally from:
1. **Rearranging** the forward process to estimate $x_0$ from $\epsilon_\theta$
2. **Plugging** this estimate into the true posterior mean formula
3. **Simplifying** the algebra using the relations between $\alpha$, $\bar{\alpha}$, and $\beta$

The key insight is that predicting noise and predicting $x_0$ are equivalent—the same information, different parameterizations.

## Navigation

- **Previous**: [Reverse SDE](reverse_sde.md)
- **Next**: [Posterior Computation](posterior_computation.md)
