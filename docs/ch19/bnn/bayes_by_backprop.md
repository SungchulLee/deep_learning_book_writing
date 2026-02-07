# Bayes by Backprop

## Overview

**Bayes by Backprop** (Blundell et al., 2015) is a practical algorithm for training Bayesian neural networks using variational inference. It learns a distribution over weights by optimizing the Evidence Lower Bound (ELBO) through standard backpropagation, enabled by the **reparameterization trick**.

!!! note "Complete Coverage"
    For advanced topics including local reparameterization, normalizing flows over weights, and comprehensive benchmarks, see **[Ch33: Bayes by Backprop](../../ch33/model_uncertainty/bayesian_methods/bayes_backprop.md)**.

---

## The Variational Objective

Approximate the intractable posterior with a tractable variational distribution:

$$
p(\theta \mid \mathcal{D}) \approx q_\phi(\theta)
$$

The ELBO objective (see [ELBO](../variational_inference/elbo.md)):

$$
\boxed{\mathcal{L}(\phi) = \mathbb{E}_{q_\phi(\theta)}[\log p(\mathcal{D} \mid \theta)] - \text{KL}(q_\phi(\theta) \| p(\theta))}
$$

The first term measures **data fit** (expected log-likelihood) and the second is a **complexity penalty** (divergence from prior).

---

## Mean-Field Gaussian Variational Family

The standard choice parameterizes each weight independently as Gaussian:

$$
q_\phi(\theta) = \prod_{j=1}^D \mathcal{N}(\theta_j \mid \mu_j, \sigma_j^2)
$$

The variational parameters are $\phi = \{\mu_j, \rho_j\}_{j=1}^D$ where $\sigma_j = \log(1 + \exp(\rho_j))$ (softplus ensures positivity).

---

## The Reparameterization Trick

To backpropagate through the stochastic sampling, reparameterize:

$$
\theta = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

This moves the stochasticity to the fixed distribution $\epsilon$, making the objective differentiable with respect to $\mu$ and $\rho$:

$$
\nabla_\phi \mathcal{L} = \nabla_\phi \mathbb{E}_{\epsilon}[\log p(\mathcal{D} \mid \mu + \sigma \odot \epsilon) - \log q_\phi(\mu + \sigma \odot \epsilon) + \log p(\mu + \sigma \odot \epsilon)]
$$

---

## Algorithm

```
Algorithm: Bayes by Backprop
─────────────────────────────
Input: Dataset D, prior p(θ), learning rate η
Initialize: μ, ρ (variational parameters)

For each epoch:
    For each minibatch (x, y) of size m:
        1. Sample ε ~ N(0, I)
        2. Compute θ = μ + softplus(ρ) ⊙ ε
        3. Compute loss:
           L = (1/m) Σ -log p(y_i | x_i, θ)     [NLL]
             + (β/N) KL(q_φ(θ) || p(θ))          [complexity cost]
        4. Compute gradients ∂L/∂μ, ∂L/∂ρ
        5. Update: μ ← μ - η ∂L/∂μ
                   ρ ← ρ - η ∂L/∂ρ
```

The **KL weighting** $\beta$ can be annealed during training (warm-up) to prevent the KL term from dominating early and collapsing the posterior to the prior.

---

## KL Divergence for Gaussian Prior and Posterior

When both prior and approximate posterior are Gaussian, the KL has closed form:

$$
\text{KL}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, \sigma_p^2)) = \log\frac{\sigma_p}{\sigma} + \frac{\sigma^2 + \mu^2}{2\sigma_p^2} - \frac{1}{2}
$$

For the full network with $D$ weights:

$$
\text{KL}(q_\phi \| p) = \sum_{j=1}^D \left[\log\frac{\sigma_p}{\sigma_j} + \frac{\sigma_j^2 + \mu_j^2}{2\sigma_p^2} - \frac{1}{2}\right]
$$

---

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BayesLinear(nn.Module):
    """
    Bayesian linear layer with learnable weight distributions.
    
    Weights: w ~ N(mu_w, softplus(rho_w)^2)
    Biases:  b ~ N(mu_b, softplus(rho_b)^2)
    """
    
    def __init__(self, in_features: int, out_features: int,
                 prior_sigma: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma
        
        # Variational parameters for weights
        self.mu_w = nn.Parameter(torch.empty(out_features, in_features))
        self.rho_w = nn.Parameter(torch.empty(out_features, in_features))
        
        # Variational parameters for biases
        self.mu_b = nn.Parameter(torch.empty(out_features))
        self.rho_b = nn.Parameter(torch.empty(out_features))
        
        self.reset_parameters()
        self.kl = 0.0
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.mu_w, a=math.sqrt(5))
        nn.init.constant_(self.rho_w, -3.0)  # small initial variance
        nn.init.zeros_(self.mu_b)
        nn.init.constant_(self.rho_b, -3.0)
    
    @property
    def sigma_w(self):
        return F.softplus(self.rho_w)
    
    @property
    def sigma_b(self):
        return F.softplus(self.rho_b)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sample weights via reparameterization
        eps_w = torch.randn_like(self.mu_w)
        eps_b = torch.randn_like(self.mu_b)
        
        w = self.mu_w + self.sigma_w * eps_w
        b = self.mu_b + self.sigma_b * eps_b
        
        # Compute KL divergence
        self.kl = self._kl_divergence(
            self.mu_w, self.sigma_w, self.mu_b, self.sigma_b)
        
        return F.linear(x, w, b)
    
    def _kl_divergence(self, mu_w, sigma_w, mu_b, sigma_b):
        """Closed-form KL(q || prior) for Gaussian distributions."""
        sp = self.prior_sigma
        
        kl_w = (torch.log(sp / sigma_w) + 
                (sigma_w**2 + mu_w**2) / (2 * sp**2) - 0.5).sum()
        kl_b = (torch.log(sp / sigma_b) + 
                (sigma_b**2 + mu_b**2) / (2 * sp**2) - 0.5).sum()
        
        return kl_w + kl_b


class BayesianMLP(nn.Module):
    """Bayesian MLP for regression or classification."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 prior_sigma: float = 1.0):
        super().__init__()
        self.fc1 = BayesLinear(input_dim, hidden_dim, prior_sigma)
        self.fc2 = BayesLinear(hidden_dim, hidden_dim, prior_sigma)
        self.fc3 = BayesLinear(hidden_dim, output_dim, prior_sigma)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def kl_loss(self):
        """Total KL divergence across all Bayesian layers."""
        return sum(m.kl for m in self.modules() if isinstance(m, BayesLinear))
    
    def predict(self, x: torch.Tensor, n_samples: int = 50):
        """Monte Carlo predictive distribution."""
        self.eval()
        preds = torch.stack([self(x) for _ in range(n_samples)])
        return preds.mean(dim=0), preds.var(dim=0)


def train_bnn(model, train_loader, n_epochs=100, lr=1e-3, 
              n_train=None, kl_weight=1.0):
    """
    Train a Bayesian neural network with KL annealing.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(n_epochs):
        # KL annealing: gradually increase KL weight
        beta = min(1.0, (epoch + 1) / (n_epochs * 0.3)) * kl_weight
        
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            output = model(x_batch)
            nll = F.mse_loss(output, y_batch, reduction='sum')
            kl = model.kl_loss()
            
            # ELBO loss: NLL + scaled KL
            loss = nll + beta * kl / (n_train or len(train_loader.dataset))
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    
    return model
```

---

## MC Dropout as Approximate BNN

MC Dropout (Gal & Ghahramani, 2016) provides a simpler alternative that reinterprets standard dropout as approximate variational inference:

$$
p(y^* \mid x^*, \mathcal{D}) \approx \frac{1}{T} \sum_{t=1}^T p(y^* \mid x^*, \hat{w}_t)
$$

where $\hat{w}_t$ are weights with random dropout masks applied at test time.

```python
class MCDropoutModel(nn.Module):
    """Standard network usable as approximate BNN via MC Dropout."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def mc_predict(self, x, n_samples=100):
        """Run T forward passes with dropout active."""
        self.train()  # keep dropout active
        preds = torch.stack([self(x) for _ in range(n_samples)])
        self.eval()
        return preds.mean(dim=0), preds.var(dim=0)
```

| Aspect | Bayes by Backprop | MC Dropout |
|--------|-------------------|------------|
| Parameters | $2D$ (mean + variance per weight) | $D$ (standard weights) |
| Training cost | ~2x standard training | Standard training |
| Inference cost | $T$ forward passes | $T$ forward passes |
| Uncertainty quality | Better calibration | May underestimate |
| Implementation | Custom layers | Standard + test-time dropout |

---

## Summary

| Concept | Key Point |
|---------|-----------|
| **ELBO objective** | Data fit - KL complexity penalty |
| **Reparameterization trick** | Enables gradient-based optimization through sampling |
| **Mean-field Gaussian** | Independent Gaussian per weight; $2D$ parameters |
| **KL annealing** | Warm up KL weight to prevent posterior collapse |
| **MC Dropout** | Simpler alternative; reinterprets dropout as variational inference |

---

## References

- Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015). Weight Uncertainty in Neural Networks. *ICML*.
- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation. *ICML*.
- Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. *ICLR*.
- Graves, A. (2011). Practical Variational Inference for Neural Networks. *NeurIPS*.
