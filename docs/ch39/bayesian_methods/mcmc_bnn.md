# MCMC Methods for Bayesian Neural Networks
## Overview

Markov Chain Monte Carlo (MCMC) methods provide the most principled approach to posterior inference in Bayesian neural networks, generating samples that asymptotically converge to the true posterior distribution. While computationally expensive, they serve as the gold standard for uncertainty quantification and are essential benchmarks against which approximate methods are evaluated.

## Hamiltonian Monte Carlo (HMC)

### The Algorithm

HMC augments the parameter space with momentum variables $\mathbf{r}$ and simulates Hamiltonian dynamics:

$$H(\theta, \mathbf{r}) = U(\theta) + K(\mathbf{r})$$

where $U(\theta) = -\log p(\theta | \mathcal{D})$ is the potential energy (negative log-posterior) and $K(\mathbf{r}) = \frac{1}{2}\mathbf{r}^T M^{-1} \mathbf{r}$ is the kinetic energy.

The leapfrog integrator alternates half-steps:

$$\mathbf{r}_{t+\epsilon/2} = \mathbf{r}_t - \frac{\epsilon}{2} \nabla_\theta U(\theta_t)$$

$$\theta_{t+\epsilon} = \theta_t + \epsilon M^{-1} \mathbf{r}_{t+\epsilon/2}$$

$$\mathbf{r}_{t+\epsilon} = \mathbf{r}_{t+\epsilon/2} - \frac{\epsilon}{2} \nabla_\theta U(\theta_{t+\epsilon})$$

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import numpy as np
from typing import List, Callable, Tuple


class HamiltonianMonteCarlo:
    """
    HMC sampler for neural network posteriors.
    
    Note: Full HMC requires computing gradients over the entire dataset,
    making it impractical for large-scale problems. Use SGLD/SGHMC
    for scalable alternatives.
    """
    
    def __init__(
        self,
        model: nn.Module,
        log_posterior_fn: Callable,
        step_size: float = 0.001,
        n_leapfrog: int = 10,
        mass_matrix: str = 'identity'
    ):
        self.model = model
        self.log_posterior_fn = log_posterior_fn
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog
        
        # Flatten parameters for sampling
        self.param_shapes = [p.shape for p in model.parameters()]
        self.n_params = sum(p.numel() for p in model.parameters())
    
    def _flatten_params(self) -> torch.Tensor:
        return torch.cat([p.data.flatten() for p in self.model.parameters()])
    
    def _unflatten_params(self, flat: torch.Tensor):
        idx = 0
        for p, shape in zip(self.model.parameters(), self.param_shapes):
            n = p.numel()
            p.data = flat[idx:idx+n].reshape(shape)
            idx += n
    
    def _compute_potential_energy(self) -> torch.Tensor:
        """U(θ) = -log p(θ|D)"""
        return -self.log_posterior_fn(self.model)
    
    def _compute_gradient(self) -> torch.Tensor:
        """∇U(θ)"""
        self.model.zero_grad()
        U = self._compute_potential_energy()
        U.backward()
        return torch.cat([p.grad.flatten() for p in self.model.parameters()])
    
    def step(self) -> Tuple[bool, float]:
        """
        One HMC step: leapfrog integration + Metropolis accept/reject.
        
        Returns:
            accepted: Whether proposal was accepted
            log_prob: Log probability at current state
        """
        # Save current state
        current_params = self._flatten_params().clone()
        
        # Sample momentum
        momentum = torch.randn(self.n_params)
        current_momentum = momentum.clone()
        
        # Current Hamiltonian
        current_U = self._compute_potential_energy().item()
        current_K = 0.5 * torch.sum(current_momentum ** 2).item()
        
        # Leapfrog integration
        grad = self._compute_gradient()
        momentum = momentum - 0.5 * self.step_size * grad
        
        for i in range(self.n_leapfrog - 1):
            params = self._flatten_params() + self.step_size * momentum
            self._unflatten_params(params)
            
            grad = self._compute_gradient()
            momentum = momentum - self.step_size * grad
        
        # Final half-step
        params = self._flatten_params() + self.step_size * momentum
        self._unflatten_params(params)
        grad = self._compute_gradient()
        momentum = momentum - 0.5 * self.step_size * grad
        
        # Proposed Hamiltonian
        proposed_U = self._compute_potential_energy().item()
        proposed_K = 0.5 * torch.sum(momentum ** 2).item()
        
        # Metropolis accept/reject
        log_accept = (current_U + current_K) - (proposed_U + proposed_K)
        
        if np.log(np.random.uniform()) < log_accept:
            return True, -proposed_U
        else:
            self._unflatten_params(current_params)
            return False, -current_U
    
    def sample(
        self, n_samples: int, burn_in: int = 100, thin: int = 1
    ) -> List[torch.Tensor]:
        """Collect posterior samples."""
        samples = []
        n_accepted = 0
        
        for i in range(burn_in + n_samples * thin):
            accepted, log_prob = self.step()
            n_accepted += int(accepted)
            
            if i >= burn_in and (i - burn_in) % thin == 0:
                samples.append(self._flatten_params().clone())
        
        total_steps = burn_in + n_samples * thin
        accept_rate = n_accepted / total_steps
        print(f"Acceptance rate: {accept_rate:.3f}")
        
        return samples
```

## Stochastic Gradient Langevin Dynamics (SGLD)

SGLD enables scalable Bayesian inference by using minibatch gradients with injected noise:

$$\theta_{t+1} = \theta_t + \frac{\epsilon_t}{2}\left(\nabla \log p(\theta_t) + \frac{N}{n}\sum_{i \in S_t} \nabla \log p(y_i | x_i, \theta_t)\right) + \eta_t$$

where $\eta_t \sim \mathcal{N}(0, \epsilon_t I)$ and $\epsilon_t$ is a decaying learning rate.

```python
class SGLDOptimizer(torch.optim.Optimizer):
    """
    Stochastic Gradient Langevin Dynamics optimizer.
    
    Combines SGD with Gaussian noise injection for posterior sampling.
    As learning rate decays, samples converge to the posterior.
    """
    
    def __init__(self, params, lr=1e-3, weight_decay=0.0,
                 noise_scale=1.0, temperature=1.0):
        defaults = dict(lr=lr, weight_decay=weight_decay,
                       noise_scale=noise_scale, temperature=temperature)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad.data
                
                # Weight decay (prior)
                if group['weight_decay'] != 0:
                    d_p = d_p + group['weight_decay'] * p.data
                
                # SGD update
                lr = group['lr']
                p.data.add_(d_p, alpha=-lr)
                
                # Langevin noise injection
                noise = torch.randn_like(p.data)
                noise_std = (2.0 * lr * group['temperature']) ** 0.5
                p.data.add_(noise, alpha=noise_std * group['noise_scale'])


def train_with_sgld(
    model: nn.Module,
    train_loader,
    n_epochs: int = 100,
    lr: float = 1e-3,
    collect_every: int = 10,
    burn_in_epochs: int = 50,
    weight_decay: float = 1e-4,
    dataset_size: int = None
) -> List[dict]:
    """
    Train with SGLD and collect posterior samples.
    
    Returns list of parameter snapshots (posterior samples).
    """
    optimizer = SGLDOptimizer(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    
    samples = []
    
    for epoch in range(n_epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            
            # Scale gradient for minibatch
            if dataset_size is not None:
                loss = loss * dataset_size / len(y)
            
            loss.backward()
            optimizer.step()
        
        # Collect samples after burn-in
        if epoch >= burn_in_epochs and epoch % collect_every == 0:
            snapshot = {
                name: param.data.clone()
                for name, param in model.named_parameters()
            }
            samples.append(snapshot)
    
    print(f"Collected {len(samples)} posterior samples")
    return samples
```

## SGHMC: Stochastic Gradient Hamiltonian Monte Carlo

SGHMC adds momentum to SGLD for better exploration:

$$\theta_{t+1} = \theta_t + \epsilon_t \mathbf{v}_t$$

$$\mathbf{v}_{t+1} = (1 - \alpha)\mathbf{v}_t + \epsilon_t \hat{\nabla} \log p(\theta_t | \mathcal{D}) + \mathcal{N}(0, 2\alpha\epsilon_t I)$$

where $\alpha$ is the friction coefficient and $\hat{\nabla}$ denotes the stochastic gradient.

```python
class SGHMCOptimizer(torch.optim.Optimizer):
    """Stochastic Gradient Hamiltonian Monte Carlo."""
    
    def __init__(self, params, lr=1e-4, momentum_decay=0.01,
                 noise_scale=1.0):
        defaults = dict(lr=lr, momentum_decay=momentum_decay,
                       noise_scale=noise_scale)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                if len(state) == 0:
                    state['velocity'] = torch.zeros_like(p.data)
                
                v = state['velocity']
                lr = group['lr']
                alpha = group['momentum_decay']
                
                # Friction + gradient + noise
                noise = torch.randn_like(p.data)
                noise_std = (2.0 * alpha * lr) ** 0.5
                
                v.mul_(1 - alpha).add_(
                    p.grad.data, alpha=-lr
                ).add_(noise, alpha=noise_std * group['noise_scale'])
                
                p.data.add_(v, alpha=lr)
```

## Prediction with MCMC Samples

```python
def predict_with_mcmc_samples(
    model: nn.Module,
    samples: List[dict],
    x: torch.Tensor,
    task: str = 'classification'
) -> dict:
    """
    Make predictions using collected MCMC posterior samples.
    """
    all_outputs = []
    
    model.eval()
    with torch.no_grad():
        for sample in samples:
            # Load sampled weights
            for name, param in model.named_parameters():
                param.data.copy_(sample[name])
            
            output = model(x)
            all_outputs.append(output)
    
    outputs = torch.stack(all_outputs)  # (S, batch, dim)
    
    if task == 'classification':
        probs = torch.softmax(outputs, dim=-1)
        mean_probs = probs.mean(dim=0)
        pred_class = mean_probs.argmax(dim=-1)
        epistemic = probs.var(dim=0).mean(dim=-1)
        
        return {
            'probs': mean_probs,
            'pred_class': pred_class,
            'epistemic_uncertainty': epistemic
        }
    else:
        mean = outputs.mean(dim=0)
        epistemic_var = outputs.var(dim=0)
        
        return {
            'mean': mean,
            'epistemic_var': epistemic_var,
            'total_std': torch.sqrt(epistemic_var)
        }
```

## Practical Considerations

### When to Use MCMC for BNNs

- Research requiring gold-standard posterior approximations
- Small to medium models (< 10M parameters)
- When uncertainty quality is more important than computational cost
- Benchmarking approximate methods

### Limitations

- **Scalability**: Full HMC requires full-dataset gradients
- **Mixing**: Poor mixing in high dimensions leads to correlated samples
- **Diagnostics**: Convergence assessment is challenging for neural networks
- **Multimodality**: Standard MCMC may not explore all posterior modes

### Recommendations

| Setting | Method | Notes |
|---------|--------|-------|
| Small model, gold standard | HMC | Best accuracy, high cost |
| Medium model, scalable | SGLD | Minibatch-compatible |
| Better exploration needed | SGHMC | Momentum helps mixing |
| Large-scale production | Use ensembles/SWAG instead | MCMC too expensive |

## References

- Welling, M., & Teh, Y. W. (2011). "Bayesian Learning via Stochastic Gradient Langevin Dynamics." ICML.
- Chen, T., et al. (2014). "Stochastic Gradient Hamiltonian Monte Carlo." ICML.
- Neal, R. M. (2011). "MCMC Using Hamiltonian Dynamics." Handbook of MCMC.

## Exercises

**Exercise 1.**
For a two-layer neural network with ReLU activations and Gaussian weight priors, derive the form of the approximate posterior under the method described in this section.

??? success "Solution to Exercise 1"
    With weights $W_1, W_2$ and Gaussian prior $p(W) = \mathcal{N}(0, \sigma_p^2 I)$, the posterior $p(W | D) \propto p(D | W) p(W)$ is intractable. The approximation method from this section produces a tractable form: for variational inference, each weight has an independent Gaussian posterior $q(w_{ij}) = \mathcal{N}(\mu_{ij}, \sigma_{ij}^2)$; for Laplace approximation, the posterior is a single Gaussian centered at the MAP estimate with covariance equal to the inverse Hessian; for MC Dropout, the posterior is implicitly defined by the dropout mask distribution. Each approximation captures different aspects of the true posterior's shape. $\square$

---

**Exercise 2.**
Design an experiment to compare the calibration of uncertainty estimates from this method against MC Dropout and deep ensembles. Specify the metrics and visualization.

??? success "Solution to Exercise 2"
    Metrics: (1) Expected Calibration Error (ECE) with 15 bins; (2) Brier score; (3) negative log-likelihood (NLL); (4) AUROC for OOD detection. Visualization: reliability diagrams plotting observed frequency vs. predicted confidence for each method. Protocol: train all methods on CIFAR-10 (in-distribution), evaluate calibration on CIFAR-10 test set, and OOD detection on SVHN. Use temperature scaling as a post-hoc baseline. Report means and standard errors over 5 random seeds. A well-calibrated method has points close to the diagonal in the reliability diagram and low ECE. $\square$

---

**Exercise 3.**
Prove that the predictive variance from a Bayesian neural network decomposes into epistemic and aleatoric components. Show how each component behaves as the training set size $N \to \infty$.

??? success "Solution to Exercise 3"
    The predictive variance decomposes via the law of total variance: $\text{Var}[y | x, D] = \underbrace{\mathbb{E}_{p(\theta|D)}[\text{Var}[y | x, \theta]]}_{\text{aleatoric}} + \underbrace{\text{Var}_{p(\theta|D)}[\mathbb{E}[y | x, \theta]]}_{\text{epistemic}}$. The aleatoric component captures irreducible noise in the data-generating process and remains constant as $N \to \infty$. The epistemic component reflects parameter uncertainty, which decreases as $O(1/N)$ because the posterior concentrates around the true parameters. In the limit, only aleatoric uncertainty remains. This decomposition is crucial for deciding when to collect more data (high epistemic) vs. accepting inherent noise (high aleatoric). $\square$

---

**Exercise 4.**
Discuss how the uncertainty quantification method from this section could be used for position sizing in a trading system. Propose a concrete decision rule.

??? success "Solution to Exercise 4"
    Decision rule: the position size is inversely proportional to the epistemic uncertainty. Let $\hat{y}$ be the predicted return and $\sigma_e^2$ be the epistemic variance. The position is $w = \frac{\hat{y}}{\lambda \sigma_e^2}$ where $\lambda$ is a risk aversion parameter. When epistemic uncertainty is high (novel market conditions), positions are reduced; when low (familiar regimes), the system trades with higher conviction. Additionally, set a maximum epistemic uncertainty threshold above which no trade is placed (abstention). This framework naturally implements a Kelly-criterion-like sizing scaled by model confidence. Backtest with walk-forward validation to calibrate $\lambda$. $\square$
