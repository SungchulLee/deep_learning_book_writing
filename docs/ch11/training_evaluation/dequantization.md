# Dequantization

## Introduction

Discrete data (like image pixels with values 0-255) poses a fundamental challenge for normalizing flows, which model continuous densities. **Dequantization** transforms discrete data into continuous data, enabling flow-based density estimation while maintaining a valid probabilistic interpretation.

## The Problem with Discrete Data

Normalizing flows define a probability density $p(x)$ over continuous $x \in \mathbb{R}^d$. But images have discrete pixel values:

$$x_{\text{discrete}} \in \{0, 1, 2, \ldots, 255\}^d$$

**The issue**: A continuous density assigns probability zero to any single discrete point. We can't directly compute $\log p(x)$ for discrete $x$.

### Naive Approaches (Don't Work)

1. **Treat discrete as continuous**: Evaluate $p(x)$ at integer values. This gives a density value, not a probability, and isn't comparable across models.

2. **Bin counting**: Integrate density over bins. Expensive and loses gradient information.

## Uniform Dequantization

The standard solution: add uniform noise to spread discrete values over bins.

### Method

For 8-bit images (values 0-255), transform:

$$\tilde{x} = \frac{x + u}{256}, \quad u \sim \text{Uniform}(0, 1)$$

Then $\tilde{x} \in [0, 1)$ is continuous, and:

$$P(x) = \int_{x/256}^{(x+1)/256} p(\tilde{x}) \, d\tilde{x}$$

The probability mass of the discrete value $x$ equals the integral of the density over its "bin."

### Mathematical Justification

The relationship between discrete and continuous probabilities:

$$P(x_{\text{discrete}}) = \int_0^1 p_X\left(\frac{x + u}{256}\right) \cdot \frac{1}{256} \, du = \mathbb{E}_{u}\left[\frac{p_X((x + u)/256)}{256}\right]$$

In log-space, Jensen's inequality gives:
$$\log P(x) = \log \mathbb{E}_u\left[\frac{p_X((x+u)/256)}{256}\right] \geq \mathbb{E}_u\left[\log \frac{p_X((x+u)/256)}{256}\right]$$

So training with dequantized data optimizes a **lower bound** on the true discrete log-likelihood.

### Implementation

```python
def dequantize_uniform(x: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    """
    Add uniform noise for dequantization.
    
    Args:
        x: Integer tensor with values in [0, 2^num_bits - 1]
        num_bits: Bit depth (8 for standard images)
    
    Returns:
        Continuous tensor in [0, 1)
    """
    num_levels = 2 ** num_bits  # 256 for 8-bit
    
    # Add uniform noise
    noise = torch.rand_like(x.float())
    
    # Scale to [0, 1)
    x_dequant = (x.float() + noise) / num_levels
    
    return x_dequant


def quantize(x: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    """
    Reverse dequantization (for sampling).
    
    Args:
        x: Continuous tensor in [0, 1)
        num_bits: Bit depth
    
    Returns:
        Integer tensor with discrete values
    """
    num_levels = 2 ** num_bits
    
    # Scale and floor
    x_discrete = torch.floor(x * num_levels).clamp(0, num_levels - 1)
    
    return x_discrete.long()
```

### Bits Per Dimension

A common evaluation metric that accounts for dequantization:

$$\text{BPD} = -\frac{\log_2 p(x)}{d} = -\frac{\log p(x)}{d \cdot \ln 2}$$

where $d$ is the data dimensionality. For images:

$$\text{BPD} = -\frac{\mathbb{E}[\log p(\tilde{x})]}{H \times W \times C \times \ln 2}$$

Lower BPD means better density estimation. Typical values:

| Dataset | Good BPD |
|---------|----------|
| MNIST | ~1.0 |
| CIFAR-10 | ~3.5 |
| ImageNet | ~4.0 |

## Variational Dequantization

Uniform dequantization adds noise independently of the data content. **Variational dequantization** learns a data-dependent noise distribution, providing a tighter lower bound.

### Idea

Instead of $u \sim \text{Uniform}(0,1)$, learn:
$$u \sim q_\phi(u | x)$$

where $q_\phi$ is a normalizing flow conditioned on $x$.

### Variational Lower Bound

$$\log P(x) \geq \mathbb{E}_{u \sim q_\phi}[\log p(\tilde{x}) - \log q_\phi(u | x)]$$

The additional $-\log q_\phi$ term encourages $q_\phi$ to have high entropy (spread out), preventing it from collapsing to a point.

### Implementation

```python
class VariationalDequantization(nn.Module):
    """
    Variational dequantization using a conditional flow.
    """
    
    def __init__(self, dim: int, hidden_dim: int = 64, n_layers: int = 4):
        super().__init__()
        
        # Conditional flow for noise distribution q(u|x)
        self.noise_flow = ConditionalFlow(
            dim=dim,
            context_dim=dim,  # Condition on x
            hidden_dim=hidden_dim,
            n_layers=n_layers
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dequantize with learned noise distribution.
        
        Args:
            x: Discrete input [0, 255]
        
        Returns:
            x_dequant: Continuous dequantized data
            log_q: Log probability of noise under q(u|x)
        """
        # Sample base noise
        eps = torch.rand_like(x.float())
        
        # Transform through conditional flow
        u, log_det = self.noise_flow.forward(eps, context=x.float() / 256)
        
        # Ensure u in [0, 1) using sigmoid
        u = torch.sigmoid(u)
        
        # Jacobian correction for sigmoid
        log_sigmoid_det = torch.log(u * (1 - u) + 1e-8).sum(dim=-1)
        
        # Total log q(u|x)
        log_q = -log_det - log_sigmoid_det
        
        # Dequantize
        x_dequant = (x.float() + u) / 256
        
        return x_dequant, log_q
    
    def loss(self, x: torch.Tensor, flow_model: nn.Module) -> torch.Tensor:
        """
        Compute variational dequantization loss.
        
        Args:
            x: Discrete input
            flow_model: Main normalizing flow
        
        Returns:
            Negative ELBO
        """
        # Dequantize
        x_dequant, log_q = self.forward(x)
        
        # Flow log probability
        log_p = flow_model.log_prob(x_dequant)
        
        # ELBO: log p(x_dequant) - log q(u|x)
        elbo = log_p - log_q
        
        return -elbo.mean()
```

### Benefits of Variational Dequantization

1. **Tighter Bound**: Learns optimal noise distribution
2. **Data-Adaptive**: Different noise for different inputs
3. **Better BPD**: Typically 0.1-0.3 bits improvement

## Practical Considerations

### When to Use Each Method

| Method | Use Case |
|--------|----------|
| **Uniform** | Simple baseline, fast training |
| **Variational** | Best performance, more compute |
| **None** | Already continuous data |

### Common Pitfalls

1. **Forgetting to Dequantize**: Training on integer pixels gives wrong results
2. **Wrong Scaling**: Must match dequantization range to flow input range
3. **Ignoring Jacobian**: Scaling from [0,255] to [0,1] has Jacobian term

### Scaling Jacobian

If you scale dequantized data:

$$y = \alpha \cdot \tilde{x} + \beta$$

Add to log-likelihood:
$$\log p(y) = \log p(\tilde{x}) - d \cdot \log|\alpha|$$

### Training with Dequantization

```python
class MNISTFlowWithDequant:
    """Complete training pipeline with dequantization."""
    
    def __init__(self, use_variational: bool = False):
        self.flow = build_realnvp_model(dim=784, n_layers=8)
        
        if use_variational:
            self.dequant = VariationalDequantization(dim=784)
        else:
            self.dequant = None
    
    def train_step(self, x: torch.Tensor) -> torch.Tensor:
        """Single training step."""
        
        if self.dequant is not None:
            # Variational dequantization
            x_cont, log_q = self.dequant(x)
            log_p = self.flow.log_prob(x_cont)
            loss = -(log_p - log_q).mean()
        else:
            # Uniform dequantization
            noise = torch.rand_like(x.float())
            x_cont = (x.float() + noise) / 256
            log_p = self.flow.log_prob(x_cont)
            loss = -log_p.mean()
        
        return loss
```

## Connection to Other Concepts

### Relationship to VAE

Variational dequantization is similar to the VAE's encoder:

- VAE: $q_\phi(z|x)$ approximates $p(z|x)$
- VarDeq: $q_\phi(u|x)$ approximates optimal noise distribution

Both use variational inference to optimize a lower bound.

### Dequantization in Other Models

- **Diffusion Models**: Often use continuous data; dequantization not always needed
- **GANs**: Don't compute likelihoods; dequantization not applicable
- **Autoregressive Models**: Can model discrete directly via categorical distributions

## Summary

Dequantization bridges discrete data and continuous density models:

1. **Uniform Dequantization**: Simple, adds uniform noise to create continuous data
2. **Variational Dequantization**: Learns optimal noise distribution for tighter bounds
3. **Evaluation**: Use bits-per-dimension (BPD) for standardized comparison

Key takeaway: Always dequantize discrete data before training normalizing flows, and account for the transformation in your likelihood computation.

## References

1. Theis, L., et al. (2016). A Note on the Evaluation of Generative Models. *ICLR*.
2. Ho, J., et al. (2019). Flow++: Improving Flow-Based Generative Models with Variational Dequantization. *ICML*.
3. Dinh, L., et al. (2017). Density Estimation Using Real-NVP. *ICLR*.
