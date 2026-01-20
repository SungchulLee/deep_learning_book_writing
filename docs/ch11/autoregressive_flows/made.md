# MADE: Masked Autoencoder for Distribution Estimation

## Introduction

MADE (Masked Autoencoder for Distribution Estimation) is a clever architectural innovation that enables autoregressive density estimation in a single forward pass. By applying carefully designed masks to network weights, MADE computes all conditional distributions simultaneously, achieving the efficiency of standard neural networks while maintaining the autoregressive property.

## The Efficiency Problem

### Naive Autoregressive Models

A naive autoregressive model requires D forward passes for D-dimensional data:

```python
# Naive approach: D separate networks
def naive_log_prob(x):
    log_p = 0
    for d in range(D):
        # Network for dimension d
        params = network_d(x[:, :d])
        log_p += gaussian_log_prob(x[:, d], params)
    return log_p
```

This is inefficient: O(D) forward passes for each sample.

### MADE's Solution

MADE computes all conditionals in **one forward pass** by masking connections:

```python
# MADE: single forward pass
def made_log_prob(x):
    # All parameters at once
    all_params = masked_network(x)
    mu, log_sigma = all_params.chunk(2, dim=-1)
    return gaussian_log_prob(x, mu, log_sigma).sum()
```

## Masking Strategy

### The Autoregressive Constraint

For valid autoregressive modeling, output $x_d$ can only depend on inputs $x_1, \ldots, x_{d-1}$.

In matrix form, if $\mathbf{h} = \mathbf{W}\mathbf{x} + \mathbf{b}$, we need:

$$W_{id} = 0 \quad \text{whenever output } i \text{ should not depend on input } d$$

### Assigning Numbers to Units

MADE assigns each hidden unit a "degree" $m(k)$ representing the maximum input index it can depend on:

- **Input units**: $m(\text{input}_d) = d$
- **Hidden units**: $m(\text{hidden}_k) \in \{1, 2, \ldots, D-1\}$ (randomly assigned)
- **Output units**: $m(\text{output}_d) = d - 1$ (can depend on inputs 1 to d-1)

### Constructing Masks

For a connection from unit $j$ to unit $k$:

$$M_{kj} = \begin{cases} 
1 & \text{if } m(k) \geq m(j) \\
0 & \text{otherwise}
\end{cases}$$

For the output layer (strict inequality for autoregressive property):

$$M^{\text{out}}_{dj} = \begin{cases} 
1 & \text{if } m(\text{output}_d) \geq m(j) \\
0 & \text{otherwise}
\end{cases}$$

Since $m(\text{output}_d) = d - 1$, output $d$ receives information only from hidden units with degree $\leq d - 1$.

## Implementation

### Core MADE Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MaskedLinear(nn.Linear):
    """Linear layer with mask for autoregressive property."""
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))
    
    def set_mask(self, mask):
        self.mask.data.copy_(mask)
    
    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class MADE(nn.Module):
    """
    Masked Autoencoder for Distribution Estimation.
    
    Args:
        input_dim: Dimension of input/output
        hidden_dims: List of hidden layer dimensions
        num_outputs_per_dim: Number of outputs per input dimension
                            (e.g., 2 for Gaussian: mu and sigma)
        natural_ordering: If True, use 1,2,...,D ordering; else random
    """
    
    def __init__(self, input_dim, hidden_dims, num_outputs_per_dim=2, 
                 natural_ordering=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_outputs_per_dim = num_outputs_per_dim
        
        # Build network
        self.layers = nn.ModuleList()
        
        # Input -> First hidden
        self.layers.append(MaskedLinear(input_dim, hidden_dims[0]))
        
        # Hidden -> Hidden
        for i in range(len(hidden_dims) - 1):
            self.layers.append(MaskedLinear(hidden_dims[i], hidden_dims[i+1]))
        
        # Last hidden -> Output
        self.layers.append(MaskedLinear(
            hidden_dims[-1], 
            input_dim * num_outputs_per_dim
        ))
        
        # Create masks
        self.create_masks(natural_ordering)
    
    def create_masks(self, natural_ordering=True):
        """Create masks ensuring autoregressive property."""
        
        # Assign degrees to input units
        if natural_ordering:
            self.m_input = np.arange(1, self.input_dim + 1)
        else:
            self.m_input = np.random.permutation(self.input_dim) + 1
        
        # Assign degrees to hidden units
        self.m_hidden = []
        for dim in self.hidden_dims:
            # Random degrees in [1, input_dim - 1]
            m = np.random.randint(1, self.input_dim, size=dim)
            self.m_hidden.append(m)
        
        # Assign degrees to output units
        # Output d should depend on inputs 1, ..., d-1
        # So m(output_d) = d - 1, but we tile for multiple outputs per dim
        self.m_output = np.tile(self.m_input - 1, self.num_outputs_per_dim)
        
        # Create masks for each layer
        masks = []
        
        # First layer: input -> hidden[0]
        # hidden[0][k] receives from input[j] if m_hidden[0][k] >= m_input[j]
        mask = (self.m_hidden[0][:, None] >= self.m_input[None, :]).astype(np.float32)
        masks.append(mask)
        
        # Hidden -> hidden layers
        for i in range(len(self.hidden_dims) - 1):
            mask = (self.m_hidden[i+1][:, None] >= self.m_hidden[i][None, :]).astype(np.float32)
            masks.append(mask)
        
        # Last hidden -> output
        # Strict inequality for output layer
        mask = (self.m_output[:, None] >= self.m_hidden[-1][None, :]).astype(np.float32)
        masks.append(mask)
        
        # Set masks
        for layer, mask in zip(self.layers, masks):
            layer.set_mask(torch.from_numpy(mask))
    
    def forward(self, x):
        """
        Forward pass computing all conditional parameters.
        
        Returns:
            params: Shape (batch, input_dim, num_outputs_per_dim)
        """
        h = x
        for layer in self.layers[:-1]:
            h = F.relu(layer(h))
        
        # Output layer (no activation)
        out = self.layers[-1](h)
        
        # Reshape to (batch, input_dim, num_outputs_per_dim)
        out = out.view(x.shape[0], self.input_dim, self.num_outputs_per_dim)
        
        return out
```

### Gaussian MADE for Density Estimation

```python
class GaussianMADE(nn.Module):
    """MADE with Gaussian conditionals for density estimation."""
    
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.made = MADE(input_dim, hidden_dims, num_outputs_per_dim=2)
        self.input_dim = input_dim
    
    def get_params(self, x):
        """Get mu and sigma for all dimensions."""
        params = self.made(x)
        mu = params[:, :, 0]
        log_sigma = params[:, :, 1]
        return mu, log_sigma
    
    def log_prob(self, x):
        """Compute log p(x)."""
        mu, log_sigma = self.get_params(x)
        sigma = torch.exp(log_sigma)
        
        # Gaussian log probability
        log_p = -0.5 * (
            ((x - mu) / sigma) ** 2 
            + 2 * log_sigma 
            + np.log(2 * np.pi)
        )
        
        return log_p.sum(dim=-1)
    
    def sample(self, n_samples, device='cpu'):
        """
        Generate samples (sequential due to autoregressive nature).
        """
        samples = torch.zeros(n_samples, self.input_dim, device=device)
        
        for d in range(self.input_dim):
            # Get parameters for all dimensions
            params = self.made(samples)
            mu_d = params[:, d, 0]
            sigma_d = torch.exp(params[:, d, 1])
            
            # Sample dimension d
            samples[:, d] = mu_d + sigma_d * torch.randn(n_samples, device=device)
        
        return samples
```

## Training MADE

```python
def train_made(model, data, n_epochs=100, batch_size=128, lr=1e-3):
    """Train MADE on data."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    losses = []
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        for batch, in loader:
            optimizer.zero_grad()
            
            # Negative log-likelihood
            loss = -model.log_prob(batch).mean()
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch)
        
        avg_loss = epoch_loss / len(data)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, NLL: {avg_loss:.4f}")
    
    return losses


# Example usage
torch.manual_seed(42)

# Generate mixture of Gaussians data
n_samples = 5000
mixture = torch.distributions.MixtureSameFamily(
    torch.distributions.Categorical(torch.ones(4)),
    torch.distributions.MultivariateNormal(
        torch.tensor([[-2., -2.], [2., -2.], [-2., 2.], [2., 2.]]),
        torch.eye(2).unsqueeze(0).expand(4, -1, -1) * 0.3
    )
)
data = mixture.sample((n_samples,))

# Train MADE
model = GaussianMADE(input_dim=2, hidden_dims=[64, 64])
losses = train_made(model, data, n_epochs=200)
```

## Order Agnostic Training

### Motivation

MADE's performance depends on the input ordering. Different orderings capture different dependencies.

### Solution: Multiple Orderings

Train with random orderings and average at test time:

```python
class OrderAgnosticMADE(nn.Module):
    """MADE trained with multiple random orderings."""
    
    def __init__(self, input_dim, hidden_dims, num_masks=4):
        super().__init__()
        self.input_dim = input_dim
        self.num_masks = num_masks
        
        self.made = MADE(input_dim, hidden_dims, num_outputs_per_dim=2)
        
        # Store multiple mask configurations
        self.masks_list = []
        for _ in range(num_masks):
            self.made.create_masks(natural_ordering=False)
            masks = [layer.mask.clone() for layer in self.made.layers]
            self.masks_list.append(masks)
    
    def set_mask_index(self, idx):
        """Set masks to configuration idx."""
        masks = self.masks_list[idx]
        for layer, mask in zip(self.made.layers, masks):
            layer.set_mask(mask)
    
    def log_prob(self, x, mask_idx=None):
        """Compute log prob with specific or random mask."""
        if mask_idx is None:
            mask_idx = np.random.randint(self.num_masks)
        
        self.set_mask_index(mask_idx)
        params = self.made(x)
        
        mu = params[:, :, 0]
        log_sigma = params[:, :, 1]
        sigma = torch.exp(log_sigma)
        
        log_p = -0.5 * (
            ((x - mu) / sigma) ** 2 
            + 2 * log_sigma 
            + np.log(2 * np.pi)
        )
        
        return log_p.sum(dim=-1)
    
    def log_prob_ensemble(self, x):
        """Average log prob over all masks."""
        log_probs = []
        for idx in range(self.num_masks):
            log_probs.append(self.log_prob(x, mask_idx=idx))
        
        # Log-mean-exp for numerical stability
        log_probs = torch.stack(log_probs, dim=0)
        return torch.logsumexp(log_probs, dim=0) - np.log(self.num_masks)
```

## MADE as a Building Block for Flows

### From Density Estimator to Flow

MADE can be used as the **conditioner** in autoregressive flows:

```python
class MADEConditioner(nn.Module):
    """MADE that outputs transformation parameters."""
    
    def __init__(self, input_dim, hidden_dims, context_dim=0):
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        
        # Input includes context if provided
        full_input_dim = input_dim + context_dim
        
        self.made = MADE(
            full_input_dim, 
            hidden_dims, 
            num_outputs_per_dim=2
        )
        
        # Only output parameters for input_dim, not context
        # Adjust masks accordingly
        self._adjust_for_context()
    
    def _adjust_for_context(self):
        """Modify masks so context can influence all outputs."""
        if self.context_dim == 0:
            return
        
        # Context dimensions should connect to all hidden units
        for layer in self.made.layers:
            mask = layer.mask.clone()
            # Set context connections to 1
            mask[:, self.input_dim:] = 1
            layer.set_mask(mask)
    
    def forward(self, x, context=None):
        """
        Get transformation parameters.
        
        Args:
            x: Input tensor (batch, input_dim)
            context: Optional context (batch, context_dim)
        
        Returns:
            scale, shift: Each (batch, input_dim)
        """
        if context is not None:
            x = torch.cat([x, context], dim=-1)
        
        params = self.made(x)
        
        # Only take parameters for input dimensions
        params = params[:, :self.input_dim, :]
        
        scale = params[:, :, 0]
        shift = params[:, :, 1]
        
        return scale, shift
```

## Connectivity Constraints

### Minimum Connectivity

To ensure information flow, every hidden unit should connect to:
- At least one input
- At least one output

This is achieved by constraining degrees:
- Hidden units: $m(k) \in \{1, \ldots, D-1\}$
- Not $m(k) = 0$ (would receive no input)
- Not $m(k) = D$ (would not contribute to any output)

### Direct Input-Output Connections

MADE can include direct connections from inputs to outputs:

```python
class MADEWithDirect(nn.Module):
    """MADE with direct input-output connections."""
    
    def __init__(self, input_dim, hidden_dims, num_outputs_per_dim=2):
        super().__init__()
        
        self.made = MADE(input_dim, hidden_dims, num_outputs_per_dim)
        
        # Direct connection (also masked)
        self.direct = MaskedLinear(input_dim, input_dim * num_outputs_per_dim)
        
        # Mask for direct connection
        # Output d depends on inputs 1, ..., d-1
        mask = np.tril(np.ones((input_dim, input_dim)), k=-1)
        mask = np.tile(mask, (num_outputs_per_dim, 1))
        self.direct.set_mask(torch.from_numpy(mask.astype(np.float32)))
    
    def forward(self, x):
        hidden_out = self.made(x).view(x.shape[0], -1)
        direct_out = self.direct(x)
        return (hidden_out + direct_out).view(x.shape[0], -1, 2)
```

## Advantages and Limitations

### Advantages

1. **Single forward pass** for density evaluation
2. **Exact likelihood** computation
3. **Flexible architecture** - any number of hidden layers
4. **GPU-friendly** - standard matrix operations
5. **Foundation for flows** - enables efficient autoregressive flows

### Limitations

1. **Sequential sampling** - still O(D) steps to generate
2. **Fixed ordering** - performance depends on dimension order
3. **Expressive power** - limited by mask constraints
4. **Sparsity overhead** - many zero weights due to masking

## Comparison with Other Approaches

| Approach | Density Eval | Sampling | Parameters |
|----------|-------------|----------|------------|
| Naive AR | O(D) passes | O(D) | D networks |
| MADE | O(1) pass | O(D) | 1 network |
| RNN | O(D) steps | O(D) | 1 network |
| Coupling Flow | O(1) pass | O(1) | Multiple layers |

## Summary

MADE provides:

1. **Efficient autoregressive modeling** through weight masking
2. **Single-pass density evaluation** while maintaining autoregressive property
3. **Foundation for MAF** - MADE as conditioner in normalizing flows
4. **Order-agnostic training** for robustness

The masking insight—that careful zero-patterns in weights can enforce autoregressive structure—revolutionized efficient density estimation and enabled practical autoregressive flows.

## References

1. Germain, M., et al. (2015). MADE: Masked Autoencoder for Distribution Estimation. *ICML*.
2. Papamakarios, G., et al. (2017). Masked Autoregressive Flow for Density Estimation. *NeurIPS*.
3. Uria, B., et al. (2016). Neural Autoregressive Distribution Estimation. *JMLR*.
