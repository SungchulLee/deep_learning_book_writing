# Maximum Likelihood Training

## Introduction

Normalizing flows are trained by **maximum likelihood estimation (MLE)**—directly maximizing the probability of the training data under the model. This is in contrast to VAEs (which maximize a lower bound) or GANs (which use adversarial training). The exact likelihood computation is a key advantage of normalizing flows.

## The Training Objective

### Log-Likelihood

Given data $\{x_1, \ldots, x_N\}$, we maximize:

$$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \log p_\theta(x_i)$$

For a normalizing flow with transformation $f_\theta$:

$$\log p_\theta(x) = \log p_Z(f_\theta^{-1}(x)) + \log \left| \det \frac{\partial f_\theta^{-1}}{\partial x} \right|$$

### Loss Function

In practice, we minimize the **negative log-likelihood (NLL)**:

$$\text{Loss} = -\frac{1}{N} \sum_{i=1}^{N} \log p_\theta(x_i) = -\frac{1}{N} \sum_{i=1}^{N} \left[ \log p_Z(z_i) + \log \left| \det J_i \right| \right]$$

where $z_i = f_\theta^{-1}(x_i)$ and $J_i$ is the Jacobian.

## Training Algorithm

### Basic Training Loop

```python
def train_flow(
    flow_model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    device: str = 'cpu'
) -> List[float]:
    """
    Train a normalizing flow with maximum likelihood.
    
    Args:
        flow_model: Flow model with log_prob method
        train_loader: DataLoader for training data
        optimizer: PyTorch optimizer
        n_epochs: Number of training epochs
        device: Computation device
    
    Returns:
        List of training losses per epoch
    """
    flow_model.train()
    losses = []
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            # Handle different data formats
            if isinstance(batch, (tuple, list)):
                batch = batch[0]  # Ignore labels
            
            batch = batch.to(device)
            
            # Flatten if needed (e.g., images)
            if batch.dim() > 2:
                batch = batch.view(batch.size(0), -1)
            
            # Forward pass: compute log probability
            log_prob = flow_model.log_prob(batch)
            
            # Loss: negative log-likelihood
            loss = -log_prob.mean()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
    
    return losses
```

### Understanding the Gradient

The gradient of the loss with respect to flow parameters $\theta$:

$$\nabla_\theta \text{Loss} = -\frac{1}{N} \sum_{i=1}^{N} \nabla_\theta \log p_\theta(x_i)$$

This involves:
1. Gradient through the log-probability of the base distribution
2. Gradient through the log-determinant of the Jacobian

Both are computed automatically by PyTorch's autograd.

## Training Stability

### Gradient Clipping

Large gradients can destabilize training:

```python
def train_with_gradient_clipping(
    flow_model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    max_grad_norm: float = 1.0,
    device: str = 'cpu'
) -> List[float]:
    """Train with gradient clipping for stability."""
    
    flow_model.train()
    losses = []
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            batch = batch.to(device)
            
            if batch.dim() > 2:
                batch = batch.view(batch.size(0), -1)
            
            log_prob = flow_model.log_prob(batch)
            loss = -log_prob.mean()
            
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                flow_model.parameters(), 
                max_grad_norm
            )
            
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
    
    return losses
```

### Warm-up Period

Gradually increase learning rate at the start:

```python
def get_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int
):
    """Create a learning rate scheduler with warmup."""
    
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Constant or decay after warmup
            return 1.0
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### Numerical Stability

Handle numerical issues in log-probabilities:

```python
def stable_log_prob(
    flow_model: nn.Module,
    x: torch.Tensor,
    min_log_prob: float = -1e6
) -> torch.Tensor:
    """Compute log probability with numerical safeguards."""
    
    # Inverse transformation
    z, log_det = flow_model.inverse(x)
    
    # Base log probability
    log_pz = flow_model.base_dist.log_prob(z)
    
    # Total log probability
    log_px = log_pz + log_det
    
    # Clamp to avoid -inf
    log_px = torch.clamp(log_px, min=min_log_prob)
    
    # Check for NaN
    if torch.isnan(log_px).any():
        print("Warning: NaN in log probability")
        log_px = torch.where(
            torch.isnan(log_px),
            torch.full_like(log_px, min_log_prob),
            log_px
        )
    
    return log_px
```

## Optimization Strategies

### Optimizer Selection

Adam is the standard choice for normalizing flows:

```python
# Standard configuration
optimizer = torch.optim.Adam(
    flow_model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8
)

# For more stability
optimizer = torch.optim.AdamW(
    flow_model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)
```

### Learning Rate Scheduling

```python
# Cosine annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=n_epochs,
    eta_min=1e-6
)

# Reduce on plateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=10,
    verbose=True
)
```

### Learning Rate Recommendations

| Dataset | Starting LR | Batch Size |
|---------|-------------|------------|
| 2D toy | 1e-3 | 256 |
| MNIST | 1e-4 | 128 |
| CIFAR-10 | 1e-4 | 64 |
| High-res images | 1e-5 | 32 |

## Advanced Training Techniques

### Data Augmentation

For image data, augmentation can improve generalization:

```python
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # Dequantization
    transforms.Lambda(lambda x: (x + torch.rand_like(x) / 256)),
])
```

**Note**: Be careful with augmentations that change the distribution (e.g., rotations may not make sense for certain datasets).

### Multi-Scale Training

For high-resolution images, use progressive training:

```python
class MultiScaleTrainer:
    """Progressive training from low to high resolution."""
    
    def __init__(self, flow_model, resolutions=[8, 16, 32, 64]):
        self.flow_model = flow_model
        self.resolutions = resolutions
    
    def train(self, dataset, epochs_per_scale=50):
        for res in self.resolutions:
            print(f"Training at resolution {res}x{res}")
            
            # Resize data
            scaled_data = self.resize_data(dataset, res)
            loader = DataLoader(scaled_data, batch_size=64, shuffle=True)
            
            # Train at this scale
            optimizer = torch.optim.Adam(self.flow_model.parameters(), lr=1e-4)
            train_flow(self.flow_model, loader, optimizer, epochs_per_scale)
```

### Regularization

#### Weight Decay

```python
optimizer = torch.optim.AdamW(
    flow_model.parameters(),
    lr=1e-4,
    weight_decay=1e-5  # L2 regularization
)
```

#### Jacobian Regularization

Encourage well-conditioned Jacobians:

```python
def jacobian_regularized_loss(
    flow_model: nn.Module,
    x: torch.Tensor,
    lambda_reg: float = 0.01
) -> torch.Tensor:
    """Loss with Jacobian regularization."""
    
    # Standard NLL
    log_prob = flow_model.log_prob(x)
    nll = -log_prob.mean()
    
    # Jacobian regularization (penalize extreme log-dets)
    _, log_det = flow_model.inverse(x)
    jac_reg = (log_det ** 2).mean()
    
    return nll + lambda_reg * jac_reg
```

## Monitoring Training

### Metrics to Track

```python
class TrainingMonitor:
    """Monitor training metrics."""
    
    def __init__(self, flow_model, val_loader, device='cpu'):
        self.flow_model = flow_model
        self.val_loader = val_loader
        self.device = device
        self.history = []
    
    def log_metrics(self, epoch, train_loss):
        """Log training and validation metrics."""
        
        self.flow_model.eval()
        
        # Validation loss
        val_log_probs = []
        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, (tuple, list)):
                    batch = batch[0]
                batch = batch.to(self.device)
                if batch.dim() > 2:
                    batch = batch.view(batch.size(0), -1)
                
                log_prob = self.flow_model.log_prob(batch)
                val_log_probs.append(log_prob)
        
        val_log_prob = torch.cat(val_log_probs).mean().item()
        val_loss = -val_log_prob
        
        # Bits per dimension
        dim = next(iter(self.val_loader))[0].numel() // next(iter(self.val_loader))[0].size(0)
        bpd = val_loss / (dim * np.log(2))
        
        # Sample quality (check for NaN)
        samples = self.flow_model.sample(100, device=self.device)
        nan_ratio = torch.isnan(samples).float().mean().item()
        
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'bpd': bpd,
            'nan_ratio': nan_ratio
        }
        
        self.history.append(metrics)
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
              f"Val Loss={val_loss:.4f}, BPD={bpd:.3f}, NaN={nan_ratio:.2%}")
        
        self.flow_model.train()
        
        return metrics
```

### Early Stopping

```python
class EarlyStopping:
    """Stop training when validation loss stops improving."""
    
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
    
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Save best model
            self.best_state = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
```

## Complete Training Pipeline

```python
def train_flow_complete(
    flow_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 100,
    lr: float = 1e-4,
    device: str = 'cpu',
    checkpoint_path: str = 'checkpoints/'
) -> Dict:
    """Complete training pipeline with all best practices."""
    
    import os
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Setup
    flow_model = flow_model.to(device)
    optimizer = torch.optim.AdamW(flow_model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    early_stopping = EarlyStopping(patience=20)
    monitor = TrainingMonitor(flow_model, val_loader, device)
    
    best_loss = float('inf')
    
    for epoch in range(1, n_epochs + 1):
        flow_model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            batch = batch.to(device)
            if batch.dim() > 2:
                batch = batch.view(batch.size(0), -1)
            
            # Forward
            log_prob = flow_model.log_prob(batch)
            loss = -log_prob.mean()
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow_model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        # Logging
        train_loss = epoch_loss / n_batches
        metrics = monitor.log_metrics(epoch, train_loss)
        
        # Checkpointing
        if metrics['val_loss'] < best_loss:
            best_loss = metrics['val_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': flow_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(checkpoint_path, 'best_model.pt'))
        
        # Early stopping
        if early_stopping(metrics['val_loss'], flow_model):
            print(f"Early stopping at epoch {epoch}")
            # Load best model
            flow_model.load_state_dict(early_stopping.best_state)
            break
    
    return {
        'history': monitor.history,
        'best_loss': best_loss,
        'final_epoch': epoch
    }
```

## Common Issues and Solutions

### Issue: Loss Diverges (Goes to Infinity)

**Causes**:
- Learning rate too high
- Numerical instability in Jacobian

**Solutions**:
- Reduce learning rate
- Add gradient clipping
- Check for NaN in forward pass

### Issue: Loss Plateaus Early

**Causes**:
- Model too simple
- Learning rate too low
- Poor initialization

**Solutions**:
- Add more layers
- Try learning rate warmup
- Use proper weight initialization

### Issue: Good Training Loss, Bad Samples

**Causes**:
- Overfitting
- Mode collapse
- Numerical issues in sampling

**Solutions**:
- More regularization
- Check forward pass separately
- Verify invertibility

## Summary

Maximum likelihood training for normalizing flows:

1. **Objective**: Minimize negative log-likelihood
2. **Gradient**: Computed through inverse pass and log-determinant
3. **Stability**: Use gradient clipping, proper learning rates
4. **Monitoring**: Track both training and validation loss, BPD
5. **Best Practices**: Early stopping, checkpointing, learning rate scheduling

The direct MLE training without approximations is a key advantage of normalizing flows over other generative models.

## References

1. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.
2. Kingma, D. P., & Dhariwal, P. (2018). Glow: Generative Flow with Invertible 1×1 Convolutions. *NeurIPS*.
3. Dinh, L., et al. (2017). Density Estimation Using Real-NVP. *ICLR*.
