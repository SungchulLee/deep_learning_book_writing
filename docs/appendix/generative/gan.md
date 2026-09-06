# GAN

GAN was introduced in the 2014 paper "Generative Adversarial Networks." Two networks (Generator and Discriminator) in adversarial training.

This implementation provides a concise, educational reference for GAN. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## Code

```python
#!/usr/bin/env python3
'''
GAN - Generative Adversarial Networks
Paper: "Generative Adversarial Networks" (2014)
Key: Two networks (Generator and Discriminator) in adversarial training
'''
import torch
import torch.nn as nn

# ========================================================================
# Main
# ========================================================================

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_size=28):
        super().__init__()
        self.img_size = img_size
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, img_size * img_size * 1),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, self.img_size, self.img_size)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_size=28):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(img_size * img_size * 1, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

class GAN(nn.Module):
    def __init__(self, latent_dim=100, img_size=28):
        super().__init__()
        self.generator = Generator(latent_dim, img_size)
        self.discriminator = Discriminator(img_size)
    
    def forward(self, z):
        return self.generator(z)

# ---------------------------------------------------------------------------
# GAN Training Loop
# ---------------------------------------------------------------------------
# The minimax game alternates between:
#   1. Update D: maximize log D(x) + log(1 - D(G(z)))
#   2. Update G: minimize log(1 - D(G(z)))  [or equivalently maximize log D(G(z))]
# In practice, step 2 uses the "non-saturating" loss: -log(D(G(z)))
# because it provides stronger gradients early in training.

def update_D(X, Z, net_D, net_G, loss, trainer_D):
    """Update discriminator: maximize log D(x) + log(1 - D(G(z)))."""
    batch_size = X.shape[0]
    ones = torch.ones(batch_size, 1, device=X.device)
    zeros = torch.zeros(batch_size, 1, device=X.device)
    trainer_D.zero_grad()
    real_Y = net_D(X)
    fake_X = net_G(Z)
    fake_Y = net_D(fake_X.detach())  # detach so G is not updated
    loss_D = loss(real_Y, ones) + loss(fake_Y, zeros)
    loss_D.backward()
    trainer_D.step()
    return float(loss_D)


def update_G(Z, net_D, net_G, loss, trainer_G):
    """Update generator: maximize log D(G(z))  (non-saturating loss)."""
    batch_size = Z.shape[0]
    ones = torch.ones(batch_size, 1, device=Z.device)
    trainer_G.zero_grad()
    fake_X = net_G(Z)
    fake_Y = net_D(fake_X)
    loss_G = loss(fake_Y, ones)  # fool D into classifying fake as real
    loss_G.backward()
    trainer_G.step()
    return float(loss_G)


def train_gan(net_G, net_D, data_iter, num_epochs, latent_dim,
              lr_D=0.0002, lr_G=0.0002, device='cpu'):
    """Full GAN training loop with loss tracking."""
    loss = nn.BCELoss()
    net_G, net_D = net_G.to(device), net_D.to(device)
    trainer_D = torch.optim.Adam(net_D.parameters(), lr=lr_D, betas=(0.5, 0.999))
    trainer_G = torch.optim.Adam(net_G.parameters(), lr=lr_G, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        d_loss_sum, g_loss_sum, n = 0, 0, 0
        for X, _ in data_iter:
            X = X.to(device)
            batch_size = X.shape[0]
            Z = torch.randn(batch_size, latent_dim, device=device)
            d_loss_sum += update_D(X, Z, net_D, net_G, loss, trainer_D)
            g_loss_sum += update_G(Z, net_D, net_G, loss, trainer_G)
            n += 1
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"D loss: {d_loss_sum/n:.4f}, G loss: {g_loss_sum/n:.4f}")


if __name__ == "__main__":
    model = GAN()
    print(f"Generator Parameters: {sum(p.numel() for p in model.generator.parameters()):,}")
    print(f"Discriminator Parameters: {sum(p.numel() for p in model.discriminator.parameters()):,}")```

## Discussion

The implementation defines 3 classes (`Generator`, `Discriminator`, `GAN`) that work together to form the complete generative model architecture. Each class encapsulates a distinct component, making the code modular and easy to extend. The `forward` methods define the computational graph that PyTorch uses for automatic differentiation.

The training loop follows the standard PyTorch pattern: forward pass to compute predictions, loss calculation, backward pass for gradient computation, and parameter update via the optimizer. Tracking metrics across epochs reveals the convergence behavior and helps diagnose issues like underfitting or overfitting.

As a reference implementation, this code prioritizes clarity over optimization. Production systems would typically add mixed-precision training, distributed data parallelism, and more sophisticated data augmentation. Nevertheless, the core architectural ideas shown here remain the same regardless of scale.

## Exercises

**Exercise 1.**
Calculate the total number of learnable parameters in `Generator` with the default initialization. Break down the count by layer, including both weights and biases.

??? success "Solution to Exercise 1"
    For each `nn.Linear(in_features, out_features)`, there are `in_features * out_features` weight parameters plus `out_features` bias parameters (unless `bias=False`). For `nn.Conv2d(in_c, out_c, k)`, there are `in_c * out_c * k * k` weight parameters plus `out_c` bias parameters. For `nn.Embedding(num, dim)`, there are `num * dim` parameters. Sum across all layers. You can verify with `sum(p.numel() for p in model.parameters())`.

---

**Exercise 2.**
Replace the optimizer with Adam (use `torch.optim.Adam` with `lr=0.001`) and compare the training convergence with the original optimizer. Plot the loss curves for both on the same graph.

??? success "Solution to Exercise 2"
    Replace the optimizer creation line with `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`. Adam typically converges faster in early epochs because it maintains per-parameter adaptive learning rates and momentum estimates. The loss curve with Adam usually drops more steeply in the first few epochs but may oscillate slightly more than SGD with momentum near the optimum. For a fair comparison, run both with the same random seed and number of epochs.

---

**Exercise 3.**
Describe two potential failure modes of this implementation and explain how you would diagnose and fix each one.

??? success "Solution to Exercise 3"
    Common failure modes include: (1) **Vanishing/exploding gradients** -- diagnosed by monitoring gradient norms (`torch.nn.utils.clip_grad_norm_` or logging `param.grad.norm()` per layer). Fix with gradient clipping, better initialization (Xavier/Kaiming), or architectural changes (residual connections, normalization). (2) **Overfitting** -- diagnosed when training loss decreases but validation loss increases. Fix with regularization (dropout, weight decay, data augmentation) or reducing model capacity. Always monitor both training and validation metrics to catch these issues early.

---

**Exercise 4.**
Extend `Generator` to support a configurable number of layers or blocks. Add a `num_layers` parameter to `__init__` and use `nn.ModuleList` to create a variable-depth architecture. Test with 2, 4, and 8 layers.

??? success "Solution to Exercise 4"
    Replace the hardcoded layers with:
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    In the `forward` method, iterate: `for layer in self.layers: x = layer(x)`. Using `nn.ModuleList` (not a plain Python list) ensures PyTorch registers all parameters for optimization. Test with: `for n in [2, 4, 8]: model = Generator(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
