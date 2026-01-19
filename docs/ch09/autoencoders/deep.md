# Deep (Stacked) Autoencoder

Learn hierarchical feature representations through multiple encoding/decoding layers.

---

## Overview

**Key Concepts:**

- Stacking multiple layers for hierarchical representations
- Greedy layer-wise pretraining (historical approach)
- Deep bottleneck architectures
- Hierarchical feature extraction
- Comparison with shallow autoencoders

**Time:** ~50 minutes  
**Level:** Intermediate-Advanced

---

## Mathematical Foundation

### Shallow vs Deep Architecture

| Type | Data Flow |
|------|-----------|
| Shallow | $x \to h \to z \to h' \to \hat{x}$ |
| Deep | $x \to h_1 \to h_2 \to \cdots \to z \to \cdots \to h'_2 \to h'_1 \to \hat{x}$ |

### Hierarchical Feature Learning

Each layer $h_i$ represents progressively more abstract features:

- $h_1$: Low-level features (edges, textures)
- $h_2$: Mid-level features (parts, patterns)
- $z$: High-level abstract representation
- Decoder mirrors encoder in reverse

### Benefits of Depth

1. **Hierarchical feature learning** — captures features at multiple abstraction levels
2. **More compact representations** — aggressive compression possible
3. **Better expressivity** — with fewer neurons per layer
4. **Complex non-linear mappings** — can model intricate data distributions

### Historical Note

Before modern optimization techniques (ReLU, batch norm, Adam), deep autoencoders required **greedy layer-wise pretraining**. Modern approaches typically don't need this, but it's valuable to understand.

---

## Part 1: Deep Autoencoder Architecture

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

class DeepAutoencoder(nn.Module):
    """
    Deep autoencoder with multiple encoding and decoding layers.
    
    Architecture: 784 → 512 → 256 → 128 → 32 (bottleneck)
                  32 → 128 → 256 → 512 → 784
    
    Creates a narrow bottleneck with aggressive compression (24.5x).
    """
    
    def __init__(self):
        super(DeepAutoencoder, self).__init__()
        
        # Encoder: Progressive dimensionality reduction
        self.encoder = nn.Sequential(
            # Layer 1: 784 → 512
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            
            # Layer 2: 512 → 256
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            # Layer 3: 256 → 128
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            # Bottleneck: 128 → 32
            nn.Linear(128, 32),
            nn.ReLU()
        )
        
        # Decoder: Mirror of encoder
        self.decoder = nn.Sequential(
            # Expand from bottleneck: 32 → 128
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            # Layer 3: 128 → 256
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            # Layer 2: 256 → 512
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            
            # Output: 512 → 784
            nn.Linear(512, 784),
            nn.Sigmoid()
        )
        
        self.latent_dim = 32
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def get_layer_outputs(self, x):
        """Get outputs from all encoder layers for visualization."""
        activations = []
        h = x
        
        for layer in self.encoder:
            h = layer(h)
            if isinstance(layer, nn.ReLU):
                activations.append(h.detach())
        
        return activations
```

---

## Part 2: Very Deep Autoencoder

```python
class VeryDeepAutoencoder(nn.Module):
    """
    Very deep autoencoder with 6+ encoding/decoding layers.
    
    Architecture: 784 → 512 → 384 → 256 → 128 → 64 → 16 (bottleneck)
    
    Demonstrates that with proper regularization, very deep
    architectures can be trained end-to-end.
    """
    
    def __init__(self):
        super(VeryDeepAutoencoder, self).__init__()
        
        self.latent_dim = 16
        
        # Encoder
        self.enc1 = self._make_layer(784, 512)
        self.enc2 = self._make_layer(512, 384)
        self.enc3 = self._make_layer(384, 256)
        self.enc4 = self._make_layer(256, 128)
        self.enc5 = self._make_layer(128, 64)
        self.enc6 = nn.Sequential(nn.Linear(64, 16), nn.ReLU())
        
        # Decoder (mirror)
        self.dec6 = self._make_layer(16, 64)
        self.dec5 = self._make_layer(64, 128)
        self.dec4 = self._make_layer(128, 256)
        self.dec3 = self._make_layer(256, 384)
        self.dec2 = self._make_layer(384, 512)
        self.dec1 = nn.Sequential(nn.Linear(512, 784), nn.Sigmoid())
    
    def _make_layer(self, in_dim, out_dim):
        """Create a layer block with normalization and dropout."""
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(0.3)  # Higher dropout for very deep networks
        )
    
    def forward(self, x):
        # Encoder
        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        h3 = self.enc3(h2)
        h4 = self.enc4(h3)
        h5 = self.enc5(h4)
        z = self.enc6(h5)
        
        # Decoder
        d6 = self.dec6(z)
        d5 = self.dec5(d6)
        d4 = self.dec4(d5)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        x_recon = self.dec1(d2)
        
        return x_recon, z
```

---

## Part 3: Layer-wise Pretraining (Historical Approach)

Before modern optimization, deep autoencoders were trained using **greedy layer-wise pretraining**.

### Process

1. Train first autoencoder: $x \to h_1 \to x$
2. Fix encoder₁, train second: $h_1 \to h_2 \to h_1$
3. Continue for all layers
4. Stack all encoders, fine-tune end-to-end

```python
class StackedAutoencoder:
    """
    Greedy layer-wise pretraining for deep autoencoders.
    Historical approach (2006-2012) before modern optimization.
    """
    
    def __init__(self, layer_dims):
        """
        Parameters:
        -----------
        layer_dims : List[int]
            Dimensions for each layer, e.g., [784, 512, 256, 128, 32]
        """
        self.layer_dims = layer_dims
        self.autoencoders = []
        self.encoders = []
        
        # Create autoencoder for each layer pair
        for i in range(len(layer_dims) - 1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]
            
            ae = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, in_dim),
                nn.Sigmoid() if i == 0 else nn.ReLU()
            )
            self.autoencoders.append(ae)
    
    def pretrain_layer(self, layer_idx, data_loader, device, epochs=5):
        """Pretrain a single layer."""
        ae = self.autoencoders[layer_idx].to(device)
        optimizer = optim.Adam(ae.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        print(f"\nPretraining layer {layer_idx + 1}...")
        
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            for images, _ in data_loader:
                images = images.view(images.size(0), -1).to(device)
                
                # If not first layer, encode through previous layers
                if layer_idx > 0:
                    with torch.no_grad():
                        for prev_ae in self.autoencoders[:layer_idx]:
                            encoder = nn.Sequential(*list(prev_ae.children())[:2])
                            images = encoder(images)
                
                # Train current autoencoder
                optimizer.zero_grad()
                reconstructed = ae(images)
                loss = criterion(reconstructed, images)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {total_loss / num_batches:.6f}")
        
        # Extract and save encoder
        encoder = nn.Sequential(*list(ae.children())[:2])
        self.encoders.append(encoder)
```

---

## Part 4: Visualization of Hierarchical Features

```python
def visualize_hierarchical_features(model, test_loader, device):
    """
    Visualize activations at different layers of deep autoencoder.
    Shows how representations become more abstract in deeper layers.
    """
    model.eval()
    
    images, labels = next(iter(test_loader))
    image = images[0:1].view(1, -1).to(device)
    label = labels[0].item()
    
    with torch.no_grad():
        layer_outputs = model.get_layer_outputs(image)
        reconstructed, _ = model(image)
    
    num_layers = len(layer_outputs) + 2  # +2 for original and reconstructed
    fig, axes = plt.subplots(1, num_layers, figsize=(3 * num_layers, 3))
    
    # Original image
    axes[0].imshow(image.cpu().reshape(28, 28), cmap='gray')
    axes[0].set_title(f'Input (digit {label})')
    axes[0].axis('off')
    
    # Layer activations
    for i, activation in enumerate(layer_outputs):
        act_np = activation.cpu().numpy().flatten()
        size = int(np.ceil(np.sqrt(len(act_np))))
        padded = np.zeros(size * size)
        padded[:len(act_np)] = act_np
        act_2d = padded.reshape(size, size)
        
        axes[i + 1].imshow(act_2d, cmap='viridis')
        axes[i + 1].set_title(f'Layer {i + 1} ({len(act_np)} dim)')
        axes[i + 1].axis('off')
    
    # Reconstructed
    axes[-1].imshow(reconstructed.cpu().reshape(28, 28), cmap='gray')
    axes[-1].set_title('Reconstructed')
    axes[-1].axis('off')
    
    plt.suptitle('Hierarchical Feature Representations')
    plt.savefig('hierarchical_features.png', dpi=150)
    plt.show()
```

---

## Part 5: Training Deep Autoencoders

```python
def train_deep_autoencoder(model, train_loader, criterion, optimizer, device, epoch):
    """Train deep autoencoder for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.view(images.size(0), -1).to(device)
        
        optimizer.zero_grad()
        reconstructed, _ = model(images)
        loss = criterion(reconstructed, images)
        loss.backward()
        
        # Gradient clipping for deep networks
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches
```

### Key Training Techniques for Deep Networks

1. **Gradient clipping** — prevents exploding gradients
2. **Batch normalization** — stabilizes training
3. **Dropout** — regularization for deep networks
4. **Learning rate scheduling** — adapts LR during training

---

## Part 6: Main Execution

```python
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 15
    
    # Load data
    train_loader, test_loader = load_mnist_data(batch_size)
    
    # Initialize model
    model = DeepAutoencoder().to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        train_loss = train_deep_autoencoder(
            model, train_loader, criterion, optimizer, device, epoch
        )
        print(f"Epoch {epoch} - Loss: {train_loss:.6f}")
        
        scheduler.step(train_loss)
        
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), 'deep_autoencoder_best.pth')
    
    # Visualizations
    visualize_hierarchical_features(model, test_loader, device)

if __name__ == "__main__":
    main()
```

---

## Exercises

### Exercise 1: Depth vs Performance

Train autoencoders with different depths:

- **Shallow:** 784 → 128 → 32 → 128 → 784
- **Medium:** 784 → 512 → 256 → 32 → 256 → 512 → 784
- **Deep:** 784 → 512 → 384 → 256 → 128 → 32 → ... (mirror)
- **Very Deep:** 6-8 layers each side

**Questions:**
- Is deeper always better?
- What's the optimal depth for MNIST?

### Exercise 2: Width vs Depth Trade-off

Compare with similar parameter counts:

- **Wide & Shallow:** 784 → 1024 → 32 → 1024 → 784
- **Narrow & Deep:** 784 → 256 → 128 → 64 → 32 → 64 → 128 → 256 → 784

### Exercise 3: Layer-wise Pretraining

Compare:

a) End-to-end training (modern approach)  
b) Greedy layer-wise pretraining (historical)

**Questions:**
- Is pretraining still beneficial?
- When might it be necessary?

### Exercise 4: Residual Connections

Add skip connections: $h_{i+1} = f(h_i) + h_i$

**Questions:**
- Do residuals improve deep network training?
- How deep can you go with residuals?

### Exercise 5: Bottleneck Analysis

Fix depth, vary bottleneck size:

```python
bottleneck_dims = [8, 16, 32, 64, 128, 256]
```

**Questions:**
- How does bottleneck size affect deep networks differently than shallow?
- What's the minimum viable bottleneck?

---

## Summary

| Aspect | Shallow AE | Deep AE |
|--------|------------|---------|
| Layers | 1-2 per side | 3+ per side |
| Features | Single level | Hierarchical |
| Compression | Limited | Aggressive |
| Training | Easy | Requires regularization |
| Expressivity | Limited | High |

**Key Insight:** Deep autoencoders learn hierarchical representations where each layer captures increasingly abstract features. Modern techniques (batch norm, dropout, Adam) enable end-to-end training without the historical need for layer-wise pretraining.
