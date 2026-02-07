# Autoencoder Training

Training procedures, evaluation methods, analysis tools, and practical applications for autoencoders.

---

## Overview

**What you'll learn:**

- Training loops for standard and specialized autoencoders
- Evaluation and visualization techniques
- Latent space analysis: geometry, arithmetic, interpolation
- Failure mode identification and debugging
- Applications: anomaly detection, compression, clustering, transfer learning

---

## Part 1: Standard Training Loop

### Training and Evaluation Functions

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.utils import save_image


def loss_function(recon_x, x):
    """
    Binary Cross-Entropy reconstruction loss.
    
    BCE is appropriate when output uses sigmoid activation
    and pixels are in [0, 1] range.
    """
    return F.binary_cross_entropy(
        recon_x, x.view(-1, 784), reduction='sum'
    )


def train(epoch, model, train_loader, optimizer, device, log_interval=10):
    """
    Train for one epoch.
    
    Training loop:
    1. Forward pass: x -> encoder -> z -> decoder -> x_hat
    2. Compute BCE loss between x and x_hat
    3. Backward pass: compute gradients
    4. Update weights with optimizer
    """
    model.train()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        
        optimizer.zero_grad()
        recon_batch = model(data)
        loss = loss_function(recon_batch, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} '
                  f'[{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item() / len(data):.6f}')

    avg_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
    return avg_loss


def test(epoch, model, test_loader, device, save_path='./reconstructed_images'):
    """Evaluate on test set and save reconstruction visualizations."""
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch = model(data)
            test_loss += loss_function(recon_batch, data).item()

            if i == 0:
                n = min(data.size(0), 8)
                batch_size = data.size(0)
                comparison = torch.cat([
                    data[:n],
                    recon_batch.view(batch_size, 1, 28, 28)[:n]
                ])
                save_image(
                    comparison.cpu(),
                    f'{save_path}/reconstruction_{epoch}.png',
                    nrow=n
                )

    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')
    return test_loss
```

### Main Entry Point

```python
import torch.optim as optim
from load_data import load_data
from model import AE


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader, test_loader = load_data()
    model = AE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"Training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(1, 11):
        train(epoch, model, train_loader, optimizer, device)
        test(epoch, model, test_loader, device)
        
        # Generate samples by decoding random latent vectors
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       f'./generated_images/sample_{epoch}.png')

    torch.save(model.state_dict(), 'autoencoder_final.pth')


if __name__ == "__main__":
    main()
```

---

## Part 2: Denoising Autoencoder Training

The key difference: input is corrupted, but loss is computed against **clean** images.

```python
def train_denoising(epoch, model, train_loader, optimizer, criterion,
                    device, noise_factor=0.1, log_interval=10):
    """
    Train denoising autoencoder for one epoch.
    
    CRITICAL: Loss is between reconstruction and CLEAN images,
    NOT between reconstruction and noisy images!
    """
    model.train()
    train_loss = 0

    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.to(device)
        
        # Create noisy version
        noise = torch.randn_like(images) * noise_factor
        noisy_images = torch.clamp(images + noise, 0.0, 1.0)
        
        optimizer.zero_grad()
        recon_images = model(noisy_images)       # Forward on NOISY
        loss = criterion(recon_images, images)    # Loss against CLEAN
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} '
                  f'[{batch_idx * len(images)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item() / len(images):.6f}')

    avg_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
    return avg_loss
```

### Comparing Noise Levels at Test Time

```python
import matplotlib.pyplot as plt
import numpy as np


def compare_noise_levels(model, test_loader, device):
    """Test generalization across different noise levels."""
    model.eval()
    
    images, _ = next(iter(test_loader))
    image = images[0:1].to(device)
    
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
    
    fig, axes = plt.subplots(2, len(noise_levels), figsize=(15, 5))
    
    with torch.no_grad():
        for i, nf in enumerate(noise_levels):
            noise = torch.randn_like(image) * nf
            noisy = torch.clamp(image + noise, 0.0, 1.0)
            denoised = model(noisy)
            
            axes[0, i].imshow(noisy.cpu().squeeze(), cmap='gray')
            axes[0, i].set_title(f'sigma={nf}')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(denoised.cpu().squeeze(), cmap='gray')
            axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel('Noisy')
    axes[1, 0].set_ylabel('Denoised')
    plt.tight_layout()
    plt.savefig('noise_level_comparison.png', dpi=150)
    plt.show()
```

---

## Part 3: Deep Autoencoder Training

Deep autoencoders require additional techniques for stable training:

```python
def train_deep_autoencoder(model, train_loader, criterion, optimizer,
                           device, epoch):
    """Train deep autoencoder with gradient clipping."""
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


def train_with_scheduler(model, train_loader, device,
                         num_epochs=15, lr=0.001):
    """Full training pipeline with learning rate scheduling."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    best_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        train_loss = train_deep_autoencoder(
            model, train_loader, criterion, optimizer, device, epoch
        )
        print(f"Epoch {epoch} - Loss: {train_loss:.6f}")
        
        scheduler.step(train_loss)
        
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), 'best_autoencoder.pth')
    
    return best_loss
```

### Training Best Practices

| Technique | Purpose | When to Use |
|-----------|---------|-------------|
| **Xavier/He initialization** | Proper weight scaling | Always |
| **Adam optimizer** | Adaptive learning rates | Default choice |
| **Gradient clipping** | Prevent exploding gradients | Deep networks |
| **Batch normalization** | Stabilize training | Deep networks |
| **Dropout** | Regularization | Overfitting risk |
| **Learning rate scheduling** | Fine-tune convergence | Long training |
| **Early stopping** | Prevent overfitting | When validation available |

---

## Part 4: Latent Space Analysis

### Visualization

```python
def visualize_latent_space(model, test_loader, device):
    """Visualize the latent space colored by digit class."""
    model.eval()
    
    latents, labels = [], []
    
    with torch.no_grad():
        for images, lbls in test_loader:
            images = images.view(images.size(0), -1).to(device)
            z = model.encode(images)
            latents.append(z.cpu().numpy())
            labels.append(lbls.numpy())
    
    latents = np.concatenate(latents)
    labels = np.concatenate(labels)
    
    if model.latent_dim == 2:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(latents[:, 0], latents[:, 1],
                             c=labels, cmap='tab10', alpha=0.6, s=5)
        plt.colorbar(scatter, label='Digit')
        plt.title('2D Latent Space')
    else:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        latents_2d = tsne.fit_transform(latents[:5000])
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1],
                             c=labels[:5000], cmap='tab10', alpha=0.6, s=5)
        plt.colorbar(scatter, label='Digit')
        plt.title(f't-SNE of {model.latent_dim}D Latent Space')
    
    plt.savefig('latent_space_visualization.png', dpi=150)
    plt.show()
```

### Latent Space Geometry

```python
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA


def analyze_latent_geometry(model, test_loader, device, num_samples=2000):
    """
    Analyze geometric properties of learned latent space.
    
    Metrics:
    - Distance correlation (1.0 = perfect preservation)
    - Neighborhood preservation (1.0 = perfect)
    - Intrinsic dimensionality (effective dims used)
    """
    model.eval()
    
    original_data, latent_data = [], []
    
    with torch.no_grad():
        for images, _ in test_loader:
            if len(original_data) * test_loader.batch_size >= num_samples:
                break
            images_flat = images.view(images.size(0), -1)
            original_data.append(images_flat.numpy())
            latent = model.encode(images_flat.to(device))
            latent_data.append(latent.view(latent.size(0), -1).cpu().numpy())
    
    X_orig = np.concatenate(original_data)[:num_samples]
    X_lat = np.concatenate(latent_data)[:num_samples]
    
    # Distance preservation
    idx = np.random.choice(num_samples, min(500, num_samples), replace=False)
    d_orig = cdist(X_orig[idx], X_orig[idx])
    d_lat = cdist(X_lat[idx], X_lat[idx])
    
    triu = np.triu_indices_from(d_orig, k=1)
    correlation = np.corrcoef(d_orig[triu], d_lat[triu])[0, 1]
    
    # Neighborhood preservation
    k = 10
    scores = []
    for i in range(min(100, len(idx))):
        n_orig = set(np.argsort(d_orig[i])[:k+1])
        n_lat = set(np.argsort(d_lat[i])[:k+1])
        scores.append(len(n_orig & n_lat) / (k + 1))
    
    # Intrinsic dimensionality
    pca = PCA()
    pca.fit(X_lat)
    dim_95 = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
    
    print(f"Distance correlation:      {correlation:.4f}")
    print(f"Neighborhood preservation: {np.mean(scores):.4f}")
    print(f"Intrinsic dim (95% var):   {dim_95}/{X_lat.shape[1]}")
    
    return correlation, np.mean(scores), dim_95
```

### Latent Space Arithmetic and Interpolation

```python
def latent_arithmetic(model, test_loader, device):
    """
    Demonstrate arithmetic and interpolation in latent space.
    
    Vector arithmetic: z_0 + (z_8 - z_1) transforms digit 0 toward 8.
    Interpolation: smooth transition between two digits.
    """
    model.eval()
    
    # Collect digit examples
    digit_examples = {i: [] for i in range(10)}
    with torch.no_grad():
        for images, labels in test_loader:
            for i in range(10):
                mask = labels == i
                if mask.sum() > 0 and len(digit_examples[i]) < 5:
                    digit_examples[i].extend(
                        images[mask][:5 - len(digit_examples[i])]
                    )
            if all(len(v) >= 5 for v in digit_examples.values()):
                break
    
    # Average latent per digit
    digit_z = {}
    for i in range(10):
        imgs = torch.stack(digit_examples[i])
        imgs_flat = imgs.view(imgs.size(0), -1).to(device)
        digit_z[i] = model.encode(imgs_flat).mean(dim=0, keepdim=True)
    
    # Vector arithmetic
    z_result = digit_z[0] + (digit_z[8] - digit_z[1])
    img_result = model.decode(z_result).cpu().numpy().reshape(28, 28)
    
    # Smooth interpolation 0 -> 8
    num_steps = 10
    fig, axes = plt.subplots(1, num_steps, figsize=(15, 1.5))
    for idx, t in enumerate(np.linspace(0, 1, num_steps)):
        z_t = (1 - t) * digit_z[0] + t * digit_z[8]
        img = model.decode(z_t).cpu().numpy().reshape(28, 28)
        axes[idx].imshow(img, cmap='gray')
        axes[idx].axis('off')
    
    plt.suptitle('Interpolation: digit 0 -> digit 8')
    plt.savefig('latent_interpolation.png', dpi=150)
    plt.show()
```

---

## Part 5: Failure Mode Analysis

```python
def analyze_failure_modes(model, test_loader, device, num_show=10):
    """
    Identify best/worst reconstructions and per-class error patterns.
    """
    model.eval()
    
    all_images, all_recon, all_errors, all_labels = [], [], [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            imgs_flat = images.view(images.size(0), -1).to(device)
            recon, _ = model(imgs_flat)
            errors = torch.mean((imgs_flat - recon) ** 2, dim=1)
            
            all_images.append(images.numpy())
            all_recon.append(recon.cpu().numpy())
            all_errors.append(errors.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_images = np.concatenate(all_images)
    all_recon = np.concatenate(all_recon)
    all_errors = np.concatenate(all_errors)
    all_labels = np.concatenate(all_labels)
    
    best_idx = np.argsort(all_errors)[:num_show]
    worst_idx = np.argsort(all_errors)[-num_show:][::-1]
    
    print(f"Best error:  {all_errors[best_idx[0]]:.6f}")
    print(f"Worst error: {all_errors[worst_idx[0]]:.6f}")
    print(f"Mean error:  {np.mean(all_errors):.6f}")
    
    print("\nPer-class errors:")
    for d in range(10):
        e = all_errors[all_labels == d]
        print(f"  Digit {d}: {np.mean(e):.6f} +/- {np.std(e):.6f}")
    
    # Visualize
    fig, axes = plt.subplots(4, num_show, figsize=(15, 6))
    for i in range(num_show):
        axes[0, i].imshow(all_images[best_idx[i], 0], cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(all_recon[best_idx[i]].reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
        axes[2, i].imshow(all_images[worst_idx[i], 0], cmap='gray')
        axes[2, i].axis('off')
        axes[3, i].imshow(all_recon[worst_idx[i]].reshape(28, 28), cmap='gray')
        axes[3, i].axis('off')
    
    axes[0, 0].set_ylabel('Best')
    axes[1, 0].set_ylabel('Best recon')
    axes[2, 0].set_ylabel('Worst')
    axes[3, 0].set_ylabel('Worst recon')
    plt.suptitle('Best vs Worst Reconstructions')
    plt.tight_layout()
    plt.savefig('failure_analysis.png', dpi=150)
    plt.show()
```

---

## Part 6: Practical Applications

### Anomaly Detection

Train on normal data, flag samples with high reconstruction error as anomalies.

```python
class AnomalyDetector:
    """Anomaly detection using reconstruction error."""
    
    def __init__(self, model, threshold_percentile=95):
        self.model = model
        self.threshold_percentile = threshold_percentile
        self.threshold = None
    
    def fit(self, data_loader, device):
        """Compute threshold from normal data."""
        self.model.eval()
        errors = []
        with torch.no_grad():
            for images, _ in data_loader:
                images = images.view(images.size(0), -1).to(device)
                recon, _ = self.model(images)
                error = torch.mean((images - recon) ** 2, dim=1)
                errors.extend(error.cpu().numpy())
        self.threshold = np.percentile(errors, self.threshold_percentile)
        print(f"Anomaly threshold: {self.threshold:.6f}")
        return errors
    
    def predict(self, data_loader, device):
        """Return (errors, is_anomaly) for new data."""
        self.model.eval()
        errors = []
        with torch.no_grad():
            for images, _ in data_loader:
                images = images.view(images.size(0), -1).to(device)
                recon, _ = self.model(images)
                error = torch.mean((images - recon) ** 2, dim=1)
                errors.extend(error.cpu().numpy())
        errors = np.array(errors)
        return errors, (errors > self.threshold).astype(int)
```

### Image Compression

```python
def evaluate_compression(model, test_loader, device, num_images=5):
    """Evaluate autoencoder as a lossy compressor."""
    model.eval()
    images, _ = next(iter(test_loader))
    images = images[:num_images]
    
    original_size = 784 * 4  # float32 bytes
    compressed_size = model.latent_dim * 4
    ratio = original_size / compressed_size
    
    imgs_flat = images.view(images.size(0), -1).to(device)
    with torch.no_grad():
        latent = model.encode(imgs_flat)
        recon = model.decode(latent)
    
    mse = torch.mean((imgs_flat - recon) ** 2).item()
    psnr = 10 * np.log10(1.0 / (mse + 1e-10))
    
    print(f"Compression: {ratio:.1f}x | PSNR: {psnr:.2f} dB")
```

### Clustering in Latent Space

```python
def evaluate_clustering(model, test_loader, device, n_clusters=10):
    """Compare clustering quality: pixel space vs latent space."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    model.eval()
    X_orig, X_lat, y = [], [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            flat = images.view(images.size(0), -1)
            X_orig.append(flat.numpy())
            X_lat.append(model.encode(flat.to(device)).cpu().numpy())
            y.append(labels.numpy())
    
    X_orig = np.concatenate(X_orig)[:5000]
    X_lat = np.concatenate(X_lat)[:5000]
    y = np.concatenate(y)[:5000]
    
    km_orig = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    km_lat = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    
    sil_orig = silhouette_score(X_orig, km_orig.fit_predict(X_orig))
    sil_lat = silhouette_score(X_lat, km_lat.fit_predict(X_lat))
    
    print(f"Silhouette - Original: {sil_orig:.4f} | Latent: {sil_lat:.4f}")
```

### Transfer Learning

```python
def transfer_learning(encoder, latent_dim, train_loader, test_loader,
                      device, num_epochs=5):
    """Use pretrained encoder for classification."""
    
    class Classifier(nn.Module):
        def __init__(self, encoder, latent_dim):
            super().__init__()
            self.encoder = encoder
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.head = nn.Sequential(
                nn.Linear(latent_dim, 128), nn.ReLU(),
                nn.Dropout(0.3), nn.Linear(128, 10)
            )
        
        def forward(self, x):
            return self.head(self.encoder(x))
    
    model = Classifier(encoder, latent_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.head.parameters(), lr=0.001)
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        correct, total = 0, 0
        for images, labels in train_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
        print(f"Epoch {epoch}: Train Acc = {100. * correct / total:.2f}%")
    
    # Test
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            correct += model(images).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    print(f"Test Accuracy: {100. * correct / total:.2f}%")
```

---

## Part 7: Architecture Comparison Tool

```python
import time


class ModelComparator:
    """Compare multiple autoencoder architectures systematically."""
    
    def __init__(self):
        self.results = {}
    
    def evaluate(self, model, test_loader, device, name):
        model.eval()
        mse_losses, times = [], []
        
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.view(images.size(0), -1).to(device)
                t0 = time.time()
                recon, _ = model(images)
                times.append(time.time() - t0)
                mse = torch.mean((images - recon) ** 2, dim=1)
                mse_losses.extend(mse.cpu().numpy())
        
        self.results[name] = {
            'mse': np.mean(mse_losses),
            'params': sum(p.numel() for p in model.parameters()),
            'time_ms': np.mean(times) * 1000,
        }
        return self.results[name]
    
    def summary(self):
        print(f"{'Model':<25} {'MSE':>10} {'Params':>10} {'Time(ms)':>10}")
        print("-" * 55)
        for name, r in self.results.items():
            print(f"{name:<25} {r['mse']:>10.6f} {r['params']:>10,} "
                  f"{r['time_ms']:>10.2f}")
```

---

## Exercises

### Exercise 1: Full Training Pipeline

Build a complete autoencoder training pipeline that:
a) Trains for 20 epochs with Adam
b) Logs train and test loss per epoch
c) Saves the best model by test loss
d) Generates reconstruction comparisons every 5 epochs

### Exercise 2: Cross-Architecture Analysis

Train and compare using ModelComparator: shallow AE (2 layers), deep AE (4-6 layers), convolutional AE, sparse AE, denoising AE. Which is most parameter-efficient? Which has the smoothest latent space?

### Exercise 3: Anomaly Detection

Train on digits 0-8, use digit 9 as anomaly. Compute TPR and FPR across different threshold percentiles. Plot a ROC curve.

### Exercise 4: Semi-supervised Learning

a) Pretrain autoencoder on all 60K MNIST images (unsupervised)
b) Fine-tune classifier on small labeled subsets: 100, 500, 1000, 5000 samples
c) Compare with supervised-only baseline

### Exercise 5: Latent Space Completeness Study

With `latent_dim=2`, generate images from a grid of points in latent space. Does every region of latent space produce reasonable outputs? Compare with a VAE.

### Exercise 6: Reconstruction from Sparse Codes

a) Encode a test image and get its latent representation
b) Gradually zero out neurons (smallest first)
c) Reconstruct from increasingly sparse codes
d) Plot sparsity level vs reconstruction quality

### Final Project: Complete Autoencoder Study

1. Train 5+ different architectures on the same dataset
2. Run all analysis tools from this module
3. Create a comparison report identifying the best architecture for: compression, anomaly detection, feature extraction, and interpolation quality

---

## Summary

| Topic | Key Takeaway |
|-------|-------------|
| **Standard training** | Minimize reconstruction loss with Adam; monitor train and test loss |
| **Denoising training** | Corrupt input, reconstruct clean; forces robust features |
| **Deep training** | Use gradient clipping, batch norm, LR scheduling |
| **Latent analysis** | Check distance preservation, neighborhood preservation, intrinsic dim |
| **Failure modes** | Per-class error analysis reveals architecture weaknesses |
| **Anomaly detection** | High reconstruction error = anomaly |
| **Compression** | Compression ratio = input_dim / latent_dim |
| **Clustering** | Latent features often cluster better than raw pixels |
| **Transfer learning** | Freeze encoder, train classifier head on latent features |

**Key Insight:** Comprehensive training and analysis reveals not just how well an autoencoder performs, but *why* it succeeds or fails, enabling informed architecture and hyperparameter choices for specific applications.
