# Autoencoder Applications

Real-world applications of autoencoders for anomaly detection, compression, clustering, and transfer learning.

---

## Overview

**Applications Covered:**

1. Anomaly Detection
2. Image Compression
3. Clustering in Latent Space
4. Transfer Learning with Pretrained Encoders

**Learning Objectives:**

- Apply autoencoders to practical problems
- Understand anomaly detection using reconstruction error
- Use autoencoders for unsupervised feature learning
- Leverage latent representations for clustering
- Transfer learned representations to supervised tasks

**Time:** ~60 minutes  
**Level:** Advanced

---

## Part 1: Anomaly Detection

### Principle

Normal data should reconstruct well (low error), while anomalous data will have high reconstruction error.

**Applications:**
- Fraud detection
- Manufacturing defect detection
- Medical anomaly detection
- Network intrusion detection

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

class AnomalyDetector:
    """
    Anomaly detection using reconstruction error from autoencoder.
    """
    
    def __init__(self, model: nn.Module, threshold_percentile: float = 95):
        """
        Parameters:
        -----------
        model : nn.Module
            Trained autoencoder
        threshold_percentile : float
            Percentile of reconstruction errors to use as threshold
            (e.g., 95 means top 5% errors are anomalies)
        """
        self.model = model
        self.threshold_percentile = threshold_percentile
        self.threshold = None
    
    def fit(self, data_loader: DataLoader, device: torch.device):
        """Fit by computing threshold from normal data."""
        self.model.eval()
        errors = []
        
        with torch.no_grad():
            for images, _ in data_loader:
                images = images.view(images.size(0), -1).to(device)
                reconstructed, _ = self.model(images)
                
                # Per-sample reconstruction error
                error = torch.mean((images - reconstructed) ** 2, dim=1)
                errors.extend(error.cpu().numpy())
        
        # Set threshold based on percentile
        self.threshold = np.percentile(errors, self.threshold_percentile)
        print(f"Anomaly threshold: {self.threshold:.6f}")
        
        return errors
    
    def predict(self, data_loader: DataLoader, device: torch.device):
        """Predict anomalies in new data."""
        if self.threshold is None:
            raise ValueError("Must call fit() before predict()")
        
        self.model.eval()
        errors = []
        
        with torch.no_grad():
            for images, _ in data_loader:
                images = images.view(images.size(0), -1).to(device)
                reconstructed, _ = self.model(images)
                error = torch.mean((images - reconstructed) ** 2, dim=1)
                errors.extend(error.cpu().numpy())
        
        errors = np.array(errors)
        is_anomaly = (errors > self.threshold).astype(int)
        
        return errors, is_anomaly
```

### Demonstration

```python
def demonstrate_anomaly_detection(model, normal_loader, anomaly_loader, device):
    """
    Demo: Train on digits 0-8 (normal), test on digit 9 (anomaly)
    """
    detector = AnomalyDetector(model, threshold_percentile=95)
    
    # Fit on normal data
    print("Fitting on normal data (digits 0-8)...")
    normal_errors = detector.fit(normal_loader, device)
    
    # Predict on normal and anomalous data
    normal_test_errors, normal_predictions = detector.predict(normal_loader, device)
    anomaly_errors, anomaly_predictions = detector.predict(anomaly_loader, device)
    
    # Results
    false_positive_rate = np.mean(normal_predictions)
    true_positive_rate = np.mean(anomaly_predictions)
    
    print(f"False Positive Rate: {false_positive_rate:.2%}")
    print(f"True Positive Rate: {true_positive_rate:.2%}")
    
    # Visualization: histogram of errors + ROC curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(normal_errors, bins=50, alpha=0.7, label='Normal', density=True)
    axes[0].hist(anomaly_errors, bins=50, alpha=0.7, label='Anomaly', density=True)
    axes[0].axvline(detector.threshold, color='red', linestyle='--', label='Threshold')
    axes[0].legend()
    axes[0].set_title('Distribution of Reconstruction Errors')
    
    plt.savefig('anomaly_detection.png', dpi=150)
    plt.show()
```

---

## Part 2: Image Compression

### Principle

Use encoder to compress images to latent space, decoder to reconstruct.

| Metric | Description |
|--------|-------------|
| **Compression ratio** | Original size / Compressed size |
| **PSNR** | Peak Signal-to-Noise Ratio (higher = better) |
| **Space savings** | (1 - 1/ratio) × 100% |

```python
def demonstrate_compression(model, test_loader, device, num_images=5):
    """Demonstrate image compression using autoencoder."""
    model.eval()
    
    images, labels = next(iter(test_loader))
    images = images[:num_images]
    
    # Calculate sizes
    original_size = 784 * 4  # bytes (float32)
    compressed_size = model.latent_dim * 4
    compression_ratio = original_size / compressed_size
    
    print(f"Original size: {original_size} bytes")
    print(f"Compressed size: {compressed_size} bytes")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Space savings: {(1 - 1/compression_ratio) * 100:.1f}%")
    
    # Encode and decode
    images_flat = images.view(images.size(0), -1).to(device)
    
    with torch.no_grad():
        latent = model.encoder(images_flat)  # Compress
        reconstructed = model.decoder(latent)  # Decompress
    
    # Calculate PSNR
    images_np = images.numpy()
    recon_np = reconstructed.cpu().numpy().reshape(-1, 28, 28)
    
    psnrs = []
    for i in range(num_images):
        mse = np.mean((images_np[i, 0] - recon_np[i]) ** 2)
        psnr = 10 * np.log10(1.0 / (mse + 1e-10))
        psnrs.append(psnr)
    
    print(f"Average PSNR: {np.mean(psnrs):.2f} dB")
    print("(Typical range: 20-40 dB, higher = better)")
    
    # Visualize: original → compressed → decompressed
    fig, axes = plt.subplots(3, num_images, figsize=(12, 7))
    
    for i in range(num_images):
        axes[0, i].imshow(images_np[i, 0], cmap='gray')
        axes[0, i].axis('off')
        
        # Visualize latent as small square
        latent_np = latent[i].cpu().numpy()
        size = int(np.ceil(np.sqrt(len(latent_np))))
        padded = np.zeros(size * size)
        padded[:len(latent_np)] = latent_np
        axes[1, i].imshow(padded.reshape(size, size), cmap='viridis')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(recon_np[i], cmap='gray')
        axes[2, i].axis('off')
    
    plt.suptitle(f'Compression ({compression_ratio:.1f}x)')
    plt.savefig('compression_demo.png', dpi=150)
    plt.show()
```

---

## Part 3: Clustering in Latent Space

### Principle

Latent representations often provide better features for clustering than raw pixels.

```python
def demonstrate_clustering(model, test_loader, device, n_clusters=10):
    """
    Compare clustering in:
    - Original pixel space
    - PCA space
    - Autoencoder latent space
    """
    model.eval()
    
    # Extract features
    original_features = []
    latent_features = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images_flat = images.view(images.size(0), -1)
            images_device = images_flat.to(device)
            
            original_features.append(images_flat.numpy())
            latent = model.encoder(images_device)
            latent_features.append(latent.cpu().numpy())
            labels_list.append(labels.numpy())
    
    X_original = np.concatenate(original_features, axis=0)
    X_latent = np.concatenate(latent_features, axis=0)
    y_true = np.concatenate(labels_list, axis=0)
    
    # Limit samples for speed
    n_samples = min(5000, len(X_original))
    indices = np.random.choice(len(X_original), n_samples, replace=False)
    X_original = X_original[indices]
    X_latent = X_latent[indices]
    y_true = y_true[indices]
    
    # PCA for comparison
    from sklearn.decomposition import PCA
    pca = PCA(n_components=model.latent_dim)
    X_pca = pca.fit_transform(X_original)
    
    # K-means clustering
    kmeans_original = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_latent = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    
    labels_original = kmeans_original.fit_predict(X_original)
    labels_latent = kmeans_latent.fit_predict(X_latent)
    labels_pca = kmeans_pca.fit_predict(X_pca)
    
    # Silhouette scores (higher = better)
    sil_original = silhouette_score(X_original, labels_original)
    sil_latent = silhouette_score(X_latent, labels_latent)
    sil_pca = silhouette_score(X_pca, labels_pca)
    
    print(f"Silhouette Scores:")
    print(f"  Original space: {sil_original:.4f}")
    print(f"  PCA space:      {sil_pca:.4f}")
    print(f"  Latent space:   {sil_latent:.4f}")
    
    # Visualize with t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_latent)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_true, cmap='tab10', alpha=0.6, s=10)
    axes[0].set_title('True Labels')
    
    axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_latent, cmap='tab10', alpha=0.6, s=10)
    axes[1].set_title(f'K-means Clusters (Silhouette: {sil_latent:.3f})')
    
    plt.savefig('clustering_demo.png', dpi=150)
    plt.show()
```

---

## Part 4: Transfer Learning

### Principle

Use pretrained encoder as feature extractor, add classification head for supervised task.

```python
def demonstrate_transfer_learning(pretrained_encoder, train_loader, test_loader, device, num_epochs=5):
    """
    Transfer learning using pretrained encoder.
    
    Steps:
    1. Use pretrained encoder from autoencoder
    2. Freeze encoder weights
    3. Add classification head
    4. Train only the classifier
    """
    
    class TransferClassifier(nn.Module):
        def __init__(self, encoder, latent_dim, num_classes=10):
            super().__init__()
            self.encoder = encoder
            
            # Freeze encoder
            for param in self.encoder.parameters():
                param.requires_grad = False
            
            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
        
        def forward(self, x):
            features = self.encoder(x)
            return self.classifier(features)
    
    # Initialize
    model = TransferClassifier(pretrained_encoder, pretrained_encoder.latent_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)  # Only train classifier
    
    print("Training classifier with frozen encoder...")
    
    # Train
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
            
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        print(f"Epoch {epoch}: Train Acc = {100.0 * correct / total:.2f}%")
    
    # Evaluate
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    print(f"\nTest Accuracy: {100.0 * correct / total:.2f}%")
```

---

## Part 5: Main Execution

```python
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create and train a simple autoencoder
    class SimpleAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.latent_dim = 32
            self.encoder = nn.Sequential(
                nn.Linear(784, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, self.latent_dim), nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim, 128), nn.ReLU(),
                nn.Linear(128, 256), nn.ReLU(),
                nn.Linear(256, 784), nn.Sigmoid()
            )
        
        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z), z
    
    model = SimpleAE().to(device)
    
    # Train model...
    
    # Run all applications
    demonstrate_anomaly_detection(model, normal_loader, anomaly_loader, device)
    demonstrate_compression(model, test_loader, device)
    demonstrate_clustering(model, test_loader, device)
    demonstrate_transfer_learning(model.encoder, train_loader, test_loader, device)

if __name__ == "__main__":
    main()
```

---

## Exercises

### Exercise 1: Advanced Anomaly Detection

Improve anomaly detection:

a) Try different threshold selection methods (ROC optimization, F1-score)  
b) Use ensemble of autoencoders  
c) Add per-feature thresholds instead of global  
d) Test on real anomaly detection dataset

### Exercise 2: Learned Compression

Compare autoencoder compression with traditional methods:

- JPEG compression
- PNG compression
- Wavelet compression

**Metrics:** Compression ratio, PSNR, SSIM, visual quality

### Exercise 3: Semi-supervised Learning

a) Pretrain autoencoder on unlabeled data  
b) Fine-tune on small labeled subset  
c) Compare with supervised-only baseline

Vary labeled data: [100, 500, 1000, 5000] samples

### Exercise 4: Domain Adaptation

Train autoencoder on MNIST, apply to Fashion-MNIST:

a) Extract features using MNIST-trained encoder  
b) Train classifier on Fashion-MNIST  
c) Compare with domain-specific features

### Exercise 5: Active Learning

Use reconstruction error for active learning:

a) Train on small initial labeled set  
b) Use reconstruction error to select samples for labeling  
c) Iteratively expand labeled set

Compare with random sampling and uncertainty sampling.

---

## Summary

| Application | Principle | Key Metric |
|-------------|-----------|------------|
| **Anomaly Detection** | High reconstruction error = anomaly | TPR, FPR, AUC |
| **Compression** | Encode to smaller latent space | PSNR, compression ratio |
| **Clustering** | Cluster in latent space | Silhouette score |
| **Transfer Learning** | Pretrained encoder as feature extractor | Classification accuracy |

**Key Insight:** Autoencoders learn useful representations that transfer across multiple tasks. The latent space captures essential structure of the data, making it valuable for unsupervised and semi-supervised applications.
