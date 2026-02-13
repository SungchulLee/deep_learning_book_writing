"""
Module 40.6: Autoencoder Applications

This script demonstrates real-world applications of autoencoders:
1. Anomaly Detection
2. Image Compression
3. Feature Extraction for Downstream Tasks
4. Clustering in Latent Space
5. Transfer Learning with Pretrained Encoders

Learning Objectives:
- Apply autoencoders to practical problems
- Understand anomaly detection using reconstruction error
- Use autoencoders for unsupervised feature learning
- Leverage latent representations for clustering
- Transfer learned representations to supervised tasks

Time: 60 minutes
Level: Advanced
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, classification_report
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# PART 1: ANOMALY DETECTION
# =============================================================================

class AnomalyDetector:
    """
    Anomaly detection using reconstruction error from autoencoder.
    
    Principle: Normal data should reconstruct well (low error),
    while anomalous data will have high reconstruction error.
    
    Applications:
    - Fraud detection
    - Manufacturing defect detection
    - Medical anomaly detection
    - Network intrusion detection
    """
    
    def __init__(self, model: nn.Module, threshold_percentile: float = 95):
        """
        Initialize anomaly detector.
        
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
        """
        Fit anomaly detector by computing threshold from normal data.
        
        Parameters:
        -----------
        data_loader : DataLoader
            Data loader containing normal (non-anomalous) samples
        device : torch.device
            Device for computation
        """
        self.model.eval()
        errors = []
        
        with torch.no_grad():
            for images, _ in data_loader:
                images = images.view(images.size(0), -1).to(device)
                reconstructed, _ = self.model(images)
                
                # Compute per-sample reconstruction error
                error = torch.mean((images - reconstructed) ** 2, dim=1)
                errors.extend(error.cpu().numpy())
        
        # Set threshold based on percentile
        self.threshold = np.percentile(errors, self.threshold_percentile)
        print(f"Anomaly threshold set to: {self.threshold:.6f}")
        
        return errors
    
    def predict(self, data_loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies in new data.
        
        Returns:
        --------
        errors : np.ndarray
            Reconstruction errors for each sample
        is_anomaly : np.ndarray
            Binary predictions (1 = anomaly, 0 = normal)
        """
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


def demonstrate_anomaly_detection(
    model: nn.Module,
    normal_loader: DataLoader,
    anomaly_loader: DataLoader,
    device: torch.device
):
    """
    Demonstrate anomaly detection on MNIST.
    
    Setup: Train on digits 0-8 (normal), test on digit 9 (anomaly)
    """
    print("\n" + "="*60)
    print("ANOMALY DETECTION DEMO")
    print("="*60)
    
    # Initialize detector
    detector = AnomalyDetector(model, threshold_percentile=95)
    
    # Fit on normal data
    print("\nFitting on normal data (digits 0-8)...")
    normal_errors = detector.fit(normal_loader, device)
    
    # Predict on normal data
    print("\nEvaluating on normal test set...")
    normal_test_errors, normal_predictions = detector.predict(normal_loader, device)
    normal_fp_rate = np.mean(normal_predictions)
    
    # Predict on anomalies
    print("Evaluating on anomalous data (digit 9)...")
    anomaly_errors, anomaly_predictions = detector.predict(anomaly_loader, device)
    anomaly_tp_rate = np.mean(anomaly_predictions)
    
    print(f"\nResults:")
    print(f"False Positive Rate (normal flagged as anomaly): {normal_fp_rate:.2%}")
    print(f"True Positive Rate (anomaly detected): {anomaly_tp_rate:.2%}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of reconstruction errors
    axes[0].hist(normal_errors, bins=50, alpha=0.7, label='Normal (train)', density=True)
    axes[0].hist(normal_test_errors, bins=50, alpha=0.7, label='Normal (test)', density=True)
    axes[0].hist(anomaly_errors, bins=50, alpha=0.7, label='Anomaly (digit 9)', density=True)
    axes[0].axvline(detector.threshold, color='red', linestyle='--', 
                    label=f'Threshold ({detector.threshold:.4f})')
    axes[0].set_xlabel('Reconstruction Error', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Distribution of Reconstruction Errors', fontsize=13)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # ROC-style plot: threshold vs TPR/FPR
    thresholds = np.linspace(min(normal_errors), max(anomaly_errors), 100)
    tprs = []
    fprs = []
    
    for thresh in thresholds:
        tpr = np.mean(anomaly_errors > thresh)
        fpr = np.mean(normal_test_errors > thresh)
        tprs.append(tpr)
        fprs.append(fpr)
    
    axes[1].plot(fprs, tprs, linewidth=2)
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    axes[1].scatter([normal_fp_rate], [anomaly_tp_rate], color='red', s=100, 
                    label=f'Current (FPR={normal_fp_rate:.3f}, TPR={anomaly_tp_rate:.3f})',
                    zorder=10)
    axes[1].set_xlabel('False Positive Rate', fontsize=12)
    axes[1].set_ylabel('True Positive Rate', fontsize=12)
    axes[1].set_title('ROC-style Curve', fontsize=13)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('anomaly_detection.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved visualization to 'anomaly_detection.png'")


# =============================================================================
# PART 2: IMAGE COMPRESSION
# =============================================================================

def demonstrate_compression(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_images: int = 5
):
    """
    Demonstrate image compression using autoencoder.
    
    Compare:
    - Original size: 28×28 = 784 values
    - Compressed size: latent_dim values
    - Compression ratio: 784 / latent_dim
    """
    print("\n" + "="*60)
    print("IMAGE COMPRESSION DEMO")
    print("="*60)
    
    model.eval()
    
    # Get images
    images, labels = next(iter(test_loader))
    images = images[:num_images]
    labels = labels[:num_images]
    
    # Original size in bytes (assuming float32)
    original_size = 784 * 4  # 4 bytes per float32
    compressed_size = model.latent_dim * 4
    compression_ratio = original_size / compressed_size
    
    print(f"\nOriginal size: {original_size} bytes (784 × 4)")
    print(f"Compressed size: {compressed_size} bytes ({model.latent_dim} × 4)")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Space savings: {(1 - 1/compression_ratio) * 100:.1f}%")
    
    # Encode and decode
    images_flat = images.view(images.size(0), -1).to(device)
    
    with torch.no_grad():
        # Compress (encode)
        latent = model.encoder(images_flat)
        
        # Decompress (decode)
        reconstructed = model.decoder(latent)
    
    # Calculate compression quality (PSNR)
    images_np = images.numpy()
    recon_np = reconstructed.cpu().numpy().reshape(-1, 28, 28)
    
    psnrs = []
    for i in range(num_images):
        mse = np.mean((images_np[i, 0] - recon_np[i]) ** 2)
        psnr = 10 * np.log10(1.0 / (mse + 1e-10))
        psnrs.append(psnr)
    
    avg_psnr = np.mean(psnrs)
    print(f"\nAverage PSNR: {avg_psnr:.2f} dB")
    print("(Higher PSNR = better quality, typical range: 20-40 dB)")
    
    # Visualize
    fig, axes = plt.subplots(3, num_images, figsize=(12, 7))
    
    for i in range(num_images):
        # Original
        axes[0, i].imshow(images_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title(f'Original\n({original_size} bytes)', fontsize=10)
        
        # Compressed representation (visualize latent)
        latent_np = latent[i].cpu().numpy()
        # Create visualization of compressed data
        size = int(np.ceil(np.sqrt(len(latent_np))))
        padded = np.zeros(size * size)
        padded[:len(latent_np)] = latent_np
        axes[1, i].imshow(padded.reshape(size, size), cmap='viridis', aspect='auto')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title(f'Compressed\n({compressed_size} bytes)', fontsize=10)
        
        # Reconstructed
        axes[2, i].imshow(recon_np[i], cmap='gray', vmin=0, vmax=1)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Decompressed', fontsize=10)
        axes[2, i].text(0.5, -0.1, f'PSNR: {psnrs[i]:.1f} dB',
                       transform=axes[2, i].transAxes, ha='center', fontsize=9)
    
    plt.suptitle(f'Image Compression ({compression_ratio:.1f}x)', fontsize=14)
    plt.tight_layout()
    plt.savefig('compression_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved visualization to 'compression_demo.png'")


# =============================================================================
# PART 3: CLUSTERING IN LATENT SPACE
# =============================================================================

def demonstrate_clustering(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    n_clusters: int = 10
):
    """
    Demonstrate clustering using learned latent representations.
    
    Compares:
    - Clustering in original pixel space
    - Clustering in latent space
    - Clustering in PCA space
    """
    print("\n" + "="*60)
    print("CLUSTERING IN LATENT SPACE DEMO")
    print("="*60)
    
    model.eval()
    
    # Extract features
    original_features = []
    latent_features = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images_flat = images.view(images.size(0), -1)
            images_device = images_flat.to(device)
            
            # Original features (pixels)
            original_features.append(images_flat.numpy())
            
            # Latent features (autoencoder)
            latent = model.encoder(images_device)
            latent_features.append(latent.cpu().numpy())
            
            labels_list.append(labels.numpy())
    
    # Concatenate
    X_original = np.concatenate(original_features, axis=0)
    X_latent = np.concatenate(latent_features, axis=0)
    y_true = np.concatenate(labels_list, axis=0)
    
    # Limit to 5000 samples for speed
    n_samples = min(5000, len(X_original))
    indices = np.random.choice(len(X_original), n_samples, replace=False)
    X_original = X_original[indices]
    X_latent = X_latent[indices]
    y_true = y_true[indices]
    
    # PCA for comparison
    pca = PCA(n_components=model.latent_dim)
    X_pca = pca.fit_transform(X_original)
    
    # Perform clustering
    print(f"\nPerforming K-means clustering (k={n_clusters})...")
    
    kmeans_original = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_latent = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    
    labels_original = kmeans_original.fit_predict(X_original)
    labels_latent = kmeans_latent.fit_predict(X_latent)
    labels_pca = kmeans_pca.fit_predict(X_pca)
    
    # Compute silhouette scores
    sil_original = silhouette_score(X_original, labels_original)
    sil_latent = silhouette_score(X_latent, labels_latent)
    sil_pca = silhouette_score(X_pca, labels_pca)
    
    print(f"\nSilhouette Scores (higher = better):")
    print(f"Original space: {sil_original:.4f}")
    print(f"PCA space:      {sil_pca:.4f}")
    print(f"Latent space:   {sil_latent:.4f}")
    
    # Visualize clusters using t-SNE
    print("\nComputing t-SNE for visualization...")
    tsne = TSNE(n_components=2, random_state=42)
    
    # Use latent space for t-SNE (faster than original)
    X_tsne = tsne.fit_transform(X_latent)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # True labels
    scatter1 = axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], 
                               c=y_true, cmap='tab10', alpha=0.6, s=10)
    axes[0].set_title('True Labels', fontsize=13)
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    plt.colorbar(scatter1, ax=axes[0], label='Digit')
    
    # Cluster labels
    scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1],
                               c=labels_latent, cmap='tab10', alpha=0.6, s=10)
    axes[1].set_title(f'K-means Clusters (Silhouette: {sil_latent:.3f})', fontsize=13)
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    plt.colorbar(scatter2, ax=axes[1], label='Cluster')
    
    plt.tight_layout()
    plt.savefig('clustering_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved visualization to 'clustering_demo.png'")


# =============================================================================
# PART 4: TRANSFER LEARNING
# =============================================================================

def demonstrate_transfer_learning(
    pretrained_encoder: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 5
):
    """
    Demonstrate transfer learning using pretrained encoder.
    
    Setup:
    1. Use pretrained encoder from autoencoder
    2. Add classification head
    3. Fine-tune on labeled data
    4. Compare with training from scratch
    """
    print("\n" + "="*60)
    print("TRANSFER LEARNING DEMO")
    print("="*60)
    
    # Create classifier using pretrained encoder
    class TransferClassifier(nn.Module):
        def __init__(self, encoder, num_classes=10):
            super().__init__()
            self.encoder = encoder
            # Freeze encoder (optional - can also fine-tune)
            for param in self.encoder.parameters():
                param.requires_grad = False
            
            # Add classification head
            self.classifier = nn.Sequential(
                nn.Linear(pretrained_encoder.latent_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
        
        def forward(self, x):
            features = self.encoder(x)
            logits = self.classifier(features)
            return logits
    
    # Initialize model
    model = TransferClassifier(pretrained_encoder).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    print("\nTraining classifier with frozen encoder...")
    
    # Train
    model.train()
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        accuracy = 100.0 * correct / total
        print(f"Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}, "
              f"Train Acc={accuracy:.2f}%")
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    test_accuracy = 100.0 * correct / total
    print(f"\nTest Accuracy: {test_accuracy:.2f}%")
    print("\nNote: This demonstrates using pretrained features for classification.")
    print("For comparison, training from scratch typically requires more epochs.")


# =============================================================================
# PART 5: UTILITIES
# =============================================================================

def load_mnist_data(batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    """Load MNIST dataset."""
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.MNIST(root='./data', train=True,
                                   download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False,
                                  download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2)
    
    return train_loader, test_loader


def create_anomaly_loaders(batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    """
    Create loaders for anomaly detection demo.
    Normal: digits 0-8, Anomaly: digit 9
    """
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Load full dataset
    train_dataset = datasets.MNIST(root='./data', train=False,
                                   download=True, transform=transform)
    
    # Split into normal (0-8) and anomaly (9)
    normal_indices = [i for i, (_, label) in enumerate(train_dataset) if label != 9]
    anomaly_indices = [i for i, (_, label) in enumerate(train_dataset) if label == 9]
    
    normal_dataset = torch.utils.data.Subset(train_dataset, normal_indices)
    anomaly_dataset = torch.utils.data.Subset(train_dataset, anomaly_indices)
    
    normal_loader = DataLoader(normal_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=2)
    
    return normal_loader, anomaly_loader


# =============================================================================
# PART 6: MAIN EXECUTION
# =============================================================================

def main():
    """Main function demonstrating all applications."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load a pretrained model (assuming you have one)
    # For this demo, we'll create and quickly train a simple model
    print("\nTraining a simple autoencoder for demonstrations...")
    print("(In practice, use a well-trained model)")
    
    from torch.nn import Sequential, Linear, ReLU, Sigmoid
    
    class SimpleAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.latent_dim = 32
            self.encoder = Sequential(
                Linear(784, 256), ReLU(),
                Linear(256, 128), ReLU(),
                Linear(128, self.latent_dim), ReLU()
            )
            self.decoder = Sequential(
                Linear(self.latent_dim, 128), ReLU(),
                Linear(128, 256), ReLU(),
                Linear(256, 784), Sigmoid()
            )
        
        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z), z
    
    model = SimpleAE().to(device)
    
    # Quick training (just for demo)
    train_loader, test_loader = load_mnist_data()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(3):  # Quick training
        for images, _ in train_loader:
            images = images.view(images.size(0), -1).to(device)
            optimizer.zero_grad()
            recon, _ = model(images)
            loss = criterion(recon, images)
            loss.backward()
            optimizer.step()
        print(f"Quick training epoch {epoch+1}/3 complete")
    
    model.eval()
    
    # Run demonstrations
    print("\n" + "="*60)
    print("RUNNING AUTOENCODER APPLICATIONS")
    print("="*60)
    
    # 1. Anomaly Detection
    normal_loader, anomaly_loader = create_anomaly_loaders()
    demonstrate_anomaly_detection(model, normal_loader, anomaly_loader, device)
    
    # 2. Image Compression
    demonstrate_compression(model, test_loader, device)
    
    # 3. Clustering
    demonstrate_clustering(model, test_loader, device)
    
    # 4. Transfer Learning
    demonstrate_transfer_learning(model.encoder, train_loader, test_loader, device)
    
    print("\n" + "="*60)
    print("ALL DEMONSTRATIONS COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()


# =============================================================================
# EXERCISES
# =============================================================================

"""
EXERCISE 1: Advanced Anomaly Detection
---------------------------------------
Improve anomaly detection:
a) Try different threshold selection methods (ROC, F1-score)
b) Use ensemble of autoencoders
c) Add per-feature thresholds instead of global
d) Test on real anomaly detection dataset

Questions:
- How sensitive is performance to threshold choice?
- Can you handle multiple anomaly types?


EXERCISE 2: Learned Compression
--------------------------------
Compare autoencoder compression with traditional methods:
a) JPEG compression
b) PNG compression
c) Wavelet compression

Metrics: Compression ratio, PSNR, SSIM, visual quality

Questions:
- When does learned compression outperform traditional methods?
- Can you design task-specific compressors?


EXERCISE 3: Semi-supervised Learning
-------------------------------------
Use autoencoders for semi-supervised learning:
a) Pretrain on unlabeled data
b) Fine-tune on small labeled subset
c) Compare with supervised-only baseline

Vary amount of labeled data: [100, 500, 1000, 5000]

Questions:
- How much does pretraining help with limited labels?
- What's the minimum labeled data needed?


EXERCISE 4: Domain Adaptation
------------------------------
Train autoencoder on MNIST, apply to Fashion-MNIST:
a) Extract features using MNIST-trained encoder
b) Train classifier on Fashion-MNIST
c) Compare with native Fashion-MNIST features

Questions:
- Do MNIST features transfer to Fashion-MNIST?
- How does performance compare to domain-specific features?


EXERCISE 5: Active Learning
----------------------------
Use reconstruction error for active learning:
a) Train on small initial labeled set
b) Use reconstruction error to select samples for labeling
c) Iteratively expand labeled set

Compare with:
- Random sampling
- Uncertainty sampling
- Diversity sampling

Questions:
- Is reconstruction error a good proxy for informativeness?
- Can you combine multiple selection criteria?
"""
