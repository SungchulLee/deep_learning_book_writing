# Portfolio Compression

Using autoencoders for dimensionality reduction in portfolio management: compressing high-dimensional portfolio representations, constructing factor-mimicking portfolios, and efficient risk decomposition.

---

## Overview

**What you'll learn:**

- Compressing portfolio return profiles into low-dimensional representations
- Autoencoder-based portfolio construction and replication
- Comparison with PCA-based compression
- Risk decomposition via latent factors
- Clustering portfolios in latent space
- Transfer learning: pretrained encoders as portfolio feature extractors

---

## Mathematical Foundation

### Portfolio Compression Problem

Given a universe of $N$ assets with return matrix $R \in \mathbb{R}^{T \times N}$, portfolio compression seeks a low-dimensional representation:

$$z_t = f_\theta(r_t) \in \mathbb{R}^K, \quad K \ll N$$

such that the portfolio return can be approximately reconstructed:

$$\hat{r}_t = g_\phi(z_t) \approx r_t$$

The compression ratio $N/K$ measures how much dimensionality reduction is achieved while maintaining acceptable reconstruction fidelity.

### Applications

| Use Case | Input | Compressed Representation | Benefit |
|----------|-------|--------------------------|---------|
| **Risk management** | Asset returns | Risk factor exposures | Tractable VaR computation |
| **Portfolio replication** | Target portfolio | Simplified factor portfolio | Lower transaction costs |
| **Clustering** | Fund returns | Latent features | Strategy classification |
| **Stress testing** | Scenario returns | Compressed scenarios | Efficient simulation |

---

## Part 1: Portfolio Compression Autoencoder

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class PortfolioCompressor(nn.Module):
    """
    Autoencoder for compressing portfolio return vectors.
    
    Learns a low-dimensional representation that captures
    the dominant return patterns across assets.
    
    Args:
        n_assets: Number of assets (input dimension)
        n_components: Number of compressed dimensions
        hidden_dim: Hidden layer width
    """
    
    def __init__(self, n_assets, n_components=10, hidden_dim=64):
        super().__init__()
        
        self.n_assets = n_assets
        self.n_components = n_components
        
        # Encoder: Compress returns
        self.encoder = nn.Sequential(
            nn.Linear(n_assets, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_components),
        )
        
        # Decoder: Reconstruct returns
        self.decoder = nn.Sequential(
            nn.Linear(n_components, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_assets),
        )
    
    def encode(self, r):
        return self.encoder(r)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, r):
        z = self.encode(r)
        return self.decode(z), z
    
    def compression_ratio(self):
        return self.n_assets / self.n_components


def train_compressor(returns, n_components=10, num_epochs=100,
                     learning_rate=0.001, batch_size=64):
    """
    Train portfolio compressor on historical returns.
    
    Args:
        returns: (n_days, n_assets) numpy array
        n_components: target compressed dimension
    
    Returns:
        model, scaler, training_history
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    scaler = StandardScaler()
    returns_scaled = scaler.fit_transform(returns)
    
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(returns_scaled)
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    
    n_assets = returns.shape[1]
    model = PortfolioCompressor(n_assets, n_components).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    history = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, z = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        history.append(avg_loss)
        
        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1}: MSE = {avg_loss:.6f}")
    
    return model, scaler, history
```

---

## Part 2: Compression Quality Analysis

```python
def analyze_compression_quality(returns, model, scaler, device):
    """
    Evaluate compression quality with multiple metrics:
    - R² (explained variance ratio)
    - Per-asset RMSE
    - Correlation preservation
    - Portfolio-level tracking error
    """
    model.eval()
    returns_scaled = scaler.transform(returns)
    
    with torch.no_grad():
        x = torch.FloatTensor(returns_scaled).to(device)
        recon, z = model(x)
        recon_np = recon.cpu().numpy()
        z_np = z.cpu().numpy()
    
    # 1. Overall R²
    total_var = np.var(returns_scaled, axis=0).sum()
    residual_var = np.var(returns_scaled - recon_np, axis=0).sum()
    r_squared = 1 - residual_var / total_var
    print(f"Overall R²: {r_squared:.4f}")
    
    # 2. Per-asset RMSE
    per_asset_rmse = np.sqrt(np.mean((returns_scaled - recon_np) ** 2, axis=0))
    print(f"Per-asset RMSE: mean={np.mean(per_asset_rmse):.6f}, "
          f"max={np.max(per_asset_rmse):.6f}")
    
    # 3. Correlation preservation
    orig_corr = np.corrcoef(returns_scaled.T)
    recon_corr = np.corrcoef(recon_np.T)
    corr_error = np.mean(np.abs(orig_corr - recon_corr))
    print(f"Mean absolute correlation error: {corr_error:.4f}")
    
    # 4. Portfolio tracking error (equal-weight portfolio)
    n_assets = returns.shape[1]
    weights = np.ones(n_assets) / n_assets
    
    portfolio_returns = returns_scaled @ weights
    recon_portfolio = recon_np @ weights
    tracking_error = np.std(portfolio_returns - recon_portfolio) * np.sqrt(252)
    print(f"Equal-weight tracking error (annualized): {tracking_error:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # R² by compression level
    axes[0, 0].bar(range(len(per_asset_rmse)), 
                   sorted(per_asset_rmse, reverse=True))
    axes[0, 0].set_xlabel('Asset (sorted)')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].set_title('Per-Asset Reconstruction Error')
    
    # Correlation matrix comparison
    axes[0, 1].imshow(orig_corr - recon_corr, cmap='RdBu_r', 
                      vmin=-0.5, vmax=0.5)
    axes[0, 1].set_title('Correlation Error (Original - Reconstructed)')
    plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1])
    
    # Portfolio tracking
    cumulative_orig = np.cumsum(portfolio_returns)
    cumulative_recon = np.cumsum(recon_portfolio)
    axes[1, 0].plot(cumulative_orig, label='Original')
    axes[1, 0].plot(cumulative_recon, label='Compressed', linestyle='--')
    axes[1, 0].set_title('Cumulative Portfolio Returns')
    axes[1, 0].legend()
    
    # Latent factor time series
    for f in range(min(5, z_np.shape[1])):
        axes[1, 1].plot(z_np[:, f], alpha=0.7, label=f'z_{f}')
    axes[1, 1].set_title('Compressed Factor Time Series')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('compression_quality.png', dpi=150)
    plt.show()
    
    return r_squared, per_asset_rmse, tracking_error


def compare_compression_methods(returns, n_components_range=[2, 5, 10, 20]):
    """
    Compare autoencoder compression with PCA at different
    compression levels.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = {'n_components': [], 'pca_r2': [], 'ae_r2': []}
    
    for n_comp in n_components_range:
        print(f"\n--- {n_comp} components ---")
        
        # PCA
        scaler = StandardScaler()
        returns_scaled = scaler.fit_transform(returns)
        
        pca = PCA(n_components=n_comp)
        pca_z = pca.fit_transform(returns_scaled)
        pca_recon = pca.inverse_transform(pca_z)
        
        total_var = np.var(returns_scaled, axis=0).sum()
        pca_r2 = 1 - np.var(returns_scaled - pca_recon, axis=0).sum() / total_var
        
        # Autoencoder
        model, ae_scaler, _ = train_compressor(
            returns, n_components=n_comp, num_epochs=100
        )
        
        model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(ae_scaler.transform(returns)).to(device)
            ae_recon, _ = model(x)
            ae_recon = ae_recon.cpu().numpy()
        
        ae_r2 = 1 - np.var(ae_scaler.transform(returns) - ae_recon, axis=0).sum() / total_var
        
        results['n_components'].append(n_comp)
        results['pca_r2'].append(pca_r2)
        results['ae_r2'].append(ae_r2)
        
        print(f"  PCA R²: {pca_r2:.4f}")
        print(f"  AE R²:  {ae_r2:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(8, 5))
    plt.plot(results['n_components'], results['pca_r2'], 'o-', label='PCA')
    plt.plot(results['n_components'], results['ae_r2'], 's-', label='Autoencoder')
    plt.xlabel('Number of Components')
    plt.ylabel('R² (Explained Variance)')
    plt.title('Compression Quality: PCA vs Autoencoder')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('compression_comparison.png', dpi=150)
    plt.show()
    
    return results
```

---

## Part 3: Portfolio Clustering in Latent Space

```python
def cluster_portfolios(returns_dict, model, scaler, device, 
                       n_clusters=5):
    """
    Cluster portfolios/funds in latent space.
    
    Args:
        returns_dict: {portfolio_name: returns_array}
        model: trained compressor
        scaler: fitted StandardScaler
    
    More effective than clustering raw returns because the
    latent space captures fundamental strategy similarities.
    """
    model.eval()
    
    names = list(returns_dict.keys())
    latent_representations = {}
    
    with torch.no_grad():
        for name, returns in returns_dict.items():
            r_scaled = scaler.transform(returns)
            x = torch.FloatTensor(r_scaled).to(device)
            z = model.encode(x).cpu().numpy()
            
            # Use mean latent vector as portfolio fingerprint
            latent_representations[name] = z.mean(axis=0)
    
    # Stack into matrix
    Z = np.array([latent_representations[n] for n in names])
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(Z)
    
    # Visualize
    from sklearn.manifold import TSNE
    
    if Z.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=42)
        Z_2d = tsne.fit_transform(Z)
    else:
        Z_2d = Z
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(Z_2d[:, 0], Z_2d[:, 1], c=labels, 
                         cmap='tab10', s=100, alpha=0.8)
    
    for i, name in enumerate(names):
        plt.annotate(name, (Z_2d[i, 0], Z_2d[i, 1]), fontsize=7)
    
    plt.colorbar(scatter, label='Cluster')
    plt.title('Portfolio Clustering in Latent Space')
    plt.tight_layout()
    plt.savefig('portfolio_clustering.png', dpi=150)
    plt.show()
    
    return labels, latent_representations
```

---

## Part 4: Transfer Learning for Portfolio Classification

```python
def transfer_learning_classifier(pretrained_encoder, 
                                  train_returns, train_labels,
                                  test_returns, test_labels,
                                  scaler, device, 
                                  num_epochs=50):
    """
    Use pretrained encoder as feature extractor for portfolio
    classification (e.g., strategy type, risk category).
    
    1. Freeze pretrained encoder weights
    2. Add classification head
    3. Train only the classifier
    """
    
    class PortfolioClassifier(nn.Module):
        def __init__(self, encoder, latent_dim, n_classes):
            super().__init__()
            self.encoder = encoder
            
            # Freeze encoder
            for param in self.encoder.parameters():
                param.requires_grad = False
            
            self.classifier = nn.Sequential(
                nn.Linear(latent_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, n_classes)
            )
        
        def forward(self, x):
            with torch.no_grad():
                features = self.encoder(x)
            return self.classifier(features)
    
    n_classes = len(np.unique(train_labels))
    latent_dim = pretrained_encoder[-1].out_features  # Last linear layer
    
    model = PortfolioClassifier(pretrained_encoder, latent_dim, 
                                n_classes).to(device)
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Prepare data
    X_train = torch.FloatTensor(scaler.transform(train_returns)).to(device)
    y_train = torch.LongTensor(train_labels).to(device)
    X_test = torch.FloatTensor(scaler.transform(test_returns)).to(device)
    y_test = torch.LongTensor(test_labels).to(device)
    
    # Train classifier
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).argmax(dim=1)
        accuracy = (predictions == y_test).float().mean().item()
    
    print(f"Transfer learning classification accuracy: {accuracy:.4f}")
    return model, accuracy
```

---

## Exercises

### Exercise 1: Compression Sweep
Train autoencoders with K = 1, 2, 5, 10, 20, 50 components on a universe of 100 assets. Plot R², tracking error, and correlation error as a function of compression ratio. Compare with PCA.

### Exercise 2: Rolling Compression
Implement rolling window compression that retrains monthly. Measure whether the autoencoder adapts to changing factor structure better than static PCA.

### Exercise 3: Sparse Portfolio Compression
Add an L1 penalty on the latent representation to encourage sparse factor exposures. Compare interpretability of sparse vs dense compressed representations.

### Exercise 4: Portfolio Replication
Given a target portfolio, use the decoder to construct a factor-mimicking portfolio from the compressed representation. Measure tracking error over out-of-sample periods.

---

## Summary

| Task | Method | Key Metric |
|------|--------|------------|
| **Return compression** | Autoencoder bottleneck | R², tracking error |
| **Factor extraction** | Encoder output | Factor stability, interpretability |
| **Portfolio clustering** | K-means on latent codes | Silhouette score |
| **Strategy classification** | Transfer learning | Classification accuracy |
| **Risk decomposition** | Latent factor variance | Explained risk ratio |

**Key Insight:** Portfolio compression via autoencoders provides a flexible framework for reducing the dimensionality of large asset universes while preserving essential risk and return characteristics. Compared to PCA, autoencoders can capture nonlinear factor structures and regime-dependent loadings, though at the cost of reduced interpretability and the need for more careful training. The compressed representations serve as inputs for downstream tasks including clustering, classification, and risk management.
