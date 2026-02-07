# Anomaly Detection and Denoising

Using VAE reconstruction error and latent space properties for detecting anomalies and cleaning noisy data.

---

## Learning Objectives

By the end of this section, you will be able to:

- Use VAE reconstruction error as an anomaly score
- Implement anomaly detection pipelines for financial data
- Apply VAEs for data denoising via encode-decode
- Combine reconstruction error with KL divergence for robust anomaly scoring

---

## Anomaly Detection with VAEs

### Core Idea

A VAE trained on "normal" data learns to reconstruct normal patterns well. Anomalous data points — which differ from the training distribution — will have **high reconstruction error** because the model hasn't learned to represent them.

### Anomaly Score

```python
import torch
import torch.nn.functional as F

def anomaly_score(model, x, device='cpu'):
    """
    Compute anomaly score for input samples.
    
    Higher score = more anomalous.
    
    Args:
        model: Trained VAE (on normal data only)
        x: Input data [batch_size, data_dim]
    
    Returns:
        scores: Anomaly score per sample [batch_size]
    """
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        recon_x, mu, logvar = model(x)
        
        # Reconstruction error (per sample)
        recon_error = F.mse_loss(recon_x, x, reduction='none').sum(dim=-1)
        
        # KL divergence (per sample)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        
        # Combined anomaly score
        score = recon_error + kl
    
    return score
```

### Threshold Selection

```python
import numpy as np

def fit_anomaly_threshold(model, normal_loader, device, percentile=99):
    """
    Determine anomaly threshold from normal training data.
    
    Args:
        percentile: Scores above this percentile are anomalous
    """
    all_scores = []
    
    for data, _ in normal_loader:
        data = data.view(data.size(0), -1)
        scores = anomaly_score(model, data, device)
        all_scores.append(scores.cpu().numpy())
    
    all_scores = np.concatenate(all_scores)
    threshold = np.percentile(all_scores, percentile)
    
    return threshold, all_scores
```

---

## Financial Anomaly Detection

### Applications

| Application | Normal Data | Anomaly |
|-------------|------------|---------|
| **Fraud detection** | Legitimate transactions | Fraudulent transactions |
| **Regime change** | Normal market conditions | Crisis onset |
| **Data quality** | Clean market data | Erroneous ticks/prices |
| **Unusual trading** | Normal volume/patterns | Insider trading signals |

### Market Regime Detection

```python
def detect_regime_change(model, returns_series, window_size=20, device='cpu'):
    """
    Detect market regime changes using rolling anomaly scores.
    
    Args:
        returns_series: [T, num_assets] historical returns
        window_size: Rolling window for smoothing scores
    """
    model.eval()
    scores = []
    
    with torch.no_grad():
        for t in range(len(returns_series)):
            x = torch.FloatTensor(returns_series[t:t+1])
            score = anomaly_score(model, x, device)
            scores.append(score.item())
    
    scores = np.array(scores)
    
    # Smooth with rolling average
    smoothed = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
    
    return scores, smoothed
```

---

## Data Denoising

### Encode-Decode Denoising

VAEs can remove noise from corrupted data by projecting through the learned latent space. The encoder maps noisy input to the most likely latent code, and the decoder produces a clean reconstruction.

```python
def denoise(model, noisy_x, num_samples=10, device='cpu'):
    """
    Denoise input by averaging multiple reconstructions.
    
    Averaging over posterior samples smooths out noise
    while preserving signal structure.
    
    Args:
        noisy_x: Noisy input [batch_size, data_dim]
        num_samples: Number of posterior samples to average
    """
    model.eval()
    with torch.no_grad():
        noisy_x = noisy_x.to(device)
        mu, logvar = model.encode(noisy_x)
        
        # Average multiple samples for smoother result
        denoised = torch.zeros_like(noisy_x)
        for _ in range(num_samples):
            z = model.reparameterize(mu, logvar)
            denoised += model.decode(z)
        denoised /= num_samples
    
    return denoised
```

### Financial Denoising Applications

In finance, denoising is useful for cleaning noisy tick data (removing microstructure noise from high-frequency prices), smoothing factor exposures (extracting stable factor loadings from noisy estimates), and correlation matrix cleaning (projecting a noisy sample correlation matrix through a learned latent structure to obtain a smoother estimate).

---

## Representation Learning

### Using VAE Latent Codes as Features

The encoder's output provides a compact, meaningful representation useful for downstream tasks:

```python
def extract_features(model, dataloader, device='cpu'):
    """Extract latent features from a trained VAE."""
    model.eval()
    features, labels = [], []
    
    with torch.no_grad():
        for data, target in dataloader:
            data = data.view(data.size(0), -1).to(device)
            mu, _ = model.encode(data)
            features.append(mu.cpu())
            labels.append(target)
    
    return torch.cat(features), torch.cat(labels)

# Use features for downstream classification, clustering, etc.
features, labels = extract_features(model, dataloader)
```

### Financial Representation Learning

VAE latent representations can serve as learned market factors (complement or replace PCA-based factors), regime indicators (cluster latent codes to identify market regimes), and asset embeddings (similar assets cluster in latent space).

---

## Summary

| Application | Key Technique | Metric |
|-------------|---------------|--------|
| **Anomaly detection** | Reconstruction error + KL | High score = anomalous |
| **Regime detection** | Rolling anomaly scores | Score spikes = regime change |
| **Denoising** | Multi-sample reconstruction | Averaged decode output |
| **Representation** | Encoder output (μ) | Compact features for downstream tasks |

---

## Exercises

### Exercise 1: Anomaly Detection

Train a VAE on MNIST digits 0–8 only. Then:
a) Compute anomaly scores for digit 9
b) Compute scores for digits 0–8
c) Plot ROC curve for detecting digit 9 as anomaly

### Exercise 2: Financial Regime Detection

Train a VAE on returns from 2010–2019. Compute anomaly scores for 2020. Does the model detect the COVID crash?

### Exercise 3: Denoising

Add Gaussian noise ($\sigma = 0.3$) to MNIST images. Compare denoised outputs from: (a) mean reconstruction, (b) 1-sample, (c) 50-sample average.
