# Anomaly Detection in Finance

Using autoencoder reconstruction error for detecting market anomalies, regime changes, and fraudulent activity.

---

## Overview

**What you'll learn:**

- Anomaly detection principle: high reconstruction error signals anomalies
- Threshold selection methods for financial data
- Applications: market regime detection, fraud detection, outlier identification
- Time-series considerations: non-stationarity, volatility clustering
- Evaluation metrics for anomaly detection in finance

---

## Mathematical Foundation

### Core Principle

An autoencoder trained on "normal" data learns to reconstruct normal patterns well. Anomalous data — which deviates from learned patterns — will have **high reconstruction error**:

$$\text{anomaly\_score}(x_t) = \|x_t - g_\phi(f_\theta(x_t))\|^2$$

If $\text{anomaly\_score}(x_t) > \tau$ for some threshold $\tau$, then $x_t$ is flagged as anomalous.

### Why This Works in Finance

Financial markets exhibit strong regularities (factor structure, correlation patterns, mean-reversion). When these regularities break down — during crises, flash crashes, or regime changes — the autoencoder fails to reconstruct the observed returns, producing high reconstruction error.

### Threshold Selection

| Method | Description | Trade-off |
|--------|-------------|-----------|
| **Fixed percentile** | Flag top $\alpha$% errors | Simple but ignores time variation |
| **Rolling z-score** | $\frac{e_t - \mu_t}{\sigma_t}$ using rolling stats | Adapts to volatility clustering |
| **Extreme Value Theory** | Fit GPD to tail of error distribution | Principled for rare events |

---

## Part 1: Financial Anomaly Detector

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class FinancialAnomalyDetector(nn.Module):
    """
    Autoencoder for anomaly detection in financial time series.
    
    Trained on "normal" market data to learn the typical
    correlation and factor structure. Anomalous periods produce
    high reconstruction error.
    """
    
    def __init__(self, n_features, latent_dim=8, hidden_dim=32):
        super().__init__()
        
        self.n_features = n_features
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_features),
        )
    
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z
    
    def anomaly_score(self, x):
        """Per-sample reconstruction error."""
        self.eval()
        with torch.no_grad():
            recon, _ = self(x)
            scores = torch.mean((x - recon) ** 2, dim=1)
        return scores


class AnomalyDetectionSystem:
    """
    Complete anomaly detection system with training, threshold
    selection, and prediction.
    """
    
    def __init__(self, n_features, latent_dim=8, threshold_method='rolling_zscore'):
        self.model = FinancialAnomalyDetector(n_features, latent_dim)
        self.scaler = StandardScaler()
        self.threshold_method = threshold_method
        self.threshold = None
        self.training_errors = None
    
    def fit(self, returns, num_epochs=100, learning_rate=0.001, 
            batch_size=64):
        """
        Train on normal market data.
        
        Args:
            returns: numpy array of shape (n_days, n_features)
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        
        # Standardize
        returns_scaled = self.scaler.fit_transform(returns)
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(returns_scaled)
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # Train
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                recon, _ = self.model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if (epoch + 1) % 25 == 0:
                print(f"Epoch {epoch+1}: Loss = {epoch_loss/len(loader):.6f}")
        
        # Compute training errors for threshold calibration
        self.model.eval()
        with torch.no_grad():
            all_data = torch.FloatTensor(returns_scaled).to(device)
            self.training_errors = self.model.anomaly_score(all_data).cpu().numpy()
        
        # Set threshold
        self._calibrate_threshold()
    
    def _calibrate_threshold(self, percentile=95):
        """Calibrate anomaly threshold from training errors."""
        if self.threshold_method == 'percentile':
            self.threshold = np.percentile(self.training_errors, percentile)
        
        elif self.threshold_method == 'rolling_zscore':
            # Store rolling statistics parameters
            self.error_mean = np.mean(self.training_errors)
            self.error_std = np.std(self.training_errors)
            self.threshold = self.error_mean + 2.5 * self.error_std
        
        elif self.threshold_method == 'mad':
            # Median Absolute Deviation (robust to outliers)
            median = np.median(self.training_errors)
            mad = np.median(np.abs(self.training_errors - median))
            self.threshold = median + 3.0 * 1.4826 * mad  # 1.4826 ≈ 1/Φ⁻¹(3/4)
        
        print(f"Anomaly threshold ({self.threshold_method}): {self.threshold:.6f}")
    
    def predict(self, returns):
        """
        Predict anomalies in new data.
        
        Returns:
            scores: reconstruction error per observation
            is_anomaly: boolean array
        """
        device = next(self.model.parameters()).device
        
        returns_scaled = self.scaler.transform(returns)
        returns_tensor = torch.FloatTensor(returns_scaled).to(device)
        
        scores = self.model.anomaly_score(returns_tensor).cpu().numpy()
        is_anomaly = scores > self.threshold
        
        return scores, is_anomaly
```

---

## Part 2: Market Regime Detection

```python
def generate_regime_data(n_assets=20, n_days=2000, seed=42):
    """
    Generate synthetic market data with regime changes.
    
    Normal regime: Low volatility, stable correlations
    Crisis regime: High volatility, correlation breakdown
    """
    np.random.seed(seed)
    
    # Factor structure
    B = np.random.randn(n_assets, 3) * 0.3
    
    returns = np.zeros((n_days, n_assets))
    regimes = np.zeros(n_days, dtype=int)  # 0=normal, 1=crisis
    
    for t in range(n_days):
        # Regime transitions
        if t > 0:
            if regimes[t-1] == 0:
                regimes[t] = 1 if np.random.rand() < 0.005 else 0
            else:
                regimes[t] = 0 if np.random.rand() < 0.02 else 1
        
        if regimes[t] == 0:
            # Normal regime
            factors = np.random.randn(3) * 0.01
            noise = np.random.randn(n_assets) * 0.005
        else:
            # Crisis regime: higher vol, correlated moves
            factors = np.random.randn(3) * 0.03 - 0.01
            noise = np.random.randn(n_assets) * 0.015
        
        returns[t] = B @ factors + noise
    
    return returns, regimes


def detect_regime_changes(returns, regimes, latent_dim=5):
    """
    Use autoencoder anomaly detection to identify regime changes.
    Train on normal regime data, detect crisis periods.
    """
    # Train on normal data only
    normal_mask = regimes == 0
    normal_returns = returns[normal_mask]
    
    # Initialize and train
    detector = AnomalyDetectionSystem(
        n_features=returns.shape[1], 
        latent_dim=latent_dim,
        threshold_method='mad'
    )
    detector.fit(normal_returns, num_epochs=100)
    
    # Score all data
    scores, is_anomaly = detector.predict(returns)
    
    # Evaluation
    true_anomaly = (regimes == 1)
    true_positive = np.sum(is_anomaly & true_anomaly)
    false_positive = np.sum(is_anomaly & ~true_anomaly)
    true_negative = np.sum(~is_anomaly & ~true_anomaly)
    false_negative = np.sum(~is_anomaly & true_anomaly)
    
    precision = true_positive / (true_positive + false_positive + 1e-8)
    recall = true_positive / (true_positive + false_negative + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    print(f"\nRegime Detection Results:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Anomaly scores
    axes[0].plot(scores, alpha=0.7, color='blue', linewidth=0.5)
    axes[0].axhline(detector.threshold, color='red', linestyle='--', 
                    label='Threshold')
    axes[0].set_ylabel('Anomaly Score')
    axes[0].set_title('Reconstruction Error Over Time')
    axes[0].legend()
    
    # True regimes
    axes[1].fill_between(range(len(regimes)), regimes, alpha=0.3, 
                         color='red', label='Crisis')
    axes[1].set_ylabel('True Regime')
    axes[1].legend()
    
    # Detected anomalies
    axes[2].fill_between(range(len(is_anomaly)), is_anomaly.astype(float), 
                         alpha=0.3, color='orange', label='Detected')
    axes[2].set_ylabel('Detected Anomaly')
    axes[2].set_xlabel('Day')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('regime_detection.png', dpi=150)
    plt.show()
    
    return scores, is_anomaly
```

---

## Part 3: Per-Asset Anomaly Decomposition

```python
def decompose_anomaly(model, scaler, returns_t, device):
    """
    Decompose anomaly score into per-asset contributions.
    
    Identifies which assets are driving the anomaly signal,
    enabling targeted investigation.
    """
    model.eval()
    
    returns_scaled = scaler.transform(returns_t.reshape(1, -1))
    x = torch.FloatTensor(returns_scaled).to(device)
    
    with torch.no_grad():
        recon, z = model(x)
        per_asset_error = (x - recon).squeeze().cpu().numpy() ** 2
    
    return per_asset_error


def rolling_anomaly_detection(returns, window=252, step=21, 
                               latent_dim=5):
    """
    Rolling window anomaly detection that adapts to changing
    market conditions.
    
    Retrains the autoencoder periodically on a rolling window
    to capture evolving factor structure.
    """
    n_days, n_assets = returns.shape
    scores = np.full(n_days, np.nan)
    
    for start in range(0, n_days - window, step):
        train_end = start + window
        test_end = min(train_end + step, n_days)
        
        # Train on rolling window
        train_returns = returns[start:train_end]
        test_returns = returns[train_end:test_end]
        
        detector = AnomalyDetectionSystem(
            n_features=n_assets, latent_dim=latent_dim,
            threshold_method='rolling_zscore'
        )
        detector.fit(train_returns, num_epochs=50)
        
        # Score next period
        test_scores, _ = detector.predict(test_returns)
        scores[train_end:test_end] = test_scores
    
    return scores
```

---

## Part 4: Evaluation Metrics

```python
def evaluate_anomaly_detector(scores, true_labels, 
                               thresholds=None):
    """
    Comprehensive evaluation of anomaly detection performance.
    
    Computes ROC curve, precision-recall curve, and optimal
    threshold selection.
    """
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    
    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(
        true_labels, scores
    )
    pr_auc = auc(recall, precision)
    
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    
    # Optimal threshold (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = roc_thresholds[optimal_idx]
    
    print(f"Optimal threshold: {optimal_threshold:.6f}")
    print(f"  TPR: {tpr[optimal_idx]:.4f}, FPR: {fpr[optimal_idx]:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    axes[0].plot([0, 1], [0, 1], 'k--')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend()
    
    axes[1].plot(recall, precision, label=f'AUC = {pr_auc:.3f}')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('anomaly_evaluation.png', dpi=150)
    plt.show()
    
    return roc_auc, pr_auc, optimal_threshold
```

---

## Exercises

### Exercise 1: Threshold Sensitivity
Compare the three threshold methods (percentile, rolling z-score, MAD) on synthetic data with known anomalies. Which method provides the best precision-recall trade-off?

### Exercise 2: Latent Dimension Impact
Train anomaly detectors with different latent dimensions (K = 2, 4, 8, 16). How does the latent dimension affect detection performance? Is there a sweet spot?

### Exercise 3: Rolling vs Static Training
Compare a single static model trained on all historical data vs a rolling window approach that retrains periodically. Which adapts better to regime changes?

### Exercise 4: Ensemble Detection
Train multiple autoencoders with different architectures and combine their anomaly scores. Does ensembling improve detection reliability?

---

## Summary

| Application | Normal Data | Anomaly Signal | Key Challenge |
|-------------|-------------|----------------|---------------|
| **Regime detection** | Normal market | Crisis/dislocation | Non-stationarity |
| **Fraud detection** | Legitimate trades | Unusual patterns | Class imbalance |
| **Outlier identification** | Typical returns | Extreme moves | Fat tails |
| **Model monitoring** | Expected behavior | Model degradation | Concept drift |

**Key Insight:** Autoencoders provide a flexible, unsupervised approach to anomaly detection in finance. By learning the normal factor structure and correlation patterns of market data, they can detect anomalies as deviations from this learned structure — without requiring labeled examples of anomalies, which are rare and heterogeneous in financial markets.
