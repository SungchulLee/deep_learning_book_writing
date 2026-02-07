# MC Dropout Implementation Details

## Overview

This document provides comprehensive implementation guidance for Monte Carlo Dropout, covering architecture patterns, inference procedures, and production considerations.

## Core Implementation Patterns

### Basic MC Dropout Module

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import numpy as np


class MCDropoutLayer(nn.Module):
    """
    Dropout layer that remains active during inference for MC sampling.
    
    Unlike standard nn.Dropout, this layer applies dropout regardless of
    the model's training/eval mode, controlled by an explicit flag.
    """
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p
        self.mc_dropout_enabled = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mc_dropout_enabled and self.p > 0:
            return F.dropout(x, p=self.p, training=True)
        return x
    
    def enable_mc_dropout(self):
        """Enable MC Dropout sampling."""
        self.mc_dropout_enabled = True
    
    def disable_mc_dropout(self):
        """Disable MC Dropout (use mean weights)."""
        self.mc_dropout_enabled = False
```

### Complete MC Dropout Network

```python
class MCDropoutNetwork(nn.Module):
    """
    Neural network with MC Dropout support for uncertainty estimation.
    
    This architecture supports both deterministic inference (dropout disabled)
    and stochastic inference (MC Dropout enabled) with explicit control.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout_rate: float = 0.5,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            
            layers.append(MCDropoutLayer(p=dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer (no dropout after output)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self._mc_dropout_layers = [
            m for m in self.modules() if isinstance(m, MCDropoutLayer)
        ]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def enable_mc_dropout(self):
        """Enable MC Dropout on all dropout layers."""
        for layer in self._mc_dropout_layers:
            layer.enable_mc_dropout()
    
    def disable_mc_dropout(self):
        """Disable MC Dropout on all dropout layers."""
        for layer in self._mc_dropout_layers:
            layer.disable_mc_dropout()
    
    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 100,
        return_samples: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate predictions with uncertainty estimates via MC Dropout.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            n_samples: Number of MC samples (forward passes)
            return_samples: Whether to return all individual samples
            
        Returns:
            mean: Predictive mean, shape (batch_size, output_dim)
            std: Predictive std (epistemic uncertainty), shape (batch_size, output_dim)
            samples: If return_samples=True, all samples (n_samples, batch_size, output_dim)
        """
        self.enable_mc_dropout()
        
        samples = []
        for _ in range(n_samples):
            output = self.forward(x)
            samples.append(output)
        
        # Stack: (n_samples, batch_size, output_dim)
        samples = torch.stack(samples, dim=0)
        
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        
        if return_samples:
            return mean, std, samples
        return mean, std, None
```

## Regression Implementation

### Heteroscedastic Regression

For regression tasks where both epistemic (model) and aleatoric (data) uncertainty matter:

```python
class MCDropoutHeteroscedasticRegression(nn.Module):
    """
    MC Dropout regression model that predicts both mean and variance.
    
    The network outputs [mean, log_variance] and learns to estimate
    aleatoric uncertainty alongside the prediction.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout_rate: float = 0.5,
        min_log_var: float = -10.0,
        max_log_var: float = 10.0
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var
        
        # Shared feature extractor
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                MCDropoutLayer(p=dropout_rate)
            ])
            prev_dim = hidden_dim
        self.features = nn.Sequential(*layers)
        
        # Separate heads for mean and variance
        self.mean_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            MCDropoutLayer(p=dropout_rate),
            nn.Linear(hidden_dims[-1], output_dim)
        )
        
        self.var_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            MCDropoutLayer(p=dropout_rate),
            nn.Linear(hidden_dims[-1], output_dim)
        )
        
        self._mc_dropout_layers = [
            m for m in self.modules() if isinstance(m, MCDropoutLayer)
        ]
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean: Predicted mean
            log_var: Log variance (clamped for numerical stability)
        """
        features = self.features(x)
        mean = self.mean_head(features)
        log_var = self.var_head(features)
        
        # Clamp for numerical stability
        log_var = torch.clamp(log_var, self.min_log_var, self.max_log_var)
        
        return mean, log_var
    
    def enable_mc_dropout(self):
        for layer in self._mc_dropout_layers:
            layer.enable_mc_dropout()
    
    def disable_mc_dropout(self):
        for layer in self._mc_dropout_layers:
            layer.disable_mc_dropout()
    
    @torch.no_grad()
    def predict_with_full_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 100
    ) -> dict:
        """
        Compute full uncertainty decomposition.
        
        Returns dict with:
            - mean: Predictive mean
            - epistemic_var: Variance from model uncertainty
            - aleatoric_var: Variance from data noise (averaged)
            - total_var: epistemic_var + aleatoric_var
            - total_std: sqrt(total_var)
        """
        self.enable_mc_dropout()
        
        means = []
        log_vars = []
        
        for _ in range(n_samples):
            mean, log_var = self.forward(x)
            means.append(mean)
            log_vars.append(log_var)
        
        means = torch.stack(means, dim=0)      # (T, B, D)
        log_vars = torch.stack(log_vars, dim=0)  # (T, B, D)
        
        # Predictive mean: E[mean]
        pred_mean = means.mean(dim=0)
        
        # Epistemic uncertainty: Var[mean]
        epistemic_var = means.var(dim=0)
        
        # Aleatoric uncertainty: E[variance]
        aleatoric_var = log_vars.exp().mean(dim=0)
        
        # Total uncertainty
        total_var = epistemic_var + aleatoric_var
        
        return {
            'mean': pred_mean,
            'epistemic_var': epistemic_var,
            'aleatoric_var': aleatoric_var,
            'total_var': total_var,
            'total_std': torch.sqrt(total_var)
        }


def heteroscedastic_loss(
    mean: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Negative log-likelihood loss for heteroscedastic regression.
    
    NLL = 0.5 * [log(var) + (y - mean)^2 / var]
        = 0.5 * [log_var + (y - mean)^2 * exp(-log_var)]
    """
    precision = torch.exp(-log_var)
    return 0.5 * (log_var + precision * (target - mean) ** 2).mean()
```

## Classification Implementation

### MC Dropout Classifier with Uncertainty Metrics

```python
class MCDropoutClassifier(nn.Module):
    """
    MC Dropout classifier with comprehensive uncertainty quantification.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                MCDropoutLayer(p=dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
        
        self._mc_dropout_layers = [
            m for m in self.modules() if isinstance(m, MCDropoutLayer)
        ]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits."""
        return self.network(x)
    
    def enable_mc_dropout(self):
        for layer in self._mc_dropout_layers:
            layer.enable_mc_dropout()
    
    def disable_mc_dropout(self):
        for layer in self._mc_dropout_layers:
            layer.disable_mc_dropout()
    
    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 100
    ) -> dict:
        """
        Classification with full uncertainty decomposition.
        
        Returns:
            - probs: Mean predictive probabilities
            - pred_class: Predicted class (argmax of probs)
            - confidence: Max probability (proxy for confidence)
            - predictive_entropy: Total uncertainty H[y|x,D]
            - mutual_information: Epistemic uncertainty I[y,w|x,D]
            - expected_entropy: Aleatoric uncertainty E[H[y|x,w]]
        """
        self.enable_mc_dropout()
        
        all_probs = []
        
        for _ in range(n_samples):
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs)
        
        # (T, B, C)
        all_probs = torch.stack(all_probs, dim=0)
        
        # Mean predictive distribution: p(y|x,D) ≈ (1/T) Σ p(y|x,w_t)
        mean_probs = all_probs.mean(dim=0)  # (B, C)
        
        # Predictions
        pred_class = mean_probs.argmax(dim=-1)
        confidence = mean_probs.max(dim=-1).values
        
        # Predictive entropy: H[y|x,D] = -Σ p_c log p_c
        predictive_entropy = -torch.sum(
            mean_probs * torch.log(mean_probs + 1e-10), dim=-1
        )
        
        # Expected entropy: E[H[y|x,w]] = (1/T) Σ H[y|x,w_t]
        sample_entropies = -torch.sum(
            all_probs * torch.log(all_probs + 1e-10), dim=-1
        )  # (T, B)
        expected_entropy = sample_entropies.mean(dim=0)  # (B,)
        
        # Mutual information: I[y,w|x,D] = H[y|x,D] - E[H[y|x,w]]
        mutual_information = predictive_entropy - expected_entropy
        
        return {
            'probs': mean_probs,
            'pred_class': pred_class,
            'confidence': confidence,
            'predictive_entropy': predictive_entropy,
            'mutual_information': mutual_information,
            'expected_entropy': expected_entropy
        }
```

## Convolutional Networks

### MC Dropout for CNNs with Spatial Dropout

```python
class MCSpatialDropout2d(nn.Module):
    """
    Spatial dropout that remains active during inference.
    Drops entire feature maps for spatial consistency.
    """
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self.mc_dropout_enabled = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        """
        if self.mc_dropout_enabled and self.p > 0:
            return F.dropout2d(x, p=self.p, training=True)
        return x
    
    def enable_mc_dropout(self):
        self.mc_dropout_enabled = True
    
    def disable_mc_dropout(self):
        self.mc_dropout_enabled = False


class MCDropoutCNN(nn.Module):
    """
    Convolutional network with MC Dropout for image classification.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        dropout_rate: float = 0.25
    ):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            MCSpatialDropout2d(p=dropout_rate),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            MCSpatialDropout2d(p=dropout_rate),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            MCDropoutLayer(p=dropout_rate * 2),  # Higher rate for FC layers
            nn.Linear(256, num_classes)
        )
        
        self._mc_dropout_layers = [
            m for m in self.modules() 
            if isinstance(m, (MCDropoutLayer, MCSpatialDropout2d))
        ]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
    
    def enable_mc_dropout(self):
        for layer in self._mc_dropout_layers:
            layer.enable_mc_dropout()
    
    def disable_mc_dropout(self):
        for layer in self._mc_dropout_layers:
            layer.disable_mc_dropout()
```

## Efficient Batch Inference

### Parallelized MC Sampling

For production, parallelized sampling is more efficient than sequential loops:

```python
class EfficientMCDropout:
    """
    Efficient MC Dropout inference using batch expansion.
    
    Instead of T sequential forward passes, we expand the batch
    and perform a single forward pass with T * batch_size samples.
    """
    
    @staticmethod
    @torch.no_grad()
    def predict_efficient(
        model: nn.Module,
        x: torch.Tensor,
        n_samples: int = 100,
        max_batch_size: int = 1024
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Efficient parallel MC sampling.
        
        Args:
            model: Model with MC Dropout enabled
            x: Input tensor (B, ...)
            n_samples: Number of MC samples
            max_batch_size: Maximum expanded batch size for memory efficiency
            
        Returns:
            mean: (B, output_dim)
            std: (B, output_dim)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Determine chunking strategy
        expanded_batch = batch_size * n_samples
        
        if expanded_batch <= max_batch_size:
            # Single pass: expand batch, forward, reshape
            x_expanded = x.unsqueeze(0).expand(n_samples, -1, *x.shape[1:])
            x_expanded = x_expanded.reshape(-1, *x.shape[1:])
            
            outputs = model(x_expanded)  # (T*B, D)
            outputs = outputs.view(n_samples, batch_size, -1)  # (T, B, D)
            
        else:
            # Chunked processing
            outputs = []
            samples_per_chunk = max(1, max_batch_size // batch_size)
            
            for start in range(0, n_samples, samples_per_chunk):
                end = min(start + samples_per_chunk, n_samples)
                chunk_samples = end - start
                
                x_chunk = x.unsqueeze(0).expand(chunk_samples, -1, *x.shape[1:])
                x_chunk = x_chunk.reshape(-1, *x.shape[1:])
                
                out_chunk = model(x_chunk)
                out_chunk = out_chunk.view(chunk_samples, batch_size, -1)
                outputs.append(out_chunk)
            
            outputs = torch.cat(outputs, dim=0)  # (T, B, D)
        
        mean = outputs.mean(dim=0)
        std = outputs.std(dim=0)
        
        return mean, std


class MCDropoutEnsemble:
    """
    Wrapper for ensemble-style inference combining multiple checkpoints.
    
    Provides uncertainty estimates by combining:
    1. Within-model uncertainty (MC Dropout)
    2. Between-model uncertainty (different initializations)
    """
    
    def __init__(self, models: List[nn.Module]):
        self.models = models
        for model in self.models:
            model.eval()
    
    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        n_samples_per_model: int = 20
    ) -> dict:
        """
        Ensemble prediction with decomposed uncertainty.
        """
        all_outputs = []
        
        for model in self.models:
            model.enable_mc_dropout()
            
            for _ in range(n_samples_per_model):
                output = model(x)
                all_outputs.append(output)
        
        # (M * T, B, D)
        all_outputs = torch.stack(all_outputs, dim=0)
        
        mean = all_outputs.mean(dim=0)
        total_var = all_outputs.var(dim=0)
        
        # Decomposition requires per-model means
        n_models = len(self.models)
        per_model_outputs = all_outputs.view(n_models, n_samples_per_model, *x.shape[:-1], -1)
        per_model_means = per_model_outputs.mean(dim=1)  # (M, B, D)
        per_model_vars = per_model_outputs.var(dim=1)    # (M, B, D)
        
        # Between-model variance
        between_var = per_model_means.var(dim=0)
        
        # Within-model variance (averaged)
        within_var = per_model_vars.mean(dim=0)
        
        return {
            'mean': mean,
            'total_var': total_var,
            'between_model_var': between_var,
            'within_model_var': within_var
        }
```

## Training Utilities

### Training Loop with Validation Uncertainty Monitoring

```python
def train_mc_dropout_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    mc_samples_val: int = 50,
    device: str = 'cuda'
):
    """
    Training loop with uncertainty-aware validation metrics.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [], 'val_loss': [], 'val_acc': [],
        'val_predictive_entropy': [], 'val_mutual_info': []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        model.enable_mc_dropout()
        train_loss = 0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation with uncertainty
        model.eval()
        val_loss, correct, total = 0, 0, 0
        all_entropies, all_mi = [], []
        
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            # Standard loss (no dropout)
            model.disable_mc_dropout()
            with torch.no_grad():
                logits = model(x_batch)
                val_loss += criterion(logits, y_batch).item()
            
            # MC Dropout predictions
            if hasattr(model, 'predict_with_uncertainty'):
                results = model.predict_with_uncertainty(x_batch, n_samples=mc_samples_val)
                pred = results['pred_class']
                all_entropies.append(results['predictive_entropy'])
                all_mi.append(results['mutual_information'])
            else:
                pred = logits.argmax(dim=-1)
            
            correct += (pred == y_batch).sum().item()
            total += y_batch.size(0)
        
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(correct / total)
        
        if all_entropies:
            history['val_predictive_entropy'].append(
                torch.cat(all_entropies).mean().item()
            )
            history['val_mutual_info'].append(
                torch.cat(all_mi).mean().item()
            )
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {history['train_loss'][-1]:.4f}")
            print(f"  Val Loss: {history['val_loss'][-1]:.4f}, Acc: {history['val_acc'][-1]:.4f}")
            if all_entropies:
                print(f"  Entropy: {history['val_predictive_entropy'][-1]:.4f}, "
                      f"MI: {history['val_mutual_info'][-1]:.4f}")
    
    return history
```

## References

1. Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation. *ICML*.

2. Kendall, A., & Gal, Y. (2017). What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? *NeurIPS*.

3. Lakshminarayanan, B., et al. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. *NeurIPS*.
