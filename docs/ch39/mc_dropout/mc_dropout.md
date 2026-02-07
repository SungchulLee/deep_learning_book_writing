# Monte Carlo Dropout

## Introduction

Monte Carlo (MC) Dropout provides a simple yet powerful way to estimate uncertainty in neural networks. The key insight, established by Gal and Ghahramani (2016), is that dropout—typically used as regularization during training—can be reinterpreted as approximate Bayesian inference. By keeping dropout active during inference and running multiple forward passes, we can estimate the posterior predictive distribution without the computational burden of full Bayesian neural networks.

This technique bridges the gap between practical deep learning and principled uncertainty quantification, making it one of the most widely adopted methods for epistemic uncertainty estimation.

## Theoretical Foundation

### Dropout as Variational Inference

Standard dropout randomly zeroes activations during training:

$$\tilde{h}_j = z_j \cdot h_j \quad \text{where} \quad z_j \sim \text{Bernoulli}(1-p)$$

where $p$ is the dropout probability and $h_j$ is the activation at unit $j$.

Gal and Ghahramani showed that a neural network with dropout trained via SGD minimizes:

$$\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{N}\sum_{i=1}^N \mathcal{L}_{\text{data}}(y_i, f^{\hat{\mathbf{w}}}(\mathbf{x}_i)) + \lambda \|\boldsymbol{\theta}\|^2$$

This objective is equivalent to minimizing the KL divergence between an approximate posterior $q(\mathbf{w})$ and the true posterior $p(\mathbf{w}|\mathcal{D})$ in variational inference:

$$\mathcal{L}_{\text{VI}} = \text{KL}\left[q(\mathbf{w}) \| p(\mathbf{w})\right] - \mathbb{E}_{q(\mathbf{w})}\left[\log p(\mathcal{D}|\mathbf{w})\right]$$

### The Approximate Posterior

The key insight is that dropout implicitly defines a variational distribution:

$$q(\mathbf{W}_l) = \mathbf{M}_l \cdot \text{diag}(\mathbf{z}_l)$$

where:

- $\mathbf{M}_l \in \mathbb{R}^{K_{l-1} \times K_l}$ is the learned weight matrix for layer $l$
- $\mathbf{z}_l \in \{0, 1\}^{K_{l-1}}$ is a vector of Bernoulli random variables
- $z_{l,j} \sim \text{Bernoulli}(1-p)$ for each unit $j$

The full approximate posterior factorizes across layers:

$$q(\mathbf{w}) = \prod_{l=1}^L q(\mathbf{W}_l)$$

Each forward pass with dropout active samples a different set of weights from this distribution, effectively sampling from the approximate posterior.

### Connection to Gaussian Processes

Gal (2016) further showed that infinitely wide neural networks with dropout converge to Gaussian Processes. This provides theoretical grounding for the uncertainty estimates—they approximate the posterior variance of a GP with a specific kernel determined by the network architecture.

## MC Dropout Algorithm

### Predictive Distribution

At test time, instead of disabling dropout, we keep it active and perform $T$ stochastic forward passes. The predictive distribution is approximated by Monte Carlo integration:

$$p(y^*|\mathbf{x}^*, \mathcal{D}) = \int p(y^*|\mathbf{x}^*, \mathbf{w}) p(\mathbf{w}|\mathcal{D}) d\mathbf{w} \approx \frac{1}{T}\sum_{t=1}^T p(y^*|\mathbf{x}^*, \hat{\mathbf{w}}_t)$$

where $\hat{\mathbf{w}}_t \sim q(\mathbf{w})$ are weights sampled via dropout in forward pass $t$.

### Uncertainty Estimates

**For classification**, the predictive mean (class probabilities):

$$\mathbb{E}[y^*|\mathbf{x}^*] \approx \bar{p} = \frac{1}{T}\sum_{t=1}^T \text{softmax}(f(\mathbf{x}^*, \hat{\mathbf{w}}_t))$$

**Epistemic uncertainty** (model uncertainty) via predictive variance:

$$\text{Var}[y^*|\mathbf{x}^*] \approx \frac{1}{T}\sum_{t=1}^T \hat{p}_t^2 - \bar{p}^2$$

where $\hat{p}_t = \text{softmax}(f(\mathbf{x}^*, \hat{\mathbf{w}}_t))$.

**Predictive entropy** provides another uncertainty measure:

$$\mathbb{H}[\bar{p}] = -\sum_c \bar{p}_c \log \bar{p}_c$$

**For regression**, assume Gaussian likelihood $p(y|\mathbf{x}, \mathbf{w}) = \mathcal{N}(y; f(\mathbf{x}, \mathbf{w}), \sigma^2)$:

$$\mathbb{E}[y^*|\mathbf{x}^*] \approx \frac{1}{T}\sum_{t=1}^T f(\mathbf{x}^*, \hat{\mathbf{w}}_t)$$

$$\text{Var}[y^*|\mathbf{x}^*] \approx \sigma^2 + \frac{1}{T}\sum_{t=1}^T f(\mathbf{x}^*, \hat{\mathbf{w}}_t)^2 - \left(\frac{1}{T}\sum_{t=1}^T f(\mathbf{x}^*, \hat{\mathbf{w}}_t)\right)^2$$

The first term ($\sigma^2$) captures aleatoric uncertainty (data noise), while the remaining terms capture epistemic uncertainty (model uncertainty).

### Decomposing Uncertainty

Total predictive uncertainty can be decomposed:

$$\underbrace{\text{Var}[y^*|\mathbf{x}^*]}_{\text{Total}} = \underbrace{\mathbb{E}_{q(\mathbf{w})}[\text{Var}[y^*|\mathbf{x}^*, \mathbf{w}]]}_{\text{Aleatoric}} + \underbrace{\text{Var}_{q(\mathbf{w})}[\mathbb{E}[y^*|\mathbf{x}^*, \mathbf{w}]]}_{\text{Epistemic}}$$

MC Dropout primarily estimates the epistemic component—uncertainty due to limited training data that could be reduced with more observations.

## PyTorch Implementation

### Core MC Dropout Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple, List, Optional, Dict
import numpy as np


class MCDropoutModel(nn.Module):
    """
    Neural network with Monte Carlo Dropout for uncertainty estimation.
    
    Key differences from standard model:
    1. Dropout is applied after activations (standard practice)
    2. Dropout remains active during inference via enable_dropout()
    3. Multiple forward passes yield uncertainty estimates
    
    Theoretical basis: Gal & Ghahramani (2016) showed dropout training
    approximates variational inference with a specific posterior.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: List[int], 
        output_dim: int, 
        dropout_rate: float = 0.5,
        activation: str = 'relu'
    ):
        """
        Initialize MC Dropout model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer sizes
            output_dim: Number of output classes (classification) or 1 (regression)
            dropout_rate: Dropout probability (typically 0.1-0.5)
            activation: Activation function ('relu', 'tanh', 'elu')
        """
        super().__init__()
        
        self.dropout_rate = dropout_rate
        self.output_dim = output_dim
        
        # Select activation
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'elu': nn.ELU()
        }
        act_fn = activations.get(activation, nn.ReLU())
        
        # Build network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                act_fn,
                nn.Dropout(p=dropout_rate)  # Dropout after activation
            ])
            prev_dim = hidden_dim
        
        # Output layer (no dropout)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass.
        
        Note: Dropout behavior depends on model.train()/eval() mode
        unless enable_dropout() is called.
        """
        return self.network(x)
    
    def enable_dropout(self):
        """
        Enable dropout for MC sampling during inference.
        
        This is the KEY step for MC Dropout:
        - Standard: model.eval() disables dropout
        - MC Dropout: keep dropout active for sampling
        """
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Force dropout to training mode
    
    def disable_dropout(self):
        """Disable dropout (standard inference mode)."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.eval()
    
    def mc_predict(
        self, 
        x: torch.Tensor, 
        n_samples: int = 100,
        return_samples: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Monte Carlo prediction with uncertainty estimation.
        
        Algorithm:
            1. Enable dropout
            2. Run T forward passes with different dropout masks
            3. Compute statistics of predictions
        
        Args:
            x: Input tensor (batch_size, input_dim)
            n_samples: Number of MC samples (recommended: 50-100)
            return_samples: If True, return all individual predictions
        
        Returns:
            mean_probs: Mean predicted probabilities (batch_size, n_classes)
            uncertainty: Epistemic uncertainty estimate (batch_size,)
            (optional) samples: All T predictions (n_samples, batch_size, n_classes)
        """
        self.enable_dropout()
        
        # Collect predictions from multiple forward passes
        all_predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward(x)
                probs = F.softmax(logits, dim=-1)
                all_predictions.append(probs)
        
        # Stack: (n_samples, batch_size, n_classes)
        all_predictions = torch.stack(all_predictions)
        
        # Compute statistics
        mean_probs = all_predictions.mean(dim=0)  # (batch_size, n_classes)
        
        # Epistemic uncertainty: variance across MC samples
        variance = all_predictions.var(dim=0)  # (batch_size, n_classes)
        uncertainty = variance.mean(dim=-1)  # (batch_size,) average over classes
        
        self.disable_dropout()
        
        if return_samples:
            return mean_probs, uncertainty, all_predictions
        return mean_probs, uncertainty, None
    
    def mc_predict_regression(
        self, 
        x: torch.Tensor, 
        n_samples: int = 100,
        noise_variance: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MC prediction for regression with uncertainty decomposition.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            n_samples: Number of MC samples
            noise_variance: Assumed observation noise σ² (aleatoric)
        
        Returns:
            mean: Predictive mean (batch_size, output_dim)
            epistemic_var: Epistemic (model) uncertainty
            total_var: Total predictive variance
        """
        self.enable_dropout()
        
        all_predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                all_predictions.append(pred)
        
        all_predictions = torch.stack(all_predictions)  # (T, batch, output_dim)
        
        mean = all_predictions.mean(dim=0)
        epistemic_var = all_predictions.var(dim=0)
        total_var = epistemic_var + noise_variance  # Add aleatoric component
        
        self.disable_dropout()
        
        return mean, epistemic_var, total_var
    
    def predict_with_entropy(
        self, 
        x: torch.Tensor, 
        n_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prediction with entropy-based uncertainty.
        
        Returns:
            predictions: Predicted class labels
            confidence: Max probability (confidence)
            entropy: Predictive entropy (uncertainty)
        """
        mean_probs, _, _ = self.mc_predict(x, n_samples)
        
        confidence, predictions = mean_probs.max(dim=-1)
        
        # Entropy: H = -Σ p log p
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=-1)
        
        return predictions, confidence, entropy
    
    def mutual_information(
        self, 
        x: torch.Tensor, 
        n_samples: int = 100
    ) -> torch.Tensor:
        """
        Compute mutual information I(y; w | x, D) as uncertainty measure.
        
        MI = H[E_w[p(y|x,w)]] - E_w[H[p(y|x,w)]]
           = (entropy of mean) - (mean of entropies)
        
        High MI indicates the model is uncertain about which class,
        specifically due to model uncertainty (not data noise).
        """
        mean_probs, _, all_preds = self.mc_predict(x, n_samples, return_samples=True)
        
        # Entropy of the mean prediction
        entropy_of_mean = -torch.sum(
            mean_probs * torch.log(mean_probs + 1e-10), dim=-1
        )
        
        # Mean of individual entropies
        individual_entropies = -torch.sum(
            all_preds * torch.log(all_preds + 1e-10), dim=-1
        )  # (n_samples, batch_size)
        mean_of_entropies = individual_entropies.mean(dim=0)
        
        # Mutual information
        mi = entropy_of_mean - mean_of_entropies
        
        return mi
```

### Training Function

```python
def train_mc_dropout_model(
    model: MCDropoutModel, 
    train_loader: DataLoader,
    epochs: int = 10,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    device: str = 'cpu'
) -> Dict[str, List[float]]:
    """
    Train MC Dropout model (standard training procedure).
    
    Note: Training is identical to standard neural network.
    The L2 regularization (weight_decay) corresponds to the prior
    in the Bayesian interpretation.
    MC sampling only happens at inference time.
    
    Args:
        model: MCDropoutModel instance
        train_loader: Training data loader
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization (prior precision in Bayesian view)
        device: Device to train on
    
    Returns:
        Dictionary with training metrics
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    history = {'loss': [], 'accuracy': []}
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # Flatten if image data
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * x.size(0)
            _, predicted = logits.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)
        
        avg_loss = epoch_loss / total
        accuracy = correct / total
        history['loss'].append(avg_loss)
        history['accuracy'].append(accuracy)
        
        print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}, Acc = {accuracy:.4f}")
    
    return history
```

### Evaluation and Uncertainty Quality

```python
def evaluate_uncertainty_quality(
    model: MCDropoutModel,
    test_loader: DataLoader,
    n_samples: int = 100,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Evaluate uncertainty quality on test set.
    
    Good uncertainty should:
    1. Be higher for wrong predictions
    2. Correlate with error (calibration)
    3. Enable selective prediction (reject uncertain samples)
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_uncertainty = []
    all_confidence = []
    all_entropy = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            
            mean_probs, uncertainty, _ = model.mc_predict(x, n_samples)
            
            confidence, preds = mean_probs.max(dim=-1)
            entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=-1)
            
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
            all_uncertainty.append(uncertainty.cpu())
            all_confidence.append(confidence.cpu())
            all_entropy.append(entropy.cpu())
    
    # Aggregate
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    uncertainty = torch.cat(all_uncertainty)
    confidence = torch.cat(all_confidence)
    entropy = torch.cat(all_entropy)
    
    correct = (preds == labels)
    accuracy = correct.float().mean().item()
    
    # Uncertainty quality metrics
    unc_correct = uncertainty[correct].mean().item()
    unc_wrong = uncertainty[~correct].mean().item()
    
    # Selective prediction: accuracy when rejecting high-uncertainty samples
    threshold = uncertainty.median()
    confident_mask = uncertainty < threshold
    selective_acc = correct[confident_mask].float().mean().item() if confident_mask.sum() > 0 else 0.0
    
    results = {
        'accuracy': accuracy,
        'uncertainty_correct': unc_correct,
        'uncertainty_wrong': unc_wrong,
        'uncertainty_separation': unc_wrong - unc_correct,
        'selective_accuracy': selective_acc,
        'coverage': confident_mask.float().mean().item(),
        'mean_entropy': entropy.mean().item()
    }
    
    return results
```

## Detailed Example: MNIST Classification

```python
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split


def mc_dropout_mnist_example():
    """
    Complete MC Dropout example on MNIST demonstrating
    uncertainty estimation and evaluation.
    """
    # Setup
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Use subset for demo
    train_subset, _ = random_split(train_dataset, [10000, 50000])
    test_subset, _ = random_split(test_dataset, [2000, 8000])
    
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=256, shuffle=False)
    
    # Create model
    model = MCDropoutModel(
        input_dim=28*28,
        hidden_dims=[512, 256, 128],
        output_dim=10,
        dropout_rate=0.3
    ).to(device)
    
    print("=" * 60)
    print("MC Dropout on MNIST")
    print("=" * 60)
    print(f"Architecture: 784 → 512 → 256 → 128 → 10")
    print(f"Dropout rate: 0.3")
    print(f"Training samples: {len(train_subset)}")
    print(f"Test samples: {len(test_subset)}")
    
    # Train
    print("\nTraining...")
    history = train_mc_dropout_model(model, train_loader, epochs=5, device=device)
    
    # Evaluate with uncertainty
    print("\nEvaluating with MC Dropout (100 samples)...")
    results = evaluate_uncertainty_quality(model, test_loader, n_samples=100, device=device)
    
    print(f"\nResults:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Mean entropy: {results['mean_entropy']:.4f}")
    print(f"\nUncertainty Analysis:")
    print(f"  Correct predictions - Uncertainty: {results['uncertainty_correct']:.6f}")
    print(f"  Wrong predictions - Uncertainty: {results['uncertainty_wrong']:.6f}")
    print(f"  Separation (wrong - correct): {results['uncertainty_separation']:.6f}")
    print(f"\nSelective Prediction:")
    print(f"  Accuracy on confident samples: {results['selective_accuracy']:.4f}")
    print(f"  Coverage: {results['coverage']:.2%}")
    
    return model, results


# Out-of-Distribution Detection Example
def ood_detection_example(model: MCDropoutModel, device: str = 'cpu'):
    """
    Use MC Dropout uncertainty to detect out-of-distribution samples.
    
    Setup: Train on MNIST, test on Fashion-MNIST (OOD)
    Expected: Higher uncertainty on Fashion-MNIST
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Use MNIST stats
    ])
    
    # In-distribution: MNIST test set
    mnist_test = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    mnist_loader = DataLoader(mnist_test, batch_size=256, shuffle=False)
    
    # Out-of-distribution: Fashion-MNIST
    fashion_test = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )
    fashion_loader = DataLoader(fashion_test, batch_size=256, shuffle=False)
    
    def get_uncertainties(loader):
        uncertainties = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device).view(x.size(0), -1)
                _, unc, _ = model.mc_predict(x, n_samples=50)
                uncertainties.append(unc.cpu())
        return torch.cat(uncertainties)
    
    print("\nOOD Detection Analysis:")
    print("-" * 40)
    
    mnist_unc = get_uncertainties(mnist_loader)
    fashion_unc = get_uncertainties(fashion_loader)
    
    print(f"MNIST (in-dist) uncertainty: {mnist_unc.mean():.6f} ± {mnist_unc.std():.6f}")
    print(f"Fashion (OOD) uncertainty: {fashion_unc.mean():.6f} ± {fashion_unc.std():.6f}")
    print(f"Ratio (OOD/in-dist): {fashion_unc.mean() / mnist_unc.mean():.2f}x")
    
    # AUROC for OOD detection
    from sklearn.metrics import roc_auc_score
    labels = torch.cat([torch.zeros(len(mnist_unc)), torch.ones(len(fashion_unc))])
    scores = torch.cat([mnist_unc, fashion_unc])
    auroc = roc_auc_score(labels.numpy(), scores.numpy())
    print(f"OOD Detection AUROC: {auroc:.4f}")
    
    return mnist_unc, fashion_unc, auroc
```

## Effect of Number of MC Samples

```python
def analyze_mc_sample_convergence(
    model: MCDropoutModel, 
    x: torch.Tensor,
    sample_counts: List[int] = [5, 10, 25, 50, 100, 200, 500]
) -> List[Dict]:
    """
    Analyze how uncertainty estimates converge with more samples.
    
    Key finding: 50-100 samples typically sufficient for stable estimates.
    The law of large numbers guarantees convergence, but practical
    considerations (latency, compute) determine the optimal count.
    """
    import time
    
    print("Convergence Analysis: MC Sample Count")
    print("=" * 60)
    
    results = []
    
    # Reference: many samples
    model.enable_dropout()
    with torch.no_grad():
        ref_preds = []
        for _ in range(1000):
            logits = model(x)
            probs = F.softmax(logits, dim=-1)
            ref_preds.append(probs)
        ref_preds = torch.stack(ref_preds)
        ref_mean = ref_preds.mean(dim=0)
        ref_var = ref_preds.var(dim=0).mean(dim=-1)
    
    print(f"{'N Samples':<12} {'Mean MAE':<15} {'Var MAE':<15} {'Time (ms)':<15}")
    print("-" * 60)
    
    for n in sample_counts:
        start = time.time()
        mean_probs, uncertainty, _ = model.mc_predict(x, n_samples=n)
        elapsed_ms = (time.time() - start) * 1000
        
        # Error from reference
        mean_error = (mean_probs - ref_mean).abs().mean().item()
        var_error = (uncertainty - ref_var).abs().mean().item()
        
        print(f"{n:<12} {mean_error:<15.6f} {var_error:<15.6f} {elapsed_ms:<15.2f}")
        results.append({
            'n_samples': n,
            'mean_error': mean_error,
            'var_error': var_error,
            'time_ms': elapsed_ms
        })
    
    print("\nRecommendation: Use 50-100 samples for production")
    print("For quick prototyping: 20-30 samples may suffice")
    
    model.disable_dropout()
    return results
```

## Uncertainty Visualization

```python
import matplotlib.pyplot as plt


def visualize_mc_dropout_uncertainty(
    model: MCDropoutModel, 
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    n_samples: int = 100
):
    """
    Visualize MC Dropout uncertainty on test samples.
    """
    # Get predictions
    mean_probs, uncertainty, all_preds = model.mc_predict(
        test_images.view(test_images.size(0), -1), 
        n_samples=n_samples,
        return_samples=True
    )
    
    preds = mean_probs.argmax(dim=-1)
    confidence = mean_probs.max(dim=-1)[0]
    correct = (preds == test_labels)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    # Top row: High uncertainty (uncertain predictions)
    uncertain_idx = uncertainty.argsort(descending=True)[:5]
    for i, idx in enumerate(uncertain_idx):
        ax = axes[0, i]
        ax.imshow(test_images[idx].squeeze(), cmap='gray')
        color = 'green' if correct[idx] else 'red'
        ax.set_title(
            f'Pred: {preds[idx].item()}\n'
            f'True: {test_labels[idx].item()}\n'
            f'Unc: {uncertainty[idx]:.4f}',
            color=color
        )
        ax.axis('off')
    axes[0, 0].set_ylabel('High Uncertainty', fontsize=12)
    
    # Bottom row: Low uncertainty (confident predictions)
    confident_idx = uncertainty.argsort()[:5]
    for i, idx in enumerate(confident_idx):
        ax = axes[1, i]
        ax.imshow(test_images[idx].squeeze(), cmap='gray')
        color = 'green' if correct[idx] else 'red'
        ax.set_title(
            f'Pred: {preds[idx].item()}\n'
            f'True: {test_labels[idx].item()}\n'
            f'Unc: {uncertainty[idx]:.4f}',
            color=color
        )
        ax.axis('off')
    axes[1, 0].set_ylabel('Low Uncertainty', fontsize=12)
    
    plt.suptitle('MC Dropout Uncertainty Visualization', fontsize=14)
    plt.tight_layout()
    return fig


def plot_uncertainty_distribution(uncertainty: torch.Tensor, correct: torch.Tensor):
    """
    Plot uncertainty distribution for correct vs wrong predictions.
    
    A well-calibrated model should show:
    - Lower uncertainty for correct predictions
    - Higher uncertainty for wrong predictions
    - Clear separation between the two distributions
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    unc_correct = uncertainty[correct].numpy()
    unc_wrong = uncertainty[~correct].numpy()
    
    bins = np.linspace(0, uncertainty.max().item(), 50)
    
    ax.hist(unc_correct, bins=bins, alpha=0.6, label='Correct', color='green', density=True)
    ax.hist(unc_wrong, bins=bins, alpha=0.6, label='Wrong', color='red', density=True)
    
    ax.axvline(np.mean(unc_correct), color='green', linestyle='--', 
               linewidth=2, label=f'Correct mean: {np.mean(unc_correct):.4f}')
    ax.axvline(np.mean(unc_wrong), color='red', linestyle='--', 
               linewidth=2, label=f'Wrong mean: {np.mean(unc_wrong):.4f}')
    
    ax.set_xlabel('Uncertainty', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Uncertainty Distribution: Correct vs Wrong Predictions', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_reliability_diagram(
    confidence: torch.Tensor, 
    correct: torch.Tensor, 
    n_bins: int = 10
):
    """
    Plot reliability diagram for calibration analysis.
    
    Perfect calibration: diagonal line (confidence = accuracy)
    Below diagonal: overconfident
    Above diagonal: underconfident
    """
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    
    accuracies = []
    confidences = []
    counts = []
    
    for i in range(n_bins):
        mask = (confidence >= bin_boundaries[i]) & (confidence < bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = correct[mask].float().mean().item()
            bin_conf = confidence[mask].mean().item()
            accuracies.append(bin_acc)
            confidences.append(bin_conf)
            counts.append(mask.sum().item())
        else:
            accuracies.append(0)
            confidences.append((bin_boundaries[i] + bin_boundaries[i + 1]).item() / 2)
            counts.append(0)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Bar chart for accuracy
    bin_centers = [(bin_boundaries[i] + bin_boundaries[i + 1]).item() / 2 for i in range(n_bins)]
    width = 0.08
    ax.bar(bin_centers, accuracies, width=width, alpha=0.7, label='Actual accuracy')
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Reliability Diagram', fontsize=13)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # ECE calculation
    total = sum(counts)
    ece = sum(c * abs(a - conf) for a, conf, c in zip(accuracies, confidences, counts)) / total
    ax.text(0.05, 0.95, f'ECE = {ece:.4f}', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return fig, ece
```

## Dropout Rate Selection

The dropout rate $p$ significantly affects uncertainty quality:

| Dropout Rate | Effect | Use Case |
|-------------|--------|----------|
| $p < 0.1$ | Too little variation → underestimate uncertainty | Not recommended |
| $p \approx 0.1-0.2$ | Conservative, good for well-calibrated models | Production systems |
| $p \approx 0.2-0.3$ | Good balance for most tasks | Default choice |
| $p \approx 0.5$ | Standard dropout, may overestimate uncertainty | Research/exploration |
| $p > 0.5$ | Too much noise → poor predictions | Not recommended |

```python
def compare_dropout_rates(
    input_dim: int, 
    hidden_dims: List[int], 
    output_dim: int, 
    train_loader: DataLoader,
    test_loader: DataLoader,
    dropout_rates: List[float] = [0.1, 0.2, 0.3, 0.5],
    device: str = 'cpu'
) -> List[Dict]:
    """
    Compare different dropout rates for MC Dropout.
    
    Key metrics:
    - Accuracy: Higher is better
    - Separation: Higher is better (wrong predictions should have higher uncertainty)
    """
    results = []
    
    for rate in dropout_rates:
        print(f"\n{'='*50}")
        print(f"Dropout rate: {rate}")
        print('='*50)
        
        model = MCDropoutModel(
            input_dim, hidden_dims, output_dim, dropout_rate=rate
        ).to(device)
        train_mc_dropout_model(model, train_loader, epochs=5, device=device)
        
        # Evaluate
        metrics = evaluate_uncertainty_quality(model, test_loader, device=device)
        metrics['dropout_rate'] = rate
        results.append(metrics)
    
    print("\n" + "="*80)
    print("Comparison Summary:")
    print("-" * 80)
    print(f"{'Rate':<8} {'Accuracy':<12} {'Unc(Correct)':<15} {'Unc(Wrong)':<15} {'Separation':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['dropout_rate']:<8.1f} {r['accuracy']:<12.4f} "
              f"{r['uncertainty_correct']:<15.6f} {r['uncertainty_wrong']:<15.6f} "
              f"{r['uncertainty_separation']:<12.6f}")
    
    return results
```

## Concrete Dropout: Learning the Dropout Rate

Gal et al. (2017) proposed learning the optimal dropout rate during training:

```python
class ConcreteDropout(nn.Module):
    """
    Concrete Dropout: Learn dropout rate via continuous relaxation.
    
    Instead of fixed p, learn a parameter that is optimized during training.
    Uses the concrete (Gumbel-Softmax) distribution as a continuous
    relaxation of Bernoulli dropout.
    
    Reference: Gal, Hron, Kendall (2017) "Concrete Dropout"
    """
    
    def __init__(
        self, 
        input_dim: int,
        output_dim: int,
        weight_regularizer: float = 1e-6,
        dropout_regularizer: float = 1e-5,
        init_p: float = 0.5
    ):
        super().__init__()
        
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        
        # Linear layer
        self.linear = nn.Linear(input_dim, output_dim)
        
        # Learnable dropout probability (logit parameterization)
        init_logit = np.log(init_p / (1 - init_p))
        self.p_logit = nn.Parameter(torch.tensor(init_logit))
    
    @property
    def p(self):
        """Current dropout probability."""
        return torch.sigmoid(self.p_logit)
    
    def forward(self, x: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """
        Forward pass with concrete dropout.
        
        Uses Gumbel-Softmax trick for differentiable sampling.
        """
        p = self.p
        
        if self.training:
            # Concrete relaxation
            u = torch.rand_like(x)
            # Avoid numerical issues
            u = u.clamp(1e-8, 1 - 1e-8)
            
            # Concrete distribution sample
            z = torch.sigmoid(
                (torch.log(u) - torch.log(1 - u) + torch.log(p) - torch.log(1 - p)) 
                / temperature
            )
            
            # Scale by (1-p) for unbiased expectation
            x = x * z / (1 - p)
        else:
            # At test time, just apply dropout
            if self.training:  # MC dropout mode
                mask = torch.bernoulli(torch.ones_like(x) * (1 - p))
                x = x * mask / (1 - p)
        
        return self.linear(x)
    
    def regularization_loss(self) -> torch.Tensor:
        """
        Regularization term for variational inference.
        
        Includes both weight regularization and dropout regularization
        derived from the KL divergence in the variational objective.
        """
        p = self.p
        
        # Weight regularization (prior)
        weight_reg = self.weight_regularizer * (
            (self.linear.weight ** 2).sum() + (self.linear.bias ** 2).sum()
        )
        
        # Dropout regularization (entropy of Bernoulli)
        dropout_reg = self.dropout_regularizer * (
            p * torch.log(p + 1e-10) + (1 - p) * torch.log(1 - p + 1e-10)
        )
        
        return weight_reg + dropout_reg
```

## Limitations and Considerations

### Theoretical Limitations

1. **Approximation quality**: MC Dropout uses a specific variational family (Bernoulli masks) that may not capture the true posterior well, especially for complex posteriors with correlations.

2. **Mean-field assumption**: The factorized posterior $q(\mathbf{w}) = \prod_l q(\mathbf{W}_l)$ ignores correlations between layers.

3. **Fixed functional form**: Unlike full BNNs, the posterior approximation structure is fixed by the network architecture.

### Practical Limitations

1. **Inference cost**: $T$ forward passes required (typically 50-100x slower than single pass).

2. **Architecture dependency**: Dropout placement affects uncertainty quality. Dropout only after fully-connected layers may miss uncertainty in convolutional features.

3. **Dropout rate sensitivity**: Results depend on dropout rate choice; optimal rate may differ from regularization-optimal rate.

4. **Underestimation of uncertainty**: Studies show MC Dropout often underestimates uncertainty compared to ensembles.

### When MC Dropout Works Well

- Standard feedforward networks with sufficient depth
- Image classification tasks with dropout after dense layers
- When computational cost of multiple passes is acceptable
- As a quick uncertainty baseline for comparison
- When you need uncertainty from an existing dropout-trained model

### When to Consider Alternatives

| Scenario | Recommended Alternative |
|----------|------------------------|
| Best uncertainty quality needed | Deep Ensembles |
| Low inference latency required | SWAG, Laplace approximation |
| Principled Bayesian posterior | Bayes by Backprop, HMC |
| RNNs/Transformers | Variational dropout, specialized methods |
| Very high-stakes decisions | Multiple methods + human review |

## Comparison with Other Methods

| Method | Epistemic Unc. | Aleatoric Unc. | Compute Cost | Implementation |
|--------|---------------|----------------|--------------|----------------|
| MC Dropout | ✓ (approx) | ✗ | Medium (T passes) | Easy |
| Deep Ensemble | ✓✓ | ✓ | High (M models) | Easy |
| Bayes by Backprop | ✓ | ✓ | High | Hard |
| SWAG | ✓ | ✗ | Low-Medium | Medium |
| Laplace | ✓ | ✗ | Low | Medium |

## Key Takeaways

!!! success "Summary"
    1. **MC Dropout reinterprets dropout** as approximate Bayesian inference via variational methods
    2. **Enable dropout at test time** and run multiple forward passes to sample from approximate posterior
    3. **Variance across passes** estimates epistemic uncertainty (model uncertainty)
    4. **50-100 samples** typically sufficient for stable estimates; use 20-30 for quick prototyping
    5. **Simple to implement** with existing trained models—no architecture changes needed
    6. **Trade-off**: Computational cost vs uncertainty quality vs implementation simplicity
    7. **Dropout rate matters**: 0.2-0.3 is often a good starting point; consider Concrete Dropout for learning it

## Exercises

1. **Basic Implementation**: Implement MC Dropout for a CNN classifier on CIFAR-10. Compare uncertainty estimates with and without MC sampling at test time.

2. **Sample Convergence**: Plot uncertainty estimates vs number of MC samples. At what point do estimates stabilize? How does this depend on dropout rate?

3. **OOD Detection**: Train on MNIST, test on Fashion-MNIST and NotMNIST. Use MC Dropout uncertainty to build an OOD detector and compute AUROC.

4. **Calibration Analysis**: Compute ECE (Expected Calibration Error) for MC Dropout with different numbers of samples. Compare to a deterministic model.

5. **Concrete Dropout**: Implement Concrete Dropout and compare learned dropout rates across different datasets. Do the learned rates correlate with dataset complexity?

## References

- Gal, Y., & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." ICML.
- Gal, Y. (2016). "Uncertainty in Deep Learning." PhD Thesis, University of Cambridge.
- Gal, Y., Hron, J., & Kendall, A. (2017). "Concrete Dropout." NeurIPS.
- Srivastava, N., et al. (2014). "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." JMLR.
- Kendall, A., & Gal, Y. (2017). "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" NeurIPS.
