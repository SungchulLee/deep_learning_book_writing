# Practical Applications of Model Uncertainty

## Introduction

Uncertainty quantification transforms predictions from point estimates into probability distributions, enabling a range of practical applications. This section covers the most impactful uses of uncertainty in production systems.

## Active Learning with Uncertainty

### The Problem

Labeling data is expensive. Active learning uses uncertainty to select the most informative samples for labeling, reducing annotation cost by 50-70%.

### Acquisition Functions

**Maximum Entropy**:
$$\mathbf{x}^* = \arg\max_{\mathbf{x}} H(y|\mathbf{x}, \mathcal{D}) = -\sum_{k} p(y=k|\mathbf{x}) \log p(y=k|\mathbf{x})$$

**Maximum Variance**:
$$\mathbf{x}^* = \arg\max_{\mathbf{x}} \text{Var}[p(y|\mathbf{x}, \mathcal{D})]$$

**BALD** (Bayesian Active Learning by Disagreement):
$$\mathbf{x}^* = \arg\max_{\mathbf{x}} I(y; \mathbf{w}|\mathbf{x}, \mathcal{D}) = H(y|\mathbf{x}) - \mathbb{E}_{\mathbf{w}}[H(y|\mathbf{x}, \mathbf{w})]$$

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import List, Tuple, Optional


class ActiveLearner:
    """
    Active learning system using uncertainty for sample selection.
    
    Workflow:
    1. Train model on small labeled set
    2. Predict on unlabeled pool with uncertainty
    3. Select most uncertain samples for labeling
    4. Add to training set and retrain
    5. Repeat until budget exhausted
    """
    
    def __init__(self, model: nn.Module, acquisition: str = 'entropy'):
        """
        Args:
            model: Model with uncertainty estimation capability
            acquisition: 'entropy', 'variance', 'bald', or 'random'
        """
        self.model = model
        self.acquisition = acquisition
        self.labeled_indices = []
        self.query_history = []
    
    def compute_acquisition_scores(self, probs: torch.Tensor, 
                                   all_probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute acquisition scores for sample selection.
        
        Args:
            probs: Mean predicted probabilities (N, K)
            all_probs: All ensemble/MC predictions (M, N, K) for BALD
        
        Returns:
            Scores per sample (higher = more informative)
        """
        epsilon = 1e-10
        
        if self.acquisition == 'entropy':
            # Entropy of predictive distribution
            scores = -torch.sum(probs * torch.log(probs + epsilon), dim=-1)
        
        elif self.acquisition == 'variance':
            # Variance (max probability uncertainty)
            scores = 1 - probs.max(dim=-1)[0]
        
        elif self.acquisition == 'bald':
            if all_probs is None:
                raise ValueError("BALD requires ensemble predictions")
            
            # BALD = H(y|x,D) - E_w[H(y|x,w)]
            # Total entropy
            mean_probs = all_probs.mean(dim=0)
            total_entropy = -torch.sum(mean_probs * torch.log(mean_probs + epsilon), dim=-1)
            
            # Expected entropy
            individual_entropy = -torch.sum(all_probs * torch.log(all_probs + epsilon), dim=-1)
            expected_entropy = individual_entropy.mean(dim=0)
            
            scores = total_entropy - expected_entropy
        
        elif self.acquisition == 'random':
            scores = torch.rand(probs.shape[0])
        
        else:
            raise ValueError(f"Unknown acquisition: {self.acquisition}")
        
        return scores
    
    def select_samples(self, pool_loader: DataLoader, 
                       n_samples: int, 
                       n_mc_samples: int = 50) -> np.ndarray:
        """
        Select most informative samples from pool.
        
        Args:
            pool_loader: DataLoader for unlabeled pool
            n_samples: Number of samples to select
            n_mc_samples: MC samples for uncertainty estimation
        
        Returns:
            Indices of selected samples
        """
        self.model.eval()
        
        all_scores = []
        all_probs_list = []
        
        with torch.no_grad():
            for x, _ in pool_loader:
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)
                
                # Get MC predictions if model supports it
                if hasattr(self.model, 'mc_predict'):
                    mean_probs, _, mc_probs = self.model.mc_predict(
                        x, n_samples=n_mc_samples, return_samples=True
                    )
                    all_probs_list.append(mc_probs)
                else:
                    logits = self.model(x)
                    mean_probs = F.softmax(logits, dim=-1)
                    mc_probs = None
                
                scores = self.compute_acquisition_scores(mean_probs, mc_probs)
                all_scores.append(scores)
        
        all_scores = torch.cat(all_scores)
        
        # Select top-k
        _, top_indices = torch.topk(all_scores, k=min(n_samples, len(all_scores)))
        selected = top_indices.cpu().numpy()
        
        self.query_history.append(selected.tolist())
        
        return selected


def run_active_learning_experiment(model_class, 
                                    train_dataset,
                                    test_dataset,
                                    n_initial: int = 100,
                                    n_queries: int = 5,
                                    query_size: int = 50,
                                    acquisition: str = 'entropy') -> dict:
    """
    Run complete active learning experiment.
    
    Returns learning curve showing accuracy vs labeled samples.
    """
    # Initial random labeled set
    all_indices = np.arange(len(train_dataset))
    np.random.shuffle(all_indices)
    
    labeled_indices = all_indices[:n_initial].tolist()
    pool_indices = all_indices[n_initial:n_initial + 1000].tolist()  # Limit pool size
    
    results = {
        'n_labeled': [],
        'accuracy': [],
        'acquisition': acquisition
    }
    
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    for query_round in range(n_queries + 1):
        print(f"\nRound {query_round}: {len(labeled_indices)} labeled samples")
        
        # Create model and train
        model = model_class()
        labeled_subset = Subset(train_dataset, labeled_indices)
        train_loader = DataLoader(labeled_subset, batch_size=64, shuffle=True)
        
        # Train
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(5):
            for x, y in train_loader:
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)
                
                loss = criterion(model(x), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)
                preds = model(x).argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += len(y)
        
        accuracy = correct / total
        results['n_labeled'].append(len(labeled_indices))
        results['accuracy'].append(accuracy)
        
        print(f"  Test Accuracy: {accuracy:.4f}")
        
        # Query new samples (except last round)
        if query_round < n_queries and pool_indices:
            # Create pool loader
            pool_subset = Subset(train_dataset, pool_indices)
            pool_loader = DataLoader(pool_subset, batch_size=256, shuffle=False)
            
            # Select samples
            learner = ActiveLearner(model, acquisition=acquisition)
            selected_pool_indices = learner.select_samples(pool_loader, query_size)
            
            # Map back to original indices and update
            selected_original = [pool_indices[i] for i in selected_pool_indices]
            labeled_indices.extend(selected_original)
            pool_indices = [i for i in pool_indices if i not in selected_original]
    
    return results


def compare_acquisition_functions():
    """
    Compare different acquisition functions.
    """
    import torchvision
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST('./data', train=False, transform=transform)
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(784, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 10)
            )
        
        def forward(self, x):
            return self.fc(x)
    
    results = {}
    
    for acquisition in ['random', 'entropy', 'variance']:
        print(f"\n{'='*50}")
        print(f"Acquisition: {acquisition}")
        print('='*50)
        
        np.random.seed(42)
        torch.manual_seed(42)
        
        results[acquisition] = run_active_learning_experiment(
            SimpleModel, train_dataset, test_dataset,
            n_initial=50, n_queries=5, query_size=50,
            acquisition=acquisition
        )
    
    return results
```

## Selective Prediction (Reject Option)

### The Concept

Instead of predicting on all inputs, reject predictions when uncertainty is high:
- **Accept**: uncertainty < threshold → make prediction
- **Reject**: uncertainty ≥ threshold → abstain (request human review)

### Risk-Coverage Trade-off

$$\text{Coverage} = \frac{\text{# predictions made}}{\text{# total samples}}$$

$$\text{Selective Risk} = \frac{\text{# wrong predictions}}{\text{# predictions made}}$$

**Goal**: Maximize coverage while maintaining acceptable risk.

### Implementation

```python
class SelectivePredictor:
    """
    Selective prediction using uncertainty thresholds.
    
    Trade-off: higher threshold → higher accuracy but lower coverage
    """
    
    def __init__(self, model: nn.Module, threshold: float = 0.5):
        """
        Args:
            model: Model with uncertainty estimation
            threshold: Uncertainty threshold for rejection
        """
        self.model = model
        self.threshold = threshold
    
    def predict(self, x: torch.Tensor, 
                n_mc_samples: int = 50) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions with rejection option.
        
        Returns:
            predictions: Predicted labels
            accept_mask: Boolean mask (True = accept prediction)
            uncertainty: Uncertainty scores
        """
        if hasattr(self.model, 'mc_predict'):
            mean_probs, uncertainty = self.model.mc_predict(x, n_samples=n_mc_samples)
        else:
            with torch.no_grad():
                logits = self.model(x)
                mean_probs = F.softmax(logits, dim=-1)
                uncertainty = 1 - mean_probs.max(dim=-1)[0]
        
        predictions = mean_probs.argmax(dim=-1)
        accept_mask = uncertainty < self.threshold
        
        return predictions, accept_mask, uncertainty
    
    def evaluate(self, test_loader: DataLoader, 
                 n_mc_samples: int = 50) -> dict:
        """
        Evaluate selective prediction performance.
        """
        all_preds = []
        all_labels = []
        all_accept = []
        all_uncertainty = []
        
        for x, y in test_loader:
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            
            preds, accept, unc = self.predict(x, n_mc_samples)
            
            all_preds.append(preds)
            all_labels.append(y)
            all_accept.append(accept)
            all_uncertainty.append(unc)
        
        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        accept = torch.cat(all_accept)
        uncertainty = torch.cat(all_uncertainty)
        
        # Metrics
        coverage = accept.float().mean().item()
        
        if accept.sum() > 0:
            selective_accuracy = (preds[accept] == labels[accept]).float().mean().item()
        else:
            selective_accuracy = 0.0
        
        overall_accuracy = (preds == labels).float().mean().item()
        
        # Accuracy on rejected samples (should be lower)
        if (~accept).sum() > 0:
            rejected_accuracy = (preds[~accept] == labels[~accept]).float().mean().item()
        else:
            rejected_accuracy = 1.0
        
        return {
            'coverage': coverage,
            'selective_accuracy': selective_accuracy,
            'overall_accuracy': overall_accuracy,
            'rejected_accuracy': rejected_accuracy,
            'n_accepted': accept.sum().item(),
            'n_rejected': (~accept).sum().item()
        }


def find_optimal_threshold(model: nn.Module,
                            val_loader: DataLoader,
                            target_accuracy: float = 0.95,
                            n_mc_samples: int = 50) -> Tuple[float, dict]:
    """
    Find threshold that achieves target accuracy with maximum coverage.
    """
    # Collect all predictions and uncertainties
    all_preds = []
    all_labels = []
    all_uncertainty = []
    
    model.eval()
    
    with torch.no_grad():
        for x, y in val_loader:
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            
            if hasattr(model, 'mc_predict'):
                mean_probs, unc = model.mc_predict(x, n_samples=n_mc_samples)
            else:
                logits = model(x)
                mean_probs = F.softmax(logits, dim=-1)
                unc = 1 - mean_probs.max(dim=-1)[0]
            
            all_preds.append(mean_probs.argmax(dim=-1))
            all_labels.append(y)
            all_uncertainty.append(unc)
    
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    uncertainty = torch.cat(all_uncertainty)
    
    correct = (preds == labels)
    
    # Binary search for threshold
    thresholds = torch.linspace(uncertainty.min(), uncertainty.max(), 100)
    
    best_threshold = 0.0
    best_coverage = 0.0
    
    for thresh in thresholds:
        accept = uncertainty < thresh
        
        if accept.sum() > 0:
            sel_acc = correct[accept].float().mean().item()
            coverage = accept.float().mean().item()
            
            if sel_acc >= target_accuracy and coverage > best_coverage:
                best_coverage = coverage
                best_threshold = thresh.item()
    
    return best_threshold, {
        'threshold': best_threshold,
        'coverage': best_coverage,
        'target_accuracy': target_accuracy
    }


def plot_risk_coverage_curve(model: nn.Module,
                              test_loader: DataLoader,
                              n_mc_samples: int = 50):
    """
    Plot risk-coverage trade-off curve.
    """
    import matplotlib.pyplot as plt
    
    # Collect data
    all_preds = []
    all_labels = []
    all_uncertainty = []
    
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            
            if hasattr(model, 'mc_predict'):
                mean_probs, unc = model.mc_predict(x, n_samples=n_mc_samples)
            else:
                logits = model(x)
                mean_probs = F.softmax(logits, dim=-1)
                unc = 1 - mean_probs.max(dim=-1)[0]
            
            all_preds.append(mean_probs.argmax(dim=-1))
            all_labels.append(y)
            all_uncertainty.append(unc)
    
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    uncertainty = torch.cat(all_uncertainty)
    correct = (preds == labels)
    
    # Compute curve
    thresholds = torch.linspace(uncertainty.min(), uncertainty.max(), 100)
    coverages = []
    risks = []
    accuracies = []
    
    for thresh in thresholds:
        accept = uncertainty < thresh
        coverage = accept.float().mean().item()
        
        if accept.sum() > 0:
            risk = 1 - correct[accept].float().mean().item()  # Error rate
            accuracy = correct[accept].float().mean().item()
        else:
            risk = 0.0
            accuracy = 1.0
        
        coverages.append(coverage)
        risks.append(risk)
        accuracies.append(accuracy)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(coverages, risks, 'b-', linewidth=2)
    axes[0].set_xlabel('Coverage')
    axes[0].set_ylabel('Risk (Error Rate)')
    axes[0].set_title('Risk-Coverage Curve')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(coverages, accuracies, 'g-', linewidth=2)
    axes[1].axhline(y=correct.float().mean().item(), color='r', linestyle='--', 
                   label=f'Full accuracy: {correct.float().mean():.4f}')
    axes[1].set_xlabel('Coverage')
    axes[1].set_ylabel('Selective Accuracy')
    axes[1].set_title('Accuracy-Coverage Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

## Out-of-Distribution Detection

Use uncertainty to detect inputs that are different from training distribution:

```python
def evaluate_ood_detection(in_dist_uncertainty: np.ndarray,
                           ood_uncertainty: np.ndarray) -> dict:
    """
    Evaluate OOD detection using uncertainty scores.
    
    Args:
        in_dist_uncertainty: Uncertainty on in-distribution data
        ood_uncertainty: Uncertainty on OOD data
    
    Returns:
        Metrics: AUROC, FPR@95TPR, detection accuracy
    """
    from sklearn.metrics import roc_auc_score, roc_curve
    
    # Combine and create labels (0 = in-dist, 1 = OOD)
    all_uncertainty = np.concatenate([in_dist_uncertainty, ood_uncertainty])
    labels = np.concatenate([
        np.zeros(len(in_dist_uncertainty)),
        np.ones(len(ood_uncertainty))
    ])
    
    # AUROC
    auroc = roc_auc_score(labels, all_uncertainty)
    
    # FPR at 95% TPR
    fpr, tpr, thresholds = roc_curve(labels, all_uncertainty)
    idx_95 = np.argmin(np.abs(tpr - 0.95))
    fpr_at_95_tpr = fpr[idx_95]
    
    # Detection accuracy at optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    predictions = (all_uncertainty > optimal_threshold).astype(int)
    accuracy = (predictions == labels).mean()
    
    return {
        'auroc': auroc,
        'fpr_at_95_tpr': fpr_at_95_tpr,
        'detection_accuracy': accuracy,
        'optimal_threshold': optimal_threshold,
        'in_dist_mean': in_dist_uncertainty.mean(),
        'ood_mean': ood_uncertainty.mean()
    }


def ood_detection_example():
    """
    OOD detection example: Train on MNIST, detect Fashion-MNIST.
    """
    import torchvision
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # In-distribution: MNIST
    mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # OOD: Fashion-MNIST (same image size, different domain)
    fmnist_test = torchvision.datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    
    mnist_loader = DataLoader(mnist_test, batch_size=256, shuffle=False)
    fmnist_loader = DataLoader(fmnist_test, batch_size=256, shuffle=False)
    
    # Train model on MNIST (simplified)
    model = MCDropoutModel(784, [256, 128], 10, dropout_rate=0.3)
    # ... training code ...
    
    # Compute uncertainties
    model.eval()
    
    mnist_uncertainty = []
    fmnist_uncertainty = []
    
    with torch.no_grad():
        for x, _ in mnist_loader:
            x = x.view(x.size(0), -1)
            _, unc = model.mc_predict(x, n_samples=50)
            mnist_uncertainty.append(unc.cpu().numpy())
        
        for x, _ in fmnist_loader:
            x = x.view(x.size(0), -1)
            _, unc = model.mc_predict(x, n_samples=50)
            fmnist_uncertainty.append(unc.cpu().numpy())
    
    mnist_unc = np.concatenate(mnist_uncertainty)
    fmnist_unc = np.concatenate(fmnist_uncertainty)
    
    # Evaluate
    metrics = evaluate_ood_detection(mnist_unc, fmnist_unc)
    
    print("OOD Detection Results")
    print("=" * 40)
    print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"FPR @ 95% TPR: {metrics['fpr_at_95_tpr']:.4f}")
    print(f"Detection Accuracy: {metrics['detection_accuracy']:.4f}")
    
    return metrics
```

## Key Takeaways

!!! success "Summary"
    1. **Active Learning**: Use uncertainty to select most informative samples (50-70% label savings)
    2. **Selective Prediction**: Reject uncertain predictions for human review
    3. **OOD Detection**: High uncertainty indicates out-of-distribution inputs
    4. **Trade-offs**: Coverage vs accuracy, computational cost vs uncertainty quality

## Best Practices

| Application | Recommendation |
|-------------|----------------|
| Active Learning | Use BALD for best performance, entropy for simplicity |
| Selective Prediction | Tune threshold on validation set, monitor coverage |
| OOD Detection | Combine uncertainty with other signals (e.g., reconstruction error) |
| Production | Log uncertainty distributions, alert on drift |

## Exercises

1. **Active Learning Comparison**: Compare random, entropy, and BALD acquisition on CIFAR-10. Plot learning curves.

2. **Selective Prediction**: Find the threshold that achieves 99% accuracy on MNIST. What coverage do you get?

3. **OOD Detection**: Train on CIFAR-10, test on SVHN. How well does uncertainty distinguish them?

## References

- Gal, Y., et al. (2017). "Deep Bayesian Active Learning with Image Data"
- Geifman, Y., & El-Yaniv, R. (2017). "Selective Classification for Deep Neural Networks"
- Hendrycks, D., & Gimpel, K. (2017). "A Baseline for Detecting Misclassified and Out-of-Distribution Examples"
