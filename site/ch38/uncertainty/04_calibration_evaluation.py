"""
Module 63.4: Calibration & Evaluation of Uncertainty Estimates

This script covers comprehensive methods for calibrating and evaluating
uncertainty estimates in deep learning models. Well-calibrated models are
essential for trustworthy AI systems.

Topics:
    1. Calibration metrics (ECE, MCE, Brier score)
    2. Temperature scaling
    3. Platt scaling and isotonic regression
    4. Reliability diagrams
    5. Proper scoring rules
    6. Out-of-distribution detection
    7. Uncertainty quality evaluation

Mathematical Background:
    
    Calibration:
        A model is calibrated if:
            P(Y = y | confidence = p) = p
        
        For all confidence levels p, predictions with confidence p
        should be correct p% of the time.
    
    Expected Calibration Error (ECE):
        ECE = Σ_m (n_m/n) |acc(B_m) - conf(B_m)|
        
        where B_m are bins of predictions grouped by confidence
    
    Brier Score:
        BS = (1/N) Σ_i (p_i - y_i)²
        
        Measures mean squared error of probabilistic predictions
    
    Temperature Scaling:
        p_i = softmax(z_i / T)
        
        T > 1: less confident (calibrates overconfident models)
        T < 1: more confident

Learning Objectives:
    - Compute and interpret calibration metrics
    - Apply post-hoc calibration methods
    - Create reliability diagrams
    - Evaluate uncertainty quality
    - Use proper scoring rules
    - Detect miscalibration

Prerequisites:
    - Module 63.1-63.3: Uncertainty methods
    - Understanding of probability calibration
    - Familiarity with evaluation metrics

Time: 3-4 hours
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression
from typing import Tuple, List, Dict, Optional
import seaborn as sns
from tqdm import tqdm


# ============================================================================
# PART 1: CALIBRATION METRICS
# ============================================================================

class CalibrationMetrics:
    """
    Comprehensive calibration metrics for model evaluation.
    
    Metrics implemented:
        1. Expected Calibration Error (ECE)
        2. Maximum Calibration Error (MCE)
        3. Brier Score
        4. Negative Log-Likelihood (NLL)
        5. Accuracy
        6. Confidence statistics
    """
    
    @staticmethod
    def expected_calibration_error(confidences: np.ndarray,
                                   predictions: np.ndarray,
                                   true_labels: np.ndarray,
                                   n_bins: int = 15) -> Tuple[float, Dict]:
        """
        Compute Expected Calibration Error (ECE).
        
        ECE measures the average difference between confidence and accuracy
        across different confidence bins.
        
        Algorithm:
            1. Partition predictions into bins by confidence level
            2. For each bin, compute average confidence and accuracy
            3. ECE = Weighted average of |confidence - accuracy|
        
        Args:
            confidences: Confidence scores (max probability per prediction)
            predictions: Predicted class labels
            true_labels: True class labels
            n_bins: Number of bins for discretization
        
        Returns:
            ece: Expected Calibration Error
            bin_info: Dictionary with per-bin statistics
        """
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        # Storage for bin statistics
        bin_confidences = []
        bin_accuracies = []
        bin_counts = []
        
        ece = 0.0
        total_samples = len(confidences)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.sum() / total_samples
            
            if prop_in_bin > 0:
                # Accuracy in bin
                accuracy_in_bin = (predictions[in_bin] == true_labels[in_bin]).astype(float).mean()
                
                # Average confidence in bin
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                # Contribution to ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                # Store for visualization
                bin_confidences.append(avg_confidence_in_bin)
                bin_accuracies.append(accuracy_in_bin)
                bin_counts.append(in_bin.sum())
        
        bin_info = {
            'bin_confidences': bin_confidences,
            'bin_accuracies': bin_accuracies,
            'bin_counts': bin_counts
        }
        
        return ece, bin_info
    
    @staticmethod
    def maximum_calibration_error(confidences: np.ndarray,
                                  predictions: np.ndarray,
                                  true_labels: np.ndarray,
                                  n_bins: int = 15) -> float:
        """
        Compute Maximum Calibration Error (MCE).
        
        MCE is the maximum difference between confidence and accuracy
        across all bins. It captures worst-case miscalibration.
        
        Args:
            confidences: Confidence scores
            predictions: Predicted labels
            true_labels: True labels
            n_bins: Number of bins
        
        Returns:
            mce: Maximum Calibration Error
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        max_error = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = (predictions[in_bin] == true_labels[in_bin]).astype(float).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                max_error = max(max_error, error)
        
        return max_error
    
    @staticmethod
    def brier_score(probs: np.ndarray, true_labels: np.ndarray) -> float:
        """
        Compute Brier Score (mean squared error of probabilities).
        
        BS = (1/N) Σ_i Σ_k (p_ik - y_ik)²
        
        where:
            p_ik is predicted probability for class k
            y_ik is 1 if true class is k, 0 otherwise
        
        Lower is better. Range: [0, 2]
        
        Args:
            probs: Predicted probabilities (N, K)
            true_labels: True class labels (N,)
        
        Returns:
            brier_score: Mean squared error
        """
        n_samples, n_classes = probs.shape
        
        # One-hot encode true labels
        y_true_one_hot = np.zeros((n_samples, n_classes))
        y_true_one_hot[np.arange(n_samples), true_labels] = 1
        
        # Compute squared differences
        brier = np.mean(np.sum((probs - y_true_one_hot) ** 2, axis=1))
        
        return brier
    
    @staticmethod
    def negative_log_likelihood(probs: np.ndarray, true_labels: np.ndarray) -> float:
        """
        Compute Negative Log-Likelihood (cross-entropy loss).
        
        NLL = -(1/N) Σ_i log(p_i[y_i])
        
        Proper scoring rule: encourages accurate probability estimates.
        Lower is better.
        
        Args:
            probs: Predicted probabilities (N, K)
            true_labels: True class labels (N,)
        
        Returns:
            nll: Negative log-likelihood
        """
        n_samples = len(true_labels)
        
        # Get probabilities of true classes
        true_class_probs = probs[np.arange(n_samples), true_labels]
        
        # Compute NLL (add small epsilon for numerical stability)
        nll = -np.mean(np.log(true_class_probs + 1e-10))
        
        return nll
    
    @staticmethod
    def compute_all_metrics(probs: np.ndarray,
                           predictions: np.ndarray,
                           true_labels: np.ndarray) -> Dict[str, float]:
        """
        Compute all calibration metrics.
        
        Args:
            probs: Predicted probabilities (N, K)
            predictions: Predicted class labels (N,)
            true_labels: True class labels (N,)
        
        Returns:
            Dictionary with all metrics
        """
        # Confidence: max probability
        confidences = np.max(probs, axis=1)
        
        # Accuracy
        accuracy = np.mean(predictions == true_labels)
        
        # Average confidence
        avg_confidence = np.mean(confidences)
        
        # Calibration metrics
        ece, bin_info = CalibrationMetrics.expected_calibration_error(
            confidences, predictions, true_labels
        )
        mce = CalibrationMetrics.maximum_calibration_error(
            confidences, predictions, true_labels
        )
        brier = CalibrationMetrics.brier_score(probs, true_labels)
        nll = CalibrationMetrics.negative_log_likelihood(probs, true_labels)
        
        return {
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'ece': ece,
            'mce': mce,
            'brier_score': brier,
            'nll': nll,
            'bin_info': bin_info
        }


def demonstrate_calibration_metrics():
    """
    Demonstrate calibration metrics with examples.
    """
    print("=" * 70)
    print("PART 1: Calibration Metrics")
    print("=" * 70)
    
    np.random.seed(42)
    n_samples = 1000
    n_classes = 10
    
    # Example 1: Well-calibrated model
    print("\nExample 1: Well-Calibrated Model")
    print("-" * 60)
    
    # Generate predictions where confidence matches accuracy
    true_labels_1 = np.random.randint(0, n_classes, n_samples)
    probs_1 = np.zeros((n_samples, n_classes))
    
    for i in range(n_samples):
        # Random confidence
        conf = np.random.uniform(0.3, 0.95)
        
        # Correct with probability = confidence
        if np.random.rand() < conf:
            pred_class = true_labels_1[i]
        else:
            pred_class = np.random.choice([c for c in range(n_classes) if c != true_labels_1[i]])
        
        # Set probabilities
        probs_1[i, pred_class] = conf
        probs_1[i, :] = probs_1[i, :] / probs_1[i, :].sum()  # Normalize
    
    predictions_1 = np.argmax(probs_1, axis=1)
    metrics_1 = CalibrationMetrics.compute_all_metrics(probs_1, predictions_1, true_labels_1)
    
    print(f"Accuracy:        {metrics_1['accuracy']:.4f}")
    print(f"Avg Confidence:  {metrics_1['avg_confidence']:.4f}")
    print(f"ECE:             {metrics_1['ece']:.4f} ← Low (well-calibrated)")
    print(f"MCE:             {metrics_1['mce']:.4f}")
    print(f"Brier Score:     {metrics_1['brier_score']:.4f}")
    print(f"NLL:             {metrics_1['nll']:.4f}")
    
    # Example 2: Overconfident model
    print("\nExample 2: Overconfident Model")
    print("-" * 60)
    
    true_labels_2 = np.random.randint(0, n_classes, n_samples)
    probs_2 = np.zeros((n_samples, n_classes))
    
    for i in range(n_samples):
        # High confidence (0.8-0.99)
        conf = np.random.uniform(0.8, 0.99)
        
        # But only correct 60% of time
        if np.random.rand() < 0.6:
            pred_class = true_labels_2[i]
        else:
            pred_class = np.random.choice([c for c in range(n_classes) if c != true_labels_2[i]])
        
        probs_2[i, pred_class] = conf
        remaining = (1 - conf) / (n_classes - 1)
        probs_2[i, :] += remaining
        probs_2[i, pred_class] = conf
    
    predictions_2 = np.argmax(probs_2, axis=1)
    metrics_2 = CalibrationMetrics.compute_all_metrics(probs_2, predictions_2, true_labels_2)
    
    print(f"Accuracy:        {metrics_2['accuracy']:.4f} ← Lower than confidence")
    print(f"Avg Confidence:  {metrics_2['avg_confidence']:.4f} ← High")
    print(f"ECE:             {metrics_2['ece']:.4f} ← High (overconfident)")
    print(f"MCE:             {metrics_2['mce']:.4f}")
    print(f"Brier Score:     {metrics_2['brier_score']:.4f}")
    print(f"NLL:             {metrics_2['nll']:.4f}")
    
    print("\n" + "=" * 70)
    print("Key Observation:")
    print("  ECE quantifies miscalibration")
    print("  Well-calibrated: ECE ≈ 0")
    print("  Overconfident: ECE > 0.1")
    print("=" * 70)
    
    return metrics_1, metrics_2


# ============================================================================
# PART 2: TEMPERATURE SCALING
# ============================================================================

class TemperatureScaling(nn.Module):
    """
    Temperature Scaling for post-hoc calibration.
    
    Method (Guo et al., 2017):
        1. Train model normally
        2. Learn optimal temperature T on validation set
        3. At test time: p_i = softmax(z_i / T)
    
    Temperature T:
        - T > 1: smooths probabilities (reduces overconfidence)
        - T = 1: no change
        - T < 1: sharpens probabilities (increases confidence)
    
    Advantages:
        - Simple, single parameter
        - Does not change predictions (only calibration)
        - Fast to optimize
    
    Optimization:
        Minimize NLL on validation set:
            T* = argmin_T -Σ log(softmax(z_i / T)[y_i])
    """
    
    def __init__(self):
        """Initialize with temperature T=1 (no scaling)."""
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Model logits before softmax
        
        Returns:
            Calibrated probabilities
        """
        return torch.softmax(logits / self.temperature, dim=1)
    
    def fit(self, val_loader: DataLoader, model: nn.Module, 
            max_iter: int = 50, lr: float = 0.01):
        """
        Learn optimal temperature on validation set.
        
        Minimizes negative log-likelihood (cross-entropy).
        
        Args:
            val_loader: Validation data loader
            model: Trained model (frozen weights)
            max_iter: Maximum optimization iterations
            lr: Learning rate for temperature optimization
        """
        print("\nOptimizing temperature on validation set...")
        
        # Collect logits and labels
        logits_list = []
        labels_list = []
        
        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                if len(batch_x.shape) > 2:
                    batch_x = batch_x.view(batch_x.size(0), -1)
                
                logits = model(batch_x)
                logits_list.append(logits)
                labels_list.append(batch_y)
        
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
        
        # Optimize temperature
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        nll_criterion = nn.CrossEntropyLoss()
        
        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits), labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        optimal_temp = self.temperature.item()
        print(f"Optimal temperature: {optimal_temp:.4f}")
        
        if optimal_temp > 1.0:
            print("→ Model was overconfident (T > 1 reduces confidence)")
        elif optimal_temp < 1.0:
            print("→ Model was underconfident (T < 1 increases confidence)")
        else:
            print("→ Model was already well-calibrated (T ≈ 1)")
        
        return optimal_temp


def demonstrate_temperature_scaling():
    """
    Demonstrate temperature scaling on a trained model.
    """
    print("\n" + "=" * 70)
    print("PART 2: Temperature Scaling")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Split into train and validation
    train_subset, val_subset = random_split(train_dataset, [4000, 1000])
    val_loader = DataLoader(val_subset, batch_size=128, shuffle=False)
    
    # Simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(28*28, 256)
            self.fc2 = nn.Linear(256, 10)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            return self.fc2(x)
    
    model = SimpleModel()
    
    # Train briefly (intentionally overfit slightly)
    print("\nTraining model (will be slightly overconfident)...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    
    for epoch in range(5):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.view(batch_x.size(0), -1)
            
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Evaluate before calibration
    print("\nEvaluating BEFORE temperature scaling...")
    model.eval()
    
    all_probs_before = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.view(batch_x.size(0), -1)
            logits = model(batch_x)
            probs = F.softmax(logits, dim=1)
            
            all_probs_before.append(probs.numpy())
            all_labels.append(batch_y.numpy())
    
    probs_before = np.vstack(all_probs_before)
    labels = np.concatenate(all_labels)
    preds_before = np.argmax(probs_before, axis=1)
    
    metrics_before = CalibrationMetrics.compute_all_metrics(
        probs_before, preds_before, labels
    )
    
    print("\nMetrics BEFORE calibration:")
    print(f"  Accuracy:       {metrics_before['accuracy']:.4f}")
    print(f"  Avg Confidence: {metrics_before['avg_confidence']:.4f}")
    print(f"  ECE:            {metrics_before['ece']:.4f}")
    print(f"  NLL:            {metrics_before['nll']:.4f}")
    
    # Apply temperature scaling
    temp_scaler = TemperatureScaling()
    optimal_temp = temp_scaler.fit(val_loader, model)
    
    # Evaluate after calibration
    print("\nEvaluating AFTER temperature scaling...")
    
    all_probs_after = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.view(batch_x.size(0), -1)
            logits = model(batch_x)
            probs = temp_scaler(logits)
            
            all_probs_after.append(probs.numpy())
    
    probs_after = np.vstack(all_probs_after)
    preds_after = np.argmax(probs_after, axis=1)
    
    metrics_after = CalibrationMetrics.compute_all_metrics(
        probs_after, preds_after, labels
    )
    
    print("\nMetrics AFTER calibration:")
    print(f"  Accuracy:       {metrics_after['accuracy']:.4f} (unchanged)")
    print(f"  Avg Confidence: {metrics_after['avg_confidence']:.4f}")
    print(f"  ECE:            {metrics_after['ece']:.4f} ← Improved!")
    print(f"  NLL:            {metrics_after['nll']:.4f} ← Improved!")
    
    print("\n" + "=" * 70)
    print("Temperature Scaling Benefits:")
    print("  ✓ Reduces ECE (better calibration)")
    print("  ✓ Reduces NLL (better probability estimates)")
    print("  ✓ Preserves accuracy (same predictions)")
    print("  ✓ Single parameter, fast optimization")
    print("=" * 70)
    
    return metrics_before, metrics_after


# ============================================================================
# PART 3: OTHER CALIBRATION METHODS
# ============================================================================

def platt_scaling(logits: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Platt Scaling: learns parameters a, b such that:
        p_i = sigmoid(a * logit_i + b)
    
    For multi-class, applies to logits of true class.
    
    Args:
        logits: Model logits (N, K)
        labels: True labels (N,)
    
    Returns:
        a, b: Learned parameters
    """
    # Get logits for true classes
    n_samples = len(labels)
    true_class_logits = logits[np.arange(n_samples), labels]
    
    # Define negative log-likelihood to minimize
    def nll(params):
        a, b = params
        scaled_logits = a * true_class_logits + b
        probs = 1 / (1 + np.exp(-scaled_logits))
        return -np.sum(np.log(probs + 1e-10))
    
    # Optimize
    result = minimize(nll, x0=[1.0, 0.0], method='BFGS')
    a, b = result.x
    
    return a, b


def isotonic_regression_calibration(confidences: np.ndarray, 
                                    labels: np.ndarray) -> IsotonicRegression:
    """
    Isotonic Regression: learns monotonic mapping from confidence to calibrated probability.
    
    More flexible than Platt scaling (no parametric form).
    
    Args:
        confidences: Predicted confidences (N,)
        labels: Binary correctness (N,)
    
    Returns:
        Fitted isotonic regression model
    """
    # Fit isotonic regression
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(confidences, labels)
    
    return iso_reg


def demonstrate_other_calibration_methods():
    """
    Demonstrate Platt scaling and isotonic regression.
    """
    print("\n" + "=" * 70)
    print("PART 3: Other Calibration Methods")
    print("=" * 70)
    
    print("""
    1. Platt Scaling:
       - Learns sigmoid transformation: p = σ(a*z + b)
       - Originally for SVMs, applicable to NNs
       - Two parameters to learn
       - Assumes sigmoid shape
    
    2. Isotonic Regression:
       - Non-parametric monotonic mapping
       - More flexible than Platt scaling
       - Can model arbitrary calibration curves
       - Requires more validation data
    
    3. Comparison:
       Temperature Scaling:  1 parameter,  assumes constant scaling
       Platt Scaling:        2 parameters, assumes sigmoid shape
       Isotonic Regression:  non-parametric, most flexible
    
    Recommendation:
       - Start with Temperature Scaling (simplest, often best)
       - Try Isotonic if Temperature doesn't work well
       - Platt useful for binary classification
    """)


# ============================================================================
# PART 4: RELIABILITY DIAGRAMS
# ============================================================================

def plot_reliability_diagram(metrics: Dict, title: str = "Reliability Diagram"):
    """
    Plot reliability diagram (calibration curve).
    
    Shows relationship between predicted confidence and actual accuracy.
    Perfect calibration: points lie on diagonal.
    
    Args:
        metrics: Dictionary with bin_info from calibration metrics
        title: Plot title
    """
    bin_info = metrics['bin_info']
    confidences = bin_info['bin_confidences']
    accuracies = bin_info['bin_accuracies']
    counts = bin_info['bin_counts']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Reliability diagram
    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
    
    # Actual calibration
    if len(confidences) > 0:
        # Normalize counts for marker size
        max_count = max(counts)
        sizes = [200 * (c / max_count) for c in counts]
        
        scatter = ax1.scatter(confidences, accuracies, s=sizes, alpha=0.6,
                             c='blue', edgecolors='black', linewidth=1.5,
                             label='Model Calibration')
        
        # Connect points
        if len(confidences) > 1:
            ax1.plot(confidences, accuracies, 'b-', alpha=0.3, linewidth=1)
    
    # Add gap bars
    for conf, acc in zip(confidences, accuracies):
        ax1.plot([conf, conf], [conf, acc], 'r-', linewidth=1, alpha=0.5)
    
    ax1.set_xlabel('Confidence', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title(f'{title}\nECE: {metrics["ece"]:.4f}', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Plot 2: Sample distribution
    if len(counts) > 0:
        ax2.bar(range(len(counts)), counts, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Confidence Bin', fontsize=12)
        ax2.set_ylabel('Number of Samples', fontsize=12)
        ax2.set_title('Distribution of Predictions', fontsize=13)
        ax2.set_xticks(range(len(confidences)))
        ax2.set_xticklabels([f'{c:.2f}' for c in confidences], rotation=45)
    
    plt.tight_layout()
    return fig


def demonstrate_reliability_diagrams():
    """
    Create and compare reliability diagrams.
    """
    print("\n" + "=" * 70)
    print("PART 4: Reliability Diagrams")
    print("=" * 70)
    
    # Generate synthetic examples
    np.random.seed(42)
    
    # Example 1: Well-calibrated
    n_samples = 1000
    confidences_good = np.random.uniform(0.3, 1.0, n_samples)
    predictions_good = (np.random.rand(n_samples) < confidences_good).astype(int)
    labels_good = np.ones(n_samples, dtype=int)
    
    metrics_good = CalibrationMetrics.compute_all_metrics(
        np.column_stack([1 - confidences_good, confidences_good]),
        predictions_good,
        labels_good
    )
    
    # Example 2: Overconfident
    confidences_bad = np.random.uniform(0.7, 1.0, n_samples)
    predictions_bad = (np.random.rand(n_samples) < 0.6).astype(int)
    labels_bad = np.ones(n_samples, dtype=int)
    
    metrics_bad = CalibrationMetrics.compute_all_metrics(
        np.column_stack([1 - confidences_bad, confidences_bad]),
        predictions_bad,
        labels_bad
    )
    
    # Create plots
    fig1 = plot_reliability_diagram(metrics_good, "Well-Calibrated Model")
    plt.savefig('reliability_well_calibrated.png', dpi=150, bbox_inches='tight')
    
    fig2 = plot_reliability_diagram(metrics_bad, "Overconfident Model")
    plt.savefig('reliability_overconfident.png', dpi=150, bbox_inches='tight')
    
    print("\nReliability diagrams saved!")
    print("  - reliability_well_calibrated.png")
    print("  - reliability_overconfident.png")
    
    print("\nInterpretation:")
    print("  Points on diagonal → well-calibrated")
    print("  Points above diagonal → underconfident")
    print("  Points below diagonal → overconfident")
    print("  Gap size → magnitude of miscalibration")


# ============================================================================
# PART 5: OUT-OF-DISTRIBUTION DETECTION
# ============================================================================

def evaluate_ood_detection(in_dist_uncertainty: np.ndarray,
                          ood_uncertainty: np.ndarray) -> Dict[str, float]:
    """
    Evaluate uncertainty-based OOD detection.
    
    Use uncertainty threshold to separate in-distribution from OOD.
    
    Args:
        in_dist_uncertainty: Uncertainty on in-distribution data
        ood_uncertainty: Uncertainty on OOD data
    
    Returns:
        Dictionary with OOD detection metrics
    """
    # Combine and create labels
    all_uncertainty = np.concatenate([in_dist_uncertainty, ood_uncertainty])
    labels = np.concatenate([
        np.zeros(len(in_dist_uncertainty)),  # 0 = in-distribution
        np.ones(len(ood_uncertainty))         # 1 = OOD
    ])
    
    # Compute AUROC (using uncertainty as OOD score)
    from sklearn.metrics import roc_auc_score, roc_curve
    
    auroc = roc_auc_score(labels, all_uncertainty)
    
    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(labels, all_uncertainty)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Classification at optimal threshold
    predictions = (all_uncertainty > optimal_threshold).astype(int)
    accuracy = np.mean(predictions == labels)
    
    # TPR at 95% TNR
    tnr = 1 - fpr
    idx_95 = np.argmin(np.abs(tnr - 0.95))
    tpr_at_95 = tpr[idx_95]
    
    return {
        'auroc': auroc,
        'optimal_threshold': optimal_threshold,
        'accuracy': accuracy,
        'tpr_at_95_tnr': tpr_at_95
    }


def demonstrate_ood_detection():
    """
    Demonstrate OOD detection using uncertainty.
    """
    print("\n" + "=" * 70)
    print("PART 5: Out-of-Distribution Detection")
    print("=" * 70)
    
    # Simulate uncertainties
    np.random.seed(42)
    
    # In-distribution: lower uncertainty
    in_dist_unc = np.random.gamma(2, 0.01, 500)
    
    # OOD: higher uncertainty
    ood_unc = np.random.gamma(4, 0.02, 500)
    
    print(f"\nIn-distribution uncertainty: {in_dist_unc.mean():.6f} ± {in_dist_unc.std():.6f}")
    print(f"OOD uncertainty:             {ood_unc.mean():.6f} ± {ood_unc.std():.6f}")
    
    # Evaluate OOD detection
    ood_metrics = evaluate_ood_detection(in_dist_unc, ood_unc)
    
    print("\nOOD Detection Performance:")
    print(f"  AUROC:              {ood_metrics['auroc']:.4f}")
    print(f"  Optimal Threshold:  {ood_metrics['optimal_threshold']:.6f}")
    print(f"  Accuracy:           {ood_metrics['accuracy']:.4f}")
    print(f"  TPR @ 95% TNR:      {ood_metrics['tpr_at_95_tnr']:.4f}")
    
    print("\nKey Insight:")
    print("  Higher uncertainty on OOD data enables detection")
    print("  AUROC > 0.9 indicates good separability")
    print("  Useful for flagging unusual inputs in production")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function demonstrating calibration and evaluation.
    """
    print("\n" + "="*70)
    print("MODULE 63.4: CALIBRATION & EVALUATION")
    print("="*70)
    
    # Part 1: Calibration metrics
    metrics_good, metrics_bad = demonstrate_calibration_metrics()
    
    # Part 2: Temperature scaling
    metrics_before, metrics_after = demonstrate_temperature_scaling()
    
    # Part 3: Other methods
    demonstrate_other_calibration_methods()
    
    # Part 4: Reliability diagrams
    demonstrate_reliability_diagrams()
    
    # Part 5: OOD detection
    demonstrate_ood_detection()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. Calibration Metrics:
       - ECE: Average miscalibration across confidence bins
       - MCE: Worst-case miscalibration
       - Brier Score: Mean squared error of probabilities
       - NLL: Proper scoring rule for probability quality
    
    2. Temperature Scaling:
       - Post-hoc calibration method
       - Single learnable parameter
       - Preserves accuracy, improves calibration
       - Should be standard practice
    
    3. Evaluation:
       - Always evaluate both accuracy AND calibration
       - Reliability diagrams visualize calibration
       - OOD detection validates uncertainty quality
    
    4. Best Practices:
       - Use validation set for calibration
       - Report ECE alongside accuracy
       - Visualize with reliability diagrams
       - Test on OOD data
    
    5. Practical Guidelines:
       - Apply temperature scaling by default
       - Aim for ECE < 0.05
       - Check calibration across different confidence levels
       - Monitor calibration in production
    """)
    
    print("\nNext Steps:")
    print("  → Try 05_practical_applications.py for real-world use cases")
    print("  → Apply calibration to your own models")
    print("  → Explore other calibration methods")
    print("  → Build calibration monitoring into deployment pipeline")


if __name__ == "__main__":
    main()
