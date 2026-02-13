"""
Module 63.1: Model Uncertainty Basics (Beginner Level)

This script introduces fundamental concepts in uncertainty quantification for deep learning models.
We cover the distinction between epistemic and aleatoric uncertainty, basic probabilistic predictions,
and simple ensemble methods.

Topics:
    1. Types of uncertainty (epistemic vs aleatoric)
    2. Softmax temperature and prediction confidence
    3. Simple ensemble averaging
    4. Basic calibration concepts
    5. Visualization of uncertainty

Mathematical Background:
    - Epistemic uncertainty: uncertainty about model parameters
      Can be reduced with more training data
      Example: model doesn't know if a blur is a cat or dog
    
    - Aleatoric uncertainty: inherent noise in the data
      Cannot be reduced with more data
      Example: image is truly ambiguous or corrupted
    
    - Softmax with temperature:
      p_i = exp(z_i/T) / Σ_j exp(z_j/T)
      where T is temperature (T=1 is standard softmax)

Learning Objectives:
    - Understand different types of uncertainty
    - Implement temperature scaling for calibration
    - Build simple prediction ensembles
    - Visualize and interpret uncertainty estimates

Time: 2-3 hours
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict


# ============================================================================
# PART 1: UNDERSTANDING UNCERTAINTY TYPES
# ============================================================================

def demonstrate_uncertainty_types():
    """
    Demonstrate the difference between epistemic and aleatoric uncertainty
    using synthetic data.
    
    Epistemic uncertainty: high in regions with no training data
    Aleatoric uncertainty: high in regions with noisy/overlapping data
    """
    print("=" * 70)
    print("PART 1: Types of Uncertainty")
    print("=" * 70)
    
    # Generate synthetic 1D regression data
    np.random.seed(42)
    
    # Training data: two separate regions
    X_train_1 = np.linspace(-3, -1, 20)
    X_train_2 = np.linspace(1, 3, 20)
    X_train = np.concatenate([X_train_1, X_train_2])
    
    # True function with added noise
    def true_function(x):
        return np.sin(2 * x)
    
    # High noise region (aleatoric uncertainty)
    noise_1 = np.random.normal(0, 0.1, len(X_train_1))  # Low noise
    noise_2 = np.random.normal(0, 0.3, len(X_train_2))  # High noise
    noise = np.concatenate([noise_1, noise_2])
    
    y_train = true_function(X_train) + noise
    
    # Test data: includes gap region (epistemic uncertainty)
    X_test = np.linspace(-4, 4, 200)
    y_test = true_function(X_test)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Training data has gap in region [-1, 1]")
    print(f"Right region has 3x more noise than left region")
    print(f"\nThis demonstrates:")
    print(f"  - Epistemic uncertainty: model unsure in gap [-1, 1] (no training data)")
    print(f"  - Aleatoric uncertainty: model unsure in high-noise region [1, 3]")
    
    return X_train, y_train, X_test, y_test


# ============================================================================
# PART 2: SIMPLE NEURAL NETWORK WITH SOFTMAX TEMPERATURE
# ============================================================================

class SimpleClassifier(nn.Module):
    """
    Simple feedforward classifier for demonstrating temperature scaling.
    
    Architecture:
        - Input layer
        - Two hidden layers with ReLU activation
        - Output layer (logits)
    
    Temperature scaling is applied to logits before softmax:
        p = softmax(logits / temperature)
    
    Higher temperature → smoother probabilities (less confident)
    Lower temperature → sharper probabilities (more confident)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Initialize classifier.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Number of hidden units
            output_dim: Number of output classes
        """
        super(SimpleClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Learnable temperature parameter (initialized to 1.0)
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor, use_temperature: bool = False) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            use_temperature: If True, apply temperature scaling
        
        Returns:
            Logits of shape (batch_size, output_dim)
        """
        # Hidden layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Output logits
        logits = self.fc3(x)
        
        # Apply temperature scaling if requested
        if use_temperature:
            logits = logits / self.temperature
        
        return logits
    
    def predict_with_confidence(self, x: torch.Tensor, 
                                temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions with confidence scores.
        
        Args:
            x: Input tensor
            temperature: Temperature for scaling (default 1.0 = no scaling)
        
        Returns:
            predictions: Predicted class indices
            confidences: Confidence scores (max probability)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            
            # Apply temperature scaling
            scaled_logits = logits / temperature
            
            # Get probabilities
            probs = F.softmax(scaled_logits, dim=1)
            
            # Get predictions and confidences
            confidences, predictions = torch.max(probs, dim=1)
        
        return predictions, confidences


def demonstrate_temperature_scaling():
    """
    Demonstrate how temperature affects prediction confidence.
    
    Temperature T:
        - T < 1: sharpens probabilities (more confident)
        - T = 1: standard softmax
        - T > 1: smooths probabilities (less confident)
    """
    print("\n" + "=" * 70)
    print("PART 2: Temperature Scaling")
    print("=" * 70)
    
    # Create example logits for a 3-class problem
    logits = torch.tensor([[2.0, 1.0, 0.1]])  # Model prefers class 0
    
    temperatures = [0.5, 1.0, 2.0, 5.0]
    
    print("\nOriginal logits:", logits.numpy())
    print("\nEffect of temperature on probabilities:")
    print("-" * 60)
    print(f"{'Temperature':<15} {'Class 0':<12} {'Class 1':<12} {'Class 2':<12}")
    print("-" * 60)
    
    for temp in temperatures:
        # Apply temperature scaling
        scaled_logits = logits / temp
        probs = F.softmax(scaled_logits, dim=1)
        
        print(f"{temp:<15.1f} {probs[0, 0]:.4f}       {probs[0, 1]:.4f}       {probs[0, 2]:.4f}")
    
    print("-" * 60)
    print("\nObservations:")
    print("  - Low T (0.5): More confident (higher max probability)")
    print("  - High T (5.0): Less confident (probabilities more uniform)")
    print("  - Temperature scaling affects calibration, not predictions")


# ============================================================================
# PART 3: SIMPLE ENSEMBLE FOR UNCERTAINTY ESTIMATION
# ============================================================================

class SimpleEnsemble:
    """
    Simple ensemble of neural networks for uncertainty estimation.
    
    An ensemble trains multiple models independently and aggregates their predictions.
    Disagreement between models indicates epistemic uncertainty.
    
    Prediction: y_pred = (1/M) Σ f_m(x)  (average predictions)
    Uncertainty: σ² = (1/M) Σ (f_m(x) - y_pred)²  (variance across models)
    """
    
    def __init__(self, n_models: int, input_dim: int, 
                 hidden_dim: int, output_dim: int):
        """
        Initialize ensemble of models.
        
        Args:
            n_models: Number of models in ensemble
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (number of classes)
        """
        self.n_models = n_models
        self.models = [
            SimpleClassifier(input_dim, hidden_dim, output_dim)
            for _ in range(n_models)
        ]
    
    def train_ensemble(self, train_loader: DataLoader, 
                       epochs: int = 10, lr: float = 0.001):
        """
        Train all models in the ensemble.
        
        Each model is trained independently with different random initialization.
        This diversity is key for capturing epistemic uncertainty.
        
        Args:
            train_loader: DataLoader for training data
            epochs: Number of training epochs
            lr: Learning rate
        """
        print(f"\nTraining ensemble of {self.n_models} models...")
        
        for i, model in enumerate(self.models):
            print(f"\nTraining model {i+1}/{self.n_models}")
            
            # Each model gets its own optimizer
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            model.train()
            for epoch in range(epochs):
                total_loss = 0
                
                for batch_x, batch_y in train_loader:
                    # Forward pass
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                if (epoch + 1) % 5 == 0:
                    avg_loss = total_loss / len(train_loader)
                    print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions with uncertainty estimates.
        
        Uncertainty is measured as the variance of predictions across models.
        High variance → high epistemic uncertainty (models disagree)
        Low variance → low epistemic uncertainty (models agree)
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            mean_probs: Average probabilities across models
            uncertainty: Standard deviation across models (epistemic uncertainty)
        """
        all_probs = []
        
        # Get predictions from each model
        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs)
        
        # Stack predictions: (n_models, batch_size, n_classes)
        all_probs = torch.stack(all_probs)
        
        # Calculate mean and standard deviation
        mean_probs = torch.mean(all_probs, dim=0)
        std_probs = torch.std(all_probs, dim=0)
        
        # Uncertainty: average std across classes
        uncertainty = torch.mean(std_probs, dim=1)
        
        return mean_probs, uncertainty


def train_and_evaluate_ensemble():
    """
    Train a simple ensemble and demonstrate uncertainty estimation.
    """
    print("\n" + "=" * 70)
    print("PART 3: Simple Ensemble for Uncertainty")
    print("=" * 70)
    
    # Generate synthetic classification data
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create two-class spiral dataset
    n_samples = 300
    
    # Class 0: inner spiral
    theta_0 = np.linspace(0, 2*np.pi, n_samples // 2)
    r_0 = theta_0 / (2*np.pi) * 2
    X_0 = np.column_stack([
        r_0 * np.cos(theta_0) + np.random.normal(0, 0.1, n_samples // 2),
        r_0 * np.sin(theta_0) + np.random.normal(0, 0.1, n_samples // 2)
    ])
    y_0 = np.zeros(n_samples // 2)
    
    # Class 1: outer spiral
    theta_1 = np.linspace(0, 2*np.pi, n_samples // 2)
    r_1 = theta_1 / (2*np.pi) * 2 + 2
    X_1 = np.column_stack([
        r_1 * np.cos(theta_1) + np.random.normal(0, 0.1, n_samples // 2),
        r_1 * np.sin(theta_1) + np.random.normal(0, 0.1, n_samples // 2)
    ])
    y_1 = np.ones(n_samples // 2)
    
    # Combine data
    X = np.vstack([X_0, X_1])
    y = np.concatenate([y_0, y_1])
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    # Create DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"Training data: {len(X)} samples, 2 classes")
    print(f"Input dimension: 2, Output dimension: 2")
    
    # Create and train ensemble
    ensemble = SimpleEnsemble(
        n_models=5,
        input_dim=2,
        hidden_dim=32,
        output_dim=2
    )
    
    ensemble.train_ensemble(train_loader, epochs=20, lr=0.001)
    
    # Create test grid for visualization
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )
    
    # Get predictions on grid
    grid_points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    mean_probs, uncertainty = ensemble.predict_with_uncertainty(grid_points)
    
    # Reshape for plotting
    uncertainty = uncertainty.numpy().reshape(xx.shape)
    
    print("\nEnsemble Uncertainty Analysis:")
    print(f"  Average uncertainty: {uncertainty.mean():.4f}")
    print(f"  Max uncertainty: {uncertainty.max():.4f}")
    print(f"  Uncertainty std: {uncertainty.std():.4f}")
    print("\nHigh uncertainty regions indicate:")
    print("  - Decision boundaries (models disagree on classification)")
    print("  - Regions far from training data (epistemic uncertainty)")
    
    return X, y, xx, yy, uncertainty


# ============================================================================
# PART 4: BASIC CALIBRATION CONCEPTS
# ============================================================================

def calculate_calibration_metrics(confidences: np.ndarray, 
                                  predictions: np.ndarray,
                                  true_labels: np.ndarray,
                                  n_bins: int = 10) -> Dict[str, float]:
    """
    Calculate basic calibration metrics.
    
    A model is well-calibrated if predictions with 80% confidence are
    correct 80% of the time.
    
    Expected Calibration Error (ECE):
        ECE = Σ (|B_m|/n) |acc(B_m) - conf(B_m)|
    
    where B_m are bins of predictions grouped by confidence.
    
    Args:
        confidences: Predicted confidence scores (max probability)
        predictions: Predicted class labels
        true_labels: True class labels
        n_bins: Number of bins for calibration calculation
    
    Returns:
        Dictionary with calibration metrics
    """
    # Create bins based on confidence
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    
    for i in range(n_bins):
        # Find samples in this confidence bin
        in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        bin_count = np.sum(in_bin)
        
        if bin_count > 0:
            # Average confidence in bin
            bin_confidence = np.mean(confidences[in_bin])
            
            # Accuracy in bin (fraction of correct predictions)
            bin_accuracy = np.mean(predictions[in_bin] == true_labels[in_bin])
            
            bin_confidences.append(bin_confidence)
            bin_accuracies.append(bin_accuracy)
            bin_counts.append(bin_count)
    
    # Calculate Expected Calibration Error (ECE)
    ece = 0.0
    total_samples = len(confidences)
    
    for conf, acc, count in zip(bin_confidences, bin_accuracies, bin_counts):
        ece += (count / total_samples) * abs(acc - conf)
    
    # Maximum Calibration Error (MCE)
    if len(bin_confidences) > 0:
        mce = max(abs(np.array(bin_accuracies) - np.array(bin_confidences)))
    else:
        mce = 0.0
    
    # Overall accuracy
    accuracy = np.mean(predictions == true_labels)
    
    # Average confidence
    avg_confidence = np.mean(confidences)
    
    return {
        'ece': ece,
        'mce': mce,
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'bin_confidences': bin_confidences,
        'bin_accuracies': bin_accuracies,
        'bin_counts': bin_counts
    }


def demonstrate_calibration():
    """
    Demonstrate calibration concepts with synthetic data.
    """
    print("\n" + "=" * 70)
    print("PART 4: Calibration Concepts")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Simulate predictions for 1000 samples
    n_samples = 1000
    
    # Well-calibrated model
    print("\nExample 1: Well-Calibrated Model")
    confidences_calibrated = np.random.uniform(0.5, 1.0, n_samples)
    # Correct with probability equal to confidence
    predictions_calibrated = np.array([
        1 if np.random.rand() < conf else 0 
        for conf in confidences_calibrated
    ])
    true_labels = np.ones(n_samples)  # All samples are class 1
    
    metrics_calibrated = calculate_calibration_metrics(
        confidences_calibrated, predictions_calibrated, true_labels
    )
    
    print(f"  Accuracy: {metrics_calibrated['accuracy']:.3f}")
    print(f"  Average Confidence: {metrics_calibrated['avg_confidence']:.3f}")
    print(f"  ECE: {metrics_calibrated['ece']:.4f} (lower is better)")
    print(f"  MCE: {metrics_calibrated['mce']:.4f}")
    
    # Overconfident model
    print("\nExample 2: Overconfident Model")
    confidences_overconfident = np.random.uniform(0.8, 1.0, n_samples)
    # Actually correct only 70% of time
    predictions_overconfident = (np.random.rand(n_samples) < 0.7).astype(int)
    
    metrics_overconfident = calculate_calibration_metrics(
        confidences_overconfident, predictions_overconfident, true_labels
    )
    
    print(f"  Accuracy: {metrics_overconfident['accuracy']:.3f}")
    print(f"  Average Confidence: {metrics_overconfident['avg_confidence']:.3f}")
    print(f"  ECE: {metrics_overconfident['ece']:.4f} (higher - poorly calibrated)")
    print(f"  MCE: {metrics_overconfident['mce']:.4f}")
    
    print("\nKey Insight:")
    print("  ECE measures gap between confidence and actual accuracy")
    print("  Well-calibrated model: ECE close to 0")
    print("  Overconfident model: high confidence but lower accuracy → high ECE")


# ============================================================================
# PART 5: VISUALIZATION UTILITIES
# ============================================================================

def visualize_uncertainty_on_grid(X: np.ndarray, y: np.ndarray,
                                  xx: np.ndarray, yy: np.ndarray,
                                  uncertainty: np.ndarray):
    """
    Visualize uncertainty estimates on a 2D grid.
    
    Args:
        X: Training data points
        y: Training labels
        xx, yy: Meshgrid for visualization
        uncertainty: Uncertainty estimates on grid
    """
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Uncertainty heatmap
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, uncertainty, levels=20, cmap='RdYlBu_r', alpha=0.8)
    plt.colorbar(label='Epistemic Uncertainty')
    plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', s=20, edgecolors='k', alpha=0.6, label='Class 0')
    plt.scatter(X[y==1, 0], X[y==1, 1], c='red', s=20, edgecolors='k', alpha=0.6, label='Class 1')
    plt.title('Epistemic Uncertainty from Ensemble')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    # Plot 2: Uncertainty vs distance from training data
    plt.subplot(1, 2, 2)
    plt.hist(uncertainty.ravel(), bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Uncertainty')
    plt.ylabel('Frequency')
    plt.title('Distribution of Uncertainty Estimates')
    plt.axvline(uncertainty.mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {uncertainty.mean():.3f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('uncertainty_visualization.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'uncertainty_visualization.png'")


def plot_calibration_curve(metrics: Dict[str, float]):
    """
    Plot calibration curve (reliability diagram).
    
    Args:
        metrics: Dictionary containing calibration metrics
    """
    plt.figure(figsize=(8, 8))
    
    # Plot perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
    
    # Plot actual calibration
    confidences = metrics['bin_confidences']
    accuracies = metrics['bin_accuracies']
    counts = metrics['bin_counts']
    
    # Normalize counts for marker size
    max_count = max(counts) if counts else 1
    sizes = [100 * (c / max_count) for c in counts]
    
    plt.scatter(confidences, accuracies, s=sizes, alpha=0.6, 
                c='blue', edgecolors='black', linewidth=1.5,
                label='Model Calibration')
    
    # Connect points
    if len(confidences) > 1:
        plt.plot(confidences, accuracies, 'b-', alpha=0.3, linewidth=1)
    
    plt.xlabel('Confidence', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Calibration Curve\nECE: {metrics["ece"]:.4f}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('calibration_curve.png', dpi=150, bbox_inches='tight')
    print("Calibration curve saved as 'calibration_curve.png'")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to run all demonstrations.
    """
    print("\n" + "="*70)
    print("MODULE 63.1: MODEL UNCERTAINTY BASICS")
    print("="*70)
    
    # Part 1: Understand uncertainty types
    demonstrate_uncertainty_types()
    
    # Part 2: Temperature scaling
    demonstrate_temperature_scaling()
    
    # Part 3: Simple ensemble
    X, y, xx, yy, uncertainty = train_and_evaluate_ensemble()
    
    # Part 4: Calibration
    demonstrate_calibration()
    
    # Part 5: Visualizations
    print("\n" + "=" * 70)
    print("PART 5: Visualizations")
    print("=" * 70)
    visualize_uncertainty_on_grid(X, y, xx, yy, uncertainty)
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. Two Types of Uncertainty:
       - Epistemic: Model uncertainty (reducible with more data)
       - Aleatoric: Data uncertainty (irreducible noise)
    
    2. Temperature Scaling:
       - Controls prediction confidence
       - T > 1: less confident (smoother probabilities)
       - T < 1: more confident (sharper probabilities)
    
    3. Ensemble Methods:
       - Train multiple models independently
       - Disagreement indicates uncertainty
       - Simple but effective for epistemic uncertainty
    
    4. Calibration:
       - Model confidence should match actual accuracy
       - ECE measures calibration quality
       - Important for trustworthy predictions
    
    5. Practical Considerations:
       - Always evaluate both accuracy AND calibration
       - Visualize uncertainty to build intuition
       - Different methods capture different types of uncertainty
    """)
    
    print("\nNext Steps:")
    print("  → Try 02_intermediate_mc_dropout_ensembles.py for advanced methods")
    print("  → Experiment with different ensemble sizes")
    print("  → Apply to real datasets (MNIST, CIFAR-10)")


if __name__ == "__main__":
    main()
