"""
Module 63.2: Monte Carlo Dropout & Deep Ensembles (Intermediate Level)

This script covers more advanced uncertainty estimation techniques including
Monte Carlo (MC) Dropout and proper deep ensemble implementation. These methods
provide better uncertainty estimates than simple approaches.

Topics:
    1. Monte Carlo Dropout theory and implementation
    2. Deep Ensemble with diverse initializations
    3. Uncertainty decomposition (aleatoric vs epistemic)
    4. Bootstrap aggregating (bagging)
    5. Comparing uncertainty methods

Mathematical Background:
    
    Monte Carlo Dropout:
        - Use dropout at test time to approximate Bayesian inference
        - Predictive mean: E[y|x,D] ≈ (1/T) Σ f(x, ω_t)
        - Predictive variance: Var[y|x,D] ≈ (1/T) Σ f(x,ω_t)² - E[y|x,D]²
        - Theoretical connection to variational inference
    
    Deep Ensembles:
        - Train M models with different random initializations
        - Adversarial training for diverse predictions
        - Proper scoring rules for probabilistic predictions
    
    Uncertainty Decomposition:
        Total Uncertainty = Aleatoric + Epistemic
        Var[y|x,D] = E_w[Var[y|x,w]] + Var_w[E[y|x,w]]

Learning Objectives:
    - Implement MC Dropout correctly
    - Build and train deep ensembles
    - Decompose uncertainty into components
    - Compare different uncertainty estimation methods
    - Apply to real datasets (MNIST, CIFAR-10)

Prerequisites:
    - Module 63.1: Uncertainty Basics
    - Understanding of dropout regularization
    - Familiarity with PyTorch training loops

Time: 3-4 hours
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from tqdm import tqdm


# ============================================================================
# PART 1: MONTE CARLO DROPOUT
# ============================================================================

class MCDropoutModel(nn.Module):
    """
    Neural network with Monte Carlo Dropout for uncertainty estimation.
    
    Key Idea:
        - Keep dropout active during inference
        - Run multiple forward passes with different dropout masks
        - Aggregate predictions to estimate uncertainty
    
    Theoretical Foundation (Gal & Ghahramani, 2016):
        - MC Dropout approximates variational inference in Bayesian NNs
        - Each dropout mask samples from approximate posterior
        - Predictive distribution: p(y|x,D) ≈ (1/T) Σ p(y|x,w_t)
    
    Architecture:
        - Multiple hidden layers with dropout
        - Dropout rate typically 0.1-0.5
        - Important: dropout applied after activation
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 output_dim: int, dropout_rate: float = 0.5):
        """
        Initialize MC Dropout model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            dropout_rate: Dropout probability (0.0 to 1.0)
        """
        super(MCDropoutModel, self).__init__()
        
        self.dropout_rate = dropout_rate
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))  # Dropout after activation
            prev_dim = hidden_dim
        
        # Output layer (no dropout)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.
        
        Note: Dropout is active even in eval mode for MC Dropout.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Logits of shape (batch_size, output_dim)
        """
        return self.network(x)
    
    def enable_dropout(self):
        """
        Enable dropout for MC sampling at test time.
        
        This is the KEY difference from standard inference:
        - Standard: model.eval() disables dropout
        - MC Dropout: keep dropout active during inference
        """
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Keep dropout in training mode
    
    def mc_predict(self, x: torch.Tensor, n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo prediction with uncertainty estimation.
        
        Algorithm:
            1. Enable dropout
            2. Run T forward passes with different dropout masks
            3. Compute mean and variance of predictions
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            n_samples: Number of MC samples (typically 50-100)
        
        Returns:
            mean_probs: Mean predicted probabilities (batch_size, output_dim)
            epistemic_uncertainty: Epistemic uncertainty per sample (batch_size,)
        """
        self.enable_dropout()  # Critical: keep dropout active
        
        # Store predictions from each MC sample
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                # Forward pass with different dropout mask
                logits = self.forward(x)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs)
        
        # Stack predictions: (n_samples, batch_size, n_classes)
        predictions = torch.stack(predictions)
        
        # Compute statistics
        mean_probs = torch.mean(predictions, dim=0)  # Average prediction
        
        # Epistemic uncertainty: variance of predictions
        # High variance → models disagree → high uncertainty
        variance_probs = torch.var(predictions, dim=0)
        epistemic_uncertainty = torch.mean(variance_probs, dim=1)  # Average over classes
        
        return mean_probs, epistemic_uncertainty


def demonstrate_mc_dropout():
    """
    Demonstrate Monte Carlo Dropout on MNIST dataset.
    """
    print("=" * 70)
    print("PART 1: Monte Carlo Dropout")
    print("=" * 70)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load MNIST dataset
    print("\nLoading MNIST dataset...")
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
    
    # Use subset for faster training
    train_subset, _ = random_split(train_dataset, [5000, 55000])
    test_subset, _ = random_split(test_dataset, [1000, 9000])
    
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=128, shuffle=False)
    
    print(f"Training samples: {len(train_subset)}")
    print(f"Test samples: {len(test_subset)}")
    
    # Create MC Dropout model
    print("\nCreating MC Dropout model...")
    model = MCDropoutModel(
        input_dim=28*28,
        hidden_dims=[512, 256, 128],
        output_dim=10,
        dropout_rate=0.3
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Dropout rate: {model.dropout_rate}")
    
    # Train model
    print("\nTraining model...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(10):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            # Flatten images
            batch_x = batch_x.view(batch_x.size(0), -1)
            
            # Forward pass
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/10 - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # Evaluate with MC Dropout
    print("\n" + "=" * 70)
    print("MC Dropout Uncertainty Estimation")
    print("=" * 70)
    
    # Get test batch
    test_x, test_y = next(iter(test_loader))
    test_x_flat = test_x.view(test_x.size(0), -1)
    
    # Compare different numbers of MC samples
    n_samples_list = [1, 10, 50, 100]
    
    print("\nEffect of number of MC samples:")
    print("-" * 60)
    print(f"{'MC Samples':<15} {'Avg Uncertainty':<20} {'Std Uncertainty':<20}")
    print("-" * 60)
    
    for n_samples in n_samples_list:
        mean_probs, uncertainty = model.mc_predict(test_x_flat, n_samples=n_samples)
        
        avg_uncertainty = uncertainty.mean().item()
        std_uncertainty = uncertainty.std().item()
        
        print(f"{n_samples:<15} {avg_uncertainty:<20.6f} {std_uncertainty:<20.6f}")
    
    print("-" * 60)
    print("\nObservation:")
    print("  - More MC samples → more stable uncertainty estimates")
    print("  - 50-100 samples typically sufficient")
    print("  - Diminishing returns beyond 100 samples")
    
    # Analyze uncertainty on correct vs incorrect predictions
    mean_probs, uncertainty = model.mc_predict(test_x_flat, n_samples=100)
    predictions = torch.argmax(mean_probs, dim=1)
    correct_mask = (predictions == test_y).numpy()
    
    uncertainty_correct = uncertainty[correct_mask].numpy()
    uncertainty_incorrect = uncertainty[~correct_mask].numpy()
    
    print("\nUncertainty Analysis:")
    print(f"  Correct predictions - Avg uncertainty: {uncertainty_correct.mean():.6f}")
    print(f"  Incorrect predictions - Avg uncertainty: {uncertainty_incorrect.mean():.6f}")
    print(f"  Ratio (incorrect/correct): {uncertainty_incorrect.mean() / uncertainty_correct.mean():.2f}x")
    print("\n  ✓ Higher uncertainty on incorrect predictions!")
    
    return model, test_x, test_y, uncertainty


# ============================================================================
# PART 2: DEEP ENSEMBLES
# ============================================================================

class DeepEnsemble:
    """
    Proper implementation of Deep Ensembles (Lakshminarayanan et al., 2017).
    
    Key Ideas:
        1. Train M models with different random initializations
        2. Each model uses different random mini-batch ordering
        3. Aggregate predictions to measure epistemic uncertainty
        4. Optionally: adversarial training for diversity
    
    Why it works:
        - Different initializations explore different modes
        - Ensemble disagreement captures model uncertainty
        - More robust than single model
        - No approximation (unlike MC Dropout)
    
    Mathematical Framework:
        Predictive mean: μ*(x) = (1/M) Σ μ_m(x)
        Epistemic uncertainty: σ²_epistemic = (1/M) Σ (μ_m(x) - μ*(x))²
        Aleatoric uncertainty: σ²_aleatoric = (1/M) Σ σ²_m(x)
    """
    
    def __init__(self, n_models: int, model_fn, model_kwargs: dict):
        """
        Initialize deep ensemble.
        
        Args:
            n_models: Number of models in ensemble
            model_fn: Function that returns a model
            model_kwargs: Keyword arguments for model creation
        """
        self.n_models = n_models
        self.models = [model_fn(**model_kwargs) for _ in range(n_models)]
        self.model_kwargs = model_kwargs
    
    def train_ensemble(self, train_loader: DataLoader, 
                      epochs: int = 10, lr: float = 0.001,
                      device: str = 'cpu'):
        """
        Train all models in ensemble with different initializations.
        
        Each model:
            - Gets random initialization (automatic in PyTorch)
            - Sees mini-batches in random order (DataLoader shuffle=True)
            - Has independent optimizer
        
        Args:
            train_loader: Training data loader
            epochs: Number of training epochs
            lr: Learning rate
            device: Device to train on
        """
        print(f"\nTraining Deep Ensemble ({self.n_models} models)...")
        
        for i, model in enumerate(self.models):
            print(f"\n{'='*60}")
            print(f"Training Model {i+1}/{self.n_models}")
            print(f"{'='*60}")
            
            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            model.train()
            for epoch in range(epochs):
                total_loss = 0
                correct = 0
                total = 0
                
                # Progress bar for this model
                pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
                
                for batch_x, batch_y in pbar:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    
                    # Flatten if needed
                    if len(batch_x.shape) > 2:
                        batch_x = batch_x.view(batch_x.size(0), -1)
                    
                    # Forward pass
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Statistics
                    total_loss += loss.item()
                    _, predicted = torch.max(logits.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                    
                    # Update progress bar
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                accuracy = 100 * correct / total
                avg_loss = total_loss / len(train_loader)
                print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")
    
    def predict_with_uncertainty(self, x: torch.Tensor, 
                                device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get ensemble predictions with epistemic uncertainty.
        
        Returns three types of information:
            1. Mean prediction (ensemble average)
            2. Epistemic uncertainty (disagreement between models)
            3. Aleatoric uncertainty (average predicted variance)
        
        Args:
            x: Input tensor
            device: Device for computation
        
        Returns:
            mean_probs: Average predicted probabilities
            epistemic_uncertainty: Variance of predictions across models
            confidence: Max probability of mean prediction
        """
        all_probs = []
        
        # Get predictions from each model
        for model in self.models:
            model = model.to(device)
            model.eval()
            
            with torch.no_grad():
                x = x.to(device)
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs)
        
        # Stack: (n_models, batch_size, n_classes)
        all_probs = torch.stack(all_probs)
        
        # Mean prediction
        mean_probs = torch.mean(all_probs, dim=0)
        
        # Epistemic uncertainty: variance across models
        variance = torch.var(all_probs, dim=0)
        epistemic_uncertainty = torch.mean(variance, dim=1)
        
        # Confidence: max probability
        confidence = torch.max(mean_probs, dim=1)[0]
        
        return mean_probs, epistemic_uncertainty, confidence


def demonstrate_deep_ensemble():
    """
    Demonstrate Deep Ensemble on MNIST.
    """
    print("\n" + "=" * 70)
    print("PART 2: Deep Ensembles")
    print("=" * 70)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load MNIST
    print("\nPreparing data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Use subset for faster demo
    train_subset, _ = random_split(train_dataset, [5000, 55000])
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    
    # Model factory function
    def create_model():
        return MCDropoutModel(
            input_dim=28*28,
            hidden_dims=[256, 128],
            output_dim=10,
            dropout_rate=0.2
        )
    
    # Create and train ensemble
    ensemble = DeepEnsemble(
        n_models=5,
        model_fn=create_model,
        model_kwargs={}
    )
    
    ensemble.train_ensemble(train_loader, epochs=5, lr=0.001, device=device)
    
    print("\n" + "=" * 70)
    print("Ensemble trained successfully!")
    print("=" * 70)
    
    return ensemble


# ============================================================================
# PART 3: UNCERTAINTY DECOMPOSITION
# ============================================================================

def decompose_uncertainty(model_predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Decompose total uncertainty into epistemic and aleatoric components.
    
    Total Predictive Variance:
        Var[y|x,D] = E_w[Var[y|x,w]] + Var_w[E[y|x,w]]
                     \_______________/   \____________/
                      Aleatoric          Epistemic
    
    For classification with ensemble:
        - Epistemic: variance of mean predictions across models
        - Aleatoric: entropy of predictions (inherent uncertainty)
    
    Args:
        model_predictions: Tensor of shape (n_models, batch_size, n_classes)
    
    Returns:
        Dictionary with uncertainty components
    """
    # Mean prediction across models
    mean_pred = torch.mean(model_predictions, dim=0)
    
    # Epistemic uncertainty: variance of predictions
    epistemic = torch.var(model_predictions, dim=0)
    epistemic = torch.mean(epistemic, dim=1)  # Average over classes
    
    # Aleatoric uncertainty: average entropy
    epsilon = 1e-10  # For numerical stability
    entropy = -torch.sum(model_predictions * torch.log(model_predictions + epsilon), dim=2)
    aleatoric = torch.mean(entropy, dim=0)  # Average over models
    
    # Total uncertainty
    total = epistemic + aleatoric
    
    return {
        'epistemic': epistemic,
        'aleatoric': aleatoric,
        'total': total,
        'mean_prediction': mean_pred
    }


def demonstrate_uncertainty_decomposition():
    """
    Demonstrate uncertainty decomposition with synthetic data.
    """
    print("\n" + "=" * 70)
    print("PART 3: Uncertainty Decomposition")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # Simulate ensemble predictions for 100 samples, 3 classes
    n_models = 5
    n_samples = 100
    n_classes = 3
    
    # Case 1: High epistemic, low aleatoric
    # Models disagree, but each is confident
    print("\nCase 1: High Epistemic, Low Aleatoric Uncertainty")
    print("-" * 60)
    preds_epistemic = []
    for i in range(n_models):
        # Each model predicts different class confidently
        class_idx = i % n_classes
        probs = torch.zeros(n_samples, n_classes)
        probs[:, class_idx] = 0.9
        probs[:, (class_idx + 1) % n_classes] = 0.08
        probs[:, (class_idx + 2) % n_classes] = 0.02
        preds_epistemic.append(probs)
    
    preds_epistemic = torch.stack(preds_epistemic)
    unc_epistemic = decompose_uncertainty(preds_epistemic)
    
    print(f"Epistemic uncertainty: {unc_epistemic['epistemic'].mean():.4f}")
    print(f"Aleatoric uncertainty: {unc_epistemic['aleatoric'].mean():.4f}")
    print("→ Models disagree on prediction (high epistemic)")
    print("→ But each model is confident (low aleatoric)")
    
    # Case 2: Low epistemic, high aleatoric
    # Models agree, but all are uncertain
    print("\nCase 2: Low Epistemic, High Aleatoric Uncertainty")
    print("-" * 60)
    preds_aleatoric = []
    for i in range(n_models):
        # All models predict similar uniform distribution
        probs = torch.ones(n_samples, n_classes) / n_classes
        probs += torch.randn(n_samples, n_classes) * 0.05
        probs = F.softmax(probs, dim=1)
        preds_aleatoric.append(probs)
    
    preds_aleatoric = torch.stack(preds_aleatoric)
    unc_aleatoric = decompose_uncertainty(preds_aleatoric)
    
    print(f"Epistemic uncertainty: {unc_aleatoric['epistemic'].mean():.4f}")
    print(f"Aleatoric uncertainty: {unc_aleatoric['aleatoric'].mean():.4f}")
    print("→ Models agree on prediction (low epistemic)")
    print("→ But predictions are uncertain/uniform (high aleatoric)")
    
    print("\n" + "=" * 70)
    print("Key Insight:")
    print("  Epistemic uncertainty → can be reduced with more data/better model")
    print("  Aleatoric uncertainty → inherent in data, cannot be reduced")
    print("=" * 70)


# ============================================================================
# PART 4: COMPARISON OF METHODS
# ============================================================================

def compare_uncertainty_methods(mc_model, ensemble, test_data: torch.Tensor):
    """
    Compare MC Dropout vs Deep Ensemble on same data.
    
    Args:
        mc_model: MC Dropout model
        ensemble: Deep Ensemble
        test_data: Test samples
    """
    print("\n" + "=" * 70)
    print("PART 4: Comparing Uncertainty Methods")
    print("=" * 70)
    
    # Flatten test data
    test_data_flat = test_data.view(test_data.size(0), -1)
    
    # MC Dropout predictions
    print("\nMC Dropout (100 samples)...")
    mc_probs, mc_uncertainty = mc_model.mc_predict(test_data_flat, n_samples=100)
    
    # Ensemble predictions
    print("Deep Ensemble...")
    ens_probs, ens_uncertainty, ens_confidence = ensemble.predict_with_uncertainty(test_data_flat)
    
    # Compare statistics
    print("\n" + "=" * 70)
    print("Comparison Results:")
    print("=" * 70)
    print(f"\n{'Metric':<30} {'MC Dropout':<15} {'Deep Ensemble':<15}")
    print("-" * 60)
    print(f"{'Average Uncertainty':<30} {mc_uncertainty.mean():.6f}      {ens_uncertainty.mean():.6f}")
    print(f"{'Std of Uncertainty':<30} {mc_uncertainty.std():.6f}      {ens_uncertainty.std():.6f}")
    print(f"{'Max Uncertainty':<30} {mc_uncertainty.max():.6f}      {ens_uncertainty.max():.6f}")
    print(f"{'Min Uncertainty':<30} {mc_uncertainty.min():.6f}      {ens_uncertainty.min():.6f}")
    
    print("\n" + "=" * 70)
    print("Trade-offs:")
    print("=" * 70)
    print("""
    MC Dropout:
      ✓ Single model (less memory)
      ✓ Fast to train
      ✗ Approximate (variational approximation)
      ✗ Requires many forward passes at test time
    
    Deep Ensemble:
      ✓ No approximation
      ✓ Better calibrated
      ✓ More robust
      ✗ Multiple models (more memory)
      ✗ Slower to train
    """)


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_uncertainty_comparison(test_images: torch.Tensor,
                                    mc_uncertainty: torch.Tensor,
                                    ensemble_uncertainty: torch.Tensor,
                                    n_samples: int = 10):
    """
    Visualize uncertainty estimates from both methods.
    
    Args:
        test_images: Test images
        mc_uncertainty: MC Dropout uncertainty
        ensemble_uncertainty: Ensemble uncertainty
        n_samples: Number of samples to show
    """
    fig, axes = plt.subplots(3, n_samples, figsize=(15, 5))
    
    for i in range(n_samples):
        # Original image
        axes[0, i].imshow(test_images[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Image', fontsize=10)
        
        # MC Dropout uncertainty (heatmap)
        axes[1, i].bar([0], [mc_uncertainty[i].item()], color='blue', alpha=0.7)
        axes[1, i].set_ylim([0, max(mc_uncertainty).item() * 1.1])
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        if i == 0:
            axes[1, i].set_ylabel('MC Dropout', fontsize=10)
        
        # Ensemble uncertainty (heatmap)
        axes[2, i].bar([0], [ensemble_uncertainty[i].item()], color='red', alpha=0.7)
        axes[2, i].set_ylim([0, max(ensemble_uncertainty).item() * 1.1])
        axes[2, i].set_xticks([])
        axes[2, i].set_yticks([])
        if i == 0:
            axes[2, i].set_ylabel('Ensemble', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('uncertainty_comparison.png', dpi=150, bbox_inches='tight')
    print("\nComparison visualization saved as 'uncertainty_comparison.png'")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to run all demonstrations.
    """
    print("\n" + "="*70)
    print("MODULE 63.2: MC DROPOUT & DEEP ENSEMBLES")
    print("="*70)
    
    # Part 1: MC Dropout
    mc_model, test_x, test_y, mc_uncertainty = demonstrate_mc_dropout()
    
    # Part 2: Deep Ensemble
    ensemble = demonstrate_deep_ensemble()
    
    # Part 3: Uncertainty Decomposition
    demonstrate_uncertainty_decomposition()
    
    # Part 4: Comparison
    compare_uncertainty_methods(mc_model, ensemble, test_x)
    
    # Visualizations
    print("\n" + "=" * 70)
    print("Generating visualizations...")
    print("=" * 70)
    
    # Get ensemble uncertainty for visualization
    test_x_flat = test_x.view(test_x.size(0), -1)
    _, ens_uncertainty, _ = ensemble.predict_with_uncertainty(test_x_flat)
    
    visualize_uncertainty_comparison(test_x, mc_uncertainty, ens_uncertainty)
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. Monte Carlo Dropout:
       - Approximate Bayesian inference
       - Single model with multiple stochastic forward passes
       - Efficient but approximate
    
    2. Deep Ensembles:
       - Multiple independently trained models
       - No approximation, better calibrated
       - Higher computational cost
    
    3. Uncertainty Decomposition:
       - Epistemic: model uncertainty (reducible)
       - Aleatoric: data uncertainty (irreducible)
       - Important for interpretation and decision-making
    
    4. Practical Guidelines:
       - Use 50-100 MC samples for stable estimates
       - 3-5 ensemble members often sufficient
       - Always validate uncertainty quality
       - Consider computational budget
    
    5. When to Use What:
       - MC Dropout: Resource-constrained, fast inference needed
       - Deep Ensemble: High-stakes applications, best performance
       - Both: Maximum reliability
    """)
    
    print("\nNext Steps:")
    print("  → Try 03_advanced_bayesian_uncertainty.py for Bayesian approaches")
    print("  → Experiment with ensemble size vs uncertainty quality")
    print("  → Apply to your own datasets")


if __name__ == "__main__":
    main()
