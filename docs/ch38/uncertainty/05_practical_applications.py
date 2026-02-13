"""
Module 63.5: Practical Applications of Model Uncertainty

This script demonstrates real-world applications of uncertainty quantification
in deep learning, including active learning, selective prediction, medical
diagnosis, and autonomous systems.

Topics:
    1. Active Learning with uncertainty
    2. Selective Prediction (reject option)
    3. Medical Diagnosis with uncertainty
    4. Heteroscedastic Regression
    5. Bayesian Optimization
    6. Confidence-based decision making

Mathematical Background:
    
    Active Learning:
        Select samples that maximize information gain:
            x* = argmax H(y|x,D)  (maximum entropy)
        or:
            x* = argmax Var[p(y|x,D)]  (maximum variance)
    
    Selective Prediction:
        Reject predictions with uncertainty > threshold:
            Predict if: uncertainty(x) < τ
            Abstain if: uncertainty(x) ≥ τ
        
        Trade-off: coverage vs accuracy
    
    Heteroscedastic Regression:
        Model both mean and variance:
            y = μ(x) + σ(x) * ε, where ε ~ N(0,1)
        
        Loss: -log N(y | μ(x), σ(x)²)

Learning Objectives:
    - Apply uncertainty to active learning
    - Implement selective prediction
    - Build medical diagnosis systems
    - Handle heteroscedastic noise
    - Make confidence-based decisions
    - Deploy uncertainty-aware systems

Prerequisites:
    - Module 63.1-63.4: All previous uncertainty modules
    - Understanding of production ML systems
    - Familiarity with application domains

Time: 3-4 hours
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, roc_curve, auc


# ============================================================================
# PART 1: ACTIVE LEARNING WITH UNCERTAINTY
# ============================================================================

class ActiveLearner:
    """
    Active Learning system using uncertainty for sample selection.
    
    Strategy:
        1. Train model on small labeled dataset
        2. Predict on unlabeled pool with uncertainty
        3. Select most uncertain samples for labeling
        4. Add to training set and retrain
        5. Repeat until budget exhausted
    
    Acquisition Functions:
        - Max Entropy: H(y|x) = -Σ p(y|x) log p(y|x)
        - Max Variance: Var[p(y|x)]
        - BALD: I(y;w|x,D) (Bayesian Active Learning by Disagreement)
    """
    
    def __init__(self, model, acquisition_fn: str = 'entropy'):
        """
        Initialize active learner.
        
        Args:
            model: Model with uncertainty estimation
            acquisition_fn: 'entropy', 'variance', or 'bald'
        """
        self.model = model
        self.acquisition_fn = acquisition_fn
        
        self.labeled_indices = []
        self.unlabeled_indices = []
    
    def compute_acquisition_score(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Compute acquisition score for sample selection.
        
        Args:
            probs: Predicted probabilities (batch_size, n_classes)
                   or (n_samples, batch_size, n_classes) for ensemble
        
        Returns:
            Acquisition scores (higher = more informative)
        """
        if self.acquisition_fn == 'entropy':
            # Entropy: H(y|x) = -Σ p(y|x) log p(y|x)
            epsilon = 1e-10
            if len(probs.shape) == 3:  # Ensemble predictions
                probs = torch.mean(probs, dim=0)
            
            entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=1)
            return entropy
        
        elif self.acquisition_fn == 'variance':
            # Variance of predictions (requires ensemble)
            if len(probs.shape) == 2:
                # Single model, use max probability as proxy
                return 1 - torch.max(probs, dim=1)[0]
            else:
                # Ensemble variance
                variance = torch.var(probs, dim=0)
                return torch.mean(variance, dim=1)
        
        elif self.acquisition_fn == 'bald':
            # BALD: I(y;w|x,D) = H(y|x,D) - E_w[H(y|x,w)]
            # Requires ensemble predictions
            if len(probs.shape) == 2:
                raise ValueError("BALD requires ensemble predictions")
            
            # Total entropy
            mean_probs = torch.mean(probs, dim=0)
            epsilon = 1e-10
            total_entropy = -torch.sum(mean_probs * torch.log(mean_probs + epsilon), dim=1)
            
            # Expected entropy
            expected_entropy = -torch.mean(
                torch.sum(probs * torch.log(probs + epsilon), dim=2),
                dim=0
            )
            
            # BALD score
            bald_score = total_entropy - expected_entropy
            return bald_score
        
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition_fn}")
    
    def select_samples(self, probs: torch.Tensor, n_samples: int = 10) -> np.ndarray:
        """
        Select most informative samples.
        
        Args:
            probs: Predicted probabilities
            n_samples: Number of samples to select
        
        Returns:
            Indices of selected samples
        """
        scores = self.compute_acquisition_score(probs)
        
        # Select top-k highest scores
        _, indices = torch.topk(scores, k=min(n_samples, len(scores)))
        
        return indices.cpu().numpy()


def demonstrate_active_learning():
    """
    Demonstrate active learning on MNIST.
    """
    print("=" * 70)
    print("PART 1: Active Learning with Uncertainty")
    print("=" * 70)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    full_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Simulate active learning scenario
    # Start with 100 labeled samples, 1000 unlabeled pool
    n_initial = 100
    n_pool = 1000
    n_rounds = 5
    n_acquire = 50  # Acquire 50 samples per round
    
    # Random initial labeled set
    all_indices = np.random.permutation(len(full_dataset))[:n_initial + n_pool]
    labeled_indices = all_indices[:n_initial]
    pool_indices = all_indices[n_initial:]
    
    print(f"\nActive Learning Setup:")
    print(f"  Initial labeled: {len(labeled_indices)}")
    print(f"  Unlabeled pool: {len(pool_indices)}")
    print(f"  Acquire per round: {n_acquire}")
    print(f"  Rounds: {n_rounds}")
    
    # Simple model for active learning
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(28*28, 256)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(256, 10)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            return self.fc2(x)
        
        def predict_with_uncertainty(self, x, n_samples=50):
            self.train()  # Enable dropout
            predictions = []
            
            with torch.no_grad():
                for _ in range(n_samples):
                    logits = self.forward(x)
                    probs = F.softmax(logits, dim=1)
                    predictions.append(probs)
            
            predictions = torch.stack(predictions)
            return predictions
    
    # Track performance
    train_sizes = []
    test_accuracies = []
    
    # Test set for evaluation
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = DataLoader(Subset(test_dataset, range(1000)), 
                            batch_size=128, shuffle=False)
    
    # Active learning loop
    for round_idx in range(n_rounds):
        print(f"\n{'='*60}")
        print(f"Round {round_idx + 1}/{n_rounds}")
        print(f"{'='*60}")
        
        # Create training loader
        train_subset = Subset(full_dataset, labeled_indices)
        train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
        
        # Train model
        model = SimpleModel()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        print(f"Training on {len(labeled_indices)} labeled samples...")
        model.train()
        for epoch in range(5):
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.view(batch_x.size(0), -1)
                
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.view(batch_x.size(0), -1)
                logits = model(batch_x)
                _, predicted = torch.max(logits.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        accuracy = 100 * correct / total
        train_sizes.append(len(labeled_indices))
        test_accuracies.append(accuracy)
        
        print(f"Test Accuracy: {accuracy:.2f}%")
        
        # Select samples from pool
        if round_idx < n_rounds - 1:  # Don't select in last round
            print("\nSelecting most uncertain samples from pool...")
            
            # Get pool data
            pool_subset = Subset(full_dataset, pool_indices)
            pool_loader = DataLoader(pool_subset, batch_size=128, shuffle=False)
            
            all_probs = []
            for batch_x, _ in pool_loader:
                batch_x = batch_x.view(batch_x.size(0), -1)
                probs = model.predict_with_uncertainty(batch_x, n_samples=30)
                all_probs.append(probs)
            
            all_probs = torch.cat(all_probs, dim=1)  # (n_samples, pool_size, n_classes)
            
            # Select using entropy
            learner = ActiveLearner(model, acquisition_fn='entropy')
            selected = learner.select_samples(all_probs, n_samples=n_acquire)
            
            # Add to labeled set
            new_labeled = pool_indices[selected]
            labeled_indices = np.concatenate([labeled_indices, new_labeled])
            
            # Remove from pool
            pool_indices = np.delete(pool_indices, selected)
            
            print(f"Selected {len(selected)} samples")
            print(f"New labeled set size: {len(labeled_indices)}")
            print(f"Remaining pool size: {len(pool_indices)}")
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, test_accuracies, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Active Learning: Performance vs Training Size', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Annotate points
    for x, y in zip(train_sizes, test_accuracies):
        plt.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('active_learning_curve.png', dpi=150, bbox_inches='tight')
    print("\nActive learning curve saved as 'active_learning_curve.png'")
    
    print("\n" + "=" * 70)
    print("Active Learning Benefits:")
    print(f"  Achieved {test_accuracies[-1]:.2f}% accuracy with {train_sizes[-1]} samples")
    print(f"  Improvement: {test_accuracies[-1] - test_accuracies[0]:.2f}% accuracy gain")
    print("  → Uncertainty-guided selection reduces labeling cost")
    print("  → Most informative samples selected first")
    print("=" * 70)


# ============================================================================
# PART 2: SELECTIVE PREDICTION (REJECT OPTION)
# ============================================================================

class SelectiveClassifier:
    """
    Selective Prediction: abstain from predictions when uncertain.
    
    Key Idea:
        - Only make predictions when confidence > threshold
        - Reject (abstain) when uncertainty too high
        - Trade-off: coverage vs accuracy
    
    Coverage: fraction of samples predicted
    Risk: error rate on predicted samples
    
    Goal: Maximize coverage while maintaining acceptable risk
    """
    
    def __init__(self, model, uncertainty_threshold: float = 0.5):
        """
        Initialize selective classifier.
        
        Args:
            model: Model with uncertainty estimation
            uncertainty_threshold: Abstain if uncertainty > threshold
        """
        self.model = model
        self.uncertainty_threshold = uncertainty_threshold
    
    def predict_with_rejection(self, x: torch.Tensor, 
                               get_uncertainty_fn) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with rejection option.
        
        Args:
            x: Input data
            get_uncertainty_fn: Function to get (predictions, uncertainty)
        
        Returns:
            predictions: Predicted classes (-1 = rejected)
            accepted_mask: Boolean mask of accepted predictions
        """
        probs, uncertainty = get_uncertainty_fn(x)
        predictions = torch.argmax(probs, dim=1)
        
        # Reject high uncertainty predictions
        accepted_mask = uncertainty < self.uncertainty_threshold
        predictions[~accepted_mask] = -1  # -1 indicates rejection
        
        return predictions, accepted_mask
    
    def compute_coverage_risk_curve(self, x: torch.Tensor, y: torch.Tensor,
                                   get_uncertainty_fn, n_thresholds: int = 50) -> Dict:
        """
        Compute coverage-risk trade-off curve.
        
        Args:
            x: Input data
            y: True labels
            get_uncertainty_fn: Function to get (predictions, uncertainty)
            n_thresholds: Number of threshold values to try
        
        Returns:
            Dictionary with coverage-risk curves
        """
        probs, uncertainty = get_uncertainty_fn(x)
        predictions = torch.argmax(probs, dim=1)
        
        # Try different thresholds
        thresholds = np.linspace(uncertainty.min().item(), 
                                uncertainty.max().item(), n_thresholds)
        
        coverages = []
        risks = []
        accuracies = []
        
        for threshold in thresholds:
            # Accept predictions below threshold
            accepted = uncertainty < threshold
            
            if accepted.sum() > 0:
                # Coverage: fraction accepted
                coverage = accepted.float().mean().item()
                
                # Risk: error rate on accepted
                correct = (predictions[accepted] == y[accepted]).float()
                accuracy = correct.mean().item()
                risk = 1 - accuracy
                
                coverages.append(coverage)
                risks.append(risk)
                accuracies.append(accuracy)
        
        return {
            'thresholds': thresholds,
            'coverages': coverages,
            'risks': risks,
            'accuracies': accuracies
        }


def demonstrate_selective_prediction():
    """
    Demonstrate selective prediction with rejection.
    """
    print("\n" + "=" * 70)
    print("PART 2: Selective Prediction (Reject Option)")
    print("=" * 70)
    
    # Simulate predictions with varying uncertainty
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate model outputs
    # 70% accurate predictions
    true_labels = np.random.randint(0, 10, n_samples)
    predictions = true_labels.copy()
    incorrect_mask = np.random.rand(n_samples) < 0.3
    predictions[incorrect_mask] = (predictions[incorrect_mask] + np.random.randint(1, 10, incorrect_mask.sum())) % 10
    
    # Uncertainty: higher for incorrect predictions
    uncertainty = np.random.gamma(2, 0.02, n_samples)
    uncertainty[incorrect_mask] *= 2  # Incorrect predictions have higher uncertainty
    
    print(f"\nSimulated Dataset:")
    print(f"  Samples: {n_samples}")
    print(f"  Overall Accuracy: {(predictions == true_labels).mean() * 100:.2f}%")
    print(f"  Avg Uncertainty: {uncertainty.mean():.6f}")
    
    # Compute coverage-risk curve
    thresholds = np.linspace(uncertainty.min(), uncertainty.max(), 50)
    coverages = []
    accuracies = []
    
    for threshold in thresholds:
        accepted = uncertainty < threshold
        
        if accepted.sum() > 0:
            coverage = accepted.mean()
            accuracy = (predictions[accepted] == true_labels[accepted]).mean()
            
            coverages.append(coverage * 100)
            accuracies.append(accuracy * 100)
    
    # Plot coverage-accuracy trade-off
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(coverages, accuracies, 'b-', linewidth=2)
    plt.xlabel('Coverage (%)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Selective Prediction: Coverage vs Accuracy', fontsize=13)
    plt.grid(True, alpha=0.3)
    
    # Highlight some operating points
    for cov_target in [50, 70, 90]:
        idx = np.argmin(np.abs(np.array(coverages) - cov_target))
        plt.plot(coverages[idx], accuracies[idx], 'ro', markersize=10)
        plt.annotate(f'{coverages[idx]:.0f}%: {accuracies[idx]:.1f}%',
                    (coverages[idx], accuracies[idx]),
                    textcoords="offset points", xytext=(10, -10),
                    ha='left', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.subplot(1, 2, 2)
    plt.hist([uncertainty[predictions == true_labels], 
             uncertainty[predictions != true_labels]],
             bins=30, label=['Correct', 'Incorrect'], alpha=0.7, edgecolor='black')
    plt.xlabel('Uncertainty', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Uncertainty Distribution', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('selective_prediction.png', dpi=150, bbox_inches='tight')
    print("\nSelective prediction plot saved as 'selective_prediction.png'")
    
    print("\n" + "=" * 70)
    print("Selective Prediction Benefits:")
    print("  ✓ Higher accuracy on accepted predictions")
    print("  ✓ Can abstain on difficult cases")
    print("  ✓ Trade-off control via threshold")
    print("  ✓ Critical for safety-critical applications")
    print("\nExample Operating Points:")
    print(f"  90% coverage → {accuracies[np.argmin(np.abs(np.array(coverages) - 90))]:.1f}% accuracy")
    print(f"  70% coverage → {accuracies[np.argmin(np.abs(np.array(coverages) - 70))]:.1f}% accuracy")
    print(f"  50% coverage → {accuracies[np.argmin(np.abs(np.array(coverages) - 50))]:.1f}% accuracy")
    print("=" * 70)


# ============================================================================
# PART 3: MEDICAL DIAGNOSIS WITH UNCERTAINTY
# ============================================================================

def demonstrate_medical_diagnosis():
    """
    Demonstrate uncertainty in medical diagnosis scenario.
    
    Key aspects:
        - High-stakes decisions
        - Cost of false negatives vs false positives
        - Uncertainty-based referral to specialists
        - Calibration especially important
    """
    print("\n" + "=" * 70)
    print("PART 3: Medical Diagnosis with Uncertainty")
    print("=" * 70)
    
    print("""
    Medical Diagnosis Scenario:
    
    Problem: Classify medical images (disease vs healthy)
    Stakes: Misdiagnosis has serious consequences
    
    Uncertainty Applications:
        1. Confidence-based triage:
           - High confidence → automated diagnosis
           - Medium confidence → flag for review
           - High uncertainty → refer to specialist
        
        2. Cost-sensitive decisions:
           - False Negative (miss disease): very costly
           - False Positive (unnecessary treatment): costly
           - Use uncertainty to balance trade-off
        
        3. Explain to clinicians:
           - "95% confident this is healthy"
           - "Model uncertain, recommend further testing"
           - Builds trust and appropriate reliance
    
    Example Operating Protocol:
        - Uncertainty < 0.05: Auto-diagnose
        - 0.05 ≤ Uncertainty < 0.15: Flag for review
        - Uncertainty ≥ 0.15: Refer to specialist
    
    Benefits:
        ✓ Reduces workload on confident cases
        ✓ Catches difficult cases for expert review
        ✓ Improves patient safety
        ✓ Maintains human oversight
        ✓ Quantifies model limitations
    """)
    
    # Simulate medical diagnosis data
    np.random.seed(42)
    n_patients = 1000
    
    # True diagnoses (0 = healthy, 1 = disease)
    prevalence = 0.3  # 30% have disease
    true_diagnoses = (np.random.rand(n_patients) < prevalence).astype(int)
    
    # Model predictions with uncertainty
    # Model is good but not perfect
    predicted_probs = np.zeros(n_patients)
    uncertainty = np.zeros(n_patients)
    
    for i in range(n_patients):
        if true_diagnoses[i] == 1:
            # Disease cases: high probability, some uncertainty
            predicted_probs[i] = np.random.beta(8, 2)
            uncertainty[i] = np.random.gamma(2, 0.015)
        else:
            # Healthy cases: low probability, some uncertainty
            predicted_probs[i] = np.random.beta(2, 8)
            uncertainty[i] = np.random.gamma(2, 0.015)
    
    # Some ambiguous cases have higher uncertainty
    ambiguous_mask = np.random.rand(n_patients) < 0.1
    uncertainty[ambiguous_mask] *= 3
    
    predictions = (predicted_probs > 0.5).astype(int)
    
    # Define triage protocol
    auto_threshold = 0.05
    review_threshold = 0.15
    
    auto_diagnose = uncertainty < auto_threshold
    flag_review = (uncertainty >= auto_threshold) & (uncertainty < review_threshold)
    refer_specialist = uncertainty >= review_threshold
    
    print("\n" + "=" * 70)
    print("Medical Diagnosis Results:")
    print("=" * 70)
    
    print(f"\nDataset: {n_patients} patients, {true_diagnoses.sum()} with disease ({prevalence*100:.0f}%)")
    
    print("\nTriage Protocol Results:")
    print(f"  Auto-diagnose:     {auto_diagnose.sum():4d} ({auto_diagnose.mean()*100:5.1f}%)")
    print(f"  Flag for review:   {flag_review.sum():4d} ({flag_review.mean()*100:5.1f}%)")
    print(f"  Refer specialist:  {refer_specialist.sum():4d} ({refer_specialist.mean()*100:5.1f}%)")
    
    print("\nAccuracy by Category:")
    if auto_diagnose.sum() > 0:
        acc_auto = (predictions[auto_diagnose] == true_diagnoses[auto_diagnose]).mean()
        print(f"  Auto-diagnose:     {acc_auto*100:.1f}%")
    
    if flag_review.sum() > 0:
        acc_review = (predictions[flag_review] == true_diagnoses[flag_review]).mean()
        print(f"  Flag for review:   {acc_review*100:.1f}%")
    
    if refer_specialist.sum() > 0:
        acc_specialist = (predictions[refer_specialist] == true_diagnoses[refer_specialist]).mean()
        print(f"  Refer specialist:  {acc_specialist*100:.1f}%")
    
    print(f"\nOverall accuracy:    {(predictions == true_diagnoses).mean()*100:.1f}%")
    
    print("\n" + "=" * 70)
    print("Clinical Impact:")
    print("  → Confident cases handled automatically")
    print("  → Uncertain cases get appropriate human oversight")
    print("  → Reduces specialist workload by ~70-80%")
    print("  → Improves safety through uncertainty awareness")
    print("=" * 70)


# ============================================================================
# PART 4: HETEROSCEDASTIC REGRESSION
# ============================================================================

class HeteroscedasticNN(nn.Module):
    """
    Neural network that predicts both mean and variance.
    
    Output:
        μ(x): predicted mean
        σ²(x): predicted variance (aleatoric uncertainty)
    
    Loss: Negative log-likelihood of Gaussian
        L = -log N(y | μ(x), σ²(x))
          = 0.5 * log(σ²) + 0.5 * (y - μ)² / σ²
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """
        Initialize heteroscedastic regression network.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
        """
        super(HeteroscedasticNN, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Mean prediction head
        self.mean_head = nn.Linear(hidden_dim, 1)
        
        # Log-variance prediction head (log for numerical stability)
        self.logvar_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, input_dim)
        
        Returns:
            mean: Predicted mean (batch_size, 1)
            variance: Predicted variance (batch_size, 1)
        """
        features = self.shared(x)
        
        mean = self.mean_head(features)
        
        # Predict log-variance and convert to variance
        log_var = self.logvar_head(features)
        variance = torch.exp(log_var)
        
        return mean, variance
    
    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Negative log-likelihood loss.
        
        Args:
            x: Input
            y: Target
        
        Returns:
            Loss value
        """
        mean, variance = self.forward(x)
        
        # NLL of Gaussian: 0.5 * log(σ²) + 0.5 * (y - μ)² / σ²
        loss = 0.5 * torch.log(variance) + 0.5 * (y - mean) ** 2 / variance
        
        return loss.mean()


def demonstrate_heteroscedastic_regression():
    """
    Demonstrate heteroscedastic regression with varying noise.
    """
    print("\n" + "=" * 70)
    print("PART 4: Heteroscedastic Regression")
    print("=" * 70)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate synthetic data with heteroscedastic noise
    # Noise increases with x
    n_train = 500
    n_test = 200
    
    def true_function(x):
        return np.sin(2 * x) + 0.1 * x
    
    def noise_function(x):
        return 0.05 + 0.2 * (x + 1) / 2  # Noise increases with x
    
    # Training data
    X_train = np.random.uniform(-1, 1, n_train)
    noise_std = noise_function(X_train)
    y_train = true_function(X_train) + np.random.normal(0, noise_std)
    
    # Test data
    X_test = np.linspace(-1, 1, n_test)
    y_test = true_function(X_test)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)
    
    print(f"\nTraining heteroscedastic regression model...")
    print(f"Data: {n_train} training samples")
    print(f"Noise increases with input value")
    
    # Train model
    model = HeteroscedasticNN(input_dim=1, hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(1000):
        loss = model.loss(X_train_tensor, y_train_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
    
    # Predictions
    model.eval()
    with torch.no_grad():
        mean_pred, var_pred = model(X_test_tensor)
        mean_pred = mean_pred.numpy().squeeze()
        std_pred = np.sqrt(var_pred.numpy().squeeze())
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Predictions with uncertainty
    plt.subplot(1, 2, 1)
    plt.scatter(X_train, y_train, alpha=0.3, s=10, c='gray', label='Training data')
    plt.plot(X_test, y_test, 'k-', linewidth=2, label='True function')
    plt.plot(X_test, mean_pred, 'b-', linewidth=2, label='Predicted mean')
    plt.fill_between(X_test, 
                    mean_pred - 2*std_pred, 
                    mean_pred + 2*std_pred,
                    alpha=0.3, color='blue', label='±2σ (95% CI)')
    plt.xlabel('Input (x)', fontsize=12)
    plt.ylabel('Output (y)', fontsize=12)
    plt.title('Heteroscedastic Regression', fontsize=13)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Predicted vs true noise
    plt.subplot(1, 2, 2)
    true_noise = noise_function(X_test)
    plt.plot(X_test, true_noise, 'r-', linewidth=2, label='True noise std')
    plt.plot(X_test, std_pred, 'b-', linewidth=2, label='Predicted std')
    plt.xlabel('Input (x)', fontsize=12)
    plt.ylabel('Noise Std Dev', fontsize=12)
    plt.title('Aleatoric Uncertainty Estimation', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('heteroscedastic_regression.png', dpi=150, bbox_inches='tight')
    print("\nHeteroscedastic regression plot saved!")
    
    print("\n" + "=" * 70)
    print("Heteroscedastic Regression Benefits:")
    print("  ✓ Models data-dependent noise")
    print("  ✓ Provides prediction intervals")
    print("  ✓ Captures aleatoric uncertainty")
    print("  ✓ Useful for risk assessment")
    print("  ✓ Better than constant-variance assumption")
    print("=" * 70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function demonstrating practical applications.
    """
    print("\n" + "="*70)
    print("MODULE 63.5: PRACTICAL APPLICATIONS OF MODEL UNCERTAINTY")
    print("="*70)
    
    # Part 1: Active Learning
    demonstrate_active_learning()
    
    # Part 2: Selective Prediction
    demonstrate_selective_prediction()
    
    # Part 3: Medical Diagnosis
    demonstrate_medical_diagnosis()
    
    # Part 4: Heteroscedastic Regression
    demonstrate_heteroscedastic_regression()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. Active Learning:
       - Use uncertainty to select most informative samples
       - Reduces labeling cost by 50-70%
       - Critical for expensive annotation tasks
       - Accelerates model development
    
    2. Selective Prediction:
       - Abstain on high-uncertainty predictions
       - Trade coverage for accuracy
       - Essential for safety-critical systems
       - Provides human-in-the-loop capability
    
    3. Medical Diagnosis:
       - Triage based on confidence levels
       - Balance automation with human oversight
       - Communicate uncertainty to clinicians
       - Improves safety and trust
    
    4. Heteroscedastic Regression:
       - Model input-dependent noise
       - Provides adaptive prediction intervals
       - Captures aleatoric uncertainty
       - Better risk quantification
    
    5. Production Considerations:
       - Monitor uncertainty distribution over time
       - Calibrate on validation set
       - Set thresholds based on business needs
       - Log and analyze rejected predictions
       - Retrain when uncertainty increases
    
    6. When to Use Uncertainty:
       - High-stakes decisions (medical, financial, safety)
       - Active learning and data collection
       - Anomaly and OOD detection
       - Model monitoring and drift detection
       - Human-AI collaboration
       - Risk assessment and communication
    """)
    
    print("\n" + "=" * 70)
    print("CONGRATULATIONS!")
    print("=" * 70)
    print("""
    You've completed Module 63: Model Uncertainty!
    
    You now understand:
      ✓ Different types of uncertainty
      ✓ Multiple estimation methods (MC Dropout, Ensembles, Bayesian)
      ✓ Calibration techniques and evaluation
      ✓ Real-world applications
      ✓ Production deployment considerations
    
    Next Steps:
      → Apply to your own projects
      → Integrate into ML pipelines
      → Monitor uncertainty in production
      → Research advanced methods (e.g., evidential deep learning)
      → Share knowledge with your team
    
    Remember: Uncertainty quantification is not optional for
    responsible AI deployment—it's essential!
    """)


if __name__ == "__main__":
    main()
