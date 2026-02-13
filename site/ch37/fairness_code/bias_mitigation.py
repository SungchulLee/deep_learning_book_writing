"""
Bias Mitigation Techniques for Deep Learning
Various approaches to reduce bias in ML models.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple, Dict
from sklearn.preprocessing import StandardScaler


class ReweighingMitigation:
    """
    Pre-processing mitigation: Reweigh training samples.
    
    Assigns different weights to training samples based on their
    protected attribute and label to achieve fairness.
    """
    
    def __init__(self):
        self.weights = None
    
    def compute_weights(
        self,
        y: np.ndarray,
        sensitive_attr: np.ndarray
    ) -> np.ndarray:
        """
        Compute sample weights for reweighing.
        
        Args:
            y: Labels
            sensitive_attr: Sensitive attribute
            
        Returns:
            Sample weights
        """
        weights = np.ones(len(y))
        
        # Get unique values
        attr_values = np.unique(sensitive_attr)
        label_values = np.unique(y)
        
        # Compute expected and observed probabilities
        n = len(y)
        
        for attr_val in attr_values:
            for label_val in label_values:
                # Observed probability
                mask = (sensitive_attr == attr_val) & (y == label_val)
                p_observed = np.sum(mask) / n
                
                # Expected probability (assuming independence)
                p_attr = np.sum(sensitive_attr == attr_val) / n
                p_label = np.sum(y == label_val) / n
                p_expected = p_attr * p_label
                
                # Assign weight
                if p_observed > 0:
                    weight = p_expected / p_observed
                    weights[mask] = weight
        
        self.weights = weights
        return weights


class AdversarialDebiasing(nn.Module):
    """
    In-processing mitigation: Adversarial debiasing.
    
    Uses adversarial training to remove bias. The classifier
    learns to predict the target while an adversary tries to
    predict the sensitive attribute from the classifier's
    representations.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1
    ):
        super(AdversarialDebiasing, self).__init__()
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Classifier (predicts target label)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )
        
        # Adversary (predicts sensitive attribute)
        self.adversary = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input features
            
        Returns:
            Tuple of (classifier predictions, adversary predictions)
        """
        features = self.encoder(x)
        y_pred = self.classifier(features)
        a_pred = self.adversary(features)
        return y_pred, a_pred


def train_adversarial_debiasing(
    model: AdversarialDebiasing,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sensitive_train: np.ndarray,
    epochs: int = 100,
    learning_rate: float = 0.001,
    adversary_weight: float = 0.5
) -> AdversarialDebiasing:
    """
    Train adversarial debiasing model.
    
    Args:
        model: AdversarialDebiasing model
        X_train: Training features
        y_train: Training labels
        sensitive_train: Sensitive attributes
        epochs: Number of training epochs
        learning_rate: Learning rate
        adversary_weight: Weight for adversary loss
        
    Returns:
        Trained model
    """
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    s_tensor = torch.FloatTensor(sensitive_train).unsqueeze(1)
    
    # Optimizers
    optimizer_clf = optim.Adam(
        list(model.encoder.parameters()) + list(model.classifier.parameters()),
        lr=learning_rate
    )
    optimizer_adv = optim.Adam(model.adversary.parameters(), lr=learning_rate)
    
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        # Train adversary
        model.adversary.train()
        model.encoder.eval()
        
        optimizer_adv.zero_grad()
        _, a_pred = model(X_tensor)
        adv_loss = criterion(a_pred, s_tensor)
        adv_loss.backward()
        optimizer_adv.step()
        
        # Train classifier (maximize adversary loss, minimize classifier loss)
        model.classifier.train()
        model.encoder.train()
        model.adversary.eval()
        
        optimizer_clf.zero_grad()
        y_pred, a_pred = model(X_tensor)
        
        clf_loss = criterion(y_pred, y_tensor)
        adv_loss_for_clf = -criterion(a_pred, s_tensor)  # Negative to maximize
        
        total_loss = clf_loss + adversary_weight * adv_loss_for_clf
        total_loss.backward()
        optimizer_clf.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Classifier Loss: {clf_loss.item():.4f}, "
                  f"Adversary Loss: {adv_loss.item():.4f}")
    
    return model


class FairRepresentationLearning(nn.Module):
    """
    Learn fair representations by removing sensitive information.
    
    Uses a variational autoencoder-style approach to learn
    representations that are invariant to sensitive attributes.
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 64
    ):
        super(FairRepresentationLearning, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Sensitive attribute predictor (for regularization)
        self.sensitive_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input features
            
        Returns:
            Tuple of (latent representation, reconstruction, sensitive prediction)
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        s_pred = self.sensitive_predictor(z)
        return z, x_recon, s_pred


class ThresholdOptimization:
    """
    Post-processing mitigation: Optimize decision thresholds.
    
    Uses different classification thresholds for different groups
    to achieve fairness constraints.
    """
    
    def __init__(self, fairness_constraint: str = 'demographic_parity'):
        """
        Initialize threshold optimizer.
        
        Args:
            fairness_constraint: Type of fairness constraint
                ('demographic_parity', 'equal_opportunity', 'equalized_odds')
        """
        self.fairness_constraint = fairness_constraint
        self.thresholds = {}
    
    def optimize_thresholds(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        sensitive_attr: np.ndarray,
        num_thresholds: int = 100
    ) -> Dict[int, float]:
        """
        Find optimal thresholds for each group.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            sensitive_attr: Sensitive attribute
            num_thresholds: Number of thresholds to try
            
        Returns:
            Dictionary mapping group to optimal threshold
        """
        groups = np.unique(sensitive_attr)
        thresholds_to_try = np.linspace(0, 1, num_thresholds)
        
        best_thresholds = {}
        best_fairness = float('inf')
        
        # Grid search over all threshold combinations
        for t0 in thresholds_to_try:
            for t1 in thresholds_to_try:
                thresholds = {groups[0]: t0, groups[1]: t1}
                
                # Apply thresholds
                y_pred = np.zeros_like(y_pred_proba)
                for group, threshold in thresholds.items():
                    mask = sensitive_attr == group
                    y_pred[mask] = (y_pred_proba[mask] >= threshold).astype(int)
                
                # Calculate fairness metric
                fairness_score = self._calculate_fairness(
                    y_true, y_pred, sensitive_attr
                )
                
                if fairness_score < best_fairness:
                    best_fairness = fairness_score
                    best_thresholds = thresholds.copy()
        
        self.thresholds = best_thresholds
        return best_thresholds
    
    def _calculate_fairness(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray
    ) -> float:
        """Calculate fairness metric based on constraint."""
        groups = np.unique(sensitive_attr)
        
        if self.fairness_constraint == 'demographic_parity':
            rates = []
            for group in groups:
                mask = sensitive_attr == group
                rates.append(np.mean(y_pred[mask]))
            return abs(rates[0] - rates[1])
        
        elif self.fairness_constraint == 'equal_opportunity':
            tpr_list = []
            for group in groups:
                mask = (sensitive_attr == group) & (y_true == 1)
                if np.sum(mask) > 0:
                    tpr = np.sum((y_pred == 1) & mask) / np.sum(mask)
                    tpr_list.append(tpr)
                else:
                    tpr_list.append(0)
            return abs(tpr_list[0] - tpr_list[1])
        
        return 0.0
    
    def predict(
        self,
        y_pred_proba: np.ndarray,
        sensitive_attr: np.ndarray
    ) -> np.ndarray:
        """
        Make predictions using optimized thresholds.
        
        Args:
            y_pred_proba: Predicted probabilities
            sensitive_attr: Sensitive attribute
            
        Returns:
            Binary predictions
        """
        y_pred = np.zeros_like(y_pred_proba)
        
        for group, threshold in self.thresholds.items():
            mask = sensitive_attr == group
            y_pred[mask] = (y_pred_proba[mask] >= threshold).astype(int)
        
        return y_pred


def example_usage():
    """Example usage of bias mitigation techniques."""
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    sensitive_attr = np.random.randint(0, 2, n_samples)
    
    # Create biased labels
    y = np.random.randint(0, 2, n_samples)
    y[sensitive_attr == 0] = np.random.choice([0, 1], np.sum(sensitive_attr == 0), p=[0.3, 0.7])
    y[sensitive_attr == 1] = np.random.choice([0, 1], np.sum(sensitive_attr == 1), p=[0.7, 0.3])
    
    print("=" * 60)
    print("BIAS MITIGATION TECHNIQUES DEMO")
    print("=" * 60)
    
    # 1. Reweighing
    print("\n1. REWEIGHING")
    print("-" * 60)
    reweigh = ReweighingMitigation()
    weights = reweigh.compute_weights(y, sensitive_attr)
    print(f"Sample weights computed. Mean weight: {np.mean(weights):.4f}")
    print(f"Weight range: [{np.min(weights):.4f}, {np.max(weights):.4f}]")
    
    # 2. Threshold Optimization
    print("\n2. THRESHOLD OPTIMIZATION")
    print("-" * 60)
    y_pred_proba = np.random.rand(n_samples)
    threshold_opt = ThresholdOptimization(fairness_constraint='demographic_parity')
    optimal_thresholds = threshold_opt.optimize_thresholds(
        y, y_pred_proba, sensitive_attr, num_thresholds=20
    )
    print(f"Optimal thresholds: {optimal_thresholds}")
    
    print("\nMitigation techniques initialized successfully!")


if __name__ == "__main__":
    example_usage()
