"""
Module 56: Zero-Shot Learning - Intermediate Level (Part 1)
============================================================

Attribute-Based Zero-Shot Learning: DAP and IAP Methods

This script implements two fundamental attribute-based approaches:
1. Direct Attribute Prediction (DAP)
2. Indirect Attribute Prediction (IAP)

We'll use these on a more realistic animal dataset with multiple attributes
and explore the tradeoffs between different approaches.

Author: Deep Learning Curriculum
Level: Intermediate
Estimated Time: 60-90 minutes
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)


# ============================================================================
# PART 1: Mathematical Background
# ============================================================================

"""
DIRECT ATTRIBUTE PREDICTION (DAP)
==================================

Formulation:
1. Learn attribute classifiers: P(a_m | x) for each attribute m = 1, ..., M
2. For a test instance x, predict attribute presence probabilities
3. Classify based on match to class attribute signatures

Prediction rule:
    ŷ = argmax_c ∏_{m=1}^M P(a_m | x)^{a_c^m} * (1 - P(a_m | x))^{1 - a_c^m}

where:
- a_c^m is the m-th attribute value for class c (binary: 0 or 1)
- P(a_m | x) is the predicted probability that x has attribute m

Taking log for numerical stability:
    ŷ = argmax_c ∑_{m=1}^M [a_c^m * log P(a_m | x) + (1-a_c^m) * log(1 - P(a_m | x))]


INDIRECT ATTRIBUTE PREDICTION (IAP)
====================================

Formulation:
1. Learn probabilistic classifiers for seen classes: P(c | x)
2. Use attribute similarity to transfer to unseen classes
3. For unseen class c', compute:
   
   P(c' | x) ∝ ∑_{c ∈ seen} P(c | x) * similarity(a_c, a_c')

where similarity could be:
- Cosine similarity
- Exponential of negative distance: exp(-||a_c - a_c'||²)
- Dot product

Intuition: An unseen class that is similar (in attribute space) to highly
probable seen classes should also be probable.
"""


# ============================================================================
# PART 2: Enhanced Animal Dataset
# ============================================================================

class EnhancedAnimalDataset:
    """
    More realistic animal dataset with multiple attributes.
    
    Seen classes: dog, cat, bird, fish, snake
    Unseen classes: horse, zebra, elephant, tiger, eagle
    
    Attributes (10 total):
    - has_fur, has_feathers, has_scales
    - has_4_legs, has_2_legs, has_no_legs
    - is_large, is_carnivore, can_fly, can_swim
    """
    
    def __init__(self, n_features: int = 100, samples_per_class: int = 150):
        """
        Initialize the dataset.
        
        Args:
            n_features: Dimensionality of visual feature vectors
            samples_per_class: Number of samples to generate per class
        """
        self.n_features = n_features
        self.samples_per_class = samples_per_class
        
        # Define comprehensive attribute matrix
        # Attributes: [fur, feathers, scales, 4_legs, 2_legs, no_legs, large, carnivore, fly, swim]
        self.class_attributes = {
            # Seen classes (training)
            'dog':   np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.3, 1.0, 0.0, 0.2]),
            'cat':   np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0]),
            'bird':  np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.1, 0.3, 1.0, 0.0]),
            'fish':  np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.2, 0.5, 0.0, 1.0]),
            'snake': np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.4, 1.0, 0.0, 0.3]),
            
            # Unseen classes (testing)
            'horse':    np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0]),
            'zebra':    np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0]),
            'elephant': np.array([0.3, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.2]),
            'tiger':    np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.8, 1.0, 0.0, 0.3]),
            'eagle':    np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.6, 1.0, 1.0, 0.0]),
        }
        
        self.attribute_names = [
            'has_fur', 'has_feathers', 'has_scales',
            'has_4_legs', 'has_2_legs', 'has_no_legs',
            'is_large', 'is_carnivore', 'can_fly', 'can_swim'
        ]
        
        self.seen_classes = ['dog', 'cat', 'bird', 'fish', 'snake']
        self.unseen_classes = ['horse', 'zebra', 'elephant', 'tiger', 'eagle']
        
        self.n_attributes = len(self.attribute_names)
    
    def generate_features_from_attributes(self, attributes: np.ndarray, 
                                         n_samples: int,
                                         noise_level: float = 0.3) -> np.ndarray:
        """
        Generate visual features correlated with attributes.
        
        Args:
            attributes: Attribute vector for a class
            n_samples: Number of samples to generate
            noise_level: Amount of noise to add
        
        Returns:
            Feature matrix of shape (n_samples, n_features)
        """
        # Each attribute influences certain feature dimensions
        features = np.zeros((n_samples, self.n_features))
        
        # Map each attribute to feature dimensions
        dims_per_attr = self.n_features // self.n_attributes
        
        for attr_idx, attr_val in enumerate(attributes):
            start_dim = attr_idx * dims_per_attr
            end_dim = start_dim + dims_per_attr
            
            # Create features correlated with this attribute
            # Higher attribute value = higher feature values in these dimensions
            features[:, start_dim:end_dim] = (
                attr_val + np.random.randn(n_samples, dims_per_attr) * noise_level
            )
        
        # Add noise to remaining dimensions
        remaining_dims = self.n_features - (self.n_attributes * dims_per_attr)
        if remaining_dims > 0:
            features[:, -remaining_dims:] = np.random.randn(n_samples, remaining_dims) * 0.1
        
        return features
    
    def generate_dataset(self) -> Tuple:
        """
        Generate complete training and test datasets.
        
        Returns:
            X_train, y_train, X_test, y_test
        """
        # Generate training data (seen classes)
        X_train_list = []
        y_train_list = []
        
        for class_name in self.seen_classes:
            features = self.generate_features_from_attributes(
                self.class_attributes[class_name],
                self.samples_per_class
            )
            X_train_list.append(features)
            y_train_list.extend([class_name] * self.samples_per_class)
        
        X_train = np.vstack(X_train_list)
        y_train = np.array(y_train_list)
        
        # Generate test data (unseen classes)
        X_test_list = []
        y_test_list = []
        
        for class_name in self.unseen_classes:
            features = self.generate_features_from_attributes(
                self.class_attributes[class_name],
                self.samples_per_class // 2  # Fewer test samples
            )
            X_test_list.append(features)
            y_test_list.extend([class_name] * (self.samples_per_class // 2))
        
        X_test = np.vstack(X_test_list)
        y_test = np.array(y_test_list)
        
        return X_train, y_train, X_test, y_test


# ============================================================================
# PART 3: Direct Attribute Prediction (DAP)
# ============================================================================

class DirectAttributePrediction(nn.Module):
    """
    Direct Attribute Prediction (DAP) model.
    
    Architecture:
    - Shared feature extractor (MLP)
    - Separate binary classifier for each attribute
    - At test time: match predicted attributes to class signatures
    
    Loss: Binary cross-entropy for each attribute independently
    """
    
    def __init__(self, n_features: int, n_attributes: int, hidden_dim: int = 256):
        """
        Initialize DAP model.
        
        Args:
            n_features: Input feature dimensionality
            n_attributes: Number of attributes to predict
            hidden_dim: Hidden layer size
        """
        super().__init__()
        
        self.n_attributes = n_attributes
        
        # Shared feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Separate classifier for each attribute
        # This models independence assumption: P(a|x) = ∏_m P(a_m|x)
        self.attribute_classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(n_attributes)
        ])
    
    def forward(self, x):
        """
        Forward pass: predict attribute probabilities.
        
        Args:
            x: Input features (batch_size, n_features)
        
        Returns:
            Attribute probabilities (batch_size, n_attributes)
        """
        # Extract shared features
        features = self.feature_extractor(x)
        
        # Predict each attribute independently
        attr_logits = []
        for classifier in self.attribute_classifiers:
            logits = classifier(features)  # (batch_size, 1)
            attr_logits.append(logits)
        
        # Stack predictions
        attr_logits = torch.cat(attr_logits, dim=1)  # (batch_size, n_attributes)
        
        # Apply sigmoid to get probabilities
        attr_probs = torch.sigmoid(attr_logits)
        
        return attr_probs
    
    def predict_class(self, x, class_attributes_dict, candidate_classes):
        """
        Predict class labels using attribute matching.
        
        Args:
            x: Input features (batch_size, n_features)
            class_attributes_dict: Dictionary mapping class names to attribute vectors
            candidate_classes: List of classes to consider
        
        Returns:
            Predicted class indices
        """
        # Get attribute predictions
        attr_probs = self.forward(x)  # (batch_size, n_attributes)
        
        # Get attribute signatures for candidate classes
        class_attrs = torch.tensor(
            np.array([class_attributes_dict[c] for c in candidate_classes]),
            dtype=torch.float32
        )  # (n_classes, n_attributes)
        
        # Compute log-likelihood for each class
        # log P(c|x) = ∑_m [a_c^m * log P(a_m|x) + (1-a_c^m) * log(1-P(a_m|x))]
        
        # Clip probabilities for numerical stability
        attr_probs_clipped = torch.clamp(attr_probs, 1e-7, 1 - 1e-7)
        
        # Compute log probabilities
        log_probs = torch.log(attr_probs_clipped)  # (batch_size, n_attributes)
        log_one_minus_probs = torch.log(1 - attr_probs_clipped)
        
        # For each sample and class, compute log-likelihood
        # Broadcasting: (batch, 1, attrs) with (1, classes, attrs)
        log_probs_expanded = log_probs.unsqueeze(1)  # (batch, 1, attrs)
        log_one_minus_expanded = log_one_minus_probs.unsqueeze(1)
        class_attrs_expanded = class_attrs.unsqueeze(0)  # (1, classes, attrs)
        
        # Likelihood computation
        log_likelihood = (
            class_attrs_expanded * log_probs_expanded +
            (1 - class_attrs_expanded) * log_one_minus_expanded
        ).sum(dim=2)  # (batch, classes)
        
        # Predict class with highest likelihood
        predicted_indices = torch.argmax(log_likelihood, dim=1)
        
        return predicted_indices


# ============================================================================
# PART 4: Indirect Attribute Prediction (IAP)
# ============================================================================

class IndirectAttributePrediction(nn.Module):
    """
    Indirect Attribute Prediction (IAP) model.
    
    Architecture:
    - Train a standard classifier on seen classes
    - Transfer to unseen classes via attribute similarity
    
    Prediction:
    P(c_unseen | x) ∝ ∑_{c ∈ seen} P(c_seen | x) * similarity(a_c_seen, a_c_unseen)
    """
    
    def __init__(self, n_features: int, n_seen_classes: int, hidden_dim: int = 256):
        """
        Initialize IAP model.
        
        Args:
            n_features: Input feature dimensionality
            n_seen_classes: Number of seen classes during training
            hidden_dim: Hidden layer size
        """
        super().__init__()
        
        self.n_seen_classes = n_seen_classes
        
        # Standard classifier for seen classes
        self.classifier = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_seen_classes)
        )
    
    def forward(self, x):
        """
        Forward pass: predict seen class probabilities.
        
        Args:
            x: Input features (batch_size, n_features)
        
        Returns:
            Class logits (batch_size, n_seen_classes)
        """
        return self.classifier(x)
    
    def predict_class(self, x, seen_class_names, unseen_class_names,
                     class_attributes_dict):
        """
        Predict unseen classes using attribute-based transfer.
        
        Args:
            x: Input features
            seen_class_names: List of seen class names (in order of training)
            unseen_class_names: List of unseen class names to consider
            class_attributes_dict: Dictionary of class attributes
        
        Returns:
            Predicted class indices in unseen_class_names
        """
        # Get seen class probabilities
        seen_logits = self.forward(x)
        seen_probs = F.softmax(seen_logits, dim=1)  # (batch, n_seen)
        
        # Get attribute vectors
        seen_attrs = torch.tensor(
            np.array([class_attributes_dict[c] for c in seen_class_names]),
            dtype=torch.float32
        )  # (n_seen, n_attributes)
        
        unseen_attrs = torch.tensor(
            np.array([class_attributes_dict[c] for c in unseen_class_names]),
            dtype=torch.float32
        )  # (n_unseen, n_attributes)
        
        # Compute attribute similarity between seen and unseen classes
        # Use cosine similarity
        seen_attrs_norm = F.normalize(seen_attrs, p=2, dim=1)
        unseen_attrs_norm = F.normalize(unseen_attrs, p=2, dim=1)
        
        # Similarity matrix: (n_seen, n_unseen)
        similarity = seen_attrs_norm @ unseen_attrs_norm.T
        
        # Transfer probabilities to unseen classes
        # P(unseen_c | x) = ∑_{seen_c} P(seen_c | x) * similarity(seen_c, unseen_c)
        # Shape: (batch, n_seen) @ (n_seen, n_unseen) = (batch, n_unseen)
        unseen_probs = seen_probs @ similarity
        
        # Predict unseen class with highest probability
        predicted_indices = torch.argmax(unseen_probs, dim=1)
        
        return predicted_indices


# ============================================================================
# PART 5: Training and Evaluation
# ============================================================================

def train_dap_model(model, X_train, y_train, class_attributes_dict,
                   epochs=50, batch_size=32, lr=0.001):
    """
    Train the Direct Attribute Prediction model.
    
    Args:
        model: DAP model instance
        X_train: Training features
        y_train: Training labels (class names)
        class_attributes_dict: Dictionary of class attributes
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
    
    Returns:
        Training loss history
    """
    print("\n" + "="*70)
    print("TRAINING DIRECT ATTRIBUTE PREDICTION (DAP)")
    print("="*70)
    
    # Convert to tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    
    # Convert labels to attribute vectors
    y_attributes = np.array([class_attributes_dict[label] for label in y_train])
    y_tensor = torch.tensor(y_attributes, dtype=torch.float32)
    
    # Create data loader
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Loss function: binary cross-entropy for each attribute
    criterion = nn.BCELoss()
    
    # Training loop
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_x, batch_y_attr in dataloader:
            # Forward pass
            pred_attrs = model(batch_x)
            
            # Compute loss (averaged over attributes and batch)
            loss = criterion(pred_attrs, batch_y_attr)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f}")
    
    return loss_history


def train_iap_model(model, X_train, y_train, seen_class_names,
                   epochs=50, batch_size=32, lr=0.001):
    """
    Train the Indirect Attribute Prediction model.
    
    Args:
        model: IAP model instance
        X_train: Training features
        y_train: Training labels (class names)
        seen_class_names: List of seen class names in order
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
    
    Returns:
        Training loss and accuracy history
    """
    print("\n" + "="*70)
    print("TRAINING INDIRECT ATTRIBUTE PREDICTION (IAP)")
    print("="*70)
    
    # Convert to tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    
    # Convert class names to indices
    class_to_idx = {name: idx for idx, name in enumerate(seen_class_names)}
    y_indices = np.array([class_to_idx[label] for label in y_train])
    y_tensor = torch.tensor(y_indices, dtype=torch.long)
    
    # Create data loader
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    loss_history = []
    acc_history = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in dataloader:
            # Forward pass
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)
        
        avg_loss = epoch_loss / len(dataloader)
        accuracy = correct / total
        
        loss_history.append(avg_loss)
        acc_history.append(accuracy)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}")
    
    return loss_history, acc_history


def evaluate_zsl_model(model, X_test, y_test, class_attributes_dict,
                      candidate_classes, model_name="Model"):
    """
    Evaluate zero-shot learning model on unseen classes.
    
    Args:
        model: Trained ZSL model (DAP or IAP)
        X_test: Test features
        y_test: True test labels
        class_attributes_dict: Dictionary of class attributes
        candidate_classes: List of unseen classes to consider
        model_name: Name for printing
    
    Returns:
        Overall accuracy and per-class accuracies
    """
    print(f"\n{'='*70}")
    print(f"EVALUATING {model_name}")
    print(f"{'='*70}")
    
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        
        # Get predictions
        if isinstance(model, DirectAttributePrediction):
            pred_indices = model.predict_class(X_tensor, class_attributes_dict, candidate_classes)
        else:  # IAP model
            # Need seen class names for IAP
            seen_classes = [c for c in class_attributes_dict.keys() 
                          if c not in candidate_classes]
            pred_indices = model.predict_class(X_tensor, seen_classes, candidate_classes,
                                              class_attributes_dict)
        
        predictions = [candidate_classes[idx] for idx in pred_indices.numpy()]
    
    # Overall accuracy
    accuracy = np.mean(np.array(predictions) == y_test)
    print(f"\nOverall zero-shot accuracy: {accuracy*100:.2f}%")
    
    # Per-class analysis
    print(f"\nPer-class results:")
    per_class_acc = {}
    
    for class_name in candidate_classes:
        class_mask = y_test == class_name
        if class_mask.sum() > 0:
            class_preds = np.array(predictions)[class_mask]
            class_acc = np.mean(class_preds == class_name)
            per_class_acc[class_name] = class_acc
            print(f"  {class_name:12s}: {class_acc*100:5.2f}%")
    
    return accuracy, per_class_acc


# ============================================================================
# PART 6: Comparison and Visualization
# ============================================================================

def compare_dap_vs_iap():
    """
    Compare Direct and Indirect Attribute Prediction methods.
    """
    print("\n" + "="*70)
    print("DAP vs IAP COMPARISON")
    print("="*70)
    
    # Generate dataset
    dataset = EnhancedAnimalDataset(n_features=100, samples_per_class=150)
    X_train, y_train, X_test, y_test = dataset.generate_dataset()
    
    print(f"\nDataset statistics:")
    print(f"  Training: {X_train.shape[0]} samples, {len(dataset.seen_classes)} classes")
    print(f"  Testing:  {X_test.shape[0]} samples, {len(dataset.unseen_classes)} classes")
    print(f"  Seen classes: {dataset.seen_classes}")
    print(f"  Unseen classes: {dataset.unseen_classes}")
    
    # Train DAP model
    dap_model = DirectAttributePrediction(
        n_features=dataset.n_features,
        n_attributes=dataset.n_attributes,
        hidden_dim=256
    )
    dap_loss = train_dap_model(dap_model, X_train, y_train, 
                               dataset.class_attributes, epochs=50)
    
    # Train IAP model
    iap_model = IndirectAttributePrediction(
        n_features=dataset.n_features,
        n_seen_classes=len(dataset.seen_classes),
        hidden_dim=256
    )
    iap_loss, iap_acc = train_iap_model(iap_model, X_train, y_train,
                                        dataset.seen_classes, epochs=50)
    
    # Evaluate both models
    dap_acc, dap_per_class = evaluate_zsl_model(
        dap_model, X_test, y_test, dataset.class_attributes,
        dataset.unseen_classes, "Direct Attribute Prediction (DAP)"
    )
    
    iap_acc, iap_per_class = evaluate_zsl_model(
        iap_model, X_test, y_test, dataset.class_attributes,
        dataset.unseen_classes, "Indirect Attribute Prediction (IAP)"
    )
    
    # Visualize results
    visualize_comparison(dap_acc, iap_acc, dap_per_class, iap_per_class,
                        dataset.unseen_classes)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nDAP Accuracy: {dap_acc*100:.2f}%")
    print(f"IAP Accuracy: {iap_acc*100:.2f}%")
    print(f"\nBetter method: {'DAP' if dap_acc > iap_acc else 'IAP'}")
    print(f"Difference: {abs(dap_acc - iap_acc)*100:.2f} percentage points")


def visualize_comparison(dap_acc, iap_acc, dap_per_class, iap_per_class, classes):
    """Create comparison visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall accuracy comparison
    methods = ['DAP', 'IAP']
    accuracies = [dap_acc * 100, iap_acc * 100]
    
    axes[0].bar(methods, accuracies, color=['#3498db', '#e74c3c'], alpha=0.8)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Overall Zero-Shot Accuracy')
    axes[0].set_ylim([0, 100])
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # Per-class comparison
    x = np.arange(len(classes))
    width = 0.35
    
    dap_scores = [dap_per_class[c] * 100 for c in classes]
    iap_scores = [iap_per_class[c] * 100 for c in classes]
    
    axes[1].bar(x - width/2, dap_scores, width, label='DAP', 
                color='#3498db', alpha=0.8)
    axes[1].bar(x + width/2, iap_scores, width, label='IAP',
                color='#e74c3c', alpha=0.8)
    
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Per-Class Zero-Shot Accuracy')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(classes, rotation=45, ha='right')
    axes[1].legend()
    axes[1].set_ylim([0, 100])
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/dap_vs_iap_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n[VISUALIZATION]")
    print(f"Saved comparison to: dap_vs_iap_comparison.png")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function."""
    print("\n" + "="*70)
    print("MODULE 56: ZERO-SHOT LEARNING - INTERMEDIATE (ATTRIBUTE-BASED)")
    print("="*70)
    
    # Run comparison
    compare_dap_vs_iap()
    
    # Final summary
    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
    Direct Attribute Prediction (DAP):
    ✓ Models attributes independently: P(a|x) = ∏_m P(a_m|x)
    ✓ Direct prediction of attribute presence
    ✓ Works well when attributes are truly independent
    ✗ Independence assumption may be too strong
    ✗ Error propagation from attribute classifiers
    
    Indirect Attribute Prediction (IAP):
    ✓ Learns on seen classes directly (no independence assumption)
    ✓ Transfers via attribute similarity
    ✓ Can leverage correlations in seen class distributions
    ✗ Requires good attribute similarity measure
    ✗ Quality depends on seen-unseen attribute overlap
    
    When to use which:
    - DAP: When attributes are well-defined and independent
    - IAP: When seen classes have good attribute overlap with unseen
    - Both: Often beneficial to ensemble predictions
    
    Next steps:
    - Semantic embeddings (word vectors) instead of discrete attributes
    - Visual-semantic embedding spaces
    - Generalized zero-shot learning (seen + unseen at test time)
    """)
    print("="*70)


if __name__ == "__main__":
    main()
