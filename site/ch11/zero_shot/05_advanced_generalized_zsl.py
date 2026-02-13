"""
Module 56: Zero-Shot Learning - Advanced Level (Part 2)
=========================================================

Generalized Zero-Shot Learning (GZSL)

This script implements generalized zero-shot learning where test data may
contain both seen and unseen classes. We'll explore:
1. The generalized ZSL problem and bias toward seen classes
2. Calibration techniques
3. Gating mechanisms
4. Feature generation approaches
5. Comprehensive evaluation metrics

Author: Deep Learning Curriculum
Level: Advanced
Estimated Time: 90-120 minutes
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)


# ============================================================================
# PART 1: Generalized Zero-Shot Learning Problem
# ============================================================================

"""
CONVENTIONAL ZSL vs GENERALIZED ZSL
===================================

Conventional ZSL:
-----------------
- Training: Learn on seen classes S
- Testing: Predict only on unseen classes U
- Test set contains ONLY unseen class instances
- Assumes we know test instances are from unseen classes

Generalized ZSL (GZSL):
----------------------
- Training: Learn on seen classes S
- Testing: Predict on BOTH seen and unseen classes S ∪ U
- Test set may contain seen OR unseen class instances
- More realistic: we don't know in advance if instance is seen/unseen

Challenge: BIAS TOWARD SEEN CLASSES
-----------------------------------
Models trained on seen classes tend to:
1. Assign higher confidence to seen classes
2. Rarely predict unseen classes even when correct
3. Achieve high accuracy on seen but low on unseen

Example:
- Seen accuracy: 85%
- Unseen accuracy: 15%
- Model almost always predicts seen classes!


EVALUATION METRICS FOR GZSL
===========================

1. **Per-domain Accuracy**:
   - acc_s: Accuracy on seen classes
   - acc_u: Accuracy on unseen classes

2. **Harmonic Mean** (primary metric):
   H = 2 * (acc_s * acc_u) / (acc_s + acc_u)
   
   Harmonic mean penalizes models that do well on only one domain.
   A model predicting all seen (acc_s=1.0, acc_u=0.0) gets H=0.0

3. **Area Under Seen-Unseen Curve (AUSUC)**:
   Plot acc_s vs acc_u for different calibration values
   Compute area under curve

4. **Overall Accuracy**:
   Combined accuracy on all test samples
   Less informative than harmonic mean


CALIBRATION TECHNIQUES
=====================

Goal: Reduce bias toward seen classes

1. **Threshold Calibration**:
   Add constant γ to unseen class scores:
   score_unseen = score_unseen + γ
   
   Cross-validate γ to balance seen/unseen accuracy

2. **Temperature Scaling**:
   Divide seen class scores by temperature T > 1:
   score_seen = score_seen / T
   
   Makes seen class probabilities more uniform

3. **Learned Gating**:
   Train a gate network g(x) to predict if x is seen or unseen:
   - If g(x) > 0.5: Predict from seen classes
   - If g(x) ≤ 0.5: Predict from unseen classes

4. **Separate Classifiers**:
   Train separate models for seen and unseen
   Combine predictions with weighting


GENERATIVE APPROACHES
====================

Synthesize visual features for unseen classes:

1. **Conditional VAE/GAN**:
   G(s_c, z) → v_c  (Generate visual features from semantic + noise)
   
   Train on seen classes, generate for unseen
   Then train classifier on real + synthetic data

2. **Cycle Consistency**:
   Learn bidirectional mappings:
   V → S → V (visual to semantic back to visual)
   S → V → S (semantic to visual back to semantic)

3. **Advantages**:
   - Converts ZSL to supervised learning
   - Can handle GZSL naturally
   - Reduces bias by balancing training data
"""


# ============================================================================
# PART 2: Dataset for GZSL
# ============================================================================

class GZSLDataset(Dataset):
    """
    Dataset for Generalized Zero-Shot Learning.
    
    Contains both seen and unseen classes in test set.
    """
    
    def __init__(self, class_names: List[str], semantic_embeddings: Dict,
                 samples_per_class: int = 150, feature_dim: int = 1024,
                 is_training: bool = True):
        """
        Initialize GZSL dataset.
        
        Args:
            class_names: List of class names
            semantic_embeddings: Dictionary of semantic embeddings
            samples_per_class: Number of samples per class
            feature_dim: Feature dimensionality
            is_training: Whether this is training data
        """
        self.class_names = class_names
        self.semantic_embeddings = semantic_embeddings
        self.samples_per_class = samples_per_class
        self.feature_dim = feature_dim
        self.is_training = is_training
        
        # Generate data
        self.X, self.y, self.y_semantic = self._generate_data()
    
    def _generate_data(self):
        """Generate visual features."""
        X_list = []
        y_list = []
        y_semantic_list = []
        
        # Fixed projection matrix
        np.random.seed(42)
        semantic_dim = len(next(iter(self.semantic_embeddings.values())))
        projection = np.random.randn(semantic_dim, self.feature_dim) * 0.4
        
        for class_name in self.class_names:
            sem_emb = self.semantic_embeddings[class_name]
            
            # Project to visual space
            base_features = sem_emb @ projection
            
            # Generate samples
            features = np.tile(base_features, (self.samples_per_class, 1))
            
            # Add variation
            noise_scale = 0.5 if self.is_training else 0.6  # Test data slightly different
            features += np.random.randn(self.samples_per_class, self.feature_dim) * noise_scale
            
            X_list.append(features)
            y_list.extend([class_name] * self.samples_per_class)
            y_semantic_list.extend([sem_emb] * self.samples_per_class)
        
        X = np.vstack(X_list).astype(np.float32)
        y = np.array(y_list)
        y_semantic = np.vstack(y_semantic_list).astype(np.float32)
        
        return X, y, y_semantic
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            self.y[idx],
            torch.from_numpy(self.y_semantic[idx])
        )


# ============================================================================
# PART 3: GZSL Model with Calibration
# ============================================================================

class GZSL_Model(nn.Module):
    """
    Generalized Zero-Shot Learning model with calibration.
    
    Includes:
    - Visual-semantic compatibility
    - Calibration mechanism
    - Optional gating network
    """
    
    def __init__(self, visual_dim: int, semantic_dim: int,
                 embedding_dim: int = 256, use_gate: bool = False):
        """
        Initialize GZSL model.
        
        Args:
            visual_dim: Visual feature dimensionality
            semantic_dim: Semantic embedding dimensionality
            embedding_dim: Joint embedding space dimensionality
            use_gate: Whether to use gating mechanism
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.use_gate = use_gate
        
        # Visual encoder
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, embedding_dim)
        )
        
        # Semantic encoder
        self.semantic_encoder = nn.Sequential(
            nn.Linear(semantic_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
        
        # Optional gating network (predicts if sample is from seen/unseen)
        if use_gate:
            self.gate = nn.Sequential(
                nn.Linear(visual_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
        
        # Learnable calibration parameter
        self.calibration_bias = nn.Parameter(torch.tensor(0.0))
    
    def encode_visual(self, x):
        """Encode visual features."""
        return self.visual_encoder(x)
    
    def encode_semantic(self, s):
        """Encode semantic embeddings."""
        return self.semantic_encoder(s)
    
    def compute_gate(self, x):
        """
        Compute gating probability (probability of being seen class).
        
        Returns:
            Probability that x belongs to a seen class
        """
        if self.use_gate:
            return self.gate(x)
        return None
    
    def forward(self, x, s):
        """Compute compatibility."""
        v_proj = self.encode_visual(x)
        s_proj = self.encode_semantic(s)
        
        # Normalize
        v_norm = F.normalize(v_proj, p=2, dim=1)
        s_norm = F.normalize(s_proj, p=2, dim=1)
        
        # Cosine similarity
        return (v_norm * s_norm).sum(dim=1)
    
    def predict_gzsl(self, x, seen_embeddings, unseen_embeddings,
                    seen_names, unseen_names, calibration=0.0,
                    use_learned_calibration=False):
        """
        Predict classes for GZSL (both seen and unseen possible).
        
        Args:
            x: Visual features
            seen_embeddings: Semantic embeddings for seen classes
            unseen_embeddings: Semantic embeddings for unseen classes
            seen_names: Seen class names
            unseen_names: Unseen class names
            calibration: Manual calibration bias for unseen classes
            use_learned_calibration: Use learned calibration parameter
        
        Returns:
            Predicted classes, seen scores, unseen scores
        """
        self.eval()
        with torch.no_grad():
            # Encode visual
            v_proj = self.encode_visual(x)
            v_norm = F.normalize(v_proj, p=2, dim=1)
            
            # Encode seen and unseen semantics
            seen_proj = self.encode_semantic(seen_embeddings)
            unseen_proj = self.encode_semantic(unseen_embeddings)
            
            seen_norm = F.normalize(seen_proj, p=2, dim=1)
            unseen_norm = F.normalize(unseen_proj, p=2, dim=1)
            
            # Compute scores
            seen_scores = v_norm @ seen_norm.T  # (batch, n_seen)
            unseen_scores = v_norm @ unseen_norm.T  # (batch, n_unseen)
            
            # Apply calibration to unseen scores
            if use_learned_calibration:
                unseen_scores = unseen_scores + self.calibration_bias
            elif calibration != 0.0:
                unseen_scores = unseen_scores + calibration
            
            # Optionally use gate
            if self.use_gate:
                gate_probs = self.compute_gate(x)  # (batch, 1)
                # Weight scores by gate probabilities
                seen_scores = seen_scores * gate_probs
                unseen_scores = unseen_scores * (1 - gate_probs)
            
            # Combine scores
            all_scores = torch.cat([seen_scores, unseen_scores], dim=1)
            all_names = seen_names + unseen_names
            
            # Predict
            pred_indices = torch.argmax(all_scores, dim=1)
        
        return pred_indices, seen_scores, unseen_scores


# ============================================================================
# PART 4: Training for GZSL
# ============================================================================

def train_gzsl_model(model, train_loader, semantic_embeddings,
                    seen_classes, epochs=50, lr=0.001):
    """
    Train GZSL model.
    
    Args:
        model: GZSL model
        train_loader: Training data loader (seen classes only)
        semantic_embeddings: Dictionary of semantic embeddings
        seen_classes: List of seen class names
        epochs: Number of epochs
        lr: Learning rate
    
    Returns:
        Loss history
    """
    print("\n" + "="*70)
    print("TRAINING GZSL MODEL")
    print("="*70)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_x, batch_y, batch_s_pos in train_loader:
            batch_size = batch_x.size(0)
            
            # Positive scores
            pos_scores = model(batch_x, batch_s_pos)
            
            # Negative scores (sample from seen classes)
            neg_classes = [np.random.choice([c for c in seen_classes if c != y])
                          for y in batch_y]
            batch_s_neg = torch.stack([
                torch.from_numpy(semantic_embeddings[c]) for c in neg_classes
            ])
            neg_scores = model(batch_x, batch_s_neg)
            
            # Ranking loss
            margin = 0.1
            loss = torch.clamp(margin + neg_scores - pos_scores, min=0).mean()
            
            # If using gate, add auxiliary loss
            if model.use_gate:
                # All training samples are from seen classes, so gate should be ~1
                gate_probs = model.compute_gate(batch_x)
                gate_loss = F.binary_cross_entropy(gate_probs, torch.ones_like(gate_probs))
                loss = loss + 0.1 * gate_loss
            
            # Backward
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


# ============================================================================
# PART 5: GZSL Evaluation
# ============================================================================

def evaluate_gzsl(model, test_dataset, semantic_embeddings,
                 seen_classes, unseen_classes, calibration_values=None):
    """
    Comprehensive evaluation for GZSL.
    
    Args:
        model: Trained GZSL model
        test_dataset: Test dataset (contains both seen and unseen)
        semantic_embeddings: Dictionary of semantic embeddings
        seen_classes: List of seen class names
        unseen_classes: List of unseen class names
        calibration_values: List of calibration values to try
    
    Returns:
        Results dictionary
    """
    print("\n" + "="*70)
    print("EVALUATING GENERALIZED ZERO-SHOT LEARNING")
    print("="*70)
    
    if calibration_values is None:
        calibration_values = [0.0, 0.1, 0.2, 0.3, 0.5]
    
    # Prepare data
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    # Get embeddings
    seen_embeddings = torch.stack([
        torch.from_numpy(semantic_embeddings[c]) for c in seen_classes
    ])
    unseen_embeddings = torch.stack([
        torch.from_numpy(semantic_embeddings[c]) for c in unseen_classes
    ])
    
    # Separate test data by seen/unseen
    all_true = []
    all_x = []
    
    for batch_x, batch_y, _ in test_loader:
        all_x.append(batch_x)
        all_true.extend(batch_y)
    
    all_x = torch.cat(all_x, dim=0)
    all_true = np.array(all_true)
    
    # Evaluate for different calibration values
    results = {
        'calibration': [],
        'acc_seen': [],
        'acc_unseen': [],
        'harmonic_mean': [],
        'overall_acc': []
    }
    
    best_harmonic = 0.0
    best_calibration = 0.0
    
    for calib in calibration_values:
        pred_indices, _, _ = model.predict_gzsl(
            all_x, seen_embeddings, unseen_embeddings,
            seen_classes, unseen_classes, calibration=calib
        )
        
        all_names = seen_classes + unseen_classes
        predictions = np.array([all_names[idx] for idx in pred_indices.numpy()])
        
        # Compute metrics
        seen_mask = np.isin(all_true, seen_classes)
        unseen_mask = np.isin(all_true, unseen_classes)
        
        acc_seen = np.mean(predictions[seen_mask] == all_true[seen_mask])
        acc_unseen = np.mean(predictions[unseen_mask] == all_true[unseen_mask])
        
        # Harmonic mean
        if acc_seen + acc_unseen > 0:
            harmonic = 2 * (acc_seen * acc_unseen) / (acc_seen + acc_unseen)
        else:
            harmonic = 0.0
        
        # Overall accuracy
        overall = np.mean(predictions == all_true)
        
        results['calibration'].append(calib)
        results['acc_seen'].append(acc_seen * 100)
        results['acc_unseen'].append(acc_unseen * 100)
        results['harmonic_mean'].append(harmonic * 100)
        results['overall_acc'].append(overall * 100)
        
        if harmonic > best_harmonic:
            best_harmonic = harmonic
            best_calibration = calib
        
        print(f"\nCalibration: {calib:.2f}")
        print(f"  Seen accuracy:    {acc_seen*100:5.2f}%")
        print(f"  Unseen accuracy:  {acc_unseen*100:5.2f}%")
        print(f"  Harmonic mean:    {harmonic*100:5.2f}%")
        print(f"  Overall accuracy: {overall*100:5.2f}%")
    
    print(f"\n{'='*70}")
    print(f"BEST CALIBRATION: {best_calibration:.2f}")
    print(f"Best Harmonic Mean: {best_harmonic*100:.2f}%")
    print(f"{'='*70}")
    
    return results, best_calibration


def visualize_gzsl_results(results, seen_classes, unseen_classes):
    """Visualize GZSL evaluation results."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Accuracy vs Calibration
    calibrations = results['calibration']
    
    axes[0].plot(calibrations, results['acc_seen'], 'o-', 
                label='Seen Classes', color='#3498db', linewidth=2)
    axes[0].plot(calibrations, results['acc_unseen'], 's-',
                label='Unseen Classes', color='#e74c3c', linewidth=2)
    axes[0].plot(calibrations, results['harmonic_mean'], '^-',
                label='Harmonic Mean', color='#2ecc71', linewidth=2)
    
    axes[0].set_xlabel('Calibration Value')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('GZSL Performance vs Calibration')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot 2: Seen vs Unseen Accuracy Tradeoff
    axes[1].plot(results['acc_unseen'], results['acc_seen'], 'o-',
                color='#9b59b6', linewidth=2, markersize=8)
    
    # Annotate calibration values
    for i, calib in enumerate(calibrations):
        axes[1].annotate(f'{calib:.1f}',
                        (results['acc_unseen'][i], results['acc_seen'][i]),
                        textcoords="offset points", xytext=(5,5), ha='left')
    
    axes[1].set_xlabel('Unseen Class Accuracy (%)')
    axes[1].set_ylabel('Seen Class Accuracy (%)')
    axes[1].set_title('Seen vs Unseen Accuracy Tradeoff')
    axes[1].grid(alpha=0.3)
    
    # Add diagonal line (equal accuracy)
    lims = [0, 100]
    axes[1].plot(lims, lims, 'k--', alpha=0.3, label='Equal Accuracy')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('/home/claude/gzsl_evaluation.png', dpi=150, bbox_inches='tight')
    print(f"\n[VISUALIZATION]")
    print(f"Saved GZSL evaluation to: gzsl_evaluation.png")
    plt.close()


# ============================================================================
# PART 6: Complete GZSL Pipeline
# ============================================================================

def run_gzsl_pipeline():
    """Run complete GZSL pipeline."""
    print("\n" + "="*70)
    print("GENERALIZED ZERO-SHOT LEARNING PIPELINE")
    print("="*70)
    
    # Setup
    seen_classes = ['dog', 'cat', 'bird', 'fish', 'snake']
    unseen_classes = ['horse', 'zebra', 'elephant', 'tiger', 'eagle']
    all_classes = seen_classes + unseen_classes
    
    # Create semantic embeddings
    semantic_dim = 50
    np.random.seed(42)
    semantic_embeddings = {
        c: np.random.randn(semantic_dim).astype(np.float32) 
        for c in all_classes
    }
    for c in all_classes:
        semantic_embeddings[c] /= (np.linalg.norm(semantic_embeddings[c]) + 1e-8)
    
    # Generate training data (seen classes only)
    print("\n[Creating Training Dataset]")
    train_dataset = GZSLDataset(
        seen_classes, semantic_embeddings,
        samples_per_class=200, is_training=True
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print(f"Training: {len(train_dataset)} samples, {len(seen_classes)} classes")
    
    # Generate test data (BOTH seen and unseen)
    print("\n[Creating Test Dataset]")
    test_dataset = GZSLDataset(
        all_classes, semantic_embeddings,
        samples_per_class=100, is_training=False
    )
    print(f"Testing: {len(test_dataset)} samples, {len(all_classes)} classes")
    print(f"  Seen classes: {seen_classes}")
    print(f"  Unseen classes: {unseen_classes}")
    
    # Train model without gate
    print("\n" + "="*70)
    print("MODEL 1: GZSL WITHOUT GATING")
    print("="*70)
    
    model_no_gate = GZSL_Model(
        visual_dim=1024,
        semantic_dim=semantic_dim,
        embedding_dim=256,
        use_gate=False
    )
    
    train_gzsl_model(model_no_gate, train_loader, semantic_embeddings,
                    seen_classes, epochs=50)
    
    results_no_gate, best_calib_no_gate = evaluate_gzsl(
        model_no_gate, test_dataset, semantic_embeddings,
        seen_classes, unseen_classes
    )
    
    # Train model with gate
    print("\n" + "="*70)
    print("MODEL 2: GZSL WITH GATING MECHANISM")
    print("="*70)
    
    model_with_gate = GZSL_Model(
        visual_dim=1024,
        semantic_dim=semantic_dim,
        embedding_dim=256,
        use_gate=True
    )
    
    train_gzsl_model(model_with_gate, train_loader, semantic_embeddings,
                    seen_classes, epochs=50)
    
    results_with_gate, best_calib_with_gate = evaluate_gzsl(
        model_with_gate, test_dataset, semantic_embeddings,
        seen_classes, unseen_classes
    )
    
    # Visualize
    visualize_gzsl_results(results_no_gate, seen_classes, unseen_classes)
    
    # Compare models
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    idx_best_no_gate = results_no_gate['calibration'].index(best_calib_no_gate)
    idx_best_with_gate = results_with_gate['calibration'].index(best_calib_with_gate)
    
    print(f"\nWithout Gating (calib={best_calib_no_gate}):")
    print(f"  Seen:     {results_no_gate['acc_seen'][idx_best_no_gate]:.2f}%")
    print(f"  Unseen:   {results_no_gate['acc_unseen'][idx_best_no_gate]:.2f}%")
    print(f"  Harmonic: {results_no_gate['harmonic_mean'][idx_best_no_gate]:.2f}%")
    
    print(f"\nWith Gating (calib={best_calib_with_gate}):")
    print(f"  Seen:     {results_with_gate['acc_seen'][idx_best_with_gate]:.2f}%")
    print(f"  Unseen:   {results_with_gate['acc_unseen'][idx_best_with_gate]:.2f}%")
    print(f"  Harmonic: {results_with_gate['harmonic_mean'][idx_best_with_gate]:.2f}%")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function."""
    print("\n" + "="*70)
    print("MODULE 56: GENERALIZED ZERO-SHOT LEARNING")
    print("="*70)
    
    run_gzsl_pipeline()
    
    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
    Generalized Zero-Shot Learning (GZSL):
    ✓ More realistic than conventional ZSL
    ✓ Test set contains BOTH seen and unseen classes
    ✓ Major challenge: bias toward seen classes
    ✗ Much more difficult than conventional ZSL
    
    Evaluation Metrics:
    - Seen accuracy (acc_s): Accuracy on seen classes
    - Unseen accuracy (acc_u): Accuracy on unseen classes
    - Harmonic mean: H = 2*(acc_s * acc_u)/(acc_s + acc_u)
    - Harmonic mean is the PRIMARY metric (penalizes imbalance)
    
    Calibration Techniques:
    1. Threshold calibration: Add bias to unseen scores
    2. Temperature scaling: Divide seen scores by T > 1
    3. Gating mechanism: Predict if sample is seen/unseen
    4. Separate classifiers: Train different models
    
    Best Practices:
    1. Always report harmonic mean, not just overall accuracy
    2. Cross-validate calibration parameter
    3. Plot seen vs unseen accuracy tradeoff
    4. Consider generative approaches for severe imbalance
    5. Use proper train/val/test splits for hyperparameter tuning
    
    Advanced Techniques:
    - Feature generation with VAE/GAN
    - Transductive learning (use unlabeled test data)
    - Self-training and pseudo-labeling
    - Ensemble methods
    - Attribute-based calibration
    
    Real-World Applications:
    - Rare disease diagnosis (few/no training examples)
    - New product categorization (e-commerce)
    - Wildlife species recognition (rare species)
    - Fine-grained classification (sub-categories)
    
    Limitations:
    - Requires good semantic embeddings
    - Sensitive to domain shift
    - Hubness problem in high dimensions
    - Difficult to calibrate without validation data
    """)
    print("="*70)


if __name__ == "__main__":
    main()
