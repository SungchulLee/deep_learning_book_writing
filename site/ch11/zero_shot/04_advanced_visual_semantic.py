"""
Module 56: Zero-Shot Learning - Advanced Level (Part 1)
=========================================================

Deep Visual-Semantic Embedding (DeViSE)

This script implements advanced visual-semantic embedding approaches:
1. DeViSE (Deep Visual-Semantic Embedding Model)
2. Bilinear compatibility models
3. Multiple loss functions (ranking, cross-entropy, triplet)
4. Comprehensive training pipeline with real CNN features

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
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)


# ============================================================================
# PART 1: Mathematical Foundations
# ============================================================================

"""
DeVISE: DEEP VISUAL-SEMANTIC EMBEDDING MODEL
============================================

Paper: Frome et al. (2013) - "DeViSE: A Deep Visual-Semantic Embedding Model"

Core Idea:
----------
Learn a transformation M that maps visual features (from CNN) to the semantic
embedding space (word2vec), such that:

    M · v(image) ≈ s(class_name)

where:
- v(image): Visual features from pre-trained CNN (e.g., fc7 layer of VGG)
- s(class_name): Word embedding for the class name
- M: Learned linear/affine transformation matrix

Loss Functions:
--------------

1. **Dot Product Similarity Loss (Original DeViSE)**:
   
   L = ∑_{(x,y)} max(0, margin + M·v(x)·s(ȳ) - M·v(x)·s(y))
   
   where ȳ is a randomly sampled negative class

2. **Cosine Similarity Loss**:
   
   L = ∑_{(x,y)} max(0, margin + cos(M·v(x), s(ȳ)) - cos(M·v(x), s(y)))
   
   Cosine similarity: cos(a,b) = (a·b) / (||a|| ||b||)

3. **Cross-Entropy Loss** (alternative):
   
   Convert to classification via softmax over all classes:
   L = -log(exp(M·v(x)·s(y)) / ∑_c exp(M·v(x)·s(c)))

4. **Triplet Loss**:
   
   L = max(0, ||M·v(x) - s(y)||² - ||M·v(x) - s(ȳ)||² + margin)
   
   Encourages projected visual features to be close to positive semantic
   embedding and far from negative.


BILINEAR COMPATIBILITY MODEL
============================

Instead of linear projection, use bilinear form:

    F(v, s) = v^T W s

where W ∈ ℝ^{d_v × d_s} is a learned weight matrix.

Advantages:
- Captures interactions between visual and semantic dimensions
- More expressive than dot product after linear projection
- Can model complex relationships

Training:
- Similar loss functions as above
- More parameters to learn (d_v × d_s instead of d_v)


ADVANCED ARCHITECTURES
======================

1. **Two-branch Network**:
   - Visual branch: CNN → fully connected layers
   - Semantic branch: Word embedding → fully connected layers
   - Joint embedding space: Both branches project to same dimension
   - Compatibility: dot product or cosine similarity

2. **Cross-Modal Attention**:
   - Visual features attend to semantic embeddings
   - Learns which visual regions are relevant for each semantic concept
   - More sophisticated but requires more data

3. **Adversarial Training**:
   - Discriminator tries to distinguish visual vs semantic embeddings
   - Generator tries to make them indistinguishable
   - Encourages better alignment
"""


# ============================================================================
# PART 2: Advanced Data Generation with CNN-like Features
# ============================================================================

class AdvancedAnimalDataset(Dataset):
    """
    Advanced dataset with CNN-like visual features.
    
    Simulates features from a pre-trained CNN (like VGG16 fc7 layer).
    In practice, you would extract these from real CNNs.
    """
    
    def __init__(self, class_names: List[str], semantic_embeddings: Dict,
                 samples_per_class: int = 200, feature_dim: int = 4096):
        """
        Initialize dataset.
        
        Args:
            class_names: List of class names
            semantic_embeddings: Dictionary mapping class names to embeddings
            samples_per_class: Samples to generate per class
            feature_dim: Visual feature dimensionality (e.g., 4096 for VGG fc7)
        """
        self.class_names = class_names
        self.semantic_embeddings = semantic_embeddings
        self.samples_per_class = samples_per_class
        self.feature_dim = feature_dim
        
        # Generate data
        self.X, self.y, self.y_semantic = self._generate_data()
    
    def _generate_data(self):
        """Generate visual features correlated with semantic embeddings."""
        X_list = []
        y_list = []
        y_semantic_list = []
        
        # Create a fixed random projection from semantic to visual space
        np.random.seed(42)
        semantic_dim = len(next(iter(self.semantic_embeddings.values())))
        projection = np.random.randn(semantic_dim, self.feature_dim) * 0.3
        
        for class_name in self.class_names:
            sem_emb = self.semantic_embeddings[class_name]
            
            # Project semantic to visual space (simulating CNN features)
            base_visual = sem_emb @ projection
            
            # Add intra-class variation
            features = np.tile(base_visual, (self.samples_per_class, 1))
            features += np.random.randn(self.samples_per_class, self.feature_dim) * 0.5
            
            # Add some non-linear transformations (simulate CNN complexity)
            features = np.tanh(features)
            
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
# PART 3: DeViSE Model Implementation
# ============================================================================

class DeViSE(nn.Module):
    """
    Deep Visual-Semantic Embedding (DeViSE) Model.
    
    Architecture:
    - Visual encoder: Projects CNN features to joint embedding space
    - Learns to align visual features with semantic (word) embeddings
    - Uses ranking loss for training
    """
    
    def __init__(self, visual_dim: int, semantic_dim: int, 
                 embedding_dim: int = 300, dropout: float = 0.5):
        """
        Initialize DeViSE model.
        
        Args:
            visual_dim: CNN feature dimensionality
            semantic_dim: Word embedding dimensionality
            embedding_dim: Joint embedding space dimensionality
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Visual encoder: CNN features → Joint embedding space
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, embedding_dim)
        )
        
        # Optional semantic encoder (if semantic embeddings need transformation)
        # In original DeViSE, word embeddings are used directly
        self.semantic_encoder = nn.Sequential(
            nn.Linear(semantic_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
    
    def encode_visual(self, x):
        """
        Encode visual features to joint space.
        
        Args:
            x: Visual features (batch_size, visual_dim)
        
        Returns:
            Encoded features (batch_size, embedding_dim)
        """
        return self.visual_encoder(x)
    
    def encode_semantic(self, s):
        """
        Encode semantic vectors to joint space.
        
        Args:
            s: Semantic embeddings (batch_size or n_classes, semantic_dim)
        
        Returns:
            Encoded semantic vectors (batch_size or n_classes, embedding_dim)
        """
        return self.semantic_encoder(s)
    
    def forward(self, x, s):
        """
        Compute compatibility between visual and semantic.
        
        Args:
            x: Visual features (batch_size, visual_dim)
            s: Semantic embeddings (batch_size, semantic_dim)
        
        Returns:
            Compatibility scores
        """
        v_proj = self.encode_visual(x)
        s_proj = self.encode_semantic(s)
        
        # Normalize for cosine similarity
        v_norm = F.normalize(v_proj, p=2, dim=1)
        s_norm = F.normalize(s_proj, p=2, dim=1)
        
        # Cosine similarity
        return (v_norm * s_norm).sum(dim=1)
    
    def predict_class(self, x, class_embeddings, class_names):
        """
        Predict class for visual features.
        
        Args:
            x: Visual features (batch_size, visual_dim)
            class_embeddings: Semantic embeddings for all classes (n_classes, semantic_dim)
            class_names: List of class names
        
        Returns:
            Predicted class indices and scores
        """
        self.eval()
        with torch.no_grad():
            # Project visual to joint space
            v_proj = self.encode_visual(x)
            v_norm = F.normalize(v_proj, p=2, dim=1)
            
            # Project all class embeddings
            s_proj = self.encode_semantic(class_embeddings)
            s_norm = F.normalize(s_proj, p=2, dim=1)
            
            # Compute similarities with all classes
            # (batch_size, embedding_dim) @ (n_classes, embedding_dim).T
            # = (batch_size, n_classes)
            similarities = v_norm @ s_norm.T
            
            # Get top predictions
            scores, indices = torch.max(similarities, dim=1)
        
        return indices, scores


# ============================================================================
# PART 4: Bilinear Compatibility Model
# ============================================================================

class BilinearCompatibility(nn.Module):
    """
    Bilinear compatibility model: F(v, s) = v^T W s
    
    More expressive than DeViSE's dot product but has more parameters.
    """
    
    def __init__(self, visual_dim: int, semantic_dim: int,
                 hidden_dim: int = 1024, dropout: float = 0.5):
        """
        Initialize bilinear model.
        
        Args:
            visual_dim: Visual feature dimensionality
            semantic_dim: Semantic embedding dimensionality
            hidden_dim: Hidden layer size for visual/semantic encoders
            dropout: Dropout probability
        """
        super().__init__()
        
        # Visual encoder
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Semantic encoder
        self.semantic_encoder = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Bilinear layer: v^T W s
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)
    
    def forward(self, x, s):
        """
        Compute bilinear compatibility.
        
        Args:
            x: Visual features (batch_size, visual_dim)
            s: Semantic embeddings (batch_size, semantic_dim)
        
        Returns:
            Compatibility scores (batch_size, 1)
        """
        v_encoded = self.visual_encoder(x)
        s_encoded = self.semantic_encoder(s)
        
        # Bilinear compatibility
        scores = self.bilinear(v_encoded, s_encoded)
        
        return scores.squeeze(1)
    
    def predict_class(self, x, class_embeddings, class_names):
        """Predict class using bilinear compatibility."""
        self.eval()
        with torch.no_grad():
            batch_size = x.size(0)
            n_classes = class_embeddings.size(0)
            
            # Encode visual once
            v_encoded = self.visual_encoder(x)  # (batch_size, hidden_dim)
            
            # Encode all class embeddings
            s_encoded = self.semantic_encoder(class_embeddings)  # (n_classes, hidden_dim)
            
            # Compute compatibility for all pairs
            scores = torch.zeros(batch_size, n_classes)
            
            for i in range(n_classes):
                # Expand semantic encoding for this class
                s_i = s_encoded[i:i+1].expand(batch_size, -1)
                scores[:, i] = self.bilinear(v_encoded, s_i).squeeze(1)
            
            # Get top predictions
            max_scores, indices = torch.max(scores, dim=1)
        
        return indices, max_scores


# ============================================================================
# PART 5: Training with Multiple Loss Functions
# ============================================================================

def ranking_loss(pos_scores, neg_scores, margin=0.1):
    """
    Ranking loss (hinge loss).
    
    L = max(0, margin + neg_score - pos_score)
    
    Args:
        pos_scores: Compatibility with correct class
        neg_scores: Compatibility with incorrect class
        margin: Margin parameter
    
    Returns:
        Loss value
    """
    return torch.clamp(margin + neg_scores - pos_scores, min=0).mean()


def triplet_loss(anchor, positive, negative, margin=0.3):
    """
    Triplet loss for embedding learning.
    
    L = max(0, ||anchor - positive||² - ||anchor - negative||² + margin)
    
    Args:
        anchor: Projected visual features
        positive: Semantic embedding of correct class
        negative: Semantic embedding of incorrect class
        margin: Margin parameter
    
    Returns:
        Loss value
    """
    pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
    neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
    
    return torch.clamp(pos_dist - neg_dist + margin, min=0).mean()


def train_devise_model(model, train_loader, semantic_embeddings_dict,
                      class_names, epochs=50, lr=0.001, loss_type='ranking'):
    """
    Train DeViSE or Bilinear model.
    
    Args:
        model: DeViSE or Bilinear model
        train_loader: Training data loader
        semantic_embeddings_dict: Dictionary of semantic embeddings
        class_names: List of all class names
        epochs: Number of epochs
        lr: Learning rate
        loss_type: 'ranking', 'triplet', or 'cosine'
    
    Returns:
        Training loss history
    """
    print(f"\n{'='*70}")
    print(f"TRAINING {model.__class__.__name__} WITH {loss_type.upper()} LOSS")
    print(f"{'='*70}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_x, batch_y, batch_s_pos in train_loader:
            batch_size = batch_x.size(0)
            
            # Positive scores (correct class)
            if loss_type == 'triplet':
                # Need projected features for triplet loss
                v_proj = model.encode_visual(batch_x)
                s_pos_proj = model.encode_semantic(batch_s_pos)
                
                # Sample negative classes
                neg_classes = [np.random.choice([c for c in class_names if c != y])
                              for y in batch_y]
                batch_s_neg = torch.stack([
                    torch.from_numpy(semantic_embeddings_dict[c]) for c in neg_classes
                ])
                s_neg_proj = model.encode_semantic(batch_s_neg)
                
                loss = triplet_loss(v_proj, s_pos_proj, s_neg_proj)
            
            else:  # ranking or cosine
                pos_scores = model(batch_x, batch_s_pos)
                
                # Negative scores (random incorrect class)
                neg_classes = [np.random.choice([c for c in class_names if c != y])
                              for y in batch_y]
                batch_s_neg = torch.stack([
                    torch.from_numpy(semantic_embeddings_dict[c]) for c in neg_classes
                ])
                neg_scores = model(batch_x, batch_s_neg)
                
                loss = ranking_loss(pos_scores, neg_scores, margin=0.1)
            
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


def evaluate_model(model, test_dataset, semantic_embeddings_dict,
                  candidate_classes, model_name="Model"):
    """
    Evaluate zero-shot learning model.
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        semantic_embeddings_dict: Dictionary of semantic embeddings
        candidate_classes: List of candidate class names
        model_name: Name for printing
    
    Returns:
        Accuracy and per-class accuracies
    """
    print(f"\n{'='*70}")
    print(f"EVALUATING {model_name}")
    print(f"{'='*70}")
    
    # Prepare test data
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    # Get candidate class embeddings
    candidate_embeddings = torch.stack([
        torch.from_numpy(semantic_embeddings_dict[c]) for c in candidate_classes
    ])
    
    # Predict
    all_predictions = []
    all_true = []
    
    model.eval()
    for batch_x, batch_y, _ in test_loader:
        pred_indices, _ = model.predict_class(batch_x, candidate_embeddings, candidate_classes)
        predictions = [candidate_classes[idx] for idx in pred_indices.numpy()]
        
        all_predictions.extend(predictions)
        all_true.extend(batch_y)
    
    # Overall accuracy
    all_predictions = np.array(all_predictions)
    all_true = np.array(all_true)
    accuracy = np.mean(all_predictions == all_true)
    
    print(f"\nOverall accuracy: {accuracy*100:.2f}%")
    
    # Per-class accuracy
    print(f"\nPer-class results:")
    per_class_acc = {}
    
    for class_name in candidate_classes:
        class_mask = all_true == class_name
        if class_mask.sum() > 0:
            class_preds = all_predictions[class_mask]
            class_acc = np.mean(class_preds == class_name)
            per_class_acc[class_name] = class_acc
            print(f"  {class_name:12s}: {class_acc*100:5.2f}%")
    
    return accuracy, per_class_acc


# ============================================================================
# PART 6: Complete Pipeline
# ============================================================================

def run_advanced_zsl_pipeline():
    """Run complete advanced ZSL pipeline."""
    print("\n" + "="*70)
    print("ADVANCED VISUAL-SEMANTIC ZERO-SHOT LEARNING")
    print("="*70)
    
    # Setup
    seen_classes = ['dog', 'cat', 'bird', 'fish', 'snake']
    unseen_classes = ['horse', 'zebra', 'elephant', 'tiger', 'eagle']
    all_classes = seen_classes + unseen_classes
    
    # Create semantic embeddings (reuse from previous module)
    from types import SimpleNamespace
    
    # Simplified semantic embeddings for this example
    semantic_dim = 50
    np.random.seed(42)
    semantic_embeddings = {
        c: np.random.randn(semantic_dim).astype(np.float32) 
        for c in all_classes
    }
    
    # Normalize embeddings
    for c in all_classes:
        semantic_embeddings[c] /= (np.linalg.norm(semantic_embeddings[c]) + 1e-8)
    
    print(f"\nSemantic embeddings: {semantic_dim}D vectors")
    
    # Generate datasets
    print("\n[Generating Training Data]")
    train_dataset = AdvancedAnimalDataset(
        seen_classes, semantic_embeddings,
        samples_per_class=200, feature_dim=1024
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    print(f"Training: {len(train_dataset)} samples, {len(seen_classes)} classes")
    
    print("\n[Generating Test Data]")
    test_dataset = AdvancedAnimalDataset(
        unseen_classes, semantic_embeddings,
        samples_per_class=100, feature_dim=1024
    )
    
    print(f"Testing: {len(test_dataset)} samples, {len(unseen_classes)} classes")
    
    # Train DeViSE
    print("\n" + "="*70)
    print("METHOD 1: DeViSE")
    print("="*70)
    
    devise_model = DeViSE(
        visual_dim=1024,
        semantic_dim=semantic_dim,
        embedding_dim=256
    )
    
    train_devise_model(
        devise_model, train_loader, semantic_embeddings,
        seen_classes, epochs=50, lr=0.001, loss_type='ranking'
    )
    
    devise_acc, devise_per_class = evaluate_model(
        devise_model, test_dataset, semantic_embeddings,
        unseen_classes, "DeViSE"
    )
    
    # Train Bilinear
    print("\n" + "="*70)
    print("METHOD 2: Bilinear Compatibility")
    print("="*70)
    
    bilinear_model = BilinearCompatibility(
        visual_dim=1024,
        semantic_dim=semantic_dim,
        hidden_dim=512
    )
    
    train_devise_model(
        bilinear_model, train_loader, semantic_embeddings,
        seen_classes, epochs=50, lr=0.001, loss_type='ranking'
    )
    
    bilinear_acc, bilinear_per_class = evaluate_model(
        bilinear_model, test_dataset, semantic_embeddings,
        unseen_classes, "Bilinear Compatibility"
    )
    
    # Comparison
    visualize_model_comparison(
        devise_acc, bilinear_acc,
        devise_per_class, bilinear_per_class,
        unseen_classes
    )
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"DeViSE:   {devise_acc*100:.2f}%")
    print(f"Bilinear: {bilinear_acc*100:.2f}%")
    print(f"Random:   {100/len(unseen_classes):.2f}%")


def visualize_model_comparison(devise_acc, bilinear_acc,
                               devise_per_class, bilinear_per_class,
                               classes):
    """Visualize model comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall comparison
    models = ['DeViSE', 'Bilinear']
    accuracies = [devise_acc * 100, bilinear_acc * 100]
    
    axes[0].bar(models, accuracies, color=['#3498db', '#2ecc71'], alpha=0.8)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Overall Zero-Shot Accuracy')
    axes[0].set_ylim([0, 100])
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # Per-class comparison
    x = np.arange(len(classes))
    width = 0.35
    
    devise_scores = [devise_per_class[c] * 100 for c in classes]
    bilinear_scores = [bilinear_per_class[c] * 100 for c in classes]
    
    axes[1].bar(x - width/2, devise_scores, width, label='DeViSE',
                color='#3498db', alpha=0.8)
    axes[1].bar(x + width/2, bilinear_scores, width, label='Bilinear',
                color='#2ecc71', alpha=0.8)
    
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Per-Class Zero-Shot Accuracy')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(classes, rotation=45, ha='right')
    axes[1].legend()
    axes[1].set_ylim([0, 100])
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/advanced_model_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n[VISUALIZATION]")
    print(f"Saved comparison to: advanced_model_comparison.png")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function."""
    print("\n" + "="*70)
    print("MODULE 56: ADVANCED VISUAL-SEMANTIC EMBEDDING")
    print("="*70)
    
    run_advanced_zsl_pipeline()
    
    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
    DeViSE (Deep Visual-Semantic Embedding):
    ✓ Projects CNN features to semantic embedding space
    ✓ Uses cosine similarity for class prediction
    ✓ Simple and effective architecture
    ✓ Scales to large numbers of classes
    ✗ Linear projection may be limiting
    
    Bilinear Compatibility:
    ✓ Captures interactions between visual and semantic dimensions
    ✓ More expressive than dot product
    ✓ Can model complex relationships
    ✗ More parameters to learn (may overfit on small datasets)
    ✗ Computationally more expensive
    
    Loss Functions:
    - Ranking loss: Encourages correct class to score higher
    - Triplet loss: Encourages embeddings to cluster by class
    - Cross-entropy: Standard classification approach
    
    Practical Tips:
    1. Use pre-trained CNN features (VGG, ResNet, etc.)
    2. Start with DeViSE for simplicity
    3. Try bilinear for more capacity
    4. Experiment with different loss functions
    5. Add regularization to prevent overfitting
    6. Use batch normalization for stability
    
    Next: Generalized zero-shot learning (seen + unseen at test time)
    """)
    print("="*70)


if __name__ == "__main__":
    main()
