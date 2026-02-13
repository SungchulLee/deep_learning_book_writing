"""
Module 56: Zero-Shot Learning - Intermediate Level (Part 2)
============================================================

Semantic Embedding-Based Zero-Shot Learning

This script explores zero-shot learning using continuous semantic embeddings
(word vectors) instead of discrete attributes. We'll implement:
1. Word2Vec embeddings for class representations
2. Visual-semantic compatibility learning
3. ConSE (Convex Combination of Semantic Embeddings)
4. Ranking-based loss functions

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
from gensim.models import Word2Vec
from sklearn.manifold import TSNE

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)


# ============================================================================
# PART 1: Mathematical Background
# ============================================================================

"""
SEMANTIC EMBEDDINGS FOR ZERO-SHOT LEARNING
==========================================

Instead of binary attributes, we use continuous semantic vectors:
- Word2Vec, GloVe, FastText embeddings
- Each class c is represented by a vector s_c ∈ ℝ^d (typically d=300)
- These vectors capture semantic relationships between words/classes

Advantages over attributes:
1. No manual annotation required
2. Captures rich semantic relationships
3. Continuous space allows smoother interpolation
4. Pre-trained on large text corpora

COMPATIBILITY LEARNING
======================

Learn a function F: V × S → ℝ that measures compatibility between:
- Visual features v = φ(x) ∈ V (from image)
- Semantic embedding s_c ∈ S (from class name)

Common compatibility functions:
1. Bilinear: F(v, s) = v^T W s
2. Dot product: F(v, s) = v · s (after projection to same space)
3. Cosine similarity: F(v, s) = (v · s) / (||v|| ||s||)

Training objective (Ranking Loss):
    L = ∑_{(x,y)} max(0, Δ + F(φ(x), s_ȳ) - F(φ(x), s_y))

where:
- (x, y) is training pair (image, correct class)
- ȳ is an incorrect class
- Δ is margin parameter
- Goal: Correct class should have higher compatibility than incorrect

CONVEX COMBINATION OF SEMANTIC EMBEDDINGS (ConSE)
=================================================

At test time, instead of directly predicting unseen classes:
1. Use a classifier trained on seen classes: P(y_seen | x)
2. Project to semantic space using weighted average:
   
   ẑ = ∑_{y ∈ seen} P(y | x) · s_y

3. Find nearest unseen class in semantic space:
   
   ŷ = argmin_{c ∈ unseen} ||ẑ - s_c||

Intuition: If image looks 60% like "dog" and 40% like "cat", its semantic
representation should be between dog and cat embeddings.
"""


# ============================================================================
# PART 2: Synthetic Word Embeddings
# ============================================================================

class SemanticEmbeddings:
    """
    Generate or load semantic embeddings for animal classes.
    
    For educational purposes, we create synthetic embeddings that capture
    semantic relationships (e.g., horse and zebra are similar, both different
    from birds).
    
    In practice, you would use pre-trained Word2Vec, GloVe, or FastText.
    """
    
    def __init__(self, embedding_dim: int = 50):
        """
        Initialize semantic embeddings.
        
        Args:
            embedding_dim: Dimensionality of semantic vectors
        """
        self.embedding_dim = embedding_dim
        self.embeddings = {}
        
        # Create synthetic embeddings based on animal taxonomy
        # We'll place animals in semantic space based on:
        # - Size (small to large)
        # - Domestication (wild to domestic)
        # - Habitat (land, water, air)
        # - Diet (herbivore to carnivore)
        
        self._create_synthetic_embeddings()
    
    def _create_synthetic_embeddings(self):
        """
        Create synthetic semantic embeddings.
        
        We'll use a compositional approach where embedding dimensions
        correspond to semantic features.
        """
        # Base vectors for semantic properties
        np.random.seed(42)
        
        # Dimension groups (each property uses 10 dimensions)
        n_dims_per_property = self.embedding_dim // 5
        
        property_vectors = {
            'size': {
                'small': np.random.randn(n_dims_per_property) * 0.5,
                'medium': np.random.randn(n_dims_per_property) * 0.5,
                'large': np.random.randn(n_dims_per_property) * 0.5,
            },
            'habitat': {
                'land': np.random.randn(n_dims_per_property) * 0.5,
                'water': np.random.randn(n_dims_per_property) * 0.5,
                'air': np.random.randn(n_dims_per_property) * 0.5,
            },
            'domestication': {
                'wild': np.random.randn(n_dims_per_property) * 0.5,
                'domestic': np.random.randn(n_dims_per_property) * 0.5,
            },
            'covering': {
                'fur': np.random.randn(n_dims_per_property) * 0.5,
                'feathers': np.random.randn(n_dims_per_property) * 0.5,
                'scales': np.random.randn(n_dims_per_property) * 0.5,
            },
            'diet': {
                'herbivore': np.random.randn(n_dims_per_property) * 0.5,
                'carnivore': np.random.randn(n_dims_per_property) * 0.5,
                'omnivore': np.random.randn(n_dims_per_property) * 0.5,
            }
        }
        
        # Define animals by their properties
        animal_properties = {
            # Seen classes
            'dog':   ('medium', 'land', 'domestic', 'fur', 'omnivore'),
            'cat':   ('small', 'land', 'domestic', 'fur', 'carnivore'),
            'bird':  ('small', 'air', 'wild', 'feathers', 'omnivore'),
            'fish':  ('small', 'water', 'wild', 'scales', 'omnivore'),
            'snake': ('small', 'land', 'wild', 'scales', 'carnivore'),
            
            # Unseen classes
            'horse':    ('large', 'land', 'domestic', 'fur', 'herbivore'),
            'zebra':    ('large', 'land', 'wild', 'fur', 'herbivore'),
            'elephant': ('large', 'land', 'wild', 'fur', 'herbivore'),
            'tiger':    ('large', 'land', 'wild', 'fur', 'carnivore'),
            'eagle':    ('medium', 'air', 'wild', 'feathers', 'carnivore'),
        }
        
        # Construct embeddings by concatenating property vectors
        for animal, props in animal_properties.items():
            embedding_parts = []
            
            for i, (prop_type, prop_value) in enumerate(zip(
                ['size', 'habitat', 'domestication', 'covering', 'diet'],
                props
            )):
                embedding_parts.append(property_vectors[prop_type][prop_value])
            
            # Concatenate and add small noise
            embedding = np.concatenate(embedding_parts)
            embedding += np.random.randn(self.embedding_dim) * 0.1
            
            # Normalize (common for word embeddings)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            self.embeddings[animal] = embedding
    
    def get_embedding(self, class_name: str) -> np.ndarray:
        """Get embedding for a class."""
        return self.embeddings[class_name]
    
    def get_embeddings_matrix(self, class_names: List[str]) -> np.ndarray:
        """Get embeddings for multiple classes as a matrix."""
        return np.array([self.embeddings[name] for name in class_names])
    
    def visualize_embeddings(self, class_names: List[str] = None):
        """Visualize embeddings using t-SNE."""
        if class_names is None:
            class_names = list(self.embeddings.keys())
        
        # Get embeddings
        embeddings_matrix = self.get_embeddings_matrix(class_names)
        
        # Reduce to 2D using t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(class_names)-1))
        embeddings_2d = tsne.fit_transform(embeddings_matrix)
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        # Color seen vs unseen differently
        seen_classes = ['dog', 'cat', 'bird', 'fish', 'snake']
        
        for i, name in enumerate(class_names):
            color = 'blue' if name in seen_classes else 'red'
            marker = 'o' if name in seen_classes else '^'
            label = 'Seen' if (name in seen_classes and i == 0) else ('Unseen' if name not in seen_classes and i == len(seen_classes) else None)
            
            plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1],
                       c=color, marker=marker, s=200, alpha=0.6, label=label)
            plt.annotate(name, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                        fontsize=12, fontweight='bold')
        
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title('Semantic Embedding Space (t-SNE Visualization)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('/home/claude/semantic_embeddings_tsne.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved t-SNE visualization to: semantic_embeddings_tsne.png")
        plt.close()


# ============================================================================
# PART 3: Visual Feature Generation
# ============================================================================

def generate_visual_features(class_name: str, semantic_embedding: np.ndarray,
                            n_samples: int, feature_dim: int,
                            noise_level: float = 0.3) -> np.ndarray:
    """
    Generate synthetic visual features correlated with semantic embeddings.
    
    Args:
        class_name: Name of the class
        semantic_embedding: Semantic embedding vector
        n_samples: Number of samples to generate
        feature_dim: Visual feature dimensionality
        noise_level: Amount of noise to add
    
    Returns:
        Visual features of shape (n_samples, feature_dim)
    """
    # Project semantic embedding to visual feature space
    # In reality, this relationship is what we learn during training
    
    # Create a random projection matrix (same for all classes)
    np.random.seed(123)  # Fixed seed for projection
    projection = np.random.randn(len(semantic_embedding), feature_dim) * 0.5
    
    # Project semantic vector
    base_features = semantic_embedding @ projection
    
    # Generate samples with noise
    features = np.tile(base_features, (n_samples, 1))
    features += np.random.randn(n_samples, feature_dim) * noise_level
    
    return features


# ============================================================================
# PART 4: Compatibility Learning Model
# ============================================================================

class VisualSemanticCompatibility(nn.Module):
    """
    Learn compatibility between visual features and semantic embeddings.
    
    Architecture:
    - Visual encoder: maps raw features to embedding space
    - Semantic encoder: maps semantic vectors to same embedding space
    - Compatibility: dot product in joint space
    
    Loss: Ranking loss (hinge loss)
        L = max(0, margin + F(v, s_neg) - F(v, s_pos))
    """
    
    def __init__(self, visual_dim: int, semantic_dim: int, 
                 embedding_dim: int = 128):
        """
        Initialize compatibility model.
        
        Args:
            visual_dim: Visual feature dimensionality
            semantic_dim: Semantic embedding dimensionality
            embedding_dim: Joint embedding space dimensionality
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Visual encoder
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        # Semantic encoder
        self.semantic_encoder = nn.Sequential(
            nn.Linear(semantic_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
    
    def encode_visual(self, v):
        """Encode visual features."""
        return self.visual_encoder(v)
    
    def encode_semantic(self, s):
        """Encode semantic vectors."""
        return self.semantic_encoder(s)
    
    def compatibility(self, v_encoded, s_encoded):
        """
        Compute compatibility score between encoded visual and semantic vectors.
        
        Args:
            v_encoded: Encoded visual features (batch_size, embedding_dim)
            s_encoded: Encoded semantic vectors (batch_size or n_classes, embedding_dim)
        
        Returns:
            Compatibility scores
        """
        # Normalize vectors for cosine similarity
        v_norm = F.normalize(v_encoded, p=2, dim=1)
        s_norm = F.normalize(s_encoded, p=2, dim=1)
        
        # Compute dot product (cosine similarity)
        if v_norm.size(0) == s_norm.size(0):
            # Element-wise compatibility
            return (v_norm * s_norm).sum(dim=1)
        else:
            # Matrix compatibility: (batch_size, n_classes)
            return v_norm @ s_norm.T
    
    def forward(self, v, s):
        """
        Forward pass.
        
        Args:
            v: Visual features (batch_size, visual_dim)
            s: Semantic vectors (batch_size, semantic_dim)
        
        Returns:
            Compatibility scores
        """
        v_encoded = self.encode_visual(v)
        s_encoded = self.encode_semantic(s)
        return self.compatibility(v_encoded, s_encoded)
    
    def predict_class(self, v, candidate_embeddings, candidate_names):
        """
        Predict class for visual features.
        
        Args:
            v: Visual features (batch_size, visual_dim)
            candidate_embeddings: Semantic embeddings for candidates (n_classes, semantic_dim)
            candidate_names: List of candidate class names
        
        Returns:
            Predicted class indices
        """
        self.eval()
        with torch.no_grad():
            # Encode visual
            v_encoded = self.encode_visual(v)
            
            # Encode all candidate semantics
            s_encoded = self.encode_semantic(candidate_embeddings)
            
            # Compute compatibility with all candidates
            # Shape: (batch_size, n_classes)
            scores = self.compatibility(v_encoded, s_encoded)
            
            # Predict class with highest compatibility
            pred_indices = torch.argmax(scores, dim=1)
        
        return pred_indices


# ============================================================================
# PART 5: Training with Ranking Loss
# ============================================================================

def train_compatibility_model(model, X_train, y_train, semantic_embeddings_dict,
                              seen_classes, epochs=50, batch_size=32, lr=0.001,
                              margin=0.2):
    """
    Train visual-semantic compatibility model with ranking loss.
    
    Args:
        model: Compatibility model
        X_train: Visual features
        y_train: Class labels
        semantic_embeddings_dict: Dictionary of semantic embeddings
        seen_classes: List of seen class names
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        margin: Margin for ranking loss
    
    Returns:
        Training loss history
    """
    print("\n" + "="*70)
    print("TRAINING VISUAL-SEMANTIC COMPATIBILITY")
    print("="*70)
    
    # Prepare data
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    
    # Create dataset with positive and negative semantic embeddings
    class_to_semantic = {c: torch.tensor(semantic_embeddings_dict[c], dtype=torch.float32) 
                         for c in seen_classes}
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        # Create batches
        indices = np.random.permutation(len(X_train))
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_x = X_tensor[batch_indices]
            batch_y = y_train[batch_indices]
            
            # Get positive semantic embeddings (correct classes)
            pos_semantics = torch.stack([class_to_semantic[label] for label in batch_y])
            
            # Get negative semantic embeddings (random incorrect classes)
            neg_classes = [np.random.choice([c for c in seen_classes if c != label]) 
                          for label in batch_y]
            neg_semantics = torch.stack([class_to_semantic[label] for label in neg_classes])
            
            # Compute compatibility scores
            pos_scores = model(batch_x, pos_semantics)
            neg_scores = model(batch_x, neg_semantics)
            
            # Ranking loss (hinge loss)
            # L = max(0, margin + score_negative - score_positive)
            loss = torch.clamp(margin + neg_scores - pos_scores, min=0).mean()
            
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


# ============================================================================
# PART 6: ConSE (Convex Combination of Semantic Embeddings)
# ============================================================================

class ConSE:
    """
    ConSE: Convex Combination of Semantic Embeddings
    
    Method:
    1. Train classifier on seen classes
    2. At test time, get probabilities over seen classes
    3. Compute weighted average of seen class embeddings
    4. Find nearest unseen class in semantic space
    """
    
    def __init__(self, visual_dim: int, n_seen_classes: int, hidden_dim: int = 256):
        """Initialize ConSE model."""
        self.classifier = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_seen_classes)
        )
    
    def train_seen_classifier(self, X_train, y_train, seen_classes,
                            epochs=50, batch_size=32, lr=0.001):
        """Train classifier on seen classes."""
        print("\n" + "="*70)
        print("TRAINING ConSE CLASSIFIER")
        print("="*70)
        
        # Prepare data
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        
        class_to_idx = {c: i for i, c in enumerate(seen_classes)}
        y_indices = torch.tensor([class_to_idx[label] for label in y_train], dtype=torch.long)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_indices)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.classifier.train()
            epoch_loss = 0.0
            
            for batch_x, batch_y in dataloader:
                logits = self.classifier(batch_x)
                loss = criterion(logits, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f}")
    
    def predict_unseen(self, X_test, seen_classes, unseen_classes,
                      semantic_embeddings_dict):
        """
        Predict unseen classes using ConSE method.
        
        Args:
            X_test: Test visual features
            seen_classes: List of seen class names
            unseen_classes: List of unseen class names
            semantic_embeddings_dict: Dictionary of semantic embeddings
        
        Returns:
            Predicted class names
        """
        self.classifier.eval()
        
        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.float32)
            
            # Get probabilities over seen classes
            logits = self.classifier(X_tensor)
            probs = F.softmax(logits, dim=1).numpy()
            
            # Get seen class embeddings
            seen_embeddings = np.array([semantic_embeddings_dict[c] for c in seen_classes])
            
            # Compute weighted average (convex combination)
            # Shape: (n_test, n_seen) @ (n_seen, embedding_dim) = (n_test, embedding_dim)
            projected_embeddings = probs @ seen_embeddings
            
            # Get unseen class embeddings
            unseen_embeddings = np.array([semantic_embeddings_dict[c] for c in unseen_classes])
            
            # Find nearest unseen class for each test sample
            predictions = []
            for proj_emb in projected_embeddings:
                # Compute distances to all unseen classes
                distances = np.linalg.norm(unseen_embeddings - proj_emb, axis=1)
                pred_idx = np.argmin(distances)
                predictions.append(unseen_classes[pred_idx])
        
        return np.array(predictions)


# ============================================================================
# PART 7: Evaluation and Comparison
# ============================================================================

def run_semantic_zsl_pipeline():
    """Complete semantic embedding-based ZSL pipeline."""
    print("\n" + "="*70)
    print("SEMANTIC EMBEDDING-BASED ZERO-SHOT LEARNING")
    print("="*70)
    
    # Setup
    seen_classes = ['dog', 'cat', 'bird', 'fish', 'snake']
    unseen_classes = ['horse', 'zebra', 'elephant', 'tiger', 'eagle']
    
    # Create semantic embeddings
    print("\n[STEP 1: Creating Semantic Embeddings]")
    semantic_embedder = SemanticEmbeddings(embedding_dim=50)
    semantic_embedder.visualize_embeddings()
    
    # Generate visual features
    print("\n[STEP 2: Generating Visual Features]")
    visual_dim = 100
    samples_per_class = 150
    
    X_train_list = []
    y_train_list = []
    for class_name in seen_classes:
        sem_emb = semantic_embedder.get_embedding(class_name)
        features = generate_visual_features(class_name, sem_emb, samples_per_class, visual_dim)
        X_train_list.append(features)
        y_train_list.extend([class_name] * samples_per_class)
    
    X_train = np.vstack(X_train_list)
    y_train = np.array(y_train_list)
    
    # Generate test data
    X_test_list = []
    y_test_list = []
    for class_name in unseen_classes:
        sem_emb = semantic_embedder.get_embedding(class_name)
        features = generate_visual_features(class_name, sem_emb, samples_per_class // 2, visual_dim)
        X_test_list.append(features)
        y_test_list.extend([class_name] * (samples_per_class // 2))
    
    X_test = np.vstack(X_test_list)
    y_test = np.array(y_test_list)
    
    print(f"Training set: {X_train.shape}, {len(seen_classes)} classes")
    print(f"Test set: {X_test.shape}, {len(unseen_classes)} classes")
    
    # Method 1: Compatibility learning
    print("\n[STEP 3: Training Compatibility Model]")
    compat_model = VisualSemanticCompatibility(
        visual_dim=visual_dim,
        semantic_dim=semantic_embedder.embedding_dim,
        embedding_dim=128
    )
    
    train_compatibility_model(
        compat_model, X_train, y_train,
        semantic_embedder.embeddings, seen_classes,
        epochs=50, batch_size=32
    )
    
    # Evaluate compatibility model
    print("\n[STEP 4: Evaluating Compatibility Model]")
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    unseen_embeddings = torch.tensor(
        semantic_embedder.get_embeddings_matrix(unseen_classes),
        dtype=torch.float32
    )
    
    pred_indices = compat_model.predict_class(X_test_tensor, unseen_embeddings, unseen_classes)
    compat_predictions = [unseen_classes[idx] for idx in pred_indices.numpy()]
    compat_accuracy = np.mean(np.array(compat_predictions) == y_test)
    
    print(f"Compatibility Model Accuracy: {compat_accuracy*100:.2f}%")
    
    # Method 2: ConSE
    print("\n[STEP 5: Training ConSE Model]")
    conse_model = ConSE(visual_dim=visual_dim, n_seen_classes=len(seen_classes))
    conse_model.train_seen_classifier(X_train, y_train, seen_classes, epochs=50)
    
    print("\n[STEP 6: Evaluating ConSE Model]")
    conse_predictions = conse_model.predict_unseen(
        X_test, seen_classes, unseen_classes,
        semantic_embedder.embeddings
    )
    conse_accuracy = np.mean(conse_predictions == y_test)
    
    print(f"ConSE Model Accuracy: {conse_accuracy*100:.2f}%")
    
    # Comparison
    print("\n" + "="*70)
    print("METHOD COMPARISON")
    print("="*70)
    print(f"Compatibility Learning: {compat_accuracy*100:.2f}%")
    print(f"ConSE: {conse_accuracy*100:.2f}%")
    print(f"Random Baseline: {100/len(unseen_classes):.2f}%")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function."""
    print("\n" + "="*70)
    print("MODULE 56: ZERO-SHOT LEARNING - SEMANTIC EMBEDDINGS")
    print("="*70)
    
    run_semantic_zsl_pipeline()
    
    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
    Semantic Embeddings for ZSL:
    ✓ Use continuous word vectors instead of discrete attributes
    ✓ No manual annotation required (pre-trained on text)
    ✓ Captures rich semantic relationships automatically
    ✓ Allows smooth interpolation in semantic space
    
    Compatibility Learning:
    ✓ Learns joint visual-semantic embedding space
    ✓ Ranking loss encourages correct classes to have higher scores
    ✓ Can generalize to any class with semantic representation
    ✓ Scales well to large numbers of classes
    
    ConSE Method:
    ✓ Uses seen class probabilities as weights
    ✓ Projects to semantic space via convex combination
    ✓ Simple and interpretable
    ✓ Leverages relationships between seen and unseen classes
    
    Practical Considerations:
    - Quality of semantic embeddings matters
    - Pre-trained embeddings (Word2Vec, GloVe) work well
    - Can use BERT, GPT embeddings for better representations
    - Hubness problem in high-dimensional spaces
    
    Next: Visual-semantic embedding with deep networks
    """)
    print("="*70)


if __name__ == "__main__":
    main()
