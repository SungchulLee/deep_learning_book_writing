"""
Module 35: Multimodal Vision - Beginner Level
==============================================

Topic: Basic Multimodal Embeddings and Image-Text Matching

This script introduces fundamental concepts of multimodal learning by building
a simple dual-encoder model that maps images and text into a shared embedding space.

Key Concepts:
1. Dual-encoder architecture (separate encoders for vision and language)
2. Shared embedding space for cross-modal matching
3. Cosine similarity for measuring alignment
4. Simple contrastive loss for learning
5. Image-text retrieval basics

Learning Objectives:
- Understand how to create joint embeddings for different modalities
- Learn to compute similarity between images and text
- Implement basic image-text matching
- Build simple retrieval systems

Mathematical Background:
- Image encoder: f_v: I → R^d
- Text encoder: f_t: T → R^d  
- Cosine similarity: sim(a,b) = (a·b) / (||a|| ||b||)
- Contrastive loss: maximize similarity for matched pairs, minimize for mismatched

Author: Educational AI
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# ============================================================================
# PART 1: SIMPLE ENCODERS
# ============================================================================

class SimpleImageEncoder(nn.Module):
    """
    Basic image encoder that maps images to embedding vectors.
    
    Architecture:
    - Takes flattened image features as input
    - Uses 2-layer MLP to produce embeddings
    - L2 normalization for embeddings
    
    Args:
        input_dim: Dimension of input image features (e.g., 2048 for ResNet)
        hidden_dim: Hidden layer size
        embed_dim: Output embedding dimension
    
    Mathematical Operation:
        h = ReLU(W1 * x + b1)
        e = W2 * h + b2
        output = e / ||e||  (L2 normalization)
    """
    
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 1024, embed_dim: int = 512):
        super(SimpleImageEncoder, self).__init__()
        
        # Two-layer MLP encoder
        # Layer 1: input_dim -> hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # Layer 2: hidden_dim -> embed_dim
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        
        # Activation function (ReLU for non-linearity)
        self.relu = nn.ReLU()
        
        # Dropout for regularization (prevents overfitting)
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through image encoder.
        
        Args:
            x: Image features of shape (batch_size, input_dim)
        
        Returns:
            Normalized embeddings of shape (batch_size, embed_dim)
        """
        # First layer with activation and dropout
        # Shape: (batch_size, input_dim) -> (batch_size, hidden_dim)
        h = self.relu(self.fc1(x))
        h = self.dropout(h)
        
        # Second layer to get embeddings
        # Shape: (batch_size, hidden_dim) -> (batch_size, embed_dim)
        embeddings = self.fc2(h)
        
        # L2 normalize embeddings for cosine similarity
        # This ensures ||embeddings|| = 1, so dot product = cosine similarity
        # Shape: (batch_size, embed_dim) - normalized
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class SimpleTextEncoder(nn.Module):
    """
    Basic text encoder that maps text to embedding vectors.
    
    Architecture:
    - Takes pre-computed text features as input (e.g., from BERT)
    - Uses 2-layer MLP to produce embeddings  
    - L2 normalization for embeddings
    
    Args:
        input_dim: Dimension of input text features (e.g., 768 for BERT)
        hidden_dim: Hidden layer size
        embed_dim: Output embedding dimension
    
    Note: In practice, you'd use a proper text encoder (BERT, GPT) but we
    simplify here by assuming pre-computed features for educational purposes.
    """
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 1024, embed_dim: int = 512):
        super(SimpleTextEncoder, self).__init__()
        
        # Two-layer MLP encoder (same structure as image encoder)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through text encoder.
        
        Args:
            x: Text features of shape (batch_size, input_dim)
        
        Returns:
            Normalized embeddings of shape (batch_size, embed_dim)
        """
        # Same architecture as image encoder
        h = self.relu(self.fc1(x))
        h = self.dropout(h)
        embeddings = self.fc2(h)
        
        # L2 normalization for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


# ============================================================================
# PART 2: MULTIMODAL MODEL WITH SIMILARITY COMPUTATION
# ============================================================================

class SimpleMultimodalModel(nn.Module):
    """
    Simple multimodal model combining image and text encoders.
    
    This model learns a joint embedding space where semantically related
    images and text have high cosine similarity.
    
    Components:
    1. Image encoder: maps images to embeddings
    2. Text encoder: maps text to embeddings
    3. Shared embedding space of dimension embed_dim
    
    The key idea: matched (image, text) pairs should have similar embeddings!
    
    Args:
        image_dim: Input dimension for image features
        text_dim: Input dimension for text features
        embed_dim: Shared embedding dimension
        hidden_dim: Hidden layer dimension for encoders
    """
    
    def __init__(self, image_dim: int = 2048, text_dim: int = 768, 
                 embed_dim: int = 512, hidden_dim: int = 1024):
        super(SimpleMultimodalModel, self).__init__()
        
        # Initialize separate encoders for each modality
        self.image_encoder = SimpleImageEncoder(image_dim, hidden_dim, embed_dim)
        self.text_encoder = SimpleTextEncoder(text_dim, hidden_dim, embed_dim)
        
        # Store embedding dimension
        self.embed_dim = embed_dim
        
    def forward(self, images: torch.Tensor, texts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass encoding both images and texts.
        
        Args:
            images: Image features (batch_size, image_dim)
            texts: Text features (batch_size, text_dim)
        
        Returns:
            Tuple of (image_embeddings, text_embeddings), each (batch_size, embed_dim)
        """
        # Encode images into embedding space
        image_embeddings = self.image_encoder(images)
        
        # Encode texts into same embedding space
        text_embeddings = self.text_encoder(texts)
        
        return image_embeddings, text_embeddings
    
    def compute_similarity(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between image and text embeddings.
        
        Since embeddings are L2-normalized, dot product equals cosine similarity:
        sim(a,b) = a·b = (a·b) / (||a|| ||b||) when ||a|| = ||b|| = 1
        
        Args:
            image_emb: Image embeddings (batch_i, embed_dim)
            text_emb: Text embeddings (batch_t, embed_dim)
        
        Returns:
            Similarity matrix (batch_i, batch_t)
            Entry [i,j] = similarity between image i and text j
        """
        # Matrix multiplication: (batch_i, embed_dim) @ (embed_dim, batch_t)
        # Result shape: (batch_i, batch_t)
        # Each entry is the dot product (= cosine similarity for normalized vectors)
        similarity = torch.matmul(image_emb, text_emb.t())
        
        return similarity
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Convenience method to encode only images."""
        return self.image_encoder(images)
    
    def encode_text(self, texts: torch.Tensor) -> torch.Tensor:
        """Convenience method to encode only texts."""
        return self.text_encoder(texts)


# ============================================================================
# PART 3: SIMPLE CONTRASTIVE LOSS
# ============================================================================

class SimpleContrastiveLoss(nn.Module):
    """
    Simple contrastive loss for learning multimodal embeddings.
    
    Goal: Maximize similarity for matched (image, text) pairs,
          Minimize similarity for mismatched pairs.
    
    For each image-text pair (i, i), the loss encourages:
    - High similarity: sim(image_i, text_i) should be large
    - Low similarity: sim(image_i, text_j) for j≠i should be small
    
    Mathematical Formulation:
        For a batch of size N, we have N positive pairs (i, i)
        and N*(N-1) negative pairs (i, j) where j≠i
        
        Loss = -log(exp(sim(i,i)) / Σ_j exp(sim(i,j)))
        
    This is a simplified version. Full CLIP uses symmetric loss (both directions).
    
    Args:
        margin: Minimum margin between positive and negative pairs
    """
    
    def __init__(self, margin: float = 0.2):
        super(SimpleContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute simple contrastive loss.
        
        Args:
            image_emb: Image embeddings (batch_size, embed_dim)
            text_emb: Text embeddings (batch_size, embed_dim)
        
        Returns:
            Scalar loss value
        """
        batch_size = image_emb.shape[0]
        
        # Compute pairwise similarity matrix
        # Shape: (batch_size, batch_size)
        # similarity[i,j] = cosine similarity between image_i and text_j
        similarity = torch.matmul(image_emb, text_emb.t())
        
        # Positive pairs are on the diagonal (i, i)
        # Shape: (batch_size,)
        positive_sim = torch.diag(similarity)
        
        # For each image, compute loss against all texts
        # We want positive_sim to be larger than all other similarities by margin
        loss = 0.0
        
        for i in range(batch_size):
            # Get all similarities for image i
            # Shape: (batch_size,)
            all_sims = similarity[i, :]
            
            # For each negative pair (i, j) where j != i
            for j in range(batch_size):
                if i != j:
                    # Compute margin loss: max(0, margin - (pos_sim - neg_sim))
                    # We want pos_sim - neg_sim > margin
                    # If violated, we get positive loss
                    margin_loss = F.relu(self.margin - (positive_sim[i] - all_sims[j]))
                    loss += margin_loss
        
        # Average over all pairs
        num_pairs = batch_size * (batch_size - 1)
        loss = loss / num_pairs
        
        return loss


# ============================================================================
# PART 4: TOY DATASET
# ============================================================================

class ToyMultimodalDataset(Dataset):
    """
    Toy dataset for multimodal learning (for educational purposes).
    
    In practice, you'd use real image-text datasets like MS-COCO or Flickr30K.
    Here we create synthetic data to demonstrate the concepts.
    
    Data structure:
    - Each sample has image features and text features
    - We create 5 "classes" where images and texts of same class are related
    - Within class: images and texts are similar
    - Across classes: images and texts are different
    
    Args:
        num_samples: Number of samples per class
        image_dim: Dimension of image features
        text_dim: Dimension of text features
        num_classes: Number of semantic classes
    """
    
    def __init__(self, num_samples: int = 100, image_dim: int = 2048, 
                 text_dim: int = 768, num_classes: int = 5):
        super(ToyMultimodalDataset, self).__init__()
        
        self.num_samples = num_samples
        self.num_classes = num_classes
        
        # Generate synthetic data
        # For each class, we create a "cluster center" and add noise
        self.image_features = []
        self.text_features = []
        self.labels = []
        
        for class_idx in range(num_classes):
            # Create random cluster centers for this class
            # These centers represent the "semantic concept" of the class
            image_center = torch.randn(image_dim) * 2
            text_center = torch.randn(text_dim) * 2
            
            # Generate samples around these centers
            for _ in range(num_samples):
                # Add noise to create variations within class
                # Smaller noise = more similar within class
                image = image_center + torch.randn(image_dim) * 0.5
                text = text_center + torch.randn(text_dim) * 0.5
                
                self.image_features.append(image)
                self.text_features.append(text)
                self.labels.append(class_idx)
        
        # Convert lists to tensors
        self.image_features = torch.stack(self.image_features)
        self.text_features = torch.stack(self.text_features)
        self.labels = torch.tensor(self.labels)
        
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a single sample.
        
        Returns:
            Tuple of (image_features, text_features, label)
        """
        return self.image_features[idx], self.text_features[idx], self.labels[idx]


# ============================================================================
# PART 5: TRAINING FUNCTION
# ============================================================================

def train_multimodal_model(model: nn.Module, train_loader: DataLoader, 
                          num_epochs: int = 10, learning_rate: float = 0.001,
                          device: str = 'cpu') -> List[float]:
    """
    Train the simple multimodal model.
    
    Training procedure:
    1. For each batch of (image, text) pairs
    2. Encode images and texts into embedding space
    3. Compute contrastive loss
    4. Backpropagate and update weights
    
    Args:
        model: SimpleMultimodalModel to train
        train_loader: DataLoader providing training data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cpu' or 'cuda')
    
    Returns:
        List of losses per epoch
    """
    # Move model to device
    model = model.to(device)
    
    # Initialize optimizer (Adam is good default)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize loss function
    criterion = SimpleContrastiveLoss(margin=0.2)
    
    # Track losses
    epoch_losses = []
    
    print("Starting training...")
    print(f"Device: {device}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, texts, labels) in enumerate(train_loader):
            # Move data to device
            images = images.to(device)
            texts = texts.to(device)
            
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            
            # Forward pass: encode images and texts
            # Returns embeddings in shared space
            image_emb, text_emb = model(images, texts)
            
            # Compute contrastive loss
            # Encourages matched pairs to be similar, mismatched to be different
            loss = criterion(image_emb, text_emb)
            
            # Backward pass: compute gradients
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Track statistics
            total_loss += loss.item()
            num_batches += 1
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}], "
                      f"Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}")
        
        # Calculate average loss for epoch
        avg_epoch_loss = total_loss / num_batches
        epoch_losses.append(avg_epoch_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_epoch_loss:.4f}")
        print("-" * 50)
    
    print("Training completed!")
    return epoch_losses


# ============================================================================
# PART 6: RETRIEVAL EVALUATION
# ============================================================================

def evaluate_retrieval(model: nn.Module, test_loader: DataLoader, 
                       device: str = 'cpu', k: int = 5) -> dict:
    """
    Evaluate image-text retrieval performance.
    
    Retrieval Tasks:
    1. Image-to-Text (I2T): Given an image, find matching text
    2. Text-to-Image (T2I): Given text, find matching image
    
    Metric: Recall@K
    - For each query, check if correct match is in top-K retrieved results
    - Recall@K = (# queries with match in top-K) / (total queries)
    
    Args:
        model: Trained multimodal model
        test_loader: DataLoader with test data
        device: Device for computation
        k: Top-K results to consider
    
    Returns:
        Dictionary with retrieval metrics
    """
    model.eval()  # Set to evaluation mode
    
    # Collect all embeddings and labels
    all_image_emb = []
    all_text_emb = []
    all_labels = []
    
    print("Encoding all test samples...")
    
    with torch.no_grad():  # No gradients needed for evaluation
        for images, texts, labels in test_loader:
            images = images.to(device)
            texts = texts.to(device)
            
            # Encode all samples
            image_emb, text_emb = model(images, texts)
            
            all_image_emb.append(image_emb)
            all_text_emb.append(text_emb)
            all_labels.append(labels)
    
    # Concatenate all batches
    all_image_emb = torch.cat(all_image_emb, dim=0)  # (N, embed_dim)
    all_text_emb = torch.cat(all_text_emb, dim=0)     # (N, embed_dim)
    all_labels = torch.cat(all_labels, dim=0)         # (N,)
    
    num_samples = all_image_emb.shape[0]
    
    # Compute full similarity matrix
    # Shape: (num_images, num_texts)
    similarity = torch.matmul(all_image_emb, all_text_emb.t())
    
    # ========== Image-to-Text Retrieval ==========
    print(f"\nEvaluating Image-to-Text Retrieval (Recall@{k})...")
    
    i2t_recall = 0
    for i in range(num_samples):
        # Get similarities for image i to all texts
        # Shape: (num_samples,)
        sims = similarity[i, :]
        
        # Get top-K text indices
        # torch.topk returns (values, indices)
        _, top_k_indices = torch.topk(sims, k)
        
        # Check if correct text (same label) is in top-K
        # Correct text is at index i (assuming paired data)
        if i in top_k_indices:
            i2t_recall += 1
    
    i2t_recall = i2t_recall / num_samples
    
    # ========== Text-to-Image Retrieval ==========
    print(f"Evaluating Text-to-Image Retrieval (Recall@{k})...")
    
    t2i_recall = 0
    for i in range(num_samples):
        # Get similarities for text i to all images
        # Shape: (num_samples,)
        sims = similarity[:, i]
        
        # Get top-K image indices
        _, top_k_indices = torch.topk(sims, k)
        
        # Check if correct image (same label) is in top-K
        if i in top_k_indices:
            t2i_recall += 1
    
    t2i_recall = t2i_recall / num_samples
    
    # Print results
    print(f"\nRetrieval Results:")
    print(f"  Image-to-Text Recall@{k}: {i2t_recall:.4f} ({i2t_recall*100:.2f}%)")
    print(f"  Text-to-Image Recall@{k}: {t2i_recall:.4f} ({t2i_recall*100:.2f}%)")
    
    return {
        'i2t_recall': i2t_recall,
        't2i_recall': t2i_recall,
        'average_recall': (i2t_recall + t2i_recall) / 2
    }


# ============================================================================
# PART 7: VISUALIZATION
# ============================================================================

def visualize_embeddings(model: nn.Module, test_loader: DataLoader, 
                        device: str = 'cpu'):
    """
    Visualize embeddings using dimensionality reduction (PCA).
    
    This helps understand if the model learned good joint embeddings:
    - Images and texts of same class should cluster together
    - Different classes should be separable
    
    Args:
        model: Trained multimodal model
        test_loader: DataLoader with test data
        device: Device for computation
    """
    model.eval()
    
    # Collect embeddings
    image_embs = []
    text_embs = []
    labels = []
    
    with torch.no_grad():
        for images, texts, lbls in test_loader:
            images = images.to(device)
            texts = texts.to(device)
            
            img_emb, txt_emb = model(images, texts)
            
            image_embs.append(img_emb.cpu())
            text_embs.append(txt_emb.cpu())
            labels.append(lbls)
    
    # Concatenate
    image_embs = torch.cat(image_embs, dim=0).numpy()
    text_embs = torch.cat(text_embs, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    
    # Simple PCA for visualization (reduce to 2D)
    from sklearn.decomposition import PCA
    
    # Combine image and text embeddings
    all_embs = np.vstack([image_embs, text_embs])
    
    # Apply PCA
    pca = PCA(n_components=2)
    embs_2d = pca.fit_transform(all_embs)
    
    # Split back
    n = len(image_embs)
    image_embs_2d = embs_2d[:n]
    text_embs_2d = embs_2d[n:]
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Plot images
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(image_embs_2d[:, 0], image_embs_2d[:, 1], 
                         c=labels, cmap='tab10', marker='o', alpha=0.6, s=50)
    plt.colorbar(scatter, label='Class')
    plt.title('Image Embeddings (2D PCA)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    # Plot texts
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(text_embs_2d[:, 0], text_embs_2d[:, 1], 
                         c=labels, cmap='tab10', marker='^', alpha=0.6, s=50)
    plt.colorbar(scatter, label='Class')
    plt.title('Text Embeddings (2D PCA)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    plt.tight_layout()
    plt.savefig('embeddings_visualization.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'embeddings_visualization.png'")
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function demonstrating simple multimodal learning.
    
    Workflow:
    1. Create toy dataset
    2. Initialize model
    3. Train with contrastive loss
    4. Evaluate retrieval performance
    5. Visualize learned embeddings
    """
    print("=" * 70)
    print("SIMPLE MULTIMODAL EMBEDDINGS - BEGINNER TUTORIAL")
    print("=" * 70)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Hyperparameters
    image_dim = 2048  # Typical ResNet feature dimension
    text_dim = 768    # Typical BERT feature dimension
    embed_dim = 512   # Shared embedding space dimension
    hidden_dim = 1024
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.001
    
    # Create datasets
    print("\n" + "=" * 70)
    print("STEP 1: Creating Datasets")
    print("=" * 70)
    
    train_dataset = ToyMultimodalDataset(
        num_samples=100,  # 100 samples per class
        image_dim=image_dim,
        text_dim=text_dim,
        num_classes=5
    )
    
    test_dataset = ToyMultimodalDataset(
        num_samples=20,  # 20 samples per class for testing
        image_dim=image_dim,
        text_dim=text_dim,
        num_classes=5
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    print("\n" + "=" * 70)
    print("STEP 2: Initializing Model")
    print("=" * 70)
    
    model = SimpleMultimodalModel(
        image_dim=image_dim,
        text_dim=text_dim,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim
    )
    
    # Print model architecture
    print(f"\nModel Architecture:")
    print(f"  Image Encoder: {image_dim} -> {hidden_dim} -> {embed_dim}")
    print(f"  Text Encoder: {text_dim} -> {hidden_dim} -> {embed_dim}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    print("\n" + "=" * 70)
    print("STEP 3: Training Model")
    print("=" * 70)
    
    losses = train_multimodal_model(
        model=model,
        train_loader=train_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device
    )
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True)
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    print("\nTraining loss plot saved as 'training_loss.png'")
    
    # Evaluate retrieval
    print("\n" + "=" * 70)
    print("STEP 4: Evaluating Retrieval Performance")
    print("=" * 70)
    
    metrics = evaluate_retrieval(
        model=model,
        test_loader=test_loader,
        device=device,
        k=5
    )
    
    # Visualize embeddings
    print("\n" + "=" * 70)
    print("STEP 5: Visualizing Embeddings")
    print("=" * 70)
    
    visualize_embeddings(model, test_loader, device)
    
    print("\n" + "=" * 70)
    print("TUTORIAL COMPLETED!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Dual encoders map different modalities to a shared space")
    print("2. Contrastive loss encourages matching pairs to be similar")
    print("3. Retrieval performance shows if model learned good alignments")
    print("4. Visualization helps understand embedding quality")
    print("\nNext Steps:")
    print("- Try with real image-text datasets (MS-COCO, Flickr30K)")
    print("- Implement more sophisticated encoders (ResNet, BERT)")
    print("- Explore advanced contrastive losses (InfoNCE)")
    print("- Move to intermediate tutorial for CLIP-style learning")


if __name__ == "__main__":
    main()
