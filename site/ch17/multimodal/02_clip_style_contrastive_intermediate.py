"""
Module 35: Multimodal Vision - Intermediate Level
==================================================

Topic: CLIP-Style Contrastive Learning and Zero-Shot Classification

This script implements a CLIP-inspired model with proper InfoNCE contrastive loss,
temperature scaling, and zero-shot image classification capabilities.

Key Concepts:
1. InfoNCE (Noise Contrastive Estimation) loss
2. Temperature-scaled similarities
3. Symmetric contrastive learning (both directions)
4. Large batch training for more negatives
5. Zero-shot classification via text prompts
6. Bidirectional retrieval (I2T and T2I)

Learning Objectives:
- Understand InfoNCE loss and its connection to mutual information
- Implement temperature scaling for better training
- Build zero-shot classifiers using text descriptions
- Evaluate with standard retrieval metrics

Mathematical Background:
- InfoNCE Loss: L = -log(exp(sim(x,x+)/τ) / Σ_i exp(sim(x,x_i)/τ))
- Temperature τ controls the sharpness of the distribution
- Symmetric loss combines image-to-text and text-to-image directions

Author: Educational AI
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


# ============================================================================
# PART 1: IMPROVED ENCODERS WITH BETTER ARCHITECTURE
# ============================================================================

class ImageEncoder(nn.Module):
    """
    Improved image encoder with deeper architecture.
    
    Simulates a CNN backbone (like ResNet) followed by projection head.
    In practice, you'd use actual pretrained vision models.
    
    Architecture:
    - 3-layer MLP with BatchNorm and Dropout
    - Projects image features to embedding space
    - Final L2 normalization
    
    Args:
        input_dim: Input feature dimension (e.g., 2048 for ResNet-50)
        hidden_dims: List of hidden layer dimensions
        embed_dim: Output embedding dimension
        dropout_p: Dropout probability
    """
    
    def __init__(self, input_dim: int = 2048, hidden_dims: List[int] = [2048, 1024], 
                 embed_dim: int = 512, dropout_p: float = 0.1):
        super(ImageEncoder, self).__init__()
        
        # Build multi-layer encoder
        layers = []
        prev_dim = input_dim
        
        # Hidden layers with BatchNorm and Dropout
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),  # Normalize activations
                nn.ReLU(),
                nn.Dropout(dropout_p)
            ])
            prev_dim = hidden_dim
        
        # Final projection to embedding space (no activation)
        layers.append(nn.Linear(prev_dim, embed_dim))
        
        # Combine all layers
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to normalized embeddings.
        
        Args:
            x: Image features (batch_size, input_dim)
        
        Returns:
            L2-normalized embeddings (batch_size, embed_dim)
        """
        # Pass through encoder network
        embeddings = self.encoder(x)
        
        # L2 normalize for cosine similarity
        # After normalization: ||embeddings|| = 1
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class TextEncoder(nn.Module):
    """
    Improved text encoder with deeper architecture.
    
    Simulates a language model (like BERT) followed by projection head.
    In practice, you'd use actual pretrained language models.
    
    Args:
        input_dim: Input feature dimension (e.g., 768 for BERT-base)
        hidden_dims: List of hidden layer dimensions
        embed_dim: Output embedding dimension
        dropout_p: Dropout probability
    """
    
    def __init__(self, input_dim: int = 768, hidden_dims: List[int] = [768, 1024], 
                 embed_dim: int = 512, dropout_p: float = 0.1):
        super(TextEncoder, self).__init__()
        
        # Same architecture as image encoder for symmetry
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, embed_dim))
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode text to normalized embeddings."""
        embeddings = self.encoder(x)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


# ============================================================================
# PART 2: CLIP MODEL WITH LEARNABLE TEMPERATURE
# ============================================================================

class CLIPModel(nn.Module):
    """
    CLIP-style model with dual encoders and learnable temperature.
    
    CLIP (Contrastive Language-Image Pre-training) learns a joint embedding
    space by maximizing similarity between matched image-text pairs and
    minimizing similarity for mismatched pairs using InfoNCE loss.
    
    Key Features:
    1. Separate encoders for vision and language
    2. Learnable temperature parameter (initialized to 1/0.07 ≈ 14.29)
    3. Symmetric contrastive loss (both I2T and T2I)
    4. Zero-shot classification via text prompts
    
    The temperature parameter τ is crucial:
    - Low τ (high 1/τ): Sharp distribution, confident predictions
    - High τ (low 1/τ): Smooth distribution, uncertain predictions
    - CLIP uses τ=0.07 which works well empirically
    
    Args:
        image_dim: Input dimension for image features
        text_dim: Input dimension for text features
        embed_dim: Shared embedding dimension
        hidden_dims_img: Hidden dimensions for image encoder
        hidden_dims_txt: Hidden dimensions for text encoder
        init_temperature: Initial temperature value
    """
    
    def __init__(self, image_dim: int = 2048, text_dim: int = 768,
                 embed_dim: int = 512, hidden_dims_img: List[int] = [2048, 1024],
                 hidden_dims_txt: List[int] = [768, 1024],
                 init_temperature: float = 0.07):
        super(CLIPModel, self).__init__()
        
        # Initialize encoders
        self.image_encoder = ImageEncoder(image_dim, hidden_dims_img, embed_dim)
        self.text_encoder = TextEncoder(text_dim, hidden_dims_txt, embed_dim)
        
        # Learnable temperature parameter
        # We learn log(temperature) for numerical stability
        # Temperature is constrained to be positive via exp()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / init_temperature))
        
        # Store embedding dimension
        self.embed_dim = embed_dim
        
    def forward(self, images: torch.Tensor, texts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images and texts to shared embedding space.
        
        Args:
            images: Image features (batch_size, image_dim)
            texts: Text features (batch_size, text_dim)
        
        Returns:
            Tuple of (image_embeddings, text_embeddings)
        """
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(texts)
        return image_features, text_features
    
    def compute_logits(self, image_features: torch.Tensor, 
                      text_features: torch.Tensor) -> torch.Tensor:
        """
        Compute temperature-scaled similarity logits.
        
        The logits represent how well each image matches each text.
        Temperature scaling controls the confidence of predictions.
        
        Mathematically:
            logits[i,j] = (1/τ) * sim(image_i, text_j)
            where sim() is cosine similarity (dot product for normalized vectors)
        
        Args:
            image_features: L2-normalized image embeddings (N, embed_dim)
            text_features: L2-normalized text embeddings (M, embed_dim)
        
        Returns:
            Logits matrix (N, M)
        """
        # Get current temperature (convert from log space)
        # We clamp to prevent extreme values
        logit_scale = self.logit_scale.exp()
        logit_scale = torch.clamp(logit_scale, max=100)  # Prevent explosion
        
        # Compute cosine similarity (dot product for normalized vectors)
        # Shape: (N, embed_dim) @ (embed_dim, M) = (N, M)
        logits = logit_scale * torch.matmul(image_features, text_features.t())
        
        return logits
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Convenience method to encode only images."""
        return self.image_encoder(images)
    
    def encode_text(self, texts: torch.Tensor) -> torch.Tensor:
        """Convenience method to encode only texts."""
        return self.text_encoder(texts)
    
    def zero_shot_classifier(self, image_features: torch.Tensor,
                            text_features: torch.Tensor) -> torch.Tensor:
        """
        Perform zero-shot classification using text descriptions.
        
        Given an image and K text descriptions (one per class),
        compute probability distribution over classes.
        
        Process:
        1. Compute similarity between image and each text description
        2. Apply softmax to get probability distribution
        3. Class with highest probability is the prediction
        
        Args:
            image_features: Image embeddings (batch_size, embed_dim)
            text_features: Text embeddings for K classes (K, embed_dim)
        
        Returns:
            Class probabilities (batch_size, K)
        """
        # Compute logits
        # Shape: (batch_size, num_classes)
        logits = self.compute_logits(image_features, text_features)
        
        # Apply softmax to get probabilities
        # P(class=k | image) = exp(logit_k) / Σ_j exp(logit_j)
        probs = F.softmax(logits, dim=1)
        
        return probs


# ============================================================================
# PART 3: CONTRASTIVE LOSS (InfoNCE)
# ============================================================================

class ContrastiveLoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) loss for contrastive learning.
    
    This is the core loss function used in CLIP. It's designed to:
    1. Maximize similarity for positive (matched) pairs
    2. Minimize similarity for negative (mismatched) pairs
    
    Mathematical Formulation:
    For a batch of N image-text pairs, each image has:
    - 1 positive text (the matched one)
    - N-1 negative texts (all others in the batch)
    
    Image-to-Text Loss:
        L_i2t = -log(exp(sim(img_i, txt_i)/τ) / Σ_j exp(sim(img_i, txt_j)/τ))
    
    Text-to-Image Loss:
        L_t2i = -log(exp(sim(txt_i, img_i)/τ) / Σ_j exp(sim(txt_i, img_j)/τ))
    
    Symmetric Loss:
        L = (L_i2t + L_t2i) / 2
    
    This is equivalent to cross-entropy with targets on the diagonal!
    
    Connection to Mutual Information:
    InfoNCE maximizes a lower bound on mutual information I(image; text)
    between matched pairs. Larger batches give tighter bounds.
    
    Args:
        temperature: Temperature parameter (not used if model has learnable temp)
    """
    
    def __init__(self, temperature: float = 0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute symmetric InfoNCE loss.
        
        Args:
            logits: Temperature-scaled similarity matrix (batch_size, batch_size)
                   logits[i,j] = (1/τ) * similarity(image_i, text_j)
        
        Returns:
            Scalar loss value
        
        Implementation Notes:
        - Positive pairs are on the diagonal (i,i)
        - We use cross_entropy which is numerically stable
        - Loss is averaged over both directions (I2T and T2I)
        """
        batch_size = logits.shape[0]
        
        # Create labels: positive pairs are on diagonal
        # Labels: [0, 1, 2, ..., batch_size-1]
        # This means: for row i, the correct column is i (diagonal)
        labels = torch.arange(batch_size, device=logits.device)
        
        # Image-to-Text loss
        # For each image (row), predict which text (column) is the match
        # Cross entropy: -log(softmax(logits)[i, labels[i]])
        # This is exactly the InfoNCE formula!
        loss_i2t = F.cross_entropy(logits, labels)
        
        # Text-to-Image loss  
        # For each text (column), predict which image (row) is the match
        # We transpose logits so texts become rows
        loss_t2i = F.cross_entropy(logits.t(), labels)
        
        # Symmetric loss (average both directions)
        # This ensures both encoders learn equally well
        loss = (loss_i2t + loss_t2i) / 2
        
        return loss


# ============================================================================
# PART 4: ENHANCED DATASET
# ============================================================================

class EnhancedMultimodalDataset(Dataset):
    """
    Enhanced toy dataset with more realistic structure.
    
    Improvements over beginner version:
    1. More classes (10 instead of 5)
    2. Intra-class variation to simulate real data
    3. Clear semantic structure
    
    In practice, use real datasets like:
    - MS-COCO: 330K images with 5 captions each
    - Flickr30K: 31K images with 5 captions each
    - Conceptual Captions: 3.3M image-text pairs
    
    Args:
        num_samples: Samples per class
        image_dim: Image feature dimension
        text_dim: Text feature dimension
        num_classes: Number of semantic classes
    """
    
    def __init__(self, num_samples: int = 100, image_dim: int = 2048,
                 text_dim: int = 768, num_classes: int = 10):
        super(EnhancedMultimodalDataset, self).__init__()
        
        self.num_samples = num_samples
        self.num_classes = num_classes
        
        # Generate data with clear class structure
        self.image_features = []
        self.text_features = []
        self.labels = []
        
        for class_idx in range(num_classes):
            # Each class has distinct cluster centers
            # Scale centers to make classes well-separated
            image_center = torch.randn(image_dim) * 3
            text_center = torch.randn(text_dim) * 3
            
            for _ in range(num_samples):
                # Add moderate noise for within-class variation
                # This simulates different instances of same concept
                image = image_center + torch.randn(image_dim) * 0.8
                text = text_center + torch.randn(text_dim) * 0.8
                
                self.image_features.append(image)
                self.text_features.append(text)
                self.labels.append(class_idx)
        
        # Convert to tensors
        self.image_features = torch.stack(self.image_features)
        self.text_features = torch.stack(self.text_features)
        self.labels = torch.tensor(self.labels)
        
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        return self.image_features[idx], self.text_features[idx], self.labels[idx]


# ============================================================================
# PART 5: TRAINING WITH PROGRESS TRACKING
# ============================================================================

def train_clip_model(model: CLIPModel, train_loader: DataLoader,
                    num_epochs: int = 20, learning_rate: float = 3e-4,
                    warmup_epochs: int = 1, device: str = 'cpu') -> dict:
    """
    Train CLIP model with InfoNCE loss and learning rate warmup.
    
    Training Details:
    1. Adam optimizer with cosine annealing schedule
    2. Learning rate warmup for stable initialization
    3. Gradient clipping to prevent exploding gradients
    4. Track multiple metrics during training
    
    Args:
        model: CLIPModel to train
        train_loader: DataLoader with training data
        num_epochs: Number of training epochs
        learning_rate: Peak learning rate (after warmup)
        warmup_epochs: Epochs for linear warmup
        device: Device for training
    
    Returns:
        Dictionary with training history
    """
    model = model.to(device)
    
    # Optimizer (Adam with decoupled weight decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, 
                                  weight_decay=0.1)
    
    # Learning rate scheduler (cosine annealing after warmup)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs
    )
    
    # Loss function
    criterion = ContrastiveLoss()
    
    # Training history
    history = {
        'train_loss': [],
        'learning_rate': [],
        'temperature': []
    }
    
    print("=" * 70)
    print("TRAINING CLIP MODEL")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Warmup epochs: {warmup_epochs}")
    print(f"Batch size: {train_loader.batch_size}")
    print("=" * 70)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Progress bar for this epoch
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for images, texts, labels in pbar:
            images = images.to(device)
            texts = texts.to(device)
            
            # Learning rate warmup
            # Linearly increase LR from 0 to target over warmup_epochs
            if epoch < warmup_epochs:
                warmup_lr = learning_rate * (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            image_features, text_features = model(images, texts)
            
            # Compute temperature-scaled logits
            logits = model.compute_logits(image_features, text_features)
            
            # Compute InfoNCE loss
            loss = criterion(logits)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Track statistics
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'temp': f'{(1/model.logit_scale.exp().item()):.4f}'
            })
        
        # Learning rate scheduling (after warmup)
        if epoch >= warmup_epochs:
            scheduler.step()
        
        # Compute epoch statistics
        avg_loss = epoch_loss / num_batches
        current_lr = optimizer.param_groups[0]['lr']
        current_temp = 1 / model.logit_scale.exp().item()
        
        # Save to history
        history['train_loss'].append(avg_loss)
        history['learning_rate'].append(current_lr)
        history['temperature'].append(current_temp)
        
        # Print epoch summary
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Temperature: {current_temp:.4f}")
        print("-" * 70)
    
    print("\nTraining completed!")
    return history


# ============================================================================
# PART 6: COMPREHENSIVE EVALUATION
# ============================================================================

def evaluate_clip(model: CLIPModel, test_loader: DataLoader,
                 device: str = 'cpu') -> dict:
    """
    Comprehensive evaluation of CLIP model.
    
    Evaluation Tasks:
    1. Image-to-Text Retrieval (I2T): Find matching text for each image
    2. Text-to-Image Retrieval (T2I): Find matching image for each text
    3. Zero-Shot Classification: Classify images using text prompts
    
    Metrics:
    - Recall@K: Percentage of queries with correct result in top-K
    - Mean Reciprocal Rank (MRR): Average of 1/rank for correct results
    - Median Rank: Median position of correct result
    
    Args:
        model: Trained CLIPModel
        test_loader: Test data loader
        device: Computation device
    
    Returns:
        Dictionary with all evaluation metrics
    """
    model.eval()
    
    print("=" * 70)
    print("EVALUATING CLIP MODEL")
    print("=" * 70)
    
    # Collect all embeddings
    all_image_emb = []
    all_text_emb = []
    all_labels = []
    
    print("\nEncoding all test samples...")
    with torch.no_grad():
        for images, texts, labels in tqdm(test_loader):
            images = images.to(device)
            texts = texts.to(device)
            
            img_emb, txt_emb = model(images, texts)
            
            all_image_emb.append(img_emb.cpu())
            all_text_emb.append(txt_emb.cpu())
            all_labels.append(labels)
    
    # Concatenate all embeddings
    all_image_emb = torch.cat(all_image_emb, dim=0)
    all_text_emb = torch.cat(all_text_emb, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    num_samples = len(all_image_emb)
    
    # Compute full similarity matrix
    print("\nComputing similarity matrix...")
    similarity = torch.matmul(all_image_emb, all_text_emb.t())
    
    # ========== Retrieval Metrics ==========
    print("\nComputing retrieval metrics...")
    
    metrics = {}
    
    # Image-to-Text Retrieval
    for k in [1, 5, 10]:
        recall = 0
        mrr_sum = 0
        ranks = []
        
        for i in range(num_samples):
            # Get similarities and sort
            sims = similarity[i, :]
            sorted_indices = torch.argsort(sims, descending=True)
            
            # Find rank of correct text (index i)
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)
            
            # Recall@K: is correct result in top-K?
            if rank <= k:
                recall += 1
            
            # MRR: 1/rank
            mrr_sum += 1.0 / rank
        
        metrics[f'i2t_recall@{k}'] = recall / num_samples
        
        if k == 1:
            metrics['i2t_mrr'] = mrr_sum / num_samples
            metrics['i2t_median_rank'] = float(np.median(ranks))
    
    # Text-to-Image Retrieval
    for k in [1, 5, 10]:
        recall = 0
        mrr_sum = 0
        ranks = []
        
        for i in range(num_samples):
            sims = similarity[:, i]
            sorted_indices = torch.argsort(sims, descending=True)
            
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)
            
            if rank <= k:
                recall += 1
            
            mrr_sum += 1.0 / rank
        
        metrics[f't2i_recall@{k}'] = recall / num_samples
        
        if k == 1:
            metrics['t2i_mrr'] = mrr_sum / num_samples
            metrics['t2i_median_rank'] = float(np.median(ranks))
    
    # ========== Zero-Shot Classification ==========
    print("\nEvaluating zero-shot classification...")
    
    # Group by class to get "text prompts" for each class
    unique_labels = torch.unique(all_labels)
    num_classes = len(unique_labels)
    
    # Average text embeddings per class (simulate class descriptions)
    class_text_features = []
    for label in unique_labels:
        class_mask = (all_labels == label)
        class_texts = all_text_emb[class_mask]
        # Average all text embeddings for this class
        class_text = class_texts.mean(dim=0)
        # Re-normalize
        class_text = F.normalize(class_text.unsqueeze(0), p=2, dim=1)
        class_text_features.append(class_text)
    
    class_text_features = torch.cat(class_text_features, dim=0)
    
    # Classify each image
    correct = 0
    top5_correct = 0
    
    for i in range(num_samples):
        img_emb = all_image_emb[i:i+1]
        true_label = all_labels[i].item()
        
        # Compute similarities to all class descriptions
        # Shape: (1, num_classes)
        logits = torch.matmul(img_emb, class_text_features.t())
        
        # Top-1 prediction
        pred_label = torch.argmax(logits, dim=1).item()
        if pred_label == true_label:
            correct += 1
        
        # Top-5 prediction
        _, top5_preds = torch.topk(logits, min(5, num_classes), dim=1)
        if true_label in top5_preds[0]:
            top5_correct += 1
    
    metrics['zero_shot_top1_acc'] = correct / num_samples
    metrics['zero_shot_top5_acc'] = top5_correct / num_samples
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    print("\nImage-to-Text Retrieval:")
    print(f"  Recall@1:  {metrics['i2t_recall@1']:.4f} ({metrics['i2t_recall@1']*100:.2f}%)")
    print(f"  Recall@5:  {metrics['i2t_recall@5']:.4f} ({metrics['i2t_recall@5']*100:.2f}%)")
    print(f"  Recall@10: {metrics['i2t_recall@10']:.4f} ({metrics['i2t_recall@10']*100:.2f}%)")
    print(f"  MRR:       {metrics['i2t_mrr']:.4f}")
    print(f"  Median Rank: {metrics['i2t_median_rank']:.1f}")
    
    print("\nText-to-Image Retrieval:")
    print(f"  Recall@1:  {metrics['t2i_recall@1']:.4f} ({metrics['t2i_recall@1']*100:.2f}%)")
    print(f"  Recall@5:  {metrics['t2i_recall@5']:.4f} ({metrics['t2i_recall@5']*100:.2f}%)")
    print(f"  Recall@10: {metrics['t2i_recall@10']:.4f} ({metrics['t2i_recall@10']*100:.2f}%)")
    print(f"  MRR:       {metrics['t2i_mrr']:.4f}")
    print(f"  Median Rank: {metrics['t2i_median_rank']:.1f}")
    
    print("\nZero-Shot Classification:")
    print(f"  Top-1 Accuracy: {metrics['zero_shot_top1_acc']:.4f} ({metrics['zero_shot_top1_acc']*100:.2f}%)")
    print(f"  Top-5 Accuracy: {metrics['zero_shot_top5_acc']:.4f} ({metrics['zero_shot_top5_acc']*100:.2f}%)")
    
    print("=" * 70)
    
    return metrics


# ============================================================================
# PART 7: VISUALIZATION
# ============================================================================

def visualize_training_history(history: dict):
    """Visualize training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss curve
    axes[0].plot(history['train_loss'], marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True)
    
    # Learning rate curve
    axes[1].plot(history['learning_rate'], marker='o', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].grid(True)
    
    # Temperature curve
    axes[2].plot(history['temperature'], marker='o', color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Temperature')
    axes[2].set_title('Learned Temperature')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('clip_training_history.png', dpi=300, bbox_inches='tight')
    print("\nTraining history saved as 'clip_training_history.png'")
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function demonstrating CLIP-style learning."""
    print("=" * 70)
    print("CLIP-STYLE CONTRASTIVE LEARNING - INTERMEDIATE TUTORIAL")
    print("=" * 70)
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Hyperparameters
    config = {
        'image_dim': 2048,
        'text_dim': 768,
        'embed_dim': 512,
        'batch_size': 64,  # Larger batch = more negatives = better contrastive learning
        'num_epochs': 30,
        'learning_rate': 3e-4,
        'warmup_epochs': 2,
        'num_classes': 10
    }
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = EnhancedMultimodalDataset(
        num_samples=100,
        image_dim=config['image_dim'],
        text_dim=config['text_dim'],
        num_classes=config['num_classes']
    )
    
    test_dataset = EnhancedMultimodalDataset(
        num_samples=20,
        image_dim=config['image_dim'],
        text_dim=config['text_dim'],
        num_classes=config['num_classes']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                            shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model
    print("\nInitializing CLIP model...")
    model = CLIPModel(
        image_dim=config['image_dim'],
        text_dim=config['text_dim'],
        embed_dim=config['embed_dim'],
        hidden_dims_img=[2048, 1024],
        hidden_dims_txt=[768, 1024],
        init_temperature=0.07
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train model
    history = train_clip_model(
        model=model,
        train_loader=train_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        warmup_epochs=config['warmup_epochs'],
        device=device
    )
    
    # Visualize training
    visualize_training_history(history)
    
    # Evaluate model
    metrics = evaluate_clip(model, test_loader, device)
    
    print("\n" + "=" * 70)
    print("TUTORIAL COMPLETED!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. InfoNCE loss maximizes mutual information between matched pairs")
    print("2. Temperature scaling controls prediction confidence")
    print("3. Symmetric loss ensures both encoders learn well")
    print("4. Larger batches provide more negatives for better learning")
    print("5. Zero-shot classification works via text prompt matching")
    print("\nNext Steps:")
    print("- Train on real datasets (MS-COCO, Conceptual Captions)")
    print("- Use pretrained encoders (ResNet, BERT)")
    print("- Try different architectures (ViT, GPT)")
    print("- Explore applications (image search, VQA)")
    print("- Move to advanced tutorial for cross-attention models")


if __name__ == "__main__":
    main()
