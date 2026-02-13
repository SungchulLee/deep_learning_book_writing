"""
Module 35: Multimodal Vision - Utility Functions
=================================================

Utility functions for multimodal vision-language learning:
- Data loading and preprocessing
- Text tokenization helpers
- Evaluation metrics
- Visualization tools
- Model saving/loading

These utilities support all three difficulty levels and can be used
with real datasets like MS-COCO, Flickr30K, etc.

Author: Educational AI
Date: 2024
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import json
import os


# ============================================================================
# PART 1: DATA PREPROCESSING
# ============================================================================

class SimpleTokenizer:
    """
    Simple tokenizer for educational purposes.
    
    In practice, use proper tokenizers like:
    - transformers.BertTokenizer
    - transformers.CLIPTokenizer
    - tiktoken
    
    This demonstrates the basic concepts of tokenization.
    """
    
    def __init__(self, vocab_size: int = 30000, max_length: int = 77):
        """
        Initialize tokenizer.
        
        Args:
            vocab_size: Size of vocabulary
            max_length: Maximum sequence length
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Special tokens
        self.pad_token = '[PAD]'
        self.mask_token = '[MASK]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        
        self.special_tokens = {
            self.pad_token: 0,
            self.cls_token: 1,
            self.sep_token: 2,
            self.mask_token: 3
        }
        
    def encode(self, text: str, add_special_tokens: bool = True) -> torch.Tensor:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
            add_special_tokens: Whether to add [CLS] and [SEP]
        
        Returns:
            Token IDs tensor
        """
        # Simple word-based tokenization (not subword)
        words = text.lower().split()
        
        # Convert words to IDs (hash-based for toy example)
        token_ids = [hash(word) % (self.vocab_size - 10) + 10 for word in words]
        
        # Add special tokens
        if add_special_tokens:
            token_ids = [self.special_tokens[self.cls_token]] + token_ids + \
                       [self.special_tokens[self.sep_token]]
        
        # Truncate if too long
        token_ids = token_ids[:self.max_length]
        
        # Pad if too short
        padding_length = self.max_length - len(token_ids)
        token_ids += [self.special_tokens[self.pad_token]] * padding_length
        
        return torch.tensor(token_ids, dtype=torch.long)
    
    def decode(self, token_ids: torch.Tensor) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: Token IDs tensor
        
        Returns:
            Decoded text string
        """
        # Remove special tokens and padding
        token_ids = token_ids.tolist()
        tokens = []
        
        for tid in token_ids:
            if tid == self.special_tokens[self.pad_token]:
                break
            if tid not in self.special_tokens.values():
                tokens.append(f"word_{tid}")
        
        return " ".join(tokens)


def preprocess_image_features(images: torch.Tensor, 
                              normalize: bool = True) -> torch.Tensor:
    """
    Preprocess image features for model input.
    
    Args:
        images: Image features (batch, channels, height, width) or (batch, features)
        normalize: Whether to L2 normalize
    
    Returns:
        Preprocessed features
    """
    # If 4D (image), flatten spatial dimensions
    if images.dim() == 4:
        batch, channels, height, width = images.shape
        images = images.view(batch, channels, -1).mean(dim=-1)
    
    # L2 normalize if requested
    if normalize:
        images = F.normalize(images, p=2, dim=-1)
    
    return images


def create_attention_mask(seq_lengths: List[int], max_length: int) -> torch.Tensor:
    """
    Create attention mask for variable-length sequences.
    
    Args:
        seq_lengths: List of actual sequence lengths
        max_length: Maximum sequence length
    
    Returns:
        Attention mask (batch, max_length)
        1 = attend, 0 = ignore
    """
    batch_size = len(seq_lengths)
    mask = torch.zeros(batch_size, max_length)
    
    for i, length in enumerate(seq_lengths):
        mask[i, :length] = 1
    
    return mask


# ============================================================================
# PART 2: EVALUATION METRICS
# ============================================================================

def compute_retrieval_metrics(similarities: torch.Tensor,
                              k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """
    Compute retrieval metrics (Recall@K, MRR, Median Rank).
    
    Args:
        similarities: Similarity matrix (num_queries, num_candidates)
                     similarities[i,j] = similarity between query i and candidate j
                     Correct match is on diagonal (i,i)
        k_values: List of K values for Recall@K
    
    Returns:
        Dictionary with metrics
    """
    num_queries = similarities.shape[0]
    
    metrics = {}
    
    # Compute ranks for each query
    ranks = []
    for i in range(num_queries):
        # Get similarities for query i
        sims = similarities[i, :]
        
        # Sort in descending order
        sorted_indices = torch.argsort(sims, descending=True)
        
        # Find rank of correct answer (index i)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)
    
    ranks = np.array(ranks)
    
    # Recall@K: fraction of queries with correct answer in top-K
    for k in k_values:
        recall = np.mean(ranks <= k)
        metrics[f'recall@{k}'] = float(recall)
    
    # Mean Reciprocal Rank: average of 1/rank
    mrr = np.mean(1.0 / ranks)
    metrics['mrr'] = float(mrr)
    
    # Median Rank
    median_rank = np.median(ranks)
    metrics['median_rank'] = float(median_rank)
    
    # Mean Rank
    mean_rank = np.mean(ranks)
    metrics['mean_rank'] = float(mean_rank)
    
    return metrics


def compute_classification_metrics(predictions: torch.Tensor,
                                   targets: torch.Tensor,
                                   k: int = 5) -> Dict[str, float]:
    """
    Compute classification metrics (accuracy, top-k accuracy).
    
    Args:
        predictions: Predicted logits or probabilities (batch, num_classes)
        targets: True labels (batch,)
        k: K for top-K accuracy
    
    Returns:
        Dictionary with metrics
    """
    # Top-1 accuracy
    pred_labels = torch.argmax(predictions, dim=1)
    top1_acc = (pred_labels == targets).float().mean().item()
    
    # Top-K accuracy
    _, topk_preds = torch.topk(predictions, k, dim=1)
    topk_correct = topk_preds.eq(targets.view(-1, 1).expand_as(topk_preds))
    topk_acc = topk_correct.any(dim=1).float().mean().item()
    
    return {
        'top1_accuracy': top1_acc,
        f'top{k}_accuracy': topk_acc
    }


def compute_diversity_metrics(embeddings: torch.Tensor) -> Dict[str, float]:
    """
    Compute diversity metrics for embeddings.
    
    Useful for analyzing embedding space quality:
    - High intra-class similarity (embeddings from same class are close)
    - High inter-class diversity (embeddings from different classes are far)
    
    Args:
        embeddings: Embeddings (num_samples, embed_dim)
    
    Returns:
        Dictionary with diversity metrics
    """
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Compute pairwise similarities
    similarities = torch.matmul(embeddings, embeddings.t())
    
    # Average similarity (how clustered overall)
    avg_similarity = similarities.mean().item()
    
    # Standard deviation (how diverse)
    std_similarity = similarities.std().item()
    
    # Remove diagonal (self-similarity)
    n = similarities.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=similarities.device)
    non_diag_sims = similarities[mask]
    
    avg_pairwise_sim = non_diag_sims.mean().item()
    
    return {
        'avg_similarity': avg_similarity,
        'std_similarity': std_similarity,
        'avg_pairwise_similarity': avg_pairwise_sim
    }


# ============================================================================
# PART 3: VISUALIZATION TOOLS
# ============================================================================

def plot_similarity_matrix(similarities: torch.Tensor,
                          title: str = "Similarity Matrix",
                          save_path: Optional[str] = None):
    """
    Visualize similarity matrix as heatmap.
    
    Useful for understanding retrieval performance and alignment quality.
    
    Args:
        similarities: Similarity matrix (M, N)
        title: Plot title
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 8))
    
    # Convert to numpy
    sim_np = similarities.cpu().numpy()
    
    # Create heatmap
    plt.imshow(sim_np, cmap='viridis', aspect='auto')
    plt.colorbar(label='Similarity')
    plt.title(title)
    plt.xlabel('Text Index')
    plt.ylabel('Image Index')
    
    # Mark diagonal (correct matches)
    n = min(sim_np.shape)
    plt.plot([0, n-1], [0, n-1], 'r--', linewidth=2, label='Correct Matches')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Similarity matrix saved to {save_path}")
    
    plt.show()


def plot_retrieval_examples(image_embeddings: torch.Tensor,
                           text_embeddings: torch.Tensor,
                           num_examples: int = 5,
                           save_path: Optional[str] = None):
    """
    Visualize retrieval examples (which texts retrieved for each image).
    
    Args:
        image_embeddings: Image embeddings (N, embed_dim)
        text_embeddings: Text embeddings (N, embed_dim)
        num_examples: Number of examples to show
        save_path: Optional path to save figure
    """
    # Compute similarities
    similarities = torch.matmul(image_embeddings, text_embeddings.t())
    
    fig, axes = plt.subplots(num_examples, 1, figsize=(12, 3*num_examples))
    
    for i in range(num_examples):
        ax = axes[i] if num_examples > 1 else axes
        
        # Get similarities for image i
        sims = similarities[i, :].cpu().numpy()
        
        # Plot
        ax.bar(range(len(sims)), sims, alpha=0.6)
        ax.axvline(x=i, color='r', linestyle='--', linewidth=2, label='Correct Match')
        ax.set_xlabel('Text Index')
        ax.set_ylabel('Similarity')
        ax.set_title(f'Image {i} Retrieval Scores')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Retrieval examples saved to {save_path}")
    
    plt.show()


def plot_embedding_space(embeddings: torch.Tensor,
                        labels: torch.Tensor,
                        method: str = 'pca',
                        title: str = "Embedding Space",
                        save_path: Optional[str] = None):
    """
    Visualize embedding space using dimensionality reduction.
    
    Args:
        embeddings: Embeddings (N, embed_dim)
        labels: Labels for coloring (N,)
        method: Dimensionality reduction method ('pca' or 'tsne')
        title: Plot title
        save_path: Optional path to save figure
    """
    from sklearn.decomposition import PCA
    
    # Convert to numpy
    emb_np = embeddings.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
    else:  # tsne
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
    
    emb_2d = reducer.fit_transform(emb_np)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], 
                         c=labels_np, cmap='tab10', 
                         alpha=0.6, s=50)
    plt.colorbar(scatter, label='Class')
    plt.title(title)
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Embedding space visualization saved to {save_path}")
    
    plt.show()


def plot_attention_weights(attention_weights: torch.Tensor,
                          query_labels: List[str],
                          key_labels: List[str],
                          title: str = "Attention Weights",
                          save_path: Optional[str] = None):
    """
    Visualize attention weights as heatmap.
    
    Useful for understanding what the model attends to.
    
    Args:
        attention_weights: Attention weights (num_queries, num_keys)
        query_labels: Labels for queries (e.g., word tokens)
        key_labels: Labels for keys (e.g., image regions)
        title: Plot title
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(12, 8))
    
    # Convert to numpy
    weights_np = attention_weights.cpu().numpy()
    
    # Create heatmap
    plt.imshow(weights_np, cmap='hot', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.title(title)
    
    # Set labels
    plt.xticks(range(len(key_labels)), key_labels, rotation=45, ha='right')
    plt.yticks(range(len(query_labels)), query_labels)
    plt.xlabel('Keys (What to attend to)')
    plt.ylabel('Queries (What is attending)')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention visualization saved to {save_path}")
    
    plt.show()


# ============================================================================
# PART 4: MODEL UTILITIES
# ============================================================================

def save_model(model: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               epoch: int,
               path: str,
               **kwargs):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        path: Save path
        **kwargs: Additional items to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        **kwargs
    }
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_model(model: torch.nn.Module,
               path: str,
               optimizer: Optional[torch.optim.Optimizer] = None,
               device: str = 'cpu') -> Dict:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        path: Checkpoint path
        optimizer: Optional optimizer to load state into
        device: Device to load to
    
    Returns:
        Dictionary with checkpoint info
    """
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {path}")
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
    
    return checkpoint


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: Model to analyze
    
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }


def log_metrics(metrics: Dict[str, float], 
               log_file: str = 'metrics.json'):
    """
    Log metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        log_file: Path to log file
    """
    # Load existing metrics if file exists
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            all_metrics = json.load(f)
    else:
        all_metrics = []
    
    # Append new metrics
    all_metrics.append(metrics)
    
    # Save
    with open(log_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"Metrics logged to {log_file}")


# ============================================================================
# PART 5: BATCH PROCESSING UTILITIES
# ============================================================================

def collate_multimodal_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for multimodal data.
    
    Handles variable-length sequences with padding.
    
    Args:
        batch: List of sample dictionaries
    
    Returns:
        Batched dictionary
    """
    # Initialize lists
    images = []
    texts = []
    labels = []
    
    # Collect data
    for sample in batch:
        images.append(sample['image'])
        texts.append(sample['text'])
        if 'label' in sample:
            labels.append(sample['label'])
    
    # Stack images (fixed size)
    images = torch.stack(images)
    
    # Pad text sequences to max length in batch
    max_len = max(t.shape[0] for t in texts)
    padded_texts = []
    
    for text in texts:
        if text.shape[0] < max_len:
            padding = torch.zeros(max_len - text.shape[0], *text.shape[1:])
            text = torch.cat([text, padding], dim=0)
        padded_texts.append(text)
    
    texts = torch.stack(padded_texts)
    
    result = {
        'images': images,
        'texts': texts
    }
    
    if labels:
        result['labels'] = torch.tensor(labels)
    
    return result


def create_data_splits(dataset: torch.utils.data.Dataset,
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.1,
                      test_ratio: float = 0.1,
                      random_seed: int = 42) -> Tuple:
    """
    Split dataset into train/val/test sets.
    
    Args:
        dataset: Full dataset
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_set, val_set, test_set)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
           "Ratios must sum to 1.0"
    
    # Set seed
    torch.manual_seed(random_seed)
    
    # Calculate sizes
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Split
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"Dataset split:")
    print(f"  Train: {len(train_set)} samples")
    print(f"  Val: {len(val_set)} samples")
    print(f"  Test: {len(test_set)} samples")
    
    return train_set, val_set, test_set


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """Demonstrate utility functions."""
    print("=" * 70)
    print("MULTIMODAL VISION UTILITIES - EXAMPLES")
    print("=" * 70)
    
    # Example 1: Tokenization
    print("\n1. Tokenization Example")
    print("-" * 70)
    tokenizer = SimpleTokenizer(vocab_size=1000, max_length=20)
    text = "a photo of a cat sitting on a couch"
    tokens = tokenizer.encode(text)
    print(f"Text: {text}")
    print(f"Tokens shape: {tokens.shape}")
    print(f"Decoded: {tokenizer.decode(tokens)}")
    
    # Example 2: Retrieval Metrics
    print("\n2. Retrieval Metrics Example")
    print("-" * 70)
    # Simulate similarity matrix (10 images vs 10 texts)
    similarities = torch.randn(10, 10)
    # Add some signal: diagonal should be higher
    similarities += torch.eye(10) * 5
    
    metrics = compute_retrieval_metrics(similarities, k_values=[1, 5])
    print("Retrieval Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Example 3: Visualization
    print("\n3. Visualization Example")
    print("-" * 70)
    embeddings = torch.randn(100, 512)
    labels = torch.randint(0, 5, (100,))
    
    print("Generating embedding space visualization...")
    plot_embedding_space(embeddings, labels, method='pca',
                        title="Example Embedding Space")
    
    # Example 4: Model Parameter Count
    print("\n4. Parameter Counting Example")
    print("-" * 70)
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 512)
    )
    params = count_parameters(model)
    print("Model Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value:,}")
    
    print("\n" + "=" * 70)
    print("Utility examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    example_usage()
