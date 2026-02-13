"""
Data Loading Module for Learning Rate Scheduler Demo
=====================================================

This module handles data generation and loading for the scheduler demonstration.
It creates synthetic classification data to focus on scheduler behavior rather
than data preprocessing complexities.

The synthetic data is designed to be:
- Realistic enough to train a neural network
- Simple enough to not distract from scheduler concepts
- Fast to generate
- Reproducible with random seeds
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from typing import Tuple


# ============================================================================
# SYNTHETIC DATASET CLASS
# ============================================================================
class SyntheticDataset(Dataset):
    """
    PyTorch Dataset for synthetic classification data.
    
    This dataset generates random features with some structure that makes
    classification possible but not trivial. The data is generated such that:
    - Features have varying importance (some are more informative)
    - Classes have distinct but overlapping distributions
    - The problem is challenging enough to observe training dynamics
    
    Attributes:
        features (torch.Tensor): Input features, shape (n_samples, input_dim)
        labels (torch.Tensor): Class labels, shape (n_samples,)
    """
    
    def __init__(self, n_samples: int, input_dim: int, num_classes: int, seed: int = 42):
        """
        Initialize the synthetic dataset.
        
        Args:
            n_samples (int): Number of samples to generate
            input_dim (int): Number of features per sample
            num_classes (int): Number of classes for classification
            seed (int): Random seed for reproducibility
        
        The data generation process:
        1. Create base random features
        2. Add class-specific patterns to make classification possible
        3. Add noise to make the problem realistic
        4. Normalize features for stable training
        """
        super().__init__()
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # ====================================================================
        # GENERATE LABELS
        # ====================================================================
        # Create balanced classes (equal number of samples per class)
        self.labels = torch.from_numpy(
            np.random.randint(0, num_classes, size=n_samples)
        ).long()
        
        # ====================================================================
        # GENERATE FEATURES
        # ====================================================================
        # Start with random Gaussian features
        # Shape: (n_samples, input_dim)
        features = np.random.randn(n_samples, input_dim).astype(np.float32)
        
        # ====================================================================
        # ADD CLASS-SPECIFIC PATTERNS
        # ====================================================================
        # Make the classification task learnable by adding patterns
        # Each class gets a distinct signature in feature space
        
        for class_id in range(num_classes):
            # Get indices of samples belonging to this class
            class_mask = (self.labels == class_id).numpy()
            
            # Add class-specific shift to first few features
            # This creates separability between classes
            shift_dim = min(5, input_dim)  # Use first 5 features for class info
            shift_magnitude = 2.0  # Magnitude of the shift
            
            # Each class gets a different shift direction
            shift_pattern = np.random.randn(shift_dim) * shift_magnitude
            features[class_mask, :shift_dim] += shift_pattern
            
            # Add some class-specific correlation structure
            # This makes the problem more realistic
            if input_dim > 10:
                # Create correlated features for each class
                correlation_matrix = np.random.randn(input_dim, input_dim) * 0.5
                features[class_mask] += features[class_mask] @ correlation_matrix.T * 0.1
        
        # ====================================================================
        # ADD NOISE
        # ====================================================================
        # Add observation noise to make the problem realistic
        # Without noise, the problem would be too easy
        noise_level = 0.5
        features += np.random.randn(n_samples, input_dim).astype(np.float32) * noise_level
        
        # ====================================================================
        # NORMALIZE FEATURES
        # ====================================================================
        # Standardize features to have mean=0, std=1
        # This is important for neural network training stability
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        # Convert to PyTorch tensor
        self.features = torch.from_numpy(features).float()
        
        print(f"Generated synthetic dataset:")
        print(f"  Samples: {n_samples}")
        print(f"  Features: {input_dim}")
        print(f"  Classes: {num_classes}")
        print(f"  Feature shape: {self.features.shape}")
        print(f"  Label shape: {self.labels.shape}")
    
    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
        
        Returns:
            tuple: (features, label) where
                   features is a tensor of shape (input_dim,)
                   label is an integer class label
        """
        return self.features[idx], self.labels[idx]


# ============================================================================
# DATA LOADER BUILDER
# ============================================================================
def build_dataloaders(
    n_samples: int = 1000,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    input_dim: int = 20,
    num_classes: int = 10,
    seed: int = 42,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Build training and validation data loaders.
    
    This function:
    1. Creates a synthetic dataset
    2. Splits it into training and validation sets
    3. Creates DataLoader objects for batch iteration
    4. Calculates steps per epoch (useful for some schedulers)
    
    Args:
        n_samples (int): Total number of samples to generate
        batch_size (int): Number of samples per batch
        val_ratio (float): Fraction of data for validation (0-1)
        input_dim (int): Number of input features
        num_classes (int): Number of output classes
        seed (int): Random seed for reproducibility
        num_workers (int): Number of parallel workers for data loading
    
    Returns:
        tuple: (train_loader, val_loader, steps_per_epoch) where
               train_loader: DataLoader for training data
               val_loader: DataLoader for validation data
               steps_per_epoch: Number of batches in one training epoch
    
    Example:
        >>> train_loader, val_loader, steps = build_dataloaders(
        ...     n_samples=1000, batch_size=32, val_ratio=0.2
        ... )
        >>> print(f"Training batches per epoch: {steps}")
        >>> for features, labels in train_loader:
        ...     # Training loop here
        ...     break
    """
    
    print("\n" + "="*70)
    print("BUILDING DATA LOADERS")
    print("="*70)
    
    # ========================================================================
    # CREATE SYNTHETIC DATASET
    # ========================================================================
    print("\nGenerating synthetic dataset...")
    full_dataset = SyntheticDataset(
        n_samples=n_samples,
        input_dim=input_dim,
        num_classes=num_classes,
        seed=seed
    )
    
    # ========================================================================
    # SPLIT INTO TRAIN AND VALIDATION
    # ========================================================================
    print("\nSplitting into train and validation sets...")
    
    # Calculate split sizes
    val_size = int(n_samples * val_ratio)
    train_size = n_samples - val_size
    
    # Use random_split to split the dataset
    # The generator ensures reproducibility
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    # ========================================================================
    # CREATE DATA LOADERS
    # ========================================================================
    print("\nCreating data loaders...")
    
    # Training data loader
    # - shuffle=True: Randomize sample order each epoch (prevents overfitting)
    # - drop_last=False: Keep the last incomplete batch
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # Faster GPU transfer if available
        drop_last=False
    )
    
    # Validation data loader
    # - shuffle=False: No need to shuffle validation data
    # - drop_last=False: Evaluate on all validation samples
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    # ========================================================================
    # CALCULATE STEPS PER EPOCH
    # ========================================================================
    # This is the number of batches in one complete training epoch
    # Needed for some schedulers like OneCycleLR and CyclicLR
    steps_per_epoch = len(train_loader)
    
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    
    # ========================================================================
    # VERIFY DATA LOADER FUNCTIONALITY
    # ========================================================================
    print("\nVerifying data loaders...")
    
    # Get a sample batch to verify everything works
    try:
        sample_features, sample_labels = next(iter(train_loader))
        print(f"  Sample batch shape: {sample_features.shape}")
        print(f"  Sample labels shape: {sample_labels.shape}")
        print(f"  Feature dtype: {sample_features.dtype}")
        print(f"  Label dtype: {sample_labels.dtype}")
        print(f"  Label range: [{sample_labels.min().item()}, {sample_labels.max().item()}]")
        
        # Verify data is properly normalized
        print(f"  Feature mean: {sample_features.mean().item():.4f}")
        print(f"  Feature std: {sample_features.std().item():.4f}")
        
    except Exception as e:
        print(f"  ERROR: Failed to load sample batch: {e}")
        raise
    
    print("="*70)
    
    return train_loader, val_loader, steps_per_epoch


# ============================================================================
# VISUALIZATION HELPER (Optional)
# ============================================================================
def visualize_sample_data(data_loader: DataLoader, num_samples: int = 5):
    """
    Print sample data from the data loader for inspection.
    
    This is useful for debugging and understanding the data.
    
    Args:
        data_loader (DataLoader): Data loader to sample from
        num_samples (int): Number of samples to display
    """
    print("\n" + "="*70)
    print("SAMPLE DATA VISUALIZATION")
    print("="*70)
    
    features, labels = next(iter(data_loader))
    
    for i in range(min(num_samples, len(features))):
        print(f"\nSample {i+1}:")
        print(f"  Label: {labels[i].item()}")
        print(f"  Features (first 10): {features[i, :10].tolist()}")
        print(f"  Features mean: {features[i].mean().item():.4f}")
        print(f"  Features std: {features[i].std().item():.4f}")
    
    print("="*70)


# ============================================================================
# MODULE TEST
# ============================================================================
if __name__ == '__main__':
    """
    Test the data loader module.
    
    Run this file directly to test data generation:
        python -m scheduler.data_loader
    """
    print("Testing data loader module...\n")
    
    # Build data loaders with default parameters
    train_loader, val_loader, steps_per_epoch = build_dataloaders(
        n_samples=500,
        batch_size=32,
        val_ratio=0.2,
        input_dim=20,
        num_classes=10,
        seed=42
    )
    
    # Visualize some samples
    visualize_sample_data(train_loader, num_samples=3)
    
    # Test iteration
    print("\nTesting data loader iteration...")
    for batch_idx, (features, labels) in enumerate(train_loader):
        if batch_idx == 0:
            print(f"First batch:")
            print(f"  Features shape: {features.shape}")
            print(f"  Labels shape: {labels.shape}")
        if batch_idx >= 2:  # Just test a few batches
            break
    
    print("\nData loader module test completed successfully!")
