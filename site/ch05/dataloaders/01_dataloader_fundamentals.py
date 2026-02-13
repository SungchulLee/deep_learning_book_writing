#!/usr/bin/env python3
"""
=============================================================================
DataLoader Fundamentals: The Core Concept
=============================================================================

WHAT IS A DATALOADER?
---------------------
DataLoader is PyTorch's batch-producing iterable that wraps around a Dataset.
It handles:
  • Batching: Groups individual samples into mini-batches
  • Shuffling: Randomizes data order each epoch
  • Parallelization: Loads data using multiple worker processes
  • Memory pinning: Speeds up CPU → GPU transfers

KEY CONCEPTS:
------------
1. Dataset: Storage/retrieval of individual samples (implements __getitem__, __len__)
2. DataLoader: Iterates over Dataset, producing batches
3. Batch: Collection of samples grouped together for efficient computation

LEARNING OBJECTIVES:
-------------------
✓ Understand the Dataset → DataLoader → Batch pipeline
✓ Learn how batch_size affects iteration
✓ See the effect of shuffle parameter
✓ Master the iteration protocol (iter/next vs for-loop)

DIFFICULTY: ⭐ Beginner
TIME: 10 minutes
"""

import torch
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# STEP 1: Create a Simple Dataset
# =============================================================================
class SimpleDataset(Dataset):
    """
    A minimal custom dataset for demonstration.
    
    **Dataset Requirements:**
    Every Dataset must implement:
      - __len__(): Returns total number of samples
      - __getitem__(idx): Returns the sample at index 'idx'
    
    **This Example:**
    Creates synthetic data with 12 samples, each having:
      - X: 4 features (random numbers)
      - y: binary label (positive if sum of features > 0)
    """
    
    def __init__(self, num_samples=12, seed=0):
        """
        Initialize the dataset with synthetic data.
        
        Args:
            num_samples: Total number of samples to generate
            seed: Random seed for reproducibility
        """
        # Use a Generator for reproducible random numbers
        gen = torch.Generator().manual_seed(seed)
        
        # Generate features: [num_samples, 4] tensor of random values
        self.X = torch.randn(num_samples, 4, generator=gen)
        
        # Generate labels: 1 if sum > 0, else 0
        # This creates a binary classification problem
        self.y = (self.X.sum(dim=1) > 0).long()  # Shape: [num_samples]
        
        print(f"✓ Dataset created with {num_samples} samples")
        print(f"  - Features shape: {self.X.shape}")
        print(f"  - Labels shape: {self.y.shape}")
        print(f"  - Positive samples: {self.y.sum().item()}/{num_samples}")
    
    def __len__(self):
        """
        Required method: Returns the total number of samples.
        
        This is used by DataLoader to know when to stop iterating.
        """
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        Required method: Returns a single sample at the given index.
        
        Args:
            idx: Integer index (0 to len-1)
            
        Returns:
            Tuple of (features, label) for the sample at index idx
        """
        return self.X[idx], self.y[idx]


# =============================================================================
# STEP 2: Basic DataLoader Usage
# =============================================================================
def demo_basic_usage():
    """
    Demonstrates the most basic DataLoader usage.
    
    Key Learning Points:
    1. DataLoader wraps a Dataset
    2. It yields batches (groups of samples)
    3. Each batch is a tuple of tensors
    """
    print("\n" + "="*60)
    print("DEMO 1: Basic DataLoader Usage")
    print("="*60)
    
    # Create dataset
    dataset = SimpleDataset(num_samples=10, seed=42)
    
    # Create DataLoader with minimal configuration
    dataloader = DataLoader(
        dataset,
        batch_size=4,    # Group 4 samples per batch
        shuffle=False    # Keep original order (for now)
    )
    
    print(f"\n📦 DataLoader Configuration:")
    print(f"  - Batch size: 4")
    print(f"  - Total samples: 10")
    print(f"  - Expected batches: 3 (sizes: 4, 4, 2)")
    
    # Iterate through batches
    print(f"\n🔄 Iterating through batches:")
    for batch_idx, (features, labels) in enumerate(dataloader, 1):
        print(f"\n  Batch {batch_idx}:")
        print(f"    Features shape: {tuple(features.shape)}")
        print(f"    Labels shape: {tuple(labels.shape)}")
        print(f"    Labels: {labels.tolist()}")
        
    # Note: The last batch has only 2 samples (10 % 4 = 2 remainder)


# =============================================================================
# STEP 3: Understanding Shuffle
# =============================================================================
def demo_shuffle_effect():
    """
    Shows how shuffle=True randomizes sample order each epoch.
    
    Key Learning Points:
    1. shuffle=True randomizes the order
    2. Order changes between epochs
    3. Use generator for reproducible shuffling
    """
    print("\n" + "="*60)
    print("DEMO 2: Effect of Shuffling")
    print("="*60)
    
    dataset = SimpleDataset(num_samples=10, seed=1)
    
    # DataLoader WITH shuffle
    gen = torch.Generator().manual_seed(100)  # For reproducible shuffling
    dataloader_shuffled = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,     # ✓ Shuffle enabled
        generator=gen     # Controls shuffle randomness
    )
    
    print("\n🔀 With shuffle=True:")
    print("Running 2 epochs to see different orders...\n")
    
    for epoch in range(2):
        print(f"Epoch {epoch + 1}:")
        for batch_idx, (_, labels) in enumerate(dataloader_shuffled, 1):
            print(f"  Batch {batch_idx} labels: {labels.tolist()}")


# =============================================================================
# STEP 4: Manual Iteration with iter() and next()
# =============================================================================
def demo_manual_iteration():
    """
    Shows how to manually control batch iteration using iter() and next().
    
    Key Learning Points:
    1. DataLoader is an iterable (has __iter__ method)
    2. Can create iterator explicitly with iter()
    3. next() fetches one batch at a time
    4. Useful for debugging or fine-grained control
    """
    print("\n" + "="*60)
    print("DEMO 3: Manual Batch Iteration")
    print("="*60)
    
    dataset = SimpleDataset(num_samples=8, seed=2)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False)
    
    print("\n🎯 Manual batch fetching:")
    
    # Create an iterator explicitly
    batch_iterator = iter(dataloader)
    
    # Fetch batches one by one
    batch1 = next(batch_iterator)
    print(f"\n  First batch (manual):")
    print(f"    Features shape: {tuple(batch1[0].shape)}")
    print(f"    Labels: {batch1[1].tolist()}")
    
    batch2 = next(batch_iterator)
    print(f"\n  Second batch (manual):")
    print(f"    Features shape: {tuple(batch2[0].shape)}")
    print(f"    Labels: {batch2[1].tolist()}")
    
    # Process remaining batches with for-loop
    print(f"\n  Remaining batches (loop):")
    for batch_idx, (features, labels) in enumerate(batch_iterator, 3):
        print(f"    Batch {batch_idx} - Labels: {labels.tolist()}")


# =============================================================================
# STEP 5: Batch Size Analysis
# =============================================================================
def demo_batch_size_effects():
    """
    Shows how different batch sizes affect the number of batches.
    
    Key Learning Points:
    1. Smaller batch_size → More batches → More iterations
    2. Larger batch_size → Fewer batches → Fewer iterations
    3. Last batch may be smaller if samples don't divide evenly
    """
    print("\n" + "="*60)
    print("DEMO 4: Batch Size Effects")
    print("="*60)
    
    dataset = SimpleDataset(num_samples=10, seed=3)
    
    batch_sizes = [2, 3, 4, 5, 10]
    
    print("\n📊 Analyzing different batch sizes:")
    print(f"{'Batch Size':<12} {'Num Batches':<12} {'Batch Shapes'}")
    print("-" * 50)
    
    for bs in batch_sizes:
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=False)
        
        # Count batches and get their shapes
        batch_shapes = []
        for features, _ in dataloader:
            batch_shapes.append(features.shape[0])
        
        print(f"{bs:<12} {len(batch_shapes):<12} {batch_shapes}")
    
    print("\n💡 Observations:")
    print("  - Batch size of 10 creates 1 batch (whole dataset)")
    print("  - Batch size of 3 creates 4 batches (3+3+3+1)")
    print("  - Last batch size = num_samples % batch_size")


# =============================================================================
# STEP 6: Understanding the Iteration Protocol
# =============================================================================
def demo_iteration_protocol():
    """
    Deep dive into how DataLoader's iteration actually works.
    
    Key Learning Points:
    1. for-loop calls iter() automatically
    2. iter() creates a fresh iterator each time
    3. StopIteration signals end of epoch
    """
    print("\n" + "="*60)
    print("DEMO 5: DataLoader Iteration Protocol")
    print("="*60)
    
    dataset = SimpleDataset(num_samples=6, seed=4)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    print("\n🔍 What happens under the hood:")
    print("\n1. Creating an iterator:")
    iterator = iter(dataloader)
    print(f"   Type: {type(iterator)}")
    
    print("\n2. Fetching batches until exhausted:")
    batch_count = 0
    try:
        while True:
            batch = next(iterator)
            batch_count += 1
            print(f"   Batch {batch_count}: Got {batch[0].shape[0]} samples")
    except StopIteration:
        print(f"   StopIteration raised - Iterator exhausted")
    
    print(f"\n3. Creating a new iterator for another epoch:")
    iterator2 = iter(dataloader)
    batch = next(iterator2)
    print(f"   First batch of new epoch: {batch[0].shape[0]} samples")
    
    print("\n💡 Key Insight:")
    print("   Each call to iter(dataloader) creates a FRESH iterator")
    print("   This is why for-loops work multiple times on the same DataLoader")


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("PYTORCH DATALOADER FUNDAMENTALS")
    print("="*60)
    print("\nThis tutorial covers the absolute basics of DataLoader.")
    print("You'll learn how data flows from Dataset → DataLoader → Batches")
    
    # Run all demos
    demo_basic_usage()
    demo_shuffle_effect()
    demo_manual_iteration()
    demo_batch_size_effects()
    demo_iteration_protocol()
    
    print("\n" + "="*60)
    print("✓ TUTORIAL COMPLETE!")
    print("="*60)
    print("\n📚 Key Takeaways:")
    print("  1. DataLoader wraps Dataset and produces batches")
    print("  2. batch_size controls how many samples per batch")
    print("  3. shuffle=True randomizes order each epoch")
    print("  4. DataLoader is iterable (supports for-loops)")
    print("  5. iter() creates a fresh iterator each time")
    print("\n🎯 Next Steps:")
    print("  → Try 02_dataloader_parameters.py for advanced configuration")
    print("  → Experiment with different batch sizes on your data")
    print("  → Learn about num_workers for parallel data loading")
