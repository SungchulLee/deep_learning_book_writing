#!/usr/bin/env python3
"""
=============================================================================
Samplers: Controlling Data Access Patterns
=============================================================================

WHAT ARE SAMPLERS?
------------------
Samplers control WHICH samples are drawn and in WHAT ORDER.
  • Built-in: SequentialSampler, RandomSampler, WeightedRandomSampler
  • Custom: Create your own sampling logic
  • Replaces: shuffle parameter (sampler and shuffle are mutually exclusive)

WHY USE SAMPLERS?
-----------------
✓ Class imbalance: Over-sample minority classes
✓ Stratified sampling: Ensure class distribution in batches
✓ Curriculum learning: Start with easy samples
✓ Hard example mining: Focus on difficult samples
✓ Distributed training: Each GPU gets different samples

LEARNING OBJECTIVES:
-------------------
✓ Understand sampler vs shuffle
✓ Use WeightedRandomSampler for imbalanced data
✓ Create custom samplers
✓ Compare sampling strategies

DIFFICULTY: ⭐⭐ Intermediate
TIME: 20 minutes
"""

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data import RandomSampler, SequentialSampler, WeightedRandomSampler
import matplotlib.pyplot as plt
from collections import Counter


# =============================================================================
# Imbalanced Dataset
# =============================================================================
class ImbalancedDataset(Dataset):
    """
    Dataset with severe class imbalance (realistic scenario).
    
    Example:
      - 10% positive samples (minority class)
      - 90% negative samples (majority class)
    """
    
    def __init__(self, num_samples=100, minority_ratio=0.1, seed=0):
        """
        Args:
            num_samples: Total number of samples
            minority_ratio: Fraction of positive (minority) samples
            seed: Random seed
        """
        gen = torch.Generator().manual_seed(seed)
        
        # Features: random [num_samples, 8]
        self.X = torch.randn(num_samples, 8, generator=gen)
        
        # Labels: Create imbalanced distribution
        num_positive = int(num_samples * minority_ratio)
        num_negative = num_samples - num_positive
        
        # Create labels
        self.y = torch.cat([
            torch.ones(num_positive, dtype=torch.long),   # Positive samples
            torch.zeros(num_negative, dtype=torch.long)   # Negative samples
        ])
        
        # Shuffle labels to mix positive and negative
        perm = torch.randperm(num_samples, generator=gen)
        self.y = self.y[perm]
        
        # Print class distribution
        pos_count = (self.y == 1).sum().item()
        neg_count = (self.y == 0).sum().item()
        print(f"✓ Dataset created: {num_samples} samples")
        print(f"  - Class 0 (negative): {neg_count} ({neg_count/num_samples*100:.1f}%)")
        print(f"  - Class 1 (positive): {pos_count} ({pos_count/num_samples*100:.1f}%)")
        print(f"  - Imbalance ratio: {neg_count/pos_count:.1f}:1")
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =============================================================================
# DEMO 1: Sampler vs Shuffle
# =============================================================================
def demo_sampler_vs_shuffle():
    """
    Understanding the relationship between sampler and shuffle.
    
    Key Points:
    ----------
    • shuffle=True internally creates a RandomSampler
    • shuffle=False internally creates a SequentialSampler
    • You CANNOT use both sampler and shuffle parameters
    • sampler gives you full control over sampling logic
    """
    print("\n" + "="*60)
    print("DEMO 1: Sampler vs Shuffle")
    print("="*60)
    
    dataset = ImbalancedDataset(num_samples=20, seed=1)
    
    print("\n🔹 Using shuffle=False (SequentialSampler internally):")
    loader1 = DataLoader(dataset, batch_size=5, shuffle=False)
    labels1 = [y.tolist() for _, y in loader1]
    print(f"  Batch labels: {labels1}")
    
    print("\n🔹 Using SequentialSampler explicitly:")
    sampler = SequentialSampler(dataset)
    loader2 = DataLoader(dataset, batch_size=5, sampler=sampler)
    labels2 = [y.tolist() for _, y in loader2]
    print(f"  Batch labels: {labels2}")
    print(f"  Same as shuffle=False? {labels1 == labels2}")
    
    print("\n🔹 Using shuffle=True (RandomSampler internally):")
    gen1 = torch.Generator().manual_seed(42)
    loader3 = DataLoader(dataset, batch_size=5, shuffle=True, generator=gen1)
    labels3 = [y.tolist() for _, y in loader3]
    print(f"  Batch labels: {labels3}")
    
    print("\n🔹 Using RandomSampler explicitly:")
    gen2 = torch.Generator().manual_seed(42)  # Same seed
    sampler2 = RandomSampler(dataset, generator=gen2)
    loader4 = DataLoader(dataset, batch_size=5, sampler=sampler2)
    labels4 = [y.tolist() for _, y in loader4]
    print(f"  Batch labels: {labels4}")
    print(f"  Same as shuffle=True? {labels3 == labels4}")
    
    print("\n💡 Key Insight:")
    print("   shuffle is just a convenience parameter")
    print("   It creates RandomSampler or SequentialSampler under the hood")


# =============================================================================
# DEMO 2: WeightedRandomSampler for Imbalanced Data
# =============================================================================
def demo_weighted_sampler():
    """
    Use WeightedRandomSampler to balance classes.
    
    Strategy:
    --------
    1. Compute class frequencies
    2. Assign higher weights to minority class
    3. Sample with replacement using weights
    
    Result: Approximately balanced batches
    """
    print("\n" + "="*60)
    print("DEMO 2: WeightedRandomSampler for Class Balance")
    print("="*60)
    
    # Create heavily imbalanced dataset
    dataset = ImbalancedDataset(num_samples=100, minority_ratio=0.15, seed=2)
    
    # STEP 1: Compute per-class weights
    # Strategy: weight = 1 / class_frequency
    class_counts = torch.bincount(dataset.y)  # [count_class0, count_class1]
    class_weights = 1.0 / class_counts.float()  # Inverse frequency
    
    print(f"\n📊 Class weighting:")
    print(f"  Class counts: {class_counts.tolist()}")
    print(f"  Class weights: {class_weights.tolist()}")
    
    # STEP 2: Assign per-sample weights
    # Each sample gets the weight of its class
    sample_weights = class_weights[dataset.y]  # Shape: [num_samples]
    
    print(f"\n  Sample weights (first 10): {sample_weights[:10].tolist()}")
    print(f"  Samples from class 0 have weight: {class_weights[0].item():.4f}")
    print(f"  Samples from class 1 have weight: {class_weights[1].item():.4f}")
    
    # STEP 3: Create WeightedRandomSampler
    gen = torch.Generator().manual_seed(42)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),  # How many samples to draw
        replacement=True,          # Must be True for oversampling
        generator=gen
    )
    
    # STEP 4: Compare with and without weighted sampling
    print("\n🔹 WITHOUT weighted sampling (shuffle=True):")
    loader_unweighted = DataLoader(dataset, batch_size=20, shuffle=True, generator=gen)
    
    all_labels = []
    for _, y in loader_unweighted:
        all_labels.extend(y.tolist())
    
    counter = Counter(all_labels)
    print(f"  Class 0: {counter[0]} samples ({counter[0]/len(all_labels)*100:.1f}%)")
    print(f"  Class 1: {counter[1]} samples ({counter[1]/len(all_labels)*100:.1f}%)")
    
    print("\n🔹 WITH weighted sampling (balanced):")
    loader_weighted = DataLoader(dataset, batch_size=20, sampler=sampler)
    
    all_labels_weighted = []
    for _, y in loader_weighted:
        all_labels_weighted.extend(y.tolist())
    
    counter_weighted = Counter(all_labels_weighted)
    print(f"  Class 0: {counter_weighted[0]} samples ({counter_weighted[0]/len(all_labels_weighted)*100:.1f}%)")
    print(f"  Class 1: {counter_weighted[1]} samples ({counter_weighted[1]/len(all_labels_weighted)*100:.1f}%)")
    
    print("\n💡 Result:")
    print("   Weighted sampling produces ~50/50 class distribution!")
    print("   Minority class samples appear multiple times (replacement=True)")


# =============================================================================
# DEMO 3: Custom Sampler
# =============================================================================
class BalancedBatchSampler(Sampler):
    """
    Custom sampler that creates balanced batches.
    
    Strategy:
    --------
    Each batch contains equal numbers from each class.
    Example: batch_size=8 → 4 samples from class 0, 4 from class 1
    
    This is stronger than WeightedRandomSampler because it
    GUARANTEES balance within each batch (not just across the dataset).
    """
    
    def __init__(self, labels, batch_size, seed=0):
        """
        Args:
            labels: Tensor of class labels
            batch_size: Must be even (to split equally)
            seed: Random seed
        """
        if batch_size % 2 != 0:
            raise ValueError("batch_size must be even for balanced batches")
        
        self.labels = labels
        self.batch_size = batch_size
        self.samples_per_class = batch_size // 2
        
        # Find indices for each class
        self.class0_indices = (labels == 0).nonzero(as_tuple=True)[0].tolist()
        self.class1_indices = (labels == 1).nonzero(as_tuple=True)[0].tolist()
        
        self.gen = torch.Generator().manual_seed(seed)
        
        # Calculate number of batches
        min_class_size = min(len(self.class0_indices), len(self.class1_indices))
        self.num_batches = min_class_size // self.samples_per_class
    
    def __iter__(self):
        """Yield indices for balanced batches."""
        # Shuffle each class independently
        perm0 = torch.randperm(len(self.class0_indices), generator=self.gen).tolist()
        perm1 = torch.randperm(len(self.class1_indices), generator=self.gen).tolist()
        
        shuffled_class0 = [self.class0_indices[i] for i in perm0]
        shuffled_class1 = [self.class1_indices[i] for i in perm1]
        
        # Create balanced batches
        for i in range(self.num_batches):
            start_idx = i * self.samples_per_class
            end_idx = start_idx + self.samples_per_class
            
            # Take samples_per_class from each class
            batch_indices = (
                shuffled_class0[start_idx:end_idx] +
                shuffled_class1[start_idx:end_idx]
            )
            
            # Shuffle within batch
            batch_perm = torch.randperm(len(batch_indices), generator=self.gen).tolist()
            batch_indices = [batch_indices[i] for i in batch_perm]
            
            # Yield batch indices one by one
            for idx in batch_indices:
                yield idx
    
    def __len__(self):
        """Total number of samples (not batches)."""
        return self.num_batches * self.batch_size


def demo_custom_sampler():
    """
    Demonstrate custom sampler that guarantees balanced batches.
    """
    print("\n" + "="*60)
    print("DEMO 3: Custom Balanced Batch Sampler")
    print("="*60)
    
    dataset = ImbalancedDataset(num_samples=100, minority_ratio=0.2, seed=3)
    
    # Create custom balanced sampler
    sampler = BalancedBatchSampler(dataset.y, batch_size=8, seed=42)
    
    # Note: We use batch_size=1 because sampler already groups samples
    # The sampler's __iter__ yields indices in batch-sized groups
    loader = DataLoader(dataset, batch_size=8, sampler=sampler)
    
    print(f"\n📊 Balanced batch sampling:")
    print(f"  Batch size: 8 (4 from each class)\n")
    
    batch_distributions = []
    for batch_idx, (_, y) in enumerate(loader, 1):
        class0_count = (y == 0).sum().item()
        class1_count = (y == 1).sum().item()
        batch_distributions.append((class0_count, class1_count))
        
        print(f"  Batch {batch_idx}: Class 0: {class0_count}, Class 1: {class1_count}")
        
        if batch_idx >= 5:  # Show first 5 batches
            break
    
    print(f"\n💡 Perfect Balance:")
    print("   Every batch has EXACTLY 4 samples from each class!")
    print("   This is stronger than WeightedRandomSampler")


# =============================================================================
# DEMO 4: Sampler Comparison
# =============================================================================
def demo_sampler_comparison():
    """
    Compare different sampling strategies side-by-side.
    """
    print("\n" + "="*60)
    print("DEMO 4: Sampling Strategy Comparison")
    print("="*60)
    
    dataset = ImbalancedDataset(num_samples=100, minority_ratio=0.1, seed=4)
    
    # Strategy 1: No sampling (natural distribution)
    print("\n🔹 Strategy 1: Random (shuffle=True)")
    print("   Strategy: Pure random sampling")
    loader1 = DataLoader(dataset, batch_size=10, shuffle=True)
    labels1 = torch.cat([y for _, y in loader1])
    dist1 = Counter(labels1.tolist())
    print(f"   Class 0: {dist1[0]}, Class 1: {dist1[1]}")
    print(f"   Balance: {dist1[1]/(dist1[0]+dist1[1])*100:.1f}% positive")
    
    # Strategy 2: Weighted sampling
    print("\n🔹 Strategy 2: WeightedRandomSampler")
    print("   Strategy: Weight by inverse frequency")
    class_counts = torch.bincount(dataset.y)
    sample_weights = (1.0 / class_counts.float())[dataset.y]
    sampler2 = WeightedRandomSampler(sample_weights, len(dataset), replacement=True)
    loader2 = DataLoader(dataset, batch_size=10, sampler=sampler2)
    labels2 = torch.cat([y for _, y in loader2])
    dist2 = Counter(labels2.tolist())
    print(f"   Class 0: {dist2[0]}, Class 1: {dist2[1]}")
    print(f"   Balance: {dist2[1]/(dist2[0]+dist2[1])*100:.1f}% positive")
    
    # Strategy 3: Balanced batch sampler
    print("\n🔹 Strategy 3: BalancedBatchSampler")
    print("   Strategy: Force equal counts per batch")
    sampler3 = BalancedBatchSampler(dataset.y, batch_size=10, seed=42)
    loader3 = DataLoader(dataset, batch_size=10, sampler=sampler3)
    labels3 = torch.cat([y for _, y in loader3])
    dist3 = Counter(labels3.tolist())
    print(f"   Class 0: {dist3[0]}, Class 1: {dist3[1]}")
    print(f"   Balance: {dist3[1]/(dist3[0]+dist3[1])*100:.1f}% positive")
    
    print("\n📊 Summary:")
    print(f"{'Strategy':<25} {'Class 0':<10} {'Class 1':<10} {'Balance'}")
    print("-" * 60)
    print(f"{'Random':<25} {dist1[0]:<10} {dist1[1]:<10} {dist1[1]/(dist1[0]+dist1[1])*100:.1f}%")
    print(f"{'WeightedRandomSampler':<25} {dist2[0]:<10} {dist2[1]:<10} {dist2[1]/(dist2[0]+dist2[1])*100:.1f}%")
    print(f"{'BalancedBatchSampler':<25} {dist3[0]:<10} {dist3[1]:<10} {dist3[1]/(dist3[0]+dist3[1])*100:.1f}%")
    
    print("\n💡 When to Use:")
    print("   • Random: Balanced datasets, no special requirements")
    print("   • WeightedRandomSampler: Imbalanced data, approximate balance ok")
    print("   • BalancedBatchSampler: Need guaranteed balance per batch")


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("DATALOADER SAMPLERS: COMPLETE GUIDE")
    print("="*60)
    
    demo_sampler_vs_shuffle()
    demo_weighted_sampler()
    demo_custom_sampler()
    demo_sampler_comparison()
    
    print("\n" + "="*60)
    print("✓ TUTORIAL COMPLETE!")
    print("="*60)
    print("\n📚 Key Takeaways:")
    print("  1. Samplers control which samples are drawn and in what order")
    print("  2. shuffle=True is just a RandomSampler shortcut")
    print("  3. WeightedRandomSampler balances imbalanced datasets")
    print("  4. Custom samplers give full control over sampling logic")
    print("  5. Choose strategy based on your balancing requirements")
    print("\n🎯 Next Steps:")
    print("  → Learn about collate_fn for custom batch construction")
    print("  → Explore distributed sampling for multi-GPU training")
    print("  → Implement stratified sampling for validation sets")
