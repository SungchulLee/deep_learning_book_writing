#!/usr/bin/env python3
"""
=============================================================================
Collate Functions: Custom Batch Construction
=============================================================================

WHAT IS COLLATE_FN?
-------------------
collate_fn is a function that defines HOW to combine individual samples into a batch.

Default behavior:
  • Stack tensors with same shape → batch tensor
  • Example: [(X1, y1), (X2, y2)] → ([X1, X2], [y1, y2])

When you need custom collate_fn:
  ✓ Variable-length sequences (text, time series)
  ✓ Multiple inputs of different types
  ✓ Custom data structures (graphs, point clouds)
  ✓ Special padding or masking strategies
  ✓ Data augmentation at batch level

LEARNING OBJECTIVES:
-------------------
✓ Understand when default collate fails
✓ Write custom collate functions
✓ Handle variable-length data
✓ Implement padding strategies
✓ Apply batch-level transformations

DIFFICULTY: ⭐⭐ Intermediate
TIME: 20 minutes
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Dict
import random


# =============================================================================
# Variable-Length Sequence Dataset
# =============================================================================
class TextDataset(Dataset):
    """
    Dataset with variable-length sequences (like text).
    
    Real-world examples:
      • Sentences: "Hello" (5 tokens) vs "How are you?" (12 tokens)
      • Time series: 100 samples vs 500 samples
      • Audio: 2 seconds vs 10 seconds
    """
    
    def __init__(self, num_samples=10, min_len=3, max_len=10, vocab_size=20, seed=0):
        """
        Args:
            num_samples: Number of sequences
            min_len: Minimum sequence length
            max_len: Maximum sequence length
            vocab_size: Size of vocabulary
            seed: Random seed
        """
        random.seed(seed)
        torch.manual_seed(seed)
        
        self.sequences = []
        self.labels = []
        
        for i in range(num_samples):
            # Random length for this sequence
            seq_len = random.randint(min_len, max_len)
            
            # Random token IDs (simulating words/characters)
            seq = torch.randint(0, vocab_size, (seq_len,))
            
            # Binary label (e.g., positive/negative sentiment)
            label = torch.tensor(i % 2)
            
            self.sequences.append(seq)
            self.labels.append(label)
        
        lengths = [len(s) for s in self.sequences]
        print(f"✓ TextDataset created:")
        print(f"  - Samples: {num_samples}")
        print(f"  - Length range: [{min(lengths)}, {max(lengths)}]")
        print(f"  - Average length: {sum(lengths)/len(lengths):.1f}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# =============================================================================
# DEMO 1: Default Collate Failure
# =============================================================================
def demo_default_collate_failure():
    """
    Show when default collate_fn fails and why.
    
    Default collate expects all tensors to have the same shape.
    When they don't → RuntimeError!
    """
    print("\n" + "="*60)
    print("DEMO 1: When Default Collate Fails")
    print("="*60)
    
    dataset = TextDataset(num_samples=4, seed=1)
    
    # Print individual sample shapes
    print("\n📏 Individual sequence lengths:")
    for i in range(len(dataset)):
        seq, label = dataset[i]
        print(f"  Sample {i}: length = {len(seq)}")
    
    print("\n⚠️  Trying default DataLoader...")
    print("   This will FAIL because sequences have different lengths!")
    
    try:
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        print("   ✗ Unexpectedly succeeded!")
    except RuntimeError as e:
        print(f"   ✓ Failed as expected: {str(e)[:80]}...")
    
    print("\n💡 Solution: Use custom collate_fn to handle variable lengths!")


# =============================================================================
# DEMO 2: Simple Padding Collate
# =============================================================================
def simple_pad_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Simple padding strategy: pad all sequences to max length in batch.
    
    Args:
        batch: List of (sequence, label) tuples
        
    Returns:
        padded_seqs: [batch_size, max_len] padded sequences
        lengths: [batch_size] original lengths (useful for RNNs)
        labels: [batch_size] labels
    """
    # Separate sequences and labels
    sequences, labels = zip(*batch)
    
    # Get original lengths
    lengths = torch.tensor([len(seq) for seq in sequences])
    
    # Find max length in this batch
    max_len = lengths.max().item()
    
    # Pad sequences to max_len
    padded_seqs = torch.zeros(len(sequences), max_len, dtype=sequences[0].dtype)
    
    for i, seq in enumerate(sequences):
        padded_seqs[i, :len(seq)] = seq
    
    # Stack labels
    labels = torch.stack(labels)
    
    return padded_seqs, lengths, labels


def demo_simple_padding():
    """
    Demonstrate simple padding collate function.
    """
    print("\n" + "="*60)
    print("DEMO 2: Simple Padding Collate")
    print("="*60)
    
    dataset = TextDataset(num_samples=6, seed=2)
    
    # Use custom collate_fn
    loader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
        collate_fn=simple_pad_collate
    )
    
    print("\n📦 Batches with padding:\n")
    
    for batch_idx, (padded_seqs, lengths, labels) in enumerate(loader, 1):
        print(f"Batch {batch_idx}:")
        print(f"  Padded shape: {tuple(padded_seqs.shape)}")
        print(f"  Original lengths: {lengths.tolist()}")
        print(f"  Labels: {labels.tolist()}")
        print(f"  Max length in batch: {lengths.max().item()}")
        print(f"  Padding ratio: {(padded_seqs == 0).float().mean():.1%}\n")
    
    print("💡 Benefits:")
    print("   • All sequences have same length → can form tensor")
    print("   • Padding is per-batch (efficient)")
    print("   • Original lengths preserved for RNNs")


# =============================================================================
# DEMO 3: Using PyTorch's pad_sequence
# =============================================================================
def pytorch_pad_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Use PyTorch's built-in pad_sequence utility.
    
    pad_sequence is optimized and supports different padding modes:
      • batch_first=True: [batch, seq_len]  (most common)
      • batch_first=False: [seq_len, batch] (for RNNs)
      • padding_value: What to pad with (default: 0)
    """
    sequences, labels = zip(*batch)
    
    # PyTorch's pad_sequence
    # Default: batch_first=False → [seq_len, batch]
    # We want batch_first=True → [batch, seq_len]
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    
    lengths = torch.tensor([len(seq) for seq in sequences])
    labels = torch.stack(labels)
    
    return padded, lengths, labels


def demo_pytorch_padding():
    """
    Demonstrate PyTorch's pad_sequence utility.
    """
    print("\n" + "="*60)
    print("DEMO 3: PyTorch's pad_sequence")
    print("="*60)
    
    dataset = TextDataset(num_samples=6, seed=3)
    
    loader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
        collate_fn=pytorch_pad_collate
    )
    
    print("\n📦 Using pad_sequence:\n")
    
    for batch_idx, (padded, lengths, labels) in enumerate(loader, 1):
        print(f"Batch {batch_idx}:")
        print(f"  Shape: {tuple(padded.shape)}")
        print(f"  Lengths: {lengths.tolist()}")
        print()
    
    print("💡 Advantages of pad_sequence:")
    print("   • More efficient than manual padding")
    print("   • Supports different padding values")
    print("   • Handles batch_first parameter")


# =============================================================================
# DEMO 4: Advanced Collate with Masking
# =============================================================================
def masked_pad_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Advanced collate: Return padded sequences AND attention mask.
    
    Attention mask:
      • 1 for real tokens
      • 0 for padding tokens
      • Essential for Transformers (they need to know what to attend to)
    
    Returns:
        padded_seqs: [batch, max_len] padded sequences
        attention_mask: [batch, max_len] attention mask
        labels: [batch] labels
    """
    sequences, labels = zip(*batch)
    
    # Pad sequences
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    
    # Create attention mask
    # mask[i, j] = 1 if j < original_length[i], else 0
    batch_size, max_len = padded.shape
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    for i, seq in enumerate(sequences):
        attention_mask[i, :len(seq)] = True
    
    labels = torch.stack(labels)
    
    return padded, attention_mask, labels


def demo_masked_padding():
    """
    Demonstrate attention mask creation (used in Transformers).
    """
    print("\n" + "="*60)
    print("DEMO 4: Padding with Attention Mask")
    print("="*60)
    
    dataset = TextDataset(num_samples=4, min_len=3, max_len=7, seed=4)
    
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=masked_pad_collate
    )
    
    print("\n📦 Batches with attention masks:\n")
    
    for batch_idx, (padded, mask, labels) in enumerate(loader, 1):
        print(f"Batch {batch_idx}:")
        print(f"  Padded sequences shape: {tuple(padded.shape)}")
        print(f"  Padded sequences:")
        print(f"    {padded}")
        print(f"  Attention mask:")
        print(f"    {mask.int()}")
        print(f"  (1 = real token, 0 = padding)")
        print()
    
    print("💡 Use Case:")
    print("   Transformers use attention_mask to ignore padding tokens")
    print("   Example: BERT, GPT, T5, etc.")


# =============================================================================
# DEMO 5: Multi-Modal Collate
# =============================================================================
class MultiModalDataset(Dataset):
    """
    Dataset with multiple data types:
      • Image: [3, H, W]
      • Text: variable length sequence
      • Metadata: dictionary
    
    Real examples:
      • Image captioning: image + text caption
      • VQA: image + question text + answer
      • Multimodal transformers: image + text + audio
    """
    
    def __init__(self, num_samples=4, seed=0):
        torch.manual_seed(seed)
        self.data = []
        
        for i in range(num_samples):
            # Image: random [3, 32, 32]
            image = torch.randn(3, 32, 32)
            
            # Text: variable length
            text_len = random.randint(5, 15)
            text = torch.randint(0, 100, (text_len,))
            
            # Metadata
            metadata = {
                'id': i,
                'category': random.choice(['A', 'B', 'C']),
                'score': random.random()
            }
            
            self.data.append((image, text, metadata))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def multimodal_collate(batch):
    """
    Collate function for multi-modal data.
    
    Handles:
      • Fixed-size tensors (images): stack directly
      • Variable-length tensors (text): pad
      • Metadata (dicts): collect in list
    """
    images, texts, metadatas = zip(*batch)
    
    # Images: all same size → just stack
    images_batch = torch.stack(images)
    
    # Texts: variable length → pad
    texts_batch = pad_sequence(texts, batch_first=True, padding_value=0)
    text_lengths = torch.tensor([len(t) for t in texts])
    
    # Metadata: keep as list of dicts
    # (or convert to batch dict if needed)
    
    return {
        'images': images_batch,
        'texts': texts_batch,
        'text_lengths': text_lengths,
        'metadata': metadatas
    }


def demo_multimodal_collate():
    """
    Demonstrate collate for multi-modal data.
    """
    print("\n" + "="*60)
    print("DEMO 5: Multi-Modal Collate")
    print("="*60)
    
    dataset = MultiModalDataset(num_samples=4, seed=5)
    
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=multimodal_collate
    )
    
    print("\n📦 Multi-modal batches:\n")
    
    for batch_idx, batch in enumerate(loader, 1):
        print(f"Batch {batch_idx}:")
        print(f"  Images: {tuple(batch['images'].shape)}")
        print(f"  Texts: {tuple(batch['texts'].shape)}")
        print(f"  Text lengths: {batch['text_lengths'].tolist()}")
        print(f"  Metadata samples:")
        for i, meta in enumerate(batch['metadata']):
            print(f"    Sample {i}: {meta}")
        print()
    
    print("💡 Key Pattern:")
    print("   Return a dictionary instead of tuple for complex data!")
    print("   Makes code more readable and extensible")


# =============================================================================
# DEMO 6: Batch-Level Augmentation
# =============================================================================
def augmentation_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Apply data augmentation at BATCH level, not sample level.
    
    Benefits:
      • More efficient (GPU can do batch ops)
      • Can do cross-sample augmentations (e.g., mixup)
      • Synchronize augmentations across batch
    """
    sequences, labels = zip(*batch)
    
    # Pad sequences
    padded = pad_sequence(sequences, batch_first=True)
    labels = torch.stack(labels)
    
    # BATCH-LEVEL augmentation: Add noise to entire batch
    # In practice, this could be:
    #   • Random cropping
    #   • Mixup/CutMix
    #   • Batch normalization statistics
    noise = torch.randn_like(padded) * 0.1  # Small noise
    padded = padded + noise
    
    return padded, labels


def demo_batch_augmentation():
    """
    Demonstrate batch-level augmentation in collate_fn.
    """
    print("\n" + "="*60)
    print("DEMO 6: Batch-Level Augmentation")
    print("="*60)
    
    dataset = TextDataset(num_samples=4, seed=6)
    
    # Regular loader (no augmentation)
    loader_no_aug = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=pytorch_pad_collate
    )
    
    # Augmented loader
    loader_aug = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=augmentation_collate
    )
    
    # Get same batch from both loaders
    batch_no_aug = next(iter(loader_no_aug))
    batch_aug = next(iter(loader_aug))
    
    print("\n📊 Without augmentation:")
    print(f"  Batch stats: mean={batch_no_aug[0].float().mean():.3f}, std={batch_no_aug[0].float().std():.3f}")
    
    print("\n📊 With batch augmentation:")
    print(f"  Batch stats: mean={batch_aug[0].float().mean():.3f}, std={batch_aug[0].float().std():.3f}")
    
    print("\n💡 Use Cases:")
    print("   • Mixup: Blend samples within batch")
    print("   • CutMix: Combine patches from batch samples")
    print("   • Synchronized augmentation across batch")


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("COLLATE FUNCTIONS: COMPLETE GUIDE")
    print("="*60)
    
    demo_default_collate_failure()
    demo_simple_padding()
    demo_pytorch_padding()
    demo_masked_padding()
    demo_multimodal_collate()
    demo_batch_augmentation()
    
    print("\n" + "="*60)
    print("✓ TUTORIAL COMPLETE!")
    print("="*60)
    print("\n📚 Key Takeaways:")
    print("  1. collate_fn defines HOW to combine samples into batches")
    print("  2. Variable-length data requires custom padding")
    print("  3. pad_sequence is PyTorch's efficient padding utility")
    print("  4. Attention masks are essential for Transformers")
    print("  5. Multi-modal data needs careful collate design")
    print("  6. Batch-level augmentation can be done in collate_fn")
    print("\n🎯 Next Steps:")
    print("  → Learn about persistent_workers for efficiency")
    print("  → Explore prefetch_factor for optimization")
    print("  → Master distributed data loading for multi-GPU")
