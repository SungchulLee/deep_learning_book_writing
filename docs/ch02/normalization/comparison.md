# Normalization Methods Comparison

## Overview

This chapter provides a comprehensive comparison of all major normalization techniques used in deep learning: Batch Normalization, Layer Normalization, Instance Normalization, Group Normalization, Weight Normalization, and RMSNorm. Understanding when to use each method is crucial for building effective neural network architectures.

## Quick Reference Table

| Method | Normalizes Over | Batch Dependent | Train/Eval Same | Primary Use Case |
|--------|-----------------|-----------------|-----------------|------------------|
| **BatchNorm** | (N, H, W) per C | Yes | No | CNNs, large batches |
| **LayerNorm** | (C, H, W) per N | No | Yes | Transformers, RNNs |
| **InstanceNorm** | (H, W) per N, C | No | Yes | Style transfer, GANs |
| **GroupNorm** | (H, W, C/G) per N | No | Yes | Small batch CNNs |
| **WeightNorm** | Weights directly | No | Yes | Audio, generative |
| **RMSNorm** | Features per N | No | Yes | Modern LLMs |

## Visual Comparison

For a 4D tensor with shape $(N, C, H, W)$:

```
N = Batch size, C = Channels, H = Height, W = Width

BatchNorm:    Statistics over N×H×W for each C    → C statistics
LayerNorm:    Statistics over C×H×W for each N    → N statistics  
InstanceNorm: Statistics over H×W for each (N,C)  → N×C statistics
GroupNorm:    Statistics over H×W×(C/G) for (N,G) → N×G statistics
```

## Comprehensive Code Comparison

```python
import torch
import torch.nn as nn

def comprehensive_normalization_comparison():
    """Compare all normalization methods on the same input."""
    
    torch.manual_seed(42)
    
    # Input: 4 images, 8 channels, 4x4 spatial
    N, C, H, W = 4, 8, 4, 4
    x = torch.randn(N, C, H, W)
    
    # Initialize all normalizations
    bn = nn.BatchNorm2d(C, affine=False)
    bn.eval()
    ln = nn.LayerNorm([C, H, W], elementwise_affine=False)
    gn = nn.GroupNorm(4, C, affine=False)  # 4 groups
    in_norm = nn.InstanceNorm2d(C, affine=False)
    
    # Apply normalizations
    out_bn = bn(x)
    out_ln = ln(x)
    out_gn = gn(x)
    out_in = in_norm(x)
    
    print("Normalization Statistics Comparison")
    print("=" * 70)
    print(f"Input shape: {tuple(x.shape)}")
    
    print("\n1. Batch Normalization (per channel across batch):")
    for c in range(min(3, C)):
        mean = out_bn[:, c].mean().item()
        std = out_bn[:, c].std().item()
        print(f"   Channel {c}: mean={mean:.6f}, std={std:.4f}")
    
    print("\n2. Layer Normalization (per sample across features):")
    for n in range(min(3, N)):
        mean = out_ln[n].mean().item()
        std = out_ln[n].std().item()
        print(f"   Sample {n}: mean={mean:.6f}, std={std:.4f}")
    
    print("\n3. Instance Normalization (per sample-channel):")
    for n in range(min(2, N)):
        for c in range(min(2, C)):
            mean = out_in[n, c].mean().item()
            print(f"   Sample {n}, Channel {c}: mean={mean:.6f}")
    
    print("\n4. Group Normalization (per sample-group):")
    G = 4
    out_gn_reshaped = out_gn.view(N, G, C // G, H, W)
    for n in range(min(2, N)):
        for g in range(min(2, G)):
            mean = out_gn_reshaped[n, g].mean().item()
            print(f"   Sample {n}, Group {g}: mean={mean:.6f}")

comprehensive_normalization_comparison()
```

## Batch Size Sensitivity Analysis

```python
def test_batch_size_sensitivity():
    """Test how each normalization handles different batch sizes."""
    
    torch.manual_seed(42)
    batch_sizes = [1, 2, 4, 8, 16, 32]
    
    print("Output standard deviation by batch size:")
    print("=" * 70)
    print(f"{'Batch':>6} {'BatchNorm':>12} {'LayerNorm':>12} {'GroupNorm':>12} {'InstNorm':>12}")
    print("-" * 70)
    
    for bs in batch_sizes:
        x = torch.randn(bs, 64, 8, 8)
        
        bn = nn.BatchNorm2d(64)
        bn.train()
        ln = nn.LayerNorm([64, 8, 8])
        gn = nn.GroupNorm(8, 64)
        in_norm = nn.InstanceNorm2d(64)
        
        out_bn = bn(x).std().item()
        out_ln = ln(x).std().item()
        out_gn = gn(x).std().item()
        out_in = in_norm(x).std().item()
        
        print(f"{bs:>6} {out_bn:>12.4f} {out_ln:>12.4f} {out_gn:>12.4f} {out_in:>12.4f}")

test_batch_size_sensitivity()
```

**Key observations:**
- BatchNorm is unstable with batch size 1 (std=0)
- LayerNorm, GroupNorm, InstanceNorm are batch-independent

## Decision Tree: Choosing Normalization

```
Start
  │
  ├─► Is this for a Transformer or RNN?
  │     ├─ Yes ──► LayerNorm (or RMSNorm for efficiency)
  │     └─ No ──► Continue
  │
  ├─► Is this for style transfer or image-to-image GANs?
  │     ├─ Yes ──► InstanceNorm
  │     └─ No ──► Continue
  │
  ├─► Is this a CNN for classification/detection?
  │     ├─ Yes ──► What's your batch size?
  │     │           ├─ Large (≥16) ──► BatchNorm
  │     │           └─ Small (<16) ──► GroupNorm
  │     └─ No ──► Continue
  │
  ├─► Is this a generative audio model?
  │     ├─ Yes ──► WeightNorm
  │     └─ No ──► Continue
  │
  └─► Default recommendation:
        ├─ CNNs ──► GroupNorm (safest choice)
        ├─ Sequences ──► LayerNorm
        └─ Unsure ──► LayerNorm (most flexible)
```

## Use Case Recommendations

### Computer Vision

| Task | Recommended | Alternative | Avoid |
|------|-------------|-------------|-------|
| Image Classification | BatchNorm | GroupNorm | InstanceNorm |
| Object Detection | GroupNorm | SyncBatchNorm | - |
| Semantic Segmentation | GroupNorm | BatchNorm | InstanceNorm |
| Style Transfer | InstanceNorm | - | BatchNorm |
| Image Generation (GAN) | InstanceNorm | BatchNorm (disc.) | - |

### Natural Language Processing

| Task | Recommended | Alternative | Avoid |
|------|-------------|-------------|-------|
| Transformers (BERT, GPT) | LayerNorm | RMSNorm | BatchNorm |
| Modern LLMs (LLaMA) | RMSNorm | LayerNorm | BatchNorm |
| RNNs/LSTMs | LayerNorm | - | BatchNorm |

### Other Domains

| Task | Recommended | Alternative |
|------|-------------|-------------|
| Audio (WaveNet) | WeightNorm | LayerNorm |
| Video Understanding | GroupNorm | BatchNorm |
| Medical Imaging | GroupNorm | InstanceNorm |
| Reinforcement Learning | LayerNorm | WeightNorm |

## Common Mistakes and Solutions

### Mistake 1: Forgetting model.eval() with BatchNorm

```python
# WRONG
model = load_model()
output = model(test_input)  # BatchNorm uses batch stats!

# CORRECT
model = load_model()
model.eval()  # Critical!
with torch.no_grad():
    output = model(test_input)
```

### Mistake 2: BatchNorm with batch_size=1

```python
# WRONG - degenerate statistics
nn.BatchNorm2d(64)  # Fails with batch=1

# CORRECT - use GroupNorm
nn.GroupNorm(8, 64)  # Works with any batch size
```

### Mistake 3: Wrong normalization for style transfer

```python
# WRONG - BatchNorm mixes sample statistics
nn.BatchNorm2d(64)  # Don't use for style transfer

# CORRECT - InstanceNorm keeps samples independent
nn.InstanceNorm2d(64, affine=True)
```

### Mistake 4: Redundant bias with normalization

```python
# WRONG - bias will be negated by normalization
nn.Conv2d(64, 128, 3, bias=True)
nn.BatchNorm2d(128)

# CORRECT - no bias needed before normalization
nn.Conv2d(64, 128, 3, bias=False)
nn.BatchNorm2d(128)
```

## Performance Summary

| Method | Speed | Memory | Parameters |
|--------|-------|--------|------------|
| BatchNorm | Fast | Low (running stats) | 2C |
| LayerNorm | Fast | Low | 2×shape |
| InstanceNorm | Fast | Low | 2C (if affine) |
| GroupNorm | Fast | Low | 2C |
| WeightNorm | Fast | Low | 1 per output |
| RMSNorm | Fastest | Lowest | shape (γ only) |

## Summary Recommendations

### For CNNs (Image Tasks)
- **Large batches (≥16)**: BatchNorm
- **Small batches (<16)**: GroupNorm
- **Style transfer**: InstanceNorm
- **Multi-GPU**: SyncBatchNorm

### For Transformers (NLP)
- **Standard**: LayerNorm
- **Efficiency-focused**: RMSNorm
- **Modern LLMs**: RMSNorm (Pre-norm)

### For RNNs/LSTMs
- **Recommended**: LayerNorm
- **Alternative**: WeightNorm

### General Rules
1. When in doubt, use **GroupNorm** for vision, **LayerNorm** for sequences
2. Always check batch size requirements
3. Remember train/eval mode differences for BatchNorm
4. Use `bias=False` in layers preceding normalization

## References

1. Ioffe, S., & Szegedy, C. (2015). Batch Normalization. *ICML*.
2. Ba, J. L., et al. (2016). Layer Normalization. *arXiv*.
3. Ulyanov, D., et al. (2016). Instance Normalization. *arXiv*.
4. Wu, Y., & He, K. (2018). Group Normalization. *ECCV*.
5. Salimans, T., & Kingma, D. P. (2016). Weight Normalization. *NeurIPS*.
6. Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization. *NeurIPS*.
