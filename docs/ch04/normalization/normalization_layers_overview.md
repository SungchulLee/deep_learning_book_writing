# Normalization Layers in Deep Learning

A comprehensive guide and implementation collection for understanding and using normalization layers in deep neural networks.

## 📚 Contents

This package contains detailed implementations, explanations, and practical examples of the three most important normalization techniques:

### Files Overview

1. **batch_normalization.py**
   - Complete implementation from scratch (NumPy)
   - PyTorch examples with CNNs
   - Demonstrations of how BatchNorm works
   - Comparison with and without normalization

2. **layer_normalization.py**
   - NumPy implementation from scratch
   - RNN and Transformer examples
   - Comparison with Batch Normalization
   - Small batch size demonstrations

3. **instance_normalization.py**
   - NumPy implementation from scratch
   - Style transfer network examples
   - GAN generator with InstanceNorm
   - Comparison with other normalization methods

4. **normalization_comparison.py**
   - Side-by-side comparison of all methods
   - Performance characteristics
   - Practical recommendations
   - Common mistakes to avoid
   - Quick reference guide

5. **practical_examples.py**
   - Real-world applications
   - ResNet with BatchNorm (Image Classification)
   - Transformer with LayerNorm (NLP)
   - U-Net with GroupNorm (Segmentation)
   - CycleGAN with InstanceNorm (Style Transfer)
   - LSTM with LayerNorm (Sequence Modeling)

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running Examples

Each file can be run independently:

```bash
# Learn about Batch Normalization
python batch_normalization.py

# Learn about Layer Normalization
python layer_normalization.py

# Learn about Instance Normalization
python instance_normalization.py

# Compare all methods
python normalization_comparison.py

# See practical applications
python practical_examples.py
```

## 📖 Understanding Normalization Layers

### Why Normalization?

Normalization layers are critical for training deep neural networks because they:
- Accelerate training convergence
- Allow higher learning rates
- Reduce sensitivity to weight initialization
- Act as regularization
- Stabilize gradient flow

### Three Main Types

#### 1. Batch Normalization (BatchNorm)

**What it does:** Normalizes across the batch dimension

**Formula:** For each feature/channel, normalize using batch statistics
```
y = γ * (x - μ_batch) / √(σ²_batch + ε) + β
```

**When to use:**
- ✅ Image classification (CNNs)
- ✅ Large batch sizes (>= 32)
- ✅ Standard feedforward networks
- ❌ Small batch sizes (< 8)
- ❌ Online learning (batch size = 1)
- ❌ RNNs/Transformers

**Key characteristics:**
- Different behavior in training vs inference
- Maintains running statistics
- Sensitive to batch size
- Most common in computer vision

#### 2. Layer Normalization (LayerNorm)

**What it does:** Normalizes across the feature dimension (per sample)

**Formula:** For each sample, normalize using sample statistics
```
y = γ * (x - μ_sample) / √(σ²_sample + ε) + β
```

**When to use:**
- ✅ Transformers (BERT, GPT, etc.)
- ✅ RNNs and LSTMs
- ✅ Small batch sizes
- ✅ Variable batch sizes
- ✅ Online learning
- ❌ Traditional CNNs (BatchNorm usually better)

**Key characteristics:**
- Same behavior in training and inference
- Independent of batch size
- Standard choice for NLP models
- No running statistics needed

#### 3. Instance Normalization (InstanceNorm)

**What it does:** Normalizes each instance and each channel independently

**Formula:** For each sample and channel, normalize spatial dimensions
```
y = γ * (x - μ_instance) / √(σ²_instance + ε) + β
```

**When to use:**
- ✅ Style transfer
- ✅ GANs (especially image-to-image translation)
- ✅ When samples should be processed independently
- ❌ Image classification (BatchNorm better)
- ❌ NLP tasks (LayerNorm better)

**Key characteristics:**
- Removes instance-specific contrast
- Same behavior in training and inference
- Common in CycleGAN, Pix2Pix
- No batch statistics mixing

### Visual Comparison

```
Input shape: (N, C, H, W)
N = batch size
C = channels
H = height
W = width

Batch Norm:     Normalizes over (N, H, W) for each C
Layer Norm:     Normalizes over (C, H, W) for each N
Instance Norm:  Normalizes over (H, W) for each N and C
```

## 🎯 Decision Tree: Which Normalization to Use?

```
Start
  │
  ├─ Working with images?
  │   ├─ Yes → Classification/Detection?
  │   │   ├─ Yes → Large batch (>= 16)?
  │   │   │   ├─ Yes → Use BatchNorm
  │   │   │   └─ No → Use GroupNorm
  │   │   └─ No → Style Transfer/GANs?
  │   │       └─ Yes → Use InstanceNorm
  │   │
  │   └─ No → NLP/Sequences?
  │       ├─ Yes → Use LayerNorm
  │       └─ No → Online learning?
  │           └─ Yes → Use LayerNorm/InstanceNorm
  │
  └─ Other considerations:
      - Multi-GPU training → SyncBatchNorm
      - Video data → GroupNorm
      - Medical imaging → GroupNorm
```

## 💡 Best Practices

### DO's ✅

1. **Always call `model.eval()` before inference with BatchNorm**
   ```python
   model.eval()  # Critical for BatchNorm!
   with torch.no_grad():
       output = model(input)
   ```

2. **Use the right normalization for your task**
   - CNNs → BatchNorm
   - Transformers → LayerNorm
   - Style Transfer → InstanceNorm

3. **Standard order: Conv → Norm → Activation**
   ```python
   self.block = nn.Sequential(
       nn.Conv2d(...),
       nn.BatchNorm2d(...),
       nn.ReLU()
   )
   ```

4. **Use GroupNorm for small batches**
   ```python
   # Instead of BatchNorm with batch size < 8
   nn.GroupNorm(num_groups=8, num_channels=64)
   ```

### DON'Ts ❌

1. **Don't use BatchNorm with batch size = 1**
   - Statistics are meaningless
   - Use LayerNorm or InstanceNorm instead

2. **Don't forget train/eval mode differences**
   - BatchNorm behaves differently
   - LayerNorm/InstanceNorm: same behavior

3. **Don't use InstanceNorm for classification**
   - BatchNorm is better
   - InstanceNorm is for style-related tasks

4. **Don't mix normalization types randomly**
   - Be consistent within an architecture
   - Different norms for different purposes

## 🔬 Implementation Details

### PyTorch API Quick Reference

```python
# Batch Normalization
nn.BatchNorm1d(num_features)      # For linear layers
nn.BatchNorm2d(num_channels)      # For 2D convolutions
nn.BatchNorm3d(num_channels)      # For 3D data

# Layer Normalization
nn.LayerNorm(normalized_shape)    # Shape of features to normalize
nn.LayerNorm([C, H, W])          # For 2D data

# Instance Normalization
nn.InstanceNorm1d(num_features)   # For 1D data
nn.InstanceNorm2d(num_channels)   # For 2D data
nn.InstanceNorm3d(num_channels)   # For 3D data

# Group Normalization
nn.GroupNorm(num_groups, num_channels)  # Divide channels into groups

# Common parameters
eps=1e-5              # Numerical stability
momentum=0.1          # For BatchNorm running stats
affine=True          # Learnable scale/shift
track_running_stats=True  # For BatchNorm
```

## 📊 Performance Comparison

| Method        | Speed | Memory | Batch Dependent | Train/Eval Same |
|---------------|-------|--------|-----------------|-----------------|
| BatchNorm     | Fast  | Low    | Yes ⚠️         | No ⚠️          |
| LayerNorm     | Fast  | Low    | No ✅          | Yes ✅         |
| InstanceNorm  | Fast  | Low    | No ✅          | Yes ✅         |
| GroupNorm     | Fast  | Low    | No ✅          | Yes ✅         |

## 🎓 Learning Resources

### Papers
- **Batch Normalization:** "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" (Ioffe & Szegedy, 2015)
- **Layer Normalization:** "Layer Normalization" (Ba, Kiros & Hinton, 2016)
- **Instance Normalization:** "Instance Normalization: The Missing Ingredient for Fast Stylization" (Ulyanov et al., 2016)
- **Group Normalization:** "Group Normalization" (Wu & He, 2018)

### Key Concepts
- Internal Covariate Shift
- Running Statistics vs Batch Statistics
- Affine Transformations (γ and β)
- Train vs Eval Mode
- Gradient Flow in Deep Networks

## 🛠️ Troubleshooting

### Common Issues

**Problem:** Model works in training but fails in evaluation
- **Solution:** Remember to call `model.eval()` before inference

**Problem:** BatchNorm gives poor results with small batches
- **Solution:** Use GroupNorm or LayerNorm instead

**Problem:** Training is unstable
- **Solution:** Try GroupNorm or adjust BatchNorm momentum

**Problem:** Style transfer doesn't work well
- **Solution:** Make sure you're using InstanceNorm, not BatchNorm

## 📝 Examples Summary

Each example file includes:
- Theory and mathematical formulation
- NumPy implementation from scratch
- PyTorch implementation
- Practical demonstrations
- Comparisons and best practices

## 🤝 Contributing

Feel free to extend these examples with:
- More normalization variants (RMSNorm, etc.)
- Additional practical applications
- Performance benchmarks
- Visualization tools

## 📄 License

Educational resource - free to use and modify.

## 🔗 Additional Resources

- PyTorch Documentation: https://pytorch.org/docs/stable/nn.html
- Deep Learning Book: https://www.deeplearningbook.org/
- Papers With Code: https://paperswithcode.com/

---

**Happy Learning! 🚀**

For questions or suggestions, feel free to explore each file for detailed explanations and working code examples.
