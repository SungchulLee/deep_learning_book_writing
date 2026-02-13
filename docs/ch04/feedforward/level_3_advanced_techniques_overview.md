# Level 3: Advanced Techniques

## 🎯 Purpose

Learn production-ready techniques for training better models: regularization to prevent overfitting, normalization for stable training, learning rate scheduling for optimal convergence, and proper weight initialization.

## 📚 What You'll Master

- **Regularization**: Dropout, L1/L2, preventing overfitting
- **Normalization**: Batch normalization, layer normalization
- **Optimization**: Learning rate scheduling, adaptive methods
- **Initialization**: Xavier, He, and why it matters

## 📖 Files in This Level

### 14_dropout_regularization.py ⭐⭐
**Difficulty**: Intermediate | **Time**: 45-60 min

Prevent overfitting by randomly dropping neurons during training.

**What You'll Learn**:
- How dropout works
- `model.train()` vs `model.eval()`
- Dropout probability (p) selection
- When dropout helps vs hurts

**Dropout Visualization**:
```
Training:  X → [D] → Layer1 → [D] → Layer2 → [D] → Output
Testing:   X → Layer1 → Layer2 → Output (no dropout)
```

**Key Insight**: Dropout forces the network to learn redundant representations. Each neuron can't rely on specific other neurons being present.

**Typical Dropout Values**:
- Shallow networks: p = 0.1-0.2
- Deep networks: p = 0.3-0.5
- Too high (>0.7): underfitting
- Too low (<0.1): not effective

**Rule of Thumb**: Start with p=0.2, increase if overfitting persists.

**Why It Matters**: One of the most effective regularization techniques. Used in virtually all modern deep learning.

---

### 15_regularization_techniques_detailed.py ⭐⭐⭐
**Difficulty**: Advanced | **Time**: 60-90 min

Comprehensive comparison of all major regularization techniques.

**What You'll Learn**:
- L1 regularization (Lasso)
- L2 regularization (Ridge, weight decay)
- Dropout (covered in detail)
- Early stopping
- Data augmentation
- Ensemble methods

**Regularization Comparison**:

| Technique | How It Works | When to Use | Strength |
|-----------|--------------|-------------|----------|
| **L2 (Ridge)** | Penalize large weights | Default choice | Prevents extreme weights |
| **L1 (Lasso)** | Force some weights to zero | Feature selection | Sparse models |
| **Dropout** | Randomly drop neurons | Large networks | Prevents co-adaptation |
| **Early Stopping** | Stop when validation loss increases | Always | Free regularization |
| **Data Aug** | Create more training data | Computer vision | More data = less overfit |

**L2 Weight Decay Values**:
- Small: 1e-5 to 1e-4 (default start)
- Medium: 1e-3 to 1e-2
- Large: 0.1+ (rarely used)

**Overfitting Detection**:
```python
Train Loss ↓ + Validation Loss ↑ = OVERFITTING!
```

**Why It Matters**: Overfitting is the #1 problem in deep learning. You MUST know these techniques.

---

### 16_batch_normalization.py ⭐⭐
**Difficulty**: Intermediate | **Time**: 45-60 min

Stabilize training by normalizing layer inputs.

**What You'll Learn**:
- How batch normalization works
- Where to place BN layers
- `momentum` and `eps` parameters
- BN in training vs evaluation mode

**Batch Norm Operation**:
```
1. Normalize: x_norm = (x - mean) / sqrt(var + eps)
2. Scale & Shift: y = gamma * x_norm + beta
   (gamma and beta are learned parameters)
```

**Architecture Pattern**:
```python
# Option 1 (common)
Linear → BatchNorm → ReLU

# Option 2 (also works)
Linear → ReLU → BatchNorm
```

**Benefits**:
✅ Faster training (can use higher learning rates)  
✅ Less sensitive to initialization  
✅ Regularization effect (slight)  
✅ Allows deeper networks  

**When to Use**:
- Deep networks (>5 layers)
- Training is unstable
- Want faster convergence

**Why It Matters**: BN is crucial for training very deep networks (ResNet, etc.). Enables modern deep learning.

---

### 17_batch_normalization_detailed.py ⭐⭐⭐
**Difficulty**: Advanced | **Time**: 60-90 min

Deep dive into normalization techniques and their variants.

**What You'll Learn**:
- Batch Normalization (detailed)
- Layer Normalization
- Instance Normalization
- Group Normalization
- When to use each

**Normalization Comparison**:

| Type | Normalizes Over | Use Case | Pros | Cons |
|------|-----------------|----------|------|------|
| **Batch Norm** | Batch dimension | CNNs, large batches | Best for CV | Fails with small batches |
| **Layer Norm** | Feature dimension | NLP, RNNs, Transformers | Batch-independent | Slightly slower |
| **Instance Norm** | Spatial dimensions | Style transfer | Per-instance | Only for images |
| **Group Norm** | Channel groups | Small batches | Middle ground | Extra hyperparameter |

**Implementation Details**:
- `momentum`: How much to update running statistics (default: 0.1)
- `eps`: Small constant for numerical stability (default: 1e-5)
- `affine`: Whether to learn gamma and beta (default: True)

**Why It Matters**: Understanding different normalization types lets you handle edge cases (small batches, RNNs, etc.).

---

### 18_learning_rate_scheduling.py ⭐⭐⭐
**Difficulty**: Advanced | **Time**: 60-75 min

Dynamically adjust learning rate during training for better convergence.

**What You'll Learn**:
- StepLR: Decay at fixed intervals
- ExponentialLR: Smooth exponential decay
- ReduceLROnPlateau: Decay when training stalls
- CosineAnnealingLR: Cosine annealing
- OneCycleLR: Super-convergence

**Scheduler Comparison**:

| Scheduler | Pattern | Use Case | When to Use |
|-----------|---------|----------|-------------|
| **StepLR** | Piecewise constant | Simple baseline | Quick experiments |
| **ExponentialLR** | Smooth decay | Long training | Stable, predictable |
| **ReduceLROnPlateau** | Adaptive | Unknown plateau points | When unsure |
| **CosineAnnealingLR** | Smooth cosine | Modern standard | Good default choice |
| **OneCycleLR** | Triangular | Fast training | When time-constrained |

**Typical Schedule**:
```
LR: 0.1 ─────┐
              │  ←── StepLR
         0.01 ├────┐
              │    │
        0.001 └────┴─────→
         Epoch: 30  60  100
```

**Rule of Thumb**:
- Start: 1e-3 (Adam) or 1e-1 (SGD)
- Decay: 10x every 30-50 epochs
- Minimum: 1e-6

**Why It Matters**: Learning rate is the most important hyperparameter. Scheduling can improve accuracy by 5-10%!

---

### 19_weight_initialization.py ⭐⭐⭐
**Difficulty**: Advanced | **Time**: 45-60 min

Proper initialization is crucial for training deep networks.

**What You'll Learn**:
- Xavier (Glorot) initialization
- He (Kaiming) initialization
- Why initialization matters
- Vanishing/exploding gradients

**Initialization Methods**:

| Method | For Activation | Formula | Use Case |
|--------|----------------|---------|----------|
| **Xavier (Glorot)** | Tanh, Sigmoid | N(0, 2/(n_in + n_out)) | Symmetric activations |
| **He (Kaiming)** | ReLU | N(0, 2/n_in) | ReLU and variants |
| **LeCun** | SELU | N(0, 1/n_in) | SELU activation |
| **Orthogonal** | - | Orthogonal matrices | RNNs |

**The Problem**:
```python
# Bad initialization
W = torch.zeros(...)  # All neurons do the same thing!
W = torch.randn(...) * 10  # Exploding gradients!

# Good initialization  
W = torch.randn(...) * sqrt(2/n_in)  # He initialization for ReLU
```

**Rule of Thumb**:
- ReLU → He initialization
- Tanh → Xavier initialization
- Default PyTorch → Already uses good initialization!

**Why It Matters**: Poor initialization → vanishing/exploding gradients → network won't train. Period.

## 🎓 Learning Path

```
14_dropout_regularization.py (Required)
    ↓
15_regularization_techniques_detailed.py (Highly Recommended)
    ↓
16_batch_normalization.py (Required)
    ↓
17_batch_normalization_detailed.py (Optional but valuable)
    ↓
18_learning_rate_scheduling.py (Required)
    ↓
19_weight_initialization.py (Required)
    ↓
Ready for Level 4! 🎉
```

## 💡 Study Tips

- **Plot everything**: Loss curves, learning rates, gradient magnitudes
- **Compare techniques**: Train same model with/without each technique
- **Understand trade-offs**: More regularization = less overfitting but slower learning
- **Read the papers**: Original papers explain WHY, not just HOW

## 🧪 Critical Experiments

1. **Overfitting Demo**: Train large network on small dataset
   - No regularization → overfit
   - Add dropout → better generalization
   - Add weight decay → even better

2. **BN Impact**: Train deep network (10+ layers)
   - Without BN → unstable or fails
   - With BN → trains smoothly

3. **LR Schedule**: Train for 100 epochs
   - Constant LR → plateaus
   - With scheduling → continues improving

4. **Initialization**: Train deep network
   - Zero init → all neurons identical
   - Random init (no scaling) → exploding gradients
   - He init → trains successfully

## ✅ Level 3 Completion Checklist

Before moving to Level 4, make sure you can:

- [ ] Detect overfitting from loss curves
- [ ] Apply appropriate regularization techniques
- [ ] Use batch normalization correctly
- [ ] Implement learning rate schedules
- [ ] Understand different initialization methods
- [ ] Debug training instabilities
- [ ] Choose hyperparameters (dropout_p, weight_decay, LR)

## 🎯 Next Level Preview

**Level 4: Real-World Applications** will teach you:
- CIFAR-10 color image classification
- Regression problems (predicting continuous values)
- Multi-task learning (multiple outputs)
- Building very deep networks (20+ layers)
- Complete end-to-end workflows

---

**Outstanding!** You now know production-grade training techniques. Time to build complete systems! 🚀

*Pro tip: These techniques often work in combination. Typical recipe: Dropout + L2 + BN + LR scheduling*
