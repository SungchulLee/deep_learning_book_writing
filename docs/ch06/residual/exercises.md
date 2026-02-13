# Residual Connections - Exercises and Assignments

## 📝 Learning Exercises

These exercises are designed to deepen your understanding of residual connections. Start with Level 1 and progress through the levels.

---

## Level 1: Understanding the Basics (Beginner)

### Exercise 1.1: Implement Identity Mapping
**Goal**: Understand the simplest form of residual connection

```python
class IdentityBlock(nn.Module):
    """
    TODO: Implement a residual block that just adds input to output
    
    Forward pass should be: output = conv(x) + x
    """
    def __init__(self, channels):
        super().__init__()
        # TODO: Add a 3x3 convolution
        pass
    
    def forward(self, x):
        # TODO: Implement forward pass
        pass

# Test your implementation
block = IdentityBlock(64)
x = torch.randn(1, 64, 32, 32)
out = block(x)
assert out.shape == x.shape, "Output shape should match input shape"
```

**Questions:**
1. What happens if you remove the `+ x` term?
2. Why is this called an "identity" mapping?
3. What would happen during backpropagation without the skip connection?

---

### Exercise 1.2: Compare Gradient Magnitudes
**Goal**: Verify that residual connections help gradient flow

```python
def measure_gradient_magnitude(model, depth=10):
    """
    TODO: 
    1. Create a random input
    2. Forward pass through 'depth' layers
    3. Compute loss and backward pass
    4. Measure gradient magnitude at input
    """
    pass

# TODO: Compare plain network vs residual network
plain_grad = measure_gradient_magnitude(plain_model)
residual_grad = measure_gradient_magnitude(residual_model)

print(f"Plain network gradient: {plain_grad}")
print(f"Residual network gradient: {residual_grad}")
```

**Questions:**
1. Which network has larger gradients?
2. Why does this matter for training?
3. What happens as you increase depth to 20, 50, 100 layers?

---

### Exercise 1.3: Visualize Feature Maps
**Goal**: See what residual blocks learn

```python
def visualize_residual_components(block, input_image):
    """
    TODO: Visualize three components:
    1. Residual path output: F(x)
    2. Skip connection: x
    3. Final output: F(x) + x
    """
    pass
```

**Questions:**
1. What patterns do you see in the residual path F(x)?
2. How does F(x) + x differ from F(x) alone?
3. Why might learning F(x) be easier than learning H(x)?

---

## Level 2: Implementation Challenges (Intermediate)

### Exercise 2.1: Handle Dimension Mismatches
**Goal**: Implement proper dimension matching in skip connections

```python
class ResidualBlockWithProjection(nn.Module):
    """
    TODO: Implement a residual block that handles:
    - Spatial dimension changes (stride != 1)
    - Channel dimension changes (in_channels != out_channels)
    
    Use 1x1 convolution for the projection shortcut
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # TODO: Implement
        pass
    
    def forward(self, x):
        # TODO: Implement
        pass

# Test cases
block1 = ResidualBlockWithProjection(64, 128, stride=2)
x1 = torch.randn(1, 64, 32, 32)
out1 = block1(x1)
assert out1.shape == (1, 128, 16, 16), f"Expected (1,128,16,16), got {out1.shape}"

block2 = ResidualBlockWithProjection(64, 64, stride=1)
x2 = torch.randn(1, 64, 32, 32)
out2 = block2(x2)
assert out2.shape == (1, 64, 32, 32), f"Expected (1,64,32,32), got {out2.shape}"
```

**Questions:**
1. Why do we use 1×1 convolution for projection?
2. What happens if stride=2? How does this affect the spatial dimensions?
3. Could we use other methods besides 1×1 convolution? What are the trade-offs?

---

### Exercise 2.2: Build a Custom ResNet
**Goal**: Create a ResNet variant for your own dataset

```python
def build_custom_resnet(input_channels=3, num_classes=10, block_config=[2, 2, 2, 2]):
    """
    TODO: Build a ResNet with custom configuration
    
    Args:
        input_channels: Number of input channels (3 for RGB)
        num_classes: Number of output classes
        block_config: Number of residual blocks in each layer
    
    Architecture should follow:
    1. Initial 7x7 conv + MaxPool
    2. 4 layers of residual blocks (defined by block_config)
    3. Global average pooling
    4. Fully connected layer
    """
    pass

# Test your implementation
model = build_custom_resnet(input_channels=1, num_classes=100, block_config=[3, 4, 6, 3])
x = torch.randn(2, 1, 224, 224)
out = model(x)
assert out.shape == (2, 100), f"Expected shape (2, 100), got {out.shape}"
```

**Questions:**
1. How many parameters does your model have?
2. How does changing block_config affect model capacity?
3. What modifications would you make for small images (32×32) vs large images (224×224)?

---

### Exercise 2.3: Implement Pre-activation ResNet
**Goal**: Understand the improved residual connection design

```python
class PreActivationBlock(nn.Module):
    """
    TODO: Implement pre-activation residual block
    
    Original ResNet:  Conv -> BN -> ReLU -> Conv -> BN -> Add -> ReLU
    Pre-activation:   BN -> ReLU -> Conv -> BN -> ReLU -> Conv -> Add
    
    Key difference: BN and ReLU come BEFORE convolution
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # TODO: Implement
        pass
    
    def forward(self, x):
        # TODO: Implement
        pass
```

**Questions:**
1. Why is pre-activation better for very deep networks?
2. How does this affect gradient flow?
3. In what scenarios would you choose pre-activation over standard ResNet?

---

## Level 3: Advanced Research (Advanced)

### Exercise 3.1: Implement ResNeXt
**Goal**: Understand cardinality and aggregated transformations

```python
class ResNeXtBlock(nn.Module):
    """
    TODO: Implement ResNeXt block with grouped convolutions
    
    Key innovation: Split-transform-merge with cardinality C
    - Split input into C groups
    - Apply same transformation to each group  
    - Aggregate results
    
    Use groups parameter in nn.Conv2d
    """
    def __init__(self, in_channels, out_channels, cardinality=32, stride=1):
        super().__init__()
        # TODO: Implement
        pass
    
    def forward(self, x):
        # TODO: Implement
        pass
```

**Questions:**
1. How does cardinality affect model capacity?
2. What are the computational benefits of grouped convolutions?
3. How would you choose cardinality for your application?

---

### Exercise 3.2: Add Squeeze-and-Excitation
**Goal**: Implement channel attention mechanism

```python
class SEBlock(nn.Module):
    """
    TODO: Implement Squeeze-and-Excitation block
    
    Steps:
    1. Squeeze: Global average pooling (C × H × W → C × 1 × 1)
    2. Excitation: FC → ReLU → FC → Sigmoid
    3. Scale: Multiply input by channel weights
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        # TODO: Implement
        pass
    
    def forward(self, x):
        # TODO: Implement
        pass

class SEResidualBlock(nn.Module):
    """
    TODO: Combine SE block with residual block
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # TODO: Implement
        pass
    
    def forward(self, x):
        # TODO: Implement
        pass
```

**Questions:**
1. How does SE block improve feature representation?
2. What is the computational overhead of SE blocks?
3. Where should SE blocks be placed in a residual block?

---

### Exercise 3.3: Design Your Own Variant
**Goal**: Create a novel residual connection variant

```python
class MyCustomResidualBlock(nn.Module):
    """
    TODO: Design your own residual block variant
    
    Ideas to explore:
    - Different activation functions (GELU, Swish)
    - Attention mechanisms (spatial attention, CBAM)
    - Different normalization (Layer norm, Group norm)
    - Skip connection variants (weighted, gated)
    - Hybrid architectures (Conv + Transformer)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # TODO: Your creative implementation
        pass
    
    def forward(self, x):
        # TODO: Your creative implementation
        pass
```

**Questions:**
1. What problem does your variant solve?
2. What are the trade-offs (computation, memory, accuracy)?
3. How would you evaluate whether it's better than standard ResNet?

---

## 🎯 Mini-Projects

### Project 1: Ablation Study (Intermediate)
**Goal**: Understand the importance of each component

Systematically remove components and measure performance:
1. Train ResNet-18 normally (baseline)
2. Remove batch normalization
3. Remove skip connections (plain network)
4. Use different activation functions
5. Try different initialization schemes

**Deliverables:**
- Training curves for each variant
- Final accuracy comparison
- Analysis of why each component matters

---

### Project 2: Architecture Search (Advanced)
**Goal**: Find optimal architecture for CIFAR-10

Experiment with:
- Number of blocks per layer [2,2,2,2] vs [3,4,6,3]
- Different widths (channel multipliers)
- Block types (Basic vs Bottleneck)
- Additional components (SE, etc.)

**Deliverables:**
- Performance vs parameters plot
- Best architecture configuration
- Training time comparison

---

### Project 3: Transfer Learning (Advanced)
**Goal**: Apply pre-trained ResNet to new domain

1. Load pre-trained ResNet from torchvision
2. Fine-tune on a different dataset (e.g., medical images)
3. Compare with training from scratch
4. Experiment with different fine-tuning strategies:
   - Freeze early layers
   - Progressive unfreezing
   - Different learning rates per layer

**Deliverables:**
- Convergence comparison
- Final accuracy analysis
- Best practices guide

---

## 📊 Challenge Problems

### Challenge 1: Memory-Efficient ResNet
**Problem**: Implement a ResNet that uses 50% less memory during training

**Hints:**
- Gradient checkpointing
- In-place operations
- Mixed precision training
- Smaller batch sizes with gradient accumulation

### Challenge 2: Extremely Deep Networks
**Problem**: Train a ResNet with 500+ layers successfully

**Hints:**
- Pre-activation is essential
- Careful initialization
- Learning rate warmup
- Gradient clipping

### Challenge 3: Speed Optimization
**Problem**: Make ResNet inference 2× faster without losing accuracy

**Hints:**
- Operator fusion
- Quantization
- Knowledge distillation
- Neural architecture search

---

## 🔍 Research Questions

Explore these open questions:

1. **Why do residual connections work so well?**
   - Is it gradient flow? Loss landscape? Ensemble effect?
   - Design experiments to test hypotheses

2. **Can we do better than addition?**
   - Try multiplication, concatenation, gating
   - When does each make sense?

3. **How deep is deep enough?**
   - Find the optimal depth for your dataset
   - Diminishing returns vs computational cost

4. **Do we need skip connections at every layer?**
   - Experiment with sparse skip connections
   - Can we learn which connections to keep?

---

## 💡 Tips for Success

1. **Start Simple**: Get basic ResNet working before adding complexity
2. **Visualize Everything**: Plot gradients, features, architectures
3. **Ablation Studies**: Change one thing at a time
4. **Read Papers**: Original papers provide valuable insights
5. **Compare Results**: Benchmark against known baselines
6. **Document**: Keep detailed notes on experiments

---

## 📚 Additional Resources

**Papers to Read:**
1. Deep Residual Learning (He et al., 2015)
2. Identity Mappings in Deep Residual Networks (He et al., 2016)
3. Wide Residual Networks (Zagoruyko, 2016)
4. Aggregated Residual Transformations (Xie et al., 2017)

**Code Repositories:**
- Official PyTorch ResNet implementation
- timm library (PyTorch Image Models)
- Papers with Code leaderboards

**Tutorials:**
- PyTorch official ResNet tutorial
- Fast.ai course on CNNs
- Stanford CS231n lectures

---

## ✅ Self-Assessment Checklist

**Beginner Level:**
- [ ] Can explain what a residual connection is
- [ ] Understand gradient flow benefits
- [ ] Can implement basic residual block
- [ ] Know when dimensions need to change

**Intermediate Level:**
- [ ] Can build complete ResNet architecture
- [ ] Understand different block types
- [ ] Can train ResNet on real dataset
- [ ] Know hyperparameter best practices

**Advanced Level:**
- [ ] Can implement ResNet variants (Pre-act, Wide, Next)
- [ ] Understand SE blocks and attention
- [ ] Can design custom architectures
- [ ] Can conduct ablation studies

---

**Good luck with your learning journey! 🚀**

Remember: The best way to learn is by doing. Start coding, experiment, and don't be afraid to break things!
