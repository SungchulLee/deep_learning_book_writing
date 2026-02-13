# Vision Transformers (ViT): Bridge between CNNs and Transformers

A comprehensive Python implementation and educational resource for understanding Vision Transformers and how they bridge traditional CNNs with modern Transformer architectures.

## 📚 Overview

Vision Transformers (ViT) revolutionized computer vision by applying the Transformer architecture (originally from NLP) directly to images. This repository provides:

- **Complete ViT implementation** from scratch in PyTorch
- **CNN vs ViT comparison** to understand architectural differences
- **Hybrid models** that combine both approaches
- **Training utilities** for practical applications
- **Visualization tools** to understand attention mechanisms
- **Educational examples** demonstrating key concepts

## 🌉 The Bridge: From CNNs to Transformers

### Traditional CNN Approach
- **Input**: Continuous pixel grid
- **Processing**: Convolutional filters with local receptive fields
- **Hierarchy**: Gradually builds spatial abstractions
- **Bias**: Strong inductive biases (locality, translation equivariance)

### Vision Transformer Approach
- **Input**: Sequence of image patches (like words in NLP)
- **Processing**: Self-attention across all patches
- **Hierarchy**: Minimal - global attention from layer 1
- **Bias**: Weak inductive biases (learns patterns from data)

### The Bridge
ViT treats images as **sequences of patches**, making them compatible with Transformer architecture:
1. Split image into fixed-size patches (e.g., 16×16 pixels)
2. Linearly project each patch to an embedding
3. Add positional encoding to retain spatial information
4. Process through standard Transformer encoder
5. Use [CLS] token for classification

## 📁 File Structure

```
.
├── vit_model.py           # Core ViT implementation
├── train_vit.py           # Training utilities and loops
├── cnn_vs_vit.py          # Architecture comparison
├── visualizations.py      # Attention and patch visualization
├── examples.py            # Educational examples and demos
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from vit_model import create_vit_base
import torch

# Create model
model = create_vit_base(n_classes=10)

# Random input image
image = torch.randn(1, 3, 224, 224)

# Forward pass
output = model(image)
print(output.shape)  # (1, 10)
```

### Run Examples

```bash
# Run all educational examples
python examples.py

# Compare CNN vs ViT architectures
python cnn_vs_vit.py

# Train a model
python train_vit.py
```

## 🏗️ Architecture Details

### 1. Patch Embedding
Converts images into sequences of patch embeddings:

```python
from vit_model import PatchEmbedding

patch_embed = PatchEmbedding(
    img_size=224,
    patch_size=16,
    in_channels=3,
    embed_dim=768
)

# Input: (B, 3, 224, 224)
# Output: (B, 196, 768)  # 196 = (224/16)²
```

**Key insight**: This is similar to using a large-stride convolution, bridging continuous images to discrete tokens.

### 2. Self-Attention
Multi-head self-attention allows each patch to attend to all other patches:

```python
from vit_model import MultiHeadAttention

attention = MultiHeadAttention(
    embed_dim=768,
    n_heads=12,
    dropout=0.1
)
```

**Key insight**: Unlike CNNs which only see local neighbors, ViT has global receptive field from the first layer.

### 3. Transformer Encoder
Standard transformer blocks with residual connections:

```python
from vit_model import TransformerBlock

block = TransformerBlock(
    embed_dim=768,
    n_heads=12,
    mlp_ratio=4,
    dropout=0.1
)
```

### 4. Classification Head
Uses the [CLS] token for image classification:

```python
# CLS token is prepended to sequence
# Final classification uses only the CLS token representation
cls_output = transformer_output[:, 0]  # Take CLS token
prediction = classifier(cls_output)
```

## 📊 Model Variants

| Model      | Parameters | Embed Dim | Depth | Heads | Patch Size |
|------------|-----------|-----------|-------|-------|------------|
| ViT-Tiny   | 5M        | 192       | 12    | 3     | 16         |
| ViT-Small  | 22M       | 384       | 12    | 6     | 16         |
| ViT-Base   | 86M       | 768       | 12    | 12    | 16         |
| ViT-Large  | 307M      | 1024      | 24    | 16    | 16         |

### Creating Models

```python
from vit_model import (
    create_vit_tiny,
    create_vit_small,
    create_vit_base,
    create_vit_large
)

# For quick experiments
model = create_vit_tiny(n_classes=10)

# For production
model = create_vit_base(n_classes=1000)
```

## 🔬 CNN vs ViT Comparison

### Receptive Fields
- **CNN**: Starts local, grows with depth (hierarchical)
- **ViT**: Global from first layer (all patches visible)

### Computational Complexity
- **CNN**: O(k²·C·H·W) where k = kernel size
- **ViT**: O(N²·D) where N = number of patches

### Data Efficiency
- **CNN**: Works well with small datasets (~10k images)
- **ViT**: Needs large datasets or pretraining (~1M+ images)

### Inductive Biases
- **CNN**: Strong (locality, translation equivariance)
- **ViT**: Weak (learns from data, more flexible)

### When to Use Each

**Use CNN when:**
- Dataset is small (<100k images)
- Need fast inference
- Spatial locality is important
- Limited computational resources

**Use ViT when:**
- Large dataset available (>1M images)
- Need to model long-range dependencies
- Have computational resources for training
- Transfer learning from pretrained models

**Use Hybrid when:**
- Want benefits of both architectures
- Medium-sized dataset (100k-1M images)
- Balance efficiency and performance

## 🎨 Visualization

### Attention Maps
Visualize what the model focuses on:

```python
from visualizations import AttentionVisualizer

visualizer = AttentionVisualizer(model, device='cuda')
visualizer.visualize_attention(
    image=image,
    layer_idx=-1,  # Last layer
    head_idx=None,  # Average all heads
    save_path='attention_map.png'
)
```

### Patch Embedding
See how images are divided into patches:

```python
from visualizations import visualize_patch_embedding

visualize_patch_embedding(
    image=image,
    patch_size=16,
    save_path='patches.png'
)
```

### Receptive Field Comparison

```python
from visualizations import compare_receptive_fields

compare_receptive_fields()  # Creates comparison visualization
```

## 🎓 Training

### Basic Training

```python
from train_vit import Trainer, get_data_loaders
from vit_model import create_vit_tiny

# Prepare data
train_loader, val_loader = get_data_loaders(
    data_dir='./data',
    batch_size=128
)

# Create model and trainer
model = create_vit_tiny(n_classes=10)
trainer = Trainer(model, device='cuda')

# Train
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    n_epochs=50,
    learning_rate=3e-4
)
```

### Transfer Learning

```python
# Load pretrained model
model = create_vit_base(n_classes=1000)

# Replace classification head
n_new_classes = 10
model.head = torch.nn.Linear(model.head.in_features, n_new_classes)

# Freeze backbone (optional)
for param in model.patch_embed.parameters():
    param.requires_grad = False
for block in model.blocks:
    for param in block.parameters():
        param.requires_grad = False

# Only classification head will be trained
```

## 🔑 Key Concepts

### 1. Patch Embedding
**Why?** Transformers work with sequences. We need to convert 2D images to 1D sequences.

**How?** 
- Divide image into fixed-size patches
- Flatten each patch
- Linear projection to embedding dimension

**Analogy**: Like how text is tokenized into words, images are "tokenized" into patches.

### 2. Positional Encoding
**Why?** Self-attention is permutation-invariant. We need to tell the model where patches are located.

**How?**
- Learnable positional embeddings added to patch embeddings
- Encodes spatial relationships between patches

### 3. [CLS] Token
**Why?** Need a consistent way to aggregate information for classification.

**How?**
- Special learnable token prepended to sequence
- Attends to all patches during self-attention
- Used for final classification prediction

**Analogy**: Like the [CLS] token in BERT for sentence classification.

### 4. Self-Attention
**Why?** Allows modeling relationships between all patches simultaneously.

**How?**
- Each patch creates Query, Key, Value vectors
- Attention weights = softmax(Q @ K^T / √d)
- Output = Attention @ V

**Advantage**: Global receptive field from first layer, unlike CNNs.

### 5. Multi-Head Attention
**Why?** Allows model to focus on different aspects simultaneously.

**How?**
- Split embedding into multiple heads
- Each head performs self-attention independently
- Concatenate and project results

## 📈 Performance Tips

### Training Stability
```python
# Use label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Warmup scheduler
from torch.optim.lr_scheduler import LambdaLR

def warmup_schedule(step):
    warmup_steps = 10000
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0

scheduler = LambdaLR(optimizer, warmup_schedule)
```

### Data Augmentation
```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4),
    transforms.AutoAugment(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])
```

### Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for images, labels in train_loader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(images)
        loss = criterion(outputs, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## 🔍 Hybrid Architectures

The `HybridCNNViT` model combines strengths of both:

```python
from cnn_vs_vit import HybridCNNViT

model = HybridCNNViT(n_classes=10, embed_dim=384, depth=6)
```

**Architecture:**
1. **CNN Stem**: Extract low-level features (edges, textures)
2. **Reshape**: Convert CNN features to sequence
3. **Transformer**: Model global relationships
4. **Classification**: Final prediction

**Advantages:**
- Better inductive biases than pure ViT
- More data-efficient
- Combines local and global reasoning
- Good compromise for medium datasets

## 📚 References

1. **Original Paper**: Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ICLR 2021)

2. **Key Insights**:
   - Pure transformer architecture can match CNN performance
   - Requires large-scale pretraining (JFT-300M)
   - When pretrained, transfers well to smaller datasets
   - Attention patterns show interpretable behavior

3. **Related Work**:
   - DeiT: Data-efficient image transformers
   - Swin Transformer: Hierarchical vision transformer
   - BEiT: BERT pre-training for image transformers
   - MAE: Masked autoencoders for self-supervised learning

## 🛠️ Advanced Usage

### Custom Patch Size
```python
from vit_model import VisionTransformer

# Smaller patches = more tokens = more computation
model = VisionTransformer(
    img_size=224,
    patch_size=8,  # 28×28 = 784 patches
    embed_dim=768,
    depth=12,
    n_heads=12,
    n_classes=10
)
```

### Custom Architecture
```python
# Deeper model with fewer parameters
model = VisionTransformer(
    img_size=224,
    patch_size=16,
    embed_dim=384,  # Smaller dimension
    depth=24,       # But deeper
    n_heads=6,
    n_classes=10
)
```

## 🤝 Contributing

This is an educational resource. Feel free to:
- Add more visualizations
- Implement additional ViT variants (DeiT, Swin, etc.)
- Improve documentation
- Add more examples

## 📄 License

MIT License - feel free to use for learning and research.

## 🙏 Acknowledgments

- Original ViT paper by Google Research
- PyTorch team for the excellent framework
- Vision transformer community for insights

## 📧 Contact

For questions or suggestions, please open an issue or reach out!

---

**Happy Learning! 🚀**

Understanding how Vision Transformers bridge CNNs and Transformers opens up new possibilities in computer vision. Experiment with the code, visualize attention patterns, and see how these models process images differently from traditional CNNs!
