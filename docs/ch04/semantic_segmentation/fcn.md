# Fully Convolutional Networks (FCN)

## Learning Objectives

By the end of this section, you will be able to:

- Understand the historical significance of FCN in semantic segmentation
- Explain the transformation from classification to dense prediction networks
- Implement FCN-32s, FCN-16s, and FCN-8s architectures
- Describe skip connections and multi-scale feature fusion
- Compare FCN variants and understand their trade-offs

## Historical Context: The FCN Revolution

Before Fully Convolutional Networks, semantic segmentation relied on patch-based approaches—classifying each pixel by extracting a small patch around it and running it through a classifier. This was computationally expensive and ignored global context.

The seminal 2015 paper by Long, Shelhamer, and Darrell introduced a paradigm shift: **adapt classification networks for dense prediction** by replacing fully connected layers with convolutional layers. This simple yet profound idea enabled:

1. End-to-end training for pixel-wise prediction
2. Efficient computation through convolution
3. Arbitrary input image sizes
4. Transfer learning from ImageNet-pretrained networks

## From Classification to Segmentation

### The Fully Connected Layer Problem

Traditional CNN classifiers (AlexNet, VGG) end with fully connected layers that:

- Require fixed input dimensions
- Output a single vector (class probabilities)
- Destroy spatial information

```
Classification Network:
Input (224×224×3) → Conv layers → FC (4096) → FC (4096) → FC (1000)
                                 ↑
                           Spatial collapse!
                           Position information lost
```

### The FCN Solution: 1×1 Convolutions

FCN replaces fully connected layers with 1×1 convolutions, preserving spatial dimensions:

$$\text{FC layer}: \mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$$

can be reformulated as:

$$\text{1×1 Conv}: \mathbf{Y}_{i,j} = \mathbf{W} \cdot \mathbf{X}_{i,j} + \mathbf{b}$$

where the same weights are applied at each spatial location $(i, j)$.

```python
import torch
import torch.nn as nn
import torchvision.models as models

def convert_fc_to_conv(vgg16_classifier):
    """
    Convert VGG16 classifier (FC layers) to convolutional layers.
    
    Original VGG16 classifier:
        FC: 25088 → 4096
        FC: 4096 → 4096
        FC: 4096 → 1000
    
    Converted to:
        Conv: 512×7×7 → 4096×1×1 (kernel 7×7)
        Conv: 4096×1×1 → 4096×1×1 (kernel 1×1)
        Conv: 4096×1×1 → 1000×1×1 (kernel 1×1)
    """
    conv_classifier = nn.Sequential(
        # FC6: 512 * 7 * 7 = 25088 → 4096
        nn.Conv2d(512, 4096, kernel_size=7, padding=0),
        nn.ReLU(inplace=True),
        nn.Dropout2d(),
        
        # FC7: 4096 → 4096
        nn.Conv2d(4096, 4096, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Dropout2d(),
        
        # Score: 4096 → num_classes
        nn.Conv2d(4096, 1000, kernel_size=1)
    )
    
    return conv_classifier
```

### Adapting Pre-trained Weights

The conversion preserves pre-trained weights by reshaping:

```python
def transfer_fc_to_conv_weights(fc_weight, conv_layer, spatial_dim):
    """
    Transfer weights from FC layer to equivalent Conv layer.
    
    Args:
        fc_weight: Weight matrix of shape (out_features, in_features)
        conv_layer: Target convolutional layer
        spatial_dim: Spatial dimensions for the conv kernel (e.g., 7 for first FC)
    """
    out_features, in_features = fc_weight.shape
    in_channels = in_features // (spatial_dim * spatial_dim)
    
    # Reshape: (out, in) → (out, in_ch, h, w)
    conv_weight = fc_weight.view(out_features, in_channels, spatial_dim, spatial_dim)
    
    conv_layer.weight.data = conv_weight
    return conv_layer
```

## FCN Architecture Variants

The original paper introduced three variants with progressively finer predictions.

### FCN-32s: Single-Scale Prediction

FCN-32s produces predictions at 1/32 of the input resolution, then upsamples directly.

```
Input (H×W)
    ↓
VGG-16 Encoder
    ↓
Score layer (H/32×W/32×K)
    ↓
Upsample 32× (bilinear or learned)
    ↓
Output (H×W×K)
```

**Limitation**: Coarse predictions—fine details are lost.

```python
class FCN32s(nn.Module):
    """
    FCN-32s: Simplest FCN variant with 32× upsampling.
    
    Based on VGG-16 backbone, replaces FC layers with conv,
    adds score layer, and upsamples 32× to original resolution.
    """
    def __init__(self, num_classes=21, pretrained=True):
        super().__init__()
        
        # Load pre-trained VGG-16 features
        vgg16 = models.vgg16(pretrained=pretrained)
        self.features = vgg16.features  # Conv layers only
        
        # Converted classifier (FC → Conv)
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )
        
        # Score layer: predicts class scores at 1/32 resolution
        self.score = nn.Conv2d(4096, num_classes, kernel_size=1)
        
        # 32× upsampling (can be bilinear or transposed conv)
        self.upsample = nn.ConvTranspose2d(
            num_classes, num_classes, 
            kernel_size=64, stride=32, padding=16,
            bias=False
        )
        
        # Initialize upsampling with bilinear weights
        self._init_bilinear_upsampling()
    
    def _init_bilinear_upsampling(self):
        """Initialize transposed conv with bilinear interpolation weights."""
        factor = 32
        kernel_size = 64
        
        # Create bilinear kernel
        og = (kernel_size - 1) / 2.0
        filter_2d = torch.zeros(kernel_size, kernel_size)
        for i in range(kernel_size):
            for j in range(kernel_size):
                filter_2d[i, j] = (1 - abs(i - og) / factor) * (1 - abs(j - og) / factor)
        
        # Apply to all channels
        weight = torch.zeros(self.upsample.weight.shape)
        for i in range(weight.shape[0]):
            weight[i, i] = filter_2d
        
        self.upsample.weight.data = weight
        self.upsample.weight.requires_grad = False  # Fix bilinear weights
    
    def forward(self, x):
        input_size = x.shape[2:]  # Store original size
        
        # VGG-16 features: H×W → H/32×W/32
        x = self.features(x)
        
        # Classifier
        x = self.classifier(x)
        
        # Score
        x = self.score(x)
        
        # Upsample 32×
        x = self.upsample(x)
        
        # Crop to match input size (transposed conv may overshoot)
        x = x[:, :, :input_size[0], :input_size[1]]
        
        return x
```

### FCN-16s: Two-Scale Fusion

FCN-16s combines predictions from pool4 (1/16 resolution) with the coarse FCN-32s predictions, enabling finer details.

```
Input (H×W)
    ↓
VGG-16 pool4 → Score (H/16) ──────────┐
    ↓                                  │
VGG-16 pool5 → FCN-32s Score (H/32)   │
    ↓                                  │
Upsample 2× ───────────────────────────┤
    ↓                                  ↓
              Fuse (element-wise sum)
                       ↓
              Upsample 16×
                       ↓
              Output (H×W×K)
```

```python
class FCN16s(nn.Module):
    """
    FCN-16s: Combines pool4 and pool5 predictions.
    
    Achieves finer segmentation by fusing 1/16 and 1/32 scale features.
    """
    def __init__(self, num_classes=21, pretrained=True):
        super().__init__()
        
        vgg16 = models.vgg16(pretrained=pretrained)
        features = list(vgg16.features.children())
        
        # Split VGG features at pool4
        self.features_to_pool4 = nn.Sequential(*features[:24])  # Up to pool4
        self.features_pool4_to_pool5 = nn.Sequential(*features[24:])  # pool4 to pool5
        
        # Classifier (same as FCN-32s)
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )
        
        # Score layers for each scale
        self.score_pool5 = nn.Conv2d(4096, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        
        # 2× upsampling to fuse pool5 with pool4
        self.upsample_2x = nn.ConvTranspose2d(
            num_classes, num_classes,
            kernel_size=4, stride=2, padding=1,
            bias=False
        )
        
        # 16× upsampling for final output
        self.upsample_16x = nn.ConvTranspose2d(
            num_classes, num_classes,
            kernel_size=32, stride=16, padding=8,
            bias=False
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize score layers with zero mean and upsampling with bilinear."""
        for score in [self.score_pool5, self.score_pool4]:
            nn.init.zeros_(score.weight)
            nn.init.zeros_(score.bias)
        
        # Initialize transposed convs with bilinear interpolation
        self._init_bilinear(self.upsample_2x, 2)
        self._init_bilinear(self.upsample_16x, 16)
    
    def _init_bilinear(self, layer, factor):
        """Initialize transposed conv with bilinear weights."""
        kernel_size = layer.kernel_size[0]
        og = (kernel_size - 1) / 2.0
        
        filter_2d = torch.zeros(kernel_size, kernel_size)
        for i in range(kernel_size):
            for j in range(kernel_size):
                filter_2d[i, j] = (1 - abs(i - og) / factor) * (1 - abs(j - og) / factor)
        
        weight = torch.zeros(layer.weight.shape)
        for i in range(weight.shape[0]):
            weight[i, i] = filter_2d
        
        layer.weight.data = weight
        layer.weight.requires_grad = False
    
    def forward(self, x):
        input_size = x.shape[2:]
        
        # Extract pool4 features
        pool4 = self.features_to_pool4(x)  # H/16 × W/16 × 512
        
        # Extract pool5 features
        pool5 = self.features_pool4_to_pool5(pool4)  # H/32 × W/32 × 512
        
        # FCN-32s path
        x = self.classifier(pool5)
        score_pool5 = self.score_pool5(x)  # H/32 × W/32 × K
        
        # Upsample pool5 scores to pool4 resolution
        score_pool5_2x = self.upsample_2x(score_pool5)  # H/16 × W/16 × K
        
        # Score from pool4
        score_pool4 = self.score_pool4(pool4)  # H/16 × W/16 × K
        
        # Fuse: element-wise sum
        fused = score_pool5_2x + score_pool4
        
        # Final 16× upsampling
        output = self.upsample_16x(fused)
        
        # Crop to input size
        output = output[:, :, :input_size[0], :input_size[1]]
        
        return output
```

### FCN-8s: Three-Scale Fusion

FCN-8s adds pool3 features (1/8 resolution) for even finer predictions.

```
Input (H×W)
    ↓
VGG-16 pool3 → Score (H/8) ───────────────────────────┐
    ↓                                                  │
VGG-16 pool4 → Score (H/16) ──────────┐               │
    ↓                                  │               │
VGG-16 pool5 → FCN-32s Score (H/32)   │               │
    ↓                                  │               │
Upsample 2× ───────────────────────────┤               │
    ↓                                  ↓               │
              Fuse (sum) ──────────────┘               │
                   ↓                                   │
            Upsample 2× ───────────────────────────────┤
                   ↓                                   ↓
                         Fuse (sum) ───────────────────┘
                              ↓
                       Upsample 8×
                              ↓
                       Output (H×W×K)
```

```python
class FCN8s(nn.Module):
    """
    FCN-8s: Combines pool3, pool4, and pool5 predictions.
    
    This achieves the finest segmentation among FCN variants by
    incorporating three scales of features.
    """
    def __init__(self, num_classes=21, pretrained=True):
        super().__init__()
        
        vgg16 = models.vgg16(pretrained=pretrained)
        features = list(vgg16.features.children())
        
        # Split VGG features at pool3 and pool4
        self.features_to_pool3 = nn.Sequential(*features[:17])   # Up to pool3
        self.features_pool3_to_pool4 = nn.Sequential(*features[17:24])  # pool3 to pool4
        self.features_pool4_to_pool5 = nn.Sequential(*features[24:])    # pool4 to pool5
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )
        
        # Score layers
        self.score_pool5 = nn.Conv2d(4096, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        
        # Upsampling layers
        self.upsample_2x_pool5 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.upsample_2x_fused = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.upsample_8x = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=16, stride=8, padding=4, bias=False
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize all score and upsampling layers."""
        for score in [self.score_pool5, self.score_pool4, self.score_pool3]:
            nn.init.zeros_(score.weight)
            nn.init.zeros_(score.bias)
        
        self._init_bilinear(self.upsample_2x_pool5, 2)
        self._init_bilinear(self.upsample_2x_fused, 2)
        self._init_bilinear(self.upsample_8x, 8)
    
    def _init_bilinear(self, layer, factor):
        """Initialize with bilinear interpolation weights."""
        kernel_size = layer.kernel_size[0]
        og = (kernel_size - 1) / 2.0
        
        filter_2d = torch.zeros(kernel_size, kernel_size)
        for i in range(kernel_size):
            for j in range(kernel_size):
                filter_2d[i, j] = (1 - abs(i - og) / factor) * (1 - abs(j - og) / factor)
        
        weight = torch.zeros(layer.weight.shape)
        for i in range(weight.shape[0]):
            weight[i, i] = filter_2d
        
        layer.weight.data = weight
        layer.weight.requires_grad = False
    
    def forward(self, x):
        input_size = x.shape[2:]
        
        # Extract multi-scale features
        pool3 = self.features_to_pool3(x)         # H/8 × W/8 × 256
        pool4 = self.features_pool3_to_pool4(pool3)  # H/16 × W/16 × 512
        pool5 = self.features_pool4_to_pool5(pool4)  # H/32 × W/32 × 512
        
        # FCN-32s path
        x = self.classifier(pool5)
        score_pool5 = self.score_pool5(x)  # H/32 × W/32 × K
        
        # First fusion: pool5 + pool4
        score_pool5_2x = self.upsample_2x_pool5(score_pool5)
        score_pool4 = self.score_pool4(pool4)
        fused_16 = score_pool5_2x + score_pool4  # H/16 × W/16 × K
        
        # Second fusion: fused_16 + pool3
        fused_16_2x = self.upsample_2x_fused(fused_16)
        score_pool3 = self.score_pool3(pool3)
        fused_8 = fused_16_2x + score_pool3  # H/8 × W/8 × K
        
        # Final upsampling
        output = self.upsample_8x(fused_8)
        
        # Crop to input size
        output = output[:, :, :input_size[0], :input_size[1]]
        
        return output
```

## Understanding Skip Connections in FCN

### The Information Problem

As features propagate through the encoder, they undergo a fundamental transformation:

| Layer | Spatial Res | Channels | Information Type |
|-------|-------------|----------|------------------|
| Input | H × W | 3 | Raw pixels |
| Conv1-2 | H × W | 64 | Low-level edges |
| Pool1 | H/2 × W/2 | 64 | Texture patterns |
| Pool2 | H/4 × W/4 | 128 | Part structures |
| Pool3 | H/8 × W/8 | 256 | Object parts |
| Pool4 | H/16 × W/16 | 512 | Objects |
| Pool5 | H/32 × W/32 | 512 | Scene semantics |

**Key insight**: High-level features know "what" but not "where"; low-level features know "where" but not "what".

### Fusion Strategy

FCN uses **additive fusion** to combine multi-scale predictions:

$$\mathbf{S}_{\text{final}} = \mathbf{S}_{\text{coarse}} \uparrow_n + \mathbf{S}_{\text{fine}}$$

where $\uparrow_n$ denotes $n\times$ upsampling.

Why addition (not concatenation)?
- Memory efficient: doesn't increase channel count
- Forces features to be in the same "prediction space"
- Score layers must learn complementary information

## Training FCN

### Loss Function

FCN uses per-pixel cross-entropy loss:

$$\mathcal{L} = -\frac{1}{HW}\sum_{i=1}^{H}\sum_{j=1}^{W}\sum_{k=1}^{K} y_{ijk} \log(\hat{y}_{ijk})$$

where $y_{ijk}$ is the one-hot encoded ground truth and $\hat{y}_{ijk}$ is the predicted probability.

```python
def fcn_loss(pred, target, ignore_index=255):
    """
    Compute pixel-wise cross-entropy loss for FCN.
    
    Args:
        pred: Predicted logits (B, K, H, W)
        target: Ground truth labels (B, H, W)
        ignore_index: Label to ignore (e.g., boundary pixels)
    
    Returns:
        Average cross-entropy loss
    """
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    return criterion(pred, target)
```

### Training Strategy

The original paper recommended:

1. **Stage-wise training**:
   - First train FCN-32s
   - Initialize FCN-16s from FCN-32s, fine-tune
   - Initialize FCN-8s from FCN-16s, fine-tune

2. **Learning rate**: Start low (1e-10) for fine-tuning pre-trained layers

3. **Momentum SGD**: Standard optimizer with momentum 0.9

```python
def create_fcn_optimizer(model, lr=1e-4, momentum=0.9, weight_decay=5e-4):
    """
    Create optimizer with different learning rates for different parts.
    
    Pre-trained layers: lower learning rate
    New layers (score, upsample): higher learning rate
    """
    pretrained_params = []
    new_params = []
    
    for name, param in model.named_parameters():
        if 'score' in name or 'upsample' in name:
            new_params.append(param)
        else:
            pretrained_params.append(param)
    
    optimizer = torch.optim.SGD([
        {'params': pretrained_params, 'lr': lr * 0.1},  # Lower LR for pretrained
        {'params': new_params, 'lr': lr}                 # Higher LR for new
    ], momentum=momentum, weight_decay=weight_decay)
    
    return optimizer
```

## Performance Comparison

Results on PASCAL VOC 2012 (mIoU):

| Model | mIoU | Notes |
|-------|------|-------|
| FCN-32s | 59.4% | Single scale, coarse |
| FCN-16s | 62.4% | +3.0% from pool4 fusion |
| FCN-8s | 62.7% | +0.3% from pool3 fusion |

**Observation**: The jump from FCN-32s to FCN-16s is significant, while FCN-16s to FCN-8s provides marginal improvement. This suggests pool4 features carry the most useful fine-grained information.

## Limitations and Legacy

### Limitations

1. **Fixed encoder receptive field**: VGG's receptive field may not capture very large objects
2. **Simple upsampling**: Bilinear interpolation loses information
3. **No global context**: Limited ability to reason about scene-level information
4. **Memory intensive**: Full-resolution feature maps in decoder

### Legacy and Impact

FCN established foundational principles that persist today:

1. **Fully convolutional design**: No fully connected layers
2. **Transfer learning**: Adapt classification networks
3. **Multi-scale fusion**: Combine features from multiple resolutions
4. **End-to-end training**: Direct optimization for segmentation

Modern architectures like U-Net, DeepLab, and PSPNet build directly on these principles.

## Summary

Fully Convolutional Networks transformed semantic segmentation from a patch-based classification problem into an efficient, end-to-end trainable dense prediction task. By replacing fully connected layers with convolutions and introducing multi-scale skip connections, FCN achieved state-of-the-art results while enabling transfer learning from classification networks.

The progression from FCN-32s through FCN-8s demonstrates the importance of multi-scale feature fusion—a principle that remains central to modern segmentation architectures. Understanding FCN provides essential intuition for appreciating later advances like U-Net's symmetric encoder-decoder and DeepLab's atrous convolutions.

## References

1. Long, J., Shelhamer, E., & Darrell, T. (2015). Fully Convolutional Networks for Semantic Segmentation. CVPR.
2. Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. ICLR.
3. Noh, H., Hong, S., & Han, B. (2015). Learning Deconvolution Network for Semantic Segmentation. ICCV.
