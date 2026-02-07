# DeiT: Data-Efficient Image Transformers

## Introduction

**DeiT** (Data-efficient Image Transformer), introduced by Touvron et al. (2021), addresses the central limitation of the original ViT: the requirement for massive pretraining datasets. While ViT needed JFT-300M (300 million images) to match CNN performance, DeiT achieves competitive results training **only on ImageNet-1k** (1.28 million images)—a 230× reduction in training data. DeiT achieves this through a combination of improved training strategies, strong regularization, and a novel **distillation token** mechanism that transfers knowledge from a CNN teacher to the ViT student.

---

## The Data Efficiency Problem

### Why ViTs Need More Data

ViTs lack the inductive biases that make CNNs sample-efficient:

| Inductive Bias | CNN | ViT | Impact |
|---|---|---|---|
| Locality | Built into kernels | Must learn from data | ViT needs more data to discover local patterns |
| Translation equivariance | Exact (weight sharing) | Approximate (learned) | ViT needs diverse spatial examples |
| Scale hierarchy | Explicit (pooling) | Implicit (depth) | ViT needs more layers/data to build hierarchy |

DeiT bridges this gap not by adding inductive biases to the architecture, but by providing them through the **training procedure**—specifically through knowledge distillation from a CNN teacher that already possesses these biases.

---

## Key Contributions

### 1. Improved Training Recipe

DeiT demonstrates that much of ViT's poor performance on ImageNet-1k was due to suboptimal training, not fundamental architectural limitations. The DeiT training recipe includes:

**Optimizer and Schedule:**

```python
training_config = {
    'optimizer': 'AdamW',
    'base_lr': 5e-4,               # With linear scaling: lr = base_lr * batch_size / 512
    'weight_decay': 0.05,
    'warmup_epochs': 5,
    'total_epochs': 300,
    'lr_schedule': 'cosine',
    'batch_size': 1024,
    'gradient_clip': None,
}
```

**Data Augmentation (critical):**

```python
augmentation_config = {
    'rand_augment': 'rand-m9-mstd0.5-inc1',
    'mixup_alpha': 0.8,
    'cutmix_alpha': 1.0,
    'mixup_prob': 1.0,
    'mixup_switch_prob': 0.5,      # 50% chance of CutMix vs Mixup
    'color_jitter': 0.3,
    'random_erasing_prob': 0.25,
}
```

**Regularization:**

```python
regularization_config = {
    'label_smoothing': 0.1,
    'stochastic_depth': 0.1,       # For DeiT-S; 0.2 for DeiT-B
    'dropout': 0.0,                # No standard dropout
}
```

### 2. Distillation Token

DeiT's most novel contribution is the **distillation token**—a learnable embedding, analogous to the CLS token, that is trained to match the output of a pretrained CNN teacher:

```
Input:  [CLS]  [DIST]  [Patch₁]  [Patch₂]  ...  [Patchₙ]
          │      │         │         │              │
          ▼      ▼         ▼         ▼              ▼
     Transformer Encoder (standard self-attention)
          │      │
          ▼      ▼
      MLP Head  Distill Head
          │      │
          ▼      ▼
     ŷ_cls    ŷ_dist  ← Trained with different targets
```

The two tokens serve different purposes:

- **CLS token**: Trained with the standard cross-entropy loss against ground-truth labels
- **Distillation token**: Trained to match the CNN teacher's predictions

### 3. Hard vs. Soft Distillation

**Soft distillation** uses the teacher's class probabilities as soft targets:

$$\mathcal{L}_{\text{soft}} = (1 - \lambda) \mathcal{L}_{\text{CE}}(y, \psi(\mathbf{z}_{\text{cls}})) + \lambda \tau^2 \text{KL}\left(\sigma\!\left(\frac{\mathbf{z}_t}{\tau}\right) \bigg\| \sigma\!\left(\frac{\mathbf{z}_{\text{dist}}}{\tau}\right)\right)$$

where $\sigma(\mathbf{z}_t / \tau)$ is the teacher's softened output, $\tau$ is the temperature, and $\lambda$ balances the two losses.

**Hard distillation** uses the teacher's hard predictions (argmax) as labels:

$$\mathcal{L}_{\text{hard}} = \frac{1}{2} \mathcal{L}_{\text{CE}}(y, \psi(\mathbf{z}_{\text{cls}})) + \frac{1}{2} \mathcal{L}_{\text{CE}}(y_t, \psi(\mathbf{z}_{\text{dist}}))$$

where $y_t = \arg\max_c \mathbf{z}_t^{(c)}$ is the teacher's predicted class.

Surprisingly, **hard distillation outperforms soft distillation** in the DeiT setting. This is attributed to hard labels acting as a strong form of label smoothing combined with data augmentation through the teacher's imperfect predictions.

---

## Why Distillation from CNNs Works

### Transferring Inductive Biases

A key finding is that **CNN teachers are significantly better than transformer teachers** for distilling into ViT students. This is counterintuitive—one might expect architecture-matching to help—but the explanation is that CNN teachers provide complementary inductive biases:

| Teacher Architecture | Student (DeiT-B) Top-1 |
|---|---|
| No distillation | 81.8% |
| DeiT-B teacher (transformer) | 82.6% |
| RegNetY-16GF teacher (CNN) | **83.4%** |

The CNN teacher implicitly teaches the ViT student about locality and translation equivariance through its predictions, without modifying the ViT architecture.

### CLS vs. Distillation Token Behavior

After training, the CLS and distillation tokens learn **different representations**:

- The CLS token correlates more with the ground-truth label distribution
- The distillation token correlates more with the CNN teacher's decision boundaries
- At inference, averaging both predictions outperforms either alone

The cosine similarity between the two token representations is typically 0.06–0.10, indicating they capture substantially different information despite processing the same input.

---

## PyTorch Implementation

### DeiT with Distillation Token

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Stochastic depth: randomly drop residual branches."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = torch.rand(x.shape[0], 1, 1, device=x.device) >= self.drop_prob
        return x * keep / (1 - self.drop_prob)


class DeiTBlock(nn.Module):
    """Transformer block for DeiT (pre-norm with stochastic depth)."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.drop_path(self.attn(h, h, h, need_weights=False)[0])
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class DeiT(nn.Module):
    """
    Data-efficient Image Transformer with distillation token.

    During training, returns (cls_logits, dist_logits).
    During inference, returns the average of both.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        drop_path_rate=0.1,
    ):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

        # CLS token + Distillation token + Position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 2, embed_dim)
        )

        # Stochastic depth schedule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            DeiTBlock(embed_dim, num_heads, mlp_ratio, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Two classification heads
        self.head = nn.Linear(embed_dim, num_classes)
        self.head_dist = nn.Linear(embed_dim, num_classes)

        # Initialize
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        # Prepend CLS and distillation tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_tokens = self.dist_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, dist_tokens, x], dim=1)  # (B, N+2, d)
        x = x + self.pos_embed

        # Transformer encoder
        x = self.blocks(x)
        x = self.norm(x)

        # Outputs from both special tokens
        cls_out = self.head(x[:, 0])
        dist_out = self.head_dist(x[:, 1])

        if self.training:
            return cls_out, dist_out
        else:
            return (cls_out + dist_out) / 2


# Model factory functions
def deit_tiny(num_classes=1000):
    return DeiT(embed_dim=192, depth=12, num_heads=3, num_classes=num_classes)

def deit_small(num_classes=1000):
    return DeiT(embed_dim=384, depth=12, num_heads=6, num_classes=num_classes)

def deit_base(num_classes=1000):
    return DeiT(embed_dim=768, depth=12, num_heads=12,
                num_classes=num_classes, drop_path_rate=0.2)
```

### Training Loop with Hard Distillation

```python
def train_deit_epoch(model, teacher, dataloader, optimizer, device, alpha=0.5):
    """
    Train DeiT for one epoch with hard distillation.

    Parameters
    ----------
    model : DeiT
        Student model.
    teacher : nn.Module
        Pretrained CNN teacher (e.g., RegNetY-16GF), frozen.
    dataloader : DataLoader
        Training data loader.
    optimizer : Optimizer
        AdamW optimizer.
    device : torch.device
        Compute device.
    alpha : float
        Distillation loss weight.

    Returns
    -------
    float
        Average training loss.
    """
    model.train()
    teacher.eval()
    total_loss = 0.0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # Student forward
        cls_logits, dist_logits = model(images)

        # Teacher forward (no gradient)
        with torch.no_grad():
            teacher_logits = teacher(images)
            teacher_labels = teacher_logits.argmax(dim=-1)

        # Combined loss
        loss_cls = F.cross_entropy(cls_logits, labels)
        loss_dist = F.cross_entropy(dist_logits, teacher_labels)
        loss = (1 - alpha) * loss_cls + alpha * loss_dist

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
```

---

## DeiT Model Variants and Results

### Performance Comparison

| Model | Params | ImageNet Top-1 | Throughput (img/s) |
|---|---|---|---|
| DeiT-Ti | 5.7M | 72.2% | 2,536 |
| DeiT-S | 22M | 79.8% | 940 |
| DeiT-B | 86M | 81.8% | 292 |
| DeiT-Ti⚗ | 5.7M | 74.5% | 2,536 |
| DeiT-S⚗ | 22M | 81.2% | 940 |
| DeiT-B⚗ | 86M | 83.4% | 292 |
| DeiT-B⚗↑384 | 86M | 85.2% | 86 |

(⚗ = with distillation, ↑384 = fine-tuned at higher resolution)

### Comparison with CNNs at Similar Compute

| Model | Params | Top-1 | FLOPs |
|---|---|---|---|
| ResNet-50 | 25M | 79.0% | 4.1G |
| DeiT-S⚗ | 22M | 81.2% | 4.6G |
| EfficientNet-B3 | 12M | 81.6% | 1.8G |
| DeiT-B⚗ | 86M | 83.4% | 17.5G |

DeiT matches or exceeds comparable CNNs, though EfficientNet remains more FLOPs-efficient due to its architecture search and compound scaling.

---

## DeiT III and Later Developments

**DeiT III** (Touvron et al., 2022) further improved training with:

1. **3-Augment**: A simpler augmentation strategy (horizontal flip, color jitter, grayscale) that outperforms the complex augmentation pipeline of DeiT
2. **Simple Random Cropping**: Using a larger crop ratio (0.4–1.0 vs. 0.08–1.0) to provide more context per crop
3. **Repeated Augmentation**: Each image appears multiple times per epoch with different augmentations
4. **Binary Cross-Entropy Loss**: Replacing softmax cross-entropy with binary cross-entropy for each class independently, which works better with mixed-label augmentations

```python
# DeiT III simplified augmentation
deit3_augmentation = {
    'horizontal_flip': True,
    'color_jitter': 0.3,
    'grayscale_prob': 0.1,
    'random_resized_crop_scale': (0.4, 1.0),  # Larger min scale
    'repeated_augmentation': 3,  # 3 augmented copies per image
    'loss': 'binary_cross_entropy',
}
```

---

## Quantitative Finance Applications

### Distillation for Financial Models

The DeiT distillation paradigm transfers directly to quantitative finance, where traditional linear factor models serve as "CNN teachers" for deep learning "ViT students":

**Factor Model → Deep Learning Distillation:**

```python
class FactorModelTeacher(nn.Module):
    """Traditional linear factor model as teacher for distillation."""
    def __init__(self, n_factors, n_assets):
        super().__init__()
        # Pre-fitted factor loadings (frozen during distillation)
        self.betas = nn.Parameter(torch.randn(n_assets, n_factors), requires_grad=False)

    def forward(self, factors):
        """
        Parameters
        ----------
        factors : torch.Tensor
            Factor returns, shape (B, n_factors).

        Returns
        -------
        torch.Tensor
            Predicted asset returns, shape (B, n_assets).
        """
        return factors @ self.betas.T


class DeepAlphaStudent(nn.Module):
    """
    Deep model trained with distillation from factor model.

    The CLS-equivalent output captures alpha,
    the DIST-equivalent output captures systematic risk.
    """
    def __init__(self, input_dim, n_assets, embed_dim=128, depth=4, num_heads=4):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=4 * embed_dim,
                batch_first=True, norm_first=True,
            ),
            num_layers=depth,
        )
        self.proj = nn.Linear(input_dim, embed_dim)
        self.alpha_head = nn.Linear(embed_dim, n_assets)   # Alpha predictions
        self.risk_head = nn.Linear(embed_dim, n_assets)    # Systematic risk

    def forward(self, x):
        x = self.proj(x).unsqueeze(1)
        x = self.encoder(x).squeeze(1)
        alpha = self.alpha_head(x)
        risk = self.risk_head(x)
        return alpha, risk
```

The key insight mirrors DeiT's finding: distilling from a model with strong structural assumptions (linear factors ≈ locality) into a flexible model (transformer ≈ ViT) can outperform either approach alone.

### Data Augmentation for Financial Time Series

DeiT's emphasis on data augmentation applies to finance, where data scarcity is a perennial challenge:

| Vision Augmentation | Financial Analog |
|---|---|
| Random crop | Random sub-period sampling |
| Horizontal flip | Time reversal (for symmetric patterns) |
| Color jitter | Feature noise injection |
| Mixup | Cross-asset interpolation |
| CutMix | Regime splicing |
| Random erasing | Missing data simulation |

```python
def financial_mixup(returns_1, returns_2, alpha_1, alpha_2, lam=0.5):
    """
    Mixup augmentation for return prediction.
    Interpolate both features and targets.
    """
    mixed_returns = lam * returns_1 + (1 - lam) * returns_2
    mixed_alpha = lam * alpha_1 + (1 - lam) * alpha_2
    return mixed_returns, mixed_alpha
```

---

## Common Pitfalls

### Pitfall 1: Teacher Quality Matters

A weak or poorly calibrated teacher produces noisy distillation targets that hurt rather than help:

```python
# ❌ Wrong: Using an undertrained teacher
teacher = resnet18(pretrained=False)  # Random weights → noisy targets

# ✅ Correct: Use a strong pretrained teacher
teacher = torchvision.models.regnet_y_16gf(weights='IMAGENET1K_SWAG_E2E_V1')
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False
```

### Pitfall 2: Inference Mode

The model produces different outputs during training and inference:

```python
# During training: two separate outputs
model.train()
cls_out, dist_out = model(images)

# During inference: averaged output
model.eval()
predictions = model(images)  # Average of cls and dist
```

### Pitfall 3: Augmentation Consistency

Both teacher and student must see the **same augmented image** for distillation to be consistent:

```python
# ✅ Correct: Same augmented image to both
augmented = augment(images)
student_out = model(augmented)
teacher_out = teacher(augmented)  # Same augmentation

# ❌ Wrong: Different augmentations
student_out = model(augment(images))
teacher_out = teacher(augment(images))  # Different random augmentation
```

---

## Summary

DeiT demonstrates that Vision Transformers can be trained efficiently on moderately-sized datasets through three key innovations:

1. **Improved training recipe**: Careful optimizer selection, aggressive data augmentation, and regularization eliminate most of the gap between ViTs and CNNs on limited data
2. **Distillation token**: A second classification token trained to match CNN teacher predictions transfers convolutional inductive biases without modifying the architecture
3. **Hard distillation**: Using the teacher's argmax predictions as labels outperforms soft probability matching

For quantitative finance, DeiT's approach suggests a practical pathway: use traditional factor models as teachers to distill structural financial knowledge into flexible deep learning students, combining the interpretability of linear models with the expressiveness of transformers.

---

## References

1. Touvron, H., et al. (2021). Training Data-Efficient Image Transformers & Distillation through Attention. *ICML 2021*.
2. Touvron, H., et al. (2022). DeiT III: Revenge of the ViT. *ECCV 2022*.
3. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. *NeurIPS Workshop 2015*.
4. Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words. *ICLR 2021*.
