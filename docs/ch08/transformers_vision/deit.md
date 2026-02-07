# Data-efficient Image Transformers (DeiT)

## Introduction

Data-efficient Image Transformers (DeiT), introduced by Touvron et al. (2021), address one of ViT's main limitations: the need for massive pretraining datasets. DeiT achieves competitive performance training only on ImageNet-1K (1.2M images) through improved training strategies and a novel distillation approach.

## Motivation

Original ViT requires pretraining on JFT-300M (300M images) to match CNN performance. This creates practical barriers:

- Computational cost of large-scale pretraining
- Access to proprietary datasets
- Environmental impact of training

DeiT demonstrates that with proper training techniques, transformers can be data-efficient, making them accessible for broader research and applications.

## Key Innovations

### 1. Strong Data Augmentation

DeiT employs aggressive data augmentation:

```python
from torchvision import transforms
from timm.data.auto_augment import rand_augment_transform
from timm.data.mixup import Mixup
from timm.data.random_erasing import RandomErasing

def create_deit_transforms(img_size=224):
    """DeiT training augmentation pipeline."""
    
    # Base transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        
        # RandAugment (magnitude 9, 2 operations)
        rand_augment_transform(
            config_str='rand-m9-mstd0.5-inc1',
            hparams={'translate_const': 117}
        ),
        
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        
        # Random erasing
        RandomErasing(probability=0.25)
    ])
    
    return train_transform
```

### 2. Regularization Techniques

DeiT uses multiple regularization strategies:

```python
class DeiTTrainer:
    def __init__(self, model, n_classes=1000):
        self.model = model
        
        # Label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Mixup and CutMix
        self.mixup = Mixup(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            prob=1.0,
            switch_prob=0.5,
            num_classes=n_classes
        )
        
        # Stochastic depth (drop path)
        self.drop_path_rate = 0.1
        
    def train_step(self, images, labels):
        # Apply mixup/cutmix
        images, labels = self.mixup(images, labels)
        
        # Forward pass
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        
        return loss
```

### 3. Knowledge Distillation Token

DeiT's most innovative contribution is the distillation token—a learnable token that learns from a CNN teacher:

```python
class DeiT(nn.Module):
    """
    Data-efficient Image Transformer with distillation.
    
    Adds a distillation token that learns from a CNN teacher,
    enabling knowledge transfer without the teacher at inference.
    """
    def __init__(self, 
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 n_classes=1000,
                 embed_dim=768,
                 depth=12,
                 n_heads=12,
                 mlp_ratio=4,
                 dropout=0.0,
                 drop_path_rate=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        n_patches = self.patch_embed.n_patches
        
        # CLS token for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Distillation token (DeiT's key innovation)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embeddings (N patches + CLS + distillation)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, n_patches + 2, embed_dim)
        )
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.ModuleList([
            TransformerBlockWithDropPath(
                embed_dim, n_heads, mlp_ratio, dropout, dpr[i]
            )
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Two classification heads
        self.head = nn.Linear(embed_dim, n_classes)      # From CLS token
        self.head_dist = nn.Linear(embed_dim, n_classes) # From distillation token
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        B = x.shape[0]
        
        x = self.patch_embed(x)
        
        # Prepend CLS and distillation tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_tokens = self.dist_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, dist_tokens, x], dim=1)
        
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        # Get outputs from both tokens
        cls_output = self.head(x[:, 0])
        dist_output = self.head_dist(x[:, 1])
        
        if self.training:
            return cls_output, dist_output
        else:
            # Average at inference
            return (cls_output + dist_output) / 2
```

## Distillation Training

### Soft Distillation

Uses KL divergence between student and teacher outputs:

```python
def soft_distillation_loss(student_output, teacher_output, labels, 
                          temperature=3.0, alpha=0.5):
    """
    Soft distillation loss combining hard labels and soft teacher predictions.
    
    Args:
        student_output: Student model predictions
        teacher_output: Teacher model predictions
        labels: Ground truth labels
        temperature: Softmax temperature for distillation
        alpha: Weight for hard loss (1-alpha for soft loss)
    """
    # Hard loss (cross-entropy with true labels)
    hard_loss = F.cross_entropy(student_output, labels)
    
    # Soft loss (KL divergence with teacher)
    soft_labels = F.softmax(teacher_output / temperature, dim=1)
    soft_pred = F.log_softmax(student_output / temperature, dim=1)
    soft_loss = F.kl_div(soft_pred, soft_labels, reduction='batchmean')
    soft_loss = soft_loss * (temperature ** 2)  # Scale by T^2
    
    return alpha * hard_loss + (1 - alpha) * soft_loss
```

### Hard Distillation

Uses teacher's hard predictions as labels:

```python
def hard_distillation_loss(cls_output, dist_output, teacher_output, labels):
    """
    Hard distillation as used in DeiT.
    
    CLS token trained on true labels.
    Distillation token trained on teacher's hard predictions.
    """
    # CLS token: standard cross-entropy
    cls_loss = F.cross_entropy(cls_output, labels)
    
    # Distillation token: cross-entropy with teacher predictions
    teacher_labels = teacher_output.argmax(dim=1)
    dist_loss = F.cross_entropy(dist_output, teacher_labels)
    
    return (cls_loss + dist_loss) / 2
```

## Complete Training Pipeline

```python
class DeiTDistillationTrainer:
    """Complete DeiT training with CNN teacher distillation."""
    
    def __init__(self, student, teacher, device='cuda'):
        self.student = student.to(device)
        self.teacher = teacher.to(device).eval()
        self.device = device
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        # Optimizer with layer-wise learning rate decay
        self.optimizer = self._create_optimizer()
        
        # Mixup
        self.mixup = Mixup(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            prob=1.0,
            switch_prob=0.5,
            num_classes=1000
        )
        
    def _create_optimizer(self, lr=1e-3, weight_decay=0.05):
        """Create optimizer with layer-wise LR decay."""
        param_groups = []
        
        # Different LR for different layers
        lr_scales = {}
        depth = len(self.student.blocks)
        for i in range(depth):
            lr_scales[f'blocks.{i}'] = 0.65 ** (depth - i - 1)
        lr_scales['patch_embed'] = lr_scales['blocks.0']
        lr_scales['cls_token'] = 1.0
        lr_scales['dist_token'] = 1.0
        lr_scales['pos_embed'] = 1.0
        lr_scales['head'] = 1.0
        lr_scales['head_dist'] = 1.0
        
        for name, param in self.student.named_parameters():
            if not param.requires_grad:
                continue
                
            # Find matching LR scale
            scale = 1.0
            for key, val in lr_scales.items():
                if key in name:
                    scale = val
                    break
                    
            param_groups.append({
                'params': [param],
                'lr': lr * scale,
                'weight_decay': weight_decay if param.dim() > 1 else 0
            })
            
        return torch.optim.AdamW(param_groups)
        
    def train_epoch(self, dataloader, scheduler):
        self.student.train()
        total_loss = 0
        
        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Apply mixup
            images, labels = self.mixup(images, labels)
            
            # Get teacher predictions
            with torch.no_grad():
                teacher_output = self.teacher(images)
            
            # Student forward
            cls_output, dist_output = self.student(images)
            
            # Distillation loss
            loss = hard_distillation_loss(
                cls_output, dist_output, teacher_output, labels
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.student.parameters(), max_norm=1.0
            )
            
            self.optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
```

## Model Variants

DeiT provides several model sizes:

| Model | Params | Embed Dim | Depth | Heads | ImageNet Top-1 |
|-------|--------|-----------|-------|-------|----------------|
| DeiT-Ti | 5M | 192 | 12 | 3 | 72.2% |
| DeiT-S | 22M | 384 | 12 | 6 | 79.8% |
| DeiT-B | 86M | 768 | 12 | 12 | 81.8% |
| DeiT-B⇑ | 86M | 768 | 12 | 12 | 83.1%* |

*With distillation from RegNetY-16GF teacher

## Training Recipe

The complete DeiT training recipe:

```python
# Training hyperparameters
config = {
    # Model
    'model': 'deit_base_patch16_224',
    'img_size': 224,
    'patch_size': 16,
    
    # Optimization
    'epochs': 300,
    'batch_size': 1024,  # Effective batch size
    'lr': 5e-4,
    'min_lr': 1e-5,
    'weight_decay': 0.05,
    'warmup_epochs': 5,
    
    # Augmentation
    'color_jitter': 0.4,
    'aa': 'rand-m9-mstd0.5-inc1',  # RandAugment
    'reprob': 0.25,  # Random erasing prob
    'remode': 'pixel',
    
    # Mixup/CutMix
    'mixup': 0.8,
    'cutmix': 1.0,
    'mixup_prob': 1.0,
    'mixup_switch_prob': 0.5,
    
    # Regularization
    'drop_path': 0.1,
    'label_smoothing': 0.1,
    
    # Distillation
    'teacher': 'regnety_160',  # RegNetY-16GF
    'distillation_type': 'hard',
    'distillation_alpha': 0.5,
    'distillation_tau': 3.0,
}
```

## Key Takeaways

1. **Training Matters**: Proper training techniques can compensate for limited data
2. **Distillation Works**: CNN teachers can effectively transfer knowledge to transformers
3. **Regularization is Crucial**: Multiple regularization strategies prevent overfitting
4. **Practical Impact**: DeiT makes ViT accessible without massive compute resources

## Applications in Finance

DeiT's data efficiency makes it suitable for financial applications with limited labeled data:

- **Document Classification**: Financial reports and statements
- **Chart Analysis**: Technical pattern recognition with limited examples
- **Fraud Detection**: Visual document verification
- **OCR Enhancement**: Form and receipt processing

## References

1. Touvron, H., et al. "Training data-efficient image transformers & distillation through attention." ICML 2021.
2. Hinton, G., et al. "Distilling the knowledge in a neural network." NeurIPS Workshop 2015.
3. Zhang, H., et al. "mixup: Beyond empirical risk minimization." ICLR 2018.
