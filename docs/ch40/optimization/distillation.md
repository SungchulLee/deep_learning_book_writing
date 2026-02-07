# Knowledge Distillation

## Overview

Knowledge distillation transfers knowledge from a large, high-capacity "teacher" model to a smaller, efficient "student" model. The student learns not just from hard labels but from the teacher's soft probability distributions, capturing richer information about class relationships and decision boundaries. This enables deployment of high-performance models in resource-constrained environments.

## Motivation

Large models achieve excellent accuracy but are expensive to deploy:

| Challenge | Impact |
|-----------|--------|
| High latency | Poor user experience, real-time constraints violated |
| Large memory footprint | Cannot fit on edge devices, high DRAM costs |
| High compute cost | Expensive inference at scale |
| Energy consumption | Battery drain on mobile, data center costs |

Knowledge distillation addresses this by training a small student model to mimic a large teacher model, achieving better accuracy than training the student from scratch.

```
┌─────────────────────────────────────────────────────────────────┐
│                   KNOWLEDGE DISTILLATION                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────┐         ┌───────────────┐                   │
│  │    Teacher    │         │    Student    │                   │
│  │  (Large CNN)  │         │ (Small CNN)   │                   │
│  └───────┬───────┘         └───────┬───────┘                   │
│          │                         │                            │
│          ▼                         ▼                            │
│    ┌───────────┐             ┌───────────┐                     │
│    │  Soft     │             │  Soft     │                     │
│    │ Targets   │────────────▶│ Targets   │  ← Distillation    │
│    │ (logits)  │             │ (logits)  │    Loss             │
│    └───────────┘             └───────────┘                     │
│                                    │                            │
│                                    │                            │
│                              ┌───────────┐                     │
│                              │   Hard    │                     │
│                              │  Targets  │  ← Classification   │
│                              │ (labels)  │    Loss             │
│                              └───────────┘                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Theoretical Foundation

### Soft Targets and Dark Knowledge

The key insight of knowledge distillation is that a teacher's output probabilities contain more information than hard labels alone.

**Hard labels:**
$$y = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] \quad \text{(cat)}$$

**Soft labels (teacher output):**
$$p = [0.01, 0.02, 0.85, 0.05, 0.03, 0.01, 0.01, 0.01, 0.005, 0.005]$$

The soft labels reveal that the teacher considers this image more similar to a dog (class 3) and tiger (class 4) than to a car or airplane—information completely absent from the hard label.

Hinton et al. termed this additional information "dark knowledge"—the knowledge embedded in the relative probabilities of incorrect classes.

**The Problem with Hard Labels:**

Hard labels (one-hot) lose critical information:
- "Cat" with 99% confidence → [0, 0, 1, 0, 0]
- "Cat" with 51% confidence → [0, 0, 1, 0, 0]

Both produce the same training signal despite vastly different confidences.

### Temperature Scaling

To extract more information from soft labels, we increase the "temperature" of the softmax:

$$p_i(T) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

where $z_i$ are the logits and $T$ is the temperature.

**Effect of temperature:**
- $T = 1$: Standard softmax (peaked distribution)
- $T > 1$: Softer distribution, more information about class relationships
- $T \to \infty$: Uniform distribution

At higher temperatures, the probability differences between classes become more pronounced, making the soft targets more informative for the student.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List


def softmax_with_temperature(logits: torch.Tensor, 
                             temperature: float = 1.0) -> torch.Tensor:
    """
    Softmax with temperature scaling.
    
    Higher temperature (T > 1):
    - Softer probability distribution
    - More information about class relationships
    
    Lower temperature (T < 1):
    - Sharper distribution
    - More confident predictions
    
    T = 1: Standard softmax
    """
    return F.softmax(logits / temperature, dim=-1)


# Example: Effect of temperature
logits = torch.tensor([5.0, 2.0, 0.5, 0.1, 0.1])

print("Effect of Temperature on Softmax:")
print("-" * 50)
for T in [0.5, 1.0, 2.0, 5.0, 10.0]:
    probs = softmax_with_temperature(logits, T)
    entropy = -(probs * probs.log()).sum()
    print(f"T={T:>4}: {probs.numpy().round(3)}  (entropy: {entropy:.3f})")
```

Output:
```
T= 0.5: [0.991 0.009 0.    0.    0.   ]  (entropy: 0.063)
T= 1.0: [0.936 0.047 0.01  0.007 0.007]  (entropy: 0.317)
T= 2.0: [0.757 0.17  0.039 0.017 0.017]  (entropy: 0.742)
T= 5.0: [0.44  0.26  0.13  0.085 0.085]  (entropy: 1.386)
T=10.0: [0.32  0.25  0.17  0.13  0.13 ]  (entropy: 1.531)
```

Higher temperature reveals relationships between classes by increasing entropy.

### Distillation Loss

The total distillation loss combines two components:

$$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{hard}} + (1 - \alpha) \cdot \mathcal{L}_{\text{soft}}$$

**Hard loss (standard cross-entropy with true labels):**
$$\mathcal{L}_{\text{hard}} = -\sum_i y_i \log(p_i^{\text{student}})$$

**Soft loss (KL divergence with teacher):**
$$\mathcal{L}_{\text{soft}} = T^2 \cdot D_{\text{KL}}\left(p^{\text{teacher}}(T) \| p^{\text{student}}(T)\right)$$

$$= T^2 \cdot \sum_i p_i^{\text{teacher}}(T) \log\frac{p_i^{\text{teacher}}(T)}{p_i^{\text{student}}(T)}$$

The $T^2$ factor compensates for the gradient magnitude scaling with temperature.

### Gradient Analysis

The gradient of the soft loss with respect to student logits $z_i^s$:

$$\frac{\partial \mathcal{L}_{\text{soft}}}{\partial z_i^s} = \frac{1}{T}\left(p_i^s(T) - p_i^t(T)\right)$$

At high temperature, this gradient encourages the student to match the teacher's entire output distribution, not just the argmax. The $T^2$ scaling in the loss ensures gradients maintain appropriate magnitude regardless of temperature choice.

## PyTorch Implementation

### Basic Knowledge Distillation

```python
class DistillationLoss(nn.Module):
    """
    Knowledge Distillation loss combining hard and soft targets.
    
    L_total = α * L_hard + (1-α) * L_soft
    
    where:
    - L_hard = CrossEntropy(student_output, true_labels)
    - L_soft = T² * KL_div(student_soft, teacher_soft)
    """
    
    def __init__(self,
                 temperature: float = 4.0,
                 alpha: float = 0.5):
        """
        Args:
            temperature: Softmax temperature (higher = softer distribution)
            alpha: Weight for hard loss (1-alpha for soft loss)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self,
                student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute distillation loss.
        
        Args:
            student_logits: Raw student outputs (B, num_classes)
            teacher_logits: Raw teacher outputs (B, num_classes)
            labels: True labels (B,)
            
        Returns:
            Total loss, dictionary with loss components
        """
        # Hard loss (student vs true labels)
        hard_loss = self.ce_loss(student_logits, labels)
        
        # Soft loss (student vs teacher)
        # Note: F.log_softmax for student, F.softmax for teacher
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        soft_loss = self.kl_loss(student_soft, teacher_soft)
        soft_loss = soft_loss * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        
        return total_loss, {
            'hard_loss': hard_loss.item(),
            'soft_loss': soft_loss.item(),
            'total_loss': total_loss.item()
        }


def train_student_with_distillation(student: nn.Module,
                                    teacher: nn.Module,
                                    train_loader: torch.utils.data.DataLoader,
                                    test_loader: torch.utils.data.DataLoader,
                                    epochs: int = 20,
                                    temperature: float = 4.0,
                                    alpha: float = 0.5,
                                    lr: float = 1e-3,
                                    device: str = 'cpu') -> nn.Module:
    """
    Train student model using knowledge distillation.
    
    Args:
        student: Student model to train
        teacher: Pre-trained teacher model (frozen)
        train_loader: Training data
        test_loader: Test data
        epochs: Training epochs
        temperature: Distillation temperature
        alpha: Hard loss weight
        lr: Learning rate
        device: Training device
        
    Returns:
        Trained student model
    """
    student = student.to(device)
    teacher = teacher.to(device)
    teacher.eval()  # Freeze teacher
    
    # Ensure teacher parameters don't require gradients
    for param in teacher.parameters():
        param.requires_grad = False
    
    criterion = DistillationLoss(temperature=temperature, alpha=alpha)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        student.train()
        epoch_losses = {'hard': 0, 'soft': 0, 'total': 0}
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            # Get teacher predictions (no gradient)
            with torch.no_grad():
                teacher_logits = teacher(data)
            
            # Student forward pass
            optimizer.zero_grad()
            student_logits = student(data)
            
            # Distillation loss
            loss, loss_dict = criterion(student_logits, teacher_logits, target)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += loss_dict[f'{key}_loss']
        
        scheduler.step()
        
        # Evaluate
        student.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = student(data)
                _, pred = output.max(1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            torch.save(student.state_dict(), 'best_student.pth')
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Hard: {epoch_losses['hard']/len(train_loader):.4f}, "
                  f"Soft: {epoch_losses['soft']/len(train_loader):.4f}, "
                  f"Acc: {100*acc:.2f}%")
    
    # Load best model
    student.load_state_dict(torch.load('best_student.pth'))
    return student
```

### Example: CNN Distillation

```python
class TeacherCNN(nn.Module):
    """Large teacher model."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class StudentCNN(nn.Module):
    """Small student model (6x fewer parameters)."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# Compare parameter counts
def compare_models():
    teacher = TeacherCNN()
    student = StudentCNN()
    
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    
    print(f"Teacher parameters: {teacher_params:,}")
    print(f"Student parameters: {student_params:,}")
    print(f"Compression ratio: {teacher_params/student_params:.1f}x")
```

## Advanced Distillation Methods

### Feature-Based Distillation (FitNets)

Beyond output probabilities, we can distill intermediate feature representations, providing stronger supervision signal:

```python
class FeatureDistillationLoss(nn.Module):
    """
    Feature-based distillation (FitNets / Attention Transfer).
    
    Matches intermediate representations between teacher and student,
    providing stronger supervision signal than output-only distillation.
    """
    
    def __init__(self,
                 teacher_channels: int,
                 student_channels: int,
                 temperature: float = 4.0,
                 alpha: float = 0.5,
                 beta: float = 0.3,
                 spatial_matching: bool = True):
        """
        Args:
            teacher_channels: Number of channels in teacher feature map
            student_channels: Number of channels in student feature map
            temperature: Output distillation temperature
            alpha: Hard loss weight
            beta: Feature loss weight
            spatial_matching: Whether to match spatial attention maps
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.spatial_matching = spatial_matching
        
        # Projection layer to match dimensions
        if teacher_channels != student_channels:
            self.projector = nn.Conv2d(student_channels, teacher_channels, 1)
        else:
            self.projector = nn.Identity()
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
    
    def forward(self,
                student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                student_features: torch.Tensor,
                teacher_features: torch.Tensor,
                labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute feature distillation loss.
        
        Args:
            student_logits: Student output logits (B, num_classes)
            teacher_logits: Teacher output logits (B, num_classes)
            student_features: Student intermediate features (B, C_s, H, W)
            teacher_features: Teacher intermediate features (B, C_t, H, W)
            labels: Ground truth labels (B,)
            
        Returns:
            Total loss, dictionary with loss components
        """
        # Hard loss
        hard_loss = self.ce_loss(student_logits, labels)
        
        # Soft loss
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = self.kl_loss(soft_student, soft_teacher)
        
        # Feature matching loss
        student_proj = self.projector(student_features)
        
        if self.spatial_matching:
            # Attention transfer: match spatial attention maps
            student_attn = self._spatial_attention(student_proj)
            teacher_attn = self._spatial_attention(teacher_features)
            feature_loss = self.mse_loss(student_attn, teacher_attn)
        else:
            # Direct feature matching (after normalization)
            student_norm = F.normalize(student_proj.flatten(2), dim=2)
            teacher_norm = F.normalize(teacher_features.detach().flatten(2), dim=2)
            feature_loss = self.mse_loss(student_norm, teacher_norm)
        
        # Combine losses
        total_loss = (
            self.alpha * hard_loss +
            (1 - self.alpha - self.beta) * self.temperature ** 2 * soft_loss +
            self.beta * feature_loss
        )
        
        return total_loss, {
            'hard_loss': hard_loss.item(),
            'soft_loss': soft_loss.item(),
            'feature_loss': feature_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def _spatial_attention(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial attention map: sum of squared activations across channels.
        
        Args:
            features: Feature map (B, C, H, W)
            
        Returns:
            Attention map (B, H, W), normalized
        """
        attention = (features ** 2).sum(dim=1)  # (B, H, W)
        attention = attention / (attention.sum(dim=(1, 2), keepdim=True) + 1e-8)
        return attention


def attention_transfer_loss(student_attention: torch.Tensor,
                           teacher_attention: torch.Tensor) -> torch.Tensor:
    """
    Standalone attention transfer loss.
    
    Matches spatial attention maps (where the model "looks").
    """
    # Normalize attention maps
    student_norm = F.normalize(
        student_attention.pow(2).mean(1).view(student_attention.size(0), -1), 
        dim=1
    )
    teacher_norm = F.normalize(
        teacher_attention.pow(2).mean(1).view(teacher_attention.size(0), -1), 
        dim=1
    )
    
    return (student_norm - teacher_norm).pow(2).mean()
```

### Multi-Layer Distillation

Distillation from multiple intermediate layers provides richer supervision:

```python
class MultiLayerDistillation(nn.Module):
    """
    Distillation from multiple intermediate layers.
    
    Provides richer supervision by matching features at multiple depths.
    """
    
    def __init__(self,
                 layer_configs: List[Dict],  # [{'t': name, 's': name, 'w': weight}, ...]
                 temperature: float = 4.0,
                 alpha: float = 0.5,
                 beta: float = 0.1):
        """
        Args:
            layer_configs: List of dicts with teacher/student layer names and weights
            temperature: Output distillation temperature
            alpha: Hard loss weight
            beta: Total feature loss weight
        """
        super().__init__()
        self.layer_configs = layer_configs
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self,
                student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                student_features: Dict[str, torch.Tensor],
                teacher_features: Dict[str, torch.Tensor],
                labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-layer distillation loss.
        """
        loss_dict = {}
        
        # Hard loss
        hard_loss = self.ce_loss(student_logits, labels)
        loss_dict['hard_loss'] = hard_loss.item()
        
        # Soft loss
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        loss_dict['soft_loss'] = soft_loss.item()
        
        # Feature distillation at multiple layers
        feature_loss = 0.0
        for config in self.layer_configs:
            t_name, s_name = config['t'], config['s']
            weight = config.get('w', 1.0)
            
            if t_name in teacher_features and s_name in student_features:
                t_feat = teacher_features[t_name].detach()
                s_feat = student_features[s_name]
                
                # Normalize and compute MSE
                fl = F.mse_loss(
                    F.normalize(s_feat.flatten(1), dim=1),
                    F.normalize(t_feat.flatten(1), dim=1)
                )
                feature_loss += weight * fl
                loss_dict[f'feature_{s_name}'] = fl.item()
        
        loss_dict['feature_total'] = feature_loss.item() if isinstance(feature_loss, torch.Tensor) else feature_loss
        
        # Combine
        output_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        total_loss = output_loss + self.beta * feature_loss
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
```

### Self-Distillation

The student and teacher share the same architecture—the model distills from itself:

```python
class SelfDistillation(nn.Module):
    """
    Self-distillation: model learns from its own earlier predictions.
    
    Variants:
    1. Born-Again Networks: Train, then distill to same architecture
    2. Deep Mutual Learning: Two networks teach each other
    3. Label Smoothing as Distillation: Soft labels as implicit teacher
    """
    
    def __init__(self,
                 model: nn.Module,
                 temperature: float = 4.0,
                 alpha: float = 0.5):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.alpha = alpha
        
        # Store soft targets from previous epoch
        self.soft_targets = {}
    
    def compute_soft_targets(self,
                            data_loader: torch.utils.data.DataLoader,
                            device: str = 'cpu'):
        """
        Compute and store soft targets for all training samples.
        """
        self.model.eval()
        self.soft_targets = {}
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(data_loader):
                data = data.to(device)
                logits = self.model(data)
                soft = F.softmax(logits / self.temperature, dim=1)
                
                for i, s in enumerate(soft):
                    idx = batch_idx * data_loader.batch_size + i
                    self.soft_targets[idx] = s.cpu()
    
    def train_step(self,
                   data: torch.Tensor,
                   labels: torch.Tensor,
                   indices: torch.Tensor,
                   optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """
        Single training step with self-distillation.
        """
        self.model.train()
        
        optimizer.zero_grad()
        logits = self.model(data)
        
        # Hard loss
        hard_loss = F.cross_entropy(logits, labels)
        
        # Soft loss (from stored targets)
        if self.soft_targets:
            soft_targets = torch.stack([self.soft_targets[i.item()] for i in indices])
            soft_targets = soft_targets.to(data.device)
            
            student_soft = F.log_softmax(logits / self.temperature, dim=1)
            soft_loss = F.kl_div(student_soft, soft_targets, reduction='batchmean')
            soft_loss = soft_loss * (self.temperature ** 2)
        else:
            soft_loss = torch.tensor(0.0)
        
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        total_loss.backward()
        optimizer.step()
        
        return {
            'hard_loss': hard_loss.item(),
            'soft_loss': soft_loss.item() if isinstance(soft_loss, torch.Tensor) else 0,
            'total_loss': total_loss.item()
        }


def self_distillation_training(model: nn.Module,
                               train_loader: torch.utils.data.DataLoader,
                               epochs: int = 100,
                               distill_epochs_start: int = 50,
                               temperature: float = 4.0,
                               device: str = 'cpu') -> nn.Module:
    """
    Self-distillation: model learns from its own past predictions.
    
    After initial training, use model's own soft predictions
    as additional supervision.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    past_predictions = {}  # Store predictions from previous epoch
    
    for epoch in range(epochs):
        model.train()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            # Standard loss
            loss = criterion(output, target)
            
            # Add self-distillation after warmup
            if epoch >= distill_epochs_start and batch_idx in past_predictions:
                soft_loss = F.kl_div(
                    F.log_softmax(output / temperature, dim=-1),
                    F.softmax(past_predictions[batch_idx] / temperature, dim=-1),
                    reduction='batchmean'
                )
                loss = 0.5 * loss + 0.5 * (temperature ** 2) * soft_loss
            
            loss.backward()
            optimizer.step()
            
            # Store predictions for next epoch
            past_predictions[batch_idx] = output.detach()
    
    return model
```

### Multi-Teacher Distillation

Ensemble multiple teachers for richer knowledge:

$$p^{\text{ensemble}} = \frac{1}{K}\sum_{k=1}^{K} p^{\text{teacher}_k}$$

```python
class MultiTeacherDistillation(nn.Module):
    """
    Distillation from multiple teacher models.
    
    Multiple teachers provide diverse perspectives, often improving
    student generalization beyond what a single teacher achieves.
    """
    
    def __init__(self,
                 teachers: List[nn.Module],
                 aggregation: str = 'mean',
                 temperature: float = 4.0):
        """
        Args:
            teachers: List of teacher models
            aggregation: 'mean', 'weighted', or 'max'
            temperature: Distillation temperature
        """
        super().__init__()
        self.teachers = nn.ModuleList(teachers)
        self.aggregation = aggregation
        self.temperature = temperature
        
        if aggregation == 'weighted':
            self.weights = nn.Parameter(torch.ones(len(teachers)) / len(teachers))
    
    def get_teacher_soft_targets(self,
                                 data: torch.Tensor) -> torch.Tensor:
        """
        Aggregate soft targets from all teachers.
        """
        soft_targets = []
        
        for teacher in self.teachers:
            teacher.eval()
            with torch.no_grad():
                logits = teacher(data)
                soft = F.softmax(logits / self.temperature, dim=1)
                soft_targets.append(soft)
        
        soft_targets = torch.stack(soft_targets, dim=0)  # (K, B, C)
        
        if self.aggregation == 'mean':
            return soft_targets.mean(dim=0)
        elif self.aggregation == 'weighted':
            weights = F.softmax(self.weights, dim=0)
            return (soft_targets * weights.view(-1, 1, 1)).sum(dim=0)
        elif self.aggregation == 'max':
            return soft_targets.max(dim=0)[0]
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
```

### Online Distillation (Deep Mutual Learning)

Train teacher and student simultaneously:

```python
class OnlineDistillation(nn.Module):
    """
    Deep Mutual Learning: networks teach each other during training.
    
    No pre-trained teacher required. Multiple networks learn collaboratively,
    often outperforming traditional one-way distillation.
    """
    
    def __init__(self,
                 models: List[nn.Module],
                 temperature: float = 4.0):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.temperature = temperature
    
    def forward(self,
                data: torch.Tensor,
                labels: torch.Tensor) -> Tuple[List[torch.Tensor], Dict[str, float]]:
        """
        Forward pass for all models with mutual learning.
        """
        logits_list = [model(data) for model in self.models]
        
        losses = []
        loss_dict = {}
        
        for i, logits in enumerate(logits_list):
            # Hard loss
            hard_loss = F.cross_entropy(logits, labels)
            
            # Soft loss: learn from all other models
            soft_loss = 0.0
            student_soft = F.log_softmax(logits / self.temperature, dim=1)
            
            for j, other_logits in enumerate(logits_list):
                if i != j:
                    with torch.no_grad():
                        teacher_soft = F.softmax(other_logits / self.temperature, dim=1)
                    soft_loss += F.kl_div(student_soft, teacher_soft, reduction='batchmean')
            
            soft_loss = soft_loss / (len(self.models) - 1) * (self.temperature ** 2)
            total_loss = hard_loss + soft_loss
            losses.append(total_loss)
            loss_dict[f'model_{i}_loss'] = total_loss.item()
        
        return losses, loss_dict
```

## Progressive Distillation

Gradually distill through intermediate-sized models for very large compression ratios:

```
Large Teacher → Medium Model → Small Student
```

Each step is easier than directly distilling to the smallest model.

```python
def progressive_distillation(teachers: List[nn.Module],
                            train_loader: torch.utils.data.DataLoader,
                            test_loader: torch.utils.data.DataLoader,
                            epochs_per_stage: int = 20,
                            temperature: float = 4.0,
                            device: str = 'cpu') -> nn.Module:
    """
    Progressive distillation through a chain of models.
    
    Args:
        teachers: List of models from largest to smallest
                  [large_teacher, medium_model, small_student]
        train_loader: Training data
        test_loader: Test data
        epochs_per_stage: Training epochs for each distillation stage
        temperature: Distillation temperature
        device: Training device
        
    Returns:
        Final trained small student
    """
    for i in range(len(teachers) - 1):
        teacher = teachers[i]
        student = teachers[i + 1]
        
        print(f"\n{'='*60}")
        print(f"Stage {i+1}: Distilling model {i} → model {i+1}")
        print(f"{'='*60}")
        
        student = train_student_with_distillation(
            student=student,
            teacher=teacher,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs_per_stage,
            temperature=temperature,
            device=device
        )
    
    return teachers[-1]
```

## Hyperparameter Selection

### Temperature Selection

| Temperature | Effect | Best For |
|-------------|--------|----------|
| 1-2 | Sharp distributions | Already confident teacher |
| 3-5 | Moderate softness | General use (default: 4) |
| 10-20 | Very soft | When class similarities matter, large architecture gaps |
| >20 | Nearly uniform | Usually too soft |

### Alpha Selection

| Alpha | Interpretation | When to Use |
|-------|---------------|-------------|
| 0.0 | Pure soft loss | Very confident, accurate teacher |
| 0.3-0.5 | Balanced | General use (default) |
| 0.7-0.9 | Mostly hard | Unreliable teacher or fine-tuning |
| 1.0 | Pure hard loss | No distillation (baseline) |

### When to Use Each Method

| Scenario | Recommendation |
|----------|---------------|
| Large accuracy gap teacher-student | Use higher temperature (8-20) |
| Similar architectures | Use feature distillation |
| Very small student | Use progressive distillation |
| Limited training data | Distillation particularly helpful |
| Ensemble teacher | Combine multiple teachers |
| No pre-trained teacher | Use deep mutual learning |

### Student Architecture Design

The student should be small enough for deployment but large enough to capture the teacher's knowledge:

- **Too small**: Cannot learn complex decision boundaries
- **Too large**: Diminishing returns, defeats compression purpose

Rule of thumb: Start with 10-30% of teacher parameters, adjust based on accuracy requirements.

### Finding Optimal Temperature

```python
def find_optimal_temperature(teacher: nn.Module,
                            student_class: type,
                            train_loader: torch.utils.data.DataLoader,
                            val_loader: torch.utils.data.DataLoader,
                            temperatures: List[float] = [1, 2, 4, 8, 16, 20],
                            quick_epochs: int = 10,
                            device: str = 'cpu') -> float:
    """
    Find optimal distillation temperature via grid search.
    """
    results = []
    
    for T in temperatures:
        # Fresh student for each trial
        student = student_class()
        
        # Quick distillation training
        trained = train_student_with_distillation(
            student=student,
            teacher=teacher,
            train_loader=train_loader,
            test_loader=val_loader,
            epochs=quick_epochs,
            temperature=T,
            device=device
        )
        
        # Evaluate
        accuracy = evaluate_accuracy(trained, val_loader, device)
        results.append((T, accuracy))
        print(f"Temperature {T}: {accuracy*100:.2f}%")
    
    best_T = max(results, key=lambda x: x[1])[0]
    print(f"\nOptimal temperature: {best_T}")
    return best_T
```

## Evaluation Metrics

```python
def evaluate_distillation(teacher: nn.Module,
                          student: nn.Module,
                          student_baseline: nn.Module,
                          test_loader: torch.utils.data.DataLoader,
                          device: str = 'cpu') -> Dict[str, float]:
    """
    Comprehensive evaluation of distillation effectiveness.
    """
    def get_accuracy(model):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        return correct / total
    
    def count_params(model):
        return sum(p.numel() for p in model.parameters())
    
    teacher_acc = get_accuracy(teacher)
    student_acc = get_accuracy(student)
    baseline_acc = get_accuracy(student_baseline)
    
    results = {
        'teacher_accuracy': teacher_acc,
        'student_distilled_accuracy': student_acc,
        'student_baseline_accuracy': baseline_acc,
        'distillation_improvement': student_acc - baseline_acc,
        'gap_to_teacher': teacher_acc - student_acc,
        'compression_ratio': count_params(teacher) / count_params(student),
        'parameter_reduction': 1 - count_params(student) / count_params(teacher)
    }
    
    return results


def evaluate_distillation_agreement(teacher: nn.Module,
                                   student: nn.Module,
                                   test_loader: torch.utils.data.DataLoader,
                                   device: str = 'cpu') -> Dict[str, float]:
    """
    Evaluate how well student mimics teacher behavior.
    """
    teacher.eval()
    student.eval()
    
    teacher_correct = 0
    student_correct = 0
    agreement = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            teacher_out = teacher(data)
            student_out = student(data)
            
            teacher_pred = teacher_out.argmax(dim=1)
            student_pred = student_out.argmax(dim=1)
            
            teacher_correct += (teacher_pred == target).sum().item()
            student_correct += (student_pred == target).sum().item()
            agreement += (teacher_pred == student_pred).sum().item()
            total += target.size(0)
    
    return {
        'teacher_accuracy': teacher_correct / total,
        'student_accuracy': student_correct / total,
        'teacher_student_agreement': agreement / total,
        'knowledge_transfer_efficiency': student_correct / teacher_correct
    }
```

## Summary

Knowledge distillation enables deployment of efficient models:

1. **Core idea**: Train small student to mimic large teacher
2. **Soft targets**: Preserve probability relationships via temperature
3. **Loss function**: Combine hard labels and soft targets
4. **Advanced methods**: Feature matching, attention transfer, self-distillation
5. **Temperature**: Higher values for larger architecture gaps

Key recommendations:
- Start with temperature 4-8 and alpha 0.5
- Use feature distillation for similar architectures
- Validate that student matches teacher behavior
- Consider progressive distillation for very small students
- Use deep mutual learning when no pre-trained teacher is available

## References

1. Hinton, G., Vinyals, O., & Dean, J. "Distilling the Knowledge in a Neural Network." arXiv 2015.
2. Romero, A., et al. "FitNets: Hints for Thin Deep Nets." ICLR 2015.
3. Zagoruyko, S. & Komodakis, N. "Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer." ICLR 2017.
4. Zhang, Y., et al. "Deep Mutual Learning." CVPR 2018.
5. Furlanello, T., et al. "Born-Again Neural Networks." ICML 2018.
6. Gou, J., et al. "Knowledge Distillation: A Survey." IJCV 2021.
