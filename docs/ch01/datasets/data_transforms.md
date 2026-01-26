# Data Transforms

## Learning Objectives

By the end of this section, you will be able to:

- Implement custom transform classes following the callable pattern
- Compose multiple transforms into pipelines
- Apply separate transforms to inputs and targets
- Implement joint transforms for paired data (images + masks)
- Understand reproducibility considerations with random transforms

---

## Overview

**Transforms** are callables that modify data samples on-the-fly during `__getitem__`. They enable:

- **Preprocessing**: Normalization, standardization, type conversion
- **Data augmentation**: Random noise, crops, flips (for training)
- **Feature engineering**: Derived features, embeddings

### The Transform Pattern

```python
class MyTransform:
    """A transform is a callable with optional initialization."""
    
    def __init__(self, param1, param2):
        """Store configuration."""
        self.param1 = param1
        self.param2 = param2
    
    def __call__(self, x):
        """Apply transformation and return result."""
        return transformed_x
```

This pattern separates **configuration** (constructor) from **application** (`__call__`), enabling reuse and composition.

---

## Core Transform Classes

### Compose: Chaining Transforms

Apply multiple transforms sequentially:

```python
from typing import List, Callable, Any

class Compose:
    """
    Chain transforms: x → f_n(...f_2(f_1(x)))
    
    The output of each transform becomes the input to the next.
    """
    
    def __init__(self, transforms: List[Callable]):
        """
        Args:
            transforms: List of callables to apply in order
        """
        self.transforms = list(transforms)
    
    def __call__(self, x: Any) -> Any:
        """Apply all transforms sequentially."""
        for transform in self.transforms:
            x = transform(x)
        return x
    
    def __repr__(self) -> str:
        transform_names = [t.__class__.__name__ for t in self.transforms]
        return f"Compose({' → '.join(transform_names)})"
```

**Usage:**

```python
pipeline = Compose([
    ToTensor(),
    Standardize(mean=0.5, std=0.2),
    AddNoise(std=0.01)
])

x_raw = [1.0, 2.0, 3.0]
x_transformed = pipeline(x_raw)
```

---

### ToTensor: Type Conversion

```python
import torch

class ToTensor:
    """Convert input to PyTorch tensor if not already."""
    
    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            return x
        return torch.tensor(x)
```

---

### Standardize: Z-Score Normalization

Applies the transformation:

$$
x' = \frac{x - \mu}{\sigma + \epsilon}
$$

```python
class Standardize:
    """
    Standardize data using precomputed mean and std.
    
    This is z-score normalization: x → (x - μ) / (σ + ε)
    
    Note: Statistics should be computed on the TRAINING set only,
    then applied to validation and test sets.
    """
    
    def __init__(self, mean: float, std: float, eps: float = 1e-8):
        """
        Args:
            mean: Population mean (from training data)
            std: Population standard deviation (from training data)
            eps: Small constant to prevent division by zero
        """
        self.mean = float(mean)
        self.std = float(std)
        self.eps = eps
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)
```

**Computing Statistics:**

```python
# Compute on training data only
train_data = torch.randn(1000, 10)
mean = train_data.mean()
std = train_data.std(unbiased=False)  # Population std

# Create transform
standardize = Standardize(mean.item(), std.item())

# Apply to all splits (train, val, test)
x_train_normalized = standardize(train_data)
x_val_normalized = standardize(val_data)  # Same transform
```

---

### MinMaxScale: Range Normalization

Scales data to a target range:

$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}} \cdot (b - a) + a
$$

```python
class MinMaxScale:
    """
    Scale data from [min_val, max_val] to [out_min, out_max].
    
    Default scales to [0, 1].
    """
    
    def __init__(
        self, 
        min_val: float, 
        max_val: float, 
        out_min: float = 0.0, 
        out_max: float = 1.0,
        eps: float = 1e-8
    ):
        self.min_val = min_val
        self.max_val = max_val
        self.out_min = out_min
        self.out_max = out_max
        self.eps = eps
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Scale to [0, 1]
        scaled = (x - self.min_val) / (self.max_val - self.min_val + self.eps)
        # Scale to [out_min, out_max]
        return scaled * (self.out_max - self.out_min) + self.out_min
```

---

### AddGaussianNoise: Data Augmentation

Adds random Gaussian noise:

$$
x' = x + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
$$

```python
class AddGaussianNoise:
    """
    Add Gaussian noise for data augmentation.
    
    Uses a dedicated Generator for reproducibility.
    """
    
    def __init__(self, std: float = 0.05, seed: int = 0):
        """
        Args:
            std: Standard deviation of noise
            seed: Random seed for reproducibility
        """
        self.std = std
        self.generator = torch.Generator().manual_seed(seed)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.normal(
            mean=0, 
            std=self.std, 
            size=x.shape, 
            generator=self.generator
        )
        return x + noise
```

**Important**: Using a `Generator` ensures reproducible augmentation across runs.

---

### OneHot: Label Encoding

Converts class indices to one-hot vectors:

```python
class OneHot:
    """
    Convert class index to one-hot encoded vector.
    
    Example: 2 with 5 classes → [0, 0, 1, 0, 0]
    """
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
    
    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        one_hot = torch.zeros(self.num_classes, dtype=torch.float32)
        one_hot[y.long().item()] = 1.0
        return one_hot
```

---

## Dataset with Transforms

### Standard Pattern

```python
from typing import Optional, Callable, Tuple

class TransformableDataset(Dataset):
    """
    Dataset with separate input and target transforms.
    
    Transforms are applied on-the-fly in __getitem__.
    """
    
    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        self.X = X
        self.y = y
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        xi = self.X[idx]
        yi = self.y[idx]
        
        if self.transform is not None:
            xi = self.transform(xi)
        
        if self.target_transform is not None:
            yi = self.target_transform(yi)
        
        return xi, yi
```

### Complete Example

```python
# Generate synthetic regression data
torch.manual_seed(42)
n = 100
X = torch.empty(n, 1).uniform_(-2, 2)
y = 3 * X + 1 + 0.3 * torch.randn(n, 1)

# Compute statistics on raw data
x_mean, x_std = X.mean(), X.std()
y_mean, y_std = y.mean(), y.std()

# Build transform pipelines
x_transform = Compose([
    Standardize(x_mean.item(), x_std.item()),
    AddGaussianNoise(std=0.02, seed=123)
])

y_transform = Standardize(y_mean.item(), y_std.item())

# Create dataset
ds = TransformableDataset(
    X, y,
    transform=x_transform,
    target_transform=y_transform
)

# Verify transforms are applied
print(f"Raw X[0]: {X[0].item():.4f}")
print(f"Transformed X[0]: {ds[0][0].item():.4f}")
```

---

## Joint Transforms

For tasks like semantic segmentation, transforms must apply the **same random parameters** to both input and mask.

### The Problem

```python
# ❌ WRONG: Different random parameters for each
x = random_crop(image)      # Crops region A
y = random_crop(mask)       # Crops region B (different!)

# ✅ CORRECT: Same random parameters
x, y = random_crop(image, mask)  # Both crop region A
```

### Joint Transform Classes

```python
import random
from typing import Tuple

class RandomHorizontalFlip2D:
    """
    Randomly flip image and mask horizontally.
    
    Both get flipped (or not) together.
    """
    
    def __init__(self, p: float = 0.5, seed: Optional[int] = None):
        """
        Args:
            p: Probability of flipping
            seed: Random seed (optional)
        """
        self.p = p
        self.seed = seed
    
    def __call__(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Image tensor [C, H, W] or [H, W]
            y: Mask tensor [H, W]
        
        Returns:
            Flipped (x, y) or original (x, y)
        """
        if self.seed is not None:
            random.seed(self.seed)
        
        if random.random() < self.p:
            # Flip along width dimension
            if x.ndim == 3:
                x = x.flip(dims=[2])  # [C, H, W] → flip W
            else:
                x = x.flip(dims=[1])  # [H, W] → flip W
            
            y = y.flip(dims=[1])  # [H, W] → flip W
        
        return x, y


class RandomCrop2D:
    """
    Randomly crop image and mask with same parameters.
    
    Pads if input is smaller than target size.
    """
    
    def __init__(
        self, 
        size: Tuple[int, int], 
        seed: Optional[int] = None
    ):
        """
        Args:
            size: Target (height, width)
            seed: Random seed (optional)
        """
        self.target_h, self.target_w = size
        self.seed = seed
    
    def __call__(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply same random crop to both tensors."""
        if self.seed is not None:
            random.seed(self.seed)
        
        # Get dimensions
        if x.ndim == 3:
            C, H, W = x.shape
        else:
            H, W = x.shape
            C = None
        
        # Pad if needed
        if H < self.target_h or W < self.target_w:
            pad_h = max(0, self.target_h - H)
            pad_w = max(0, self.target_w - W)
            padding = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
            
            x = torch.nn.functional.pad(x, padding)
            y = torch.nn.functional.pad(y, padding)
            
            if C is not None:
                _, H, W = x.shape
            else:
                H, W = x.shape
        
        # Random crop position (same for both)
        top = random.randint(0, H - self.target_h)
        left = random.randint(0, W - self.target_w)
        
        # Apply crop
        if C is not None:
            x = x[:, top:top+self.target_h, left:left+self.target_w]
        else:
            x = x[top:top+self.target_h, left:left+self.target_w]
        
        y = y[top:top+self.target_h, left:left+self.target_w]
        
        return x, y
```

### Composing Joint Transforms

```python
class ComposeJoint:
    """Compose joint transforms that take (x, y) pairs."""
    
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
    
    def __call__(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for t in self.transforms:
            x, y = t(x, y)
        return x, y
```

### Segmentation Dataset Example

```python
class SegmentationDataset(Dataset):
    """
    Dataset for semantic segmentation with joint transforms.
    """
    
    def __init__(
        self,
        images: List[torch.Tensor],
        masks: List[torch.Tensor],
        joint_transform: Optional[Callable] = None
    ):
        assert len(images) == len(masks)
        self.images = images
        self.masks = masks
        self.joint_transform = joint_transform
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.images[idx].clone()
        y = self.masks[idx].clone()
        
        if self.joint_transform is not None:
            x, y = self.joint_transform(x, y)
        
        return x, y


# Usage
joint_transform = ComposeJoint([
    RandomHorizontalFlip2D(p=0.5),
    RandomCrop2D(size=(64, 64))
])

ds = SegmentationDataset(images, masks, joint_transform=joint_transform)

x, y = ds[0]
print(f"Image shape: {x.shape}")  # [C, 64, 64]
print(f"Mask shape: {y.shape}")   # [64, 64]
```

---

## Reproducibility with Random Transforms

### The Challenge

Random transforms introduce non-determinism, which can complicate:
- Debugging (hard to reproduce issues)
- Experiments (results vary across runs)
- Multi-worker loading (each worker gets different randomness)

### Solutions

#### 1. Per-Transform Generators

```python
class ReproducibleNoise:
    """Reproducible noise with per-transform RNG."""
    
    def __init__(self, std: float, seed: int):
        self.std = std
        self.generator = torch.Generator().manual_seed(seed)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(x.shape, generator=self.generator) * self.std
        return x + noise
```

**Behavior**: Same seed → same noise sequence across epochs.

#### 2. Index-Based Seeding

```python
class IndexSeededNoise:
    """Different but reproducible noise per sample."""
    
    def __init__(self, std: float, base_seed: int = 0):
        self.std = std
        self.base_seed = base_seed
    
    def __call__(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(self.base_seed + idx)
        noise = torch.randn(x.shape, generator=g) * self.std
        return x + noise
```

**Note**: Requires passing index to transform (modify dataset accordingly).

#### 3. Epoch-Aware Transforms

```python
class EpochAwareTransform:
    """Vary transformation across epochs."""
    
    def __init__(self, std: float, base_seed: int = 0):
        self.std = std
        self.base_seed = base_seed
        self.epoch = 0
    
    def set_epoch(self, epoch: int):
        self.epoch = epoch
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.Generator().manual_seed(self.base_seed + self.epoch * 1000)
        noise = torch.randn(x.shape, generator=g) * self.std
        return x + noise
```

---

## Integration with torchvision

For image tasks, `torchvision.transforms` provides production-ready transforms:

```python
from torchvision import transforms

# Image transform pipeline
image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet stats
        std=[0.229, 0.224, 0.225]
    )
])

# Use in dataset
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = load_image(self.image_paths[idx])
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
```

---

## Best Practices

### 1. Compute Statistics on Training Data Only

```python
# ✅ Correct: Use training statistics for all splits
train_mean = train_X.mean()
train_std = train_X.std()

train_ds = MyDataset(train_X, transform=Standardize(train_mean, train_std))
val_ds = MyDataset(val_X, transform=Standardize(train_mean, train_std))
test_ds = MyDataset(test_X, transform=Standardize(train_mean, train_std))
```

### 2. Different Transforms for Train vs Val/Test

```python
# Training: with augmentation
train_transform = Compose([
    Standardize(mean, std),
    AddNoise(std=0.05),
    RandomDropout(p=0.1)
])

# Validation/Test: no augmentation
eval_transform = Compose([
    Standardize(mean, std)
])
```

### 3. Document Transform Assumptions

```python
class Standardize:
    """
    Z-score normalization.
    
    Assumptions:
        - Input is a torch.Tensor
        - Statistics computed on training data
        - Applied consistently across all data splits
    """
```

---

## Summary

| Concept | Key Point |
|---------|-----------|
| **Transform Pattern** | Callable class with `__init__` and `__call__` |
| **Compose** | Chain transforms: x → f_n(...f_1(x)) |
| **Joint Transforms** | Same random params for paired data |
| **Reproducibility** | Use `torch.Generator` with explicit seeds |
| **Train vs Eval** | Augmentation for training only |

---

## Further Reading

- [torchvision.transforms Documentation](https://pytorch.org/vision/stable/transforms.html)
- Section 2.8: Data Augmentation (regularization chapter)
- Section 4.1: Image Preprocessing (computer vision chapter)
