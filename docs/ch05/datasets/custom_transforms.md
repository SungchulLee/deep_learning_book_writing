# Custom Transforms

## Overview

When built-in transforms are insufficient, custom transforms allow domain-specific preprocessing logic to integrate seamlessly into the transform pipeline. A custom transform is any callable—a function, a class with `__call__`, or a lambda.

## Callable Class Pattern

The standard approach wraps transformation logic in a class:

```python
class LogReturns:
    """Convert price series to log returns."""
    def __call__(self, prices):
        return torch.log(prices[1:] / prices[:-1])

class StandardScaler:
    """Standardize features using precomputed statistics."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / (self.std + 1e-8)
```

Classes are preferred over lambdas because they are serializable (picklable), configurable via `__init__`, and compose well within `Compose`.

## Composing Custom and Built-in Transforms

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    StandardScaler(mean=train_mean, std=train_std),
    LogReturns()
])
```

## Transforms for Dictionary Samples

When datasets return dictionaries (e.g., image + landmarks), transforms must operate on the entire sample:

```python
class RandomCropWithLandmarks:
    """Jointly crop image and adjust landmark coordinates."""
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = torch.randint(0, h - new_h + 1, (1,)).item()
        left = torch.randint(0, w - new_w + 1, (1,)).item()

        image = image[top:top + new_h, left:left + new_w]
        landmarks = landmarks - torch.tensor([left, top], dtype=torch.float32)

        return {'image': image, 'landmarks': landmarks}


class ToTensorSample:
    """Convert image and landmarks in a sample dict to tensors."""
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        # PIL images: H×W×C → C×H×W
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
        return {'image': image, 'landmarks': torch.from_numpy(landmarks).float()}
```

## Stochastic Transforms

Transforms that apply randomly during training:

```python
class RandomNoise:
    """Add Gaussian noise with given probability."""
    def __init__(self, std=0.01, p=0.5):
        self.std = std
        self.p = p

    def __call__(self, x):
        if torch.rand(1).item() < self.p:
            return x + torch.randn_like(x) * self.std
        return x
```

## Financial Data Transforms

```python
class RollingZScore:
    """Apply rolling z-score normalization to time series features."""
    def __init__(self, window=252):
        self.window = window

    def __call__(self, x):
        # x: (T, D) tensor of features
        mean = x.unfold(0, self.window, 1).mean(dim=-1)
        std = x.unfold(0, self.window, 1).std(dim=-1)
        # Align: first (window-1) values use expanding window
        normalized = (x[self.window - 1:] - mean) / (std + 1e-8)
        return normalized


class WinsorizeTails:
    """Clip extreme values at given percentiles."""
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def __call__(self, x):
        lo = torch.quantile(x, self.lower, dim=0)
        hi = torch.quantile(x, self.upper, dim=0)
        return torch.clamp(x, lo, hi)
```

## Key Takeaways

- Custom transforms are any callable that fits into `Compose`.
- Use classes (not lambdas) for serializability and configurability.
- Dictionary-based transforms enable joint processing of images and annotations.
- Financial transforms handle domain-specific preprocessing such as rolling normalization, log returns, and winsorization.
