# Custom Datasets

## Overview

Real-world data rarely comes in the form of built-in datasets. Custom `Dataset` implementations handle domain-specific data formats, loading strategies, and preprocessing requirements. This section presents patterns for building robust custom datasets.

## CSV/Tabular Dataset

```python
import pandas as pd
import torch
from torch.utils.data import Dataset

class TabularDataset(Dataset):
    def __init__(self, csv_file, feature_cols, target_col, transform=None):
        self.df = pd.read_csv(csv_file)
        self.features = self.df[feature_cols].values.astype('float32')
        self.targets = self.df[target_col].values.astype('float32')
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx])
        y = torch.tensor(self.targets[idx])
        if self.transform:
            x = self.transform(x)
        return x, y
```

## Image Dataset with Annotations

For image datasets where labels come from an annotation file (CSV, JSON, or XML):

```python
import os
from PIL import Image

class AnnotatedImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.annotations = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = self.annotations.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)
        return image, label
```

## Face Landmarks Dataset

A dataset returning both images and spatial annotations (landmarks, bounding boxes, keypoints):

```python
class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset from dlib."""
    def __init__(self, csv_file, root_dir, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = Image.open(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].values
        landmarks = landmarks.astype('float32').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)
        return sample
```

Returning dictionaries (rather than tuples) is useful when samples have multiple associated tensors of different shapes.

## Multi-Modal Dataset

Datasets that combine multiple data sources:

```python
class MultiModalDataset(Dataset):
    def __init__(self, image_dir, text_data, labels):
        self.image_dir = image_dir
        self.text_data = text_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, f'{idx}.jpg'))
        image = transforms.ToTensor()(image)
        text = self.text_data[idx]
        label = self.labels[idx]
        return {'image': image, 'text': text, 'label': label}
```

## Dataset from Tensors

When data is already in tensor form, `TensorDataset` provides a zero-boilerplate solution:

```python
from torch.utils.data import TensorDataset

X = torch.randn(1000, 10)
y = torch.randint(0, 2, (1000,))
dataset = TensorDataset(X, y)

# Equivalent to:
# dataset[i] returns (X[i], y[i])
```

## Financial Time Series Dataset

```python
class FinancialDataset(Dataset):
    """
    Rolling-window dataset for financial time series.
    Computes log returns and creates feature/target windows.
    """
    def __init__(self, prices, window_size=60, horizon=5,
                 feature_cols=None):
        self.returns = np.log(prices / prices.shift(1)).dropna()
        if feature_cols:
            self.features = self.returns[feature_cols].values
        else:
            self.features = self.returns.values
        self.window_size = window_size
        self.horizon = horizon

    def __len__(self):
        return len(self.features) - self.window_size - self.horizon + 1

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.window_size]
        y = self.features[idx + self.window_size :
                          idx + self.window_size + self.horizon, 0]
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32))
```

## Design Guidelines

**Minimize work in `__getitem__`**: Heavy computation in `__getitem__` is called for every sample access. Precompute what you can in `__init__` and let `__getitem__` do only what must be deferred (e.g., augmentation that must differ per epoch).

**Handle missing data gracefully**: Real-world datasets have missing values, corrupt files, and edge cases. Implement defensive checks:

```python
def __getitem__(self, idx):
    try:
        image = Image.open(self.paths[idx])
    except (IOError, OSError):
        # Return a default or skip to next valid sample
        return self.__getitem__((idx + 1) % len(self))
```

**Respect temporal ordering**: For time series data, never allow the dataset or sampler to access future data. Validate that `__getitem__` only uses information available up to the prediction time.

## Key Takeaways

- Custom datasets implement `__len__` and `__getitem__` to wrap any data source.
- Return tuples for simple cases, dictionaries for multi-tensor or multi-modal samples.
- `TensorDataset` provides a shortcut when data is already in tensor form.
- Financial datasets require rolling windows with strict temporal ordering to prevent look-ahead bias.
