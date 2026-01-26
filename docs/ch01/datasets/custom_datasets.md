# Custom Datasets

## Learning Objectives

By the end of this section, you will be able to:

- Design custom datasets for various data modalities (images, text, tabular, time series)
- Implement efficient lazy-loading patterns for large datasets
- Create memory-mapped datasets for datasets larger than RAM
- Handle multi-modal data in a single dataset
- Apply best practices for dataset design and debugging

---

## Overview

While PyTorch's built-in utilities handle simple cases, real-world projects often require custom datasets tailored to specific data formats, loading patterns, and preprocessing requirements. This section covers advanced patterns for building production-ready custom datasets.

---

## Design Principles

### 1. Separation of Concerns

```python
# ❌ BAD: Everything in one class
class BadDataset(Dataset):
    def __init__(self, path):
        self.data = self._load_and_preprocess_and_augment(path)

# ✅ GOOD: Separate loading, preprocessing, augmentation
class GoodDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data  # Pre-loaded or paths
        self.transform = transform
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample
```

### 2. Lazy vs Eager Loading

| Strategy | Pros | Cons | Best For |
|----------|------|------|----------|
| **Eager** | Fast access, simple | High memory | Small datasets |
| **Lazy** | Low memory | I/O overhead | Large datasets |
| **Hybrid** | Balanced | More complex | Medium datasets |

### 3. Reproducibility

Always use explicit random seeds for:
- Data augmentation
- Shuffle buffers
- Train/val/test splits

---

## Image Dataset Patterns

### Pattern 1: File Path Based

```python
import os
from PIL import Image
from typing import List, Tuple, Optional, Callable
import torch
from torch.utils.data import Dataset

class ImageFolderDataset(Dataset):
    """
    Load images from folder structure:
    
    root/
      class_0/
        img_001.jpg
        img_002.jpg
      class_1/
        img_003.jpg
        ...
    """
    
    def __init__(
        self, 
        root: str, 
        transform: Optional[Callable] = None
    ):
        self.root = root
        self.transform = transform
        
        # Build index of (path, label) pairs
        self.samples: List[Tuple[str, int]] = []
        self.class_to_idx: dict = {}
        
        self._scan_directory()
    
    def _scan_directory(self):
        """Scan root for class folders and images."""
        classes = sorted(os.listdir(self.root))
        
        for idx, class_name in enumerate(classes):
            class_dir = os.path.join(self.root, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            self.class_to_idx[class_name] = idx
            
            for fname in os.listdir(class_dir):
                if self._is_image(fname):
                    path = os.path.join(class_dir, fname)
                    self.samples.append((path, idx))
    
    def _is_image(self, fname: str) -> bool:
        """Check if file is a supported image format."""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        return os.path.splitext(fname)[1].lower() in extensions
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        
        # Lazy load image
        image = Image.open(path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
```

### Pattern 2: Annotation File Based

```python
import json

class AnnotatedImageDataset(Dataset):
    """
    Load images with annotations from JSON file.
    
    annotations.json format:
    {
        "images": [
            {"id": 1, "file_name": "img_001.jpg", "width": 224, "height": 224},
            ...
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [x, y, w, h]},
            ...
        ]
    }
    """
    
    def __init__(
        self, 
        image_dir: str, 
        annotation_file: str,
        transform: Optional[Callable] = None
    ):
        self.image_dir = image_dir
        self.transform = transform
        
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        self.images = {img['id']: img for img in annotations['images']}
        self.annotations = annotations['annotations']
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int):
        ann = self.annotations[idx]
        img_info = self.images[ann['image_id']]
        
        # Load image
        path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(path).convert('RGB')
        
        # Get target
        target = {
            'boxes': torch.tensor([ann['bbox']], dtype=torch.float32),
            'labels': torch.tensor([ann['category_id']], dtype=torch.long)
        }
        
        if self.transform:
            image, target = self.transform(image, target)
        
        return image, target
```

---

## Text Dataset Patterns

### Pattern 1: In-Memory Text

```python
from typing import List, Tuple

class TextClassificationDataset(Dataset):
    """
    In-memory text classification dataset.
    
    Suitable for small-medium text corpora that fit in memory.
    """
    
    def __init__(
        self, 
        texts: List[str], 
        labels: List[int],
        tokenizer: Callable,
        max_length: int = 512
    ):
        assert len(texts) == len(labels)
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize on-the-fly
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return (
            encoding['input_ids'].squeeze(0),
            encoding['attention_mask'].squeeze(0),
            label
        )
```

### Pattern 2: Lazy Text Loading

```python
class LazyTextDataset(Dataset):
    """
    Lazy-loading text dataset for large corpora.
    
    Stores line offsets for efficient random access.
    """
    
    def __init__(self, filepath: str, tokenizer: Callable, max_length: int = 512):
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Build line offset index
        self.offsets: List[int] = []
        self._build_index()
    
    def _build_index(self):
        """Scan file and record byte offsets of each line."""
        with open(self.filepath, 'rb') as f:
            offset = 0
            for line in f:
                self.offsets.append(offset)
                offset += len(line)
    
    def __len__(self) -> int:
        return len(self.offsets)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Seek to line and read
        with open(self.filepath, 'rb') as f:
            f.seek(self.offsets[idx])
            line = f.readline().decode('utf-8').strip()
        
        # Parse line (assuming format: label\ttext)
        parts = line.split('\t')
        label = int(parts[0])
        text = parts[1]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return (
            encoding['input_ids'].squeeze(0),
            encoding['attention_mask'].squeeze(0),
            label
        )
```

---

## Time Series Dataset Patterns

### Pattern 1: Sliding Window

```python
class TimeSeriesDataset(Dataset):
    """
    Create sliding window samples from time series data.
    
    For a series of length T with window W and horizon H:
    - Input: X[t:t+W]
    - Target: X[t+W:t+W+H]
    """
    
    def __init__(
        self, 
        data: torch.Tensor,  # Shape: [T, features]
        window_size: int = 60,
        horizon: int = 1,
        stride: int = 1
    ):
        self.data = data
        self.window_size = window_size
        self.horizon = horizon
        self.stride = stride
        
        # Calculate number of valid samples
        total_needed = window_size + horizon
        self.n_samples = max(0, (len(data) - total_needed) // stride + 1)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        
        # Input window
        x = self.data[start:start + self.window_size]
        
        # Target horizon
        y_start = start + self.window_size
        y = self.data[y_start:y_start + self.horizon]
        
        return x, y
```

### Pattern 2: Multiple Time Series

```python
class MultiTimeSeriesDataset(Dataset):
    """
    Dataset for multiple independent time series.
    
    Useful for panel data or multi-asset financial data.
    """
    
    def __init__(
        self, 
        series_dict: dict,  # {series_id: tensor of shape [T, features]}
        window_size: int = 60,
        horizon: int = 1
    ):
        self.series_dict = series_dict
        self.window_size = window_size
        self.horizon = horizon
        
        # Build index: (series_id, start_idx)
        self.index: List[Tuple[str, int]] = []
        self._build_index()
    
    def _build_index(self):
        """Create mapping from global index to (series, position)."""
        for series_id, data in self.series_dict.items():
            total_needed = self.window_size + self.horizon
            n_windows = len(data) - total_needed + 1
            
            for i in range(n_windows):
                self.index.append((series_id, i))
    
    def __len__(self) -> int:
        return len(self.index)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        series_id, start = self.index[idx]
        data = self.series_dict[series_id]
        
        x = data[start:start + self.window_size]
        y = data[start + self.window_size:start + self.window_size + self.horizon]
        
        return x, y, series_id
```

---

## Tabular Dataset Patterns

### Pattern: Mixed Types with Preprocessing

```python
import pandas as pd
import numpy as np

class TabularDataset(Dataset):
    """
    Dataset for tabular data with mixed types.
    
    Handles:
    - Numerical features (standardized)
    - Categorical features (encoded)
    - Missing values
    """
    
    def __init__(
        self, 
        df: pd.DataFrame,
        target_col: str,
        num_cols: List[str],
        cat_cols: List[str],
        num_stats: Optional[dict] = None,  # For validation/test
        cat_mappings: Optional[dict] = None
    ):
        self.df = df.copy()
        self.target_col = target_col
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        
        # Compute or use provided statistics
        if num_stats is None:
            self.num_stats = self._compute_num_stats()
        else:
            self.num_stats = num_stats
        
        if cat_mappings is None:
            self.cat_mappings = self._compute_cat_mappings()
        else:
            self.cat_mappings = cat_mappings
        
        # Preprocess
        self._preprocess()
    
    def _compute_num_stats(self) -> dict:
        """Compute mean/std for numerical columns."""
        stats = {}
        for col in self.num_cols:
            stats[col] = {
                'mean': self.df[col].mean(),
                'std': self.df[col].std()
            }
        return stats
    
    def _compute_cat_mappings(self) -> dict:
        """Create category -> index mappings."""
        mappings = {}
        for col in self.cat_cols:
            unique_vals = self.df[col].dropna().unique()
            mappings[col] = {v: i for i, v in enumerate(sorted(unique_vals))}
        return mappings
    
    def _preprocess(self):
        """Apply preprocessing to dataframe."""
        # Standardize numerical
        for col in self.num_cols:
            mean = self.num_stats[col]['mean']
            std = self.num_stats[col]['std']
            self.df[col] = (self.df[col].fillna(mean) - mean) / (std + 1e-8)
        
        # Encode categorical
        for col in self.cat_cols:
            mapping = self.cat_mappings[col]
            self.df[col] = self.df[col].map(mapping).fillna(-1).astype(int)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        
        num_features = torch.tensor(
            [row[col] for col in self.num_cols], 
            dtype=torch.float32
        )
        cat_features = torch.tensor(
            [row[col] for col in self.cat_cols], 
            dtype=torch.long
        )
        target = torch.tensor(row[self.target_col], dtype=torch.float32)
        
        return num_features, cat_features, target
    
    def get_stats(self) -> Tuple[dict, dict]:
        """Return statistics for use with validation/test sets."""
        return self.num_stats, self.cat_mappings
```

---

## Multi-Modal Dataset Patterns

### Pattern: Image + Text

```python
class ImageTextDataset(Dataset):
    """
    Multi-modal dataset combining images and text.
    
    Useful for vision-language tasks like VQA, captioning.
    """
    
    def __init__(
        self, 
        image_paths: List[str],
        texts: List[str],
        labels: List[int],
        image_transform: Optional[Callable] = None,
        tokenizer: Optional[Callable] = None,
        max_text_length: int = 128
    ):
        assert len(image_paths) == len(texts) == len(labels)
        
        self.image_paths = image_paths
        self.texts = texts
        self.labels = labels
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> dict:
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.image_transform:
            image = self.image_transform(image)
        
        # Process text
        text = self.texts[idx]
        if self.tokenizer:
            text_encoding = self.tokenizer(
                text,
                max_length=self.max_text_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            text_data = {
                'input_ids': text_encoding['input_ids'].squeeze(0),
                'attention_mask': text_encoding['attention_mask'].squeeze(0)
            }
        else:
            text_data = {'raw_text': text}
        
        return {
            'image': image,
            **text_data,
            'label': self.labels[idx]
        }
```

---

## Memory-Mapped Large Datasets

### Pattern: NumPy Memmap

```python
class LargeMemmapDataset(Dataset):
    """
    Dataset using memory-mapped files for very large data.
    
    Suitable for datasets that don't fit in RAM.
    """
    
    def __init__(
        self, 
        features_path: str,
        labels_path: str,
        features_shape: Tuple[int, ...],
        labels_shape: Tuple[int, ...],
        dtype: str = 'float32'
    ):
        self.features = np.memmap(
            features_path, 
            mode='r', 
            dtype=dtype, 
            shape=features_shape
        )
        self.labels = np.memmap(
            labels_path, 
            mode='r', 
            dtype='int64', 
            shape=labels_shape
        )
    
    def __len__(self) -> int:
        return self.features.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # np.asarray creates a view (no copy for contiguous access)
        x = torch.from_numpy(np.asarray(self.features[idx])).float()
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
    
    @staticmethod
    def create_memmap_files(
        X: np.ndarray, 
        y: np.ndarray, 
        features_path: str,
        labels_path: str
    ):
        """Helper to create memmap files from arrays."""
        features_mm = np.memmap(
            features_path, 
            mode='w+', 
            dtype=X.dtype, 
            shape=X.shape
        )
        features_mm[:] = X
        features_mm.flush()
        
        labels_mm = np.memmap(
            labels_path, 
            mode='w+', 
            dtype=y.dtype, 
            shape=y.shape
        )
        labels_mm[:] = y
        labels_mm.flush()
        
        return X.shape, y.shape
```

---

## Debugging Tips

### 1. Validate Dataset

```python
def validate_dataset(ds: Dataset, n_samples: int = 5):
    """Quick validation of dataset integrity."""
    print(f"Dataset length: {len(ds)}")
    
    for i in range(min(n_samples, len(ds))):
        try:
            sample = ds[i]
            if isinstance(sample, tuple):
                shapes = [s.shape if hasattr(s, 'shape') else type(s) for s in sample]
                print(f"  [{i}] shapes: {shapes}")
            else:
                print(f"  [{i}] type: {type(sample)}")
        except Exception as e:
            print(f"  [{i}] ERROR: {e}")
```

### 2. Check for Slow `__getitem__`

```python
import time

def benchmark_dataset(ds: Dataset, n_iterations: int = 100):
    """Measure __getitem__ performance."""
    indices = torch.randint(0, len(ds), (n_iterations,)).tolist()
    
    start = time.perf_counter()
    for idx in indices:
        _ = ds[idx]
    elapsed = time.perf_counter() - start
    
    print(f"Average __getitem__ time: {elapsed/n_iterations*1000:.2f} ms")
    print(f"Samples per second: {n_iterations/elapsed:.1f}")
```

### 3. Memory Profiling

```python
import sys

def estimate_memory(ds: Dataset):
    """Estimate memory usage of dataset."""
    size = sys.getsizeof(ds)
    
    # Check common attributes
    for attr in ['data', 'X', 'y', 'samples', 'paths']:
        if hasattr(ds, attr):
            obj = getattr(ds, attr)
            if isinstance(obj, torch.Tensor):
                size += obj.element_size() * obj.nelement()
            elif isinstance(obj, list):
                size += sys.getsizeof(obj) + sum(sys.getsizeof(x) for x in obj)
    
    print(f"Estimated memory: {size / 1024 / 1024:.2f} MB")
```

---

## Summary

| Pattern | Use Case | Memory | Speed |
|---------|----------|--------|-------|
| **In-Memory** | Small data | High | Fast |
| **Lazy Loading** | Large files | Low | Medium |
| **Memmap** | Huge arrays | Very Low | Medium |
| **Hybrid** | Medium data | Medium | Fast |

**Key Design Principles:**
1. Separate data loading from transforms
2. Use lazy loading for large datasets
3. Always include reproducibility (seeds)
4. Validate datasets before training
5. Profile performance bottlenecks

---

## Further Reading

- Section 1.13: DataLoaders (multiprocessing, collation)
- Section 2.8: Data Augmentation techniques
- [PyTorch Data Loading Best Practices](https://pytorch.org/docs/stable/data.html)
