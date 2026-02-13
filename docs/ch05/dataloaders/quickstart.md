# Quick Start Guide

Get started with PyTorch DataLoader in 5 minutes!

---

## Installation

```bash
pip install torch torchvision matplotlib numpy
```

---

## Run Your First Tutorial

```bash
cd pytorch_dataloader_tutorial
python 01_basics/01_dataloader_fundamentals.py
```

You should see output explaining DataLoader basics with examples.

---

## Basic Usage Example

```python
import torch
from torch.utils.data import Dataset, DataLoader

# 1. Create a Dataset
class MyDataset(Dataset):
    def __init__(self, size=100):
        self.data = torch.randn(size, 10)
        self.labels = torch.randint(0, 2, (size,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 2. Create DataLoader
dataset = MyDataset(size=100)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2
)

# 3. Iterate through batches
for batch_idx, (data, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx}: data shape = {data.shape}")
    # Your training code here...
```

---

## What's Next?

### For Beginners
Start with basics:
1. `01_basics/01_dataloader_fundamentals.py`
2. `01_basics/02_dataloader_parameters.py`

### For Intermediate Users
Jump to performance:
1. `02_intermediate/01_multiprocess_loading.py`
2. `02_intermediate/02_performance_profiling.py`

### For Advanced Users
Multi-GPU training:
1. `03_advanced/01_distributed_loading.py`

---

## Common First Steps

### 1. Understanding Batching
```python
# Create loader with different batch sizes
loader_small = DataLoader(dataset, batch_size=8)
loader_large = DataLoader(dataset, batch_size=128)
```

### 2. Using Multiple Workers
```python
# Speed up loading with workers
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4  # Parallel loading
)
```

### 3. GPU Training
```python
# Optimize for GPU
loader = DataLoader(
    dataset,
    batch_size=32,
    pin_memory=True,  # Faster GPU transfer
    num_workers=4
)
```

---

## Tips

✅ **Always start with num_workers=0 for debugging**  
✅ **Use pin_memory=True when training on GPU**  
✅ **Increase num_workers if GPU is underutilized**  
✅ **Profile before optimizing - measure everything**  

---

## Need Help?

- Read the detailed comments in each tutorial file
- Check the main README.md for comprehensive guide
- See troubleshooting section for common issues

---

Happy coding! 🚀
