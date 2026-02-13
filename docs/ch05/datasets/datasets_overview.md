# PyTorch Dataset & DataLoader Examples

A comprehensive collection of 16 PyTorch examples demonstrating Dataset and DataLoader concepts from basic to advanced.

## 📦 Package Contents

### Map-Style Datasets (01-10)
Map-style datasets support random access via indexing: `dataset[i]`

- **01_map_style_minimal.py** - Simplest map-style dataset implementation
- **02_map_style_with_init.py** - Dataset with initialization logic
- **03_map_style_with_transforms.py** - Adding data transformations
- **04_transforms_core.py** - Core transform operations
- **05_transforms_joint_transforms_spatial.py** - Spatial transforms for images
- **06_map_style_in_ram_memory.py** - Loading entire dataset into RAM
- **07_lazy_text_dataset.py** - Lazy loading for text data
- **08_memmap_dataset.py** - Memory-mapped files for large datasets
- **09_manual_batching_without_dataloader.py** - Manual batch creation
- **10_random_split_without_dataloader.py** - Splitting datasets manually

### Iterable-Style Datasets (11-16)
Iterable-style datasets return data sequentially, ideal for streams

- **11_iterable_style_minimal.py** - Basic iterable dataset
- **12_iterable_generator.py** - Generator-based dataset
- **13_iterable_iterator_obj.py** - Custom iterator object
- **14_iterable_shuffle_buffer.py** - Implementing shuffle buffer
- **15_iterable_block_shuffle.py** - Block-based shuffling strategy
- **16_iterable_dataloader_workers_demo.py** - Multi-worker DataLoader demo

## 🚀 Quick Start

```python
# Run any example
python 01_map_style_minimal.py
```

## 📚 Learning Path

**Beginner** (Start here):
1. 01_map_style_minimal.py
2. 02_map_style_with_init.py
3. 03_map_style_with_transforms.py

**Intermediate**:
4. 06_map_style_in_ram_memory.py
5. 07_lazy_text_dataset.py
6. 09_manual_batching_without_dataloader.py

**Advanced**:
7. 08_memmap_dataset.py
8. 11-16_iterable_style_*.py

## 🔑 Key Concepts

### Map-Style vs Iterable-Style

**Map-Style (Random Access)**:
- Implements `__len__()` and `__getitem__()`
- Random access: `dataset[42]`
- Best for: Images, tabular data, finite datasets

**Iterable-Style (Sequential)**:
- Implements `__iter__()`
- Sequential access: `for item in dataset`
- Best for: Streams, databases, infinite data

### When to Use Each

**Map-Style**: 
- Fixed-size datasets
- Need shuffling
- Random sampling
- Image classification

**Iterable-Style**:
- Streaming data
- Very large datasets (won't fit in memory)
- Database queries
- Real-time data feeds

## 💡 Tips

1. **Memory Management**: Use lazy loading (07) or memmap (08) for large datasets
2. **Performance**: Use DataLoader with `num_workers > 0` for parallel loading
3. **Shuffling**: Map-style datasets shuffle easily; iterable needs shuffle buffer (14)
4. **Batching**: DataLoader handles batching automatically, but see (09) for manual approach

## 📖 Requirements

```bash
pip install torch torchvision numpy
```

## 🎯 Common Patterns

**Basic Training Loop**:
```python
from torch.utils.data import Dataset, DataLoader

# 1. Create dataset
dataset = MyDataset(data)

# 2. Create dataloader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. Training loop
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        # Train on batch
        pass
```

## 📚 Further Reading

- [PyTorch Dataset Documentation](https://pytorch.org/docs/stable/data.html)
- [DataLoader Documentation](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
- [Writing Custom Datasets](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)

---

**Total Files**: 16 Python examples  
**Difficulty**: Beginner → Advanced  
**Focus**: PyTorch Dataset & DataLoader mastery
