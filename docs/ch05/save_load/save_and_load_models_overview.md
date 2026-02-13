# PyTorch Model Saving and Loading - Complete Guide

A comprehensive collection of tutorials covering all aspects of PyTorch model saving, loading, and deployment.

## 📚 Overview

This collection provides in-depth, fully-commented tutorials on PyTorch model persistence, from basic save/load operations to advanced deployment strategies. Each tutorial is standalone and includes runnable examples with detailed explanations.

## 📁 Tutorial Collection

### Core Tutorials

#### 01_basic_save_load.py
**Fundamentals of Model Saving and Loading**
- Three core methods: `torch.save()`, `torch.load()`, `load_state_dict()`
- Complete vs. State Dict saving
- Training checkpoint management
- Device-specific loading (CPU/GPU)
- Best practices and common pitfalls

**Topics Covered:**
- Saving entire models vs state dictionaries
- Creating and loading training checkpoints
- Cross-device model transfer
- Evaluation vs training modes
- File size considerations

### Advanced Checkpoint Management

#### 02_checkpoint_manager.py
**Professional Checkpoint Management System**
- Automated checkpoint saving with metadata
- Best model tracking based on metrics
- Automatic cleanup of old checkpoints
- Resume training from any checkpoint
- Checkpoint versioning

**Features:**
- `CheckpointManager` class for production use
- Configurable retention policies
- Metric-based best model tracking
- Timestamp and metadata logging
- List and inspect all checkpoints

#### 03_model_versioning.py
**Model Versioning and Metadata Tracking**
- Version numbering for models
- Configuration management
- Hyperparameter logging
- Model fingerprinting (hash verification)
- Training history tracking

**Features:**
- `VersionedModel` wrapper class
- Automatic version tracking
- JSON metadata export
- Integrity verification
- Reproducibility support

### Deployment and Export

#### 04_onnx_export.py
**ONNX Export for Cross-Platform Deployment**
- Export PyTorch models to ONNX format
- Dynamic input shapes
- Model verification and validation
- ONNX Runtime inference
- Output comparison

**Use Cases:**
- Cross-framework compatibility
- Mobile deployment preparation
- Hardware acceleration
- Production inference optimization

#### 05_torchscript_export.py
**TorchScript for Production Deployment**
- Tracing vs Scripting methods
- Model optimization for inference
- Mobile deployment preparation
- C++ runtime compatibility

**Features:**
- `torch.jit.trace()` for simple models
- `torch.jit.script()` for control flow
- Model freezing and optimization
- Performance improvements
- Production-ready exports

### Specialized Scenarios

#### 06_transfer_learning.py
**Transfer Learning Save/Load Patterns**
- Loading pre-trained models
- Freezing and unfreezing layers
- Saving fine-tuned models
- Partial state dict loading
- Handling architecture changes

**Topics:**
- Pre-trained model modification
- Layer freezing strategies
- Optimizer configuration for frozen layers
- Missing/unexpected key handling
- Feature extractor patterns

#### 07_distributed_training.py
**Multi-GPU and Distributed Training Checkpoints**
- DataParallel checkpoint handling
- DistributedDataParallel support
- 'module.' prefix management
- Multi-GPU best practices

**Key Concepts:**
- Saving wrapped vs unwrapped models
- Prefix cleaning utilities
- Portable checkpoint creation
- Cross-GPU-count compatibility

## 🚀 Quick Start

Each tutorial can be run independently:

```bash
# Run any tutorial
python 01_basic_save_load.py
python 02_checkpoint_manager.py
# ... etc
```

### Installation Requirements

```bash
# Core requirements
pip install torch torchvision

# For ONNX export (optional)
pip install onnx onnxruntime

# For visualization (optional)
pip install matplotlib
```

## 💡 Key Concepts Summary

### Saving Methods

**Method 1: Save Entire Model**
```python
# Simple but not recommended for production
torch.save(model, 'model.pth')
model = torch.load('model.pth')
model.eval()
```

**Method 2: Save State Dict (RECOMMENDED)**
```python
# Production standard
torch.save(model.state_dict(), 'model.pth')
model = Model(*args, **kwargs)
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

**Method 3: Save Complete Training Checkpoint**
```python
# For resuming training
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# Load and resume
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

### Device Transfer

```python
# CPU to GPU
device = torch.device('cuda')
model.load_state_dict(torch.load('model.pth', map_location=device))
model.to(device)

# GPU to CPU
device = torch.device('cpu')
model.load_state_dict(torch.load('model.pth', map_location=device))

# GPU to different GPU
model.load_state_dict(torch.load('model.pth', map_location='cuda:1'))
```

## 📊 Tutorial Comparison

| Tutorial | Level | Topics | Use Case |
|----------|-------|--------|----------|
| 01_basic_save_load | Beginner | Core concepts | Learning fundamentals |
| 02_checkpoint_manager | Intermediate | Automation | Production training |
| 03_model_versioning | Intermediate | Tracking | Experiment management |
| 04_onnx_export | Advanced | Deployment | Cross-platform |
| 05_torchscript_export | Advanced | Optimization | Production inference |
| 06_transfer_learning | Intermediate | Fine-tuning | Pre-trained models |
| 07_distributed_training | Advanced | Multi-GPU | Large-scale training |

## ✅ Best Practices

### DO's
- ✓ Use `state_dict()` for production models
- ✓ Always call `model.eval()` before inference
- ✓ Save checkpoints regularly during training
- ✓ Include metadata in checkpoints
- ✓ Use meaningful filenames with version/epoch
- ✓ Verify loaded models before deployment
- ✓ Handle device placement explicitly

### DON'Ts
- ✗ Don't save entire model in production
- ✗ Don't forget to set eval mode for inference
- ✗ Don't ignore device compatibility
- ✗ Don't mix training and inference modes
- ✗ Don't save temporary/cached tensors
- ✗ Don't assume same device when loading

## 🔍 Common Issues and Solutions

### Issue 1: State Dict Size Mismatch
**Problem:** Model architecture doesn't match saved state dict

**Solution:**
```python
# Use strict=False for partial loading
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
```

### Issue 2: CUDA Out of Memory
**Problem:** Loading large model on GPU

**Solution:**
```python
# Load to CPU first
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
# Then move to GPU if needed
model.to('cuda')
```

### Issue 3: Module Prefix with DataParallel
**Problem:** State dict has unexpected 'module.' prefix

**Solution:**
```python
# Remove prefix when saving
state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
torch.save(state_dict, 'model.pth')
```

### Issue 4: Different Behavior After Loading
**Problem:** Model produces different results

**Solution:**
```python
# Ensure eval mode
model.eval()

# Disable gradient computation
with torch.no_grad():
    output = model(input)
```

## 📈 Learning Path

### Beginner
1. Start with **01_basic_save_load.py**
2. Understand state dict concept
3. Practice checkpoint saving

### Intermediate
4. Study **02_checkpoint_manager.py**
5. Learn **06_transfer_learning.py**
6. Explore **03_model_versioning.py**

### Advanced
7. Master **04_onnx_export.py**
8. Understand **05_torchscript_export.py**
9. Handle **07_distributed_training.py**

## 🎯 Use Case Guide

**Training Deep Learning Models**
→ Use `02_checkpoint_manager.py` patterns

**Fine-tuning Pre-trained Models**
→ Follow `06_transfer_learning.py`

**Deploying to Production**
→ Apply `04_onnx_export.py` or `05_torchscript_export.py`

**Multi-GPU Training**
→ Implement `07_distributed_training.py` strategies

**Experiment Tracking**
→ Utilize `03_model_versioning.py` approach

## 📚 Additional Resources

- [Official PyTorch Docs](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- [TorchScript Documentation](https://pytorch.org/docs/stable/jit.html)
- [ONNX Official Site](https://onnx.ai/)
- [PyTorch Mobile](https://pytorch.org/mobile/home/)

## 🤝 Contributing

These tutorials are designed to be:
- **Clear**: Every concept thoroughly explained
- **Practical**: Real-world applicable examples
- **Complete**: Cover all major use cases
- **Runnable**: All code is tested and working

## 📄 License

Educational resource for PyTorch model persistence.
Free to use for learning and reference.

## 🙏 Acknowledgments

Based on patterns from:
- PyTorch official tutorials
- Production deployment experience
- Community best practices
- Patrick Loeber's PyTorch Tutorial

---

**Last Updated:** November 2025
**PyTorch Version:** Compatible with PyTorch 1.10+
**Python Version:** 3.8+

For questions or improvements, refer to official PyTorch documentation.

## 🎓 Summary

This collection transforms you from basic save/load operations to production-ready model deployment. Each tutorial builds on previous concepts while remaining independently useful.

**Start with `01_basic_save_load.py` and progress through the series based on your needs!**
