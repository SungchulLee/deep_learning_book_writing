# Memory Management

## Overview

Efficient memory management during inference prevents out-of-memory errors, reduces latency from memory allocation, and enables deployment of larger models on constrained hardware. This is particularly important in quantitative finance where production systems run alongside other critical processes.

## GPU Memory Management

### Memory Monitoring

```python
import torch

def gpu_memory_stats():
    """Print current GPU memory usage."""
    if not torch.cuda.is_available():
        print("No GPU available")
        return
    
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    max_allocated = torch.cuda.max_memory_allocated() / 1024**2
    
    print(f"Allocated: {allocated:.1f} MB")
    print(f"Reserved:  {reserved:.1f} MB")
    print(f"Peak:      {max_allocated:.1f} MB")

def estimate_model_memory(model, input_shape, batch_size=1, dtype=torch.float32):
    """Estimate memory requirements for inference."""
    # Parameter memory
    param_mem = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_mem = sum(b.numel() * b.element_size() for b in model.buffers())
    
    # Input memory
    input_mem = 1
    for d in input_shape:
        input_mem *= d
    input_mem *= batch_size * torch.tensor([], dtype=dtype).element_size()
    
    # Activation memory (rough estimate: 2x parameters for forward pass)
    activation_mem = param_mem * 2
    
    total = param_mem + buffer_mem + input_mem + activation_mem
    
    return {
        'parameters_mb': param_mem / 1024**2,
        'buffers_mb': buffer_mem / 1024**2,
        'input_mb': input_mem / 1024**2,
        'activation_estimate_mb': activation_mem / 1024**2,
        'total_estimate_mb': total / 1024**2,
    }
```

### Memory-Efficient Inference

```python
@torch.no_grad()
def memory_efficient_inference(model, dataloader, device='cuda'):
    """Inference with minimal memory footprint."""
    model.eval()
    results = []
    
    for batch in dataloader:
        inputs = batch[0].to(device, non_blocking=True)
        
        # Forward pass
        outputs = model(inputs)
        
        # Move results to CPU immediately
        results.append(outputs.cpu())
        
        # Free GPU memory
        del inputs, outputs
    
    # Clear cache periodically
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    return torch.cat(results)
```

### Model Sharding for Large Models

```python
def shard_model_inference(model_parts, input_tensor, devices):
    """Run inference across multiple GPUs via model sharding."""
    x = input_tensor.to(devices[0])
    
    for part, device in zip(model_parts, devices):
        x = x.to(device)
        x = part(x)
    
    return x.cpu()
```

## CPU Memory Optimization

```python
import gc

def optimize_cpu_memory():
    """Force garbage collection and release unused memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Memory-mapped model loading for large models
def load_model_mmap(path, model):
    """Load model with memory-mapped tensors (reduces RAM usage)."""
    state_dict = torch.load(path, map_location='cpu', mmap=True, 
                           weights_only=True)
    model.load_state_dict(state_dict)
    return model
```

## Best Practices

- **Always use `torch.no_grad()`** for inference to avoid storing computation graphs
- **Delete intermediate tensors** explicitly and call `torch.cuda.empty_cache()`
- **Move results to CPU** as soon as they are computed
- **Use `non_blocking=True`** for async CPU-GPU transfers with pinned memory
- **Profile memory** with `torch.cuda.memory_summary()` to identify leaks
- **Consider model quantization** to reduce memory footprint by 2-4×
- **Use gradient checkpointing** during training to reduce activation memory

## References

1. PyTorch Memory Management: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
2. PyTorch CUDA Best Practices: https://pytorch.org/docs/stable/notes/cuda.html
