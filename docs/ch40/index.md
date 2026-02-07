# Model Deployment Overview

## Introduction

Model deployment bridges the gap between research and production. While training focuses on achieving high accuracy, deployment requires balancing accuracy with latency, throughput, model size, and resource constraints. Training a model is only half the battle—converting a research prototype into a production-ready system involves serialization, optimization, serving, and monitoring.

## Why Deployment Matters

Converting a research prototype into a production-ready system involves:

1. **Serialization**: Converting models to formats that can run efficiently outside Python
2. **Optimization**: Reducing latency and memory footprint while maintaining accuracy
3. **Serving**: Building APIs and infrastructure to handle prediction requests
4. **Monitoring**: Tracking model performance and behavior in production

## The Deployment Pipeline

A typical deployment workflow consists of five stages:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Training  │────▶│   Convert   │────▶│  Optimize   │────▶│  Benchmark  │────▶│   Deploy    │
│   (PyTorch) │     │   (Export)  │     │ (Quantize)  │     │   (Test)    │     │   (Serve)   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

**Stage 1: Train** — Train or fine-tune the model to achieve target accuracy on the validation set.

**Stage 2: Convert** — Export the model to a deployment-friendly format like ONNX or TorchScript that enables cross-platform execution and hardware acceleration.

**Stage 3: Optimize** — Apply optimization techniques such as quantization, pruning, and knowledge distillation to reduce model size and improve inference speed while maintaining acceptable accuracy.

**Stage 4: Benchmark** — Measure key performance metrics including latency, throughput, memory usage, and accuracy on the target hardware.

**Stage 5: Deploy** — Deploy the optimized model using appropriate serving infrastructure and monitoring systems.

## Key Performance Metrics

### Latency

Latency measures the time required for a single inference:

$$\text{Latency} = t_{\text{preprocess}} + t_{\text{inference}} + t_{\text{postprocess}}$$

Target latency depends on the application:

| Application Type | Target Latency |
|-----------------|----------------|
| Interactive (UI) | < 10 ms |
| Real-time systems | < 100 ms |
| Batch processing | < 1 s |

```python
import time
import torch

def measure_latency(model, input_tensor, num_runs=100, warmup=10):
    """Measure average inference latency."""
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(input_tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)
    
    return {
        'mean_ms': sum(latencies) / len(latencies),
        'p50_ms': sorted(latencies)[len(latencies)//2],
        'p95_ms': sorted(latencies)[int(len(latencies)*0.95)],
        'p99_ms': sorted(latencies)[int(len(latencies)*0.99)]
    }
```

### Throughput

Throughput measures samples processed per unit time:

$$\text{Throughput} = \frac{\text{batch\_size} \times \text{num\_batches}}{\text{total\_time}}$$

Higher batch sizes generally improve throughput at the cost of increased latency:

```python
def measure_throughput(model, input_shape, batch_sizes=[1, 8, 16, 32], num_runs=100):
    """Benchmark throughput across different batch sizes."""
    device = next(model.parameters()).device
    model.eval()
    results = {}
    
    for batch_size in batch_sizes:
        test_input = torch.randn(batch_size, *input_shape).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(test_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        throughput = (batch_size * num_runs) / elapsed
        latency_ms = (elapsed / num_runs) * 1000
        
        results[batch_size] = {
            'throughput': throughput,
            'latency_ms': latency_ms
        }
    
    return results
```

### Model Size

Model size impacts storage requirements, loading time, and memory footprint:

```python
import os
import tempfile

def get_model_size_mb(model):
    """Calculate model size in megabytes."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        torch.save(model.state_dict(), f.name)
        size_mb = os.path.getsize(f.name) / (1024 * 1024)
        os.unlink(f.name)
    return size_mb

def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}
```

### Accuracy Retention

Optimization techniques often trade accuracy for efficiency. The acceptable accuracy drop depends on the application:

| Application | Acceptable Accuracy Drop |
|-------------|-------------------------|
| Safety-critical systems | < 0.1% |
| Production ML systems | < 1-2% |
| Edge/mobile deployment | < 5% |

## Optimization Techniques Summary

| Technique | Size Reduction | Speed Gain | Accuracy Impact |
|-----------|---------------|------------|-----------------|
| Dynamic Quantization | 2-4× | 1.5-2× | Minimal |
| Static Quantization | 4× | 2-4× | Low |
| Pruning (30%) | 1.3× | Variable | Minimal |
| Knowledge Distillation | Variable | Variable | Low |
| ONNX Optimization | 1.2-1.5× | 1.2-1.8× | None |
| TensorRT | 1× | 2-5× | Low (FP16) |

## Key Serialization Formats

### 1. PyTorch Native (.pt, .pth)

The default serialization format using `torch.save()`:

```python
# Save model weights (recommended)
torch.save(model.state_dict(), 'model.pt')

# Load weights into model
model = ModelClass()
model.load_state_dict(torch.load('model.pt'))
```

**Advantages**: Simple, preserves full PyTorch functionality  
**Disadvantages**: Requires Python and PyTorch to load

### 2. TorchScript (.pt)

Serializable representation that can run independently from Python:

```python
# Trace model with example input (for models without control flow)
traced_model = torch.jit.trace(model, example_input)
traced_model.save('model_traced.pt')

# Or script model (handles control flow)
scripted_model = torch.jit.script(model)
scripted_model.save('model_scripted.pt')

# Load without Python
loaded_model = torch.jit.load('model_traced.pt')
```

**Advantages**: Language-independent, optimized execution, C++ deployment  
**Disadvantages**: Some Python features not supported

### 3. ONNX (.onnx)

Open Neural Network Exchange format for cross-framework compatibility:

```python
import torch.onnx

torch.onnx.export(
    model,
    example_input,
    'model.onnx',
    export_params=True,
    opset_version=17,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
```

**Advantages**: Framework-agnostic, broad runtime support, optimization tools  
**Disadvantages**: Not all operations supported

### 4. TensorRT (.engine)

NVIDIA's inference optimizer for maximum GPU performance:

```python
import torch_tensorrt

trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input(shape=[1, 3, 224, 224])],
    enabled_precisions={torch.float16}
)
```

**Advantages**: Maximum GPU inference speed, automatic kernel tuning  
**Disadvantages**: NVIDIA GPUs only, requires calibration for INT8

## Format Selection Guide

| Scenario | Recommended Format | Rationale |
|----------|-------------------|-----------|
| Development/Prototyping | PyTorch (.pt) | Full flexibility, easy debugging |
| Production (C++/Mobile) | TorchScript | Python-independent, optimized |
| Cross-Framework | ONNX | Maximum compatibility |
| Maximum GPU Speed | TensorRT | Hardware-optimized kernels |
| Edge Devices | ONNX + Runtime | Lightweight, portable |
| Cloud Deployment | TorchScript or ONNX | Balanced speed and compatibility |

## Deployment Strategies

### Online Inference (REST API)

Real-time prediction via HTTP endpoints:

- **Latency-critical**: Single request, immediate response
- **Use cases**: Recommendation systems, fraud detection, chatbots

### Batch Inference

Process large datasets offline:

- **Throughput-focused**: Maximize samples per second
- **Use cases**: Daily predictions, report generation, ETL pipelines

### Edge Deployment

On-device inference without network dependency:

- **Resource-constrained**: Limited memory, compute, power
- **Use cases**: Mobile apps, IoT sensors, autonomous systems

### Streaming Inference

Process continuous data streams:

- **Low-latency, high-throughput**: Balance both requirements
- **Use cases**: Financial trading, video analytics, monitoring

## Production Considerations

### Reliability Requirements

Production systems must handle:

- **Graceful degradation**: Fallback behavior when model fails
- **Error handling**: Proper responses for invalid inputs
- **Health checks**: Readiness and liveness probes
- **Rollback capability**: Quick revert to previous versions

### Scalability Patterns

As traffic grows, consider:

- **Horizontal scaling**: Multiple model replicas behind load balancer
- **Vertical scaling**: Larger machines with more GPU memory
- **Auto-scaling**: Dynamic capacity based on demand
- **Caching**: Store predictions for repeated inputs

## Deployment Checklist

Before deploying a model to production:

- [ ] Model accuracy validated on held-out test set
- [ ] Latency meets requirements on target hardware (p50, p95, p99)
- [ ] Throughput sufficient for expected load
- [ ] Model size acceptable for deployment environment
- [ ] Memory usage within available limits
- [ ] Cross-platform compatibility verified (if applicable)
- [ ] Error handling implemented for edge cases
- [ ] Health checks configured (readiness/liveness probes)
- [ ] Monitoring and logging configured
- [ ] Rollback strategy defined
- [ ] A/B testing framework prepared

## Chapter Organization

This chapter covers deployment in depth:

1. **[ONNX Export](onnx.md)** — Converting models to ONNX format for cross-platform deployment
2. **[TorchScript](torchscript.md)** — Compiling models for optimized execution without Python
3. **[TensorRT](tensorrt.md)** — Maximum performance optimization for NVIDIA GPUs
4. **[Serving Frameworks](serving.md)** — Production serving with TorchServe, Triton, and cloud platforms
5. **[Model Validation](validation.md)** — Ensuring correctness and consistency across formats
6. **[Performance Monitoring](monitoring.md)** — Tracking model behavior in production

## Learning Objectives

After completing this section, you will be able to:

1. Choose the appropriate serialization format for your deployment target
2. Export PyTorch models to TorchScript and ONNX
3. Apply optimization techniques for production performance
4. Build production-ready inference APIs
5. Implement monitoring and logging for deployed models
6. Design scalable and reliable ML serving systems

## References

1. PyTorch Deployment Documentation: https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html
2. ONNX Runtime Performance Tuning: https://onnxruntime.ai/docs/performance/
3. TensorRT Developer Guide: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/
4. MLOps: Continuous Delivery for Machine Learning: https://ml-ops.org/
5. Google ML Best Practices: https://developers.google.com/machine-learning/guides/rules-of-ml
