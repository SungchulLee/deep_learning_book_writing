# CPU Optimization

## Overview

CPU inference optimization is critical for deployment scenarios where GPUs are unavailable or cost-prohibitive—common in edge devices, serverless functions, and many production trading systems where deterministic latency matters more than raw throughput.

## PyTorch CPU Optimizations

### Thread Configuration

```python
import torch
import os

# Set number of threads for intra-op parallelism
torch.set_num_threads(4)

# Set number of threads for inter-op parallelism
torch.set_num_interop_threads(2)

# Environment variables (set before importing torch)
# OMP_NUM_THREADS=4
# MKL_NUM_THREADS=4
```

### Operator Fusion with TorchScript

```python
model.eval()
example_input = torch.randn(1, 50)

# Trace and optimize
traced = torch.jit.trace(model, example_input)
frozen = torch.jit.freeze(traced)
optimized = torch.jit.optimize_for_inference(frozen)

# Benchmark
import time
n = 1000
start = time.perf_counter()
with torch.no_grad():
    for _ in range(n):
        optimized(example_input)
cpu_time = (time.perf_counter() - start) / n * 1000
print(f"Optimized CPU inference: {cpu_time:.2f} ms")
```

### Quantization for CPU

INT8 quantization provides the largest CPU speedups:

```python
# Dynamic quantization (simplest)
quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Static quantization (best performance)
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
prepared = torch.quantization.prepare(model)
# Run calibration data...
quantized = torch.quantization.convert(prepared)
```

### ONNX Runtime CPU

```python
import onnxruntime as ort

# Create optimized CPU session
session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 4
session_options.inter_op_num_threads = 2
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    'model.onnx',
    session_options,
    providers=['CPUExecutionProvider']
)

# Run inference
result = session.run(None, {'input': input_numpy})
```

## Memory Layout Optimization

```python
# Ensure contiguous memory layout
x = x.contiguous()

# Use channels_last format for CNNs (faster on CPU)
model = model.to(memory_format=torch.channels_last)
x = x.to(memory_format=torch.channels_last)
```

## Best Practices

- **Set thread counts** appropriately for your hardware (typically = physical cores)
- **Use INT8 quantization** for maximum CPU speedup (2-4×)
- **Profile with `torch.profiler`** to identify bottlenecks
- **Batch requests** when possible—CPU inference also benefits from batching
- **Use ONNX Runtime** for production CPU deployment (often faster than native PyTorch)
- **Enable `channels_last`** memory format for convolutional models

## References

1. PyTorch CPU Performance: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
2. ONNX Runtime CPU Optimization: https://onnxruntime.ai/docs/performance/tune-performance.html
