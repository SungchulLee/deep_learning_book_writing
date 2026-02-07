# TensorRT

## Introduction

TensorRT is NVIDIA's high-performance deep learning inference optimizer and runtime. It delivers the lowest latency and highest throughput for GPU inference by applying kernel optimization, precision calibration, and layer fusion.

## Why TensorRT?

TensorRT provides maximum performance on NVIDIA hardware:

**Kernel Auto-Tuning** — Selects optimal kernel implementations for the target GPU architecture.

**Precision Optimization** — Supports FP32, FP16, and INT8 with automatic calibration for reduced precision.

**Layer Fusion** — Combines multiple operations into single kernels to reduce memory bandwidth and launch overhead.

**Tensor Memory Optimization** — Minimizes memory footprint through intelligent memory reuse.

Performance gains of 2-10× over native PyTorch are common, with larger gains for transformer models.

## TensorRT Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TensorRT Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐ │
│  │  ONNX   │───▶│  Parser  │───▶│ Optimizer│───▶│ Engine │ │
│  │  Model  │    │          │    │          │    │ (.trt) │ │
│  └─────────┘    └──────────┘    └──────────┘    └────────┘ │
│                       │                │                    │
│                       ▼                ▼                    │
│              ┌────────────────────────────────┐            │
│              │     Optimization Steps         │            │
│              ├────────────────────────────────┤            │
│              │ • Layer fusion                 │            │
│              │ • Kernel selection             │            │
│              │ • Precision calibration        │            │
│              │ • Memory allocation            │            │
│              └────────────────────────────────┘            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Installation

TensorRT can be used through several interfaces:

```bash
# Torch-TensorRT (recommended for PyTorch users)
pip install torch-tensorrt

# ONNX-TensorRT (for ONNX models)
pip install tensorrt

# Verify installation
python -c "import torch_tensorrt; print(torch_tensorrt.__version__)"
```

Requirements:

- NVIDIA GPU with compute capability >= 7.0 (Volta or newer)
- CUDA 11.x or 12.x
- cuDNN 8.x

## Torch-TensorRT

### Basic Compilation

Convert PyTorch models directly to TensorRT:

```python
import torch
import torch_tensorrt

class ResNetBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(channels)
        self.conv2 = torch.nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(channels)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

model = ResNetBlock(64).cuda().eval()

# Compile with TensorRT
trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input(
        shape=[1, 64, 32, 32],
        dtype=torch.float32
    )],
    enabled_precisions={torch.float32}
)

# Run inference
input_tensor = torch.randn(1, 64, 32, 32).cuda()
output = trt_model(input_tensor)
```

### Dynamic Shapes

Support variable batch sizes and dimensions:

```python
trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input(
        min_shape=[1, 64, 32, 32],
        opt_shape=[8, 64, 32, 32],   # Optimization target
        max_shape=[32, 64, 32, 32],
        dtype=torch.float32
    )],
    enabled_precisions={torch.float32}
)

# Works with any batch size from 1 to 32
batch_1 = torch.randn(1, 64, 32, 32).cuda()
batch_16 = torch.randn(16, 64, 32, 32).cuda()

output_1 = trt_model(batch_1)
output_16 = trt_model(batch_16)
```

### FP16 Precision

Enable half-precision for approximately 2× speedup on modern GPUs:

```python
trt_model_fp16 = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input(
        shape=[1, 64, 32, 32],
        dtype=torch.float16
    )],
    enabled_precisions={torch.float16}
)

# Input must match specified precision
input_fp16 = torch.randn(1, 64, 32, 32, dtype=torch.float16).cuda()
output = trt_model_fp16(input_fp16)
```

### INT8 Quantization

INT8 provides maximum performance with calibration:

```python
def calibration_data_generator():
    """Generate calibration data for INT8 quantization."""
    for _ in range(100):
        yield torch.randn(1, 64, 32, 32).cuda()

trt_model_int8 = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input(
        shape=[1, 64, 32, 32],
        dtype=torch.float32  # Input as FP32
    )],
    enabled_precisions={torch.float32, torch.int8},
    calibrator=torch_tensorrt.ptq.DataLoaderCalibrator(
        calibration_data_generator(),
        cache_file="calibration.cache",
        use_cache=True,
        algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2
    )
)
```

Calibration algorithms:

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| `ENTROPY_CALIBRATION` | Minimizes KL divergence | Default choice |
| `ENTROPY_CALIBRATION_2` | Improved entropy method | Better accuracy |
| `MINMAX_CALIBRATION` | Uses min/max ranges | Fast calibration |

## ONNX to TensorRT

Convert ONNX models to TensorRT for maximum compatibility:

```python
import tensorrt as trt

def build_engine_from_onnx(onnx_path, engine_path, fp16=False, int8=False):
    """
    Build TensorRT engine from ONNX model.
    
    Args:
        onnx_path: Path to ONNX model
        engine_path: Path to save TensorRT engine
        fp16: Enable FP16 precision
        int8: Enable INT8 precision (requires calibration)
        
    Returns:
        TensorRT engine
    """
    logger = trt.Logger(trt.Logger.WARNING)
    
    # Create builder
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX model
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"ONNX Parse Error: {parser.get_error(error)}")
            raise RuntimeError("Failed to parse ONNX model")
    
    # Configure builder
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1 GB workspace
    
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        # INT8 requires calibration (see below)
    
    # Build engine
    print("Building TensorRT engine (this may take a few minutes)...")
    engine = builder.build_engine(network, config)
    
    if engine is None:
        raise RuntimeError("Failed to build TensorRT engine")
    
    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    print(f"Engine saved to {engine_path}")
    return engine

# Example usage
engine = build_engine_from_onnx("model.onnx", "model.trt", fp16=True)
```

### Dynamic Shapes with ONNX

```python
def build_engine_dynamic_shapes(onnx_path, engine_path):
    """Build engine with dynamic batch size."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())
    
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30
    
    # Create optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()
    
    # Get input name and set shape ranges
    input_name = network.get_input(0).name
    
    # (min_shape, optimal_shape, max_shape)
    profile.set_shape(
        input_name,
        min=(1, 3, 224, 224),      # Minimum batch size 1
        opt=(8, 3, 224, 224),      # Optimal batch size 8
        max=(32, 3, 224, 224)      # Maximum batch size 32
    )
    
    config.add_optimization_profile(profile)
    
    engine = builder.build_engine(network, config)
    
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    return engine
```

## TensorRT Runtime

Run inference with a TensorRT engine:

```python
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

class TensorRTInference:
    """TensorRT inference wrapper."""
    
    def __init__(self, engine_path):
        """
        Initialize TensorRT inference.
        
        Args:
            engine_path: Path to serialized TensorRT engine
        """
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Load engine
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers (use pinned memory for speed)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
    
    def infer(self, input_data):
        """
        Run inference.
        
        Args:
            input_data: Input numpy array
            
        Returns:
            Output numpy array
        """
        # Copy input to host buffer
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        
        # Transfer to GPU (async)
        cuda.memcpy_htod_async(
            self.inputs[0]['device'],
            self.inputs[0]['host'],
            self.stream
        )
        
        # Execute inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        
        # Transfer back to CPU (async)
        cuda.memcpy_dtoh_async(
            self.outputs[0]['host'],
            self.outputs[0]['device'],
            self.stream
        )
        
        # Synchronize
        self.stream.synchronize()
        
        return self.outputs[0]['host'].copy()

# Usage
trt_inference = TensorRTInference("model.trt")
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
output = trt_inference.infer(input_data)
print(f"Output shape: {output.shape}")
```

## Precision Modes

### FP32 (Default)

Full 32-bit floating point precision:

```python
config = builder.create_builder_config()
# FP32 is default, no flag needed
```

### FP16 (Half Precision)

16-bit floating point for ~2x speedup:

```python
config.set_flag(trt.BuilderFlag.FP16)

# Check if GPU supports FP16
if builder.platform_has_fast_fp16:
    print("GPU supports native FP16")
```

**When to use**: Most cases where accuracy loss is acceptable (~0.1-1% typically).

### INT8 (Quantization)

8-bit integer for maximum performance (requires calibration):

```python
class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    """INT8 calibrator using entropy method."""
    
    def __init__(self, calibration_data, cache_file="calibration.cache"):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.data = calibration_data
        self.cache_file = cache_file
        self.current_index = 0
        self.batch_size = 1
        self.device_input = cuda.mem_alloc(
            calibration_data[0:1].nbytes
        )
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_batch(self, names):
        if self.current_index >= len(self.data):
            return None
        
        batch = self.data[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size
        
        cuda.memcpy_htod(self.device_input, np.ascontiguousarray(batch))
        return [int(self.device_input)]
    
    def read_calibration_cache(self):
        try:
            with open(self.cache_file, 'rb') as f:
                return f.read()
        except FileNotFoundError:
            return None
    
    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)

# Build INT8 engine with calibration
def build_int8_engine(onnx_path, calibration_data, engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())
    
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30
    
    # Enable INT8
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = Int8Calibrator(calibration_data)
    
    engine = builder.build_engine(network, config)
    
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    return engine
```

### Precision Selection Guide

| Model Type | Recommended Precision | Notes |
|------------|----------------------|-------|
| Classification | FP16 | Safe, minimal accuracy loss |
| Detection | FP16 | Safe for most cases |
| Segmentation | FP16 | Verify on edge cases |
| NLP/Transformers | FP16 | Verify attention outputs |
| Quantization-sensitive | FP32 | When accuracy is critical |

## Optimization Techniques

### Layer Fusion

TensorRT automatically fuses compatible operations:

```
Before:  Conv → BatchNorm → ReLU  (3 kernels, 3 memory transfers)
After:   ConvBNReLU               (1 kernel, 1 memory transfer)
```

Common fusion patterns:
- Conv + BatchNorm + ReLU
- Conv + Add + ReLU (residual connections)
- MatMul + Add (linear layers)
- Multiple element-wise operations

### Precision Calibration

For representative calibration data:

```python
class CalibrationDataset:
    """Calibration dataset for INT8 quantization."""
    
    def __init__(self, dataloader, num_batches=100):
        self.data = []
        for i, (images, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            self.data.append(images.numpy())
    
    def __iter__(self):
        return iter(self.data)

# Use with calibrator
calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
    CalibrationDataset(train_loader),
    cache_file="calibration.cache"
)
```

### Memory Optimization

Configure workspace and memory pools:

```python
config = builder.create_builder_config()

# Workspace for layer algorithms
config.max_workspace_size = 2 << 30  # 2 GB

# Timing cache for faster builds
timing_cache = config.create_timing_cache(b"")
config.set_timing_cache(timing_cache, ignore_mismatch=False)

# Limit memory for constrained environments
config.max_workspace_size = 512 << 20  # 512 MB
```

## Benchmarking TensorRT

Compare performance across configurations:

```python
import time
import torch

def benchmark_trt(model_name, pytorch_model, trt_model, input_shape, 
                  num_runs=1000, warmup=100):
    """Benchmark PyTorch vs TensorRT."""
    input_tensor = torch.randn(*input_shape).cuda()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = pytorch_model(input_tensor)
            _ = trt_model(input_tensor)
    
    torch.cuda.synchronize()
    
    # Benchmark PyTorch
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = pytorch_model(input_tensor)
    torch.cuda.synchronize()
    pytorch_time = time.perf_counter() - start
    
    # Benchmark TensorRT
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = trt_model(input_tensor)
    torch.cuda.synchronize()
    trt_time = time.perf_counter() - start
    
    pytorch_latency = (pytorch_time / num_runs) * 1000
    trt_latency = (trt_time / num_runs) * 1000
    speedup = pytorch_time / trt_time
    
    print(f"\n=== {model_name} Benchmark ===")
    print(f"PyTorch latency:  {pytorch_latency:.3f} ms")
    print(f"TensorRT latency: {trt_latency:.3f} ms")
    print(f"Speedup: {speedup:.2f}x")
    
    return {
        'pytorch_latency_ms': pytorch_latency,
        'trt_latency_ms': trt_latency,
        'speedup': speedup
    }
```

Typical speedups by model type:

| Model | FP32 Speedup | FP16 Speedup | INT8 Speedup |
|-------|-------------|-------------|--------------|
| ResNet-50 | 1.5-2× | 2-3× | 3-5× |
| BERT-base | 2-3× | 3-5× | 4-6× |
| ViT-B/16 | 2-4× | 4-6× | 5-8× |
| GPT-2 | 2-3× | 4-6× | 5-7× |

## Common Issues

### Issue 1: Unsupported Operations

```python
# Check for unsupported ops during parsing
parser = trt.OnnxParser(network, logger)
success = parser.parse(onnx_model)

if not success:
    for i in range(parser.num_errors):
        error = parser.get_error(i)
        print(f"Error {i}: {error.desc()}")
        print(f"  Node: {error.node()}")
```

**Solutions:**
1. Update TensorRT to latest version
2. Use ONNX opset that TensorRT supports
3. Implement custom plugin for unsupported ops

### Issue 2: Accuracy Degradation

```python
def verify_accuracy(pytorch_model, trt_engine, test_data, tolerance=1e-3):
    """Verify TensorRT accuracy against PyTorch."""
    pytorch_model.eval()
    
    max_diff = 0
    for batch in test_data:
        # PyTorch inference
        with torch.no_grad():
            pytorch_out = pytorch_model(batch).cpu().numpy()
        
        # TensorRT inference
        trt_out = trt_engine.infer(batch.numpy())
        
        # Compare
        diff = np.abs(pytorch_out - trt_out).max()
        max_diff = max(max_diff, diff)
    
    print(f"Maximum difference: {max_diff}")
    
    if max_diff > tolerance:
        print("⚠ Accuracy degradation detected!")
        print("Consider:")
        print("  - Using FP32 instead of FP16/INT8")
        print("  - Checking specific layers")
        print("  - Adjusting calibration data")
    
    return max_diff < tolerance
```

### Issue 3: Memory Issues

```python
# Limit workspace size
config.max_workspace_size = 512 << 20  # 512 MB instead of 1 GB

# Enable memory reuse
config.set_flag(trt.BuilderFlag.REFIT)
```

### Fallback Handling

Use PyTorch for unsupported ops with Torch-TensorRT:

```python
# Allow fallback to PyTorch for unsupported ops
trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input(shape=[1, 3, 224, 224])],
    enabled_precisions={torch.float16},
    torch_executed_ops=["aten::grid_sampler"],  # Run in PyTorch
    truncate_long_and_double=True
)
```

## Production Deployment

### Serialization

Save and load TensorRT engines:

```python
# Save Torch-TensorRT model
torch.jit.save(trt_model, "model_trt.pt")

# Load model
loaded_model = torch.jit.load("model_trt.pt")
```

### Version Compatibility

TensorRT engines are version-specific:

- GPU architecture (compute capability)
- TensorRT version
- CUDA version

Rebuild engines when any of these change.

### Multi-GPU Deployment

```python
# Build engine for specific GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Or use device context
with torch.cuda.device(0):
    trt_model = torch_tensorrt.compile(model, ...)
```

## Best Practices

1. **Start with FP16** — Often provides 2× speedup with minimal accuracy loss.

2. **Profile first** — Identify which layers benefit most from optimization:
   ```python
   config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
   # Use: nsys profile python inference.py
   ```

3. **Use calibration cache** — Speed up INT8 calibration on subsequent builds.

4. **Match input shapes** — Use optimization profiles that match production workloads.

5. **Validate accuracy** — Always compare outputs against baseline:
   ```python
   max_diff = torch.max(torch.abs(pytorch_out - trt_out)).item()
   assert max_diff < 1e-3, f"Accuracy regression: {max_diff}"
   ```

6. **Version lock** — Pin TensorRT, CUDA, and driver versions in production.

7. **Calibration data guidelines** for INT8:
   - Use representative production data
   - Include edge cases
   - 500-1000 samples typically sufficient
   - Cache calibration results for reproducibility

8. **Optimize input pipeline**:
   ```python
   # Use pinned memory for faster transfers
   host_input = cuda.pagelocked_empty(input_size, dtype=np.float32)
   # Async transfers
   cuda.memcpy_htod_async(device_input, host_input, stream)
   ```

## Complete Workflow

```python
import torch
import torch_tensorrt
import time

def tensorrt_deployment_workflow(model, input_shape, precision='fp16'):
    """Complete TensorRT deployment workflow."""
    print("=== TensorRT Deployment Workflow ===\n")
    
    model = model.cuda().eval()
    example_input = torch.randn(1, *input_shape).cuda()
    
    # Step 1: Determine precision
    print(f"1. Configuring for {precision.upper()} precision...")
    if precision == 'fp16':
        precisions = {torch.float16}
        input_dtype = torch.float16
        example_input = example_input.half()
    elif precision == 'int8':
        precisions = {torch.float32, torch.int8}
        input_dtype = torch.float32
    else:
        precisions = {torch.float32}
        input_dtype = torch.float32
    
    # Step 2: Compile
    print("2. Compiling with TensorRT...")
    start = time.time()
    trt_model = torch_tensorrt.compile(
        model,
        inputs=[torch_tensorrt.Input(
            shape=[1, *input_shape],
            dtype=input_dtype
        )],
        enabled_precisions=precisions,
        truncate_long_and_double=True
    )
    compile_time = time.time() - start
    print(f"   ✓ Compilation completed in {compile_time:.1f}s")
    
    # Step 3: Verify accuracy
    print("3. Verifying accuracy...")
    with torch.no_grad():
        if precision == 'fp16':
            pytorch_output = model(example_input.float()).half()
        else:
            pytorch_output = model(example_input)
    trt_output = trt_model(example_input)
    
    max_diff = torch.max(torch.abs(pytorch_output - trt_output)).item()
    print(f"   Max difference: {max_diff:.2e}")
    
    # Step 4: Benchmark
    print("4. Benchmarking...")
    num_runs = 500
    
    # PyTorch
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(example_input.float() if precision == 'fp16' else example_input)
    torch.cuda.synchronize()
    pytorch_time = time.time() - start
    
    # TensorRT
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        _ = trt_model(example_input)
    torch.cuda.synchronize()
    trt_time = time.time() - start
    
    print(f"   PyTorch: {(pytorch_time/num_runs)*1000:.2f} ms")
    print(f"   TensorRT: {(trt_time/num_runs)*1000:.2f} ms")
    print(f"   Speedup: {pytorch_time/trt_time:.2f}x")
    
    # Step 5: Save
    print("5. Saving optimized model...")
    torch.jit.save(trt_model, "model_tensorrt.pt")
    print("   ✓ Model saved to model_tensorrt.pt")
    
    print("\n✓ Workflow complete!")
    return trt_model

# Usage
if torch.cuda.is_available():
    model = ResNetBlock(64)
    trt_model = tensorrt_deployment_workflow(model, (64, 32, 32), precision='fp16')
```

## Summary

TensorRT provides maximum GPU inference performance:

1. **Automatic optimization**: Kernel selection, layer fusion, precision tuning
2. **Multiple precision modes**: FP32, FP16, INT8 with calibration
3. **Dynamic shapes**: Variable batch sizes and dimensions
4. **Integration options**: Direct API or Torch-TensorRT

Key considerations:
- Build time is long but one-time
- Engines are hardware-specific (not portable)
- Always verify numerical accuracy
- Profile to understand bottlenecks

## References

1. TensorRT Documentation: https://docs.nvidia.com/deeplearning/tensorrt/
2. Torch-TensorRT: https://pytorch.org/TensorRT/
3. NVIDIA Developer Blog - TensorRT: https://developer.nvidia.com/blog/tag/tensorrt/
4. TensorRT Best Practices: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#best-practices
