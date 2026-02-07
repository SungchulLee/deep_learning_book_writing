# ONNX Export

## Introduction

ONNX (Open Neural Network Exchange) is an open format for representing machine learning models. It enables model interoperability across frameworks (PyTorch, TensorFlow, scikit-learn) and deployment on diverse hardware through ONNX Runtime's execution providers.

## Why ONNX?

ONNX addresses a fundamental challenge in model deployment: framework lock-in. Benefits include:

**Cross-Platform Compatibility** — A single ONNX model runs on CPU, GPU, NPU, and custom accelerators without modification.

**Framework Independence** — Train in PyTorch or TensorFlow, deploy anywhere. No need to port model code.

**Hardware Optimization** — Leverage vendor-specific optimizations from Intel, NVIDIA, Qualcomm, and others.

**Optimized Runtime** — ONNX Runtime provides hardware-specific optimizations and graph-level transformations.

**Standardized Format** — Models are stored in a well-defined protobuf format with versioned operator specifications.

**Language Independence** — Deploy without Python dependencies in production environments.

## ONNX Architecture

An ONNX model consists of:

```
┌─────────────────────────────────────────────────────────────┐
│                       ONNX Model                            │
├─────────────────────────────────────────────────────────────┤
│  Metadata (version, producer, domain)                       │
├─────────────────────────────────────────────────────────────┤
│  Graph                                                       │
│  ├── Inputs (name, type, shape)                             │
│  ├── Outputs (name, type, shape)                            │
│  ├── Nodes (operations with inputs/outputs)                 │
│  └── Initializers (constant tensors/weights)                │
└─────────────────────────────────────────────────────────────┘
```

The graph representation enables framework-independent optimizations and hardware-specific execution.

## Converting PyTorch to ONNX

### Basic Export

The `torch.onnx.export()` function traces model execution and converts operations to ONNX operators:

```python
import torch
import torch.nn as nn

class SimpleClassifier(nn.Module):
    """Simple MLP for ONNX export demonstration."""
    
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Create and prepare model
model = SimpleClassifier()
model.eval()  # Critical: set to evaluation mode

# Create example input (defines input shape)
batch_size = 1
example_input = torch.randn(batch_size, 784)

# Export to ONNX
torch.onnx.export(
    model,                          # Model to export
    example_input,                  # Example input tensor
    "classifier.onnx",              # Output file path
    export_params=True,             # Include trained weights
    opset_version=17,               # ONNX opset version
    do_constant_folding=True,       # Optimize constants
    input_names=['input'],          # Input tensor names
    output_names=['output'],        # Output tensor names
)

print("Model exported to classifier.onnx")
```

### Dynamic Batch Sizes

For variable batch sizes, specify dynamic axes:

```python
torch.onnx.export(
    model,
    example_input,
    "classifier_dynamic.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},      # Variable batch dimension
        'output': {0: 'batch_size'}
    }
)
```

This allows inference with any batch size at runtime.

### Exporting CNNs

```python
class SimpleCNN(nn.Module):
    """Simple CNN for image classification."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(torch.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleCNN()
model.eval()

# Image input: (batch, channels, height, width)
example_input = torch.randn(1, 1, 28, 28)

torch.onnx.export(
    model,
    example_input,
    "cnn_classifier.onnx",
    opset_version=17,
    input_names=['image'],
    output_names=['logits'],
    dynamic_axes={
        'image': {0: 'batch_size'},
        'logits': {0: 'batch_size'}
    }
)
```

### Sequence Models

For models with variable sequence lengths (e.g., transformers, RNNs):

```python
class SequenceModel(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out[:, -1, :])
        return output

model = SequenceModel()
model.eval()

# Example with batch_size=2, seq_len=32
dummy_input = torch.randint(0, 10000, (2, 32))

torch.onnx.export(
    model,
    dummy_input,
    "sequence_model.onnx",
    opset_version=17,
    input_names=['input_ids'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'logits': {0: 'batch_size'}
    }
)
```

### Multiple Inputs and Outputs

```python
class MultiIOModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(784, 256)
        self.classifier = nn.Linear(256, 10)
        self.regressor = nn.Linear(256, 1)
    
    def forward(self, x, auxiliary):
        features = torch.relu(self.encoder(x))
        features = features + auxiliary  # Use auxiliary input
        
        class_logits = self.classifier(features)
        regression = self.regressor(features)
        
        return class_logits, regression

model = MultiIOModel()
model.eval()

# Multiple inputs
x = torch.randn(1, 784)
aux = torch.randn(1, 256)

torch.onnx.export(
    model,
    (x, aux),  # Tuple of inputs
    "multi_io.onnx",
    opset_version=17,
    input_names=['main_input', 'auxiliary'],
    output_names=['classification', 'regression'],
    dynamic_axes={
        'main_input': {0: 'batch'},
        'auxiliary': {0: 'batch'},
        'classification': {0: 'batch'},
        'regression': {0: 'batch'}
    }
)
```

### Export Parameters

Key parameters for `torch.onnx.export()`:

| Parameter | Description | Recommendation |
|-----------|-------------|----------------|
| `opset_version` | ONNX operator set version | Use 14+ for modern ops, 17 recommended |
| `do_constant_folding` | Optimize constants at export | Always True |
| `export_params` | Include weights in ONNX file | True for deployment |
| `keep_initializers_as_inputs` | Treat weights as inputs | False (default) |
| `verbose` | Print export details | True for debugging |
| `dynamic_axes` | Axes with variable dimensions | Set for batch dimension |

## Verifying ONNX Models

### Structure Validation

Check that the ONNX model is well-formed:

```python
import onnx

def verify_onnx_model(onnx_path):
    """Verify ONNX model validity."""
    try:
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print("✓ ONNX model is valid")
        return True
    except Exception as e:
        print(f"✗ ONNX model validation failed: {e}")
        return False

verify_onnx_model("classifier.onnx")
```

### Model Inspection

Examine model structure, inputs, and outputs:

```python
def print_onnx_model_info(onnx_path):
    """Print detailed ONNX model information."""
    model = onnx.load(onnx_path)
    
    print("\n=== Model Information ===")
    print(f"IR Version: {model.ir_version}")
    print(f"Producer: {model.producer_name}")
    print(f"Opset Version: {model.opset_import[0].version}")
    
    print("\n=== Inputs ===")
    for input_tensor in model.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param 
                 for d in input_tensor.type.tensor_type.shape.dim]
        print(f"  Name: {input_tensor.name}")
        print(f"  Shape: {shape}")
        print(f"  Type: {input_tensor.type.tensor_type.elem_type}")
    
    print("\n=== Outputs ===")
    for output_tensor in model.graph.output:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param 
                 for d in output_tensor.type.tensor_type.shape.dim]
        print(f"  Name: {output_tensor.name}")
        print(f"  Shape: {shape}")

print_onnx_model_info("classifier.onnx")
```

### Numerical Verification

Verify numerical equivalence between PyTorch and ONNX:

```python
import numpy as np
import onnxruntime as ort

def compare_pytorch_onnx_outputs(pytorch_model, onnx_path, test_input, rtol=1e-3, atol=1e-5):
    """Compare outputs between PyTorch and ONNX models."""
    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input).numpy()
    
    # ONNX inference
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    onnx_output = session.run(None, {input_name: test_input.numpy()})[0]
    
    # Compare
    max_diff = np.max(np.abs(pytorch_output - onnx_output))
    match = np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol)
    
    print(f"\n=== Output Comparison ===")
    print(f"Max difference: {max_diff:.2e}")
    print(f"Outputs match (rtol={rtol}, atol={atol}): {'✓' if match else '✗'}")
    
    return match, max_diff

# Test with multiple random inputs
def comprehensive_verification(pytorch_model, onnx_path, input_shape, num_tests=10):
    """Run comprehensive numerical verification."""
    onnx_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = onnx_session.get_inputs()[0].name
    
    np.random.seed(42)
    max_diff = 0.0
    
    print("Comparing PyTorch vs ONNX outputs:")
    for i in range(num_tests):
        np_input = np.random.randn(*input_shape).astype(np.float32)
        
        # PyTorch inference
        torch_input = torch.from_numpy(np_input)
        with torch.no_grad():
            pytorch_output = pytorch_model(torch_input).numpy()
        
        # ONNX inference
        onnx_output = onnx_session.run(None, {input_name: np_input})[0]
        
        diff = np.abs(pytorch_output - onnx_output).max()
        max_diff = max(max_diff, diff)
        print(f"  Test {i+1}: max difference = {diff:.2e}")
    
    print(f"\nMaximum difference: {max_diff:.2e}")
    
    if max_diff < 1e-5:
        print("✓ Outputs match within acceptable tolerance")
    else:
        print("⚠ Warning: Significant numerical differences detected")
    
    return max_diff

# Usage
test_input = torch.randn(5, 784)
match, diff = compare_pytorch_onnx_outputs(model, "classifier.onnx", test_input)
```

## ONNX Runtime Inference

### Basic Inference

Run inference using ONNX Runtime:

```python
import onnxruntime as ort
import numpy as np

class ONNXInferenceSession:
    """Wrapper for ONNX Runtime inference."""
    
    def __init__(self, onnx_path, providers=None):
        """
        Initialize ONNX Runtime session.
        
        Args:
            onnx_path: Path to ONNX model
            providers: Execution providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        """
        if providers is None:
            providers = ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        print(f"Loaded model with providers: {self.session.get_providers()}")
    
    def predict(self, input_data):
        """Run inference."""
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.numpy()
        
        outputs = self.session.run(self.output_names, {self.input_name: input_data})
        return outputs[0] if len(outputs) == 1 else outputs
    
    def get_metadata(self):
        """Get model metadata."""
        return {
            'inputs': [(i.name, i.shape, i.type) for i in self.session.get_inputs()],
            'outputs': [(o.name, o.shape, o.type) for o in self.session.get_outputs()],
            'providers': self.session.get_providers()
        }

# Usage
session = ONNXInferenceSession("classifier.onnx")
input_data = np.random.randn(5, 784).astype(np.float32)
output = session.predict(input_data)
print(f"Output shape: {output.shape}")
```

### Batch Inference

```python
def batch_inference(session, data, batch_size=32):
    """
    Perform batched inference with ONNX Runtime.
    
    Args:
        session: ONNX Runtime inference session
        data: Input data array
        batch_size: Number of samples per batch
        
    Returns:
        Concatenated predictions
    """
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    all_outputs = []
    num_samples = len(data)
    
    for i in range(0, num_samples, batch_size):
        batch = data[i:i+batch_size].astype(np.float32)
        outputs = session.run([output_name], {input_name: batch})
        all_outputs.append(outputs[0])
    
    return np.concatenate(all_outputs, axis=0)

# Example usage
data = np.random.randn(1000, 784)
session = ort.InferenceSession("classifier.onnx", providers=['CPUExecutionProvider'])
predictions = batch_inference(session, data, batch_size=64)
print(f"Processed {len(predictions)} samples")
```

### GPU Inference

Enable CUDA execution for GPU inference:

```python
# Check available providers
print(ort.get_available_providers())
# ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']

# GPU inference with fallback to CPU
session = ONNXInferenceSession(
    "classifier.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

# Verify GPU is being used
assert 'CUDAExecutionProvider' in session.session.get_providers()
```

### Optimized Session Configuration

```python
def create_optimized_session(onnx_path, use_gpu=False):
    """Create optimized ONNX Runtime session."""
    sess_options = ort.SessionOptions()
    
    # Enable all optimizations
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Enable parallel execution
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 4
    
    # Enable memory pattern optimization
    sess_options.enable_mem_pattern = True
    
    # Execution providers
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
    
    session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
    return session
```

## ONNX Model Optimization

### Graph Optimizations

ONNX provides graph-level optimizations:

```python
from onnx import optimizer
import os

def optimize_onnx_model(input_path, output_path):
    """Apply ONNX graph optimizations."""
    model = onnx.load(input_path)
    
    # Available optimization passes
    passes = [
        'eliminate_identity',
        'eliminate_nop_transpose',
        'eliminate_nop_pad',
        'eliminate_unused_initializer',
        'fuse_consecutive_transposes',
        'fuse_transpose_into_gemm',
        'fuse_bn_into_conv',
        'fuse_add_bias_into_conv'
    ]
    
    optimized_model = optimizer.optimize(model, passes)
    onnx.save(optimized_model, output_path)
    
    # Report size reduction
    original_size = os.path.getsize(input_path) / (1024 * 1024)
    optimized_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"Original size: {original_size:.2f} MB")
    print(f"Optimized size: {optimized_size:.2f} MB")
    print(f"Reduction: {(1 - optimized_size/original_size)*100:.1f}%")
```

### ONNX Simplifier

The `onnx-simplifier` tool removes redundant operations:

```bash
pip install onnx-simplifier
python -m onnxsim model.onnx model_simplified.onnx
```

Or programmatically:

```python
import onnxsim

def simplify_onnx_model(input_path, output_path):
    """Simplify ONNX model by removing redundant operations."""
    model = onnx.load(input_path)
    simplified_model, check = onnxsim.simplify(model)
    
    if check:
        onnx.save(simplified_model, output_path)
        print(f"✓ Model simplified and saved to {output_path}")
    else:
        print("✗ Simplification failed verification")
    
    return check
```

### Transformer Model Optimization

```python
from onnxruntime.transformers import optimizer

# Optimize for specific model architectures
optimized_model = optimizer.optimize_model(
    "model.onnx",
    model_type='bert',  # or 'gpt2', 'vit', etc.
    num_heads=12,
    hidden_size=768
)

optimized_model.save_model_to_file("model_optimized.onnx")
```

### ONNX Quantization

Quantize ONNX models for faster inference:

```python
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType

def quantize_onnx_dynamic(input_path, output_path):
    """Apply dynamic quantization to ONNX model."""
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8
    )
    
    # Compare sizes
    original_size = os.path.getsize(input_path) / (1024 * 1024)
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"Original: {original_size:.2f} MB")
    print(f"Quantized: {quantized_size:.2f} MB")
    print(f"Compression: {original_size/quantized_size:.2f}x")

quantize_onnx_dynamic("classifier.onnx", "classifier_quantized.onnx")
```

For static quantization (requires calibration data):

```python
from onnxruntime.quantization import CalibrationDataReader

class DataReader(CalibrationDataReader):
    """Calibration data reader for static quantization."""
    
    def __init__(self, calibration_data, input_name):
        self.data = calibration_data
        self.input_name = input_name
        self.index = 0
    
    def get_next(self):
        if self.index >= len(self.data):
            return None
        
        batch = self.data[self.index]
        self.index += 1
        return {self.input_name: batch}
    
    def rewind(self):
        self.index = 0

def quantize_onnx_static(input_path, output_path, calibration_data):
    """Apply static quantization with calibration."""
    data_reader = DataReader(calibration_data, 'input')
    
    quantize_static(
        model_input=input_path,
        model_output=output_path,
        calibration_data_reader=data_reader
    )
```

## Common Export Issues

### Issue 1: Dynamic Control Flow

```python
# ❌ Problematic: data-dependent branching
class DynamicModel(nn.Module):
    def forward(self, x):
        if x.sum() > 0:  # Depends on input values
            return self.branch_a(x)
        else:
            return self.branch_b(x)

# ✓ Solution 1: Use torch.where for element-wise branching
class StaticModel(nn.Module):
    def forward(self, x):
        out_a = self.branch_a(x)
        out_b = self.branch_b(x)
        condition = (x.sum(dim=1, keepdim=True) > 0).float()
        return condition * out_a + (1 - condition) * out_b

# ✓ Solution 2: Use torch.jit.script before export
scripted_model = torch.jit.script(DynamicModel())
torch.onnx.export(scripted_model, dummy_input, "dynamic_model.onnx", opset_version=17)
```

### Issue 2: Unsupported Operations

```python
# ❌ Some operations lack ONNX support
class ModelWithCustomOp(nn.Module):
    def forward(self, x):
        # torch.unique not supported in older opsets
        return torch.unique(x)

# ✓ Solution 1: Replace with supported operations
class ModelFixed(nn.Module):
    def forward(self, x):
        sorted_x, _ = torch.sort(x.flatten())
        return sorted_x

# ✓ Solution 2: Register custom symbolic function
@torch.onnx.symbolic_helper.parse_args('v')
def symbolic_custom_op(g, input):
    return g.op("CustomOp", input)

torch.onnx.register_custom_op_symbolic('aten::my_custom_op', symbolic_custom_op, 17)

# ✓ Solution 3: Upgrade opset version (if operation added in newer version)
```

### Issue 3: In-place Operations

```python
# ❌ In-place operations can cause issues
class InPlaceOps(nn.Module):
    def forward(self, x):
        x += 1  # In-place addition
        x.relu_()  # In-place ReLU
        return x

# ✓ Use out-of-place operations
class OutOfPlaceOps(nn.Module):
    def forward(self, x):
        x = x + 1
        x = torch.relu(x)
        return x
```

### Issue 4: Shape Inference Failures

```python
# ❌ Problematic shape operations
class ShapeIssues(nn.Module):
    def forward(self, x):
        mid = x.shape[1] // 2
        return x[:, :mid]

# ✓ Use tensor operations for shapes
class FixedShapes(nn.Module):
    def forward(self, x):
        mid = x.size(1) // 2
        return torch.narrow(x, 1, 0, mid)

# Use verbose mode to debug shape issues
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    verbose=True,  # Print detailed export log
    opset_version=17
)
```

## Performance Benchmarking

```python
import time
import numpy as np

def benchmark_inference(session, input_shape, num_iterations=1000, warmup=100):
    """
    Benchmark ONNX model inference performance.
    
    Args:
        session: ONNX Runtime session
        input_shape: Shape of input tensor
        num_iterations: Number of inference iterations
        warmup: Number of warmup iterations
        
    Returns:
        Dictionary with latency statistics
    """
    input_name = session.get_inputs()[0].name
    
    # Generate random input
    input_data = np.random.randn(*input_shape).astype(np.float32)
    
    # Warmup
    for _ in range(warmup):
        session.run(None, {input_name: input_data})
    
    # Benchmark
    latencies = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        session.run(None, {input_name: input_data})
        latencies.append((time.perf_counter() - start) * 1000)  # ms
    
    latencies = np.array(latencies)
    
    return {
        'mean_ms': np.mean(latencies),
        'std_ms': np.std(latencies),
        'min_ms': np.min(latencies),
        'max_ms': np.max(latencies),
        'p50_ms': np.percentile(latencies, 50),
        'p95_ms': np.percentile(latencies, 95),
        'p99_ms': np.percentile(latencies, 99),
        'throughput': 1000 / np.mean(latencies)  # samples/second
    }

# Run benchmark
session = ort.InferenceSession("classifier.onnx", providers=['CPUExecutionProvider'])
results = benchmark_inference(session, (1, 784))

print(f"Mean latency: {results['mean_ms']:.3f} ms")
print(f"P95 latency: {results['p95_ms']:.3f} ms")
print(f"P99 latency: {results['p99_ms']:.3f} ms")
print(f"Throughput: {results['throughput']:.1f} samples/sec")
```

## Best Practices

### Export Checklist

1. **Set model to eval mode**: `model.eval()`
2. **Use representative input shapes**: Match production data
3. **Enable dynamic axes**: For variable batch sizes
4. **Verify numerically**: Compare PyTorch vs ONNX outputs
5. **Test edge cases**: Empty batches, extreme values
6. **Benchmark performance**: Measure latency and throughput

### Production Recommendations

1. **Version control ONNX models**: Track model files with metadata
2. **Document input/output specifications**: Shape, dtype, preprocessing
3. **Include validation data**: For CI/CD numerical verification
4. **Monitor inference metrics**: Latency, errors, drift
5. **Plan for model updates**: Versioning and rollback strategy

### Common Pitfalls to Avoid

- **Forgetting `model.eval()`**: Causes BatchNorm and Dropout to behave incorrectly
- **Hardcoded shapes**: Use dynamic axes for production flexibility
- **Skipping verification**: Always compare outputs numerically
- **Ignoring opset version**: Higher versions support more operations but require newer runtimes

## Complete Workflow Example

```python
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import os

def complete_onnx_workflow(model, input_shape, output_dir="./deployment"):
    """Complete ONNX export and validation workflow."""
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    dummy_input = torch.randn(1, *input_shape)
    
    # Step 1: Export to ONNX
    print("1. Exporting to ONNX...")
    onnx_path = f"{output_dir}/model.onnx"
    torch.onnx.export(
        model, dummy_input, onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"   ✓ Exported to {onnx_path}")
    
    # Step 2: Verify model structure
    print("2. Verifying ONNX model...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("   ✓ Model structure is valid")
    
    # Step 3: Numerical verification
    print("3. Comparing outputs...")
    test_input = torch.randn(5, *input_shape)
    
    with torch.no_grad():
        pytorch_output = model(test_input).numpy()
    
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    onnx_output = session.run(None, {'input': test_input.numpy()})[0]
    
    max_diff = np.max(np.abs(pytorch_output - onnx_output))
    print(f"   Max difference: {max_diff:.2e}")
    
    if max_diff < 1e-5:
        print("   ✓ Numerical verification passed")
    else:
        print("   ⚠ Warning: Numerical differences detected")
    
    # Step 4: Benchmark
    print("4. Benchmarking...")
    import time
    
    num_runs = 100
    # Warmup
    for _ in range(10):
        session.run(None, {'input': test_input.numpy()})
    
    start = time.time()
    for _ in range(num_runs):
        session.run(None, {'input': test_input.numpy()})
    elapsed = time.time() - start
    
    print(f"   Latency: {(elapsed/num_runs)*1000:.2f} ms")
    print(f"   Throughput: {(5 * num_runs)/elapsed:.1f} samples/sec")
    
    # Step 5: Report file size
    file_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"5. Model size: {file_size:.2f} MB")
    
    print(f"\n✓ Workflow complete. Model saved to {output_dir}")
    
    return onnx_path

# Run workflow
model = SimpleClassifier()
complete_onnx_workflow(model, input_shape=(784,))
```

## Summary

ONNX provides a standardized format for deploying PyTorch models across diverse platforms. Key takeaways:

- **Export carefully**: Set eval mode, verify numerical accuracy, handle edge cases
- **Optimize for target**: Use ONNX Runtime optimizations for your hardware
- **Test thoroughly**: Benchmark latency, throughput, and accuracy
- **Document everything**: Input/output specs, preprocessing, versions

## References

1. ONNX Documentation: https://onnx.ai/onnx/
2. ONNX Runtime: https://onnxruntime.ai/
3. PyTorch ONNX Export: https://pytorch.org/docs/stable/onnx.html
4. ONNX Simplifier: https://github.com/daquexian/onnx-simplifier
5. ONNX Runtime Performance Tuning: https://onnxruntime.ai/docs/performance/
