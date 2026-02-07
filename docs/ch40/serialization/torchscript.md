# TorchScript

## Introduction

TorchScript is PyTorch's mechanism for creating serializable and optimizable models that can run independently from Python. It bridges the gap between research and production by enabling deployment to C++, mobile, and embedded environments.

## Why TorchScript?

Standard PyTorch models are Python objects that rely on the Python interpreter. TorchScript addresses key production requirements:

**Python-Free Execution** — Run models in C++ environments without a Python interpreter, reducing overhead and enabling deployment in resource-constrained settings.

**Graph Optimizations** — The TorchScript compiler applies optimizations like operator fusion, constant folding, and dead code elimination.

**Serialization** — Models can be saved and loaded as single files containing both architecture and weights.

**Production Stability** — Eliminates Python's Global Interpreter Lock (GIL) for better multithreading performance.

**Portability** — Deploy to C++, mobile (iOS/Android), and embedded systems.

## TorchScript vs Regular PyTorch

| Aspect | Regular PyTorch | TorchScript |
|--------|-----------------|-------------|
| Runtime | Python required | Python-free |
| Execution | Eager (line-by-line) | JIT compiled |
| Flexibility | Full Python | Subset of Python |
| Serialization | Weights only | Code + weights |
| Optimization | Limited | Graph-level |
| Debugging | Easy | More complex |

## Two Conversion Methods

TorchScript offers two approaches to convert PyTorch models:

### Method 1: Tracing

**How it works**: Records operations performed on example inputs by running the model once.

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Model +   │────▶│    Run      │────▶│   Static    │
│Example Input│     │  Forward    │     │   Graph     │
└─────────────┘     └─────────────┘     └─────────────┘
```

**Best for**: Simple feedforward models without data-dependent control flow.

### Method 2: Scripting

**How it works**: Compiles Python source code directly to TorchScript IR.

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Python    │────▶│   Compile   │────▶│ TorchScript │
│   Source    │     │   to IR     │     │    Code     │
└─────────────┘     └─────────────┘     └─────────────┘
```

**Best for**: Models with control flow (if/else, loops), RNNs, transformers.

## Tracing in Detail

### Basic Tracing

Tracing records operations as the model executes with example inputs:

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    """Simple feedforward model suitable for tracing."""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Create model
model = SimpleModel()
model.eval()  # Critical: set to evaluation mode

# Create example input
example_input = torch.randn(1, 784)

# Trace the model
traced_model = torch.jit.trace(model, example_input)

# Save traced model
traced_model.save("model_traced.pt")

# Verify output matches
with torch.no_grad():
    original_output = model(example_input)
    traced_output = traced_model(example_input)
    
diff = torch.abs(original_output - traced_output).max()
print(f"Max difference: {diff:.2e}")  # Should be ~0
```

### Inspecting Traced Graph

```python
# View the traced graph
print("Traced Graph:")
print(traced_model.graph)

# View optimized graph
print("\nOptimized Graph:")
print(traced_model.inlined_graph)

# View code representation
print("\nCode:")
print(traced_model.code)
```

### Tracing CNNs

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleCNN()
model.eval()

# Trace with image input
example_image = torch.randn(1, 1, 28, 28)
traced_cnn = torch.jit.trace(model, example_image)
traced_cnn.save("cnn_traced.pt")
```

### Tracing Limitations

Tracing records **only one execution path**:

```python
class ConditionalModel(nn.Module):
    """Model with data-dependent control flow - tracing will fail to capture."""
    
    def __init__(self):
        super().__init__()
        self.fc_positive = nn.Linear(10, 5)
        self.fc_negative = nn.Linear(10, 5)
    
    def forward(self, x):
        # ⚠ This condition depends on input values
        if x.sum() > 0:
            return self.fc_positive(x)
        else:
            return self.fc_negative(x)

model = ConditionalModel()
model.eval()

# Trace with positive-sum input
positive_input = torch.ones(1, 10)
traced = torch.jit.trace(model, positive_input)

# Warning: traced model ALWAYS takes positive path
# regardless of actual input values
negative_input = -torch.ones(1, 10)
traced_output = traced(negative_input)  # Uses fc_positive, NOT fc_negative!
```

**Key Point**: When control flow depends on tensor values, use scripting instead.

## Scripting in Detail

### Basic Scripting

Scripting analyzes Python code and converts it to TorchScript IR:

```python
class ModelWithControlFlow(nn.Module):
    """Model with control flow that requires scripting."""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor, use_dropout: bool = False) -> torch.Tensor:
        """
        Forward pass with optional dropout.
        
        Type annotations are required for scripting.
        """
        x = torch.relu(self.fc1(x))
        
        # Control flow based on boolean parameter (scriptable)
        if use_dropout:
            x = self.dropout(x)
        
        return self.fc2(x)

# Script the model
model = ModelWithControlFlow()
model.eval()

scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")

# Both paths work correctly
input_tensor = torch.randn(1, 784)
output_with_dropout = scripted_model(input_tensor, True)
output_without_dropout = scripted_model(input_tensor, False)
```

### Scripting with Loops

```python
class RNNModel(nn.Module):
    """RNN with explicit loop - requires scripting."""
    
    def __init__(self, input_size=10, hidden_size=128, output_size=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = nn.RNNCell(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Process variable-length sequence.
        
        Args:
            sequence: Shape (batch_size, seq_len, input_size)
        
        Returns:
            Output from final hidden state
        """
        batch_size = sequence.size(0)
        seq_len = sequence.size(1)
        
        # Initialize hidden state
        hidden = torch.zeros(batch_size, self.hidden_size, device=sequence.device)
        
        # Loop through sequence (scripting handles this correctly)
        for t in range(seq_len):
            hidden = self.rnn_cell(sequence[:, t, :], hidden)
        
        return self.output_layer(hidden)

model = RNNModel()
model.eval()

scripted_rnn = torch.jit.script(model)

# Works with different sequence lengths
seq_short = torch.randn(2, 5, 10)
seq_long = torch.randn(2, 20, 10)

output_short = scripted_rnn(seq_short)
output_long = scripted_rnn(seq_long)
print(f"Short sequence output: {output_short.shape}")
print(f"Long sequence output: {output_long.shape}")
```

### Type Annotations

TorchScript requires explicit type annotations:

```python
from typing import Tuple, List, Optional, Dict

class AnnotatedModel(nn.Module):
    """Model with comprehensive type annotations."""
    
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    
    # Return single tensor
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
    
    # Return tuple
    @torch.jit.export
    def forward_with_hidden(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.fc(x)
        hidden = torch.relu(output)
        return output, hidden
    
    # Optional parameter
    @torch.jit.export
    def forward_optional(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = self.fc(x)
        if mask is not None:
            output = output * mask
        return output
```

## Hybrid Approach

Combine tracing and scripting for complex models:

```python
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.classifier = nn.Linear(128, 10)
    
    @torch.jit.export
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Separate method for encoding (can be traced)."""
        return self.encoder(x)
    
    def forward(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Forward with control flow (must be scripted)."""
        features = self.encode(x)
        logits = self.classifier(features)
        
        # Data-dependent control flow
        if logits.max() > threshold:
            return torch.softmax(logits, dim=-1)
        else:
            return torch.sigmoid(logits)

# Script the entire model
model = HybridModel()
scripted = torch.jit.script(model)
```

Or use `torch.jit.trace` on submodules and `torch.jit.script` on the main module:

```python
class MainModel(nn.Module):
    def __init__(self):
        super().__init__()
        encoder = Encoder()
        # Trace encoder
        self.encoder = torch.jit.trace(encoder, torch.randn(1, 3, 224, 224))
        self.processor = Processor()
    
    @torch.jit.export
    def forward(self, x):
        features = self.encoder(x)
        return self.processor(features)

scripted_model = torch.jit.script(MainModel())
```

## Optimization for Inference

### Basic Optimization

Apply inference-time optimizations:

```python
def optimize_for_inference(model, example_input):
    """Apply TorchScript optimizations for inference."""
    model.eval()
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    
    # Freeze the model (inline parameters, remove training ops)
    frozen_model = torch.jit.freeze(traced_model)
    
    # Optimize for inference
    optimized_model = torch.jit.optimize_for_inference(frozen_model)
    
    return optimized_model

model = SimpleModel()
example_input = torch.randn(1, 784)
optimized = optimize_for_inference(model, example_input)
```

### Freezing

Freezing inlines constant attributes and removes training-related code:

```python
model.eval()
traced_model = torch.jit.trace(model, example_input)
frozen_model = torch.jit.freeze(traced_model)

# Frozen model has constants inlined
print(frozen_model.graph)
```

Benefits of freezing:

- Parameters become constants in the graph
- Removes gradient computation paths
- Enables additional optimizations

### Fusion

TorchScript automatically fuses compatible operations:

```python
# Before fusion: separate ops
# relu(bn(conv(x))) = 3 kernel launches

# After fusion: single fused kernel
# fused_conv_bn_relu(x) = 1 kernel launch
```

Check which fusions are applied:

```python
optimized = torch.jit.optimize_for_inference(traced_model)
print(optimized.graph_for(example_input))
```

## Performance Benchmarking

Compare original vs TorchScript performance:

```python
import time

def benchmark(model, input_tensor, num_iterations=1000, warmup=100):
    """Benchmark model inference."""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    
    # Synchronize if using GPU
    if input_tensor.is_cuda:
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(input_tensor)
    
    if input_tensor.is_cuda:
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    
    return {
        'total_time_s': elapsed,
        'time_per_iteration_ms': (elapsed / num_iterations) * 1000,
        'throughput': num_iterations / elapsed
    }

# Compare PyTorch vs TorchScript
model_pytorch = SimpleModel()
model_scripted = torch.jit.script(SimpleModel())
model_optimized = optimize_for_inference(SimpleModel(), torch.randn(1, 784))

input_tensor = torch.randn(32, 784)

pytorch_results = benchmark(model_pytorch, input_tensor)
scripted_results = benchmark(model_scripted, input_tensor)
optimized_results = benchmark(model_optimized, input_tensor)

print(f"PyTorch:   {pytorch_results['time_per_iteration_ms']:.3f} ms/iter")
print(f"Scripted:  {scripted_results['time_per_iteration_ms']:.3f} ms/iter")
print(f"Optimized: {optimized_results['time_per_iteration_ms']:.3f} ms/iter")
print(f"Speedup (optimized): {pytorch_results['time_per_iteration_ms'] / optimized_results['time_per_iteration_ms']:.2f}x")
```

Typical speedup ranges:

| Model Type | CPU Speedup | GPU Speedup |
|------------|-------------|-------------|
| MLP | 1.5-2× | 1.1-1.5× |
| CNN | 1.2-1.8× | 1.1-1.3× |
| Transformer | 1.3-2× | 1.1-1.4× |

## Saving and Loading

### Save TorchScript Model

```python
# Save traced/scripted model
traced_model.save("model.pt")

# Or use torch.jit.save
torch.jit.save(traced_model, "model.pt")
```

### Load TorchScript Model

```python
# Load without Python class definition
loaded_model = torch.jit.load("model.pt")
loaded_model.eval()

# Optionally specify device
loaded_model = torch.jit.load("model.pt", map_location='cuda:0')

# Run inference
with torch.no_grad():
    output = loaded_model(input_tensor)
```

### Device Mapping

```python
# Load GPU model to CPU
model_cpu = torch.jit.load("model_gpu.pt", map_location='cpu')

# Load CPU model to GPU
model_gpu = torch.jit.load("model_cpu.pt", map_location='cuda:0')

# Load to available device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.jit.load("model.pt", map_location=device)
```

### Model Metadata

Add metadata to saved models:

```python
# Save with extra files
extra_files = {
    'config.json': '{"input_size": [784], "num_classes": 10}',
    'version.txt': '1.0.0'
}
torch.jit.save(traced_model, "model.pt", _extra_files=extra_files)

# Load and read metadata
extra_files = {'config.json': '', 'version.txt': ''}
loaded_model = torch.jit.load("model.pt", _extra_files=extra_files)
print(extra_files['config.json'])
```

## C++ Deployment

TorchScript models can run in pure C++ using LibTorch:

```cpp
#include <torch/script.h>
#include <iostream>

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: inference <path_to_model.pt>\n";
        return -1;
    }

    // Load model
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading model\n";
        return -1;
    }

    // Create input tensor
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::randn({1, 784}));

    // Run inference
    at::Tensor output = module.forward(inputs).toTensor();
    
    std::cout << "Output shape: " << output.sizes() << std::endl;
    
    return 0;
}
```

Build with CMake:

```cmake
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(inference)

find_package(Torch REQUIRED)

add_executable(inference main.cpp)
target_link_libraries(inference "${TORCH_LIBRARIES}")
set_property(TARGET inference PROPERTY CXX_STANDARD 14)
```

## Common Issues and Solutions

### Issue 1: Type Annotation Requirements

TorchScript requires type annotations for scripted functions:

```python
# ❌ Error: Missing type annotation
class BadModel(nn.Module):
    def forward(self, x, scale):  # scale type unclear
        return x * scale

# ✓ Fixed: Add type annotations
class GoodModel(nn.Module):
    def forward(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        return x * scale
```

### Issue 2: Unsupported Python Features

Some Python features don't translate to TorchScript:

```python
# ❌ Unsupported: arbitrary Python objects
class BadModel(nn.Module):
    def forward(self, x):
        return {"output": x}  # dict with string keys problematic

# ✓ Solution: Use NamedTuple or multiple returns
from typing import NamedTuple

class ModelOutput(NamedTuple):
    output: torch.Tensor
    confidence: torch.Tensor

class GoodModel(nn.Module):
    def forward(self, x) -> ModelOutput:
        return ModelOutput(output=x, confidence=torch.ones(1))
```

### Issue 3: Tensor Shape Operations

```python
# ❌ Problematic: Python int operations on shapes
def forward(self, x):
    half = x.shape[1] // 2  # Python division
    return x[:, :half]

# ✓ Fixed: Use tensor operations
def forward(self, x: torch.Tensor) -> torch.Tensor:
    half = x.size(1) // 2
    return torch.narrow(x, 1, 0, half)
```

### Issue 4: Default Mutable Arguments

```python
# ❌ Problematic: Mutable default
def forward(self, x, indices=[0, 1]):
    return x[:, indices]

# ✓ Fixed: Use None default with check
def forward(self, x: torch.Tensor, indices: Optional[List[int]] = None) -> torch.Tensor:
    if indices is None:
        indices = [0, 1]
    return x[:, indices]
```

### Issue 5: Module Containers

```python
# ❌ Problematic: Plain Python list
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(10, 10) for _ in range(3)]  # Not tracked
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# ✓ Fixed: Use ModuleList
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
```

### Debugging TorchScript

Inspect the generated graph:

```python
traced_model = torch.jit.trace(model, example_input)

# View high-level IR
print(traced_model.graph)

# View code
print(traced_model.code)

# Debug specific inputs
with torch.jit.optimized_execution(False):
    output = traced_model(input_tensor)
```

## Decision Guide: Tracing vs Scripting

```
┌─────────────────────────────────────────────────────────────┐
│                  DECISION FLOWCHART                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Does your model have data-dependent control flow?          │
│  (if/else based on tensor values, dynamic loops)            │
│                                                             │
│       NO                              YES                   │
│        │                               │                    │
│        ▼                               ▼                    │
│   ┌─────────┐                    ┌──────────┐              │
│   │ TRACING │                    │ SCRIPTING│              │
│   └─────────┘                    └──────────┘              │
│                                                             │
│  Simpler, works with any         Full Python subset,       │
│  model without modifications     requires type annotations │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Use Tracing When:**
- Simple feedforward models (CNNs, MLPs)
- Pre-trained models you don't want to modify
- Quick prototyping
- Model has no conditionals based on input values

**Use Scripting When:**
- Model has if/else based on tensor values
- Variable-length loops
- RNNs, transformers with dynamic behavior
- Need to guarantee all code paths work

## Best Practices

1. **Always use `model.eval()`** before tracing/scripting for inference.

2. **Freeze models** with `torch.jit.freeze()` to inline parameters and enable more optimizations.

3. **Use `optimize_for_inference()`** for production deployment.

4. **Test thoroughly** — verify outputs match between original and TorchScript models.

5. **Add type annotations** for scripted modules to avoid runtime errors.

6. **Avoid Python-specific constructs** like arbitrary dicts, sets with non-tensor values.

7. **Profile on target hardware** — speedups vary by architecture and device.

## Complete Workflow

```python
import torch
import torch.nn as nn
import time
import os

def torchscript_deployment_workflow(model, input_shape, output_path="model.pt"):
    """Complete TorchScript deployment workflow."""
    model.eval()
    example_input = torch.randn(1, *input_shape)
    
    print("=== TorchScript Deployment Workflow ===\n")
    
    # Step 1: Try tracing first (preferred for most models)
    print("1. Creating TorchScript model...")
    try:
        traced_model = torch.jit.trace(model, example_input)
        print("   ✓ Model traced successfully")
    except Exception as e:
        print(f"   Tracing failed, trying scripting: {e}")
        traced_model = torch.jit.script(model)
        print("   ✓ Model scripted successfully")
    
    # Step 2: Freeze model
    print("2. Freezing model...")
    frozen_model = torch.jit.freeze(traced_model)
    print("   ✓ Model frozen")
    
    # Step 3: Optimize for inference
    print("3. Optimizing for inference...")
    optimized_model = torch.jit.optimize_for_inference(frozen_model)
    print("   ✓ Optimizations applied")
    
    # Step 4: Verify outputs
    print("4. Verifying output consistency...")
    with torch.no_grad():
        original_output = model(example_input)
    optimized_output = optimized_model(example_input)
    
    max_diff = torch.max(torch.abs(original_output - optimized_output)).item()
    print(f"   Max difference: {max_diff:.2e}")
    
    # Step 5: Benchmark
    print("5. Benchmarking...")
    num_runs = 100
    
    with torch.no_grad():
        start = time.time()
        for _ in range(num_runs):
            _ = model(example_input)
        original_time = time.time() - start
    
    start = time.time()
    for _ in range(num_runs):
        _ = optimized_model(example_input)
    optimized_time = time.time() - start
    
    print(f"   Original: {original_time*1000/num_runs:.2f} ms/inference")
    print(f"   Optimized: {optimized_time*1000/num_runs:.2f} ms/inference")
    print(f"   Speedup: {original_time/optimized_time:.2f}x")
    
    # Step 6: Save
    print(f"6. Saving to {output_path}...")
    torch.jit.save(optimized_model, output_path)
    print(f"   ✓ Model saved")
    
    # Report file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"   File size: {size_mb:.2f} MB")
    
    print("\n✓ Workflow complete!")
    return optimized_model

# Usage
model = SimpleModel()
optimized = torchscript_deployment_workflow(model, input_shape=(784,))
```

## Summary

TorchScript enables production deployment of PyTorch models without Python:

1. **Tracing**: Records operations, best for simple models
2. **Scripting**: Compiles source, handles control flow
3. **Both methods**: Produce self-contained, optimizable models
4. **Type annotations**: Required for scripting, helpful for tracing
5. **Performance**: JIT compilation and graph optimizations

Key recommendations:

- Start with tracing for simple models
- Use scripting when tracing fails or warns
- Test thoroughly after conversion
- Benchmark performance improvements
- Use freezing and optimization for production

## References

1. PyTorch TorchScript Documentation: https://pytorch.org/docs/stable/jit.html
2. TorchScript Language Reference: https://pytorch.org/docs/stable/jit_language_reference.html
3. PyTorch C++ API: https://pytorch.org/cppdocs/
4. Production Deployment Guide: https://pytorch.org/tutorials/beginner/deploy_seq2seq_hybrid_frontend_tutorial.html
