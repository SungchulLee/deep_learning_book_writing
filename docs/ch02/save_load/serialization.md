# Model Serialization

## Overview

**Model serialization** encompasses the techniques for converting trained PyTorch models into formats suitable for storage, distribution, and deployment. Beyond basic save/load operations, production deployment often requires converting models to optimized formats like ONNX or TorchScript that enable cross-platform inference, hardware acceleration, and independence from Python runtime.

## Learning Objectives

By the end of this section, you will be able to:

- Understand PyTorch's serialization protocol and pickle-based persistence
- Export models to ONNX format for cross-framework compatibility
- Convert models to TorchScript for production deployment
- Choose the appropriate serialization format for different deployment targets
- Handle common serialization issues including versioning and compatibility

## Serialization Fundamentals

### PyTorch Serialization Protocol

PyTorch uses Python's `pickle` module extended with custom handlers for tensor storage:

```python
import torch
import pickle
import io

# Standard PyTorch save
torch.save(obj, filepath)

# Equivalent to:
# with open(filepath, 'wb') as f:
#     pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
```

The serialization process:

1. **Pickle protocol**: Serializes Python object graph
2. **Tensor storage**: Extracts tensor data into efficient binary format
3. **ZIP archive**: Combines pickled metadata with tensor storage files

### Inspecting Saved Files

```python
import zipfile
import torch

# Save a model
model = torch.nn.Linear(10, 5)
torch.save(model.state_dict(), 'model.pt')

# Inspect the saved file structure
with zipfile.ZipFile('model.pt', 'r') as z:
    print("Archive contents:")
    for name in z.namelist():
        info = z.getinfo(name)
        print(f"  {name}: {info.file_size} bytes")
```

**Output:**
```
Archive contents:
  data.pkl: 234 bytes
  byteorder: 6 bytes
  data/0: 200 bytes
  data/1: 20 bytes
  version: 2 bytes
```

### Serialization Options

```python
# Default serialization
torch.save(state_dict, 'model.pt')

# Use specific pickle protocol
torch.save(state_dict, 'model.pt', pickle_protocol=4)

# Save without ZIP compression (legacy format)
torch.save(state_dict, 'model.pt', _use_new_zipfile_serialization=False)

# Load with weights_only for security (recommended)
state_dict = torch.load('model.pt', weights_only=True)
```

!!! warning "Security Consideration"
    `torch.load()` uses pickle which can execute arbitrary code. Use `weights_only=True` when loading untrusted files, or verify file integrity before loading.

## ONNX Export

### What is ONNX?

**ONNX** (Open Neural Network Exchange) is an open standard for representing machine learning models. It provides:

- Cross-framework interoperability (PyTorch ↔ TensorFlow ↔ etc.)
- Hardware-optimized inference runtimes
- Standardized operator definitions
- Model visualization and analysis tools

### Basic ONNX Export

```python
import torch
import torch.nn as nn
import torch.onnx

class ConvClassifier(nn.Module):
    """Example CNN for ONNX export demonstration."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# Create and prepare model
model = ConvClassifier(num_classes=10)
model.eval()  # CRITICAL: Set to eval mode before export

# Create example input (batch_size, channels, height, width)
dummy_input = torch.randn(1, 1, 28, 28)

# Export to ONNX
torch.onnx.export(
    model,                              # Model to export
    dummy_input,                        # Example input tensor
    "classifier.onnx",                  # Output file path
    export_params=True,                 # Store trained weights
    opset_version=13,                   # ONNX operator set version
    do_constant_folding=True,           # Optimize constants
    input_names=['input'],              # Input tensor names
    output_names=['output'],            # Output tensor names
    dynamic_axes={                      # Dynamic dimensions
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("Model exported to classifier.onnx")
```

### ONNX Export Parameters

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `opset_version` | ONNX operator set version | 13+ for modern ops |
| `export_params` | Include trained weights | `True` |
| `do_constant_folding` | Fold constant expressions | `True` |
| `dynamic_axes` | Axes that can vary at runtime | Batch dimension typically |
| `input_names` | Names for input tensors | Descriptive names |
| `output_names` | Names for output tensors | Descriptive names |
| `verbose` | Print export details | `False` for production |

### ONNX Model Verification

```python
import onnx
import onnxruntime as ort
import numpy as np

def verify_onnx_model(
    onnx_path: str,
    pytorch_model: nn.Module,
    example_input: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-6
) -> bool:
    """
    Verify ONNX model produces identical outputs to PyTorch model.
    
    Args:
        onnx_path: Path to ONNX model file
        pytorch_model: Original PyTorch model
        example_input: Example input tensor
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
    
    Returns:
        True if outputs match within tolerance
    """
    # Load and validate ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")
    
    # Create ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_path)
    
    # Get input name
    input_name = ort_session.get_inputs()[0].name
    
    # Run PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(example_input).numpy()
    
    # Run ONNX Runtime inference
    ort_inputs = {input_name: example_input.numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]
    
    # Compare outputs
    max_diff = np.abs(pytorch_output - ort_output).max()
    outputs_match = np.allclose(pytorch_output, ort_output, rtol=rtol, atol=atol)
    
    print(f"Max difference: {max_diff:.2e}")
    print(f"Outputs match: {outputs_match}")
    
    return outputs_match


# Usage
model = ConvClassifier(num_classes=10)
model.eval()
example_input = torch.randn(1, 1, 28, 28)

verify_onnx_model("classifier.onnx", model, example_input)
```

### ONNX Runtime Inference

```python
import onnxruntime as ort
import numpy as np


class ONNXInferenceEngine:
    """
    Production-ready ONNX inference wrapper.
    
    Args:
        model_path: Path to ONNX model file
        providers: Execution providers in priority order
    """
    
    def __init__(
        self,
        model_path: str,
        providers: list = None
    ):
        # Default to CUDA if available, then CPU
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get input/output metadata
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"Loaded ONNX model: {model_path}")
        print(f"  Input: {self.input_name} {self.input_shape}")
        print(f"  Provider: {self.session.get_providers()[0]}")
    
    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on input data."""
        return self.session.run(
            [self.output_name],
            {self.input_name: input_data}
        )[0]
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Alias for __call__."""
        return self(input_data)
    
    def predict_batch(
        self,
        inputs: list,
        batch_size: int = 32
    ) -> np.ndarray:
        """Run inference on multiple inputs with batching."""
        results = []
        for i in range(0, len(inputs), batch_size):
            batch = np.stack(inputs[i:i + batch_size])
            results.append(self(batch))
        return np.concatenate(results, axis=0)


# Usage
engine = ONNXInferenceEngine("classifier.onnx")

# Single inference
input_data = np.random.randn(1, 1, 28, 28).astype(np.float32)
output = engine(input_data)
print(f"Output shape: {output.shape}")

# Batch inference
batch_inputs = [np.random.randn(1, 28, 28).astype(np.float32) for _ in range(100)]
outputs = engine.predict_batch(batch_inputs, batch_size=16)
print(f"Batch output shape: {outputs.shape}")
```

## TorchScript

### What is TorchScript?

**TorchScript** is PyTorch's intermediate representation (IR) that enables:

- Serialization independent of Python runtime
- Ahead-of-time compilation and optimization
- Deployment to C++ runtime (LibTorch)
- Mobile deployment (iOS/Android)

### Tracing vs Scripting

PyTorch provides two methods to create TorchScript models:

| Method | Mechanism | Best For |
|--------|-----------|----------|
| **Tracing** | Records operations during example execution | Models without control flow |
| **Scripting** | Compiles Python source code | Models with control flow |

### Method 1: Tracing

```python
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """Model without control flow - suitable for tracing."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Create model
model = SimpleModel(input_dim=784, hidden_dim=256, output_dim=10)
model.eval()

# Create example input
example_input = torch.randn(1, 784)

# Trace the model
traced_model = torch.jit.trace(model, example_input)

# Save traced model
traced_model.save("traced_model.pt")

# Load traced model (no Python model class needed!)
loaded_model = torch.jit.load("traced_model.pt")

# Verify outputs match
with torch.no_grad():
    original_output = model(example_input)
    traced_output = traced_model(example_input)
    loaded_output = loaded_model(example_input)

print(f"Original == Traced: {torch.allclose(original_output, traced_output)}")
print(f"Original == Loaded: {torch.allclose(original_output, loaded_output)}")
```

!!! warning "Tracing Limitations"
    Tracing records the exact operations executed with the example input. Control flow (if/else, loops with data-dependent bounds) will be "baked in" based on the trace, not preserved as dynamic logic.

### Method 2: Scripting

```python
class ModelWithControlFlow(nn.Module):
    """Model with control flow - requires scripting."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.threshold = 0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        
        # Control flow that depends on input
        if x.mean() > self.threshold:
            x = x * 2
        else:
            x = x + 1
        
        x = self.fc2(x)
        return x


# Create model
model = ModelWithControlFlow(input_dim=784, hidden_dim=256, output_dim=10)
model.eval()

# Script the model (compiles Python source)
scripted_model = torch.jit.script(model)

# Save scripted model
scripted_model.save("scripted_model.pt")

# Test with different inputs
input_high = torch.ones(1, 784)   # Mean > threshold
input_low = -torch.ones(1, 784)   # Mean < threshold

with torch.no_grad():
    output_high = scripted_model(input_high)
    output_low = scripted_model(input_low)

print("Control flow preserved - different code paths executed")
```

### Hybrid Approach: Trace with Script Submodules

For complex models, combine both methods:

```python
class HybridModel(nn.Module):
    """Complex model using both tracing and scripting."""
    
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784)
        )
    
    @torch.jit.export
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)
    
    @torch.jit.export
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor, mode: str = 'full') -> torch.Tensor:
        if mode == 'encode':
            return self.encode(x)
        elif mode == 'decode':
            return self.decode(x)
        else:
            z = self.encode(x)
            return self.decode(z)


# Script the model (handles control flow)
model = HybridModel()
model.eval()
scripted_model = torch.jit.script(model)

# Access exported methods
z = scripted_model.encode(torch.randn(1, 784))
x_reconstructed = scripted_model.decode(z)
```

### TorchScript Optimization

```python
def optimize_torchscript_model(model_path: str, output_path: str) -> None:
    """
    Apply production optimizations to TorchScript model.
    
    Args:
        model_path: Path to input TorchScript model
        output_path: Path for optimized model
    """
    # Load model
    model = torch.jit.load(model_path)
    
    # Freeze model (inlines parameters as constants)
    frozen_model = torch.jit.freeze(model)
    
    # Optimize for inference (fuses operations, etc.)
    optimized_model = torch.jit.optimize_for_inference(frozen_model)
    
    # Save optimized model
    optimized_model.save(output_path)
    
    print(f"Optimized model saved to {output_path}")
    print("Applied optimizations:")
    print("  - Constant propagation")
    print("  - Dead code elimination")
    print("  - Operator fusion")
    print("  - Memory layout optimization")


# Usage
optimize_torchscript_model("scripted_model.pt", "optimized_model.pt")
```

## Model Versioning and Compatibility

### Versioned Model Wrapper

```python
import hashlib
import json
from datetime import datetime
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class VersionedModelSaver:
    """
    Save models with comprehensive versioning and metadata.
    
    Args:
        base_dir: Directory for saving models
    """
    
    def __init__(self, base_dir: str = "models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_model_hash(self, model: nn.Module) -> str:
        """Compute SHA256 hash of model weights."""
        params = torch.cat([p.flatten() for p in model.parameters()])
        param_bytes = params.detach().cpu().numpy().tobytes()
        return hashlib.sha256(param_bytes).hexdigest()[:16]
    
    def save(
        self,
        model: nn.Module,
        version: str,
        config: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
        description: str = ""
    ) -> Path:
        """
        Save model with versioning metadata.
        
        Args:
            model: Model to save
            version: Semantic version string (e.g., "1.0.0")
            config: Model configuration/hyperparameters
            metrics: Training/evaluation metrics
            description: Human-readable description
        
        Returns:
            Path to saved model directory
        """
        # Create version directory
        version_dir = self.base_dir / f"v{version}"
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        weights_path = version_dir / "weights.pt"
        torch.save(model.state_dict(), weights_path)
        
        # Save TorchScript version
        model.eval()
        try:
            scripted = torch.jit.script(model)
            scripted.save(str(version_dir / "model.torchscript"))
            has_torchscript = True
        except Exception as e:
            print(f"Warning: Could not create TorchScript: {e}")
            has_torchscript = False
        
        # Create metadata
        metadata = {
            "version": version,
            "description": description,
            "config": config,
            "metrics": metrics or {},
            "model_hash": self.compute_model_hash(model),
            "pytorch_version": torch.__version__,
            "timestamp": datetime.now().isoformat(),
            "has_torchscript": has_torchscript,
            "files": {
                "weights": "weights.pt",
                "torchscript": "model.torchscript" if has_torchscript else None
            }
        }
        
        # Save metadata
        with open(version_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model v{version} saved to {version_dir}")
        return version_dir
    
    def load(
        self,
        model_class: type,
        version: str,
        device: torch.device = None,
        use_torchscript: bool = False,
        **model_kwargs
    ) -> tuple:
        """
        Load a versioned model.
        
        Args:
            model_class: Class to instantiate model (ignored if use_torchscript)
            version: Version string to load
            device: Device to load model to
            use_torchscript: Load TorchScript version if available
            **model_kwargs: Arguments for model class
        
        Returns:
            Tuple of (model, metadata)
        """
        version_dir = self.base_dir / f"v{version}"
        
        # Load metadata
        with open(version_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        device = device or torch.device("cpu")
        
        if use_torchscript and metadata["has_torchscript"]:
            # Load TorchScript model
            model = torch.jit.load(
                str(version_dir / "model.torchscript"),
                map_location=device
            )
        else:
            # Load standard model
            model = model_class(**model_kwargs)
            state_dict = torch.load(
                version_dir / "weights.pt",
                map_location=device
            )
            model.load_state_dict(state_dict)
            model.to(device)
        
        model.eval()
        
        # Verify hash
        if not use_torchscript:
            current_hash = self.compute_model_hash(model)
            if current_hash != metadata["model_hash"]:
                print(f"Warning: Model hash mismatch!")
        
        print(f"Loaded model v{version}")
        return model, metadata
    
    def list_versions(self) -> list:
        """List all available model versions."""
        versions = []
        for path in sorted(self.base_dir.glob("v*")):
            if path.is_dir():
                metadata_path = path / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    versions.append(metadata)
        return versions
```

## Format Comparison

### When to Use Each Format

| Format | Use Case | Advantages | Limitations |
|--------|----------|------------|-------------|
| **State Dict (.pt)** | Training, fine-tuning | Flexible, preserves gradients | Requires model class |
| **TorchScript** | Production deployment | No Python needed, optimized | Some ops unsupported |
| **ONNX** | Cross-platform | Framework agnostic | Limited dynamic features |

### Decision Flowchart

```
Need to deploy model?
├── Yes: Python available in production?
│   ├── Yes: State Dict or TorchScript
│   └── No: TorchScript (.pt) or ONNX
│       ├── PyTorch ecosystem only? → TorchScript
│       └── Multiple frameworks/hardware? → ONNX
└── No: Need to resume training?
    ├── Yes: State Dict with full checkpoint
    └── No: State Dict (weights only)
```

## Common Serialization Issues

### Issue 1: Pickle Compatibility

```python
# Problem: Loading model saved with different Python/PyTorch version

# Solution: Use weights_only mode (PyTorch 2.0+)
state_dict = torch.load('model.pt', weights_only=True)

# Or catch and handle unpickling errors
try:
    state_dict = torch.load('model.pt')
except Exception as e:
    print(f"Failed to load: {e}")
    # Try alternative loading strategies
```

### Issue 2: Custom Layers in ONNX

```python
# Problem: Custom operation not supported in ONNX

# Solution: Register custom symbolic function
@torch.onnx.symbolic_helper.parse_args('v', 'v')
def my_custom_op(g, input, weight):
    # Define ONNX graph representation
    return g.op("MatMul", input, weight)

torch.onnx.register_custom_op_symbolic('my_namespace::custom_op', my_custom_op, 13)
```

### Issue 3: Dynamic Shapes in TorchScript

```python
# Problem: Dynamic tensor shapes cause tracing issues

# Solution: Use scripting instead of tracing for dynamic shapes
@torch.jit.script
def dynamic_forward(x: torch.Tensor) -> torch.Tensor:
    batch_size = x.size(0)  # Dynamic dimension
    # ... processing that depends on batch_size
    return x
```

## Summary

Model serialization enables the transition from research to production:

1. **State Dict**: Default choice for training and PyTorch-to-PyTorch transfer
2. **TorchScript**: Production deployment without Python dependency
3. **ONNX**: Cross-framework interoperability and hardware optimization

Choose the format based on your deployment target, performance requirements, and ecosystem constraints. When in doubt, save both state dict (for flexibility) and TorchScript/ONNX (for deployment).

## References

- [PyTorch Serialization Docs](https://pytorch.org/docs/stable/notes/serialization.html)
- [TorchScript Documentation](https://pytorch.org/docs/stable/jit.html)
- [ONNX Official Documentation](https://onnx.ai/onnx/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [PyTorch Mobile](https://pytorch.org/mobile/)
