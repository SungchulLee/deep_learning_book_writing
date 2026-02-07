# Quantization

## Overview

Quantization reduces the numerical precision of neural network weights and activations from floating-point (typically 32-bit) to lower bit-width representations (8-bit, 4-bit, or even binary). This achieves significant reductions in memory footprint and computational cost, with specialized hardware providing substantial inference speedups.

## Quantization Fundamentals

### Why Quantize?

Neural network weights and activations are typically stored as 32-bit floats:

```
FP32: ±[1.18e-38, 3.4e+38] with ~7 decimal digits precision
      Sign (1) | Exponent (8) | Mantissa (23)
```

Most neural network computations don't require this precision. Quantization maps values to lower precision:

```
INT8: [-128, 127] or [0, 255]
      8 bits total

FP16: ±[6.1e-5, 65504] with ~3 decimal digits
      Sign (1) | Exponent (5) | Mantissa (10)
```

### Benefits of Quantization

| Metric | FP32 → INT8 | FP32 → FP16 |
|--------|-------------|-------------|
| Model Size | 4× reduction | 2× reduction |
| Memory Bandwidth | 4× reduction | 2× reduction |
| Compute Speed | 2-4× faster* | 1.5-2× faster* |
| Accuracy Loss | 0-2% typical | <0.5% typical |

*Speedup depends on hardware support

## Mathematical Foundation

### Quantization Mapping

Quantization maps continuous floating-point values to a discrete set of integers:

$$x_{\text{float}} \in \mathbb{R} \rightarrow x_{\text{quant}} \in \{0, 1, \ldots, 2^b - 1\}$$

where $b$ is the bit-width.

### Uniform Affine Quantization

The most common quantization scheme uses a linear mapping with scale and zero-point:

**Quantization:**
$$x_q = \text{round}\left(\frac{x - z}{s}\right) = \text{round}\left(\frac{x}{s}\right) - z_q$$

**Dequantization:**
$$\hat{x} = s \cdot (x_q + z_q) = s \cdot x_q + z$$

where:
- $s$ is the scale factor
- $z$ is the zero-point (offset)
- $z_q$ is the quantized zero-point

### Scale and Zero-Point Computation

For a tensor with range $[x_{\min}, x_{\max}]$:

$$s = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}}$$

$$z = q_{\min} - \text{round}\left(\frac{x_{\min}}{s}\right)$$

For INT8 with unsigned representation: $q_{\min} = 0$, $q_{\max} = 255$

For INT8 with signed representation: $q_{\min} = -128$, $q_{\max} = 127$

### Symmetric vs. Asymmetric Quantization

**Symmetric Quantization** ($z = 0$):
$$x_q = \text{round}\left(\frac{x}{s}\right), \quad s = \frac{\max(|x_{\min}|, |x_{\max}|)}{q_{\max}}$$

Simpler but wastes dynamic range for asymmetric distributions.

**Asymmetric Quantization** ($z \neq 0$):
Uses the full formula above. Better utilizes the quantization range for skewed distributions (common for activations after ReLU).

## PyTorch Implementation

### Basic Quantization Operations

```python
import torch
import torch.nn as nn
import torch.quantization
from typing import Tuple, Dict, List
import time


def compute_quantization_params(tensor: torch.Tensor, 
                                num_bits: int = 8, 
                                symmetric: bool = True) -> Tuple[float, int]:
    """
    Compute scale and zero-point for quantization.
    
    Args:
        tensor: Input tensor to quantize
        num_bits: Number of bits for quantization
        symmetric: Whether to use symmetric quantization
        
    Returns:
        scale, zero_point
    """
    if symmetric:
        # Symmetric: range is [-max_abs, max_abs], zero_point = 0
        max_abs = tensor.abs().max()
        qmax = 2 ** (num_bits - 1) - 1
        scale = max_abs / qmax
        zero_point = 0
    else:
        # Asymmetric: use full range
        min_val = tensor.min()
        max_val = tensor.max()
        qmin = 0
        qmax = 2 ** num_bits - 1
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = int(round(-min_val / scale))
    
    return scale.item(), zero_point


def quantize(tensor: torch.Tensor, scale: float, zero_point: int, 
             num_bits: int = 8) -> torch.Tensor:
    """Quantize tensor to integer."""
    qmin = 0 if zero_point != 0 else -(2 ** (num_bits - 1))
    qmax = 2 ** num_bits - 1 if zero_point != 0 else 2 ** (num_bits - 1) - 1
    
    q = torch.round(tensor / scale) + zero_point
    q = torch.clamp(q, qmin, qmax)
    return q.to(torch.int8)


def dequantize(q_tensor: torch.Tensor, scale: float, 
               zero_point: int) -> torch.Tensor:
    """Dequantize integer tensor back to float."""
    return scale * (q_tensor.float() - zero_point)
```

## Quantization Error Analysis

### Quantization Noise

The quantization error for uniform quantization follows an approximately uniform distribution:

$$\epsilon = x - \hat{x} \sim \mathcal{U}\left(-\frac{s}{2}, \frac{s}{2}\right)$$

**Mean Squared Error:**
$$\text{MSE} = \mathbb{E}[\epsilon^2] = \frac{s^2}{12}$$

**Signal-to-Quantization-Noise Ratio (SQNR):**
$$\text{SQNR} = 10 \log_{10}\left(\frac{\sigma_x^2}{\text{MSE}}\right) \approx 6.02b + 4.77 \text{ dB}$$

Each additional bit approximately adds 6 dB to SQNR.

### Error Propagation

In a multi-layer network, quantization errors accumulate:

$$\epsilon_{\text{output}} = f(x + \epsilon_x, W + \epsilon_W) - f(x, W)$$

For linear layers, the error approximately scales with layer width and depth, motivating per-layer quantization strategies.

## Types of Quantization

### 1. Dynamic Quantization

Weights quantized statically, activations quantized dynamically at runtime.

**Best for**: Models with variable input sizes, RNNs, LSTMs, Transformers.

```python
class LinearModel(nn.Module):
    """Example model for dynamic quantization."""
    
    def __init__(self, input_size: int = 784, hidden_size: int = 256, 
                 num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def apply_dynamic_quantization(model: nn.Module,
                               dtype: torch.dtype = torch.qint8) -> nn.Module:
    """
    Apply dynamic quantization to a model.
    
    Dynamic quantization:
    - Weights: quantized statically (INT8)
    - Activations: quantized dynamically at inference time
    
    Args:
        model: Pre-trained FP32 model
        dtype: Quantization dtype (torch.qint8 or torch.float16)
        
    Returns:
        Dynamically quantized model
    """
    model.eval()
    
    # Get size before
    size_before = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Specify which layer types to quantize
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.Linear, nn.LSTM, nn.GRU},
        dtype=dtype
    )
    
    # Get size after
    size_after = sum(
        p.numel() * p.element_size() 
        for p in quantized_model.parameters()
    )
    
    print(f"Size before: {size_before / 1e6:.2f} MB")
    print(f"Size after: {size_after / 1e6:.2f} MB")
    print(f"Compression: {size_before / size_after:.2f}x")
    
    return quantized_model
```

### 2. Static Quantization

Both weights and activations quantized ahead of time using calibration data.

**Best for**: Fixed input sizes, CNNs, vision models.

```python
class QuantizableModel(nn.Module):
    """
    Model prepared for static quantization.
    
    Key modifications:
    1. Add QuantStub/DeQuantStub for input/output
    2. Replace functional operations with module equivalents
    3. Fuse layers where possible
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        # Quantization stubs
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        # Model layers
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()  # Use module, not F.relu
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(64 * 7 * 7, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)  # Quantize input
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        x = self.dequant(x)  # Dequantize output
        return x
    
    def fuse_model(self):
        """
        Fuse Conv-BN-ReLU sequences for better quantization.
        
        Fusing reduces quantization error by avoiding intermediate
        quantization/dequantization steps.
        """
        torch.quantization.fuse_modules(
            self,
            [['conv1', 'bn1', 'relu1'],
             ['conv2', 'bn2', 'relu2']],
            inplace=True
        )


def apply_static_quantization(model: nn.Module,
                              calibration_loader: torch.utils.data.DataLoader,
                              backend: str = 'fbgemm') -> nn.Module:
    """
    Apply static quantization with calibration.
    
    Static quantization:
    - Weights: quantized statically
    - Activations: quantized using pre-computed scale/zero-point from calibration
    
    Requires representative calibration data to determine activation ranges.
    
    Args:
        model: QuantizableModel (must have quant/dequant stubs)
        calibration_loader: DataLoader with representative samples
        backend: 'fbgemm' (x86) or 'qnnpack' (ARM)
        
    Returns:
        Statically quantized model
    """
    model.eval()
    
    # Fuse layers if method exists
    if hasattr(model, 'fuse_model'):
        model.fuse_model()
    
    # Set quantization configuration
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    
    # Insert observers
    torch.quantization.prepare(model, inplace=True)
    
    # Calibration: run representative data through model
    print("Calibrating...")
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(calibration_loader):
            model(data)
            if batch_idx >= 100:  # Use 100 batches for calibration
                break
    
    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)
    
    return model
```

### 3. Quantization-Aware Training (QAT)

Simulate quantization during training so the network learns to be robust to quantization errors.

**Best for**: When static quantization causes unacceptable accuracy loss.

**Forward Pass:**
$$\hat{x} = \text{FakeQuant}(x) = s \cdot \text{round}\left(\frac{\text{clamp}(x, x_{\min}, x_{\max})}{s}\right)$$

**Backward Pass (Straight-Through Estimator):**
$$\frac{\partial \mathcal{L}}{\partial x} \approx \frac{\partial \mathcal{L}}{\partial \hat{x}} \cdot \mathbf{1}[x_{\min} \leq x \leq x_{\max}]$$

The gradient passes through the quantization as if it were an identity function (within the clipping range).

```python
def train_with_qat(model: nn.Module,
                   train_loader: torch.utils.data.DataLoader,
                   test_loader: torch.utils.data.DataLoader,
                   epochs: int = 10,
                   lr: float = 1e-3,
                   backend: str = 'fbgemm',
                   device: str = 'cpu') -> nn.Module:
    """
    Quantization-Aware Training (QAT).
    
    Simulates quantization during training using fake quantization,
    allowing the model to learn weights robust to quantization error.
    
    QAT typically achieves 1-2% better accuracy than PTQ.
    
    Args:
        model: QuantizableModel to train
        train_loader: Training data
        test_loader: Test data for evaluation
        epochs: Training epochs
        lr: Learning rate
        backend: Quantization backend
        device: Training device
        
    Returns:
        Quantized model
    """
    model = model.to(device)
    model.train()
    
    # Fuse layers
    if hasattr(model, 'fuse_model'):
        model.fuse_model()
    
    # Set QAT configuration
    model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
    
    # Prepare for QAT (inserts fake quantization modules)
    torch.quantization.prepare_qat(model, inplace=True)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print("Starting Quantization-Aware Training...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # Freeze BatchNorm stats after warmup
        if epoch >= epochs // 2:
            model.apply(torch.quantization.disable_observer)
        if epoch >= epochs * 3 // 4:
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        # Evaluate
        if (epoch + 1) % 2 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    _, pred = output.max(1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
            
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Loss: {train_loss/len(train_loader):.4f}, "
                  f"Acc: {100*correct/total:.2f}%")
    
    # Convert to fully quantized model
    model.eval()
    model_quantized = torch.quantization.convert(model.cpu(), inplace=False)
    
    return model_quantized
```

## Mixed Precision (FP16)

Use half precision for faster computation on modern GPUs:

```python
def mixed_precision_inference(model: nn.Module, 
                              data: torch.Tensor) -> torch.Tensor:
    """
    Run inference with automatic mixed precision (AMP).
    
    AMP automatically uses FP16 where safe and FP32 where needed.
    """
    model.eval()
    device = next(model.parameters()).device
    data = data.to(device)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            output = model(data)
    
    return output


def benchmark_precision(model: nn.Module, data: torch.Tensor, 
                        iterations: int = 100) -> Dict[str, float]:
    """Compare FP32 vs FP16 performance."""
    device = torch.device('cuda')
    model = model.to(device)
    data = data.to(device)
    
    # FP32 benchmark
    model_fp32 = model.float()
    data_fp32 = data.float()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model_fp32(data_fp32)
    torch.cuda.synchronize()
    fp32_time = time.perf_counter() - start
    
    # FP16 benchmark
    model_fp16 = model.half()
    data_fp16 = data.half()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model_fp16(data_fp16)
    torch.cuda.synchronize()
    fp16_time = time.perf_counter() - start
    
    print(f"FP32: {fp32_time*1000:.2f} ms")
    print(f"FP16: {fp16_time*1000:.2f} ms")
    print(f"Speedup: {fp32_time/fp16_time:.2f}x")
    
    return {
        'fp32_time_ms': fp32_time * 1000,
        'fp16_time_ms': fp16_time * 1000,
        'speedup': fp32_time / fp16_time
    }
```

## Calibration Strategies

### Min-Max Calibration

```python
class MinMaxObserver:
    """Track min/max values for calibration."""
    
    def __init__(self):
        self.min_val = float('inf')
        self.max_val = float('-inf')
    
    def update(self, tensor: torch.Tensor):
        self.min_val = min(self.min_val, tensor.min().item())
        self.max_val = max(self.max_val, tensor.max().item())
    
    def compute_params(self, num_bits: int = 8) -> Tuple[float, int]:
        qmin, qmax = 0, 2**num_bits - 1
        scale = (self.max_val - self.min_val) / (qmax - qmin)
        zero_point = int(round(-self.min_val / scale))
        return scale, zero_point
```

### Histogram Calibration

```python
import numpy as np

class HistogramObserver:
    """Track histogram for better calibration."""
    
    def __init__(self, num_bins: int = 2048):
        self.num_bins = num_bins
        self.histogram = None
        self.bin_edges = None
    
    def update(self, tensor: torch.Tensor):
        values = tensor.flatten().cpu().numpy()
        
        if self.histogram is None:
            self.histogram, self.bin_edges = np.histogram(
                values, bins=self.num_bins
            )
        else:
            hist, _ = np.histogram(values, bins=self.bin_edges)
            self.histogram += hist
    
    def compute_params_entropy(self, num_bits: int = 8):
        """
        Compute optimal threshold using KL divergence (entropy calibration).
        
        Find threshold that minimizes information loss when quantizing.
        """
        # Use PyTorch's built-in HistogramObserver for production
        pass
```

### Percentile Calibration

```python
class PercentileObserver:
    """Use percentile for outlier robustness."""
    
    def __init__(self, percentile: float = 99.99):
        self.percentile = percentile
        self.values = []
    
    def update(self, tensor: torch.Tensor):
        self.values.extend(tensor.flatten().tolist())
    
    def compute_params(self, num_bits: int = 8) -> Tuple[float, int]:
        values = np.array(self.values)
        min_val = np.percentile(values, 100 - self.percentile)
        max_val = np.percentile(values, self.percentile)
        
        qmin, qmax = 0, 2**num_bits - 1
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = int(round(-min_val / scale))
        
        return scale, zero_point
```

## Per-Channel vs. Per-Tensor Quantization

### Per-Tensor Quantization

Single scale/zero-point for the entire tensor:
- Simpler implementation
- May lose precision if weight distributions vary across channels

### Per-Channel Quantization

Separate scale/zero-point for each output channel:
- Better preserves weight distributions
- Essential for convolutional layers
- Standard for modern quantization

```python
def analyze_weight_distribution(conv: nn.Conv2d) -> Dict:
    """
    Analyze weight distribution per output channel.
    
    Per-channel quantization is beneficial when channels have
    significantly different weight ranges.
    """
    weights = conv.weight.data  # (out_ch, in_ch, kH, kW)
    
    channel_stats = []
    for ch in range(weights.size(0)):
        ch_weights = weights[ch].flatten()
        channel_stats.append({
            'channel': ch,
            'min': ch_weights.min().item(),
            'max': ch_weights.max().item(),
            'mean': ch_weights.mean().item(),
            'std': ch_weights.std().item(),
            'range': (ch_weights.max() - ch_weights.min()).item()
        })
    
    # Compute range variation
    ranges = [s['range'] for s in channel_stats]
    range_ratio = max(ranges) / (min(ranges) + 1e-8)
    
    return {
        'channel_stats': channel_stats,
        'range_ratio': range_ratio,
        'recommendation': 'per-channel' if range_ratio > 2.0 else 'per-tensor'
    }
```

## Mixed-Precision Quantization

Different layers have different sensitivity to quantization. Mixed-precision assigns bit-widths per layer:

```python
def mixed_precision_sensitivity(model: nn.Module,
                                test_loader: torch.utils.data.DataLoader,
                                bit_widths: List[int] = [4, 8, 16],
                                device: str = 'cpu') -> Dict:
    """
    Analyze layer sensitivity to different quantization bit-widths.
    
    Args:
        model: Model to analyze
        test_loader: Test data
        bit_widths: Bit-widths to test
        device: Device for inference
        
    Returns:
        Sensitivity scores per layer per bit-width
    """
    import copy
    
    baseline_acc = evaluate_model(model, test_loader, device)
    results = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            results[name] = {}
            
            for bits in bit_widths:
                # Simulate quantization for this layer only
                model_copy = copy.deepcopy(model)
                
                for n, m in model_copy.named_modules():
                    if n == name:
                        # Quantize weights
                        weight = m.weight.data
                        q_min = -(2 ** (bits - 1))
                        q_max = 2 ** (bits - 1) - 1
                        scale = (weight.max() - weight.min()) / (q_max - q_min)
                        
                        quantized = torch.clamp(
                            torch.round(weight / scale),
                            q_min, q_max
                        )
                        m.weight.data = quantized * scale
                        break
                
                acc = evaluate_model(model_copy, test_loader, device)
                acc_drop = baseline_acc - acc
                
                results[name][bits] = {
                    'accuracy': acc,
                    'accuracy_drop': acc_drop
                }
    
    return results


def evaluate_model(model: nn.Module, 
                   test_loader: torch.utils.data.DataLoader,
                   device: str = 'cpu') -> float:
    """Evaluate model accuracy."""
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    return correct / total
```

## Hardware Considerations

### Supported Operations

| Operation | INT8 CPU | INT8 GPU | FP16 GPU |
|-----------|----------|----------|----------|
| Conv2d | ✓ | ✓ | ✓ |
| Linear | ✓ | ✓ | ✓ |
| BatchNorm | Fused | Fused | ✓ |
| ReLU | Fused | Fused | ✓ |
| Add | ✓ | ✓ | ✓ |
| Concat | ✓ | ✓ | ✓ |
| Softmax | FP32 fallback | FP32 fallback | ✓ |

### Backend Selection

```python
def select_quantization_backend(target_device: str) -> str:
    """
    Select appropriate quantization backend based on target hardware.
    
    Args:
        target_device: 'x86_cpu', 'arm_cpu', 'nvidia_gpu', 'apple_neural_engine'
        
    Returns:
        Backend string for PyTorch quantization
    """
    backends = {
        'x86_cpu': 'fbgemm',      # Intel/AMD CPUs
        'arm_cpu': 'qnnpack',      # ARM CPUs (mobile)
        'nvidia_gpu': 'tensorrt',  # NVIDIA GPUs (requires TensorRT)
        'apple_neural_engine': 'coreml'  # Apple devices (requires coremltools)
    }
    
    return backends.get(target_device, 'fbgemm')
```

## Best Practices

### Layer Sensitivity

First and last layers are typically most sensitive to quantization:

```python
def apply_mixed_precision_strategy(model: nn.Module,
                                   sensitive_layers: List[str] = None) -> nn.Module:
    """
    Apply higher precision to sensitive layers.
    
    Common strategy:
    - First conv layer: FP32 or FP16 (preserves input information)
    - Last linear layer: FP32 or FP16 (preserves output precision)
    - Middle layers: INT8
    """
    if sensitive_layers is None:
        # Default: first and last layers
        layer_names = [name for name, _ in model.named_modules() 
                      if isinstance(_, (nn.Conv2d, nn.Linear))]
        sensitive_layers = [layer_names[0], layer_names[-1]] if layer_names else []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if name in sensitive_layers:
                module.qconfig = None  # Keep in FP32
            else:
                module.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    return model
```

### Batch Normalization Folding

Fold BatchNorm into preceding Conv/Linear for efficiency:

```python
def fold_batchnorm(model: nn.Module) -> nn.Module:
    """
    Fold BatchNorm layers into preceding convolutions.
    
    For Conv-BN sequence:
    y = γ * (Wx + b - μ) / σ + β
      = (γ/σ) * Wx + (γ(b-μ)/σ + β)
      = W' * x + b'
    
    where:
    W' = W * γ/σ
    b' = γ(b-μ)/σ + β
    """
    # PyTorch provides this automatically during quantization
    torch.quantization.fuse_modules(
        model,
        # Specify fusion patterns
        [['conv', 'bn', 'relu']],  # Example pattern
        inplace=True
    )
    return model
```

### Calibration Data Guidelines

```python
def prepare_calibration_data(dataset, num_samples: int = 1000) -> List[torch.Tensor]:
    """
    Prepare representative calibration data.
    
    Guidelines:
    - Use 500-2000 samples
    - Include diverse examples
    - Match production data distribution
    - Don't use training data augmentation
    """
    indices = torch.randperm(len(dataset))[:num_samples]
    
    calibration_samples = []
    for idx in indices:
        sample, _ = dataset[idx]
        calibration_samples.append(sample.unsqueeze(0))
    
    return calibration_samples
```

## Quantization Error Debugging

```python
def debug_quantization_error(original: nn.Module,
                             quantized: nn.Module,
                             sample_input: torch.Tensor) -> Dict:
    """
    Analyze where quantization error originates.
    
    Compares intermediate activations between FP32 and quantized models.
    """
    original.eval()
    quantized.eval()
    
    activations_fp32 = {}
    activations_quant = {}
    
    # Register hooks to capture activations
    def make_hook(storage, name):
        def hook(module, input, output):
            storage[name] = output.detach()
        return hook
    
    hooks = []
    for name, module in original.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU)):
            hooks.append(module.register_forward_hook(
                make_hook(activations_fp32, name)
            ))
    
    for name, module in quantized.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU)):
            hooks.append(module.register_forward_hook(
                make_hook(activations_quant, name)
            ))
    
    # Forward pass
    with torch.no_grad():
        _ = original(sample_input)
        _ = quantized(sample_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Compare activations
    errors = {}
    for name in activations_fp32:
        if name in activations_quant:
            fp32_act = activations_fp32[name].float()
            quant_act = activations_quant[name].float()
            
            mse = ((fp32_act - quant_act) ** 2).mean().item()
            rel_error = (mse / (fp32_act ** 2).mean().item()) ** 0.5
            
            errors[name] = {
                'mse': mse,
                'relative_error': rel_error,
                'max_error': (fp32_act - quant_act).abs().max().item()
            }
    
    return errors


def validate_quantization(original_model: nn.Module, 
                          quantized_model: nn.Module, 
                          test_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
    """
    Compare accuracy of original and quantized models.
    """
    original_model.eval()
    quantized_model.eval()
    
    original_correct = 0
    quantized_correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            # Original model
            out_orig = original_model(data)
            pred_orig = out_orig.argmax(dim=1)
            original_correct += (pred_orig == target).sum().item()
            
            # Quantized model
            out_quant = quantized_model(data)
            pred_quant = out_quant.argmax(dim=1)
            quantized_correct += (pred_quant == target).sum().item()
            
            total += target.size(0)
    
    orig_acc = original_correct / total
    quant_acc = quantized_correct / total
    
    print(f"Original accuracy: {orig_acc*100:.2f}%")
    print(f"Quantized accuracy: {quant_acc*100:.2f}%")
    print(f"Accuracy drop: {(orig_acc - quant_acc)*100:.2f}%")
    
    return orig_acc, quant_acc


def measure_quantization_impact(original: nn.Module,
                                quantized: nn.Module,
                                test_loader: torch.utils.data.DataLoader,
                                device: str = 'cpu') -> Dict:
    """
    Compare original and quantized model performance.
    """
    # Model sizes
    def get_size_mb(model):
        param_size = sum(p.nelement() * p.element_size() 
                        for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() 
                         for b in model.buffers())
        return (param_size + buffer_size) / (1024 ** 2)
    
    orig_size = get_size_mb(original)
    quant_size = get_size_mb(quantized)
    
    # Inference time
    sample_input = next(iter(test_loader))[0][:1].to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = original(sample_input)
            _ = quantized(sample_input.cpu())
    
    # Timing
    n_runs = 100
    
    start = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = original(sample_input)
    orig_time = (time.time() - start) / n_runs
    
    start = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = quantized(sample_input.cpu())
    quant_time = (time.time() - start) / n_runs
    
    return {
        'original_size_mb': orig_size,
        'quantized_size_mb': quant_size,
        'compression_ratio': orig_size / quant_size,
        'original_latency_ms': orig_time * 1000,
        'quantized_latency_ms': quant_time * 1000,
        'speedup': orig_time / quant_time
    }
```

## Summary

Quantization is essential for efficient deployment:

1. **Dynamic quantization**: Easy to apply, good for RNNs/Transformers
2. **Static quantization**: Best compression, requires calibration
3. **QAT**: Best accuracy, requires training infrastructure
4. **FP16**: Simple, good GPU speedup, minimal accuracy loss

| Scenario | Recommended Method |
|----------|-------------------|
| Quick deployment | Dynamic quantization |
| Maximum accuracy | QAT |
| CNNs/Vision | Static quantization |
| RNNs/Transformers | Dynamic quantization |
| GPU deployment | FP16 / INT8 TensorRT |

Key recommendations:
- Start with dynamic quantization for quick wins
- Use static quantization for CNNs/vision models
- Apply QAT when accuracy drop is unacceptable
- Always validate accuracy after quantization
- Use representative calibration data
- Keep first and last layers at higher precision if needed

## References

1. Jacob, B., et al. "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." CVPR 2018.
2. Banner, R., et al. "Post Training 4-bit Quantization of Convolutional Networks for Rapid-Deployment." NeurIPS 2019.
3. Nagel, M., et al. "Data-Free Quantization Through Weight Equalization and Bias Correction." ICCV 2019.
4. Gholami, A., et al. "A Survey of Quantization Methods for Efficient Neural Network Inference." arXiv 2021.
5. Krishnamoorthi, R. "Quantizing Deep Convolutional Networks for Efficient Inference." arXiv 2018.
