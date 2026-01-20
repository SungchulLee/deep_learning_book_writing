# 1.1.3 GPU Configuration

## Learning Objectives

By the end of this section, you will be able to:
- Understand GPU computing architecture and CUDA fundamentals
- Verify GPU availability and capabilities on your system
- Install and configure NVIDIA drivers and CUDA toolkit
- Set up PyTorch with GPU support
- Configure Apple Silicon MPS acceleration
- Monitor GPU usage and troubleshoot common issues
- Optimize GPU memory management for deep learning workloads

---

## Introduction

Graphics Processing Units (GPUs) have revolutionized deep learning by providing massive parallelism for matrix operations. A modern NVIDIA GPU can accelerate training by 10-100x compared to CPUs, making previously intractable problems feasible. This section covers the complete GPU setup process for deep learning.

### Why GPUs for Deep Learning?

Deep learning operations are fundamentally parallel matrix computations:

$$\mathbf{Y} = \mathbf{W} \cdot \mathbf{X} + \mathbf{b}$$

A forward pass through a neural network involves millions of such operations. CPUs execute these sequentially (a few at a time), while GPUs execute thousands simultaneously:

| Hardware | Typical Cores | Architecture | Use Case |
|----------|---------------|--------------|----------|
| **CPU** | 8-32 | Few powerful cores | Sequential tasks |
| **GPU** | 2,000-16,000+ | Many simple cores | Parallel computation |

**Performance comparison** (approximate, varies by model):

| Task | CPU Time | GPU Time | Speedup |
|------|----------|----------|---------|
| ResNet-50 training (1 epoch) | 2 hours | 8 minutes | 15x |
| GPT-2 fine-tuning | 24 hours | 2 hours | 12x |
| BERT inference (batch=32) | 500 ms | 20 ms | 25x |

---

## GPU Computing Fundamentals

### CUDA Architecture

NVIDIA's CUDA (Compute Unified Device Architecture) is the dominant platform for GPU computing in deep learning:

```
┌─────────────────────────────────────────────────────────────┐
│                      Application Layer                       │
│              (PyTorch, TensorFlow, JAX)                     │
├─────────────────────────────────────────────────────────────┤
│                    Framework Libraries                       │
│           (cuDNN, cuBLAS, NCCL, TensorRT)                   │
├─────────────────────────────────────────────────────────────┤
│                     CUDA Runtime API                         │
│              (Memory management, kernels)                    │
├─────────────────────────────────────────────────────────────┤
│                      CUDA Driver                             │
│               (Hardware communication)                       │
├─────────────────────────────────────────────────────────────┤
│                    NVIDIA GPU Driver                         │
│                 (OS-level interface)                         │
├─────────────────────────────────────────────────────────────┤
│                       GPU Hardware                           │
│                   (NVIDIA GPU Card)                          │
└─────────────────────────────────────────────────────────────┘
```

**Key Components:**

- **GPU Driver**: Operating system interface to the hardware
- **CUDA Driver**: Low-level GPU control
- **CUDA Toolkit**: Compiler (nvcc), libraries, tools
- **cuDNN**: Optimized deep learning primitives (convolutions, RNNs, etc.)
- **cuBLAS**: Optimized linear algebra (matrix multiplication)
- **NCCL**: Multi-GPU communication

### CUDA Version Compatibility

CUDA versions must align across the stack:

| Component | Must Match |
|-----------|------------|
| GPU Driver | ≥ CUDA Toolkit version |
| CUDA Toolkit | Framework requirement |
| PyTorch/TensorFlow | Compiled against specific CUDA |
| cuDNN | CUDA Toolkit version |

**Example**: PyTorch 2.6 with CUDA 12.1 requires:
- NVIDIA Driver ≥ 530.30
- CUDA Toolkit 12.1 (bundled with PyTorch)
- cuDNN 8.x (bundled with PyTorch)

---

## Checking GPU Availability

### NVIDIA GPU Detection

**Linux/Windows:**
```bash
# Check if NVIDIA GPU is detected
nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.12   Driver Version: 535.104.12   CUDA Version: 12.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA RTX 4090     Off  | 00000000:01:00.0  On |                  Off |
| 30%   35C    P8    22W / 450W |    512MiB / 24564MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

**Key information:**
- **Driver Version**: 535.104.12 (your installed driver)
- **CUDA Version**: 12.2 (maximum supported CUDA for this driver)
- **Memory-Usage**: 512MiB / 24564MiB (used / total VRAM)
- **GPU-Util**: 0% (current utilization)

### PyTorch GPU Check

```python
import torch

# Basic availability
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    # CUDA version
    print("CUDA version:", torch.version.cuda)
    
    # Number of GPUs
    print("GPU count:", torch.cuda.device_count())
    
    # GPU name
    print("GPU name:", torch.cuda.get_device_name(0))
    
    # GPU memory
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU memory: {total:.1f} GB")
    
    # cuDNN version
    print("cuDNN version:", torch.backends.cudnn.version())
    print("cuDNN enabled:", torch.backends.cudnn.enabled)
```

### Apple Silicon (MPS) Check

For M1/M2/M3/M4 Macs:

```python
import torch

# MPS (Metal Performance Shaders) availability
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device:", device)
```

---

## NVIDIA Driver Installation

### Linux (Ubuntu/Debian)

#### Method 1: Ubuntu Drivers (Recommended)

```bash
# Check available drivers
ubuntu-drivers devices

# Install recommended driver automatically
sudo ubuntu-drivers autoinstall

# Or install specific version
sudo apt install nvidia-driver-535

# Reboot required
sudo reboot
```

#### Method 2: Official NVIDIA Repository

```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install driver
sudo apt update
sudo apt install nvidia-driver-535

# Reboot
sudo reboot
```

### Linux (Fedora/RHEL)

```bash
# Enable RPM Fusion (required for NVIDIA drivers)
sudo dnf install https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm
sudo dnf install https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm

# Install NVIDIA driver
sudo dnf install akmod-nvidia

# Wait for kernel module to build (may take several minutes)
# Reboot
sudo reboot
```

### Windows

#### Method 1: GeForce Experience (Consumer GPUs)

1. Download GeForce Experience from https://www.nvidia.com/en-us/geforce/geforce-experience/
2. Install and run
3. Go to "Drivers" tab
4. Click "Download" for latest driver
5. Install and restart

#### Method 2: Manual Download

1. Visit https://www.nvidia.com/Download/index.aspx
2. Select your GPU model
3. Download and run installer
4. Choose "Custom Installation"
5. Select "Clean Installation" for fresh setup
6. Restart computer

#### Method 3: Chocolatey

```powershell
choco install nvidia-display-driver -y
```

### Verify Driver Installation

```bash
nvidia-smi
```

If this shows your GPU info, the driver is working.

---

## CUDA Toolkit Installation

For deep learning, you typically don't need to install the CUDA Toolkit separately—PyTorch and TensorFlow bundle the necessary CUDA components. However, if you need the full toolkit:

### Conda Method (Recommended)

```bash
# CUDA runtime (usually sufficient)
conda install -c conda-forge cudatoolkit=12.1

# Or with NVIDIA channel
conda install -c nvidia cuda-toolkit=12.1
```

### System Installation (Linux)

```bash
# Download CUDA installer
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run

# Run installer (select components you need)
sudo sh cuda_12.1.0_530.30.02_linux.run

# Add to PATH (add to ~/.bashrc)
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
```

### Verify CUDA Installation

```bash
nvcc --version
```

---

## PyTorch GPU Installation

### Option 1: Conda (Recommended)

```bash
# Remove CPU version first if installed
conda remove pytorch torchvision torchaudio cpuonly

# Install with CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Or with CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Or with CUDA 12.4 (latest)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

### Option 2: Pip

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Verify PyTorch GPU Installation

```python
import torch

# Comprehensive check
print("=" * 50)
print("PyTorch GPU Configuration")
print("=" * 50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Total memory: {props.total_memory / 1e9:.1f} GB")
        print(f"  Multi-processors: {props.multi_processor_count}")
    
    # Simple computation test
    print("\n" + "=" * 50)
    print("GPU Computation Test")
    print("=" * 50)
    
    x = torch.randn(10000, 10000, device='cuda')
    y = torch.randn(10000, 10000, device='cuda')
    
    # Warm up
    _ = torch.mm(x, y)
    torch.cuda.synchronize()
    
    # Timed test
    import time
    start = time.time()
    for _ in range(10):
        _ = torch.mm(x, y)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"Matrix multiplication (10000x10000) x 10: {elapsed:.3f}s")
    print(f"Throughput: {10 * 2 * 10000**3 / elapsed / 1e12:.1f} TFLOPS")
    
    print("\n✅ GPU is working correctly!")
else:
    print("\n❌ No GPU available. Using CPU.")
```

---

## Apple Silicon MPS Setup

Apple Silicon Macs (M1/M2/M3/M4) use Metal Performance Shaders (MPS) for GPU acceleration instead of CUDA.

### Installation

MPS support is included in standard PyTorch builds:

```bash
# No special installation needed
conda install pytorch torchvision torchaudio -c pytorch
# Or
pip install torch torchvision torchaudio
```

### Using MPS in Code

```python
import torch

# Check availability
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Move tensors to MPS
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)
z = torch.mm(x, y)  # Computed on MPS

# Move models to MPS
model = torch.nn.Linear(100, 10)
model = model.to(device)
```

### MPS Limitations

MPS is less mature than CUDA. Current limitations:

- Not all PyTorch operations supported
- Some operations fall back to CPU
- Less memory than dedicated GPUs (shared with system)
- No multi-GPU support
- Some numerical precision differences

**Workaround for unsupported operations:**

```python
# Some operations might fail on MPS
try:
    result = some_operation(tensor.to("mps"))
except RuntimeError:
    # Fall back to CPU
    result = some_operation(tensor.to("cpu")).to("mps")
```

---

## Device Management in PyTorch

### Automatic Device Selection

```python
import torch

def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()
print(f"Using: {device}")
```

### Moving Data to GPU

```python
# Create on GPU directly
x = torch.randn(1000, 1000, device='cuda')

# Move existing tensor
x_cpu = torch.randn(1000, 1000)
x_gpu = x_cpu.to('cuda')  # Creates copy on GPU
x_gpu = x_cpu.cuda()       # Equivalent

# Move model
model = MyModel()
model = model.to('cuda')   # Moves all parameters to GPU
model = model.cuda()       # Equivalent
```

### Multi-GPU Usage

```python
import torch
import torch.nn as nn

# Check multiple GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    
    # DataParallel (simple, older)
    model = nn.DataParallel(model)
    
    # DistributedDataParallel (recommended)
    # Requires torch.distributed setup
    
model = model.cuda()
```

---

## GPU Memory Management

### Understanding GPU Memory

```python
import torch

# Current memory usage
allocated = torch.cuda.memory_allocated() / 1e9
reserved = torch.cuda.memory_reserved() / 1e9
print(f"Allocated: {allocated:.2f} GB")
print(f"Reserved: {reserved:.2f} GB")

# Peak memory
max_allocated = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak: {max_allocated:.2f} GB")

# Reset peak stats
torch.cuda.reset_peak_memory_stats()
```

### Memory Optimization Techniques

#### 1. Clear Cache

```python
# Force garbage collection
import gc
gc.collect()
torch.cuda.empty_cache()
```

#### 2. Gradient Checkpointing

Trade computation for memory:

```python
from torch.utils.checkpoint import checkpoint

class MyModel(nn.Module):
    def forward(self, x):
        # Checkpoint intermediate layers
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        return self.layer3(x)
```

#### 3. Mixed Precision Training

Use 16-bit floats to halve memory:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for inputs, targets in dataloader:
    optimizer.zero_grad()
    
    # Forward pass in mixed precision
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    # Backward pass with scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

#### 4. Reduce Batch Size

```python
# If you get OOM, reduce batch size
# Or use gradient accumulation

accumulation_steps = 4
for i, (inputs, targets) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## Monitoring GPU Usage

### nvidia-smi

```bash
# Single snapshot
nvidia-smi

# Continuous monitoring (every 1 second)
nvidia-smi -l 1

# Or using watch
watch -n 1 nvidia-smi

# Query specific metrics
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 1
```

### PyTorch Profiler

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# Profile a training step
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    with record_function("forward"):
        output = model(input_tensor)
    with record_function("backward"):
        output.sum().backward()

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export for TensorBoard
prof.export_chrome_trace("trace.json")
```

### nvitop (Recommended)

```bash
pip install nvitop

# Run interactive monitor
nvitop

# Or as process monitor
nvitop -m full
```

---

## Troubleshooting

### Common Issues and Solutions

#### "CUDA out of memory"

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# 1. Reduce batch size
batch_size = batch_size // 2

# 2. Clear cache
torch.cuda.empty_cache()

# 3. Use gradient checkpointing
from torch.utils.checkpoint import checkpoint

# 4. Use mixed precision
from torch.cuda.amp import autocast

# 5. Monitor memory
print(torch.cuda.memory_summary())
```

#### "CUDA driver version is insufficient"

**Cause**: Driver too old for installed CUDA/PyTorch.

**Solutions**:
```bash
# Check minimum driver for CUDA version
# CUDA 12.1 requires driver >= 530.30

# Update driver
sudo apt install nvidia-driver-535
# Or download from nvidia.com
```

#### "CUDA not available" (even with GPU)

**Diagnostic**:
```python
import torch
print(torch.__version__)  # Check PyTorch version
print(torch.version.cuda)  # Check CUDA version (None if CPU-only)
```

**Solutions**:
```bash
# Reinstall PyTorch with CUDA
conda remove pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

#### GPU not utilized (shows 0% usage)

**Cause**: Data on CPU, only model on GPU.

**Fix**:
```python
# Move BOTH data and model to GPU
inputs = inputs.to('cuda')
targets = targets.to('cuda')
model = model.to('cuda')
```

#### Slow GPU performance

**Causes and fixes**:

1. **PCIe bottleneck**: Use more GPU computation per data transfer
2. **Small batch size**: Increase batch size
3. **CPU preprocessing**: Use DataLoader with `num_workers > 0`
4. **Synchronization**: Avoid `.item()` in training loops

```python
# SLOW: Forces synchronization each iteration
for batch in dataloader:
    loss = model(batch)
    print(loss.item())  # Synchronizes every iteration!

# FAST: Only synchronize periodically
losses = []
for batch in dataloader:
    loss = model(batch)
    losses.append(loss)
    
total_loss = torch.stack(losses).mean().item()  # Single synchronization
```

---

## Environment File with GPU Support

Complete `environment.yml` for GPU development:

```yaml
name: dl_gpu
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  # Core Python
  - python=3.10
  
  # Deep Learning (GPU)
  - pytorch>=2.0
  - torchvision>=0.15
  - torchaudio>=2.0
  - pytorch-cuda=12.1
  
  # CUDA Tools
  - cudatoolkit=12.1
  
  # Scientific Computing
  - numpy>=2.0
  - pandas>=2.0
  - scipy>=1.10
  - scikit-learn>=1.3
  
  # Visualization
  - matplotlib>=3.7
  - seaborn>=0.12
  
  # Development
  - jupyterlab>=4.0
  - ipykernel>=6.0
  
  # GPU Monitoring
  - pip
  - pip:
    - nvitop>=1.0
    - wandb>=0.15
```

---

## Summary

GPU acceleration is essential for practical deep learning. Key takeaways:

1. **Verify GPU availability** using `nvidia-smi` and PyTorch checks before starting projects.

2. **Match versions carefully**: Driver ≥ CUDA Toolkit ≥ PyTorch CUDA version.

3. **Use conda for installation**: It manages CUDA dependencies automatically.

4. **Move both data and model** to GPU—a common mistake is leaving data on CPU.

5. **Monitor memory usage**: OOM errors are the most common GPU issue.

6. **Use mixed precision** for larger models and reduced memory usage.

7. **Apple Silicon uses MPS**, not CUDA—the API is similar but with some limitations.

---

## Exercises

### Exercise 1: GPU Benchmark

Write a script that benchmarks CPU vs GPU performance for matrix multiplication of various sizes (100x100 to 10000x10000). Plot the speedup factor.

### Exercise 2: Memory Profiling

Train a small neural network and profile GPU memory usage. Implement gradient checkpointing and measure the memory savings.

### Exercise 3: Multi-GPU Setup

If you have multiple GPUs, implement DataParallel and measure the training speedup compared to single GPU.

---

## References

1. NVIDIA CUDA Documentation. https://docs.nvidia.com/cuda/
2. PyTorch CUDA Semantics. https://pytorch.org/docs/stable/notes/cuda.html
3. PyTorch Mixed Precision Training. https://pytorch.org/docs/stable/amp.html
4. Apple MPS Backend. https://pytorch.org/docs/stable/notes/mps.html
5. NVIDIA GPU Computing SDK. https://developer.nvidia.com/gpu-computing-sdk
