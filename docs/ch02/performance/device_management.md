# Device Management

## Overview

PyTorch tensors carry a **device** attribute that determines where their data resides and where operations on them execute. Correct device management—creating tensors on the right device, transferring them when necessary, and ensuring all operands share a device—is a prerequisite for GPU-accelerated training. This section covers the practical mechanics of device handling in PyTorch.

---

## The `torch.device` Object

Every tensor in PyTorch is associated with a `torch.device` that specifies both a device type and an optional index:

```python
import torch

cpu_device = torch.device("cpu")
gpu_device = torch.device("cuda")       # default GPU (index 0)
gpu1_device = torch.device("cuda:1")    # second GPU
```

The string `"cuda"` without an index refers to the **current default CUDA device**, which is GPU 0 unless changed with `torch.cuda.set_device()`.

Querying a tensor's device is straightforward:

```python
x = torch.randn(3, 3)
print(x.device)  # cpu

y = torch.randn(3, 3, device="cuda")
print(y.device)  # cuda:0
```

---

## Creating Tensors on a Device

Tensors can be placed on a specific device at creation time, avoiding unnecessary transfers:

```python
# Direct creation on GPU
x = torch.randn(1000, 1000, device="cuda")
w = torch.zeros(256, 128, device="cuda")
indices = torch.arange(100, device="cuda")
```

This is more efficient than creating on CPU and transferring:

```python
# Less efficient: CPU allocation + host-to-device copy
x = torch.randn(1000, 1000).to("cuda")  # involves an extra copy
```

When a tensor is created from another tensor, PyTorch preserves the source device by default:

```python
x = torch.randn(3, 3, device="cuda")
y = torch.zeros_like(x)   # also on cuda:0
z = x + 1                  # also on cuda:0
```

---

## Moving Tensors Between Devices

The `.to()` method transfers tensors between devices:

```python
x_cpu = torch.randn(3, 3)
x_gpu = x_cpu.to("cuda")      # CPU → GPU
x_back = x_gpu.to("cpu")      # GPU → CPU
```

Equivalent shorthand methods exist:

```python
x_gpu = x_cpu.cuda()          # CPU → default GPU
x_gpu1 = x_cpu.cuda(1)        # CPU → GPU 1
x_cpu = x_gpu.cpu()           # GPU → CPU
```

The `.to()` method is a **no-op** when the tensor is already on the target device, making it safe to call unconditionally:

```python
x = torch.randn(3, 3, device="cuda")
y = x.to("cuda")  # no copy, returns x itself
print(x.data_ptr() == y.data_ptr())  # True
```

### Transfer Cost

Host-to-device (CPU → GPU) and device-to-host (GPU → CPU) copies traverse the **PCIe bus** (or NVLink for GPU-to-GPU). PCIe 4.0 provides roughly 25 GB/s of bidirectional bandwidth—orders of magnitude slower than GPU memory bandwidth (900+ GB/s). This asymmetry has practical consequences:

- Transferring a $1000 \times 1000$ float32 tensor (4 MB) takes approximately 0.16 ms over PCIe 4.0.
- A batch of images ($64 \times 3 \times 224 \times 224$, ~37 MB) takes roughly 1.5 ms.
- Frequent small transfers can dominate total runtime if repeated every iteration.

The guiding principle: **minimise transfers by keeping data on the GPU throughout the training loop**.

---

## The Same-Device Constraint

PyTorch requires all operands in a computation to reside on the same device. Mixing devices produces a `RuntimeError`:

```python
x = torch.randn(3, 3, device="cuda")
y = torch.randn(3, 3, device="cpu")

z = x + y
# RuntimeError: Expected all tensors to be on the same device,
# but found at least two devices, cuda:0 and cpu!
```

This constraint applies equally to model parameters and input tensors:

```python
model = torch.nn.Linear(128, 64).to("cuda")
x = torch.randn(32, 128)  # on CPU

output = model(x)
# RuntimeError: Input and parameter tensors are not on the same device
```

The fix is to ensure inputs are on the same device as the model:

```python
x = x.to("cuda")
output = model(x)  # works
```

---

## Device-Agnostic Code

Production code should work on both CPU and GPU without hard-coding device names. The standard pattern:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create model and data on the resolved device
model = MyModel().to(device)
x = torch.randn(batch_size, input_dim, device=device)
```

For more robust device selection that also handles Apple Silicon:

```python
def get_device():
    """Select the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon GPU
    else:
        return torch.device("cpu")

device = get_device()
```

### Moving Entire Models

`nn.Module.to()` moves all parameters and buffers recursively:

```python
model = torch.nn.Sequential(
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10),
)
model.to(device)  # moves all parameters and buffers
```

Unlike tensor `.to()`, module `.to()` operates **in-place**—it modifies the module and returns it:

```python
# Tensor .to() returns a new tensor (or same if already on device)
x_gpu = x.to("cuda")  # x is unchanged if it was on CPU

# Module .to() modifies in-place
model.to("cuda")  # model's parameters are now on GPU
```

---

## Multi-GPU Basics

Systems with multiple GPUs expose them as `cuda:0`, `cuda:1`, etc. PyTorch provides several mechanisms for multi-GPU computation:

### Manual Placement

```python
# Place different models on different GPUs
encoder = Encoder().to("cuda:0")
decoder = Decoder().to("cuda:1")

# Manually transfer intermediate results
hidden = encoder(x.to("cuda:0"))
output = decoder(hidden.to("cuda:1"))
```

### `DataParallel` (Simple)

`DataParallel` replicates a model across GPUs and splits each batch:

```python
model = MyModel().to("cuda:0")
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
# Forward pass automatically splits batch across 4 GPUs
```

`DataParallel` is simple but has known bottlenecks: all gradients are gathered to GPU 0, creating a memory and bandwidth asymmetry.

### `DistributedDataParallel` (Production)

For serious multi-GPU training, `DistributedDataParallel` (DDP) is preferred. It uses one process per GPU and overlaps gradient communication with backward computation:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
model = MyModel().to(local_rank)
model = DDP(model, device_ids=[local_rank])
```

DDP is the recommended approach for multi-GPU training in production but requires understanding process groups and distributed launchers. Full coverage is deferred to later chapters on distributed training.

---

## Common Device Management Patterns

### Training Loop Pattern

A typical training loop with proper device management:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MyModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        # Transfer data to device
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Forward, loss, backward, step
        output = model(batch_x)
        loss = criterion(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Saving and Loading Across Devices

When saving a model trained on GPU and loading on CPU (or vice versa), use `map_location`:

```python
# Save
torch.save(model.state_dict(), "model.pt")

# Load on CPU regardless of where it was trained
state_dict = torch.load("model.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.to(device)
```

Without `map_location`, loading a GPU-trained model on a CPU-only machine raises an error.

### Checking Device Availability

```python
# CUDA availability
print(torch.cuda.is_available())       # True/False
print(torch.cuda.device_count())       # number of GPUs
print(torch.cuda.get_device_name(0))   # e.g., "NVIDIA A100-SXM4-80GB"

# Current memory usage
print(torch.cuda.memory_allocated())   # bytes currently allocated
print(torch.cuda.memory_reserved())    # bytes reserved by caching allocator
```

---

## Key Takeaways

- Every PyTorch tensor has a `device` attribute; all operands in a computation must share the same device.
- Create tensors directly on the target device when possible to avoid unnecessary CPU → GPU copies.
- Use `torch.device("cuda" if torch.cuda.is_available() else "cpu")` for device-agnostic code.
- `nn.Module.to()` moves parameters in-place; `Tensor.to()` returns a new tensor (or the same one if already on the target device).
- Minimise host–device transfers—they traverse PCIe and are orders of magnitude slower than on-device memory access.
- Use `map_location` when loading checkpoints to handle cross-device portability.
