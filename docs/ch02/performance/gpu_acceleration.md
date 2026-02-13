# GPU Acceleration

## Overview

Modern deep learning is computationally intensive. A single training run may involve billions of floating-point operations per batch, repeated over millions of parameter updates. **Graphics Processing Units (GPUs)** provide the parallel arithmetic throughput that makes this feasible. Understanding *why* GPUs accelerate tensor computation—and where they do not—is essential for writing efficient PyTorch code.

---

## CPU vs GPU Architecture

CPUs and GPUs solve fundamentally different engineering problems.

A CPU has a small number of powerful cores (typically 4–64), each optimised for low-latency sequential execution. Deep pipelines, branch predictors, and large caches allow a single core to retire complex instruction streams quickly. This makes CPUs ideal for control-heavy, branching workloads such as operating-system scheduling, data parsing, and general-purpose scripting.

A GPU, by contrast, packs thousands of simpler cores (modern NVIDIA data-centre GPUs exceed 10,000 CUDA cores) organised into **Streaming Multiprocessors (SMs)**. Each SM schedules groups of 32 threads—called **warps**—that execute the same instruction simultaneously. The resulting architecture is a **Single Instruction, Multiple Thread (SIMT)** machine: it sacrifices per-thread sophistication for massive data-parallel throughput.

| Property | CPU | GPU |
|---|---|---|
| Core count | 4–64 | 1,000–16,000+ |
| Clock speed | 3–5 GHz | 1–2 GHz |
| Cache per core | Large (MB) | Small (KB) |
| Memory bandwidth | 50–100 GB/s | 900–3,000+ GB/s |
| Strength | Sequential, branching logic | Data-parallel arithmetic |

Deep learning workloads are dominated by **matrix multiplications**, **element-wise activations**, and **reductions** (sums, means, norms)—operations that map naturally onto SIMT execution. A single forward pass through a linear layer computes $Y = XW + b$, where every output element is an independent dot product. Thousands of these dot products execute simultaneously on a GPU, producing speedups of 10–100× over a CPU for large tensors.

---

## Why GPUs Suit Deep Learning

Three properties of neural network training align with GPU strengths:

**Regularity.** Training repeatedly applies the same operations (convolutions, matrix multiplies, activations) to different data. The GPU's SIMT model is purpose-built for this pattern: one instruction stream, many data elements.

**Arithmetic intensity.** Matrix multiplication has $O(n^3)$ arithmetic for $O(n^2)$ data movement, giving a high ratio of compute to memory access. This keeps GPU cores busy rather than stalled waiting for data.

**Parallelism across the batch.** Mini-batch training processes $B$ samples independently through identical layers. Increasing $B$ adds parallelism with negligible overhead until GPU memory or compute saturates—a property that CPUs, with their limited core counts, cannot exploit.

---

## GPU Acceleration in PyTorch

PyTorch delegates tensor operations to optimised GPU libraries:

- **cuBLAS** for linear algebra (matrix multiplies, solves)
- **cuDNN** for neural network primitives (convolutions, RNNs, batch normalisation)
- **cuFFT** for Fourier transforms
- **cuRAND** for random number generation

These libraries are called transparently when tensors reside on a CUDA device. No manual kernel writing is required:

```python
import torch

# CPU computation
x_cpu = torch.randn(1000, 1000)
y_cpu = x_cpu @ x_cpu.T  # uses CPU BLAS

# GPU computation — same syntax, different device
x_gpu = torch.randn(1000, 1000, device="cuda")
y_gpu = x_gpu @ x_gpu.T  # dispatches to cuBLAS on GPU
```

The dispatch is automatic: PyTorch inspects the device attribute of input tensors and routes the operation accordingly.

---

## When GPUs Help Less

Not all workloads benefit from GPU acceleration:

**Small tensors.** Kernel launch overhead (typically 5–20 μs) dominates when the computation itself is trivial. A $10 \times 10$ matrix multiply finishes faster on a CPU than on a GPU once launch and synchronisation costs are included.

**Sequential, branching logic.** Python control flow, conditional operations on individual elements, and variable-length sequence processing under-utilise the SIMT pipeline. Operations like `if tensor[i] > 0` executed in a loop negate GPU advantages.

**I/O-bound tasks.** Data loading, disk reads, and network transfers are bottlenecked by I/O bandwidth, not arithmetic throughput. Adding a faster GPU does not help if the training loop spends most of its time waiting for data.

**Memory-bound operations.** Element-wise operations like `relu(x)` or `x + y` move data through memory at the bandwidth limit. Speedups depend on GPU memory bandwidth rather than compute, and the advantage over modern CPU memory subsystems is smaller (roughly 10–30× rather than 50–100×).

---

## Quantitative Finance Applications

GPU acceleration is particularly impactful in quantitative finance:

**Monte Carlo simulation.** Pricing derivatives by simulating millions of paths through a stochastic differential equation is embarrassingly parallel. Each path is independent, and the arithmetic (random number generation, cumulative products, discounting) maps directly onto GPU kernels. A GPU can evaluate $10^6$–$10^8$ paths in the time a CPU handles $10^4$–$10^5$.

**Neural network calibration.** Calibrating parametric models (e.g., neural stochastic volatility models) to observed option surfaces requires repeated forward passes and gradient computations. GPU-accelerated automatic differentiation reduces calibration time from hours to minutes.

**Real-time risk.** Value-at-Risk (VaR) and Expected Shortfall (ES) computations over large portfolios involve thousands of scenario evaluations. GPU parallelism enables intra-day risk updates that would be infeasible on CPUs alone.

**Portfolio optimisation.** Large-scale mean–variance or factor-model optimisations involve repeated matrix factorisations and solves. GPU-accelerated linear algebra libraries handle covariance matrices with thousands of assets efficiently.

---

## Key Takeaways

- GPUs achieve throughput by executing thousands of threads in lockstep (SIMT), trading single-thread performance for data-parallel capacity.
- Deep learning workloads—matrix multiplies, element-wise operations, batch processing—are naturally suited to this architecture.
- PyTorch transparently dispatches to optimised GPU libraries (cuBLAS, cuDNN) based on tensor device placement.
- GPU advantages diminish for small tensors, sequential logic, and I/O-bound workloads.
- In quantitative finance, GPU acceleration enables large-scale Monte Carlo, real-time risk, and neural model calibration at production speeds.
