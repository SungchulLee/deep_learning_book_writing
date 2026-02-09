# Chapter 40: Model Deployment

## Introduction

Model deployment bridges the gap between research and production. While training focuses on achieving high accuracy, deployment requires balancing accuracy with latency, throughput, model size, and resource constraints. Training a model is only half the battle—converting a research prototype into a production-ready system involves serialization, optimization, efficient inference, scalable serving, distributed training for large-scale models, and robust MLOps practices to maintain reliability over time.

In quantitative finance, deployment carries additional demands: microsecond-level latency for trading systems, strict regulatory requirements for model governance, deterministic reproducibility for backtesting, and the ability to handle real-time market data streams without interruption.

## Why Deployment Matters

A model that performs well in a Jupyter notebook may fail entirely in production. The deployment process transforms a trained model into a reliable, performant, and maintainable system through several interconnected concerns:

1. **Serialization**: Converting models to formats that can run efficiently outside Python—from simple state dicts to cross-platform formats like ONNX and TorchScript
2. **Optimization**: Reducing latency and memory footprint through quantization, pruning, knowledge distillation, and neural architecture search while maintaining acceptable accuracy
3. **Inference**: Executing predictions efficiently across diverse hardware—batching strategies, streaming pipelines, GPU/CPU-specific optimizations, and memory management
4. **Serving**: Building APIs and infrastructure to handle prediction requests at scale using REST, gRPC, TorchServe, Triton, and BentoML
5. **Distributed Training**: Scaling model training across multiple GPUs and nodes using data parallelism, model parallelism, pipeline parallelism, FSDP, and DeepSpeed
6. **MLOps**: Managing the full model lifecycle—experiment tracking, model registries, CI/CD pipelines, monitoring, and A/B testing
7. **Finance-Specific Deployment**: Meeting the unique requirements of financial systems including ultra-low latency, real-time data processing, backtesting infrastructure, and production pipeline orchestration

## The Deployment Pipeline

A typical deployment workflow consists of five stages:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Training  │────▶│   Convert   │────▶│  Optimize   │────▶│  Benchmark  │────▶│   Deploy    │
│   (PyTorch) │     │   (Export)  │     │ (Quantize)  │     │   (Test)    │     │   (Serve)   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

**Stage 1: Train** — Train or fine-tune the model to achieve target accuracy on the validation set. For large models, distributed training strategies (data parallel, model parallel, pipeline parallel) may be required to fit within available hardware.

**Stage 2: Convert** — Export the model to a deployment-friendly format like ONNX or TorchScript that enables cross-platform execution and hardware acceleration. Save model weights using state dicts and establish checkpointing strategies for fault tolerance.

**Stage 3: Optimize** — Apply optimization techniques such as quantization, pruning, knowledge distillation, and neural architecture search to reduce model size and improve inference speed while maintaining acceptable accuracy.

**Stage 4: Benchmark** — Measure key performance metrics including latency (p50, p95, p99), throughput, memory usage, and accuracy on the target hardware. For finance applications, measure tail latencies and jitter under realistic market conditions.

**Stage 5: Deploy** — Deploy the optimized model using appropriate serving infrastructure (REST APIs, gRPC, TorchServe, Triton) with monitoring, logging, and A/B testing frameworks in place.

## Key Performance Metrics

### Latency

Latency measures the time required for a single inference:

$$\text{Latency} = t_{\text{preprocess}} + t_{\text{inference}} + t_{\text{postprocess}}$$

Target latency depends on the application:

| Application Type | Target Latency |
|-----------------|----------------|
| High-Frequency Trading | < 1 ms |
| Interactive (UI) | < 10 ms |
| Real-time systems | < 100 ms |
| Batch processing | < 1 s |

```python
import time
import torch

def measure_latency(model, input_tensor, num_runs=100, warmup=10):
    """Measure average inference latency with percentile statistics."""
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Warmup runs to stabilize GPU clocks and caches
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

    latencies.sort()
    return {
        "mean_ms": sum(latencies) / len(latencies),
        "p50_ms": latencies[len(latencies) // 2],
        "p95_ms": latencies[int(len(latencies) * 0.95)],
        "p99_ms": latencies[int(len(latencies) * 0.99)],
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
            "throughput_samples_per_sec": throughput,
            "latency_ms": latency_ms,
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
    return {"total": total, "trainable": trainable}
```

### Accuracy Retention

Optimization techniques often trade accuracy for efficiency. The acceptable accuracy drop depends on the application:

| Application | Acceptable Accuracy Drop |
|-------------|-------------------------|
| Safety-critical / financial risk | < 0.1% |
| Production ML systems | < 1–2% |
| Edge / mobile deployment | < 5% |

## Optimization Techniques Summary

| Technique | Size Reduction | Speed Gain | Accuracy Impact |
|-----------|---------------|------------|-----------------|
| Dynamic Quantization | 2–4× | 1.5–2× | Minimal |
| Static Quantization | 4× | 2–4× | Low |
| Pruning (30%) | 1.3× | Variable | Minimal |
| Knowledge Distillation | Variable | Variable | Low |
| Neural Architecture Search | Variable | Variable | Optimized |
| ONNX Optimization | 1.2–1.5× | 1.2–1.8× | None |
| TensorRT | 1× | 2–5× | Low (FP16) |

## Key Serialization Formats

### PyTorch Native (.pt, .pth)

The default serialization format using `torch.save()`:

```python
# Save model weights (recommended)
torch.save(model.state_dict(), "model.pt")

# Load weights into model
model = ModelClass()
model.load_state_dict(torch.load("model.pt", weights_only=True))
```

**Advantages**: Simple, preserves full PyTorch functionality, supports checkpointing for training resumption.

**Disadvantages**: Requires Python and PyTorch to load, not optimized for inference.

### TorchScript (.pt)

Serializable representation that can run independently from Python:

```python
# Trace model with example input (for models without control flow)
traced_model = torch.jit.trace(model, example_input)
traced_model.save("model_traced.pt")

# Script model (handles dynamic control flow)
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")

# Load without Python runtime dependency
loaded_model = torch.jit.load("model_traced.pt")
```

**Advantages**: Language-independent, optimized execution, C++ deployment support.

**Disadvantages**: Some Python features not supported, debugging can be harder.

### ONNX (.onnx)

Open Neural Network Exchange format for cross-framework compatibility:

```python
import torch.onnx

torch.onnx.export(
    model,
    example_input,
    "model.onnx",
    export_params=True,
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)
```

**Advantages**: Framework-agnostic, broad runtime support, rich optimization ecosystem.

**Disadvantages**: Not all PyTorch operations supported, version compatibility considerations.

### TensorRT (.engine)

NVIDIA's inference optimizer for maximum GPU performance:

```python
import torch_tensorrt

trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input(shape=[1, 3, 224, 224])],
    enabled_precisions={torch.float16},
)
```

**Advantages**: Maximum GPU inference speed, automatic kernel tuning and fusion.

**Disadvantages**: NVIDIA GPUs only, hardware-specific builds, calibration required for INT8.

### Format Selection Guide

| Scenario | Recommended Format | Rationale |
|----------|-------------------|-----------|
| Development / Prototyping | PyTorch (.pt) | Full flexibility, easy debugging |
| Training Resumption | State Dict + Checkpoint | Optimizer state, epoch tracking |
| Production (C++ / Mobile) | TorchScript | Python-independent, optimized |
| Cross-Framework | ONNX | Maximum compatibility |
| Maximum GPU Speed | TensorRT | Hardware-optimized kernels |
| Edge Devices | ONNX + Runtime | Lightweight, portable |
| Cloud Deployment | TorchScript or ONNX | Balanced speed and compatibility |
| Financial Trading Systems | TensorRT or ONNX | Ultra-low latency requirements |

## Serving Architectures

### Online Inference (REST / gRPC)

Real-time prediction via HTTP or gRPC endpoints:

- **REST API**: Simple HTTP endpoints using Flask or FastAPI; ideal for prototyping and moderate-traffic services
- **gRPC**: Binary protocol with lower overhead; preferred for inter-service communication and latency-sensitive applications
- **Use cases**: Recommendation systems, fraud detection, credit scoring, chatbots

### Managed Serving Frameworks

Production-grade serving with built-in scaling and monitoring:

- **TorchServe**: PyTorch-native model serving with multi-model management and custom handlers
- **Triton Inference Server**: NVIDIA's multi-framework server supporting dynamic batching, model ensembles, and concurrent execution
- **BentoML**: Framework-agnostic serving with packaging, versioning, and deployment automation

### Batch Inference

Process large datasets offline:

- **Throughput-focused**: Maximize samples per second with large batch sizes
- **Use cases**: Daily portfolio rebalancing, report generation, risk model recalibration

### Streaming Inference

Process continuous data streams:

- **Low-latency, high-throughput**: Balance both requirements under sustained load
- **Use cases**: Real-time trading signals, market surveillance, tick-level feature computation

## Distributed Training Strategies

Training large models requires distributing computation across multiple GPUs and nodes:

| Strategy | What is Distributed | Best For |
|----------|-------------------|----------|
| Data Parallel (DP) | Data batches | Single-node multi-GPU, prototyping |
| Distributed Data Parallel (DDP) | Data batches + gradient sync | Multi-node training, production |
| Model Parallel | Model layers | Models too large for single GPU |
| Pipeline Parallel | Model stages | Very deep models with sequential structure |
| FSDP | Parameters, gradients, optimizer states | Large models with memory constraints |
| DeepSpeed | All of the above + offloading | Extreme-scale training with ZeRO optimization |

## MLOps Lifecycle

Deploying a model is not a one-time event but a continuous lifecycle:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Experiment  │────▶│    Model     │────▶│   CI / CD    │────▶│  Production  │
│  Tracking    │     │  Registry    │     │  Pipelines   │     │  Monitoring  │
└──────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                      │
                                              ┌──────────────┐        │
                                              │  A/B Testing │◀───────┘
                                              └──────────────┘
```

- **Experiment Tracking**: Log hyperparameters, metrics, and artifacts using tools like MLflow, Weights & Biases, or Neptune
- **Model Registry**: Version, stage, and catalog models with metadata for reproducibility and governance
- **CI/CD**: Automate testing, validation, and deployment of model updates
- **Monitoring**: Track data drift, prediction drift, latency degradation, and system health in production
- **A/B Testing**: Compare model versions with statistical rigor before full rollout

## Finance-Specific Deployment

Quantitative finance imposes unique deployment requirements that go beyond standard ML serving:

- **Latency Requirements**: Trading systems demand sub-millisecond inference with deterministic tail latencies; even small delays can result in significant financial impact
- **Real-Time Systems**: Continuous processing of market data feeds with strict ordering guarantees and failover mechanisms
- **Backtesting Infrastructure**: Reproducible historical simulation that matches production logic exactly, avoiding look-ahead bias and survivorship bias
- **Production Pipelines**: End-to-end orchestration from data ingestion through feature computation, model inference, risk checks, and order generation

## Production Considerations

### Reliability Requirements

Production systems must handle:

- **Graceful degradation**: Fallback behavior when model fails or latency exceeds thresholds
- **Error handling**: Proper responses for invalid inputs, missing features, and out-of-distribution data
- **Health checks**: Readiness and liveness probes for orchestration systems
- **Rollback capability**: Quick revert to previous model versions with minimal downtime

### Scalability Patterns

As traffic grows, consider:

- **Horizontal scaling**: Multiple model replicas behind a load balancer
- **Vertical scaling**: Larger machines with more GPU memory
- **Auto-scaling**: Dynamic capacity adjustment based on demand patterns
- **Caching**: Store predictions for repeated or similar inputs

## Deployment Checklist

Before deploying a model to production:

- [ ] Model accuracy validated on held-out test set
- [ ] Serialization format chosen and export verified
- [ ] Optimization techniques applied and accuracy retention confirmed
- [ ] Latency meets requirements on target hardware (p50, p95, p99)
- [ ] Throughput sufficient for expected peak load
- [ ] Model size acceptable for deployment environment
- [ ] Memory usage within available limits
- [ ] Distributed training validated (if applicable)
- [ ] Serving infrastructure configured and load-tested
- [ ] Cross-platform compatibility verified (if applicable)
- [ ] Error handling implemented for edge cases
- [ ] Health checks configured (readiness / liveness probes)
- [ ] Monitoring and logging configured (metrics, drift detection)
- [ ] Model registered with metadata and versioning
- [ ] CI/CD pipeline established for model updates
- [ ] Rollback strategy defined and tested
- [ ] A/B testing framework prepared
- [ ] Finance-specific validations completed (if applicable)

## Chapter Organization

This chapter covers deployment in depth across seven sections:

### 40.1 Serialization

Saving, loading, and exporting models for deployment:

- **[Model Saving](serialization/saving.md)** — Fundamentals of persisting PyTorch models to disk
- **[State Dict](serialization/state_dict.md)** — Working with model state dictionaries for flexible weight management
- **[Checkpointing](serialization/checkpointing.md)** — Saving and resuming training with full optimizer and scheduler state
- **[ONNX Export](serialization/onnx.md)** — Converting models to ONNX format for cross-platform deployment
- **[TorchScript](serialization/torchscript.md)** — Compiling models for optimized execution without Python

### 40.2 Optimization

Reducing model size and improving inference speed:

- **[Quantization](optimization/quantization.md)** — Reducing precision from FP32 to INT8/FP16 for faster inference
- **[Pruning](optimization/pruning.md)** — Removing redundant weights to shrink model size
- **[Knowledge Distillation](optimization/distillation.md)** — Training smaller student models to mimic larger teachers
- **[Neural Architecture Search](optimization/nas.md)** — Automated discovery of efficient model architectures

### 40.3 Inference

Efficient prediction across diverse hardware and deployment patterns:

- **[Batch Inference](inference/batch.md)** — Processing large datasets with optimized batching strategies
- **[Streaming Inference](inference/streaming.md)** — Continuous prediction on real-time data streams
- **[GPU Optimization](inference/gpu.md)** — Maximizing GPU utilization with kernel fusion, mixed precision, and memory optimization
- **[CPU Optimization](inference/cpu.md)** — Efficient inference on CPU with vectorization and thread tuning
- **[Memory Management](inference/memory.md)** — Controlling memory allocation, fragmentation, and peak usage

### 40.4 Serving

Building production-ready inference APIs and infrastructure:

- **[REST API](serving/rest_api.md)** — HTTP-based model serving with Flask and FastAPI
- **[gRPC](serving/grpc.md)** — High-performance binary protocol for low-latency serving
- **[TorchServe](serving/torchserve.md)** — PyTorch-native serving with model management and scaling
- **[Triton](serving/triton.md)** — NVIDIA's multi-framework inference server with dynamic batching
- **[BentoML](serving/bentoml.md)** — Framework-agnostic model packaging and deployment

### 40.5 Distributed Training

Scaling training across multiple GPUs and nodes:

- **[Data Parallel](distributed/data_parallel.md)** — Simple multi-GPU training with replicated models
- **[Distributed Data Parallel](distributed/ddp.md)** — Efficient multi-node training with gradient synchronization
- **[Model Parallel](distributed/model_parallel.md)** — Splitting large models across devices
- **[Pipeline Parallel](distributed/pipeline.md)** — Overlapping computation and communication for deep models
- **[FSDP](distributed/fsdp.md)** — Fully Sharded Data Parallel for memory-efficient large model training
- **[DeepSpeed](distributed/deepspeed.md)** — Microsoft's library for extreme-scale training with ZeRO optimization

### 40.6 MLOps

Managing the model lifecycle from experiment to production:

- **[Experiment Tracking](mlops/tracking.md)** — Logging parameters, metrics, and artifacts across runs
- **[Model Registry](mlops/registry.md)** — Versioning, staging, and cataloging models
- **[CI/CD](mlops/cicd.md)** — Automated testing and deployment pipelines for models
- **[Monitoring](mlops/monitoring.md)** — Tracking model performance, data drift, and system health
- **[A/B Testing](mlops/ab_testing.md)** — Statistical comparison of model versions in production

### 40.7 Finance Deployment

Meeting the unique requirements of quantitative finance systems:

- **[Latency Requirements](finance/latency.md)** — Achieving sub-millisecond inference for trading applications
- **[Real-Time Systems](finance/realtime.md)** — Processing live market data with strict ordering and failover
- **[Backtesting Infrastructure](finance/backtesting.md)** — Reproducible historical simulation matching production logic
- **[Production Pipelines](finance/pipelines.md)** — End-to-end orchestration from data ingestion to order generation

## Learning Objectives

After completing this chapter, you will be able to:

1. Serialize PyTorch models using state dicts, checkpoints, TorchScript, and ONNX export
2. Apply quantization, pruning, knowledge distillation, and NAS to optimize models for production
3. Design efficient inference pipelines for batch, streaming, GPU, and CPU deployment
4. Build production-ready serving APIs using REST, gRPC, TorchServe, Triton, and BentoML
5. Scale model training with data parallelism, model parallelism, FSDP, and DeepSpeed
6. Implement MLOps practices including experiment tracking, model registries, CI/CD, and monitoring
7. Deploy ML models in quantitative finance settings with appropriate latency, reliability, and reproducibility guarantees

## References

1. PyTorch Deployment Documentation: https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html
2. ONNX Runtime Performance Tuning: https://onnxruntime.ai/docs/performance/
3. TensorRT Developer Guide: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/
4. TorchServe Documentation: https://pytorch.org/serve/
5. Triton Inference Server: https://docs.nvidia.com/deeplearning/triton-inference-server/
6. BentoML Documentation: https://docs.bentoml.com/
7. DeepSpeed Documentation: https://www.deepspeed.ai/
8. PyTorch FSDP Tutorial: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
9. MLflow Documentation: https://mlflow.org/docs/latest/index.html
10. MLOps: Continuous Delivery for Machine Learning: https://ml-ops.org/
11. Google ML Best Practices: https://developers.google.com/machine-learning/guides/rules-of-ml
