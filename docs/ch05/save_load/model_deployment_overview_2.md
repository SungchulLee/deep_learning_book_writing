# Model Deployment Basics: ONNX, Quantization & Inference Optimization

This repository contains comprehensive Python examples for deploying machine learning models in production, covering ONNX conversion, model quantization, and inference optimization techniques.

## 📚 Contents

1. **`1_onnx_basics.py`** - ONNX conversion and inference fundamentals
2. **`2_quantization.py`** - Model quantization techniques (static, dynamic, QAT)
3. **`3_inference_optimization.py`** - Performance optimization strategies
4. **`4_deployment_pipeline.py`** - Complete end-to-end deployment workflow

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running Examples

```bash
# ONNX basics
python 1_onnx_basics.py

# Quantization techniques
python 2_quantization.py

# Inference optimization
python 3_inference_optimization.py

# Complete deployment pipeline
python 4_deployment_pipeline.py
```

## 📖 Module Overview

### 1. ONNX Basics (`1_onnx_basics.py`)

Learn how to work with ONNX format for model deployment:

**Key Topics:**
- Converting PyTorch models to ONNX
- Verifying ONNX model validity
- Running ONNX inference with ONNXRuntime
- Comparing PyTorch vs ONNX outputs
- Optimizing ONNX models

**Example Usage:**
```python
from 1_onnx_basics import convert_pytorch_to_onnx, ONNXInferenceSession

# Convert model
convert_pytorch_to_onnx(model, dummy_input, "model.onnx")

# Run inference
session = ONNXInferenceSession("model.onnx")
output = session.predict(input_data)
```

**When to Use ONNX:**
- ✅ Cross-platform deployment
- ✅ Framework interoperability
- ✅ Hardware acceleration (CPU, GPU, NPU)
- ✅ Production environments

---

### 2. Quantization (`2_quantization.py`)

Master model quantization for reduced size and faster inference:

#### Dynamic Quantization
- **What:** Quantizes weights at load time, activations on-the-fly
- **Best for:** LSTMs, Transformers, variable input sizes
- **Benefits:** 2-4x smaller, 1.5-2x faster, no calibration needed

```python
from 2_quantization import dynamic_quantization_demo

quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear, nn.LSTM},
    dtype=torch.qint8
)
```

#### Static Quantization
- **What:** Quantizes both weights and activations
- **Best for:** CNNs, fixed input sizes
- **Benefits:** 4x smaller, 2-4x faster, maximum performance
- **Requires:** Calibration dataset

```python
from 2_quantization import static_quantization_demo

# Requires calibration data
quantized_model = static_quantization_demo(model, calibration_loader, example_input)
```

#### Quantization-Aware Training (QAT)
- **What:** Train with quantization simulation
- **Best for:** Models with accuracy loss from PTQ
- **Benefits:** Better accuracy retention, production-ready models

**Quantization Decision Tree:**
```
Start
├─ Is it an LSTM/Transformer? → Dynamic Quantization
├─ Is it a CNN with fixed inputs? 
│  ├─ Accuracy drop < 2%? → Static Quantization
│  └─ Accuracy drop > 2%? → QAT
└─ Quick deployment needed? → Dynamic Quantization
```

---

### 3. Inference Optimization (`3_inference_optimization.py`)

Advanced techniques for maximizing inference performance:

#### Batch Processing
- Accumulate requests and process together
- 3-10x throughput improvement
- Essential for high-volume deployments

```python
from 3_inference_optimization import DynamicBatcher

batcher = DynamicBatcher(model, max_batch_size=32, timeout_ms=100)
output = batcher.predict(input_data)
```

#### Model Pruning
- **Unstructured:** Remove individual weights (30-50% sparsity)
- **Structured:** Remove entire channels/neurons (actual speedup)

```python
from 3_inference_optimization import prune_model

pruned_model = prune_model(model, amount=0.3)  # Remove 30% of weights
```

#### TorchScript Compilation
- Removes Python overhead
- Enables graph optimizations
- 1.5-2x speedup

```python
from 3_inference_optimization import torchscript_optimization

optimized = torchscript_optimization(model, example_input)
```

#### Mixed Precision (FP16)
- 2-3x speedup on modern GPUs
- Minimal accuracy impact
- Requires CUDA-capable GPU

```python
with torch.cuda.amp.autocast():
    output = model(input)
```

#### Profiling & Benchmarking
- Identify bottlenecks
- Measure latency and throughput
- Optimize critical paths

```python
from 3_inference_optimization import profile_model, benchmark_comprehensive

profile_model(model, input_data)
results = benchmark_comprehensive(model, input_size)
```

---

### 4. Deployment Pipeline (`4_deployment_pipeline.py`)

Complete end-to-end deployment workflow:

**Pipeline Steps:**
1. **Train** - Train or fine-tune your model
2. **Optimize** - Apply quantization and pruning
3. **Convert** - Export to ONNX format
4. **Benchmark** - Measure performance metrics
5. **Deploy** - Production-ready artifacts

```python
from 4_deployment_pipeline import DeploymentPipeline

pipeline = DeploymentPipeline(model)
pipeline.train_model(train_loader, val_loader)
pipeline.apply_quantization(val_loader)
onnx_path = pipeline.convert_to_onnx(input_shape=(3, 32, 32))
pipeline.optimize_onnx(onnx_path)
pipeline.save_deployment_report()
```

**Output:**
- Optimized model files
- Performance benchmarks
- Deployment recommendations
- Comprehensive metrics report

---

## 🎯 Best Practices

### Model Size Optimization
| Technique | Size Reduction | Accuracy Impact | Speed Gain |
|-----------|---------------|-----------------|------------|
| Dynamic Quantization | 2-4x | Minimal | 1.5-2x |
| Static Quantization | 4x | Low | 2-4x |
| Pruning (30%) | 1.3x | Minimal | Variable |
| ONNX Optimization | 1.2-1.5x | None | 1.2-1.8x |

### Inference Speed Optimization
1. **Batching** - Accumulate and process together (3-10x throughput)
2. **Quantization** - INT8 operations (2-4x faster)
3. **TorchScript** - Remove Python overhead (1.5-2x faster)
4. **Mixed Precision** - FP16 on GPU (2-3x faster)
5. **Model Pruning** - Fewer operations (1.2-2x faster)

### Deployment Checklist
- [ ] Model accuracy validated on test set
- [ ] Latency meets requirements (< target ms)
- [ ] Throughput sufficient for load (> target QPS)
- [ ] Model size acceptable (< target MB)
- [ ] Memory usage within limits
- [ ] Cross-platform compatibility verified
- [ ] Error handling implemented
- [ ] Monitoring and logging configured

---

## 🔧 Common Issues & Solutions

### Issue: Accuracy Drop After Quantization
**Solutions:**
1. Use static quantization with good calibration data
2. Try quantization-aware training (QAT)
3. Increase calibration dataset size
4. Use per-channel quantization
5. Skip quantizing sensitive layers

### Issue: ONNX Conversion Fails
**Solutions:**
1. Use supported PyTorch operations
2. Avoid dynamic control flow
3. Set appropriate opset version
4. Simplify complex operations
5. Check input/output shapes

### Issue: Slow Inference Despite Optimization
**Solutions:**
1. Verify correct execution provider (CPU vs CUDA)
2. Increase batch size for better GPU utilization
3. Profile to find bottlenecks
4. Check memory transfers (CPU-GPU)
5. Consider model architecture changes

### Issue: Large Model Size
**Solutions:**
1. Apply quantization (4x reduction)
2. Use pruning (30-50% reduction)
3. Reduce model architecture
4. Knowledge distillation to smaller model
5. Remove unnecessary layers

---

## 📊 Performance Metrics

### Latency
- **Definition:** Time for single inference
- **Target:** < 100ms for real-time, < 10ms for interactive
- **Measure:** Average over 100+ runs after warmup

### Throughput
- **Definition:** Samples processed per second
- **Target:** Depends on expected load
- **Improve:** Batching, quantization, pruning

### Model Size
- **Definition:** Disk space for model weights
- **Target:** < 100MB for mobile, < 1GB for server
- **Reduce:** Quantization, pruning, distillation

### Accuracy
- **Definition:** Task-specific metric (accuracy, F1, etc.)
- **Target:** Minimal drop from baseline (< 1-2%)
- **Validate:** Full test set evaluation

---

## 🌐 Production Deployment Options

### 1. ONNX Runtime
```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
output = session.run(None, {"input": input_data})
```
**Pros:** Fast, cross-platform, multiple backends
**Cons:** Limited to ONNX format

### 2. TorchServe
```bash
torch-model-archiver --model-name model --version 1.0 \
    --serialized-file model.pt --handler image_classifier
torchserve --start --model-store model_store
```
**Pros:** Built for PyTorch, production-ready, RESTful API
**Cons:** PyTorch-specific

### 3. FastAPI + Docker
```python
from fastapi import FastAPI
app = FastAPI()

@app.post("/predict")
async def predict(data: ImageData):
    return {"prediction": model(data)}
```
**Pros:** Flexible, customizable, easy scaling
**Cons:** Requires more setup

### 4. Cloud Platforms
- **AWS SageMaker:** Managed deployment
- **Azure ML:** Enterprise integration  
- **Google Vertex AI:** AutoML + custom models
- **Hugging Face Spaces:** Easy sharing

---

## 📚 Additional Resources

### Documentation
- [ONNX Documentation](https://onnx.ai/onnx/)
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
- [ONNX Runtime Performance Tuning](https://onnxruntime.ai/docs/performance/)

### Tutorials
- [PyTorch Production Guide](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html)
- [Quantization Tutorial](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)
- [ONNX Optimization](https://github.com/onnx/optimizer)

### Tools
- **Netron:** Visualize ONNX models
- **ONNX Simplifier:** Optimize ONNX graphs
- **PyTorch Profiler:** Performance analysis
- **TensorBoard:** Metrics visualization

---

## 🤝 Contributing

Feel free to extend these examples with:
- Additional optimization techniques
- More deployment scenarios
- Platform-specific optimizations
- Real-world use cases

---

## 📝 License

These examples are provided for educational purposes. Feel free to use and modify for your projects.

---

## 🎓 Key Takeaways

1. **ONNX** enables cross-platform deployment and hardware acceleration
2. **Quantization** reduces model size 2-4x with minimal accuracy loss
3. **Optimization** techniques (batching, pruning, compilation) improve speed 2-10x
4. **Complete pipeline** from training to deployment ensures production readiness
5. **Benchmarking** is essential to validate improvements
6. **Trade-offs** between size, speed, and accuracy must be carefully balanced

---

**Happy Deploying! 🚀**
