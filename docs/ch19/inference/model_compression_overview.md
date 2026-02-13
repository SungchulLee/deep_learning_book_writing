# Module 65: Model Compression

## Overview
This module covers essential techniques for compressing deep neural networks to reduce memory footprint, computational cost, and inference latency while maintaining acceptable accuracy. These techniques are critical for deploying models on resource-constrained devices (mobile, edge, IoT) and production environments.

## Learning Objectives
- Understand the computational and memory bottlenecks in deep neural networks
- Master quantization techniques (post-training and quantization-aware training)
- Implement various pruning strategies (magnitude-based, structured, unstructured)
- Apply knowledge distillation to compress models
- Evaluate the accuracy-efficiency trade-offs in compressed models

## Prerequisites
- Module 20: Feedforward Networks
- Module 23: Convolutional Neural Networks
- Module 14: Loss Functions
- Module 15: Optimizers
- Basic understanding of PyTorch nn.Module

---

## Part 1: Theoretical Foundations

### 1.1 Why Model Compression?

Modern deep neural networks face three critical challenges:

1. **Memory Footprint**: 
   - ResNet-50: ~100MB (25M parameters × 4 bytes/float32)
   - GPT-3: ~700GB (175B parameters)
   
2. **Computational Cost**:
   - Mobile devices: Limited FLOPS, battery constraints
   - Cloud inference: Cost scales with compute time
   
3. **Latency Requirements**:
   - Real-time applications need <100ms inference
   - Autonomous vehicles, medical diagnosis require immediate responses

### 1.2 Quantization

**Definition**: Reducing the numerical precision of weights and activations from floating-point (32-bit) to lower bit-width representations (8-bit, 4-bit, or even binary).

#### Mathematical Formulation

For a tensor **x** with floating-point values, quantization maps:

```
x_float ∈ ℝ → x_quant ∈ {0, 1, ..., 2^b - 1}
```

where `b` is the bit-width.

**Quantization function**:
```
x_quant = round((x_float - zero_point) / scale)

where:
scale = (x_max - x_min) / (2^b - 1)
zero_point = round(-x_min / scale)
```

**Dequantization** (for computation):
```
x_dequant = scale × x_quant + zero_point
```

#### Types of Quantization

1. **Post-Training Quantization (PTQ)**:
   - Quantize a pre-trained model without retraining
   - Faster but may lose more accuracy
   - Suitable for models with redundancy

2. **Quantization-Aware Training (QAT)**:
   - Simulate quantization during training
   - Learns weights robust to quantization errors
   - Better accuracy but requires retraining

#### Theoretical Benefits

- **Memory reduction**: 4× for FP32→INT8, 8× for FP32→INT4
- **Speed improvement**: INT8 operations are faster on specialized hardware
- **Energy efficiency**: Lower precision = lower power consumption

**Accuracy Trade-off**:
```
Δ_accuracy ∝ quantization_error²
where quantization_error = x_float - x_dequant
```

### 1.3 Pruning

**Definition**: Removing redundant or less important weights/neurons from the network to create sparse models.

#### Types of Pruning

1. **Unstructured Pruning**:
   - Remove individual weights based on magnitude
   - Higher sparsity achievable (90%+)
   - Requires sparse tensor operations for speedup
   
2. **Structured Pruning**:
   - Remove entire filters, channels, or layers
   - Lower sparsity but works with standard operations
   - Direct speedup on regular hardware

#### Mathematical Framework

For a weight tensor **W**, define importance score:
```
importance(w_i) = |w_i|  (magnitude-based)
                = |∂L/∂w_i|  (gradient-based)
                = w_i² × (∂L/∂w_i)²  (Hessian approximation)
```

**Pruning criterion**:
```
Prune w_i if importance(w_i) < threshold_τ
```

**Sparsity ratio**:
```
sparsity = (# pruned parameters) / (# total parameters)
```

#### Iterative Magnitude Pruning (IMP)

The Lottery Ticket Hypothesis suggests:
1. Train network to convergence
2. Prune p% of smallest magnitude weights
3. Reset remaining weights to initialization
4. Repeat training

**Convergence theorem** (informal):
```
∃ subnetwork S ⊂ Network that achieves:
accuracy(S) ≥ accuracy(Network) - ε
with |S| << |Network|
```

### 1.4 Knowledge Distillation

**Definition**: Training a smaller "student" model to mimic a larger "teacher" model by matching output distributions.

#### Mathematical Formulation

**Soft targets** from teacher:
```
p_i^teacher = softmax(z_i / T)
where T is temperature (T > 1 makes distribution softer)
```

**Student training objective**:
```
L_total = α × L_hard + (1-α) × L_soft

L_hard = CrossEntropy(y_student, y_true)
L_soft = KL_divergence(p_student, p_teacher) × T²
```

The T² term compensates for magnitude scaling in gradients.

**Intuition**:
- Hard labels: {cat: 1, dog: 0, car: 0}
- Soft labels: {cat: 0.9, dog: 0.08, car: 0.02}
- Soft labels provide richer information about class relationships

#### Why It Works

1. **Dark knowledge**: Teacher's incorrect probabilities encode similarity structure
2. **Regularization**: Prevents student from overfitting to hard labels
3. **Generalization**: Soft targets act as label smoothing

---

## Part 2: Implementation Structure

### Module Organization

```
65_model_compression/
├── README.md                          # This file
├── requirements.txt                   # Dependencies
├── utils.py                           # Shared utilities
│
├── 01_quantization_basics.py         # BEGINNER: Post-training quantization
├── 02_pruning_basics.py               # BEGINNER: Magnitude-based pruning
├── 03_knowledge_distillation.py       # BEGINNER: Basic distillation
│
├── 04_quantization_aware_training.py  # INTERMEDIATE: QAT
├── 05_structured_pruning.py           # INTERMEDIATE: Filter/channel pruning
├── 06_iterative_pruning.py            # INTERMEDIATE: IMP algorithm
│
├── 07_mixed_precision_quantization.py # ADVANCED: Per-layer bit-widths
├── 08_neural_architecture_search.py   # ADVANCED: AutoML for compression
└── 09_combined_compression.py         # ADVANCED: Pruning + Quantization + Distillation
```

### Difficulty Levels

**BEGINNER** (01-03):
- Focus on understanding core concepts
- Simple implementations with detailed comments
- Small models (LeNet, simple CNNs)
- Clear accuracy vs compression trade-offs

**INTERMEDIATE** (04-06):
- More sophisticated techniques
- Training integration
- Medium models (ResNet-18, MobileNet)
- Fine-tuning and recovery strategies

**ADVANCED** (07-09):
- State-of-the-art methods
- Combined compression pipelines
- Large models (ResNet-50, Vision Transformers)
- Production-ready implementations

---

## Part 3: Practical Considerations

### Hardware Support

| Hardware | INT8 | INT4 | FP16 | Sparse |
|----------|------|------|------|--------|
| CPU      | ✓    | ✗    | ✗    | ✗      |
| GPU      | ✓    | ✓    | ✓    | Limited|
| TPU      | ✓    | ✗    | ✓    | ✗      |
| ARM/Mobile| ✓   | ✓    | ✓    | ✗      |

### Compression Guidelines

1. **Always measure**:
   - Accuracy drop
   - Memory reduction
   - Inference speedup (not just theoretical)
   
2. **Start conservative**:
   - Begin with 8-bit quantization
   - Prune 50% before aggressive compression
   - Use distillation to recover accuracy

3. **Layer sensitivity**:
   - First and last layers are most sensitive
   - Middle layers can handle aggressive compression
   - Batch norm layers should often remain FP32

---

## Part 4: Evaluation Metrics

### Model Size
```python
size_mb = (num_parameters × bytes_per_parameter) / (1024²)
compression_ratio = size_original / size_compressed
```

### Inference Time
```python
latency = forward_pass_time (averaged over multiple runs)
speedup = latency_original / latency_compressed
```

### Accuracy
```python
accuracy_drop = accuracy_original - accuracy_compressed
acceptable_drop < 1-2% for most applications
```

### Efficiency Score
```python
efficiency = accuracy / (latency × model_size)
```

---

## References

1. **Quantization**:
   - Jacob et al. "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (2018)
   - Banner et al. "Post training 4-bit quantization of convolutional networks for rapid-deployment" (2019)

2. **Pruning**:
   - Han et al. "Learning both Weights and Connections for Efficient Neural Networks" (2015)
   - Frankle & Carbin "The Lottery Ticket Hypothesis" (2019)

3. **Knowledge Distillation**:
   - Hinton et al. "Distilling the Knowledge in a Neural Network" (2015)
   - Romero et al. "FitNets: Hints for Thin Deep Nets" (2015)

4. **Surveys**:
   - Cheng et al. "Model Compression and Acceleration for Deep Neural Networks" (2020)
   - Gholami et al. "A Survey of Quantization Methods for Efficient Neural Network Inference" (2021)

---

## Usage

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run beginner examples
python 01_quantization_basics.py
python 02_pruning_basics.py
python 03_knowledge_distillation.py

# Run intermediate examples
python 04_quantization_aware_training.py
python 05_structured_pruning.py

# Run advanced examples
python 09_combined_compression.py
```

### Jupyter Notebooks
All scripts can be run in Jupyter notebooks for interactive exploration:
```bash
jupyter notebook
```

---

## Learning Path

**Week 1**: Quantization Theory + PTQ (01)
**Week 2**: Pruning Theory + Magnitude Pruning (02)
**Week 3**: Knowledge Distillation (03)
**Week 4**: QAT + Structured Pruning (04-05)
**Week 5**: Advanced Techniques (06-07)
**Week 6**: Combined Methods + Projects (08-09)

---

## Exercises

1. **Quantization Analysis**: Measure how quantization error propagates through layers
2. **Pruning Sensitivity**: Identify which layers can be pruned most aggressively
3. **Distillation Experiments**: Compare different temperature values and α weights
4. **Compression Pipeline**: Build end-to-end pipeline for your favorite model
5. **Mobile Deployment**: Deploy compressed model to mobile device (iOS/Android)

---

## Additional Resources

- PyTorch Quantization Tutorial: https://pytorch.org/docs/stable/quantization.html
- TensorFlow Model Optimization: https://www.tensorflow.org/model_optimization
- ONNX Runtime Quantization: https://onnxruntime.ai/docs/performance/quantization.html
- Papers With Code (Model Compression): https://paperswithcode.com/task/model-compression
