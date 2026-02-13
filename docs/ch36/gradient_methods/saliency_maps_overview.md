# Module 60: Saliency Maps for Neural Network Interpretability

## Overview
This module provides a comprehensive introduction to **saliency maps** - visualization techniques that help interpret neural network decisions by highlighting which parts of an input are most important for the model's predictions.

## Learning Objectives
By completing this module, students will:
- Understand the mathematical foundations of gradient-based saliency methods
- Implement various saliency map techniques from scratch
- Apply saliency methods to real image classification tasks
- Compare and evaluate different interpretability approaches
- Understand the strengths and limitations of each method

## Mathematical Foundations

### 1. Vanilla Gradient (Basic Saliency)
The simplest saliency map uses the gradient of the output with respect to the input:

**Formula:**
```
S(x) = |∂y_c/∂x|
```
Where:
- `x`: input image
- `y_c`: output score for class c
- `S(x)`: saliency map

**Interpretation:** Pixels with high gradient magnitude have strong influence on the prediction.

### 2. Gradient × Input
Multiplies the gradient by the input to account for input magnitude:

**Formula:**
```
S(x) = x ⊙ (∂y_c/∂x)
```
Where `⊙` denotes element-wise multiplication.

**Advantage:** Considers both sensitivity (gradient) and actual input values.

### 3. Integrated Gradients
Accumulates gradients along a path from a baseline to the input:

**Formula:**
```
IG(x) = (x - x') ⊙ ∫₀¹ (∂f(x' + α(x - x'))/∂x) dα
```
Where:
- `x'`: baseline input (usually zeros or blurred image)
- `α`: interpolation coefficient [0, 1]
- Integration is approximated by Riemann sum

**Properties:**
- Satisfies axioms: Sensitivity and Implementation Invariance
- More theoretically grounded than vanilla gradients

### 4. SmoothGrad
Averages gradients with added Gaussian noise to reduce visual noise:

**Formula:**
```
SG(x) = (1/n) Σᵢ (∂y_c/∂(x + N(0, σ²)))
```
Where:
- `n`: number of noisy samples
- `N(0, σ²)`: Gaussian noise with std σ

**Advantage:** Produces cleaner, more interpretable visualizations.

### 5. Grad-CAM (Gradient-weighted Class Activation Mapping)
Uses gradients of target class flowing into final convolutional layer:

**Formula:**
```
α_k = (1/Z) Σᵢ Σⱼ (∂y_c/∂A_k^(i,j))    [Global average pooling of gradients]
L_Grad-CAM = ReLU(Σ_k α_k A_k)           [Weighted combination of feature maps]
```
Where:
- `A_k`: k-th feature map in last conv layer
- `α_k`: importance weight for feature map k
- `Z`: normalization constant (number of pixels)

**Advantages:**
- Class-discriminative
- Localizes regions in coarse resolution
- Works for any CNN architecture

### 6. Guided Backpropagation
Modifies ReLU backward pass to only backpropagate positive gradients:

**Forward pass:**
```
y = ReLU(x) = max(0, x)
```

**Standard backward pass:**
```
∂L/∂x = (∂L/∂y) · 1(x > 0)
```

**Guided backward pass:**
```
∂L/∂x = (∂L/∂y) · 1(x > 0) · 1(∂L/∂y > 0)
```

**Effect:** Produces sharper, cleaner visualizations by suppressing negative gradients.

### 7. Guided Grad-CAM
Combines Grad-CAM and Guided Backpropagation:

**Formula:**
```
Guided Grad-CAM = Guided Backprop ⊙ Upsample(Grad-CAM)
```

**Advantages:**
- High-resolution (from Guided Backprop)
- Class-discriminative (from Grad-CAM)
- Best of both worlds

## Module Structure

### Beginner Level (01-03)
1. **01_vanilla_gradient_saliency.py**
   - Basic gradient-based saliency
   - Simple visualization
   - Single image example

2. **02_gradient_input_saliency.py**
   - Gradient × Input method
   - Comparison with vanilla gradients
   - Multiple examples

3. **03_smoothgrad.py**
   - Noise-smoothed gradients
   - Hyperparameter effects
   - Visual quality improvement

### Intermediate Level (04-06)
4. **04_integrated_gradients.py**
   - Path integration method
   - Baseline selection strategies
   - Theoretical properties

5. **05_gradcam.py**
   - Class activation mapping
   - Convolutional layer selection
   - Multi-class visualization

6. **06_guided_backpropagation.py**
   - Modified ReLU gradients
   - High-resolution saliency
   - Custom backward hooks

### Advanced Level (07-09)
7. **07_guided_gradcam.py**
   - Combining multiple methods
   - Best practices
   - Production-ready implementation

8. **08_comparative_analysis.py**
   - All methods side-by-side
   - Quantitative evaluation
   - Method selection guidelines

9. **09_advanced_techniques.py**
   - Layer-wise Relevance Propagation (LRP) basics
   - Attention rollout
   - Custom architectures

### Utility Module
10. **utils.py**
    - Image preprocessing
    - Visualization helpers
    - Common functions

## Prerequisites
- **Module 02**: Tensors and autograd
- **Module 04**: Gradients
- **Module 19**: Activation functions
- **Module 20**: Feedforward networks
- **Module 23**: Convolutional Neural Networks
- **Module 43**: Model interpretability (overview)

## Installation Requirements
```bash
pip install torch torchvision matplotlib numpy pillow
```

## Quick Start
```python
# Example: Generate vanilla gradient saliency map
from torchvision import models
import torch

# Load pretrained model
model = models.resnet50(pretrained=True)
model.eval()

# Load and preprocess image
from utils import load_image, preprocess_image
image = load_image('dog.jpg')
input_tensor = preprocess_image(image)
input_tensor.requires_grad = True

# Forward pass
output = model(input_tensor)
class_idx = output.argmax(dim=1)

# Backward pass
model.zero_grad()
output[0, class_idx].backward()

# Get saliency map
saliency = input_tensor.grad.abs().max(dim=1)[0]

# Visualize
from utils import visualize_saliency
visualize_saliency(image, saliency)
```

## Key Concepts

### When to Use Each Method?

| Method | Best For | Limitations |
|--------|----------|-------------|
| Vanilla Gradient | Quick debugging, gradient flow analysis | Noisy, not class-discriminative |
| Gradient × Input | Considering input magnitude | Still noisy |
| SmoothGrad | Clean visualizations | Computationally expensive |
| Integrated Gradients | Theoretical guarantees, faithful attribution | Requires baseline selection |
| Grad-CAM | Localization in images, CNNs | Coarse resolution |
| Guided Backprop | High-resolution details | Not class-discriminative alone |
| Guided Grad-CAM | Best overall visualization | Requires CNN architecture |

### Evaluation Metrics

1. **Insertion/Deletion Curves**: Remove pixels in order of importance
2. **Pointing Game**: Check if saliency highlights ground truth object
3. **Human Agreement**: Correlation with human attention maps
4. **Sensitivity-n**: Measure attribution completeness

## Applications

1. **Model Debugging**: Identify what features the model uses
2. **Bias Detection**: Check for spurious correlations
3. **Trust Building**: Explain predictions to users
4. **Model Improvement**: Guide data augmentation strategies
5. **Scientific Discovery**: Understand learned representations

## Common Pitfalls

1. **Over-interpretation**: Saliency ≠ causality
2. **Architecture dependency**: Some methods only work with specific architectures
3. **Adversarial robustness**: Saliency maps can be manipulated
4. **Resolution mismatch**: Grad-CAM gives coarse localization
5. **Baseline selection**: Integrated Gradients sensitive to baseline choice

## Further Reading

### Papers
1. **Simonyan et al. (2013)**: "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps"
2. **Sundararajan et al. (2017)**: "Axiomatic Attribution for Deep Networks" (Integrated Gradients)
3. **Smilkov et al. (2017)**: "SmoothGrad: removing noise by adding noise"
4. **Selvaraju et al. (2017)**: "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
5. **Springenberg et al. (2015)**: "Striving for Simplicity: The All Convolutional Net" (Guided Backprop)

### Resources
- Captum: PyTorch interpretability library
- TensorFlow Model Analysis
- SHAP: Unified approach to explaining predictions

## Learning Path

### Week 1: Foundations
- Complete beginner modules (01-03)
- Understand gradient computation
- Practice with simple models

### Week 2: Advanced Methods
- Complete intermediate modules (04-06)
- Compare different techniques
- Apply to pretrained models

### Week 3: Integration & Analysis
- Complete advanced modules (07-09)
- Conduct comparative studies
- Build interpretability pipeline

## Assessment Ideas

1. Implement saliency method from scratch
2. Compare methods on misclassified examples
3. Design evaluation protocol
4. Apply to domain-specific model (medical, etc.)
5. Identify model biases using saliency

## Directory Structure
```
saliency_maps/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── utils.py                           # Shared utilities
├── 01_vanilla_gradient_saliency.py   # Beginner
├── 02_gradient_input_saliency.py     # Beginner
├── 03_smoothgrad.py                  # Beginner
├── 04_integrated_gradients.py        # Intermediate
├── 05_gradcam.py                     # Intermediate
├── 06_guided_backpropagation.py      # Intermediate
├── 07_guided_gradcam.py              # Advanced
├── 08_comparative_analysis.py        # Advanced
├── 09_advanced_techniques.py         # Advanced
├── data/                             # Sample images
│   ├── dog.jpg
│   ├── cat.jpg
│   └── bird.jpg
└── outputs/                          # Generated visualizations
```

## Tips for Students

1. **Start simple**: Begin with vanilla gradients before advanced methods
2. **Visualize everything**: Compare saliency maps for correct vs incorrect predictions
3. **Try different models**: ResNet, VGG, EfficientNet behave differently
4. **Experiment with baselines**: For Integrated Gradients, try zeros, blur, random
5. **Check class discrimination**: Verify saliency changes for different classes
6. **Combine methods**: Guided Grad-CAM often gives best results

## Common Questions

**Q: Why are my saliency maps so noisy?**
A: Try SmoothGrad or Integrated Gradients. Vanilla gradients are inherently noisy.

**Q: Can I use these methods with transformers?**
A: Some methods (vanilla gradient, integrated gradients) work. Grad-CAM requires adaptation (see attention rollout).

**Q: How do I choose the right method?**
A: Depends on goals: Grad-CAM for localization, Integrated Gradients for attribution, Guided Grad-CAM for visualization.

**Q: Are saliency maps causal explanations?**
A: No! They show correlations and sensitivity, not causality. Be careful with interpretation.

## License
Educational use only. Pretrained models follow their respective licenses.

## Acknowledgments
Based on foundational work in neural network interpretability and extensive research in explainable AI.
