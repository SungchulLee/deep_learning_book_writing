# CNN Visualization and Decomposition Methods

## Introduction

CNN-specific interpretability methods exploit the hierarchical, spatial structure of convolutional neural networks to provide insights unavailable to model-agnostic approaches. This section covers **Layer-wise Relevance Propagation (LRP)** and **DeepLIFT**—two decomposition methods that propagate relevance from the output back through the network, satisfying conservation properties that gradient methods lack.

## Layer-wise Relevance Propagation (LRP)

### The Conservation Principle

LRP decomposes a neural network's prediction by propagating relevance scores backward, layer by layer, satisfying a conservation property—the total relevance is preserved across layers:

$$
\sum_j R_j^{(l)} = \sum_i R_i^{(l+1)} = \ldots = f(\mathbf{x})
$$

### Propagation Rules

**LRP-0 (Basic Rule):**
$$
R_i^{(l)} = \sum_j \frac{a_i w_{ij}}{\sum_{i'} a_{i'} w_{i'j}} R_j^{(l+1)}
$$

**LRP-epsilon (Stabilized):**
$$
R_i^{(l)} = \sum_j \frac{a_i w_{ij}}{\epsilon + \sum_{i'} a_{i'} w_{i'j}} R_j^{(l+1)}
$$

**LRP-gamma (Positive-emphasis):**
$$
R_i^{(l)} = \sum_j \frac{a_i (w_{ij} + \gamma w_{ij}^+)}{\sum_{i'} a_{i'} (w_{i'j} + \gamma w_{i'j}^+)} R_j^{(l+1)}
$$

### Composite Strategy

Best practice uses different rules at different depth levels:

| Layer Type | Rule | Rationale |
|------------|------|-----------|
| Input layers | LRP-zB (bounded) | Respects input domain bounds |
| Lower conv layers | LRP-gamma ($\gamma=0.25$) | Emphasizes positive evidence |
| Upper layers | LRP-epsilon ($\epsilon=0.01$) | Suppresses noise |
| Dense layers | LRP-epsilon | Stable decomposition |

### Implementation

```python
import torch
import torch.nn as nn

class LRP:
    """Layer-wise Relevance Propagation."""
    
    def __init__(self, model, epsilon=1e-6):
        self.model = model
        self.epsilon = epsilon
        self.activations = {}
        self._register_hooks()
    
    def _register_hooks(self):
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.register_forward_hook(get_activation(name))
    
    def propagate_linear(self, layer, relevance, activation):
        """LRP-epsilon through linear layer."""
        W = layer.weight
        z = activation.unsqueeze(1) * W.unsqueeze(0)
        z_sum = z.sum(dim=2, keepdim=True) + self.epsilon
        s = relevance.unsqueeze(2) / z_sum
        c = z * s
        return c.sum(dim=1)
    
    def __call__(self, image_tensor, target_class, device):
        self.model.eval()
        image_tensor = image_tensor.to(device)
        output = self.model(image_tensor)
        
        relevance = torch.zeros_like(output)
        relevance[0, target_class] = output[0, target_class]
        
        return relevance
```

### Using Captum

```python
from captum.attr import LRP as CaptumLRP

lrp = CaptumLRP(model)
attribution = lrp.attribute(input_tensor, target=target_class)
```

## DeepLIFT

### Core Concept

DeepLIFT explains predictions by comparing activations to reference activations, addressing the saturation problem where gradients vanish for saturated neurons:

$$
\Delta y = f(\mathbf{x}) - f(\mathbf{x}^0) = \sum_i C_{\Delta x_i \Delta y}
$$

### The Saturation Advantage

For a sigmoid neuron at saturation ($x = 10$, $x^0 = 0$):
- **Gradient**: $\sigma'(10) \approx 0.000045$ (nearly zero)
- **DeepLIFT**: $(\sigma(10) - \sigma(0)) / (10 - 0) \approx 0.05$ (meaningful)

### Implementation

```python
from captum.attr import DeepLift, DeepLiftShap

# Single baseline
deeplift = DeepLift(model)
attribution = deeplift.attribute(
    input_tensor, target=target_class,
    baselines=torch.zeros_like(input_tensor)
)

# Multiple baselines (DeepLIFT SHAP)
deeplift_shap = DeepLiftShap(model)
attribution = deeplift_shap.attribute(
    input_tensor, target=target_class,
    baselines=baseline_distribution
)
```

## Feature Visualization

Beyond attribution, CNN visualization includes techniques to understand what features each neuron or layer has learned:

### Activation Maximization

Find the input that maximally activates a specific neuron:

$$
\mathbf{x}^* = \arg\max_{\mathbf{x}} a_k(\mathbf{x}) - \lambda \|\mathbf{x}\|^2
$$

### Filter Visualization

```python
def visualize_filters(model, layer_name):
    """Visualize learned convolutional filters."""
    for name, module in model.named_modules():
        if name == layer_name and isinstance(module, nn.Conv2d):
            weights = module.weight.data.cpu()
            n_filters = min(weights.shape[0], 64)
            
            fig, axes = plt.subplots(8, 8, figsize=(12, 12))
            for i, ax in enumerate(axes.flat):
                if i < n_filters:
                    w = weights[i]
                    if w.shape[0] == 3:
                        w = (w - w.min()) / (w.max() - w.min())
                        ax.imshow(w.permute(1, 2, 0))
                    else:
                        ax.imshow(w[0], cmap='gray')
                ax.axis('off')
            return fig
```

## Comparison

| Method | Conservation | Computation | Handles Saturation | Theory |
|--------|-------------|-------------|-------------------|--------|
| LRP | Yes | Medium | Depends on rule | Taylor decomposition |
| DeepLIFT | Yes | Fast | Yes | Reference comparison |
| Integrated Gradients | Yes (completeness) | Slow | Yes | Path integral |
| Vanilla Gradient | No | Very fast | No | Local sensitivity |

## Summary

CNN-specific decomposition methods—LRP and DeepLIFT—provide conservation-preserving attributions that distribute the prediction exactly across input features. Combined with feature visualization techniques, they offer comprehensive insight into CNN decision-making.

## References

1. Bach, S., et al. (2015). "On Pixel-wise Explanations for Non-Linear Classifier Decisions by Layer-wise Relevance Propagation." *PLoS ONE*.

2. Shrikumar, A., et al. (2017). "Learning Important Features Through Propagating Activation Differences." *ICML*.

3. Montavon, G., et al. (2019). "Layer-wise Relevance Propagation: An Overview." *Explainable AI*.

4. Olah, C., et al. (2017). "Feature Visualization." *Distill*.\n