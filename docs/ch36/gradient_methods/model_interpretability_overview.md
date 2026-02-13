# Model Interpretability: Grad-CAM & Attention Visualization

A comprehensive Python toolkit for visualizing and interpreting deep learning models, focusing on CNNs (via Grad-CAM) and Transformers (via attention visualization).

## 📋 Contents

This package includes:

1. **gradcam.py** - Grad-CAM and Grad-CAM++ implementations for CNNs
2. **attention_visualization.py** - Attention visualization tools for Transformers
3. **example_gradcam.py** - Example scripts for Grad-CAM usage
4. **example_attention.py** - Example scripts for attention visualization
5. **requirements.txt** - Required Python packages

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage - Grad-CAM

```python
import torch
import torchvision.models as models
from gradcam import GradCAM

# Load a pre-trained model
model = models.resnet50(pretrained=True)
model.eval()

# Initialize Grad-CAM with target layer
target_layer = model.layer4[-1]
gradcam = GradCAM(model, target_layer)

# Generate visualization
input_image = torch.randn(1, 3, 224, 224)
cam = gradcam.generate_cam(input_image)
visualization = gradcam.visualize_cam(input_image)
```

### Basic Usage - Attention Visualization

```python
from transformers import BertTokenizer, BertModel
from attention_visualization import BERTAttentionVisualizer

# Load BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

# Get attention weights
text = "The quick brown fox jumps over the lazy dog"
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)

# Visualize
visualizer = BERTAttentionVisualizer(model)
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
fig = visualizer.visualize_all_heads(outputs.attentions[-1], tokens)
```

## 📚 Features

### Grad-CAM Features

- **Standard Grad-CAM**: Visualize CNN decisions using gradient-weighted class activation mapping
- **Grad-CAM++**: Improved version with better localization for multiple objects
- **Multi-Architecture Support**: Works with ResNet, VGG, MobileNet, and custom CNNs
- **Multi-Class Visualization**: Compare activations for different predicted classes
- **Flexible Layer Selection**: Target any convolutional layer in your model

### Attention Visualization Features

- **Multi-Head Visualization**: View attention patterns across all heads
- **Layer-wise Analysis**: Compare attention across different layers
- **Attention Flow**: Track how specific tokens attend to others
- **Pattern Analysis**: Identify common attention patterns (local, global, causal)
- **Statistics**: Compute entropy and distance metrics for attention
- **BERT/GPT Support**: Built-in support for popular transformer architectures

## 📖 Documentation

### Grad-CAM Class

#### `GradCAM(model, target_layer)`

Initialize Grad-CAM visualizer.

**Parameters:**
- `model` (nn.Module): The neural network model
- `target_layer` (nn.Module): The convolutional layer to visualize (typically the last conv layer)

**Methods:**
- `generate_cam(input_image, target_class=None)`: Generate CAM heatmap
- `visualize_cam(input_image, target_class=None, original_image=None)`: Create overlay visualization

#### `GradCAMPlusPlus(model, target_layer)`

Improved Grad-CAM with better localization. Same interface as GradCAM.

### Attention Visualizer Classes

#### `AttentionVisualizer(model)`

General-purpose attention visualizer.

**Methods:**
- `register_hooks(layer_names=None)`: Register hooks to capture attention
- `visualize_attention_head(attention_weights, tokens, head_idx)`: Visualize specific head
- `visualize_all_heads(attention_weights, tokens)`: Visualize all heads in grid
- `visualize_layer_attention(layer_name, tokens)`: Visualize specific layer
- `plot_attention_flow(attention_weights, tokens, query_idx)`: Plot attention from one token

#### `BERTAttentionVisualizer(model)`

Specialized for BERT-like models. Inherits from AttentionVisualizer.

**Additional Methods:**
- `extract_attention_from_output(outputs)`: Extract attention from model outputs

## 🎯 Use Cases

### 1. Debugging CNNs
- Understand what features your CNN is focusing on
- Verify the model is learning relevant patterns
- Identify potential biases or shortcuts

### 2. Model Comparison
- Compare different architectures on the same input
- Evaluate Grad-CAM vs Grad-CAM++
- Analyze attention patterns across transformer variants

### 3. Research & Analysis
- Create publication-quality visualizations
- Analyze attention patterns in transformers
- Study layer-wise feature learning

### 4. Education
- Teach deep learning concepts visually
- Demonstrate attention mechanisms
- Show CNN feature hierarchies

## 📊 Examples

### Example 1: CNN Interpretability

```python
from example_gradcam import example_resnet_gradcam
example_resnet_gradcam()
```

Generates visualizations showing what regions of an image activate a ResNet model.

### Example 2: Transformer Attention

```python
from example_attention import example_bert_attention
example_bert_attention()
```

Visualizes multi-head attention patterns in BERT for text understanding.

### Example 3: Attention Patterns

```python
from example_attention import example_attention_patterns
example_attention_patterns()
```

Demonstrates different types of attention patterns (local, global, causal, dilated).

## 🔬 Technical Details

### Grad-CAM Algorithm

1. Forward pass to get activations and predictions
2. Backward pass to compute gradients w.r.t. target class
3. Global average pooling of gradients to get importance weights
4. Weighted combination of activation maps
5. ReLU and normalization

### Attention Visualization

1. Extract attention weights from transformer layers
2. Select specific heads or average across heads
3. Visualize as heatmap or graph
4. Analyze patterns and statistics

## 📦 Requirements

- Python >= 3.7
- PyTorch >= 1.7
- torchvision >= 0.8
- numpy >= 1.19
- matplotlib >= 3.3
- seaborn >= 0.11
- opencv-python >= 4.5
- Pillow >= 8.0
- transformers >= 4.0 (optional, for BERT/GPT examples)

## 🎨 Customization

### Custom Layer Selection

```python
# For custom models, find the right layer
def get_target_layer(model, layer_name):
    return dict(model.named_modules())[layer_name]

target_layer = get_target_layer(my_model, 'features.conv5')
```

### Custom Colormap

```python
import cv2
cam = gradcam.generate_cam(input_image)
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_VIRIDIS)
```

### Custom Attention Hooks

```python
visualizer = AttentionVisualizer(model)
visualizer.register_hooks(['encoder.layer.11.attention'])
```

## 🐛 Troubleshooting

### Issue: "No gradients computed"
- Ensure model is in eval mode: `model.eval()`
- Check that target layer has requires_grad=True

### Issue: "Attention weights not captured"
- For transformers, use `output_attentions=True`
- Verify hooks are registered before forward pass

### Issue: "CAM is all zeros"
- Check if correct target layer is selected
- Verify input preprocessing matches training

## 📄 References

### Grad-CAM
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (ICCV 2017)
- Chattopadhyay et al., "Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks" (WACV 2018)

### Attention Mechanisms
- Vaswani et al., "Attention Is All You Need" (NeurIPS 2017)
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (NAACL 2019)

## 📝 License

This code is provided for educational and research purposes. Feel free to use and modify.

## 🤝 Contributing

Suggestions and improvements are welcome! Common areas for enhancement:
- Support for additional architectures
- More attention pattern analysis tools
- Interactive visualizations
- Video/sequence support

## 📧 Support

For questions or issues:
1. Check the example scripts
2. Review the troubleshooting section
3. Consult the original papers for algorithm details

---

**Happy Visualizing! 🎨🔍**
