# 50 Well-Known Neural Network Architectures

A comprehensive collection of 50 influential neural network architectures implemented in PyTorch with detailed comments and documentation.

## 📚 Collection Overview

This repository contains fully commented implementations of 50 landmark neural network architectures that have shaped the field of deep learning. Each implementation includes:

- **Comprehensive comments** explaining architecture design choices
- **Paper references** with links to original publications
- **Key innovations** that made each model influential
- **Runnable examples** demonstrating basic usage
- **Parameter counts** for reference

## 🏗️ Architecture Categories

### Computer Vision - Convolutional Networks

#### Classic CNNs
1. **AlexNet** (2012) - The breakthrough that started the deep learning revolution
2. **VGGNet** (2014) - Demonstrated the importance of network depth
3. **GoogLeNet/Inception v1** (2014) - Introduced inception modules with parallel paths
4. **ResNet-50** (2015) - Residual connections enabling very deep networks
5. **DenseNet-121** (2017) - Dense connections for feature reuse

#### Modern CNNs
6. **ConvNeXt** (2022) - Modernized ResNet with transformer-inspired designs
7. **EfficientNet** (2019) - Compound scaling for optimal efficiency
11. **MobileNetV2** (2018) - Efficient architecture for mobile devices
12. **SqueezeNet** (2016) - Extremely parameter-efficient design
13. **ShuffleNet** (2017) - Channel shuffle for efficiency

#### Advanced Architectures
14. **Inception v3** (2015) - Improved inception with factorized convolutions
15. **Xception** (2017) - Extreme inception with depthwise separable convolutions
16. **NASNet** (2018) - Architecture discovered by neural architecture search
17. **EfficientNetV2** (2021) - Faster training with fused operations
20. **ConvNeXt V2** (2023) - Global response normalization improvements

#### Specialized CNNs
21. **RegNet** (2020) - Quantized network design spaces
22. **ResNeXt** (2017) - Aggregated residual transformations
23. **SENet** (2018) - Squeeze-and-excitation channel attention
24. **CBAM** (2018) - Convolutional block attention module
25. **ResNeSt** (2020) - Split-attention networks
26. **GhostNet** (2020) - Generate more features from cheap operations
27. **MixNet** (2019) - Mixed depthwise convolutional kernels
28. **HRNet** (2019) - High-resolution representations throughout
29. **PyramidNet** (2017) - Gradual dimension increase
30. **CoordConv** (2018) - Adding spatial coordinates to convolutions

### Computer Vision - Transformers
18. **Vision Transformer (ViT)** (2020) - Pure transformer for images
19. **Swin Transformer** (2021) - Hierarchical vision transformer with shifted windows

### Semantic Segmentation
31. **U-Net** (2015) - Skip connections for biomedical segmentation
32. **FCN** (2015) - First fully convolutional network for segmentation
33. **DeepLabV3** (2017) - Atrous spatial pyramid pooling
34. **PSPNet** (2017) - Pyramid scene parsing network

### Object Detection
35. **Mask R-CNN** (2017) - Instance segmentation framework
36. **YOLOv3** (2018) - Real-time object detection
37. **RetinaNet** (2017) - Focal loss for dense detection
38. **Faster R-CNN** (2015) - Region proposal networks
39. **SSD** (2016) - Single shot multibox detector

### Sequence Models - RNNs
40. **LSTM** (1997) - Long short-term memory networks
41. **GRU** (2014) - Gated recurrent units
42. **Seq2Seq** (2014) - Encoder-decoder for sequence transduction
43. **Attention Seq2Seq** (2014) - Attention mechanism in sequence models

### Transformers - NLP
44. **Transformer** (2017) - Attention is all you need
8. **BERT** (2018) - Bidirectional encoder representations
9. **GPT** (2018) - Generative pre-trained transformer

### Audio Processing
10. **Wav2Vec 2.0** (2020) - Self-supervised speech representation learning

### Generative Models
45. **GAN** (2014) - Generative adversarial networks
46. **DCGAN** (2015) - Deep convolutional GAN
47. **VAE** (2013) - Variational autoencoder
48. **Autoencoder** - Classic dimensionality reduction

### Specialized Architectures
49. **CapsuleNet** (2017) - Capsule networks with dynamic routing
50. **Neural ODE** (2018) - Continuous depth models (NeurIPS Best Paper)

## 🚀 Quick Start

Each model can be run independently:

```python
import torch
from model_file import ModelName

# Create model instance
model = ModelName(num_classes=1000)

# Check parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Forward pass
x = torch.randn(2, 3, 224, 224)  # Batch of 2 images
output = model(x)
print(f"Output shape: {output.shape}")
```

## 📖 File Structure

```
50_well_known_models_enhanced/
├── 01_alex_net.py              # AlexNet (2012)
├── 02_vgg_net.py               # VGG16 (2014)
├── 03_google_net_inception.py  # GoogLeNet/Inception v1
├── 04_resnet.py                # ResNet-50
├── 05_densenet.py              # DenseNet-121
├── ...
└── 50_neural_ode.py            # Neural ODE
```

Each file is standalone and includes:
- Full architecture implementation
- Detailed comments explaining design choices
- Paper citation and link
- Key innovations summary
- Runnable example code

## 💡 Key Features

### Comprehensive Documentation
- **Architecture diagrams in comments**: Visual descriptions of layer flow
- **Design rationale**: Why specific choices were made
- **Historical context**: Impact on the field
- **Performance notes**: Typical use cases and results

### Production-Ready Code
- Clean, readable PyTorch implementations
- Consistent naming conventions
- Type hints for better IDE support
- Modular design for easy modification

### Educational Value
- Perfect for learning deep learning architectures
- Understand evolution of neural network design
- Compare different approaches to similar problems
- Reference implementations for research

## 🔍 Notable Innovations by Year

- **1997**: LSTM - Solving vanishing gradients in RNNs
- **2012**: AlexNet - Deep learning revolution begins
- **2013**: VAE - Probabilistic generative models
- **2014**: GAN - Adversarial training
- **2014**: VGG - Importance of depth
- **2014**: GoogLeNet - Inception modules
- **2014**: Seq2Seq - Neural machine translation
- **2015**: ResNet - Residual connections
- **2015**: U-Net - Medical image segmentation
- **2015**: Faster R-CNN - End-to-end object detection
- **2017**: Transformer - Attention mechanism
- **2017**: CapsNet - Dynamic routing
- **2017**: DenseNet - Feature reuse
- **2018**: BERT - Bidirectional transformers
- **2018**: GPT - Generative pre-training
- **2018**: Neural ODE - Continuous depth
- **2019**: EfficientNet - Compound scaling
- **2020**: Vision Transformer - Transformers for vision
- **2020**: Wav2Vec 2.0 - Self-supervised speech
- **2021**: Swin Transformer - Hierarchical vision transformers
- **2022**: ConvNeXt - Modernizing CNNs
- **2023**: ConvNeXt V2 - Further improvements

## 📊 Architecture Comparison

### Parameters (Approximate)
- **SqueezeNet**: ~1.2M (smallest)
- **MobileNetV2**: ~3.5M
- **EfficientNet-B0**: ~5.3M
- **ResNet-50**: ~25.6M
- **VGG16**: ~138M (largest among classics)
- **Vision Transformer**: ~86M
- **BERT-base**: ~110M
- **GPT-2**: ~1.5B (not full implementation)

### Best Use Cases
- **Image Classification**: ResNet, EfficientNet, ViT
- **Object Detection**: YOLOv3, Faster R-CNN, RetinaNet
- **Semantic Segmentation**: U-Net, DeepLabV3, PSPNet
- **Instance Segmentation**: Mask R-CNN
- **NLP**: BERT, GPT, Transformer
- **Generative**: GAN, DCGAN, VAE
- **Mobile/Edge**: MobileNet, SqueezeNet, GhostNet

## 🛠️ Requirements

```bash
pip install torch torchvision
```

Tested with:
- Python 3.8+
- PyTorch 1.10+

## 📝 Citation Format

Each model file includes the original paper citation. Example:

```
Paper: "Deep Residual Learning for Image Recognition" (2015)
Authors: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
Link: https://arxiv.org/abs/1512.03385
```

## 🎓 Learning Path

### Beginner
1. Start with AlexNet (01) - Understand basic CNN structure
2. Study VGGNet (02) - Learn about depth
3. Explore ResNet (04) - Master residual connections

### Intermediate
4. GoogLeNet (03) - Multi-path architectures
5. DenseNet (05) - Feature reuse patterns
6. U-Net (31) - Encoder-decoder structure
7. Transformer (44) - Attention mechanism

### Advanced
8. Vision Transformer (18) - Transformers for vision
9. BERT (08) - Pre-training strategies
10. Neural ODE (50) - Continuous models
11. Swin Transformer (19) - Hierarchical processing

## 🔬 Research Applications

These implementations serve as:
- **Baseline models** for comparison
- **Building blocks** for novel architectures
- **Educational resources** for understanding design principles
- **Transfer learning** starting points
- **Ablation study** references

## 🤝 Contributing

This collection represents well-established architectures. Each implementation prioritizes:
1. **Clarity** over optimization
2. **Educational value** over production efficiency
3. **Completeness** over brevity

## 📜 License

These are educational implementations of published architectures. Please cite the original papers when using these models in research.

## 🙏 Acknowledgments

This collection stands on the shoulders of giants. Each model represents years of research and countless experiments by brilliant researchers in the deep learning community.

Special recognition to:
- ImageNet competition for driving CNN innovation
- The transformer paper authors for revolutionizing sequence modeling
- The open-source community for making deep learning accessible
- All researchers who published their architectures and insights

## 📚 Additional Resources

- [Papers with Code](https://paperswithcode.com/) - Latest research
- [PyTorch Hub](https://pytorch.org/hub/) - Pre-trained models
- [Hugging Face](https://huggingface.co/) - NLP models
- [TensorFlow Model Garden](https://github.com/tensorflow/models) - Alternative implementations

---

**Last Updated**: November 2025
**Total Models**: 50
**Framework**: PyTorch
**Focus**: Education & Research

For questions or suggestions, please refer to the individual paper links in each model file.
