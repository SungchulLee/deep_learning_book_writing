# AlexNet

## Learning Objectives

By the end of this section, you will be able to:

- Understand the key innovations introduced by AlexNet (2012)
- Identify how AlexNet influenced subsequent architecture design

## Overview

**Year**: 2012 | **Parameters**: 61M | **Key Innovation**: Deep CNN with GPU training, ReLU, dropout

AlexNet (Krizhevsky et al., 2012) won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2012 by a large margin, reducing top-5 error from 26% to 16.4%. This result launched the deep learning revolution in computer vision.

## Key Innovations

1. **ReLU activation**: Replaced tanh/sigmoid with $f(x) = \max(0, x)$, enabling faster training through non-saturating gradients
2. **GPU training**: Split the network across two GPUs, establishing GPU computing as essential for deep learning
3. **Dropout**: Randomly zeroed 50% of activations during training, providing effective regularization
4. **Data augmentation**: Random crops, horizontal flips, and color jittering expanded the effective training set
5. **Local Response Normalization (LRN)**: Lateral inhibition across feature maps (later superseded by batch normalization)

## Architecture

```python
import torchvision.models as models

# AlexNet is available in torchvision
model = models.alexnet(weights='DEFAULT')
# Architecture: 5 conv layers + 3 FC layers
# Input: 224×224×3 → Output: 1000 classes
```

AlexNet demonstrated that depth (8 layers), scale (61M parameters), and data (1.2M ImageNet images) were the keys to visual recognition—principles that continue to drive the field.

## References

1. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. NeurIPS.
