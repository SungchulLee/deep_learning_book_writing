# Self-Supervised Learning: Contrastive Learning & MAE

This collection contains implementations of key self-supervised learning methods.

## Files Included

1. **contrastive_simclr.py** - SimCLR implementation for contrastive learning
2. **contrastive_moco.py** - MoCo (Momentum Contrast) implementation
3. **mae_vision_transformer.py** - Masked Autoencoder (MAE) for images
4. **data_augmentation.py** - Augmentation strategies for self-supervised learning
5. **utils.py** - Utility functions and helpers

## Overview

### Contrastive Learning
Contrastive learning learns representations by pulling similar samples together and pushing dissimilar samples apart in the embedding space.

- **SimCLR**: Uses data augmentation to create positive pairs and maximizes agreement using contrastive loss
- **MoCo**: Uses a momentum encoder and queue mechanism for efficient contrastive learning

### Masked Autoencoder (MAE)
MAE learns by masking random patches of input images and reconstructing the masked pixels. It uses a Vision Transformer architecture.

## Requirements

```bash
pip install torch torchvision numpy --break-system-packages
```

## Usage

Each file can be run independently to train models or used as modules for your projects.

```python
# Example: Using SimCLR
from contrastive_simclr import SimCLR
model = SimCLR(base_model='resnet50')

# Example: Using MAE
from mae_vision_transformer import MAE
model = MAE(image_size=224, patch_size=16, embed_dim=768)
```
