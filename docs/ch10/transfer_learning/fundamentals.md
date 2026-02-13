# Transfer Learning Overview

Transfer learning uses knowledge from models pretrained on large datasets to improve performance on new tasks, especially when training data is limited. This section develops the theoretical foundation and provides a decision framework for choosing transfer strategies.

## Why Transfer Learning Works

### Hierarchical Feature Learning

Deep networks learn hierarchical representations:

| Layer depth | Features learned | Transferability |
|-------------|------------------|-----------------|
| Early (1–2) | Edges, colours, gradients | Universal |
| Middle (3–5) | Textures, patterns | Mostly universal |
| Late (6–10) | Object parts, shapes | Domain-dependent |
| Final | Task-specific concepts | Must be replaced |

Early and middle layers transfer well across tasks because low-level visual features are shared across domains.

### Mathematical Foundation

Let the source domain be $\mathcal{D}_S$ with distribution $P_S(X, Y)$ and target domain $\mathcal{D}_T$ with $P_T(X, Y)$. Transfer learning relaxes the standard assumption $P_S = P_T$, aiming to improve learning in $\mathcal{D}_T$ by leveraging $\mathcal{D}_S$, especially when $n_T \ll n_S$.

**Domain adaptation bound** (Ben-David et al., 2010):

$$\epsilon_T(h) \leq \epsilon_S(h) + d_{\mathcal{H}\Delta\mathcal{H}}(S, T) + \lambda$$

where $\epsilon_T$ and $\epsilon_S$ are target and source errors, $d_{\mathcal{H}\Delta\mathcal{H}}$ measures domain divergence, and $\lambda$ is the error of the ideal joint hypothesis. Target performance depends on source performance plus domain similarity.

## When to Use Transfer Learning

### Decision Framework

| Target data | Domain similarity | Strategy |
|-------------|-------------------|----------|
| Small (<1K) | High | Feature extraction |
| Small (<1K) | Low | Feature extraction + augmentation |
| Large (>10K) | High | Full fine-tuning |
| Large (>10K) | Low | Fine-tune with small LR |

Transfer learning is most beneficial when: (1) target data is limited, (2) domains share underlying structure, (3) a high-quality pretrained model exists.

### Negative Transfer

When source and target are too dissimilar, transfer can hurt performance. Signs include:

- Worse accuracy than random initialisation
- Learning curves that plateau early
- Features that don't activate on target data

Always compare against a randomly initialised baseline.

## Transfer Strategies Overview

| Strategy | Trainable params | Expected accuracy | Training time | Overfitting risk |
|----------|------------------|-------------------|---------------|------------------|
| Feature extraction | <1% | 80–85% | Fast | Low |
| Unfreeze layer4 | ~25% | 83–87% | Medium | Low |
| Unfreeze layer3+4 | ~50% | 85–90% | Medium | Medium |
| Full fine-tuning | 100% | 88–92% | Slow | High |

The following sections cover each strategy in detail: [feature extraction](feature_extraction.md), [fine-tuning](fine_tuning.md), [layer freezing](layer_freezing.md), and [discriminative learning rates](discriminative_lr.md).

## Data Preparation

### ImageNet Normalisation

Pretrained models expect ImageNet statistics:

```python
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
```

### Handling Different Input Channels

```python
import torch
import torch.nn as nn

# Grayscale → 3 channels
transforms.Grayscale(num_output_channels=3)

# Multi-spectral: replace first conv
def adapt_input_channels(model, in_channels):
    old = model.conv1
    model.conv1 = nn.Conv2d(in_channels, old.out_channels,
                            old.kernel_size, old.stride, old.padding, bias=False)
    with torch.no_grad():
        model.conv1.weight[:, :3] = old.weight
        if in_channels > 3:
            model.conv1.weight[:, 3:] = old.weight.mean(dim=1, keepdim=True)
    return model
```

## Complete Training Pipeline

```python
import torch
import torch.nn as nn
import torch.optim as optim
import copy


class TransferLearner:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.best_weights = None
        self.best_acc = 0

    def train(self, train_loader, val_loader, optimizer, epochs=30, patience=5):
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3)
        no_improve = 0

        for epoch in range(epochs):
            # Train
            self.model.train()
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(x), y)
                loss.backward()
                optimizer.step()

            # Validate
            val_acc = self.evaluate(val_loader)
            scheduler.step(val_acc)

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_weights = copy.deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            print(f"Epoch {epoch+1}: Val Acc = {val_acc:.2f}%")

            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        self.model.load_state_dict(self.best_weights)
        return self.best_acc

    def evaluate(self, loader):
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                correct += (self.model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        return 100 * correct / total
```

## Practical Guidelines

1. **Always start with pretrained** when data is limited
2. **Use ImageNet normalisation** for ImageNet-pretrained models
3. **Start with feature extraction** as a baseline
4. **Use discriminative LR** (10:1 or 100:1 ratio) when fine-tuning
5. **Monitor for overfitting** — large train/val gap means too much fine-tuning
6. **More data + different domain** → more layers to unfreeze

## Summary

| Aspect | Recommendation |
|--------|----------------|
| Small data, similar domain | Feature extraction |
| Small data, different domain | Fine-tune last block with small LR |
| Large data | Full fine-tuning with discriminative LR |
| Very small data (<100/class) | Few-shot transfer + augmentation |
| Normalisation | Always use ImageNet mean/std |
| Learning rate | 10–100× smaller for pretrained layers |

## References

1. Yosinski, J., et al. (2014). "How Transferable are Features in Deep Neural Networks?" *NeurIPS*.
2. Ben-David, S., et al. (2010). "A Theory of Learning from Different Domains." *Machine Learning*.
3. Kornblith, S., et al. (2019). "Do Better ImageNet Models Transfer Better?" *CVPR*.
4. Howard, J., & Ruder, S. (2018). "Universal Language Model Fine-tuning for Text Classification." *ACL*.
