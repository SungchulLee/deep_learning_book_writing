# ConvNeXt: Modernizing ConvNets

## Overview

ConvNeXt demonstrates that pure convolutional networks can match or exceed Vision Transformers (ViT) when modernized with contemporary training techniques and architectural refinements.

!!! info "Key Paper"
    Liu et al., 2022 - "A ConvNet for the 2020s" ([arXiv:2201.03545](https://arxiv.org/abs/2201.03545))

## Key Modernizations

| Change | Top-1 Acc |
|--------|-----------|
| ResNet-50 baseline | 76.1% |
| + Modern training | 78.8% |
| + Stage ratio [3,3,9,3] | 79.4% |
| + Patchify stem | 79.5% |
| + Depthwise conv | 80.5% |
| + Inverted bottleneck | 80.6% |
| + Large kernel 7×7 | 80.6% |
| + GELU, LayerNorm | 81.5% |
| **ConvNeXt-T** | **82.1%** |

## Implementation

```python
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))
    
    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return residual + x
```

## Comparison with Transformers

| Model | Params | FLOPs | Top-1 |
|-------|--------|-------|-------|
| Swin-T | 28M | 4.5B | 81.3% |
| ConvNeXt-T | 29M | 4.5B | 82.1% |

## Key Takeaways

1. Training recipe matters as much as architecture
2. Large kernels + depthwise conv can match attention
3. LayerNorm and GELU work well for ConvNets
4. ConvNets remain competitive when modernized

---

**Previous**: [DenseNet](densenet.md) | **Next**: [Model Comparison](model_comparison.md)
