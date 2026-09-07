# ConvNeXt V2

2023년 논문 "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders"에서 나온 ConvNeXt V2는 전체 반응 고르게 맞추기(GRN)와, 가린 자기부호기를 쓰는 더 나은 스스로 살피는 미리 익히기 전략으로 ConvNeXt를 넓힌다.

## 코드

```python
import torch
import torch.nn as nn


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
    
    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return input + x


class ConvNeXtV2(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Conv2d(3, 96, kernel_size=4, stride=4)
        self.blocks = nn.Sequential(*[ConvNeXtV2Block(96) for _ in range(3)])
        self.head = nn.Linear(96, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = x.mean([2, 3])
        return self.head(x)


if __name__ == "__main__":
    model = ConvNeXtV2()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## 논의

Global Response Normalization (GRN) addresses feature collapse in self-supervised learning. It computes the $L^2$ norm across spatial dimensions for each channel, normalizes by the mean norm, and applies learnable scale and bias parameters. This encourages feature diversity by preventing all channels from learning similar representations.

ConvNeXt V2는 예전에는 변환기만의 것으로 여겨지던 가린 자기부호기 미리 익히기를 누비기 그물도 누릴 수 있음을 보여 준다. GRN과 가린 그림 나타내기 미리 익히기가 어우러져 ConvNeXt V2는 모든 모델 크기에서 앞선 판을 앞선다.

## 연습문제

**연습문제 1.**
GRN이 무엇을 셈하는지 수학으로 밝히고, 왜 특징이 무너지는 것을 막는지 설명하여라.

??? success "연습문제 1 풀이"
    GRN computes: $\text{GRN}(X) = \gamma \cdot X \cdot \frac{\|X\|_2}{\text{mean}(\|X\|_2)} + \beta + X$, where norms are computed spatially for each channel. If a channel has a large spatial norm relative to others, it gets amplified; if small, it gets suppressed. This inter-channel competition prevents all channels from converging to the same representation during self-supervised pre-training.

---

**연습문제 2.**
가린 자기부호기 미리 익히기는 변환기를 위해 꾸며졌는데도 왜 ConvNeXt에 이로운가?

??? success "연습문제 2 풀이"
    가린 자기부호기는 들임 조각을 마구잡이로 가리고 모델이 그것을 되살리도록 익혀 돌아간다. ConvNeXt는 조각으로 나눈 줄기의 내놓음을 ViT의 조각 묻힘처럼 다루어 여기에 맞출 수 있다. GRN이 결정적인데, 그것이 없으면 성기게 가린 들임 탓에 누비기 그물에서 특징이 무너지기 때문이다. GRN이 있으면 ConvNeXt는 가린 미리 익히기에서 센 나타냄을 배운다.

---

**연습문제 3.**
ConvNeXt V2를 위한 단순한 가린 자기부호기 미리 익히기 목표를 짜라.

??? success "연습문제 3 풀이"
    ```python
class MAEPretraining(nn.Module):
    def __init__(self, encoder, decoder_dim=256, mask_ratio=0.6):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.decoder = nn.Sequential(
            nn.Conv2d(96, decoder_dim, 1),
            nn.GELU(),
            nn.Conv2d(decoder_dim, 3 * 16, 1)  # 4x4 조각 어림
        )
    
    def forward(self, x):
        # 조각에 마구잡이 가리기를 하게 된다
        features = self.encoder.stem(x)
        features = self.encoder.blocks(features)
        reconstruction = self.decoder(features)
        return reconstruction
```
