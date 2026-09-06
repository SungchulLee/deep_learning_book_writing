# ConvNeXt

Liu 외의 2022년 논문 "A ConvNet for the 2020s"에서 나온 ConvNeXt는 보기 변환기의 꾸밈 고름을 받아들여 ResNet 얼개를 요즘 것으로 바꾼다. 그 결과는 ImageNet 갈래 매기기에서 Swin 변환기와 맞먹거나 그를 넘어서는 순수 누비기 그물이며, 제대로 꾸미면 누비기가 여전히 경쟁력 있음을 보여 준다.

## 코드

```python
import torch
import torch.nn as nn


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return input + x


class ConvNeXt(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, 4, stride=4),
            nn.LayerNorm([96, 56, 56])
        )
        self.stages = nn.Sequential(*[ConvNeXtBlock(96) for _ in range(3)])
        self.norm = nn.LayerNorm(96)
        self.head = nn.Linear(96, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = x.mean([2, 3])
        x = self.norm(x)
        return self.head(x)


if __name__ == "__main__":
    model = ConvNeXt()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## 논의

ConvNeXt systematically adopts Transformer design choices into a convolutional framework. The $7 \times 7$ depthwise convolution mirrors the large receptive fields of self-attention. LayerNorm replaces BatchNorm for more stable training. GELU replaces ReLU, and the inverted bottleneck (narrow-wide-narrow) follows the MLP block pattern in Transformers. Layer scale -- a learnable per-channel scaling parameter initialized to a small value -- stabilizes training of deep networks.

The patchify stem uses a $4 \times 4$ convolution with stride 4, analogous to the patch embedding in Vision Transformers. This aggressive downsampling at the input differs from the gradual downsampling in traditional ConvNets but proves effective. The result is an architecture that bridges the gap between ConvNets and Transformers.

## 연습문제

**연습문제 1.**
Compare the receptive field of a $7 \times 7$ depthwise convolution to that of a $3 \times 3$ standard convolution. How many $3 \times 3$ layers would you need to match the receptive field?

??? success "연습문제 1 풀이"
    A single $7 \times 7$ convolution has a receptive field of $7 \times 7$. Each $3 \times 3$ convolution adds 2 pixels to the receptive field per layer. Starting from 3, after $n$ layers the receptive field is $2n + 1$. Setting $2n + 1 = 7$ gives $n = 3$. So three stacked $3 \times 3$ convolutions match the receptive field of one $7 \times 7$.

---

**연습문제 2.**
Explain why layer scale initialization with a small value (e.g., $10^{-6}$) helps training stability.

??? success "연습문제 2 풀이"
    층 잣수 매개변수를 작은 값으로 첫자리매김하면 익히기가 시작될 때 잔차 덩이가 내놓음에 거의 아무것도 보태지 않는다. 그물이 처음에는 항등 함수처럼 굴어, 익히기를 뒤흔들 수 있는 큰 깨어남과 기울기를 막는다. 익히기가 나아가면서 층 잣수 매개변수가 알맞은 값으로 자라 덩이마다 뜻있게 이바지하게 된다.

---

**연습문제 3.**
ConvNeXtBlock이 LayerNorm 대신 BatchNorm을 쓰도록 고쳐라. 무엇을 바꿔야 하는지 밝히고 있을 수 있는 맞바꿈을 논하여라.

??? success "연습문제 3 풀이"
    ```python
    class ConvNeXtBlockBN(nn.Module):
        def __init__(self, dim, layer_scale_init=1e-6):
            super().__init__()
            self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
            self.norm = nn.BatchNorm2d(dim)  # LayerNorm에서 바꿈
            self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1)  # 대신 Conv2d 쓰기
            self.act = nn.GELU()
            self.pwconv2 = nn.Conv2d(4 * dim, dim, 1)
            self.gamma = nn.Parameter(layer_scale_init * torch.ones(1, dim, 1, 1))

        def forward(self, x):
            input = x
            x = self.norm(self.dwconv(x))
            x = self.act(self.pwconv1(x))
            x = self.gamma * self.pwconv2(x)
            return input + x
    ```

    맞바꿈: BatchNorm은 묶음 통계에 기대므로 묶음 크기에 민감하고 아주 작은 묶음에는 맞지 않는다. LayerNorm은 표본마다 고르게 맞춰 더 든든하지만 조금 느릴 수 있다. BatchNorm은 익히는 동안 벌주기 효과를 볼 수 있다.
