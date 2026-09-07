# EfficientNet V2

Tan과 Le의 2021년 논문 "EfficientNetV2: Smaller Models and Faster Training"에서 나온 EfficientNetV2는 앞쪽 층의 깊이별 누비기를 Fused-MBConv 덩이로 갈음하고 차츰 배우기 전략을 써서 익히기 빠르기를 낫게 한다.

## 코드

```python
import torch
import torch.nn as nn


class FusedMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=4):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        return self.conv(x)


class EfficientNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Conv2d(3, 24, 3, 2, 1, bias=False)
        self.classifier = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


if __name__ == "__main__":
    model = EfficientNetV2()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## 논의

Fused-MBConv replaces the depthwise-then-pointwise pattern with a single $3 \times 3$ standard convolution followed by a $1 \times 1$ projection. This is used in early stages where depthwise convolutions have lower hardware utilization. Later stages still use standard MBConv with depthwise separable convolutions where the channel count is large enough to benefit.

차츰 배우기는 익히는 동안 그림 크기와 벌주기 세기를 함께 키운다. 모델은 작은 그림과 약한 불리기로 익히기를 시작해 둘을 차츰 키운다. 이러면 앞쪽 익히기가 빨라지고(작은 그림은 빠르다) 끝에 가서는 큰 그림으로 높은 정확도를 낸다.

## 연습문제

**연습문제 1.**
Fused-MBConv이 매개변수가 더 많은데도 앞쪽 층에서 MBConv보다 빠른 까닭을 설명하여라.

??? success "연습문제 1 풀이"
    In early layers, channel counts are small (e.g., 24-48). Depthwise convolutions at these small channel counts have poor hardware utilization because they cannot saturate the GPU's parallel computation units. A single standard $3 \times 3$ convolution, while having more parameters, executes as one efficient matrix operation and achieves better GPU throughput.

---

**연습문제 2.**
차츰 배우기 일정과 그것이 벌주기에 미치는 영향을 설명하여라.

??? success "연습문제 2 풀이"
    Training starts with small images (e.g., $128 \times 128$) with minimal augmentation and dropout. Over the course of training, the image size increases to the target (e.g., $384 \times 384$) while augmentation strength and dropout rate also increase proportionally. This works because small images need less regularization (less overfitting risk), while large images benefit from stronger regularization.

---

**연습문제 3.**
세대가 지날수록 그림 크기를 키우는 단순한 차츰 배우기 일정 짜개를 짜라.

??? success "연습문제 3 풀이"
    ```python
def get_image_size(epoch, max_epochs, min_size=128, max_size=384):
    progress = epoch / max_epochs
    size = int(min_size + (max_size - min_size) * progress)
    return (size // 32) * 32  # 32의 배수로 반올림

# 쓰는 보기:
for epoch in range(0, 100, 20):
    size = get_image_size(epoch, 100)
    print(f"Epoch {epoch}: image size = {size}x{size}")
# 내놓음: 128, 176, 224, 288, 336
```
