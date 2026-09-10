# EfficientNet V2

Tan과 Le의 2021년 논문 "EfficientNetV2: Smaller Models and Faster Training"에서 나온 EfficientNetV2는 앞쪽 층의 깊이별 누비기를 Fused-MBConv 덩이로 갈음하고 차츰 배우기 전략을 써서 익히기 빠르기를 낫게 한다.

## 1. 코드

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

**출력:**

```
Parameters: 1,281,648
```

## 2. 논의

녹여 붙인 MBConv은 깊이별 다음 점별이라는 무늬를 여느 $3 \times 3$ 합성곱 하나와 그 뒤의 $1 \times 1$ 투영으로 갈음한다. 깊이별 합성곱이 하드웨어를 덜 쓰는 이른 단계에서 이를 쓴다. 뒤 단계는 채널 수가 넉넉해 이득이 있으므로 여전히 깊이별 분리 합성곱을 쓰는 여느 MBConv을 쓴다.

차츰 배우기는 익히는 동안 그림 크기와 벌주기 세기를 함께 키운다. 모델은 작은 그림과 약한 불리기로 익히기를 시작해 둘을 차츰 키운다. 이러면 앞쪽 익히기가 빨라지고(작은 그림은 빠르다) 끝에 가서는 큰 그림으로 높은 정확도를 낸다.

## 연습문제

**연습문제 1.**
Fused-MBConv이 매개변수가 더 많은데도 앞쪽 층에서 MBConv보다 빠른 까닭을 설명하여라.

??? success "연습문제 1 풀이"
    이른 켜에서는 채널 수가 작다(예: 24~48). 채널 수가 이렇게 작으면 깊이별 합성곱이 GPU의 병렬 연산 단위를 채우지 못해 하드웨어를 잘 쓰지 못한다. 여느 $3 \times 3$ 합성곱 하나는 매개변수는 더 많지만 잘 드는 행렬 연산 하나로 돌아 GPU 처리량이 더 낫다.

---

**연습문제 2.**
차츰 배우기 일정과 그것이 벌주기에 미치는 영향을 설명하여라.

??? success "연습문제 2 풀이"
    학습은 증강과 드롭아웃을 가장 적게 둔 작은 그림(예: $128 \times 128$)에서 비롯한다. 학습이 나아가면서 그림 크기가 목표(예: $384 \times 384$)까지 커지고 증강 세기와 드롭아웃 비율도 비례해 커진다. 작은 그림은 정칙화가 덜 필요하고(과적합 위험이 작다) 큰 그림은 센 정칙화가 이롭기에 이 방식이 잘 듣는다.

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

## 정리하며

**다룬 것** — EfficientNet V2

녹여 붙인 MBConv은 깊이별 다음 점별이라는 무늬를 여느 $3 \times 3$ 합성곱 하나와 그 뒤의 $1 \times 1$ 투영으로 갈음한다.

고갱이 갈래는 `FusedMBConv`, `EfficientNetV2`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
