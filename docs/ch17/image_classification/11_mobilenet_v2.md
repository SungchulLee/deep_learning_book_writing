# MobileNet V2

MobileNetV2는 Sandler 외의 2018년 논문 "MobileNetV2: Inverted Residuals and Linear Bottlenecks"에서 나왔다. 처음 MobileNet 위에, 깊이별로 갈라지는 누비기를 쓴 뒤집은 잔차 덩이를 들여왔으며 손전화 기기에서 효율적으로 미루도록 꾸몄다.

## 코드

```python
import torch
import torch.nn as nn


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = stride == 1 and in_channels == out_channels
        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        self.classifier = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        return self.classifier(x)


if __name__ == "__main__":
    model = MobileNetV2()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## 논의

MobileNetV2의 뒤집은 잔차 덩이는 예로부터의 잔차 병목과 정반대다. 가운데에서 채널을 눌러 담는 대신 부풀리고, 깊이별 누비기로 다룬 뒤 다시 내리쬔다. 내리쬐기 단계의 선형 병목(깨어남 없음)이 좁은 나타냄 속의 앎을 지킨다. 깨어남을 6에서 자르는 ReLU6은 고정소수점 미룸과 맞물리게 하려고 쓴다.

너비 곱셈수 매개변수는 모든 채널 수를 고르게 키워 정확도와 늦음 사이를 매끄럽게 맞바꾸게 한다. 너비 1.0인 MobileNetV2는 매개변수가 약 350만 개로 실시간 손전화 미룸에 알맞다.

## 연습문제

**연습문제 1.**
들임과 내놓음 차원이 같을 때 뒤집은 잔차 덩이와 보통 잔차 덩이의 FLOPs를 견주어라.

??? success "연습문제 1 풀이"
    들임 채널 $C$개, 날임 채널 $C$개, 넓힘 비율 6, 공간 크기 $H \times W$일 때

    - 뒤집힌 잔차: 넓힘($C \times 6C$) + 깊이별($6C \times 9$) + 투영($6C \times C$) = $C \times 6C \times H \times W + 6C \times 9 \times H \times W + 6C \times C \times H \times W = (12C^2 + 54C) \times H \times W$
    - Standard residual ($3 \times 3$): $2 \times C \times C \times 9 \times H \times W = 18C^2 \times H \times W$

    For large $C$, inverted residuals are cheaper since $12C^2 < 18C^2$.

---

**연습문제 2.**
MobileNetV2는 왜 보통의 ReLU 대신 ReLU6을 쓰는가?

??? success "연습문제 2 풀이"
    ReLU6은 내놓음을 6에서 자르는데, 이는 손전화에 펼칠 때의 고정소수점 양자화에 도움이 된다. 깨어남이 $[0, 6]$에 갇히면 자릿수가 적은 정수로도 정밀도를 크게 잃지 않고 나타낼 수 있어 손전화 하드웨어에서 더 빨리 미룰 수 있다.

---

**연습문제 3.**
MobileNetV3 방식의 하드 스위시 깨어남을 짜고 ReLU6과 견주어라.

??? success "연습문제 3 풀이"
    ```python
    class HardSwish(nn.Module):
        def forward(self, x):
            return x * torch.clamp(x + 3, 0, 6) / 6
    ```

    하드 스위시는 시그모이드 셈 없이 스위시 함수를 어림한다. 손전화 미룸에 효율적이면서도 ReLU6보다 조금 더 정확하다. MobileNetV3은 뒤쪽 층에 하드 스위시를, 앞쪽 층에 ReLU를 쓴다.
