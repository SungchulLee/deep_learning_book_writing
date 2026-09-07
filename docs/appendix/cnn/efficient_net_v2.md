# EfficientNet V2

EfficientNetV2은 2021년 글 "EfficientNetV2: 더 작은 모형과 더 빠른 익힘"에서 나왔으며, 녹여 붙인 MBConv 켜와 차근차근 배우는 꾀로 본디 EfficientNet을 낫게 한다. 이로써 맞음은 지키거나 올리면서 익힘 때는 크게 준다. 얼개는 신경 얼개 찾기와 손질을 아울러 찾아냈으며, 익힘 빠르기와 매개변수의 잘 듦을 함께 다듬었다.

## 코드

```python
#!/usr/bin/env python3
'''
EfficientNetV2 - Improved Efficiency and Speed
Paper: "EfficientNetV2: Smaller Models and Faster Training" (2021)
Key: Fused-MBConv layers, progressive learning, improved training speed
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

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

EfficientNetV2의 고갱이 얼개 새로움은 녹여 붙인 MBConv 덩이다. 이른 도막에서 깊이별로 가른 엮음을 여느 $3 \times 3$ 엮음으로 갈음한다. 깊이별로 가른 엮음은 매개변수가 적지만 셈의 밀도가 낮아 요즘 빠르게 하는 쇠 붙임새(GPU/TPU)를 덜 쓴다. 녹여 붙인 갈래는 여느 $3 \times 3$ 엮음으로 갈래를 넓힌 뒤 $1 \times 1$으로 되비추어, 매개변수가 조금 늘지만 쇠 붙임새를 더 잘 쓴다. EfficientNetV2은 결 그림이 큰 이른 도막에는 녹여 붙인 MBConv을, 늦은 도막에는 여느 MBConv을 쓴다.

차근차근 배우는 꾀도 고갱이 이바지다. 익히는 동안 그림의 결과 다독임의 셈(드롭아웃, 자료 불리기)을 차츰 올린다. 이른 판에는 작은 그림과 여린 불리기를 써서 거친 결을 빨리 배우게 하고, 늦은 판에는 온 결의 그림과 센 다독임을 쓴다. 이 맞추어 가는 길은 결을 붙박은 익힘보다 익힘 때를 최대 11배까지 줄일 수 있다.

EfficientNetV2은 어디에나 SiLU(Swish) 살림을 쓰는데, 여러 자리에서 ReLU보다 낫다고 밝혀졌다. SiLU 함수 $f(x) = x \cdot \sigma(x)$은 매끄럽고 한 방향으로만 오르지 않아 익히는 동안 기울기가 더 잘 흐른다.

## 익힘 문제

**익힘 1.**
들임 꼴이 $(1, 32, 56, 56)$, 넓힘 견줌 4, 날임 갈래 32일 때 여느 MBConv 덩이와 녹여 붙인 MBConv 덩이의 셈 값(뜨는 셈 횟수)을 견주어라.

??? success "익힘 1 풀이"
    MBConv에서는 (1) $1 \times 1$ 넓힘: $32 \times 128 \times 56 \times 56 \approx 12.8M$번. (2) $3 \times 3$ 깊이별: $128 \times 9 \times 56 \times 56 \approx 3.6M$번. (3) $1 \times 1$ 되비춤: $128 \times 32 \times 56 \times 56 \approx 12.8M$번. 모두 $\approx 29.2M$번이다. 녹여 붙인 MBConv에서는 (1) $3 \times 3$ 여느 엮음: $32 \times 128 \times 9 \times 56 \times 56 \approx 115.6M$번. (2) $1 \times 1$ 되비춤: $128 \times 32 \times 56 \times 56 \approx 12.8M$번. 모두 $\approx 128.4M$번이다. 녹여 붙인 MBConv은 셈이 약 4.4배 많지만 쇠 붙임새를 더 잘 써서 GPU에서는 벽시계 때가 더 빠른 일이 잦다.

---

**익힘 2.**
차근차근 배우기(익히는 동안 결을 올리기)가 ResNet처럼 결이 붙박인 모형보다 EfficientNet 결의 모형에 더 이로운 까닭을 밝혀라.

??? success "익힘 2 풀이"
    EfficientNet 모형은 깊이, 너비, 결을 함께 손보는 겹 잣대를 쓴다. 그 얼개는 여러 결에서 두루 듣도록 꾸며졌다. 차근차근 배우기는 이를 써서 낮은 결(다룰 낱그림점이 적어 되돌이가 빠름)에서 비롯해 차츰 온 결로 올린다. 그물 얼개가 결을 가리지 않으므로 낮은 결에서 배운 결이 자연스레 옮아간다. ResNet은 얼개의 가정이 붙박여 결이 바뀌면 됨됨이가 더 흔들린다. 게다가 차근차근 배우기의 다독임 짜임(낮은 결에는 여린 불리기, 높은 결에는 센 불리기)은 모형이 불린 작은 그림에 지나치게 맞춰지는 것을 막는데, 결을 붙박고 익히는 ResNet에는 덜 걸리는 탈이다.

---

**익힘 3.**
쥐어짜 북돋우기(SE) 눈길과 건너뛰는 이음을 갖춘 온전한 `FusedMBConv` 덩이를 짜라.

??? success "익힘 3 풀이"
    ```python
    class FusedMBConvSE(nn.Module):
        def __init__(self, in_ch, out_ch, expand_ratio=4, se_ratio=0.25):
            super().__init__()
            hidden = in_ch * expand_ratio
            self.use_skip = (in_ch == out_ch)
            self.expand = nn.Sequential(
                nn.Conv2d(in_ch, hidden, 3, 1, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.SiLU(inplace=True),
            )
            se_ch = max(1, int(in_ch * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden, se_ch, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(se_ch, hidden, 1),
                nn.Sigmoid(),
            )
            self.project = nn.Sequential(
                nn.Conv2d(hidden, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )

        def forward(self, x):
            out = self.expand(x)
            out = out * self.se(out)
            out = self.project(out)
            if self.use_skip:
                out = out + x
            return out
    ```
