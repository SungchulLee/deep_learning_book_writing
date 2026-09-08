# PyramidNet

"Deep Pyramidal Residual Learning for Image Recognition"(2017)에서 나온 PyramidNet은 줄여 뽑는 단계에서 계단처럼 늘리는 흔한 방식 대신, 잔차 덩이를 지나며 특징 지도 차원을 차츰 늘린다. 이 매끄러운 피라미드 넓힘은 나타냄의 짐을 층 사이에 더 고르게 나누어, 넘어가는 지점에서의 앎 잃음을 줄이고 익히는 동안 기울기 흐름을 낫게 한다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
PyramidNet — 깊은 피라미드 잔차 그물
논문: "Deep Pyramidal Residual Networks" (2017)
핵심: 특징 지도 차원을 차츰 늘린다
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class PyramidBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(2, 2),
            )
    
    def forward(self, x):
        out = self.conv1(torch.nn.functional.relu(self.bn1(x)))
        out = self.conv2(torch.nn.functional.relu(self.bn2(out)))
        out = self.bn3(out)
        
        # 차원을 맞추려 0으로 덧대기
        shortcut = self.shortcut(x)
        if shortcut.size(1) != out.size(1):
            pad_size = out.size(1) - shortcut.size(1)
            shortcut = torch.nn.functional.pad(shortcut, (0, 0, 0, 0, 0, pad_size))
        
        out += shortcut
        return out

class PyramidNet(nn.Module):
    def __init__(self, num_classes=1000, alpha=48, depth=110):
        super().__init__()
        n = (depth - 2) // 6
        
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        
        add_rate = alpha / (3 * n)
        in_channels = 16
        
        layers = []
        for i in range(3 * n):
            out_channels = round(16 + add_rate * (i + 1))
            stride = 2 if i == n or i == 2 * n else 1
            layers.append(PyramidBasicBlock(in_channels, out_channels, stride))
            in_channels = out_channels
        
        self.layers = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layers(x)
        x = torch.nn.functional.relu(self.bn(x))
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

if __name__ == "__main__":
    model = PyramidNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## 2. 논의

여기 짠 것은 함께 어울려 온전한 그림 가르기 얼개를 이루는 클래스 2개(`PyramidBasicBlock`, `PyramidNet`)를 정한다. 클래스마다 뚜렷한 조각 하나를 감싸므로 코드가 단원별로 나뉘고 넓히기 쉽다. `forward` 메서드가 PyTorch의 자동 미분에 쓰이는 셈 그래프를 정한다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 컴퓨터 비전 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
`PyramidBasicBlock`의 앞먹임을 따라가며 텐서 꼴을 좇아라. 붙박이 매개변수로 들임 표본 4개짜리 묶음에 대해 주요 연산(누비기, 모으기, 선형 층)마다 그 뒤의 꼴을 적어라.

??? success "연습문제 1 풀이"
    입력 모양에서 출발하여 각 층을 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 (덧대기가 없으면) $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌거나 (`padding=k//2`이면) 그대로 유지된다. 커널이 2인 풀링은 공간 차원을 절반으로 만든다. 선형 층은 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임에 유의하며 추적한다. 중간 모양을 합성곱 층에서는 $(B, C, H, W)$로, 평탄화 후에는 $(B, F)$로 적는다.

---

**연습문제 2.**
$64 \times 64$ 크기의 RGB 이미지(입력 모양 $3 \times 64 \times 64$)를 받도록 구조를 수정하라. 모든 층의 차원을 그에 맞게 고치고 모델이 오류 없이 실행되는지 확인하라.

??? success "연습문제 2 풀이"
    첫 엮음 켜의 `in_channels`을 이제 값에서 3으로 바꾼다. 엮음과 모으기 켜마다 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 자리 차수를 다시 셈한다. 첫 선형 켜의 `in_features`을 마지막 엮음/모으기 켜의 편 날임에 맞춘다. `model = PyramidBasicBlock(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`으로 따진다.

---

**연습문제 3.**
같은 입출력 차원에서 표준 합성곱과 깊이별 분리 합성곱의 매개변수 개수와 FLOPs를 비교하라. 계산 절감이 가장 큰 것은 언제인가?

??? success "연습문제 3 풀이"
    표준 `Conv2d(C_in, C_out, k)`은 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개의 매개변수를 갖는다. 깊이별 분리 합성곱은 이를 둘로 나눈다. (1) 깊이별: $C_{{\text{{in}}}} \times k^2$개(입력 채널마다 필터 하나), (2) 점별: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 합성곱)이다. 매개변수의 비는 대략 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 매개변수가 약 $8{-}9\times$ 적어진다. 절감은 $C_{{\text{{out}}}}$과 $k$가 모두 클 때 가장 크다.

---

**연습문제 4.**
층이나 덩이의 개수를 정할 수 있도록 `PyramidBasicBlock`을 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`로 깊이가 바뀌는 얼개를 만들어라. 층 2, 4, 8개로 시험하여라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 되풀이한다. (그냥 파이썬 목록이 아니라) `nn.ModuleList`를 써야 PyTorch가 가장 좋게 하기에 쓸 매개변수를 모두 등록한다. 시험: `for n in [2, 4, 8]: model = PyramidBasicBlock(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.

## 정리하며

**다룬 것** — PyramidNet

여기 짠 것은 함께 어울려 온전한 그림 가르기 얼개를 이루는 클래스 2개(`PyramidBasicBlock`, `PyramidNet`)를 정한다.

고갱이 갈래는 `PyramidBasicBlock`, `PyramidNet`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
