# DenseNet

DenseNet은 하나의 dense block 안에서 앞선 특징 지도를 채널 축으로 연결해 재사용한다. 각 층이 새로운 특징을 추가하며, ResNet의 원소별 덧셈과는 다른 연결 방식이다. Huang 외의 CVPR 2017 논문에 기반한 아래 구현은 DenseNet-121의 병목·전이 블록 구성을 보여 준다.

## 1. 코드

```python
import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, 3, padding=1, bias=False)
    
    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, dim=1)
        out = self.conv1(torch.relu(self.bn1(x)))
        out = self.conv2(torch.relu(self.bn2(out)))
        return out


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super().__init__()
        self.layers = nn.ModuleList([
            DenseLayer(in_channels + i * growth_rate, growth_rate)
            for i in range(num_layers)
        ])
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, dim=1)


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pool = nn.AvgPool2d(2, 2)
    
    def forward(self, x):
        return self.pool(self.conv(torch.relu(self.bn(x))))


class DenseNet121(nn.Module):
    def __init__(self, growth_rate=32, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        num_channels = 64
        self.dense1 = DenseBlock(6, num_channels, growth_rate)
        num_channels += 6 * growth_rate
        self.trans1 = Transition(num_channels, num_channels // 2)
        num_channels = num_channels // 2
        
        self.dense2 = DenseBlock(12, num_channels, growth_rate)
        num_channels += 12 * growth_rate
        self.trans2 = Transition(num_channels, num_channels // 2)
        num_channels = num_channels // 2
        
        self.dense3 = DenseBlock(24, num_channels, growth_rate)
        num_channels += 24 * growth_rate
        self.trans3 = Transition(num_channels, num_channels // 2)
        num_channels = num_channels // 2
        
        self.dense4 = DenseBlock(16, num_channels, growth_rate)
        num_channels += 16 * growth_rate
        
        self.bn_final = nn.BatchNorm2d(num_channels)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_channels, num_classes)
    
    def forward(self, x):
        x = self.maxpool(torch.relu(self.bn1(self.conv1(x))))
        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.trans3(self.dense3(x))
        x = self.dense4(x)
        x = torch.relu(self.bn_final(x))
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


if __name__ == "__main__":
    model = DenseNet121()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 7,978,856
```

## 2. 논의

### 블록 안의 밀집 연결

입력 x₀와 이전 층의 새 특징 x₁,…,xₗ₋₁을 이어 붙여 xₗ=Hₗ([x₀,…,xₗ₋₁])를 계산한다. 성장률 k는 층마다 추가하는 채널 수이므로 L개 층을 지난 블록 출력은 C₀+Lk채널이다. 연결은 같은 해상도의 블록 안에서 유지되며, 전체 네트워크의 모든 층이 해상도와 무관하게 직접 연결되는 것은 아니다.

`DenseLayer`는 입력 특징을 합쳐 BN–ReLU–1×1과 BN–ReLU–3×3을 적용한 뒤 새로운 k개 특징만 반환한다. `DenseBlock`이 이 반환값을 목록에 추가하고 마지막에 전체 특징을 연결한다. 개별 층과 블록이 반환하는 채널 수를 구분한다.

### 병목과 압축

병목의 중간 폭은 4k다. 이 폭이 현재 입력 폭보다 작으면 3×3의 입력을 줄이지만, 초기의 좁은 층에서는 오히려 확장할 수 있다. 따라서 병목이 모든 층에서 항상 매개변수를 줄인다고 단정하지 않는다.

전이 블록은 BN–ReLU–1×1 뒤 2×2 평균 풀링을 적용한다. 예제의 채널 수는 정수 나눗셈으로 절반을 취하며 공간 크기도 줄어든다. 압축이 없더라도 유한한 깊이에서는 채널 수가 C₀+Lk로 유한하게 증가한다. ‘끝없이 늘어난다’는 설명 대신 깊이에 따른 누적 비용으로 이해한다.

### 매개변수와 활성 메모리

특징 재사용 덕분에 좁은 성장률로 표현을 구성할 수 있지만, 이전 특징의 저장과 반복 연결은 활성 메모리·메모리 접근 비용을 늘릴 수 있다. 매개변수가 적다는 사실만으로 학습 메모리와 지연시간도 더 작다고 결론 내리지 않는다. ResNet도 잔차 경로로 정보를 유지하므로 그 특징을 매번 모두 버리는 구조라고 대비하면 부정확하다.

표준 1,000클래스 DenseNet-121은 약 800만 규모지만 성장률·초기 채널·분류기를 바꾸면 달라진다. 아래 CIFAR 변형은 초기 7×7 stride=2를 3×3 stride=1로 바꾸고 최대 풀링을 제거한다. 나머지 구조는 해당 클래스의 규칙을 유지하므로 다른 문헌의 ‘CIFAR DenseNet’과 자동으로 동일하지 않다. 실행하지 않은 코드에 근거해 정확한 파라미터 수나 정확도를 단정하지 않는다.

연결과 계산 절감의 근거는 [원 논문](https://arxiv.org/abs/1608.06993)을 참고한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
C₀=64, 성장률 k=32, 6개 DenseLayer의 출력 채널과 합성곱 가중치 수를 계산하라. BN은 별도로 세라.

</div>

??? success "연습문제 1 풀이"
    출력은 64+6×32=256채널이다. 층 i의 합성곱은 (64+32i)×128+128×32×9개이므로 i=0,…,5의 합은 331,776개다. BN의 학습 가능한 스케일·편향은 2∑ᵢ[(64+32i)+128]=3,264개다. 따라서 합성곱과 BN affine을 포함한 총 학습 매개변수는 335,040개다. BN의 이동 통계 버퍼는 학습 매개변수로 세지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
병목이 항상 채널을 줄이는가? 992→128→32의 예와 초기 64→128→32를 비교하라.

</div>

??? success "연습문제 2 풀이"
    4k=128이므로 입력 992에서는 축소지만 입력 64에서는 확장이다. 992→32의 단일 3×3은 285,696개, 병목 두 합성곱은 126,976+36,864=163,840개로 줄어든다. 입력 64에서는 단일 3×3의 18,432개보다 병목의 45,056개가 크다. 계산량과 활성 메모리 절약 여부도 실제 폭에 따라 다르다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 CIFAR용 수정이 원 구조에서 바꾸는 항목과 유지하는 항목을 설명하라.

</div>

??? success "연습문제 3 풀이"

    성장률을 12, 클래스 수를 10으로 설정하고 stem을 3×3 stride=1로 바꾸며 최대 풀링을 제거한다. 초기 64채널과 [6,12,24,16] 블록 수, 전이의 절반 압축은 유지한다. 서로 다른 초기 폭을 사용하는 문헌 모델의 파라미터 수를 그대로 인용할 수 없으며 정확한 수와 성능은 해당 구성에서 확인해야 한다.

    ```python
    model = DenseNet121(growth_rate=12, num_classes=10)
    # 32x32 들임에 맞게 첫 누비기를 고친다:
    # 7x7 성큼 2 누비기를 3x3 성큼 1로 갈음
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()
    params = sum(p.numel() for p in model.parameters())
    print(f"CIFAR DenseNet: {params:,} parameters")
    ```


## 정리하며

DenseNet의 연결은 블록 안에서 채널을 누적하며 성장률은 각 층의 새 특징 수다. 병목·전이의 실제 폭을 계산하고 합성곱·BN·활성 메모리를 구분해야 한다. 적은 매개변수와 적은 실행비용은 같은 주장이 아니다.
