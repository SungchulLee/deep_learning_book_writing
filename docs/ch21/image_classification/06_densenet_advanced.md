# 심화 DenseNet

이 구현은 성장률, 블록별 층 수, 압축 계수와 드롭아웃을 조절하는 DenseNet이다. 각 DenseLayer가 새 특징과 기존 특징을 즉시 연결하므로 앞의 목록 기반 구현과 같은 밀집 연결을 다른 방식으로 표현한다. 다만 stem과 기본 클래스 수가 ImageNet용 표준 모델과 다르다.

## 1. 코드

```python
import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    """BN -> ReLU -> Conv(1x1) -> BN -> ReLU -> Conv(3x3), 떨구기는 있어도 되고 없어도 된다."""
    def __init__(self, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, 3, padding=1, bias=False)
        self.drop_rate = drop_rate
    
    def forward(self, x):
        new_features = self.conv1(self.relu1(self.bn1(x)))
        new_features = self.conv2(self.relu2(self.bn2(new_features)))
        if self.drop_rate > 0:
            new_features = nn.functional.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(
                in_channels + i * growth_rate, growth_rate, bn_size, drop_rate))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)
    
    def forward(self, x):
        return self.pool(self.conv(self.relu(self.bn(x))))


class DenseNet(nn.Module):
    def __init__(self, growth_rate=12, block_config=(6, 12, 24, 16),
                 compression=0.5, num_classes=10, drop_rate=0.0):
        super().__init__()
        num_init_features = 2 * growth_rate
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True)
        )
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, num_features, growth_rate,
                             bn_size=4, drop_rate=drop_rate)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = TransitionLayer(num_features, int(num_features * compression))
                self.features.add_module(f'transition{i+1}', trans)
                num_features = int(num_features * compression)
        self.features.add_module('bn_final', nn.BatchNorm2d(num_features))
        self.features.add_module('relu_final', nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))


def densenet121(num_classes=10):
    return DenseNet(32, (6, 12, 24, 16), 0.5, num_classes)


if __name__ == "__main__":
    pass
```

## 2. 논의

### 성장률과 채널 추적

입력 C₀채널의 블록에서 L개 층이 각각 k채널을 더하면 최종 채널은 C₀+Lk다. 각 DenseLayer는 BN–ReLU–1×1과 BN–ReLU–3×3으로 새 특징을 계산하고 입력과 연결한다. 순차 모듈의 다음 층은 이미 누적된 특징을 받는다. 드롭아웃은 새 특징에만 적용하며 평가 모드에서는 꺼진다.

성장률은 채널 수를 조절하는 구조값이지 실제 정보량의 직접적인 척도는 아니다. 밀집 연결은 짧은 경로와 특징 재사용을 제공하지만 경사 소실을 없애거나 모든 데이터에서 최선의 흐름을 보장하지 않는다.

### 압축과 입력 구조

전이에서는 출력 채널을 int(Cθ)로 정하고 평균 풀링으로 해상도를 줄인다. 보통 0<θ≤1을 압축 계수로 해석하지만 코드에 별도 검증이 없어 0채널이나 확장을 만드는 값도 입력할 수 있다. 유효한 양의 채널 수와 반복 풀링에 충분한 공간 크기가 필요하다.

초기 채널은 2k이며 stem은 3×3 stride=1이다. 기본 [6,12,24,16]과 k=32가 DenseNet-121의 블록 반복 수와 일치해도, 공개 ImageNet 모델의 7×7 stem·초기 풀링·1,000클래스 분류기를 모두 재현한 것은 아니다. DenseNet-169 등의 변형도 해당 블록 구성을 명시해야 하며 함수 이름만으로 별도 구현이 제공되지는 않는다.

### 연결 수와 학습 메모리

L개 새 층과 입력을 노드로 놓으면 ℓ번째 층은 ℓ개의 이전 노드를 참조하므로 직접 연결(edge) 수는 ∑ₗ₌₁ᴸℓ=L(L+1)/2다. 이는 입력에서 출력까지 가능한 경로(path)의 수가 아니다. 성장률 k는 채널 폭에 영향을 주지만 이 노드 단위 연결 수에는 들어가지 않는다.

저장된 고유 특징의 채널 합은 깊이에 선형으로 늘 수 있지만, 각 층의 연결된 입력을 별도 텐서로 저장하면 그 입력 채널 합은 LC₀+kL(L−1)/2로 이차항을 갖는다. 학습 메모리는 구현의 저장·재계산·연결 방식에 따라 달라진다. ResNet도 역전파에 필요한 중간 활성화를 저장하므로 현재 특징 하나만 보유한다고 비교하면 부정확하다. 매개변수 메모리, 옵티마이저 상태와 활성 메모리를 나누어 비교한다.

성장률 실험은 같은 블록·압축·분류기에서 비용을 비교하는 데 사용한다. 정수 압축과 고정·선형 항이 있으므로 모든 매개변수가 정확히 k²배가 되는 것은 아니다. 아래 코드는 파라미터만 세며 정확도나 메모리를 측정하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
입력 노드와 L개 새 층 사이의 직접 참조 연결 수를 구하라. 경로 수와 같은가?

</div>

??? success "연습문제 1 풀이"
    층 ℓ이 입력과 이전 ℓ−1개 층을 참조하므로 연결 수는 ∑ℓ=L(L+1)/2이고 L=6이면 21이다. 이는 edge 수다. 입력에서 특정 층까지 여러 연결을 거치는 path의 수와 같지 않으며 별도로 정의한 그래프에서 계산해야 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
DenseNet의 작은 매개변수 수가 작은 학습 메모리를 보장하지 않는 이유를 설명하라.

</div>

??? success "연습문제 2 풀이"
    기존 특징을 재사용하면서 연결된 입력과 BN·활성화의 역전파 상태를 저장할 수 있다. 연결 입력의 총 채널은 LC₀+kL(L−1)/2로 이차항을 가지지만 실제 메모리는 공유·재계산 최적화에 따라 달라진다. ResNet도 중간 상태를 저장한다. 같은 입력·배치·정밀도·체크포인팅 조건에서 비교해야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
성장률 실험의 규모와 성능을 비교할 때 고정할 조건을 설명하라.

</div>

??? success "연습문제 3 풀이"

    블록 반복 수·압축·클래스 수·입력 크기·학습 설정을 고정한다. 여기서는 초기 채널도 2k로 변하고 압축에 정수 절삭이 있으므로 문헌의 다른 모델 수치를 그대로 인용하지 않는다. 대략적인 이차 증가 경향과 정확한 파라미터 수를 구분하며, 아래 코드는 수를 셀 뿐 정확도·메모리 실험은 수행하지 않는다.

    ```python
    for k in [8, 12, 24, 32]:
        model = DenseNet(growth_rate=k, block_config=(6, 12, 24, 16),
                        compression=0.5, num_classes=10)
        params = sum(p.numel() for p in model.parameters())
        print(f"k={k}: {params:,} parameters")
    ```


## 정리하며

밀집 연결의 edge 수와 가능한 path 수, 성장률과 실제 정보량을 구분한다. 압축·stem·클래스 수가 모델의 규모를 결정하며 학습 메모리는 활성 저장 방식에도 크게 영향을 받는다.
