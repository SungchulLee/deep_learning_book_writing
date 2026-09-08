# 항등 사상

He 등이 "Identity Mappings in Deep Residual Networks"(2016)에서 소개한 심층 잔차 신경망의 항등 사상은 원래 ResNet 설계를 크게 다듬은 것이다. 잔차 블록 안의 부품 순서를 바꾸어 배치 정규화와 ReLU를 합성곱 뒤가 아니라 *앞*에 두면 건너뛰기 연결이 참된 항등 사상이 되어, 기울기가 더 깨끗하게 흐르고 1000층이 넘는 신경망도 학습할 수 있게 된다.

이 절은 왜 순수한 항등 지름길이 최적인지, 사전 활성화 블록 설계가 그것을 어떻게 이루는지, 그리고 기울기 전파에 어떤 수학적 결과가 따르는지를 살펴본다.

---

## 1. 왜 항등 사상이 중요한가

원래 ResNet 논문은 다음 식으로 잔차 함수를 배우자고 제안했다.

$$y_l = h(x_l) + F(x_l, W_l)$$

$$x_{l+1} = f(y_l)$$

여기서 $h(x_l)$은 건너뛰기 연결, $F$은 잔차 함수, $f$은 더하기 뒤의 활성화 함수(ReLU)이다.

### 건너뛰기 연결의 변형 실험

He 등은 $h(x_l)$의 여러 형태를 체계적으로 시험했다.

| 지름길의 종류 | $h(x_l)$ | 학습 오차 |
|---------------|----------|----------------|
| 항등 (원래) | $x_l$ | 가장 좋음 |
| 배율 조정 (0.5) | $0.5 \cdot x_l$ | 더 나쁨 |
| 문 달기 | $g(x_l) \cdot x_l$ | 더 나쁨 |
| 1×1 합성곱 | $W \cdot x_l$ | 더 나쁨 |
| 드롭아웃 | $\text{dropout}(x_l)$ | 더 나쁨 |

**핵심 발견**: 순수한 항등 사상이 가장 잘 통한다. 학습된 문 달기처럼 이로워 보이는 것을 포함하여, 건너뛰기 연결에 무엇을 손대든 성능이 나빠진다.

### 수학적 설명

건너뛰기 연결이 항등이면 순전파가 깔끔하게 펼쳐진다.

$$x_L = x_l + \sum_{i=l}^{L-1} F_i(x_i)$$

어떤 깊은 층도 얕은 층에 누적된 잔차를 더한 것으로 곧바로 나타난다. 항등이 아닌 지름길 $h(x_l) = \lambda x_l$을 쓰면 펼침이 다음과 같이 된다.

$$x_L = \lambda^{L-l} x_l + \sum_{i=l}^{L-1} \lambda^{L-1-i} F_i(x_i)$$

지수 인수 $\lambda^{L-l}$이 신호를 키우거나($\lambda > 1$) 줄여($\lambda < 1$), 건너뛰기 연결이 풀려던 바로 그 기울기 흐름 문제를 되살린다.

---

## 2. 사전 활성화라는 착상

### 원래 ResNet 블록 (사후 활성화)

```
Input ─────┬─────────────────────────────────────┐
           │                                      │
           ▼                                      │
      [Conv 3×3]                                  │
           ▼                                      │
      [BatchNorm]                                 │
           ▼                                      │
        [ReLU]                                    │ (identity or projection)
           ▼                                      │
      [Conv 3×3]                                  │
           ▼                                      │
      [BatchNorm]                                 │
           ▼                                      │
         (+)  ◄───────────────────────────────────┘
           ▼
        [ReLU]  ◄─── This ReLU affects the next identity path!
           ▼
        Output
```

**문제**: 더하기 뒤에 ReLU가 있으므로 건너뛰기 연결을 지나 다음 블록으로 가는 신호*까지* ReLU를 거치게 되어 순수한 항등 사상이 깨진다. 출력 $x_{l+1} = \text{ReLU}(x_l + F(x_l))$은 언제나 음이 아니므로 항등 경로가 $\mathbb{R}_{\geq 0}$으로 제약된다.

### 사전 활성화 ResNet 블록

```
Input ─────┬─────────────────────────────────────┐
           │                                      │
           ▼                                      │
      [BatchNorm]                                 │
           ▼                                      │
        [ReLU]                                    │
           ▼                                      │
      [Conv 3×3]                                  │
           ▼                                      │
      [BatchNorm]                                 │
           ▼                                      │
        [ReLU]                                    │ (pure identity)
           ▼                                      │
      [Conv 3×3]                                  │
           ▼                                      │
         (+)  ◄───────────────────────────────────┘
           ▼
        Output (directly connects to next block's input)
```

**해결**: 배치 정규화와 ReLU를 합성곱 앞으로 옮기면 건너뛰기 연결이 참된 항등 사상이 된다. 신호가 잇따른 블록을 손대지 않은 채 흐르고 출력은 다음과 같이 간단해진다.

$$x_{l+1} = x_l + F(\hat{f}(x_l), W_l)$$

여기서 $\hat{f}$은 잔차 가지 안에서만 적용되는 사전 활성화(배치 정규화 뒤 ReLU)를 뜻한다.

---

## 3. 수학적 분석

### 정보의 전파

사전 활성화를 쓰면 순전파가 다음과 같이 된다.

$$x_{l+1} = x_l + F(\hat{f}(x_l), W_l)$$

이 점화식을 펼치면 다음과 같다.

$$x_L = x_l + \sum_{i=l}^{L-1} F(\hat{f}(x_i), W_i)$$

이는 어떤 깊은 층 $x_L$도 얕은 층 $x_l$과 중간의 모든 잔차 함수의 합임을 보인다. 곱해지는 인수가 전혀 없고 관계가 순전히 덧셈적이다.

### 기울기의 흐름

사전 활성화를 쓸 때의 기울기는 다음과 같다.

$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \left(1 + \frac{\partial}{\partial x_l}\sum_{i=l}^{L-1}F_i\right)$$

"1" 항이 $\mathcal{L}$에서 $x_l$까지 어떤 비선형에도 막히지 않는 곧바른 기울기 경로를 준다. 원래의 (사후 활성화) ResNet에서는 기울기가 블록마다 더하기 뒤의 ReLU를 지나야 하고, 그 과정에서 음의 기울기가 0이 될 수 있다. 사전 활성화는 이 병목을 아예 없앤다.

### 기울기 경로의 비교

| 경로 | 원래 ResNet | 사전 활성화 ResNet |
|------|-----------------|----------------------|
| 항등 기울기 | ReLU를 $L-l$번 지난다 | 곧바르고 손대지 않는다 |
| 기울기의 하한 | $\prod_{i=l}^{L-1} \mathbb{1}[y_i > 0]$ | 언제나 1 |
| 사라질 수 있는가? | 그렇다 (죽은 ReLU가 연쇄된다) | 아니다 (항등은 언제나 살아 있다) |

---

## 4. 구현

### 사전 활성화 기본 블록

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Type, List

class PreActBasicBlock(nn.Module):
    """
    사전 활성화 기본 블록.
    
    구조: BN → ReLU → Conv → BN → ReLU → Conv → 더하기
    
    배치 정규화와 ReLU가 합성곱 뒤에 오는 원래 BasicBlock과 달리
    여기서는 합성곱 앞에 와서 더 깨끗한 항등 사상을 만든다.
    """
    
    expansion: int = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ):
        super(PreActBasicBlock, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        # 사전 활성화: 합성곱 앞에 배치 정규화와 ReLU
        self.bn1 = norm_layer(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        
        self.bn2 = norm_layer(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 사전 활성화
        out = self.bn1(x)
        out = F.relu(out, inplace=True)
        
        # 건너뛰기 연결을 위해 저장 (첫 활성화 뒤)
        # 참고: 하향 표본화는 사전 활성화된 입력에 적용된다
        identity = out if self.downsample is None else self.downsample(out)
        
        # 첫 합성곱 (사전 활성화된 입력에)
        out = self.conv1(out)
        
        # 둘째 사전 활성화와 합성곱
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        
        # 건너뛰기 연결 (순수한 덧셈, 뒤에 활성화 없음)
        out = out + identity
        
        return out
```

### 사전 활성화 병목 블록

```python
class PreActBottleneck(nn.Module):
    """
    사전 활성화 병목 블록.
    
    구조: BN → ReLU → Conv1×1 → BN → ReLU → Conv3×3 → BN → ReLU → Conv1×1 → 더하기
    
    확장 배수 = 4 (병목 블록의 표준).
    """
    
    expansion: int = 4
    
    def __init__(
        self,
        in_channels: int,
        width: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ):
        super(PreActBottleneck, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        # 실제 너비 계산
        actual_width = int(width * (base_width / 64.0)) * groups
        
        # 1×1 축소를 위한 사전 활성화
        self.bn1 = norm_layer(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, actual_width,
            kernel_size=1, bias=False
        )
        
        # 3×3 처리를 위한 사전 활성화
        self.bn2 = norm_layer(actual_width)
        self.conv2 = nn.Conv2d(
            actual_width, actual_width,
            kernel_size=3, stride=stride, padding=1,
            groups=groups, bias=False
        )
        
        # 1×1 확장을 위한 사전 활성화
        self.bn3 = norm_layer(actual_width)
        self.conv3 = nn.Conv2d(
            actual_width, width * self.expansion,
            kernel_size=1, bias=False
        )
        
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 첫 사전 활성화
        out = self.bn1(x)
        out = F.relu(out, inplace=True)
        
        # 사전 활성화된 입력에서 오는 건너뛰기 연결
        identity = out if self.downsample is None else self.downsample(out)
        
        # 1×1 축소
        out = self.conv1(out)
        
        # 3×3 처리
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        
        # 1×1 확장
        out = self.bn3(out)
        out = F.relu(out, inplace=True)
        out = self.conv3(out)
        
        # 덧셈 (뒤에 활성화 없음)
        out = out + identity
        
        return out
```

### 완전한 사전 활성화 ResNet

```python
class PreActResNet(nn.Module):
    """
    사전 활성화 ResNet.
    
    원래 ResNet과의 핵심 차이:
    1. 잔차 블록에서 배치 정규화와 ReLU를 합성곱 앞으로 옮겼다
    2. 분류기 앞에 마지막 배치 정규화를 두었다 (마지막 블록에 사후 활성화가 없으므로)
    3. 항등 경로를 지나는 기울기의 흐름이 더 깨끗하다
    """
    
    def __init__(
        self,
        block: Type[PreActBasicBlock],
        layers: List[int],
        num_classes: int = 1000,
        in_channels: int = 3,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ):
        super(PreActResNet, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        self.in_planes = 64
        self.groups = groups
        self.base_width = width_per_group
        
        # 첫 합성곱 (배치 정규화와 ReLU 없음 — 첫 블록에 들어간다)
        self.conv1 = nn.Conv2d(
            in_channels, self.in_planes,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 잔차 단계들
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # 마지막 배치 정규화와 활성화 (사전 활성화 방식을 완성한다)
        self.bn_final = norm_layer(512 * block.expansion)
        self.relu_final = nn.ReLU(inplace=True)
        
        # 분류기
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, block, planes, num_blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        
        if stride != 1 or self.in_planes != planes * block.expansion:
            # 참고: 하향 표본화에는 배치 정규화가 없다 (사전 활성화가 정규화를 맡는다)
            downsample = nn.Conv2d(
                self.in_planes, planes * block.expansion,
                kernel_size=1, stride=stride, bias=False
            )
        
        layers = []
        layers.append(block(
            self.in_planes, planes, stride, downsample,
            norm_layer=norm_layer
        ))
        
        self.in_planes = planes * block.expansion
        
        for _ in range(1, num_blocks):
            layers.append(block(
                self.in_planes, planes,
                norm_layer=norm_layer
            ))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 풀링 전 마지막 사전 활성화
        x = self.bn_final(x)
        x = self.relu_final(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# 생성 함수들
def preact_resnet18(num_classes: int = 1000) -> PreActResNet:
    return PreActResNet(PreActBasicBlock, [2, 2, 2, 2], num_classes)

def preact_resnet50(num_classes: int = 1000) -> PreActResNet:
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3], num_classes)

def preact_resnet152(num_classes: int = 1000) -> PreActResNet:
    return PreActResNet(PreActBottleneck, [3, 8, 36, 3], num_classes)
```

---

## 5. 실험 결과

### CIFAR-10에서의 성능 비교

| 깊이 | 원래 ResNet | 사전 활성화 ResNet |
|-------|-----------------|----------------------|
| 110 | 오차 6.61% | 오차 6.37% |
| 164 | 오차 5.93% | 오차 5.46% |
| 1001 | 수렴 실패 | 오차 4.92% |

**핵심 관찰**: 깊이가 깊어질수록 사전 활성화가 꼭 필요해진다. 1001층에서는 사전 활성화 판본만 수렴한다. 보통 깊이에서는 개선이 미미하지만 극단적인 깊이에서는 결정적이다.

### 활성화 순서에 대한 제거 실험

He 등은 여러 활성화 순서를 시험했다.

| 순서 | CIFAR-10 오차 (164층) |
|----------|-----------------------------|
| 사후 활성화 (원래) | 5.93% |
| ReLU만 앞으로 | 5.71% |
| 배치 정규화만 앞으로 | 5.63% |
| 완전한 사전 활성화 (배치 정규화 + ReLU) | **5.46%** |

배치 정규화와 ReLU가 모두 개선에 이바지하며, 둘을 함께 앞으로 옮긴 완전한 사전 활성화가 최적이다.

---

## 6. 비교: 원래 방식과 사전 활성화

| 항목 | 원래 ResNet | 사전 활성화 ResNet |
|--------|-----------------|----------------------|
| 배치 정규화와 ReLU의 위치 | 합성곱 뒤 | 합성곱 앞 |
| 건너뛰기 연결 | 항등에 사후 ReLU | 순수한 항등 |
| 마지막 층의 출력 | 활성화됨 | 마지막에 배치 정규화와 ReLU가 필요 |
| 기울기 경로 | 활성화를 $L$번 지남 | 곧바른 항등 경로 |
| 아주 깊을 때 (1000층 이상) | 수렴하지 못함 | 잘 수렴함 |
| 미리 학습된 가중치 | 널리 있음 | 드묾 |

---

## 7. 사전 활성화 ResNet을 쓸 때

### 권장하는 상황

1. **아주 깊은 신경망 (100층 이상)**: 극단적인 깊이에서 수렴에 꼭 필요하다
2. **초심층 구조 연구**: 500~1000층이 넘는 신경망을 가능하게 한다
3. **학습의 안정성이 중요할 때**: 학습 동역학이 더 안정적이다
4. **조밀 예측 과제**: 분할을 위한 특징 표현이 더 깨끗하다

### 원래 ResNet으로 충분할 때

1. **보통 깊이 (18~50층)**: 두 판본의 성능이 비슷하다
2. **미리 학습된 가중치를 쓸 때**: 대부분의 사전 학습 모델이 원래 ResNet의 순서를 쓴다
3. **실전 배포**: 프레임워크의 지원이 낫고 사전 학습 검사점이 더 많다

---

## 8. 트랜스포머 구조와의 관계

사전 활성화 설계는 오늘날 주류인 "사전 정규화" 트랜스포머 구조에 곧바로 영향을 주었다. 사전 정규화 트랜스포머에서는 층 정규화를 어텐션과 순방향 부분층 *앞*에 적용하여 똑같이 순수한 항등 건너뛰기 연결을 만든다.

$$x_{l+1} = x_l + \text{Attention}(\text{LN}(x_l))$$

$$x_{l+2} = x_{l+1} + \text{FFN}(\text{LN}(x_{l+1}))$$

이 관계는 항등 사상의 원리가 구조를 가리지 않음을 보여 준다. 잔차 연결이 있는 어떤 깊은 신경망에도 이롭다. 퀀트 금융에서 시계열 모형화와 금융 텍스트의 자연어 처리에 널리 쓰이는 순차열 모델도 여기에 든다.

---

## 연습문제

**연습문제 1.**
잔차 연결 $y = x + F(x)$을 지나는 기울기의 크기가 적어도 1임을 증명하라.

??? success "연습문제 1 풀이"
    $\frac{\partial y}{\partial x} = I + \frac{\partial F}{\partial x}$이다. 기울기는 $I$에 그 함수의 야코비 행렬을 더한 것이다. $\frac{\partial F}{\partial x} \approx 0$이더라도 항등항이 기울기가 사라지지 않게 해 준다. 잔차 블록 $L$개의 사슬에서는 $\frac{\partial y_L}{\partial x_0} = \prod_{l=1}^L (I + \frac{\partial F_l}{\partial x_{l-1}})$이며, 펼치면 언제나 항 $I$을 품는다.

---

**연습문제 2.**
사전 활성화 ResNet(BN-ReLU-Conv)과 사후 활성화(Conv-BN-ReLU)를 비교하라. 아주 깊은 신경망에는 어느 쪽이 나은가?

??? success "연습문제 2 풀이"
    아주 깊은 신경망(100층 초과)에는 사전 활성화(He 등, 2016)가 낫다. 사후 활성화에서는 건너뛰기 경로에서도 신호가 배치 정규화와 ReLU를 지나 깨끗한 항등 사상이 깨진다. 사전 활성화는 건너뛰기 경로를 순수한 항등으로 두어 기울기의 흐름을 최적으로 지킨다.

---

**연습문제 3.**
차원이 바뀔 때 쓰는 사영 지름길을 갖춘 잔차 블록을 PyTorch로 구현하라.

??? success "연습문제 3 풀이"
    ```python
    class ResBlock(nn.Module):
        def __init__(self, in_ch, out_ch, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1)
            self.bn1 = nn.BatchNorm2d(out_ch)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
            self.bn2 = nn.BatchNorm2d(out_ch)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride), nn.BatchNorm2d(out_ch)
            ) if stride != 1 or in_ch != out_ch else nn.Identity()
        def forward(self, x):
            return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))
    ```

---

**연습문제 4.**
잔차 신경망과 앙상블 학습의 관계를 설명하라.

??? success "연습문제 4 풀이"
    Veit 등(2016)은 ResNet이 얕은 신경망의 앙상블처럼 움직임을 보였다. 잔차 연결을 풀어 보면 블록이 $L$개인 ResNet에 길이가 서로 다른 경로가 $2^L$개 있다. 기울기는 대부분 짧은 경로(블록 3~5개)로 흐르는데, 이는 ResNet이 얕은 부분 신경망의 앙상블을 암묵적으로 학습시킴을 시사한다.

## 정리하며

사전 활성화 ResNet은 중요한 순서 변경을 들여온다.

| 변경 | 영향 |
|--------|--------|
| 합성곱 앞의 배치 정규화 | 합성곱마다의 입력을 정규화한다 |
| 합성곱 앞의 ReLU | 정규화된 특징에 활성화를 적용한다 |
| 더하기 뒤의 항등 | 지름길로 기울기가 순수하게 흐른다 |
| 분류기 앞의 마지막 배치 정규화 | 사전 활성화 방식을 완성한다 |

핵심 원리는 **건너뛰기 연결이 손대지 않은 항등 사상이어야 한다**는 것이다. 지름길 경로에 어떤 변환을 적용하든, 그것이 학습된 것(합성곱)이든 고정된 것(배율 조정)이든 비선형(ReLU)이든 기울기의 흐름과 학습 성능을 해친다. 이 원리는 깊이와 구조와 분야를 가리지 않고 성립한다.

**참고 문헌**

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity Mappings in Deep Residual Networks. *ECCV 2016*.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.
3. Xiong, R., Yang, Y., He, J., Zheng, K., Zheng, S., Xing, C., Zhang, H., Lan, Y., Wang, L., & Liu, T. (2020). On Layer Normalization in the Transformer Architecture. *ICML 2020*.
