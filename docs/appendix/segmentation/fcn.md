# FCN

FCN은 2015년 논문 "Fully Convolutional Networks for Semantic Segmentation"에서 나왔다. 시맨틱 분할(픽셀마다 클래스 라벨을 매기는 과제)을 엔드투엔드로 트레인한 첫 CNN이며 완전 연결 레이어를 컨볼루션 레이어로 대체한다.

여기 구현은 FCN을 짧고 트레인하기 좋게 보인 예제다. 코드는 핵심 구조와 순전파(forward pass)에 초점을 두어, 핵심 구성 요소를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
FCN - 시맨틱 분할을 위한 완전 컨볼루션 네트워크
논문: "Fully Convolutional Networks for Semantic Segmentation" (2015)
핵심: 시맨틱 분할을 엔드투엔드로 하는 첫 컨볼루션 뉴럴 네트워크로, 완전 연결 레이어를 컨볼루션 레이어로 대체한다
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class FCN8s(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        # VGG 스타일 인코더
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        
        # 스킵 연결을 위한 스코어 레이어
        self.score_pool3 = nn.Conv2d(256, num_classes, 1)
        
        # 업샘플링
        self.upscore = nn.ConvTranspose2d(num_classes, num_classes, 16, stride=8)
    
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        
        score = self.score_pool3(conv3)
        upscore = self.upscore(score)
        
        return upscore

if __name__ == "__main__":
    model = FCN8s()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 1,853,802
```

## 2. 논의

`FCN8s` 클래스는 PyTorch의 `nn.Module` 인터페이스로 모델 구조를 캡슐화한다. 이름은 FCN-8s를 따랐지만 실제로는 단순화한 것이다. 인코더는 VGG의 앞 세 블록만 두었고, 1/8 해상도의 `conv3` 출력을 $1 \times 1$ 컨볼루션으로 클래스 점수로 바꾼 뒤 전치 컨볼루션(transposed convolution, 트레인되는 업샘플링) 하나로 8배 키운다. 원래 FCN-8s처럼 더 깊은 레이어의 점수와 얕은 레이어의 점수를 더하는 스킵 연결은 없다. 또 커널 16, 스트라이드 8인 전치 컨볼루션은 $H/8$을 $(H/8 - 1) \times 8 + 16 = H + 8$로 키우므로, 입력과 크기를 맞추려면 원 논문처럼 가장자리를 잘라 내야 한다. `forward` 메서드가 계산 그래프를 정의하므로 트레인하는 동안 PyTorch의 autograd가 그래디언트 계산을 자동으로 처리한다. 이런 모듈식 설계 덕분에 구성 요소 하나하나를 고치거나 더 큰 파이프라인에 넣기가 쉽다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 트레인에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 트레인, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`FCN8s`의 순전파를 따라 텐서의 형태를 추적하라. 기본 파라메터로 입력 표본 4개의 배치를 넣었을 때, 주요 연산(컨볼루션, 풀링, 선형 레이어)마다 형태가 어떻게 되는지 적어라.

</div>

??? success "풀이"
    입력의 형태에서 출발해 레이어를 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 $H_{\text{out}} = H_{\text{in}} - k + 1$로 바뀌고(패딩 없이) `padding=k//2`이면 그대로다. 커널 크기 2로 풀링하면 공간 차원이 반이 된다. 선형 레이어는 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임을 확인하라. 컨볼루션 레이어에서는 $(B, C, H, W)$, 평탄화(flatten) 뒤에는 $(B, F)$로 중간 형태를 적어라. `FCN8s`에는 선형 레이어도 평탄화도 없다. $224 \times 224$ 입력이면 $(4, 3, 224, 224) \to (4, 64, 112, 112) \to (4, 128, 56, 56) \to (4, 256, 28, 28)$이고, `score_pool3`이 $(4, 21, 28, 28)$을, 전치 컨볼루션이 $(4, 21, 232, 232)$를 낸다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
구조를 크기 $64 \times 64$의 RGB 이미지(입력 형태: $3 \times 64 \times 64$)를 받도록 고쳐라. 레이어의 차원을 모두 그에 맞게 조정하고 모델이 오류 없이 실행되는지 확인하라.

</div>

??? success "풀이"
    첫 컨볼루션 레이어는 이미 입력 채널이 3이다. 컨볼루션과 풀링 레이어마다 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$로 공간 차원을 다시 계산하면 $64 \to 32 \to 16 \to 8$이고, 전치 컨볼루션이 이를 $(8 - 1) \times 8 + 16 = 72$로 키운다. 완전 컨볼루션 네트워크라 선형 레이어가 없으므로 고칠 차원은 없다. 다만 출력이 $72 \times 72$로 입력보다 8 크므로, 픽셀마다 라벨과 맞추려면 가운데 $64 \times 64$를 잘라 내야 한다. `model = FCN8s(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`로 확인한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
입력과 출력의 차원이 같을 때 표준 컨볼루션과 깊이별 분리 컨볼루션(depthwise separable convolution)의 파라메터 수와 부동소수점 연산 횟수(FLOPs)를 비교하여라. 계산이 가장 크게 절약되는 때는 언제인가?

</div>

??? success "풀이"
    표준 `Conv2d(C_in, C_out, k)`의 파라메터는 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개다. 깊이별 분리 컨볼루션은 이를 (1) 깊이별(depthwise): $C_{{\text{{in}}}} \times k^2$개(입력 채널마다 필터 하나)와 (2) 점별(pointwise): $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 컨볼루션)로 나눈다. 파라메터 수의 비는 대략 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 파라메터가 약 $8{-}9\times$ 적다. $C_{{\text{{out}}}}$과 $k$가 모두 클 때 가장 크게 절약된다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`FCN8s`를 레이어나 블록의 수를 설정할 수 있게 확장하라. `__init__`에 `num_layers` 인자를 더하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 만들어라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩된 레이어를 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 반복한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 그 안의 파라메터를 모두 모듈에 등록하여 옵티마이저가 갱신할 수 있다. `for n in [2, 4, 8]: model = FCN8s(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — FCN

`FCN8s` 클래스는 PyTorch의 `nn.Module` 인터페이스로 모델 구조를 캡슐화한다.

핵심 클래스는 `FCN8s`이며 앞의 연습문제 4개로 스스로 확인해 볼 수 있다.
