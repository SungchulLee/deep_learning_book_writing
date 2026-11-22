# NASNet

NASNet(Neural Architecture Search Network)은 뉴럴 네트워크 설계의 패러다임을 바꾸었다. 구조를 손으로 설계하는 대신 강화학습으로 가장 좋은 셀(cell) 구조를 자동으로 찾아낸다. 2018년 구글 브레인이 발표했으며, 작은 데이터셋에서 컨볼루션 셀 두 종류(일반 셀과 축소 셀)를 찾고, 찾아낸 셀을 쌓아 더 큰 과제로 전이한다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
NASNet - 뉴럴 아키텍처 탐색 네트워크
논문: "Learning Transferable Architectures for Scalable Image Recognition" (2018)
핵심: 뉴럴 아키텍처 탐색(AutoML)으로 찾아낸 구조
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class NASNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, 3, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(1056, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

if __name__ == "__main__":
    model = NASNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 1,059,784
```

## 2. 논의

`NASNet` 클래스는 PyTorch의 `nn.Module` 인터페이스로 모델 구조를 담는다. `forward` 메서드가 계산 그래프를 정의하므로 트레인하는 동안 PyTorch의 autograd가 그래디언트 계산을 자동으로 처리한다. 이렇게 모듈화한 설계 덕에 구성 요소 하나하나를 고치거나 더 큰 파이프라인에 넣기가 쉽다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 트레인에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 트레인, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`NASNet`의 순전파를 따라 텐서의 형태를 추적하여라. 기본 파라메터로 샘플 4개짜리 배치를 넣었을 때, 주요 연산(컨볼루션, 풀링, 선형 레이어)마다 형태가 어떻게 되는지 적어라.

</div>

??? success "풀이"
    입력의 형태에서 출발해 레이어를 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌고(패딩이 없을 때) `padding=k//2`이면 그대로다. 커널 크기 2로 풀링하면 공간 차원이 반이 된다. 선형 레이어는 마지막 차원을 바꾼다. 배치 차원은 끝까지 그대로임을 확인하여라. 컨볼루션 레이어에서는 $(B, C, H, W)$, 평탄화한 뒤에는 $(B, F)$로 중간 형태를 적어라.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
구조를 크기 $64 \times 64$의 RGB 이미지(입력 형태: $3 \times 64 \times 64$)를 받도록 고쳐라. 레이어의 차원을 모두 그에 맞게 조정하고 모델이 오류 없이 실행되는지 확인하여라.

</div>

??? success "풀이"
    첫 컨볼루션 레이어의 `in_channels`를 현재 값에서 3으로 바꾼다. 컨볼루션과 풀링 레이어마다 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 공간 차원을 다시 계산한다. 첫 선형 레이어의 `in_features`를 마지막 컨볼루션/풀링 레이어의 평탄화된 출력에 맞춘다. `model = NASNet(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`으로 확인한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
입력과 출력의 차원이 같을 때 표준 컨볼루션과 깊이별 분리 컨볼루션(depthwise separable convolution)의 파라메터 수와 FLOPs(부동소수점 연산 횟수)를 비교하여라. 계산을 가장 크게 절약하는 때는 언제인가?

</div>

??? success "풀이"
    표준 `Conv2d(C_in, C_out, k)`의 파라메터는 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개다. 깊이별 분리 컨볼루션은 이를 (1) 깊이별(depthwise): $C_{{\text{{in}}}} \times k^2$개(입력 채널마다 필터 하나)와 (2) 점별(pointwise): $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 컨볼루션)로 나눈다. 파라메터 수의 비는 대략 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 파라메터가 약 $8{-}9\times$ 적다. $C_{{\text{{out}}}}$과 $k$가 모두 클 때 가장 크게 절약한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`NASNet`을 레이어나 블록의 수를 설정할 수 있게 확장하여라. `__init__`에 `num_layers` 인자를 더하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 구현하여라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩된 레이어를 다음으로 대체한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`으로 반복한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 그 안의 파라메터를 모두 모듈에 등록하여 옵티마이저가 갱신할 수 있다. `for n in [2, 4, 8]: model = NASNet(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — NASNet

`NASNet` 클래스는 PyTorch의 `nn.Module` 인터페이스로 모델 구조를 담는다.

핵심 클래스는 `NASNet`이며 앞의 연습문제 4개로 스스로 확인해 볼 수 있다.
