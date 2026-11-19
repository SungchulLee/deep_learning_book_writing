# LeNet

LeNet-5는 얀 르쿤과 동료들이 1998년 논문 "Gradient-Based Learning Applied to Document Recognition"에서 발표했으며, 가장 이르고 가장 큰 영향을 준 컨볼루션 뉴럴 네트워크 구조의 하나다. 원래 MNIST 데이터셋의 손글씨 숫자 인식을 위해 설계되었고, 컨볼루션으로 학습한 피처를 쓰는 뉴럴 네트워크가 손으로 설계한 피처 추출기보다 나을 수 있음을 보였다. 그 설계는 컨볼루션 레이어와 풀링 레이어를 번갈아 두고 그 뒤에 완전 연결 레이어를 두는 CNN의 기본 패턴을 처음 선보였다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
LeNet-5 - 컨볼루션 뉴럴 네트워크
논문: "Gradient-Based Learning Applied to Document Recognition" (1998)
저자: 얀 르쿤 외
핵심: 컨볼루션, 평균 풀링, 완전 연결 레이어를 쓰는 초기 CNN 구조.
MNIST 숫자 분류에 널리 쓰였다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # 컨볼루션 레이어
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        # 완전 연결 레이어
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # x: (batch, 1, 28, 28)
        x = F.relu(self.conv1(x))      # -> (batch, 6, 24, 24)
        x = F.avg_pool2d(x, 2)         # -> (batch, 6, 12, 12)

        x = F.relu(self.conv2(x))      # -> (batch, 16, 8, 8)
        x = F.avg_pool2d(x, 2)         # -> (batch, 16, 4, 4)

        x = torch.flatten(x, 1)        # -> (batch, 16*4*4)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


if __name__ == "__main__":
    model = LeNet()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    print(y.shape)  # torch.Size([1, 10])
```

**출력:**

```
torch.Size([1, 10])
```

## 2. 논의

LeNet-5는 오늘날의 CNN이 거의 다 따르는 틀을 세웠다. 채널 수가 늘어나는(1→6→16) 컨볼루션 레이어 둘을 두고, 레이어마다 뒤에 공간 차원을 반으로 줄이는 평균 풀링을 둔다. 그렇게 나온 피처 맵을 평탄화하여 차원을 차츰 줄이는(256→120→84→10) 완전 연결 레이어 셋에 넣는다. 레이어마다 비선형 활성화 함수를 적용해 네트워크가 계층적인 피처 표현을 학습하게 한다.

컨볼루션 레이어는 국소적인 공간 피처를 학습한다. 첫 레이어는 흔히 에지와 밝기 변화를 검출하도록 학습되고, 둘째 레이어는 이를 조합해 곡선이나 모서리 같은 더 높은 수준의 패턴을 이룬다. 평균 풀링은 이동 불변성(translation invariance)과 공간 압축을 주지만, 요즘 구조는 이를 최대 풀링이나 스트라이드가 있는 컨볼루션으로 거의 대체했다. 완전 연결 레이어는 학습된 피처 표현 위에서 동작하는 분류기 역할을 한다.

오늘날의 기준으로는 단순하지만 LeNet에는 지금도 유효한 핵심 원칙이 담겨 있다. 컨볼루션을 통한 웨이트 공유(완전 연결 레이어보다 파라메터가 적다), 컨볼루션과 풀링을 쌓아 이루는 공간적 계층 구조, 역전파를 통한 종단간(end-to-end) 학습이다. 파라메터가 6만 개쯤뿐인데도 LeNet은 MNIST에서 99%가 넘는 정확도를 이루어 그 과제에 놀랍도록 효율적이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
형태가 $(32, 1, 28, 28)$인 입력에 대해 LeNet의 순전파 전체에 걸쳐 텐서 형태가 어떻게 바뀌는지 추적하여라.

</div>

??? success "풀이"
    입력: $(32, 1, 28, 28)$. conv1($5 \times 5$, 필터 6개) 뒤: $28 - 5 + 1 = 24$이므로 $(32, 6, 24, 24)$. avg_pool2d(커널 2) 뒤: $(32, 6, 12, 12)$. conv2($5 \times 5$, 필터 16개) 뒤: $12 - 5 + 1 = 8$이므로 $(32, 16, 8, 8)$. avg_pool2d(커널 2) 뒤: $(32, 16, 4, 4)$. 평탄화한 뒤: $16 \times 4 \times 4 = 256$이므로 $(32, 256)$. fc1 뒤: $(32, 120)$. fc2 뒤: $(32, 84)$. fc3 뒤: $(32, 10)$.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
LeNet-5에서 학습 가능한 파라메터의 총수를 계산하여라(바이어스도 포함한다).

</div>

??? success "풀이"
    conv1: $1 \times 6 \times 5 \times 5 + 6 = 156$개. conv2: $6 \times 16 \times 5 \times 5 + 16 = 2,416$개. fc1: $256 \times 120 + 120 = 30,840$개. fc2: $120 \times 84 + 84 = 10,164$개. fc3: $84 \times 10 + 10 = 850$개. 모두: $156 + 2,416 + 30,840 + 10,164 + 850 = 44,426$개. 파라메터의 대부분(약 93%)이 완전 연결 레이어에 있음에 주목하여라.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
LeNet을 $32 \times 32$ RGB 이미지(CIFAR-10 등)를 받고 평균 풀링 대신 최대 풀링을 쓰도록 고쳐라. 차원 계산도 모두 조정하여라.

</div>

??? success "풀이"
    ```python
    class LeNetCIFAR(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)       # (B,3,32,32) -> (B,6,28,28)
            self.conv2 = nn.Conv2d(6, 16, 5)      # (B,6,14,14) -> (B,16,10,10)
            self.fc1 = nn.Linear(16 * 5 * 5, 120) # after pool: (B,16,5,5) -> 400
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_classes)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # (B,6,14,14)
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # (B,16,5,5)
            x = torch.flatten(x, 1)                      # (B,400)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
    ```
    입력 채널이 1에서 3으로, 공간 차원이 28에서 32로 바뀌고, 평탄화한 크기는 $16 \times 4 \times 4 = 256$이 아니라 $16 \times 5 \times 5 = 400$이 된다.

## 정리하며

**다룬 것** — LeNet

LeNet-5는 오늘날의 CNN이 거의 다 따르는 틀을 세웠다.

핵심 클래스는 `LeNet`, `LeNetCIFAR`이며 앞의 연습문제 3개로 스스로 확인해 볼 수 있다.
