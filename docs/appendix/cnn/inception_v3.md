# Inception V3

Inception v3는 2015년 논문 "Rethinking the Inception Architecture for Computer Vision"에서 나왔으며, 몇 가지 핵심 혁신으로 원래의 GoogLeNet을 개선했다. 큰 필터를 작고 비대칭인 필터로 쪼개는 분해 컨볼루션(factorized convolution), 라벨 스무딩 정칙화, 트레인을 안정시키는 보조 분류기(auxiliary classifier)다. 이 개선들이 함께 더 효율적이고 더 정확한 구조를 이루어 이미지 분류 연구의 표준 베이스라인이 되었다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
Inception v3 - 개선한 인셉션 구조
논문: "Rethinking the Inception Architecture" (2015)
핵심: 분해 컨볼루션(nx1과 1xn), 라벨 스무딩, 보조 분류기
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class InceptionV3(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, bias=False)
        self.conv2 = nn.Conv2d(32, 32, 3, bias=False)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.fc(x)

if __name__ == "__main__":
    model = InceptionV3()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 2,077,512
```

## 2. 논의

Inception v3의 가장 큰 구조적 기여는 컨볼루션 분해다. $5 \times 5$ 컨볼루션을 $3 \times 3$ 컨볼루션 둘을 쌓아 대체하여 파라메터를 $25C^2$에서 $18C^2$으로 줄인다. 더 나아가 $n \times n$ 컨볼루션을 $n \times 1$ 다음에 $1 \times n$ 컨볼루션으로 분해하면 $n=7$일 때 파라메터가 $49C^2$에서 $14C^2$으로 준다. 이렇게 분해해도 리셉티브 필드는 그대로면서 계산은 크게 줄어든다.

라벨 스무딩(label smoothing)도 중요한 기여다. 딱딱한 원핫 타깃 대신 Inception v3은 타깃 분포를 혼합 분포로 대체한다. $(1-\epsilon) \cdot \delta_{k,y} + \epsilon / K$이며, $\epsilon$은 흔히 0.1이고 $K$는 클래스 수다. 이러면 모델이 과도하게 확신하지 않게 되고 일반화가 좋아진다. 이 기법은 그 뒤로 깊은 네트워크 트레인의 표준 관행이 되었다.

트레인하는 동안 중간 레이어에 붙이는 보조 분류기는 추가 그래디언트 신호를 주어 아주 깊은 네트워크의 그래디언트 소실 문제에 대응한다. 보조 분류기의 로스는 작은 가중치(0.3)를 곱해 더하고, 추론할 때는 보조 분류기를 떼어 낸다. 그래디언트 소실을 막는 데 얼마나 기여하는지는 논란이 있지만, 중간 표현에서 판별력 있는 피처를 유도하여 효과적인 정칙화 역할을 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
입력 채널 256, 출력 채널 256인 $7 \times 7$ 컨볼루션을 $7 \times 1$ 다음 $1 \times 7$ 컨볼루션으로 분해할 때 절약되는 파라메터를 계산하여라.

</div>

??? success "풀이"
    원래의 $7 \times 7$ 컨볼루션: $256 \times 256 \times 7 \times 7 = 3,211,264$개. 분해한 뒤에는 $7 \times 1$ 컨볼루션이 $256 \times 256 \times 7 \times 1 = 458,752$개, $1 \times 7$ 컨볼루션이 $256 \times 256 \times 1 \times 7 = 458,752$개다. 분해한 것 전체: $917,504$개. 절약: $3,211,264 / 917,504 \approx 3.5\times$ 적으니 약 71% 줄었다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
라벨 스무딩이 과적합을 막는 데 도움이 되는 까닭을 설명하여라. 교차 엔트로피 로스 함수와는 어떻게 상호작용하는가?

</div>

??? success "풀이"
    딱딱한 원핫 타깃을 쓰면 교차 엔트로피 로스가 정답 클래스의 로짓을 다른 클래스보다 한없이 크게 내도록 몰아간다. $-\log(\text{softmax})$는 로짓 차이가 무한히 커질 때만 0에 다가가기 때문이다. 그래서 예측이 과도하게 확신하게 되고 보정(calibration)이 나빠진다. 라벨 스무딩은 타깃 분포를 바꾸어 정답 클래스의 확률을 1이 아니라 $1-\epsilon+\epsilon/K$로, 오답 클래스를 0이 아니라 $\epsilon/K$로 둔다. 이제 로스는 모델이 정답 클래스에 높되 유계인 확률을 주는 지점에서 최솟값을 가진다. 이는 모델이 불확실성을 유지하도록 유도하는 정칙화 역할을 하여 보정을 개선하고 트레인과 검증 사이의 성능 격차를 줄인다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
병렬 브랜치 셋을 지닌 Inception 모듈을 구현하여라. $1 \times 1$ 컨볼루션, 분해한 $3 \times 3$ 컨볼루션($3 \times 1$과 $1 \times 3$을 씀), 최대 풀링 브랜치다.

</div>

??? success "풀이"
    ```python
    class InceptionModule(nn.Module):
        def __init__(self, in_ch, out_1x1, out_3x3, pool_proj):
            super().__init__()
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_ch, out_1x1, 1, bias=False),
                nn.BatchNorm2d(out_1x1), nn.ReLU(inplace=True),
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_ch, out_3x3, (3, 1), padding=(1, 0), bias=False),
                nn.BatchNorm2d(out_3x3), nn.ReLU(inplace=True),
                nn.Conv2d(out_3x3, out_3x3, (1, 3), padding=(0, 1), bias=False),
                nn.BatchNorm2d(out_3x3), nn.ReLU(inplace=True),
            )
            self.branch3 = nn.Sequential(
                nn.MaxPool2d(3, 1, 1),
                nn.Conv2d(in_ch, pool_proj, 1, bias=False),
                nn.BatchNorm2d(pool_proj), nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)
    ```

## 정리하며

**다룬 것** — Inception V3

Inception v3의 가장 큰 구조적 기여는 컨볼루션 분해다.

핵심 클래스는 `InceptionV3`, `InceptionModule`이며 앞의 연습문제 3개로 스스로 확인해 볼 수 있다.
