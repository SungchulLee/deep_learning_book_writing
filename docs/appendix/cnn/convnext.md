# ConvNeXt

ConvNeXt는 2022년 논문 "A ConvNet for the 2020s"에서 나왔으며, 비전 트랜스포머(ViT)에서 빌려 온 설계 원칙을 넣어 기존 ResNet 구조를 체계적으로 현대화한다. 그 결과는 순수한 컨볼루션 구조이면서도 여러 비전 벤치마크에서 스윈 트랜스포머에 맞먹거나 앞선다. 최신 학습 기법과 구조 개선을 곁들이면 컨볼루션의 귀납 편향(inductive bias, 모델 구조에 미리 담긴 가정)이 여전히 매우 유효함을 보여 준다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
ConvNeXt - 2020년대를 위한 컨볼루션 네트워크
논문: "A ConvNet for the 2020s" (2022)
핵심: 비전 트랜스포머의 설계를 들여와 현대화한 ResNet
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return input + x

class ConvNeXt(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, 4, stride=4),
            nn.LayerNorm([96, 56, 56])
        )
        self.stages = nn.Sequential(*[ConvNeXtBlock(96) for _ in range(3)])
        self.norm = nn.LayerNorm(96)
        self.head = nn.Linear(96, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = x.mean([2, 3])
        x = self.norm(x)
        return self.head(x)

if __name__ == "__main__":
    model = ConvNeXt()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 941,896
```

## 2. 논의

ConvNeXt 블록은 트랜스포머에서 여러 설계를 가져온다. $7 \times 7$ 깊이별 컨볼루션(depthwise convolution)은 셀프 어텐션의 넓은 리셉티브 필드를 본떴고, `groups=dim`으로 계산을 효율화한다. 배치 정규화 대신 레이어 정규화를 써서 트랜스포머의 관례와 맞추고, 배치 통계량을 믿기 어려운 전이 학습 상황에서 성능을 올린다. 역 병목(inverted bottleneck) 설계(차원을 $4 \times$로 넓혔다가 되돌리기)는 트랜스포머 블록의 피드포워드 네트워크와 그대로 대응한다.

아주 작은 값($10^{-6}$)으로 초기화한 학습 가능한 파라메터 `gamma`가 조절하는 레이어 스케일(layer scale)도 중요한 요소다. 이는 네트워크가 항등 함수에 가까운 잔차 연결로 학습을 시작하게 하여 깊은 모델의 최적화를 안정시킨다. ReLU 대신 GELU 활성화 함수를 쓰고, 여러 레이어로 급격히 다운샘플링하던 기존 스템 대신 패치화 스템($4 \times 4$ 스트라이드 4 컨볼루션 하나)을 쓰는 것도 현대화한 대목이며, 이들이 함께 트랜스포머 구조와의 격차를 메운다.

ConvNeXt가 주는 더 큰 교훈은, 비전 트랜스포머가 우월하다고 여겨진 까닭이 셀프 어텐션 그 자체가 아니라 함께 도입된 설계와 학습 기법의 개선에 있다는 것이다. 그 개선을 컨볼루션 구조에 다시 적용하면 순수한 컨볼루션 네트워크도 높은 성능을 내며, 이동 등변성(translation equivariance, 입력을 옮기면 출력도 같이 옮겨지는 성질)과 국소 연결에서 오는 효율도 그대로 지닌다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`dim=96`인 `ConvNeXtBlock` 하나의 파라메터 수를 계산하여라.

</div>

??? success "풀이"
    깊이별 컨볼루션은 $96 \times 7 \times 7 + 96 = 4800$개(웨이트 + 바이어스)다. 레이어 정규화는 $96 + 96 = 192$개(웨이트와 바이어스)다. 첫 점별(pointwise) 선형 레이어는 $96 \times 384 + 384 = 37,248$개, 둘째 점별 선형 레이어는 $384 \times 96 + 96 = 36,960$개다. 레이어 스케일 gamma는 $96$개다. 모두 $4800 + 192 + 37,248 + 36,960 + 96 = 79,296$개다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`ConvNeXtBlock`의 순전파에 있는 `permute` 연산이 무엇을 하는지 설명하여라. NCHW 텐서에 `nn.LayerNorm`을 곧바로 적용하지 않는 까닭은 무엇인가?

</div>

??? success "풀이"
    PyTorch의 `nn.LayerNorm`은 입력의 마지막 차원을 따라 정규화한다. NCHW 형태 텐서에 적용하면 W(너비)만 정규화하거나 `[C, H, W]`를 명시해야 하는데, 그러면 정규화가 특정 공간 해상도에 묶인다. NHWC 형태로 바꾸면 레이어 정규화가 위치마다 채널 차원 C를 따라 자연스럽게 정규화하는데, 이것이 원하는 동작이다(트랜스포머에서 피처 차원을 따라 레이어 정규화를 하는 것과 같다). 선형 레이어도 마지막 차원에 작용하므로 NHWC 형태에서 채널을 곧바로 다룰 수 있다. 마지막 permute는 깊이별 컨볼루션과 맞도록 NCHW로 되돌린다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
`ConvNeXt` 모델을 스테이지 사이에 다운샘플링을 두는 다중 스테이지 구조로 고쳐라. `dim=192`인 둘째 스테이지와 두 스테이지 사이의 다운샘플링 레이어를 더하여라.

</div>

??? success "풀이"
    ```python
    class ConvNeXtMultiStage(nn.Module):
        def __init__(self, num_classes=1000):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 96, 4, stride=4),
                nn.LayerNorm([96, 56, 56])
            )
            self.stage1 = nn.Sequential(*[ConvNeXtBlock(96) for _ in range(3)])
            self.downsample = nn.Sequential(
                nn.LayerNorm([96, 56, 56]),
                nn.Conv2d(96, 192, 2, stride=2),
            )
            self.stage2 = nn.Sequential(*[ConvNeXtBlock(192) for _ in range(3)])
            self.norm = nn.LayerNorm(192)
            self.head = nn.Linear(192, num_classes)

        def forward(self, x):
            x = self.stem(x)
            x = self.stage1(x)
            x = self.downsample(x)
            x = self.stage2(x)
            x = x.mean([2, 3])
            x = self.norm(x)
            return self.head(x)
    ```
    다운샘플링 레이어는 레이어 정규화 다음에 $2 \times 2$ 스트라이드 2 컨볼루션을 쓰며, 공간 해상도를 반으로 줄이면서 채널 수를 96에서 192로 두 배 늘린다.

## 정리하며

**다룬 것** — ConvNeXt

ConvNeXt 블록은 트랜스포머에서 여러 설계를 가져온다.

핵심 클래스는 `ConvNeXtBlock`, `ConvNeXt`, `ConvNeXtMultiStage`이며 앞의 연습문제 3개로 스스로 확인해 볼 수 있다.
