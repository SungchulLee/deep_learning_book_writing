# Stable Diffusion

Stable Diffusion은 2022년 논문 "High-Resolution Image Synthesis with Latent Diffusion Models"에서 나왔다. *잠재 공간*에서 확산을 수행하며, 크로스 어텐션으로 텍스트를 조건으로 삼는다.

여기 구현은 Stable Diffusion을 짧게 줄인 학습용 예제다. 코드는 핵심 구조와 순전파(forward pass)에 초점을 두어, 핵심 구성 요소를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
Stable Diffusion (개념 수준)
논문: "High-Resolution Image Synthesis with Latent Diffusion Models" (2022)
핵심 아이디어:
  - *잠재 공간*에서의 확산
  - 크로스 어텐션으로 텍스트를 조건으로 삼는다

파일: appendix/generative/stable_diffusion.py
"""

import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================


class LatentUNet(nn.Module):
    """잠재 공간에서 잡음을 예측하는 네트워크."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

    def forward(self, z, text_emb):
        # 실제 모델에서는 text_emb가 크로스 어텐션으로 조건이 된다
        return self.net(z)


class StableDiffusion(nn.Module):
    """개념 수준의 Stable Diffusion."""
    def __init__(self):
        super().__init__()
        self.unet = LatentUNet()

    def forward(self, z, text_emb):
        return self.unet(z, text_emb)


if __name__ == "__main__":
    pass
```

## 2. 논의

이 구현은 클래스 2개(`LatentUNet`, `StableDiffusion`)를 정의하고, 이들이 함께 완전한 생성 모델 구조를 이룬다. 클래스마다 고유한 역할을 맡아 코드를 모듈로 나누고 확장하기 쉽게 한다. `forward` 메서드가 PyTorch의 자동 미분이 쓰는 계산 그래프를 정의한다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기본 설정으로 초기화한 `LatentUNet`의 학습 가능한 파라메터를 모두 세어라. 웨이트와 바이어스를 함께 레이어마다 나누어 적어라.

</div>

??? success "풀이"
    `nn.Linear(in_features, out_features)`마다 웨이트 파라메터가 `in_features * out_features`개이고 바이어스가 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 웨이트가 `in_c * out_c * k * k`개, 바이어스가 `out_c`개다. `nn.Embedding(num, dim)`은 파라메터가 `num * dim`개다. 레이어를 모두 더한다. `sum(p.numel() for p in model.parameters())`으로 확인할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
어텐션 웨이트 뒤(값과 곱하기 앞)에 드롭아웃 레이어를 추가하라. 학습하는 동안 드롭아웃 비율 0.1을 써라. 어텐션 드롭아웃이 정칙화에 왜 도움이 되는지 설명하라.

</div>

??? success "풀이"
    `__init__`에 `self.attn_dropout = nn.Dropout(0.1)`을 추가하고 소프트맥스 뒤에 적용한다. `attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))`. 어텐션 드롭아웃은 학습하는 동안 어텐션 웨이트의 일부를 무작위로 0으로 만들어, 모델이 특정 토큰 간 관계에만 의존하는 것을 막는다. 그래서 모델이 어텐션을 더 고루 분산하고 더 강건한 표현을 학습하게 되는데, 이는 일반 드롭아웃이 뉴런끼리의 공적응(co-adaptation)을 막는 것과 같다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
셀프 어텐션의 계산 복잡도를 시퀀스 길이 $n$과 모델 차원 $d$의 함수로 구하라. 이것이 긴 시퀀스에 Longformer나 Linformer 같은 구조를 쓰게 되는 까닭은 무엇인가?

</div>

??? success "풀이"
    일반 셀프 어텐션은 $n \times n$ 어텐션 행렬을 계산하므로 시간 복잡도는 $O(n^2 d)$, 어텐션 웨이트의 메모리는 $O(n^2)$이다. 시퀀스가 길면($n = 4096$ 따위) 감당할 수 없다. Longformer는 국소 슬라이딩 윈도 어텐션($O(n \cdot w \cdot d)$, $w$는 윈도 크기)과 선택된 토큰에 대한 희소한 전역 어텐션을 함께 쓴다. Linformer는 키와 값을 더 낮은 차원 $k \ll n$으로 사영하여 복잡도를 $O(n \cdot k \cdot d)$로 줄인다. 둘 다 표현력을 얼마쯤 내주고 긴 입력에서 효율을 얻는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`LatentUNet`을 레이어나 블록의 수를 설정할 수 있게 확장하라. `__init__`에 `num_layers` 파라메터를 추가하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 구조를 구성하라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩된 레이어를 다음으로 대체한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`으로 반복한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`을 써야 PyTorch가 파라메터를 모두 최적화 대상으로 등록한다. `for n in [2, 4, 8]: model = LatentUNet(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — Stable Diffusion

이 구현은 클래스 2개(`LatentUNet`, `StableDiffusion`)를 정의하고, 이들이 함께 완전한 생성 모델 구조를 이룬다.

핵심 클래스는 `LatentUNet`, `StableDiffusion`이며 앞의 연습문제 4개로 스스로 점검해 볼 수 있다.
