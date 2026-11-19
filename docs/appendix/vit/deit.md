# DeiT

DeiT는 2021년 논문 "Training data-efficient image transformers & distillation through attention"에서 나왔다. 지식 증류(knowledge distillation, 교사 모델의 예측을 학생 모델이 따라 하게 하는 학습)로 더 적은 데이터로 ViT를 학습하며, 클래스 토큰에 더해 *증류 토큰*을 둔다.

이 쪽의 구현은 DeiT의 핵심만 남긴 교육용 예시다. 코드는 핵심 구조와 순전파(forward pass, 입력에서 출력까지의 계산)에 집중하므로, 주요 구성 요소를 살펴보고 여러 가지로 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
DeiT - 데이터 효율적인 이미지 트랜스포머
논문: "Training data-efficient image transformers & distillation through attention" (2021)
저자: 위고 투브롱 외
핵심 아이디어:
  - 지식 증류로 더 적은 데이터로 ViT를 학습한다
  - 클래스 토큰에 더해 *증류 토큰*을 둔다

파일: appendix/vit/deit.py
참고: 교육용 구현이다(순전파만 담았다).
"""

import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================


class PatchEmbedding(nn.Module):
    """이미지를 패치로 나누어 임베딩한다."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.proj(x)                 # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2) # (B, N, D)
        return x


class DeiT(nn.Module):
    """
    DeiT = ViT + 증류 토큰.

    토큰:
      - [CLS] 토큰: 표준 분류 토큰
      - [DIST] 토큰: 교사의 예측에서 배운다
    """
    def __init__(self, num_classes=1000, embed_dim=768, num_patches=196):
        super().__init__()

        self.patch_embed = PatchEmbedding(embed_dim=embed_dim)

        # 학습 가능한 토큰
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 위치 임베딩에 두 토큰이 모두 들어간다
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))

        # 트랜스포머 인코더(단순화함)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=12, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=12)

        # 헤드 둘: 하나는 분류, 하나는 지식 증류
        self.head_cls = nn.Linear(embed_dim, num_classes)
        self.head_dist = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)

        x = self.patch_embed(x)  # (B, N, D)

        # 토큰을 배치 크기로 확장한다
        cls = self.cls_token.expand(B, -1, -1)
        dist = self.dist_token.expand(B, -1, -1)

        # 토큰과 패치 임베딩을 이어 붙인다
        x = torch.cat([cls, dist, x], dim=1)
        x = x + self.pos_embed

        # 트랜스포머 인코딩
        x = self.encoder(x)

        # 출력을 나눈다
        cls_out = x[:, 0]
        dist_out = x[:, 1]

        # 예측 헤드 둘
        logits_cls = self.head_cls(cls_out)
        logits_dist = self.head_dist(dist_out)

        return logits_cls, logits_dist


if __name__ == "__main__":
    pass
```

## 2. 논의

이 구현은 클래스 2개(`PatchEmbedding`, `DeiT`)를 정의하고, 이들이 함께 완전한 비전 트랜스포머 구조를 이룬다. 클래스마다 고유한 역할을 맡아 코드가 모듈로 나뉘고 확장하기 쉽다. `forward` 메서드가 PyTorch의 자동 미분이 쓰는 계산 그래프를 정의한다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`PatchEmbedding`의 순전파를 따라 텐서의 형태를 추적하여라. 기본 파라메터로 입력 샘플 4개를 배치로 넣었을 때, 주요 연산(컨볼루션, 풀링, 선형 레이어)마다 형태가 어떻게 되는지 적어라.

</div>

??? success "풀이"
    입력의 형태에서 시작해 레이어를 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌고(패딩 없이) `padding=k//2`이면 그대로다. 커널 크기 2로 풀링하면 공간 차원이 반이 된다. 선형 레이어는 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임을 확인하여라. 컨볼루션 레이어에서는 $(B, C, H, W)$, 평탄화한 뒤에는 $(B, F)$로 중간 형태를 적어라.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
구조를 크기 $64 \times 64$의 RGB 이미지(입력 형태: $3 \times 64 \times 64$)를 받도록 고쳐라. 레이어의 차원을 모두 그에 맞게 조정하고 모델이 오류 없이 실행되는지 확인하여라.

</div>

??? success "풀이"
    첫 컨볼루션 레이어의 `in_channels`를 현재 값에서 3으로 바꾼다. 컨볼루션과 풀링 레이어마다 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 공간 차원을 다시 계산한다. 첫 선형 레이어의 `in_features`를 마지막 컨볼루션/풀링 레이어의 평탄화된 출력에 맞춘다. `model = PatchEmbedding(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`로 확인한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
셀프 어텐션의 계산 복잡도를 시퀀스 길이 $n$과 모델 차원 $d$의 함수로 나타내어라. 이것이 긴 시퀀스를 위해 Longformer나 Linformer 같은 구조가 나온 동기가 되는 까닭은 무엇인가?

</div>

??? success "풀이"
    표준 셀프 어텐션은 $n \times n$ 어텐션 행렬을 계산하므로 시간 복잡도는 $O(n^2 d)$, 어텐션 웨이트의 메모리는 $O(n^2)$이다. 시퀀스가 길면($n = 4096$ 따위) 감당할 수 없다. Longformer는 슬라이딩 윈도 국소 어텐션($O(n \cdot w \cdot d)$, $w$는 윈도 크기)과 선택한 토큰에 대한 희소한 전역 어텐션을 함께 쓴다. Linformer는 키와 값을 더 낮은 차원 $k \ll n$으로 투영하여 복잡도를 $O(n \cdot k \cdot d)$로 줄인다. 둘 다 표현력을 얼마쯤 내주고 긴 입력에서 효율을 얻는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`PatchEmbedding`을 레이어나 블록의 수를 설정할 수 있게 확장하여라. `__init__`에 `num_layers` 인자를 추가하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 구현하여라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩한 레이어를 다음으로 대체한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 순회한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 그 안의 파라메터를 모델에 등록하고, 옵티마이저가 그것들을 갱신할 수 있다. `for n in [2, 4, 8]: model = PatchEmbedding(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — DeiT

이 구현은 클래스 2개(`PatchEmbedding`, `DeiT`)를 정의하고, 이들이 함께 완전한 비전 트랜스포머 구조를 이룬다.

핵심 클래스는 `PatchEmbedding`, `DeiT`이며 앞의 연습문제 4개로 스스로 확인할 수 있다.
