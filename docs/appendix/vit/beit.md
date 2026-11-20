# BEiT

BEiT는 2021년 논문 "BEiT: BERT Pre-Training of Image Transformers"에서 나왔다. 자기 지도 방식으로 사전 학습하며, 픽셀 대신 *이산 시각 토큰*(이미지 패치를 코드북의 인덱스로 바꾼 것)을 예측한다. BERT의 마스크 언어 모델에서 착안했다.

이 쪽의 구현은 BEiT의 핵심만 남긴 교육용 예시다. 코드는 핵심 구조와 순전파(forward pass, 입력에서 출력까지의 계산)에 집중하므로, 주요 구성 요소를 살펴보고 여러 가지로 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
BEiT - 이미지 트랜스포머의 BERT식 사전 학습
논문: "BEiT: BERT Pre-Training of Image Transformers" (2021)
저자: 항보 바오 외
핵심 아이디어:
  - 자기 지도 방식으로 사전 학습한다
  - 픽셀 대신 *이산 시각 토큰*을 예측한다
  - BERT의 마스크 언어 모델에서 착안했다

파일: appendix/vit/beit.py
참고: 교육용 구현이다(마스킹된 패치 예측).
"""

import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================


class BEiT(nn.Module):
    """
    BEiT 사전 학습 모델.

    스텝:
      1) 이미지를 패치로 나누어 토큰처럼 만든다
      2) 패치 일부를 마스킹한다
      3) 그 패치의 이산 시각 토큰(코드북 인덱스)을 예측한다
    """
    def __init__(self, vocab_size=8192, embed_dim=768, num_patches=196):
        super().__init__()

        self.patch_embed = nn.Linear(768, embed_dim)  # assume pre-extracted patch features
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=12, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=12)

        # 이산 토큰 ID를 예측한다
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, patch_feats, mask):
        """
        patch_feats: (B, N, D) 패치 임베딩
        mask:        (B, N) 불리언 마스크(True이면 마스킹된 것)
        """
        x = self.patch_embed(patch_feats)

        # 마스킹된 패치를 마스크 토큰으로 대체한다
        mask_token = self.mask_token.expand(x.size(0), x.size(1), -1)
        x = torch.where(mask.unsqueeze(-1), mask_token, x)

        x = x + self.pos_embed
        x = self.encoder(x)

        # 시각 토큰을 예측한다
        logits = self.head(x)  # (B, N, vocab_size)
        return logits


if __name__ == "__main__":
    pass
```

## 2. 논의

`BEiT` 클래스는 PyTorch의 `nn.Module` 인터페이스를 써서 모델 구조를 담는다. `forward` 메서드가 계산 그래프를 정의하므로 학습하는 동안 PyTorch의 autograd가 그래디언트 계산을 자동으로 처리한다. 이렇게 모듈로 나눈 설계 덕에 구성 요소 하나하나를 고치거나 더 큰 파이프라인에 넣기가 쉽다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기본 설정으로 만든 `BEiT`의 학습 가능한 파라메터를 모두 세어라. 웨이트와 바이어스를 포함해 레이어마다 나누어 적어라.

</div>

??? success "풀이"
    `nn.Linear(in_features, out_features)`마다 웨이트 파라메터가 `in_features * out_features`개이고 바이어스가 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 웨이트가 `in_c * out_c * k * k`개, 바이어스가 `out_c`개다. `nn.Embedding(num, dim)`은 파라메터가 `num * dim`개다. 모든 레이어의 값을 더한다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 형태와 자료형이 기대한 대로인지 확인하는 입력 검증을 주요 함수나 클래스에 추가하여라. 올바르지 않은 입력에는 이해하기 쉬운 오류 메시지를 띄워라.

</div>

??? success "풀이"
    `forward` 메서드(또는 알맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 검사를 추가한다. 형태를 검사하려면 중요한 차원을 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 이해하기 쉬운 오류 메시지는 디버깅을 크게 앞당기고 코드를 재사용하기에 더 견고하게 만든다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현에서 일어날 만한 실패 유형 둘을 들고, 각각 어떻게 탐지하고 고칠지 설명하여라.

</div>

??? success "풀이"
    흔한 실패 유형은 이렇다. (1) **그래디언트 소실 또는 폭발** -- 그래디언트 크기를 관찰해 탐지한다(`torch.nn.utils.clip_grad_norm_`이나 레이어마다 `param.grad.norm()` 기록하기). 그래디언트 클리핑, 더 나은 초기화(Xavier/Kaiming), 구조 변경(잔차 연결, 정규화 레이어)으로 고친다. (2) **과적합** -- 학습 로스는 줄어드는데 검증 로스가 오르면 탐지한다. 정칙화(드롭아웃, 웨이트 감쇠, 데이터 어그멘테이션)나 모델 크기 줄이기로 고친다. 학습 지표와 검증 지표를 늘 함께 관찰해 이를 일찍 잡아야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`BEiT`를 레이어나 블록의 수를 설정할 수 있게 확장하여라. `__init__`에 `num_layers` 인자를 추가하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 구현하여라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩한 레이어를 다음으로 대체한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 순회한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 그 안의 파라메터를 모델에 등록하고, 옵티마이저가 그것들을 갱신할 수 있다. `for n in [2, 4, 8]: model = BEiT(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — BEiT

`BEiT` 클래스는 PyTorch의 `nn.Module` 인터페이스를 써서 모델 구조를 담는다.

핵심 클래스는 `BEiT`이며 앞의 연습문제 4개로 스스로 확인할 수 있다.
