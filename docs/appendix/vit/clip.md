# CLIP

CLIP은 2021년 논문 "Learning Transferable Visual Models From Natural Language Supervision"에서 나왔다. 이미지 인코더와 텍스트 인코더를 함께 학습하며, 둘이 공유하는 임베딩 공간에서 대조 로스(짝인 이미지와 텍스트는 가깝게, 짝이 아닌 것은 멀게 하는 로스)로 둘을 정렬한다.

이 쪽의 구현은 CLIP의 핵심만 남긴 교육용 예시다. 코드는 핵심 구조와 순전파(forward pass, 입력에서 출력까지의 계산)에 집중하므로, 주요 구성 요소를 살펴보고 여러 가지로 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
CLIP - 언어-이미지 대조 사전 학습
논문: "Learning Transferable Visual Models From Natural Language Supervision" (2021)
저자: 알렉 래드퍼드 외
핵심 아이디어:
  - 이미지 인코더와 텍스트 인코더를 함께 학습한다
  - 공유 임베딩 공간에서 대조 로스로 둘을 정렬한다

파일: appendix/vit/clip.py
참고: 교육용 구현이다(핵심 아이디어만 담았다).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class ImageEncoder(nn.Module):
    """단순한 ViT 방식의 이미지 인코더."""
    def __init__(self, embed_dim=512):
        super().__init__()
        self.encoder = nn.Linear(768, embed_dim)  # assume patch pooled features

    def forward(self, img_feat):
        return self.encoder(img_feat)


class TextEncoder(nn.Module):
    """단순한 트랜스포머 기반 텍스트 인코더."""
    def __init__(self, vocab_size=50000, embed_dim=512):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

    def forward(self, tokens):
        x = self.emb(tokens)
        x = self.encoder(x)
        return x[:, 0]  # use [CLS]-like token


class CLIP(nn.Module):
    """
    CLIP 모델: 대조 학습 목적 함수를 쓰는 이미지 인코더 + 텍스트 인코더.
    """
    def __init__(self, embed_dim=512):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)

        # 온도 파라메터(학습된다)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.659)

    def forward(self, image_feat, text_tokens):
        img_emb = self.image_encoder(image_feat)
        txt_emb = self.text_encoder(text_tokens)

        # 임베딩을 정규화한다
        img_emb = F.normalize(img_emb, dim=-1)
        txt_emb = F.normalize(txt_emb, dim=-1)

        # 온도로 스케일한 코사인 유사도
        scale = self.logit_scale.exp()
        logits = scale * img_emb @ txt_emb.t()

        return logits


if __name__ == "__main__":
    pass
```

## 2. 논의

이 구현은 클래스 3개(`ImageEncoder`, `TextEncoder`, `CLIP`)를 정의하고, 이들이 함께 이미지와 텍스트를 같은 임베딩 공간에 놓는 듀얼 인코더(dual encoder)를 이룬다. 이미지 인코더는 비전 트랜스포머가 아니라, 미리 뽑아 풀링한 768차원 피처를 512차원으로 사영하는 선형 레이어 하나로 줄였다. `CLIP.forward`는 배치 안의 모든 이미지-텍스트 쌍의 코사인 유사도에 스케일을 곱한 로짓 행렬을 돌려줄 뿐이고, 대조 로스는 따로 계산해야 한다. 이 행렬의 대각선이 짝인 쌍이므로, 행 방향과 열 방향으로 각각 정답이 대각선인 크로스 엔트로피를 구해 평균하면 CLIP의 대조 로스가 된다. `logit_scale`의 초깃값 2.659는 $\ln(1/0.07)$로, 온도(temperature) 0.07에 해당한다. 온도가 낮을수록 소프트맥스가 뾰족해진다. `forward` 메서드가 PyTorch의 자동 미분이 쓰는 계산 그래프를 정의한다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기본 설정으로 만든 `ImageEncoder`의 학습 가능한 파라메터를 모두 세어라. 웨이트와 바이어스를 포함해 레이어마다 나누어 적어라.

</div>

??? success "풀이"
    `nn.Linear(in_features, out_features)`마다 웨이트 파라메터가 `in_features * out_features`개이고 바이어스가 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 웨이트가 `in_c * out_c * k * k`개, 바이어스가 `out_c`개다. `nn.Embedding(num, dim)`은 파라메터가 `num * dim`개다. 모든 레이어의 값을 더한다. 기본 `ImageEncoder`에는 768에서 512로 가는 선형 레이어 하나뿐이므로 웨이트 $768 \times 512 = 393{,}216$개와 바이어스 512개, 모두 393,728개다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

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
`ImageEncoder`를 레이어나 블록의 수를 설정할 수 있게 확장하여라. `__init__`에 `num_layers` 인자를 추가하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 구현하여라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩한 레이어를 다음으로 대체한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 순회한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 그 안의 파라메터를 모델에 등록하고, 옵티마이저가 그것들을 갱신할 수 있다. `for n in [2, 4, 8]: model = ImageEncoder(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — CLIP

이 구현은 클래스 3개(`ImageEncoder`, `TextEncoder`, `CLIP`)를 정의하고, 이들이 함께 이미지와 텍스트를 같은 임베딩 공간에 놓는 듀얼 인코더를 이룬다.

핵심 클래스는 `ImageEncoder`, `TextEncoder`, `CLIP`이며 앞의 연습문제 4개로 스스로 확인할 수 있다.
