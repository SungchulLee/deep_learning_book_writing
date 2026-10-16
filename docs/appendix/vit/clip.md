# CLIP

CLIP은 2021년 글 "Learning Transferable Visual Models From Natural Language Supervision"에서 나왔다. - 그림 인코더와 텍스트 인코더를 함께 학습된이다 - 나누어 쓰는 담음 밭에서 맞대어 로스로 둘을 맞춘다.

여기 구현은 CLIP을 짧고 학습하기 좋게 보인 본이다. 코드는 핵심 구조와 앞으로 스텝에 마음을 두어, 핵심 꾸밈새를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
CLIP - 맞대어 배우는 말-그림 미리 익히기
글: "흔한 말의 이끎에서 옮길 수 있는 보기 모형 배우기" (2021)
지은이: 알렉 래드퍼드 외
핵심 깨침:
  - 그림 인코더와 글월 인코더를 함께 익힌다
  - 나누어 쓰는 담음 밭에서 맞대어 잃음으로 둘을 맞춘다

두루마리: appendix/vit/clip.py
눈여겨볼 것: 배우기 위한 구현이다(핵심 깨침만 담았다).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class ImageEncoder(nn.Module):
    """단순한 ViT 결의 그림 인코더."""
    def __init__(self, embed_dim=512):
        super().__init__()
        self.encoder = nn.Linear(768, embed_dim)  # assume patch pooled features

    def forward(self, img_feat):
        return self.encoder(img_feat)


class TextEncoder(nn.Module):
    """단순한 변환기 바탕 글월 인코더."""
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
    CLIP 모형: 맞대어 배우는 목표를 지닌 그림 인코더 + 글월 인코더.
    """
    def __init__(self, embed_dim=512):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)

        # 온도 파라메터(배운다)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.659)

    def forward(self, image_feat, text_tokens):
        img_emb = self.image_encoder(image_feat)
        txt_emb = self.text_encoder(text_tokens)

        # 담음의 잣대를 맞춘다
        img_emb = F.normalize(img_emb, dim=-1)
        txt_emb = F.normalize(txt_emb, dim=-1)

        # 온도로 잣대를 맞춘 코사인 닮음
        scale = self.logit_scale.exp()
        logits = scale * img_emb @ txt_emb.t()

        return logits


if __name__ == "__main__":
    pass
```

## 2. 논의

이 구현은 클래스 3개(`ImageEncoder`, `TextEncoder`, `CLIP`)를 매기고, 이들이 어울려 온전한 보기 트랜스포머 구조를 이룬다. 클래스마다 남다른 몫을 담아 코드를 묶음으로 나누고 넓히기 쉽게 한다. `forward` 방법이 PyTorch가 절로 미분하는 데 쓰는 계산 그림을 매긴다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 증강 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기본 첫자리로 잡은 `ImageEncoder`의 배울 수 있는 파라메터를 모두 세어라. 짐과 바이어스를 아울러 켜마다 나누어 적어라.

</div>

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)`마다 짐 파라메터가 `in_features * out_features`개이고 바이어스가 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 짐이 `in_c * out_c * k * k`개, 바이어스가 `out_c`개다. `nn.Embedding(num, dim)`은 파라메터가 `num * dim`개다. 켜를 모두 더한다. `sum(p.numel() for p in model.parameters())`으로 따져 볼 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 꼴과 데이터 클래스가 바라는 대로인지 살피는 입력 살피기를 으뜸 함수나 클래스에 더하여라. 올바르지 않은 입력에는 알아듣기 쉬운 어긋남 알림을 띄워라.

</div>

??? success "연습문제 2 풀이"
    `forward` 방법(또는 알맞은 함수)의 소개에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 살핌을 더한다. 꼴을 살피려면 중요한 차수를 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아듣기 쉬운 어긋남 알림은 디버깅을 크게 앞당기고 코드를 되쓰기 든든하게 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현이 무너질 만한 결 둘을 밝히고, 저마다 어떻게 짚어내고 고칠지 밝혀라.

</div>

??? success "연습문제 3 풀이"
    흔히 무너지는 결은 이렇다. (1) **그래디언트가 사라지거나 터짐** -- 그래디언트 크기를 지켜보아 짚어낸다(`torch.nn.utils.clip_grad_norm_`이나 켜마다 `param.grad.norm()` 적기). 그래디언트 자르기, 더 나은 첫자리 잡기(Xavier/Kaiming), 구조 고치기(나머지 이음, 잣대 잡기)로 고친다. (2) **지나치게 맞추기** -- 학습 로스는 줄어드는데 따짐 로스가 오르면 짚어낸다. 다독임(드롭아웃, 짐 줄이기, 데이터 증강)이나 모델 크기 줄이기로 고친다. 학습과 따짐 자를 늘 함께 지켜보아 이를 일찍 잡아야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`ImageEncoder`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 파라메터를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 구조를 짜라. 켜 2개, 4개, 8개로 시험하여라.

</div>

??? success "연습문제 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 파라메터를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = ImageEncoder(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — CLIP

이 구현은 클래스 3개(`ImageEncoder`, `TextEncoder`, `CLIP`)를 매기고, 이들이 어울려 온전한 보기 트랜스포머 구조를 이룬다.

핵심 클래스는 `ImageEncoder`, `TextEncoder`, `CLIP`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
