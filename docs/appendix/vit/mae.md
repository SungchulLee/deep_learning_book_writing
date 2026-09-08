# MAE

MAE은 2021년 글 "Masked Autoencoders Are Scalable Vision Learners"에서 나왔다. - 그림 조각 대부분을 가린다(예: 75%) - 보이는 조각만 부호로 바꾼다 - 가벼운 풀개가 가린 조각을 되살린다.

여기 짜보기는 MAE을 짧고 배우기 좋게 보인 본이다. 코드는 고갱이 얼개와 앞으로 걸음에 마음을 두어, 고갱이 꾸밈새를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
MAE - 가린 제 부호기
글: "가린 제 부호기는 크게 키울 수 있는 보기 배움꾼이다" (2021)
지은이: 카이밍 허 외
고갱이 깨침:
  - 그림 조각 대부분을 가린다(예: 75%)
  - 보이는 조각만 부호로 바꾼다
  - 가벼운 풀개가 가린 조각을 되살린다

두루마리: appendix/vit/mae.py
눈여겨볼 것: 배우기 위한 짜보기다(부호기-풀개 얼개).
"""

import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================


class MAE(nn.Module):
    """
    ViT 결의 부호기와 풀개를 지닌 가린 제 부호기.
    """
    def __init__(self, embed_dim=768, decoder_dim=512, num_patches=196):
        super().__init__()

        # 부호기는 보이는 조각만 다룬다
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=12, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=12)

        # 풀개가 가린 조각을 되살린다
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim, nhead=8, batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=8)

        self.enc_to_dec = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        self.head = nn.Linear(decoder_dim, embed_dim)

    def forward(self, x, mask):
        """
        x   : (B, N, D) 조각 담음
        mask: (B, N) 가린 조각이면 True
        """
        # 보이는 조각만 남긴다
        visible = x[~mask].view(x.size(0), -1, x.size(-1))

        # 보이는 조각을 부호로 바꾼다
        enc = self.encoder(visible)

        # 풀개 차수로 되비춘다
        dec_input = self.enc_to_dec(enc)

        # 되살리기를 위해 가림 낱말을 덧붙인다
        num_masked = mask.sum(dim=1).max()
        mask_tokens = self.mask_token.expand(x.size(0), num_masked, -1)

        dec_input = torch.cat([dec_input, mask_tokens], dim=1)

        # 푼다
        dec = self.decoder(dec_input)

        # 되살린 조각 담음을 미루어 본다
        recon = self.head(dec)
        return recon


if __name__ == "__main__":
    pass```

## 2. 논의

`MAE` 갈래는 PyTorch의 `nn.Module` 낯을 써서 모형 얼개를 담는다. `forward` 방법이 셈 그림을 매기므로 익히는 동안 PyTorch의 autograd가 기울기 셈을 절로 다룬다. 이렇게 묶음으로 나눈 꾸밈 덕에 몫 하나하나를 고치거나 더 큰 흐름에 넣기가 쉽다.

여기 실린 코드는 본보기 짜보기라 다듬기보다 알아보기 쉬움을 앞세운다. 서비스 얼개라면 흔히 섞인 촘촘함 익히기, 흩은 자료 나란히, 더 정교한 자료 불리기를 더한다. 그래도 여기서 보인 얼개의 고갱이 깨침은 크기와 상관없이 그대로다.

## 연습문제

**연습문제 1.**
기본 첫자리로 잡은 `MAE`의 배울 수 있는 매개변수를 모두 세어라. 짐과 치우침을 아울러 켜마다 나누어 적어라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)`마다 짐 매개변수가 `in_features * out_features`개이고 치우침이 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 짐이 `in_c * out_c * k * k`개, 치우침이 `out_c`개다. `nn.Embedding(num, dim)`은 매개변수가 `num * dim`개다. 켜를 모두 더한다. `sum(p.numel() for p in model.parameters())`으로 따져 볼 수 있다.

---

**연습문제 2.**
들임의 꼴과 자료 갈래가 바라는 대로인지 살피는 들임 살피기를 으뜸 함수나 갈래에 더하여라. 올바르지 않은 들임에는 알아듣기 쉬운 어긋남 알림을 띄워라.

??? success "연습문제 2 풀이"
    `forward` 방법(또는 알맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 살핌을 더한다. 꼴을 살피려면 종요로운 차수를 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아듣기 쉬운 어긋남 알림은 벌레잡기를 크게 앞당기고 코드를 되쓰기 든든하게 한다.

---

**연습문제 3.**
이 짜보기가 무너질 만한 결 둘을 밝히고, 저마다 어떻게 짚어내고 고칠지 밝혀라.

??? success "연습문제 3 풀이"
    흔히 무너지는 결은 이렇다. (1) **기울기가 사라지거나 터짐** -- 기울기 크기를 지켜보아 짚어낸다(`torch.nn.utils.clip_grad_norm_`이나 켜마다 `param.grad.norm()` 적기). 기울기 자르기, 더 나은 첫자리 잡기(Xavier/Kaiming), 얼개 고치기(나머지 이음, 잣대 잡기)로 고친다. (2) **지나치게 맞추기** -- 익힘 잃음은 줄어드는데 따짐 잃음이 오르면 짚어낸다. 다독임(드롭아웃, 짐 줄이기, 자료 불리기)이나 모형 크기 줄이기로 고친다. 익힘과 따짐 자를 늘 함께 지켜보아 이를 일찍 잡아야 한다.

---

**연습문제 4.**
`MAE`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 짜라. 켜 2개, 4개, 8개로 시험하여라.

??? success "연습문제 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 매개변수를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = MAE(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — MAE

`MAE` 갈래는 PyTorch의 `nn.Module` 낯을 써서 모형 얼개를 담는다.

고갱이 갈래는 `MAE`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
