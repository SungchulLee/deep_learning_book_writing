# DeiT

DeiT은 2021년 글 "Training data-efficient image transformers & distillation through attention"에서 나왔다. - 앎 옮기기로 더 적은 자료로 ViT을 익힌다 - 갈래 낱말에 더해 *옮김 낱말*을 둔다.

여기 짜보기는 DeiT을 짧고 배우기 좋게 보인 본이다. 코드는 고갱이 얼개와 앞으로 걸음에 마음을 두어, 고갱이 꾸밈새를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
DeiT - 자료를 아끼는 그림 변환기
글: "자료를 아끼는 그림 변환기 익히기와 눈길을 거친 앎 옮기기" (2021)
지은이: 위고 투브롱 외
고갱이 깨침:
  - 앎 옮기기로 더 적은 자료로 ViT을 익힌다
  - 갈래 낱말에 더해 *옮김 낱말*을 둔다

두루마리: appendix/vit/deit.py
눈여겨볼 것: 배우기 위한 짜보기다(앞으로 걸음만 담았다).
"""

import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================


class PatchEmbedding(nn.Module):
    """그림을 조각으로 나누어 담는다."""
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
    DeiT = ViT + 옮김 낱말.

    낱말:
      - [CLS] 낱말: 여느 가름 낱말
      - [DIST] 낱말: 스승의 미루어 봄에서 배운다
    """
    def __init__(self, num_classes=1000, embed_dim=768, num_patches=196):
        super().__init__()

        self.patch_embed = PatchEmbedding(embed_dim=embed_dim)

        # 배울 수 있는 낱말
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 자리 담기에 두 낱말이 모두 들어간다
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))

        # 변환기 부호기(단순하게 만듦)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=12, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=12)

        # 머리 둘: 하나는 가름, 하나는 앎 옮기기
        self.head_cls = nn.Linear(embed_dim, num_classes)
        self.head_dist = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)

        x = self.patch_embed(x)  # (B, N, D)

        # 낱말을 묶음 크기로 넓힌다
        cls = self.cls_token.expand(B, -1, -1)
        dist = self.dist_token.expand(B, -1, -1)

        # 낱말과 조각 담음을 이어 붙인다
        x = torch.cat([cls, dist, x], dim=1)
        x = x + self.pos_embed

        # 변환기 부호로 바꾸기
        x = self.encoder(x)

        # 날임을 갈라 낸다
        cls_out = x[:, 0]
        dist_out = x[:, 1]

        # 미루어 봄 머리 둘
        logits_cls = self.head_cls(cls_out)
        logits_dist = self.head_dist(dist_out)

        return logits_cls, logits_dist


if __name__ == "__main__":
    pass```

## 2. 논의

이 짜보기는 갈래 2개(`PatchEmbedding`, `DeiT`)를 매기고, 이들이 어울려 온전한 보기 변환기 얼개를 이룬다. 갈래마다 남다른 몫을 담아 코드를 묶음으로 나누고 넓히기 쉽게 한다. `forward` 방법이 PyTorch가 절로 미분하는 데 쓰는 셈 그림을 매긴다.

여기 실린 코드는 본보기 짜보기라 다듬기보다 알아보기 쉬움을 앞세운다. 서비스 얼개라면 흔히 섞인 촘촘함 익히기, 흩은 자료 나란히, 더 정교한 자료 불리기를 더한다. 그래도 여기서 보인 얼개의 고갱이 깨침은 크기와 상관없이 그대로다.

## 연습문제

**연습문제 1.**
`PatchEmbedding`의 앞으로 걸음을 따라 텐서의 꼴을 좇아라. 기본 매개변수로 들임 보기 4개를 묶어 넣었을 때, 큰 셈(엮음, 모으기, 선형 켜)마다 꼴이 어떻게 되는지 적어라.

??? success "연습문제 1 풀이"
    들임의 꼴에서 비롯해 켜를 차례로 건다. `Conv2d(in_c, out_c, k)`마다 자리 차수는 $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌고(덧대기 없이) `padding=k//2`이면 그대로다. 알갱이 2로 모으면 자리 차수가 반이 된다. 선형 켜는 마지막 차수를 바꾼다. 묶음 차수는 내내 그대로임을 좇아라. 엮음 켜에서는 $(B, C, H, W)$, 편 뒤에는 $(B, F)$으로 가운데 꼴을 적어라.

---

**연습문제 2.**
얼개를 크기 $64 \times 64$의 RGB 그림(들임 꼴: $3 \times 64 \times 64$)을 받도록 고쳐라. 켜의 차수를 모두 그에 맞게 손보고 모형이 어긋남 없이 도는지 따져라.

??? success "연습문제 2 풀이"
    첫 엮음 켜의 `in_channels`을 이제 값에서 3으로 바꾼다. 엮음과 모으기 켜마다 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 자리 차수를 다시 셈한다. 첫 선형 켜의 `in_features`을 마지막 엮음/모으기 켜의 편 날임에 맞춘다. `model = PatchEmbedding(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`으로 따진다.

---

**연습문제 3.**
스스로 눈길의 셈 번거로움을 이음 길이 $n$과 모형 차수 $d$의 함수로 밝혀라. 이것이 긴 이음에 Longformer이나 Linformer 같은 얼개를 이끄는 까닭은 무엇인가?

??? success "연습문제 3 풀이"
    여느 스스로 눈길은 $n \times n$ 눈길 행렬을 셈하므로 때는 $O(n^2 d)$, 눈길 짐의 기억은 $O(n^2)$이다. 이음이 길면($n = 4096$ 따위) 감당할 수 없다. Longformer는 그 자리 미닫이 창 눈길($O(n \cdot w \cdot d)$, $w$은 창 크기)과 고른 낱말에 대한 성긴 두루 눈길을 아울러 쓴다. Linformer는 열쇠와 값을 더 낮은 차수 $k \ll n$으로 되비추어 번거로움을 $O(n \cdot k \cdot d)$으로 줄인다. 둘 다 드러내는 힘을 얼마쯤 내주고 긴 들임에서 잘 들게 한다.

---

**연습문제 4.**
`PatchEmbedding`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 짜라. 켜 2개, 4개, 8개로 시험하여라.

??? success "연습문제 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 매개변수를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = PatchEmbedding(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — DeiT

이 짜보기는 갈래 2개(`PatchEmbedding`, `DeiT`)를 매기고, 이들이 어울려 온전한 보기 변환기 얼개를 이룬다.

고갱이 갈래는 `PatchEmbedding`, `DeiT`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
