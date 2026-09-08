# DETR

DETR은 2020년 글 "End-to-End Object Detection with Transformers"에서 나왔다. CNN 등뼈 + 변환기 부호기/풀개 + 붙박인 물체 물음 묶음. 상자와 갈래를 곧바로 한 벌로 미루어 본다(본디 꼴에는 닻도 NMS도 없다).

여기 짜보기는 DETR을 짧고 배우기 좋게 보인 본이다. 코드는 고갱이 얼개와 앞으로 걸음에 마음을 두어, 고갱이 꾸밈새를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
DETR - 변환기로 끝에서 끝까지 물체 알아내기
글: "변환기로 끝에서 끝까지 물체 알아내기" (2020)
지은이: 니콜라 카리옹 외
고갱이: CNN 등뼈 + 변환기 부호기/풀개 + 붙박인 물체 물음 묶음.
     상자와 갈래를 곧바로 한 벌로 미루어 본다(본디 꼴에는 닻도 NMS도 없다).

두루마리: appendix/detection/detr.py
눈여겨볼 것: 모형의 앞으로 걸음에 마음을 둔, 배우기 위한 단순한 짜보기다.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class PositionalEncoding2D(nn.Module):
    """
    결 그림을 위한 단순한 2차원 사인-코사인 자리 담기.
    참 DETR은 배운 담음이나 사인-코사인 담음을 쓴다. 여기서는 옹골찬 사인-코사인 갈래다.
    """
    def __init__(self, d_model: int = 256, temperature: int = 10000):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 for 2D sine-cos encoding.")
        self.d_model = d_model
        self.temperature = temperature

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (B, C, H, W)
        B, C, H, W = feat.shape
        device = feat.device

        y = torch.arange(H, device=device).unsqueeze(1).repeat(1, W)  # (H, W)
        x = torch.arange(W, device=device).unsqueeze(0).repeat(H, 1)  # (H, W)

        # [0, 2pi] 사이로 잣대를 맞춘다
        y = y / (H - 1 + 1e-6) * 2 * math.pi
        x = x / (W - 1 + 1e-6) * 2 * math.pi

        # 잦기를 셈한다
        dim_t = torch.arange(self.d_model // 4, device=device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (self.d_model // 2))

        # (H, W, D/4)
        pos_x = x[..., None] / dim_t
        pos_y = y[..., None] / dim_t

        # 사인-코사인 짝
        pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=-1).flatten(-2)  # (H, W, D/2)
        pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=-1).flatten(-2)  # (H, W, D/2)

        pos = torch.cat((pos_y, pos_x), dim=-1)  # (H, W, D)
        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1)  # (B, D, H, W)
        return pos


class TinyBackbone(nn.Module):
    """
    결 그림을 내는 아주 작은 CNN 등뼈.
    참 DETR에서는 ResNet + d_model으로 가는 1x1 되비춤.
    """
    def __init__(self, d_model=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4
            nn.Conv2d(128, d_model, 3, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)  # (B, d_model, H/4, W/4)


class DETR(nn.Module):
    """
    단순하게 만든 DETR:
    - 등뼈 -> 결 그림
    - 자리 담기를 더한다
    - 변환기 부호기/풀개
    - 물체 물음 -> 풀개 날임
    - 머리: 갈래 로짓 + 상자 자리 값

    날임:
      - pred_logits: (B, num_queries, num_classes+1)  (+1은 "물체 없음")
      - pred_boxes : (B, num_queries, 4), 잣대 맞춘 cx,cy,w,h (시그모이드)
    """
    def __init__(self, num_classes=20, num_queries=100, d_model=256, nhead=8, num_enc=6, num_dec=6):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.d_model = d_model

        self.backbone = TinyBackbone(d_model=d_model)
        self.pos_enc = PositionalEncoding2D(d_model=d_model)

        # PyTorch 변환기는 이음을 앞에 둔다: (S, B, E)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_enc,
            num_decoder_layers=num_dec,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=False,
        )

        # 배울 수 있는 물체 물음 (num_queries, d_model)
        self.query_embed = nn.Embedding(num_queries, d_model)

        # 미루어 봄 머리
        self.class_head = nn.Linear(d_model, num_classes + 1)  # +1: "no-object"
        self.box_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),
        )

    def forward(self, x):
        """
        x: (B, 3, H, W)

        걸음:
        1) 등뼈 -> feat (B, C, H', W')
        2) 자리를 펼침 -> src 이음 (S=H'*W', B, C)
        3) 물음 이음을 짓는다 (T=num_queries, B, C)
        4) transformer(src, tgt) -> hs (T, B, C)
        5) 머리 -> 로짓/상자
        """
        feat = self.backbone(x)               # (B, C, H', W')
        pos = self.pos_enc(feat)              # (B, C, H', W')

        B, C, H, W = feat.shape
        src = feat.flatten(2).permute(2, 0, 1)  # (S=H*W, B, C)
        pos = pos.flatten(2).permute(2, 0, 1)   # (S, B, C)

        # src에 자리를 더한다(DETR에서 흔한 솜씨)
        src = src + pos

        # 물체 물음을 첫 과녁 낱말로 쓴다 (T, B, C)
        query = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)

        # 변환기: 부호기가 src을 다루고, 풀개가 src에 눈길을 주며 물음을 다룬다
        hs = self.transformer(src=src, tgt=query)  # (T, B, C)

        # 묶음을 앞에 두도록 바꾼다: (B, T, C)
        hs = hs.permute(1, 0, 2)

        pred_logits = self.class_head(hs)           # (B, T, num_classes+1)
        pred_boxes = torch.sigmoid(self.box_head(hs))  # (B, T, 4) normalized

        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}


if __name__ == "__main__":
    model = DETR(num_classes=20, num_queries=100)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print("pred_logits:", y["pred_logits"].shape)  # (2, 100, 21)
    print("pred_boxes :", y["pred_boxes"].shape)   # (2, 100, 4)
```

## 2. 논의

이 짜보기는 갈래 3개(`PositionalEncoding2D`, `TinyBackbone`, `DETR`)를 매기고, 이들이 어울려 온전한 물체 알아내기 얼개를 이룬다. 갈래마다 남다른 몫을 담아 코드를 묶음으로 나누고 넓히기 쉽게 한다. `forward` 방법이 PyTorch가 절로 미분하는 데 쓰는 셈 그림을 매긴다.

여기 실린 코드는 본보기 짜보기라 다듬기보다 알아보기 쉬움을 앞세운다. 서비스 얼개라면 흔히 섞인 촘촘함 익히기, 흩은 자료 나란히, 더 정교한 자료 불리기를 더한다. 그래도 여기서 보인 얼개의 고갱이 깨침은 크기와 상관없이 그대로다.

## 연습문제

**연습문제 1.**
`PositionalEncoding2D`의 앞으로 걸음을 따라 텐서의 꼴을 좇아라. 기본 매개변수로 들임 보기 4개를 묶어 넣었을 때, 큰 셈(엮음, 모으기, 선형 켜)마다 꼴이 어떻게 되는지 적어라.

??? success "연습문제 1 풀이"
    들임의 꼴에서 비롯해 켜를 차례로 건다. `Conv2d(in_c, out_c, k)`마다 자리 차수는 $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌고(덧대기 없이) `padding=k//2`이면 그대로다. 알갱이 2로 모으면 자리 차수가 반이 된다. 선형 켜는 마지막 차수를 바꾼다. 묶음 차수는 내내 그대로임을 좇아라. 엮음 켜에서는 $(B, C, H, W)$, 편 뒤에는 $(B, F)$으로 가운데 꼴을 적어라.

---

**연습문제 2.**
얼개를 크기 $64 \times 64$의 RGB 그림(들임 꼴: $3 \times 64 \times 64$)을 받도록 고쳐라. 켜의 차수를 모두 그에 맞게 손보고 모형이 어긋남 없이 도는지 따져라.

??? success "연습문제 2 풀이"
    첫 엮음 켜의 `in_channels`을 이제 값에서 3으로 바꾼다. 엮음과 모으기 켜마다 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 자리 차수를 다시 셈한다. 첫 선형 켜의 `in_features`을 마지막 엮음/모으기 켜의 편 날임에 맞춘다. `model = PositionalEncoding2D(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`으로 따진다.

---

**연습문제 3.**
들임과 날임의 차수가 같을 때 여느 엮음과 깊이별로 가른 엮음의 매개변수 수와 뜨는 셈 횟수를 견주어라. 셈을 가장 크게 아끼는 때는 언제인가?

??? success "연습문제 3 풀이"
    여느 `Conv2d(C_in, C_out, k)`의 매개변수는 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개다. 깊이별로 가른 엮음은 이를 (1) 깊이별: $C_{{\text{{in}}}} \times k^2$개(들임 갈래마다 거르개 하나)와 (2) 점별: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 엮음)로 나눈다. 매개변수의 견줌은 어림잡아 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 매개변수가 약 $8{-}9\times$ 적다. $C_{{\text{{out}}}}$과 $k$이 모두 클 때 가장 크게 아낀다.

---

**연습문제 4.**
`PositionalEncoding2D`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 짜라. 켜 2개, 4개, 8개로 시험하여라.

??? success "연습문제 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 매개변수를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = PositionalEncoding2D(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — DETR

이 짜보기는 갈래 3개(`PositionalEncoding2D`, `TinyBackbone`, `DETR`)를 매기고, 이들이 어울려 온전한 물체 알아내기 얼개를 이룬다.

고갱이 갈래는 `PositionalEncoding2D`, `TinyBackbone`, `DETR`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
