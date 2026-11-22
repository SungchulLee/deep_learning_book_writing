# DETR

DETR은 2020년 논문 "End-to-End Object Detection with Transformers"에서 나왔다. CNN 백본, 트랜스포머 인코더/디코더, 고정된 개수의 객체 쿼리(object query, 디코더가 객체 하나씩을 찾아 내도록 트레인되는 벡터) 집합으로 이루어지며, 상자와 클래스의 집합을 직접 예측한다(원 논문의 DETR에는 앵커도, 겹친 상자를 걸러 내는 NMS도 없다).

여기 구현은 DETR을 짧게 줄인 학습용 예제다. 코드는 핵심 구조와 순전파(forward pass)에 초점을 두어, 핵심 구성 요소를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
DETR - 트랜스포머를 이용한 엔드투엔드 객체 탐지
논문: "End-to-End Object Detection with Transformers" (2020)
저자: Nicolas Carion 외
핵심: CNN 백본 + 트랜스포머 인코더/디코더 + 고정된 개수의 객체 쿼리 집합.
     상자와 클래스의 집합을 직접 예측한다(원래 형태에는 앵커도 NMS도 없다).

파일: appendix/detection/detr.py
참고: 모델의 순전파에 초점을 둔, 트레인용의 단순한 구현이다.
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
    피처 맵을 위한 단순한 2차원 사인-코사인 위치 인코딩.
    실제 DETR은 학습된 인코딩이나 사인-코사인 인코딩을 쓴다. 여기서는 간결한 사인-코사인 변형이다.
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

        # [0, 2pi] 범위로 정규화한다
        y = y / (H - 1 + 1e-6) * 2 * math.pi
        x = x / (W - 1 + 1e-6) * 2 * math.pi

        # 주파수를 계산한다
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
    피처 맵을 출력하는 아주 작은 CNN 백본.
    실제 DETR에서는 ResNet + d_model로 가는 1x1 사영.
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
    단순화한 DETR:
    - 백본 -> 피처 맵
    - 위치 인코딩을 더한다
    - 트랜스포머 인코더/디코더
    - 객체 쿼리 -> 디코더 출력
    - 헤드: 클래스 로짓 + 상자 좌표

    출력:
      - pred_logits: (B, num_queries, num_classes+1)  (+1은 "객체 없음")
      - pred_boxes : (B, num_queries, 4), 정규화된 cx,cy,w,h (시그모이드)
    """
    def __init__(self, num_classes=20, num_queries=100, d_model=256, nhead=8, num_enc=6, num_dec=6):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.d_model = d_model

        self.backbone = TinyBackbone(d_model=d_model)
        self.pos_enc = PositionalEncoding2D(d_model=d_model)

        # PyTorch 트랜스포머는 시퀀스 차원을 앞에 둔다: (S, B, E)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_enc,
            num_decoder_layers=num_dec,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=False,
        )

        # 트레인 가능한 객체 쿼리 (num_queries, d_model)
        self.query_embed = nn.Embedding(num_queries, d_model)

        # 예측 헤드
        self.class_head = nn.Linear(d_model, num_classes + 1)  # +1: "no-object"
        self.box_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),
        )

    def forward(self, x):
        """
        x: (B, 3, H, W)

        단계:
        1) 백본 -> feat (B, C, H', W')
        2) 공간 차원을 평탄화 -> src 시퀀스 (S=H'*W', B, C)
        3) 쿼리 시퀀스를 만든다 (T=num_queries, B, C)
        4) transformer(src, tgt) -> hs (T, B, C)
        5) 헤드 -> 로짓/상자
        """
        feat = self.backbone(x)               # (B, C, H', W')
        pos = self.pos_enc(feat)              # (B, C, H', W')

        B, C, H, W = feat.shape
        src = feat.flatten(2).permute(2, 0, 1)  # (S=H*W, B, C)
        pos = pos.flatten(2).permute(2, 0, 1)   # (S, B, C)

        # src에 위치 인코딩을 더한다(DETR에서 흔한 기법)
        src = src + pos

        # 객체 쿼리를 초기 타깃 토큰으로 쓴다 (T, B, C)
        query = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)

        # 트랜스포머: 인코더가 src를 처리하고, 디코더가 src에 어텐션을 주며 쿼리를 처리한다
        hs = self.transformer(src=src, tgt=query)  # (T, B, C)

        # 배치를 앞에 두도록 바꾼다: (B, T, C)
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

**출력:**

```
pred_logits: torch.Size([2, 100, 21])
pred_boxes : torch.Size([2, 100, 4])
```

## 2. 논의

이 구현은 클래스 3개(`PositionalEncoding2D`, `TinyBackbone`, `DETR`)를 정의하고, 이들이 함께 완전한 객체 탐지 구조를 이룬다. 클래스마다 고유한 역할을 맡아 코드를 모듈로 나누고 확장하기 쉽게 한다. `forward` 메서드가 PyTorch의 자동 미분이 쓰는 계산 그래프를 정의한다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 트레인에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 트레인, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`PositionalEncoding2D`의 순전파를 따라 텐서의 형태를 추적하라. 기본 파라메터로 입력 샘플 4개를 한 배치로 넣었을 때, 주요 연산(컨볼루션, 풀링, 선형 레이어)마다 형태가 어떻게 되는지 적어라.

</div>

??? success "풀이"
    입력의 형태에서 출발해 레이어를 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 $H_{\text{out}} = H_{\text{in}} - k + 1$로 바뀌고(패딩 없이) `padding=k//2`이면 그대로다. 커널 크기 2로 풀링하면 공간 차원이 반이 된다. 선형 레이어는 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임을 확인하라. 컨볼루션 레이어에서는 $(B, C, H, W)$, 평탄화한 뒤에는 $(B, F)$로 중간 형태를 적어라.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
구조를 크기 $64 \times 64$의 RGB 이미지(입력 형태: $3 \times 64 \times 64$)를 받도록 고쳐라. 레이어의 차원을 모두 그에 맞게 조정하고 모델이 오류 없이 실행되는지 확인하라.

</div>

??? success "풀이"
    첫 컨볼루션 레이어의 `in_channels`가 3인지 확인한다(이 쪽의 모델은 이미 RGB 3채널을 받는다). 컨볼루션과 풀링 레이어마다 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$로 공간 차원을 다시 계산한다. 첫 선형 레이어의 `in_features`를 마지막 컨볼루션/풀링 레이어의 평탄화된 출력에 맞춘다. `model = PositionalEncoding2D(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`로 확인한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
입력과 출력의 차원이 같을 때 일반 컨볼루션과 깊이별 분리 컨볼루션(depthwise separable convolution)의 파라메터 수와 부동소수점 연산 횟수(FLOPs)를 비교하여라. 계산이 가장 크게 절약되는 때는 언제인가?

</div>

??? success "풀이"
    일반 `Conv2d(C_in, C_out, k)`의 파라메터는 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개다. 깊이별 분리 컨볼루션은 이를 (1) 깊이별(depthwise): $C_{{\text{{in}}}} \times k^2$개(입력 채널마다 필터 하나)와 (2) 점별(pointwise): $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 컨볼루션)로 나눈다. 파라메터 수의 비는 대략 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 파라메터가 약 $8{-}9\times$ 적다. $C_{{\text{{out}}}}$과 $k$가 모두 클 때 가장 크게 절약된다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`PositionalEncoding2D`를 레이어나 블록의 수를 설정할 수 있게 확장하라. `__init__`에 `num_layers` 파라메터를 추가하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 구성하라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩된 레이어를 다음으로 대체한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 반복한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 파라메터를 모두 최적화 대상으로 등록한다. `for n in [2, 4, 8]: model = PositionalEncoding2D(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`로 시험한다.

## 정리하며

**다룬 것** — DETR

이 구현은 클래스 3개(`PositionalEncoding2D`, `TinyBackbone`, `DETR`)를 정의하고, 이들이 함께 완전한 객체 탐지 구조를 이룬다.

핵심 클래스는 `PositionalEncoding2D`, `TinyBackbone`, `DETR`이며 앞의 연습문제 4개로 스스로 점검해 볼 수 있다.
