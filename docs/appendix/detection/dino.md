# DINO

DINO는 2022년 논문 "DINO: DETR with Improved Denoising Anchor Boxes for End-to-End Object Detection"에서 나왔다. DETR 계열의 탐지기로, 잡음을 넣은 정답 상자를 디코더가 원래대로 되돌리게 하는 디노이징 트레인과 앵커 상자 형태의 쿼리를 더해 DETR의 느린 수렴을 개선한다.

여기 구현은 DINO를 짧게 줄인 학습용 예제다. 코드는 핵심 구조와 순전파(forward pass)에 초점을 두어, 핵심 구성 요소를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
DINO - 디노이징 앵커 박스를 개선한 DETR
논문: "DINO: DETR with Improved Denoising Anchor Boxes for End-to-End Object Detection" (2022)
저자: IDEA Research (Feng Li 외)
핵심 아이디어(개략):
  1) 디노이징 트레인(DN): 잡음을 넣은 정답(GT) 쿼리를 더해 트레인을 안정시킨다.
  2) 더 나은 쿼리 초기화(흔히 앵커나 기준점 형태).
  3) 멀티스케일 피처(실제로는 흔히 디포머블 어텐션 변형을 쓴다).

파일: appendix/detection/dino.py
참고: 트레인용의 단순한 구현이다.
  - "기준점" 아이디어와 DN 방식의 쿼리 증강 훅을 보인다.
  - nn.Transformer를 쓴다(디포머블 어텐션이 아니다).
  - 완전한 매칭과 로스의 세부는 생략한다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class TinyBackbone(nn.Module):
    """피처 맵을 출력하는 작은 CNN 백본(단순화를 위해 단일 스케일만 쓴다)."""
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


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """로짓 공간에서 상자를 정제하는, DETR 계열에서 흔한 기법."""
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


class DINO(nn.Module):
    """
    단순화한 DINO 방식 탐지기:
    - 객체 쿼리 + "기준점"(정규화된 상자)으로 반복 정제하는 아이디어를 보인다
    - 트레인 모드에서 선택적으로 쓰는 디노이징 쿼리(DN)

    출력:
      - pred_logits: (B, num_queries, num_classes+1)
      - pred_boxes : (B, num_queries, 4), 정규화됨
    """
    def __init__(
        self,
        num_classes=20,
        num_queries=300,
        d_model=256,
        nhead=8,
        num_enc=6,
        num_dec=6,
        dn_num_queries=50,     # number of denoising queries (educational default)
        dn_noise_scale=0.4,    # how much noise to add to GT boxes (educational)
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.d_model = d_model

        self.backbone = TinyBackbone(d_model=d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_enc,
            num_decoder_layers=num_dec,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=False,
        )

        # 트레인 가능한 쿼리(콘텐츠 임베딩)
        self.query_embed = nn.Embedding(num_queries, d_model)

        # 기준점(정규화된 좌표에서 앵커처럼 초기화한다)
        # 실제 DINO에서는 디코더 레이어마다 이를 예측하거나 갱신할 수 있다.
        self.refpoint_embed = nn.Embedding(num_queries, 4)

        # 헤드
        self.class_head = nn.Linear(d_model, num_classes + 1)
        self.box_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),
        )

        # DN 설정(정답이 주어질 때만 쓴다)
        self.dn_num_queries = dn_num_queries
        self.dn_noise_scale = dn_noise_scale

    def make_denoising_queries(self, gt_boxes: torch.Tensor, gt_labels: torch.Tensor):
        """
        정답에서 디노이징(DN) 쿼리를 만든다:
          - 정답 상자에 잡음을 더한다
          - 트레인하는 동안 추가 쿼리로 써서 트레인을 안정시킨다

        gt_boxes: (B, M, 4), 정규화된 (cx,cy,w,h)
        gt_labels: (B, M), [0, num_classes-1] 안의 클래스 번호

        반환값:
          dn_query_embed: (Tdn, B, C)
          dn_refpoints : (Tdn, B, 4)
        """
        B, M, _ = gt_boxes.shape
        device = gt_boxes.device

        # 배치마다 정답 상자를 dn_num_queries개까지 고른다(단순히 잘라 낸다)
        M_use = min(M, self.dn_num_queries)
        boxes = gt_boxes[:, :M_use, :]  # (B, M_use, 4)

        # 정규화된 좌표 공간에서 잡음을 더한다(트레인용)
        noise = (torch.rand_like(boxes) - 0.5) * self.dn_noise_scale
        noisy_boxes = (boxes + noise).clamp(0.0, 1.0)

        # DN 콘텐츠 임베딩을 만든다:
        # 실제 방법은 라벨과 마스크를 인코딩한다. 여기서는 트레인된 "dn 토큰"을 되풀이해 쓸 뿐이다.
        dn_token = torch.zeros(B, M_use, self.d_model, device=device)
        dn_token = dn_token.permute(1, 0, 2).contiguous()  # (Tdn=M_use, B, C)

        dn_ref = noisy_boxes.permute(1, 0, 2).contiguous()  # (Tdn, B, 4)
        return dn_token, dn_ref

    def forward(self, x, gt_boxes=None, gt_labels=None):
        """
        x: (B, 3, H, W)
        gt_boxes/gt_labels: 선택 인자. 트레인 모드에서 DN 쿼리를 더할 때 쓴다
        """
        feat = self.backbone(x)
        B, C, H, W = feat.shape

        # 인코더 입력: 공간 피처 맵을 평탄화한다
        src = feat.flatten(2).permute(2, 0, 1)  # (S=H*W, B, C)

        # 일반 학습된 쿼리
        q_content = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # (T, B, C)
        q_ref = torch.sigmoid(self.refpoint_embed.weight).unsqueeze(1).repeat(1, B, 1)  # (T, B, 4)

        # 선택적으로 일반 쿼리 앞에 디노이징 쿼리(DN)를 덧붙인다
        if (gt_boxes is not None) and (gt_labels is not None):
            dn_content, dn_ref = self.make_denoising_queries(gt_boxes, gt_labels)
            q_content = torch.cat([dn_content, q_content], dim=0)  # (Tdn+T, B, C)
            q_ref = torch.cat([dn_ref, q_ref], dim=0)              # (Tdn+T, B, 4)

        # 디코더 출력 토큰
        hs = self.transformer(src=src, tgt=q_content)  # (Ttotal, B, C)
        hs = hs.permute(1, 0, 2)                       # (B, Ttotal, C)

        # 클래스 로짓을 예측한다
        pred_logits = self.class_head(hs)  # (B, Ttotal, num_classes+1)

        # 상자 변위를 예측한 뒤 기준점 둘레에서 "정제한다"(DETR 계열에서 흔한 아이디어)
        # 실제로는 레이어마다 정제한다. 여기서는 한 번에 정제한다.
        delta = self.box_head(hs)                 # (B, Ttotal, 4)
        ref = q_ref.permute(1, 0, 2).contiguous() # (B, Ttotal, 4)

        # 로짓 공간에서 정제: inv_sigmoid(ref) + delta -> sigmoid
        pred_boxes = torch.sigmoid(inverse_sigmoid(ref) + delta)  # (B, Ttotal, 4)

        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}


if __name__ == "__main__":
    model = DINO(num_classes=20, num_queries=300)

    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print("no-DN pred_logits:", y["pred_logits"].shape)
    print("no-DN pred_boxes :", y["pred_boxes"].shape)

    # 가짜 정답 상자와 라벨로 DN 모드를 보이는 예
    gt_boxes = torch.rand(2, 10, 4)     # (B=2, M=10, 4) normalized
    gt_labels = torch.randint(0, 20, (2, 10))
    y_dn = model(x, gt_boxes=gt_boxes, gt_labels=gt_labels)
    print("DN pred_logits:", y_dn["pred_logits"].shape)
    print("DN pred_boxes :", y_dn["pred_boxes"].shape)
```

**출력:**

```
no-DN pred_logits: torch.Size([2, 300, 21])
no-DN pred_boxes : torch.Size([2, 300, 4])
DN pred_logits: torch.Size([2, 310, 21])
DN pred_boxes : torch.Size([2, 310, 4])
```

## 2. 논의

이 구현은 클래스 2개(`TinyBackbone`, `DINO`)를 정의하고, 이들이 함께 완전한 객체 탐지 구조를 이룬다. 클래스마다 고유한 역할을 맡아 코드를 모듈로 나누고 확장하기 쉽게 한다. `forward` 메서드가 PyTorch의 자동 미분이 쓰는 계산 그래프를 정의한다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 트레인에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 트레인, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`TinyBackbone`의 순전파를 따라 텐서의 형태를 추적하라. 기본 파라메터로 입력 샘플 4개를 한 배치로 넣었을 때, 주요 연산(컨볼루션, 풀링, 선형 레이어)마다 형태가 어떻게 되는지 적어라.

</div>

??? success "풀이"
    입력의 형태에서 출발해 레이어를 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 $H_{\text{out}} = H_{\text{in}} - k + 1$로 바뀌고(패딩 없이) `padding=k//2`이면 그대로다. 커널 크기 2로 풀링하면 공간 차원이 반이 된다. 선형 레이어는 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임을 확인하라. 컨볼루션 레이어에서는 $(B, C, H, W)$, 평탄화한 뒤에는 $(B, F)$로 중간 형태를 적어라.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
구조를 크기 $64 \times 64$의 RGB 이미지(입력 형태: $3 \times 64 \times 64$)를 받도록 고쳐라. 레이어의 차원을 모두 그에 맞게 조정하고 모델이 오류 없이 실행되는지 확인하라.

</div>

??? success "풀이"
    첫 컨볼루션 레이어의 `in_channels`가 3인지 확인한다(이 쪽의 모델은 이미 RGB 3채널을 받는다). 컨볼루션과 풀링 레이어마다 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$로 공간 차원을 다시 계산한다. 첫 선형 레이어의 `in_features`를 마지막 컨볼루션/풀링 레이어의 평탄화된 출력에 맞춘다. `model = TinyBackbone(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`로 확인한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
셀프 어텐션의 계산 복잡도를 시퀀스 길이 $n$과 모델 차원 $d$의 함수로 구하라. 이것이 긴 시퀀스에 Longformer나 Linformer 같은 구조를 쓰게 되는 까닭은 무엇인가?

</div>

??? success "풀이"
    일반 셀프 어텐션은 $n \times n$ 어텐션 행렬을 계산하므로 시간 복잡도는 $O(n^2 d)$, 어텐션 웨이트의 메모리는 $O(n^2)$이다. 시퀀스가 길면($n = 4096$ 따위) 감당할 수 없다. Longformer는 국소 슬라이딩 윈도 어텐션($O(n \cdot w \cdot d)$, $w$는 윈도 크기)과 선택된 토큰에 대한 희소한 전역 어텐션을 함께 쓴다. Linformer는 키와 밸류를 더 낮은 차원 $k \ll n$으로 사영하여 복잡도를 $O(n \cdot k \cdot d)$로 줄인다. 둘 다 표현력을 얼마쯤 내주고 긴 입력에서 효율을 얻는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`TinyBackbone`을 레이어나 블록의 수를 설정할 수 있게 확장하라. `__init__`에 `num_layers` 파라메터를 추가하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 구성하라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩된 레이어를 다음으로 대체한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 반복한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 파라메터를 모두 최적화 대상으로 등록한다. `for n in [2, 4, 8]: model = TinyBackbone(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`로 시험한다.

## 정리하며

**다룬 것** — DINO

이 구현은 클래스 2개(`TinyBackbone`, `DINO`)를 정의하고, 이들이 함께 완전한 객체 탐지 구조를 이룬다.

핵심 클래스는 `TinyBackbone`, `DINO`이며 앞의 연습문제 4개로 스스로 점검해 볼 수 있다.
