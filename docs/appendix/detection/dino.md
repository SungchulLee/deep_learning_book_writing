# DINO

DINO은 2022년 글 "DINO: DETR with Improved Denoising Anchor Boxes for End-to-End Object Detection"에서 나왔다.

여기 짜보기는 DINO을 짧고 배우기 좋게 보인 본이다. 코드는 고갱이 얼개와 앞으로 걸음에 마음을 두어, 고갱이 꾸밈새를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 코드

```python
#!/usr/bin/env python3
"""
DINO - 잡소리 없애는 닻 상자를 나아지게 한 DETR
글: "DINO: 끝에서 끝까지 물체 알아내기를 위해 잡소리 없애는 닻 상자를 나아지게 한 DETR" (2022)
지은이: IDEA Research (펑 리 외)
고갱이 깨침(크게 보아):
  1) 잡소리 없애며 익히기(DN): 시끄러운 참값 물음을 더해 익힘을 든든하게 한다.
  2) 더 나은 물음 첫자리 잡기(흔히 닻이나 기준 점 꼴).
  3) 여러 잣대의 결(참으로는 흔히 일그러뜨리는 눈길 갈래를 쓴다).

두루마리: appendix/detection/dino.py
눈여겨볼 것: 배우기 위한 단순한 짜보기다.
  - "기준 점" 깨침과 DN 결의 물음 불리기 걸개를 보인다.
  - nn.Transformer을 쓴다(일그러뜨리는 눈길이 아니다).
  - 온전한 짝짓기와 잃음의 속내는 건너뛴다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class TinyBackbone(nn.Module):
    """결 그림을 내는 작은 CNN 등뼈(쉽게 하려고 한 잣대만 쓴다)."""
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
    """로짓 밭에서 상자를 다듬는, DETR 집안에서 흔한 솜씨."""
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


class DINO(nn.Module):
    """
    단순하게 만든 DINO 꼴 알아내개:
    - 물체 물음 + "기준 점"(잣대 맞춘 상자)으로 거듭 다듬는 깨침을 보인다
    - 익힘 결에서 골라 쓰는 잡소리 없애기 물음(DN)

    날임:
      - pred_logits: (B, num_queries, num_classes+1)
      - pred_boxes : (B, num_queries, 4), 잣대 맞춤
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

        # 배울 수 있는 물음(속내 담음)
        self.query_embed = nn.Embedding(num_queries, d_model)

        # 기준 점(잣대 맞춘 자리 값에서 닻처럼 첫자리를 잡는다)
        # 참 DINO에서는 풀개 켜마다 이를 미루어 보거나 고칠 수 있다.
        self.refpoint_embed = nn.Embedding(num_queries, 4)

        # 머리
        self.class_head = nn.Linear(d_model, num_classes + 1)
        self.box_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),
        )

        # DN 설정(참값이 주어질 때만 쓴다)
        self.dn_num_queries = dn_num_queries
        self.dn_noise_scale = dn_noise_scale

    def make_denoising_queries(self, gt_boxes: torch.Tensor, gt_labels: torch.Tensor):
        """
        참값에서 잡소리 없애기(DN) 물음을 짓는다:
          - 참값 상자에 시끄러움을 더한다
          - 익히는 동안 덧붙은 물음으로 써서 배움을 든든하게 한다

        gt_boxes: (B, M, 4), 잣대 맞춘 (cx,cy,w,h)
        gt_labels: (B, M), [0, num_classes-1] 안의 갈래 번호

        돌려주는 것:
          dn_query_embed: (Tdn, B, C)
          dn_refpoints : (Tdn, B, 4)
        """
        B, M, _ = gt_boxes.shape
        device = gt_boxes.device

        # 묶음마다 참값 상자를 dn_num_queries개까지 고른다(단순히 잘라 낸다)
        M_use = min(M, self.dn_num_queries)
        boxes = gt_boxes[:, :M_use, :]  # (B, M_use, 4)

        # 잣대 맞춘 밭에서 시끄러움을 더한다(배우기 위함)
        noise = (torch.rand_like(boxes) - 0.5) * self.dn_noise_scale
        noisy_boxes = (boxes + noise).clamp(0.0, 1.0)

        # DN 속내 담음을 짓는다:
        # 참 방법은 이름표와 가림을 담는다. 여기서는 배운 "dn 낱말"을 되풀이해 쓸 뿐이다.
        dn_token = torch.zeros(B, M_use, self.d_model, device=device)
        dn_token = dn_token.permute(1, 0, 2).contiguous()  # (Tdn=M_use, B, C)

        dn_ref = noisy_boxes.permute(1, 0, 2).contiguous()  # (Tdn, B, 4)
        return dn_token, dn_ref

    def forward(self, x, gt_boxes=None, gt_labels=None):
        """
        x: (B, 3, H, W)
        gt_boxes/gt_labels: 골라 쓴다. 익힘 결에서 DN 물음을 더할 때 쓴다
        """
        feat = self.backbone(x)
        B, C, H, W = feat.shape

        # 부호기 들임: 자리 결 그림을 펼친다
        src = feat.flatten(2).permute(2, 0, 1)  # (S=H*W, B, C)

        # 여느 배운 물음
        q_content = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # (T, B, C)
        q_ref = torch.sigmoid(self.refpoint_embed.weight).unsqueeze(1).repeat(1, B, 1)  # (T, B, 4)

        # 골라서 여느 물음 앞에 잡소리 없애기 물음(DN)을 더한다
        if (gt_boxes is not None) and (gt_labels is not None):
            dn_content, dn_ref = self.make_denoising_queries(gt_boxes, gt_labels)
            q_content = torch.cat([dn_content, q_content], dim=0)  # (Tdn+T, B, C)
            q_ref = torch.cat([dn_ref, q_ref], dim=0)              # (Tdn+T, B, 4)

        # 풀개 날임 낱말
        hs = self.transformer(src=src, tgt=q_content)  # (Ttotal, B, C)
        hs = hs.permute(1, 0, 2)                       # (B, Ttotal, C)

        # 갈래 로짓을 미루어 본다
        pred_logits = self.class_head(hs)  # (B, Ttotal, num_classes+1)

        # 상자 차이를 미루어 본 뒤 기준 점 둘레에서 "다듬는다"(DETR 집안에서 흔한 깨침)
        # 참으로는 켜마다 다듬는다. 여기서는 한 번에 다듬는다.
        delta = self.box_head(hs)                 # (B, Ttotal, 4)
        ref = q_ref.permute(1, 0, 2).contiguous() # (B, Ttotal, 4)

        # 로짓 밭에서 다듬기: inv_sigmoid(ref) + delta -> sigmoid
        pred_boxes = torch.sigmoid(inverse_sigmoid(ref) + delta)  # (B, Ttotal, 4)

        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}


if __name__ == "__main__":
    model = DINO(num_classes=20, num_queries=300)

    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print("no-DN pred_logits:", y["pred_logits"].shape)
    print("no-DN pred_boxes :", y["pred_boxes"].shape)

    # 거짓 참값 상자와 이름표로 DN 결을 보이는 보기
    gt_boxes = torch.rand(2, 10, 4)     # (B=2, M=10, 4) normalized
    gt_labels = torch.randint(0, 20, (2, 10))
    y_dn = model(x, gt_boxes=gt_boxes, gt_labels=gt_labels)
    print("DN pred_logits:", y_dn["pred_logits"].shape)
    print("DN pred_boxes :", y_dn["pred_boxes"].shape)```

## 논의

이 짜보기는 갈래 2개(`TinyBackbone`, `DINO`)를 매기고, 이들이 어울려 온전한 물체 알아내기 얼개를 이룬다. 갈래마다 남다른 몫을 담아 코드를 묶음으로 나누고 넓히기 쉽게 한다. `forward` 방법이 PyTorch가 절로 미분하는 데 쓰는 셈 그림을 매긴다.

여기 실린 코드는 본보기 짜보기라 다듬기보다 알아보기 쉬움을 앞세운다. 서비스 얼개라면 흔히 섞인 촘촘함 익히기, 흩은 자료 나란히, 더 정교한 자료 불리기를 더한다. 그래도 여기서 보인 얼개의 고갱이 깨침은 크기와 상관없이 그대로다.

## 익힘 문제

**익힘 1.**
`TinyBackbone`의 앞으로 걸음을 따라 텐서의 꼴을 좇아라. 기본 매개변수로 들임 보기 4개를 묶어 넣었을 때, 큰 셈(엮음, 모으기, 선형 켜)마다 꼴이 어떻게 되는지 적어라.

??? success "익힘 1 풀이"
    들임의 꼴에서 비롯해 켜를 차례로 건다. `Conv2d(in_c, out_c, k)`마다 자리 차수는 $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌고(덧대기 없이) `padding=k//2`이면 그대로다. 알갱이 2로 모으면 자리 차수가 반이 된다. 선형 켜는 마지막 차수를 바꾼다. 묶음 차수는 내내 그대로임을 좇아라. 엮음 켜에서는 $(B, C, H, W)$, 편 뒤에는 $(B, F)$으로 가운데 꼴을 적어라.

---

**익힘 2.**
얼개를 크기 $64 \times 64$의 RGB 그림(들임 꼴: $3 \times 64 \times 64$)을 받도록 고쳐라. 켜의 차수를 모두 그에 맞게 손보고 모형이 어긋남 없이 도는지 따져라.

??? success "익힘 2 풀이"
    첫 엮음 켜의 `in_channels`을 이제 값에서 3으로 바꾼다. 엮음과 모으기 켜마다 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 자리 차수를 다시 셈한다. 첫 선형 켜의 `in_features`을 마지막 엮음/모으기 켜의 편 날임에 맞춘다. `model = TinyBackbone(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`으로 따진다.

---

**익힘 3.**
스스로 눈길의 셈 번거로움을 이음 길이 $n$과 모형 차수 $d$의 함수로 밝혀라. 이것이 긴 이음에 Longformer이나 Linformer 같은 얼개를 이끄는 까닭은 무엇인가?

??? success "익힘 3 풀이"
    여느 스스로 눈길은 $n \times n$ 눈길 행렬을 셈하므로 때는 $O(n^2 d)$, 눈길 짐의 기억은 $O(n^2)$이다. 이음이 길면($n = 4096$ 따위) 감당할 수 없다. Longformer은 그 자리 미닫이 창 눈길($O(n \cdot w \cdot d)$, $w$은 창 크기)과 고른 낱말에 대한 성긴 두루 눈길을 아울러 쓴다. Linformer은 열쇠와 값을 더 낮은 차수 $k \ll n$으로 되비추어 번거로움을 $O(n \cdot k \cdot d)$으로 줄인다. 둘 다 드러내는 힘을 얼마쯤 내주고 긴 들임에서 잘 들게 한다.

---

**익힘 4.**
`TinyBackbone`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 짜라. 켜 2개, 4개, 8개로 시험하여라.

??? success "익힘 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 매개변수를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = TinyBackbone(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.
