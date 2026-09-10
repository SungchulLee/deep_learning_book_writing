# FCOS: 온통 누비기인 한 단계 알아내기
FCOS는 자리마다 상자 모서리까지의 거리를 되돌려 상자를 어림한다:

```
┌─────────────────────────────────────────────────────────────────┐
│                      FCOS Architecture                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  For each pixel location (x, y):                                │
│                                                                  │
│       l ←───┬───→ r                                             │
│             │                                                   │
│       t     ●     (x, y)  feature map location                 │
│       ↑     │                                                   │
│       │     ↓                                                   │
│             b                                                   │
│                                                                  │
│  Predict:                                                        │
│    • (l, t, r, b): Distances to box edges                       │
│    • class score: What object (if any)?                         │
│    • centerness: How close to object center?                    │
│                                                                  │
│  Box = (x - l, y - t, x + r, y + b)                             │
└─────────────────────────────────────────────────────────────────┘
```

### FCOS 짜기

```python
class FCOSHead(nn.Module):
    """
    가운데다움 어림을 갖춘 FCOS 알아내기 머리.
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_convs: int = 4,
        prior_prob: float = 0.01
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # 갈래 매기기 가지
        cls_tower = []
        for _ in range(num_convs):
            cls_tower.append(nn.Conv2d(in_channels, in_channels, 3, padding=1))
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU(inplace=True))
        self.cls_tower = nn.Sequential(*cls_tower)
        
        # 되돌리기 가지
        reg_tower = []
        for _ in range(num_convs):
            reg_tower.append(nn.Conv2d(in_channels, in_channels, 3, padding=1))
            reg_tower.append(nn.GroupNorm(32, in_channels))
            reg_tower.append(nn.ReLU(inplace=True))
        self.reg_tower = nn.Sequential(*reg_tower)
        
        # 내놓는 머리
        self.cls_logits = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, 3, padding=1)
        
        # FPN 켜마다 배울 수 있는 잣수
        self.scales = nn.ModuleList([nn.Conv2d(1, 1, 1) for _ in range(5)])
        
        # 초점 손실을 위한 치우침 첫자리매김
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)
    
    def forward(self, features: list) -> tuple:
        """
        인수:
            features: FPN에서 온 특징 지도의 목록
            
        반환값:
            cls_scores: 켜마다의 (B, num_classes, H, W) 목록
            bbox_preds: 켜마다의 (B, 4, H, W) 목록
            centernesses: 켜마다의 (B, 1, H, W) 목록
        """
        cls_scores = []
        bbox_preds = []
        centernesses = []
        
        for i, feature in enumerate(features):
            cls_feat = self.cls_tower(feature)
            reg_feat = self.reg_tower(feature)
            
            # 분류
            cls_score = self.cls_logits(cls_feat)
            cls_scores.append(cls_score)
            
            # 양수를 얻으려 exp를 쓴 상자 되돌리기
            bbox_pred = self.bbox_pred(reg_feat)
            bbox_pred = F.relu(bbox_pred) * self.scales[i](torch.ones(1, 1, 1, 1, device=feature.device))
            bbox_preds.append(bbox_pred)
            
            # 가운데다움(되돌리기 특징에서 어림)
            centerness = self.centerness(reg_feat)
            centernesses.append(centerness)
        
        return cls_scores, bbox_preds, centernesses

def compute_centerness(
    left: torch.Tensor,
    top: torch.Tensor,
    right: torch.Tensor,
    bottom: torch.Tensor
) -> torch.Tensor:
    """
    가운데다움 목표를 셈한다.
    
    가운데다움은 어떤 자리가 물체 가운데에 얼마나 가까운지를 잰다.
    0(모서리)부터 1(가운데)까지이다.
    
    가운데다움 = sqrt(min(l, r) / max(l, r) * min(t, b) / max(t, b))
    """
    lr_min = torch.min(left, right)
    lr_max = torch.max(left, right)
    tb_min = torch.min(top, bottom)
    tb_max = torch.max(top, bottom)
    
    centerness = torch.sqrt(
        (lr_min / (lr_max + 1e-6)) * (tb_min / (tb_max + 1e-6))
    )
    
    return centerness
```

### FCOS 익히기

**양성 표본 배정**:

- 참값 상자 안에 있는 자리는 양성이다
- 여러 잣수 배정: FPN 켜마다 다른 물체 크기를 맡는다

**손실 함수**:

$$L = L_{cls} + \lambda_1 L_{reg} + \lambda_2 L_{centerness}$$

```python
class FCOSLoss(nn.Module):
    """FCOS 손실 함수."""
    
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(
        self,
        cls_scores: list,
        bbox_preds: list,
        centernesses: list,
        targets: dict
    ) -> dict:
        """FCOS 손실을 셈한다."""
        
        # 어림 펴기
        all_cls_scores = torch.cat([
            s.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            for s in cls_scores
        ])
        all_bbox_preds = torch.cat([
            b.permute(0, 2, 3, 1).reshape(-1, 4)
            for b in bbox_preds
        ])
        all_centernesses = torch.cat([
            c.permute(0, 2, 3, 1).reshape(-1)
            for c in centernesses
        ])
        
        # 목표 얻기
        labels = targets['labels']
        bbox_targets = targets['bbox_targets']
        centerness_targets = targets['centerness_targets']
        
        # 양성 마스크
        pos_mask = labels > 0
        num_pos = pos_mask.sum().float()
        
        # 갈래 매기기 손실(초점 손실)
        cls_loss = sigmoid_focal_loss(
            all_cls_scores,
            labels,
            reduction='sum'
        ) / num_pos
        
        if pos_mask.sum() > 0:
            # 되돌리기 손실(겹침 비 손실)
            reg_loss = iou_loss(
                all_bbox_preds[pos_mask],
                bbox_targets[pos_mask],
                reduction='sum'
            ) / num_pos
            
            # 가운데다움 손실(두 갈래 엇갈린 엔트로피)
            centerness_loss = F.binary_cross_entropy_with_logits(
                all_centernesses[pos_mask],
                centerness_targets[pos_mask],
                reduction='sum'
            ) / num_pos
        else:
            reg_loss = all_bbox_preds.sum() * 0
            centerness_loss = all_centernesses.sum() * 0
        
        return {
            'cls_loss': cls_loss,
            'reg_loss': reg_loss,
            'centerness_loss': centerness_loss,
            'total': cls_loss + reg_loss + centerness_loss
        }
```

---

## 1. 견줌: 닻 바탕과 닻 없음

| 갈래 | 닻 바탕 | 닻 없음 |
|--------|-------------|-------------|
| 웃매개변수 | 닻 크기, 비, 겹침 비 문턱값 | 더 적다(닻 꾸밈 없음) |
| 유연함 | 붙박이 가로세로비 | 아무 꼴이나 |
| 복잡도 | 닻 짝짓기 논리 | 더 단순한 익히기 |
| 빠르기 | NMS 필요 | 일부 방법은 NMS 없이 |
| 정확도 | 무르익고 잘 다듬어짐 | 견줄 만함 |

### 무엇을 언제 쓸 것인가

- **닻 바탕**: 물체 크기와 비를 아는, 잘 이해된 문제
- **닻 없음**: 새로운 갈래, 흔치 않은 가로세로비, 빠른 시제품 만들기

---

## 연습문제

**연습문제 1.**
한 단계 알아내개와 두 단계 알아내개의 차이를 설명하여라. 빠르기와 정확도 사이의 근본 맞바꿈은 무엇인가?

??? success "연습문제 1 풀이"
    **두 단계 알아내개**(보기로 더 빠른 R-CNN)는 먼저 자리 제안을 만들고 제안마다 갈래를 매기고 다듬는다. 정확하지만 제안마다 다루기 때문에 느리다. **한 단계 알아내개**(보기로 YOLO, SSD)는 특징 지도에서 두름 상자와 갈래 확률을 한 번에 곧바로 어림하여 정확도를 조금 내주고 훨씬 빠른 미룸을 얻는다. 맞바꿈은 이렇다. 두 단계 알아내개는 작고 겹치는 물체를 잘 알아내지만 초당 5~15틀로 돌고, 한 단계 알아내개는 mAP가 조금 낮은 대신 초당 30~155틀 넘게 낸다.

---

**연습문제 2.**
겹침 비(교집합 나누기 합집합) 식을 이끌어 내고 두름 상자를 값매김할 때 왜 단순한 L2 거리보다 낫게 여기는지 설명하여라.

??? success "연습문제 2 풀이"
    두 두름 상자 $A$과 $B$에 대해:

    $$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}$$

    겹침 비를 낫게 여기는 까닭은 이렇다. (1) 잣수에 안 바뀐다(화소 10개의 어긋남은 큰 물체보다 작은 물체에 더 크게 다가온다). (2) 자연스레 $[0, 1]$에 놓여 좋음 점수로 읽을 수 있다. (3) 상자 자리표 사이의 L2 거리는 겹침을 담아내지 못해 두 상자의 L2 거리가 작아도 겹침이 0일 수 있다(보기로 하나가 다른 하나 안에 있는 경우와 나란히 놓인 경우).

---

**연습문제 3.**
최대가 아닌 것 누르기(NMS)를 짜고 알아내기 물길에서 그것이 하는 몫을 설명하여라.

??? success "연습문제 3 풀이"
    ```python
    import numpy as np

    def nms(boxes, scores, iou_threshold=0.5):
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            remaining = order[1:]
            ious = compute_iou(boxes[i], boxes[remaining])
            mask = ious <= iou_threshold
            order = remaining[mask]
        return keep
    ```
    NMS는 같은 물체를 거듭 알아낸 것을 없앤다. 후보 상자에 점수를 매긴 뒤 점수가 가장 높은 상자를 고르고 겹침 비가 문턱값을 넘는 상자(겹친 것일 가능성이 높다)를 모두 없애기를 되풀이한다.

---

**연습문제 4.**
물체 알아내기의 갈래 치우침 문제와 초점 손실이 그것을 어떻게 다루는지 설명하여라.

??? success "연습문제 4 풀이"
    한 단계 알아내개에서는 닻 상자 대부분이 바탕(쉬운 아님 보기)이고 물체를 담은 것은 몇 안 된다. 여느 엇결 엔트로피 잃음은 쉬운 아님 보기가 워낙 많아 그쪽에 휘둘리므로 어려운 맞음 보기의 기울기 신호가 묻힌다. **초점 잃음**은 조절 값을 더한다. $\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$이다. $\gamma > 0$이면 쉬운 보기($p_t$이 높은 것)의 짐이 지수로 줄어 어려운 보기에 익힘이 몰린다. $\gamma = 2$과 $\alpha = 0.25$을 쓰면 RetinaNet은 한 단계의 빠르기를 지키면서 두 단계 알아내개에 맞먹는 맞음을 이룬다.

## 정리하며

이 마당은 견줌: 닻 바탕과 닻 없음을 차례로 짚었다.

**참고 문헌**

1. Tian, Z., et al. (2019). FCOS: Fully Convolutional One-Stage Object Detection. ICCV.
2. Lin, T.-Y., et al. (2017). Feature Pyramid Networks for Object Detection. CVPR.
