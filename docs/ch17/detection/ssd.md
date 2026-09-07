# SSD: 한 방 여러 상자 알아내개
## 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

- SSD 얼개와 여러 잣수 알아내기 전략을 설명한다
- 붙박이 상자(닻 앞선 것)와 SSD에서의 몫을 이해한다
- 특징 지도 여럿을 쓰는 SSD 알아내기 머리를 짠다
- 어려운 음성 캐기 전략과 그 중요함을 설명한다
- SSD와 YOLO를 견주고 그 맞바꿈을 이해한다

## 개요

SSD(한 방 여러 상자 알아내개)는 다음으로 YOLO의 빠르기와 자리 바탕 방법의 정확도 사이 균형을 잡는다:

1. 제안 만들기 단계를 없앤다(한 방).
2. 알아내기에 **여러 잣수 특징 지도**를 쓴다.
3. 자리마다 **붙박이 상자**(닻 앞선 것)를 둔다.

## 여러 잣수 특징 지도

SSD의 핵심 새로움은 점점 작아지는 특징 지도를 써서 여러 잣수에서 물체를 알아내는 것이다:

| 특징 지도 | 크기 | 받는 자리 | 흔한 물체 |
|-------------|------|-----------------|-----------------|
| Conv4_3 | 38×38 | 작음 | 작은 물체 |
| Conv7 | 19×19 | 가운데 | 가운데 크기 물체 |
| Conv8_2 | 10×10 | 큼 | 큰 물체 |
| Conv9_2 | 5×5 | 아주 큼 | 아주 큰 물체 |
| Conv10_2 | 3×3 | 엄청 큼 | 가장 큰 물체 |
| Conv11_2 | 1×1 | 전체 | 장면 수준 |

**왜 여러 잣수인가?**

- **작은 특징 지도**는 받는 자리가 넓다 → 큰 물체를 알아낸다
- **큰 특징 지도**는 자리의 세밀함을 지킨다 → 작은 물체를 알아낸다
- 잣수가 달라도 같은 등뼈를 나눠 쓴다 → 셈이 효율적이다

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class SSDExtraLayers(nn.Module):
    """
    여러 잣수 알아내기를 위해 VGG 바탕 뒤에 붙인 덧누비기 층.
    """
    def __init__(self, in_channels: int = 1024):
        super().__init__()
        
        # Conv8: 19×19 → 10×10
        self.conv8_1 = nn.Conv2d(in_channels, 256, kernel_size=1)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        
        # Conv9: 10×10 → 5×5
        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        
        # Conv10: 5×5 → 3×3
        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3)
        
        # Conv11: 3×3 → 1×1
        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        
        x = F.relu(self.conv8_1(x), inplace=True)
        x = F.relu(self.conv8_2(x), inplace=True)
        features.append(x)
        
        x = F.relu(self.conv9_1(x), inplace=True)
        x = F.relu(self.conv9_2(x), inplace=True)
        features.append(x)
        
        x = F.relu(self.conv10_1(x), inplace=True)
        x = F.relu(self.conv10_2(x), inplace=True)
        features.append(x)
        
        x = F.relu(self.conv11_1(x), inplace=True)
        x = F.relu(self.conv11_2(x), inplace=True)
        features.append(x)
        
        return features
```

## 붙박이 상자(닻 앞선 것)

특징 지도의 자리마다 SSD는 가로세로비와 잣수가 다른 **붙박이 상자** 한 벌을 놓는다.

### 붙박이 상자 자리매김

크기가 $f_k$인 특징 지도마다:

**잣수**:

$$s_k = s_{min} + \frac{s_{max} - s_{min}}{m - 1}(k - 1), \quad k \in [1, m]$$

where $s_{min} = 0.2$ and $s_{max} = 0.9$.

**가로세로 견줌**: ${1, 2, 3, 1/2, 1/3}$에 잣대가 $\sqrt{s_k \cdot s_{k+1}}$인 상자 하나를 더한다.

```python
class DefaultBoxGenerator:
    """SSD 붙박이 상자를 만든다."""
    def __init__(
        self,
        image_size: int = 300,
        feature_maps: List[int] = [38, 19, 10, 5, 3, 1],
        aspect_ratios: List[List[float]] = None
    ):
        self.image_size = image_size
        self.feature_maps = feature_maps
        
        if aspect_ratios is None:
            self.aspect_ratios = [
                [1, 2, 1/2],
                [1, 2, 3, 1/2, 1/3],
                [1, 2, 3, 1/2, 1/3],
                [1, 2, 3, 1/2, 1/3],
                [1, 2, 1/2],
                [1, 2, 1/2],
            ]
        else:
            self.aspect_ratios = aspect_ratios
        
        # 잣수 셈하기
        m = len(feature_maps)
        self.scales = [0.2 + (0.9 - 0.2) * k / (m - 1) for k in range(m)]
        self.scales.append(1.0)
    
    def generate(self, device: torch.device) -> torch.Tensor:
        default_boxes = []
        
        for k, f_k in enumerate(self.feature_maps):
            s_k = self.scales[k]
            s_k_prime = (self.scales[k] * self.scales[k + 1]) ** 0.5
            
            for i in range(f_k):
                for j in range(f_k):
                    cx = (j + 0.5) / f_k
                    cy = (i + 0.5) / f_k
                    
                    default_boxes.append([cx, cy, s_k, s_k])
                    default_boxes.append([cx, cy, s_k_prime, s_k_prime])
                    
                    for ar in self.aspect_ratios[k]:
                        if ar != 1:
                            w = s_k * (ar ** 0.5)
                            h = s_k / (ar ** 0.5)
                            default_boxes.append([cx, cy, w, h])
        
        return torch.tensor(default_boxes, device=device).clamp(0, 1)
```

## 알아내기 머리

```python
class SSDHead(nn.Module):
    """특징 지도 하나를 위한 SSD 알아내기 머리."""
    def __init__(self, in_channels: int, num_boxes: int, num_classes: int):
        super().__init__()
        
        self.loc_conv = nn.Conv2d(in_channels, num_boxes * 4, 3, padding=1)
        self.conf_conv = nn.Conv2d(in_channels, num_boxes * num_classes, 3, padding=1)
        self.num_classes = num_classes
        self.num_boxes = num_boxes
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        
        loc = self.loc_conv(x).permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        conf = self.conf_conv(x).permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes)
        
        return loc, conf
```

## SSD 손실 함수

SSD는 어려운 음성 캐기를 곁들인 여러 일 손실을 쓴다:

$$L = \frac{1}{N}(L_{conf} + \alpha L_{loc})$$

### 어려운 음의 보기 캐기

가장 어려운 음성(손실이 가장 큰 것)을 골라 음성과 양성의 비를 3:1로 지킨다.

```python
class SSDLoss(nn.Module):
    def __init__(self, num_classes: int, neg_pos_ratio: float = 3.0):
        super().__init__()
        self.num_classes = num_classes
        self.neg_pos_ratio = neg_pos_ratio
    
    def forward(self, loc_pred, conf_pred, loc_target, conf_target):
        pos_mask = conf_target > 0
        num_pos = pos_mask.sum(dim=1)
        
        # 자리 잡기 손실
        loc_loss = F.smooth_l1_loss(
            loc_pred[pos_mask],
            loc_target[pos_mask],
            reduction='sum'
        )
        
        # 어려운 음성 캐기를 곁들인 믿음도 손실
        conf_loss_all = F.cross_entropy(
            conf_pred.view(-1, self.num_classes),
            conf_target.view(-1),
            reduction='none'
        ).view(conf_pred.size(0), -1)
        
        # 어려운 음성 캐기
        conf_loss_neg = conf_loss_all.clone()
        conf_loss_neg[pos_mask] = 0
        
        _, loss_idx = conf_loss_neg.sort(dim=1, descending=True)
        _, idx_rank = loss_idx.sort(dim=1)
        
        num_neg = torch.clamp(num_pos * self.neg_pos_ratio, max=conf_pred.size(1) - 1).long()
        neg_mask = idx_rank < num_neg.unsqueeze(1)
        
        conf_loss = (conf_loss_all * (pos_mask.float() + neg_mask.float())).sum()
        
        N = max(num_pos.sum().item(), 1)
        return loc_loss / N, conf_loss / N
```

## 미리 익힌 SSD 쓰기

```python
import torchvision
from torchvision.models.detection import ssd300_vgg16

model = ssd300_vgg16(weights='DEFAULT')
model.eval()

with torch.no_grad():
    predictions = model([torch.rand(3, 300, 300)])

boxes = predictions[0]['boxes']
scores = predictions[0]['scores']
labels = predictions[0]['labels']
```

## SSD와 YOLO 견줌

| 갈래 | SSD | YOLO |
|--------|-----|------|
| **여러 잣수** | 그렇다(6가지 잣수) | v3 이후: 그렇다 |
| **빠르기** | 초당 약 46틀 | 초당 약 45틀 |
| **작은 물체** | 더 낫다 | 어렵다 |
| **얼개** | VGG-16 | Darknet |

## 요약

SSD는 한 방 알아내기에 핵심적인 새로움을 들여왔다:

1. **여러 잣수 특징 지도**: 잣수가 다른 물체를 효율적으로 알아낸다
2. **붙박이 상자**: 자리마다 미리 정해 둔 닻
3. **어려운 음성 캐기**: 갈래 치우침을 다룬다
4. **끝에서 끝까지 익히기**: 그물 하나로 모든 어림을 낸다

SSD는 빠르기와 정확도 사이 균형을 잘 잡아 실시간 쓰임새에 알맞다.

## 참고 문헌

1. Liu, W., et al. (2016). SSD: Single Shot MultiBox Detector. *ECCV*.
2. Fu, C.Y., et al. (2017). DSSD: Deconvolutional Single Shot Detector. *arXiv*.

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
