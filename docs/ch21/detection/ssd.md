# SSD: 기본 상자와 다중 해상도 객체 탐지

SSD는 여러 해상도의 특징 맵에서 기본 상자(default box)별 위치 보정과 클래스를 예측하는 단일 단계 탐지기다. 영역 제안 단계를 두지 않지만 기본 상자와 정답의 매칭·학습 손실·추론 후처리는 필요하다. SSD의 특징을 단순히 YOLO와 두 단계 방식의 속도·정확도 중간값으로 정의하지 않는다.

## 1. 여러 해상도의 특징

대표적인 SSD300은 300×300 입력에서 38·19·10·5·3·1 크기의 특징 맵을 사용한다. 큰 특징 맵은 위치를 더 촘촘하게 표본화하고, 깊은 단계의 작은 특징 맵은 넓은 수용 영역을 활용한다. 그러나 어느 단계가 특정 크기 객체만 담당하거나 마지막 1×1 특징이 장면 분류만 한다는 뜻은 아니다. 기본 상자의 크기와 정답 매칭도 담당 객체에 영향을 준다.

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

`SSDExtraLayers`는 전체 backbone이 아니다. B×1024×19×19 특징을 받으면 차례로 B×512×10×10, B×256×5×5, B×256×3×3, B×256×1×1을 반환한다. 마지막 두 3×3 합성곱에는 padding이 없어 크기가 2씩 줄어든다. 앞의 38×38·19×19 특징은 backbone에서 별도로 받아야 여섯 단계가 된다. 임의의 작은 특징을 넣으면 마지막 합성곱을 적용할 수 없을 수 있다.

## 2. 기본 상자의 크기와 순서

기본 상자는 위치마다 미리 정한 중심·크기·종횡비를 갖는 기준이다. 모델은 이 기준에 대한 오프셋을 학습하므로 ‘고정’은 최종 예측 상자가 움직이지 않는다는 뜻이 아니다. 정규화 좌표에서 단계 k의 기본 스케일을 s_k, 종횡비를 a라 하면 너비는 s_k√a, 높이는 s_k/√a다. 비율 1에는 기하평균 스케일 √(s_k s_{k+1})의 정사각형을 하나 더 둔다.

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

이 생성기는 각 특징 단계에서 행→열→기본 상자 순으로 `[cx,cy,w,h]`를 만든다. 기본 단계별 상자 수는 [4,6,6,6,4,4]이고 총수는 38²×4+19²×6+10²×6+5²×6+3²×4+1²×4=8732다. head를 펼치는 순서와 반드시 맞춰야 한다.

코드는 스케일을 0.2~0.9로 선형 배치하고 마지막에 1.0을 추가한다. 이는 이 예제의 설정이며 원 SSD300이나 라이브러리의 모든 세부 스케일을 그대로 재현한다고 보지 않는다. `image_size`는 저장만 하므로 인수만 바꿔도 특징 크기·스케일이 자동 조정되지는 않는다. 단계가 하나면 스케일 계산의 m−1 분모도 문제가 된다.

반환값의 `clamp(0,1)`은 중심·너비·높이 각각을 자른다. xyxy 모서리를 이미지 경계에 맞춰 자르는 연산이 아니며, 중심과 크기가 각각 0~1이어도 모서리는 밖에 있을 수 있다. 정답과 IoU를 계산할 때는 표현을 변환하고 경계 처리 규칙을 일치시킨다.

## 3. 단계별 탐지 head

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

특징 B×C×H×W와 위치별 기본 상자 수 A에 대해 loc는 B×(HWA)×4, conf는 B×(HWA)×K다. 채널을 공간 뒤로 옮겨 펼치므로 각 위치의 앵커 순서가 생성기 순서와 같아야 한다. K는 배경 포함 클래스 수이고 conf는 아직 softmax 전의 로짓이다. 여러 단계 결과는 같은 순서로 연결한다. 이 head만으로 기본 상자 매칭이나 오프셋 디코딩이 수행되지는 않는다.

## 4. 위치 손실과 hard negative mining

원 SSD는 양성 상자의 smooth L1 위치 손실과 분류 손실을 양성 매칭 수로 정규화한다. 배경 후보 중 분류 손실이 큰 것을 선택하는 hard negative mining으로 쉬운 음성이 압도하는 현상을 줄인다. 음성:양성 3:1은 최대 선택 비율이며 음성이 부족하거나 양성이 없을 때까지 항상 정확한 비율을 만들 수 있는 것은 아니다.

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

목표는 이미 기본 상자와 매칭·인코딩된 loc_target과 배경 0·객체 1 이상인 conf_target이다. 이 함수가 그 목표를 생성하지는 않는다. 반환값은 정규화된 위치·분류 손실 두 개이므로 학습 루프에서 계수를 정해 합해야 한다. N은 배치 전체 양성 수이며 0이면 분모를 1로 둔다. 양성이 없는 이미지는 num_neg도 0이 되어 이 구현에서 배경 분류 학습에 기여하지 않는다.

음성 선택에는 추가 주의가 필요하다. 코드는 양성의 정렬 점수를 0으로 만들지만 `neg_mask`에서 양성을 명시적으로 제외하지 않는다. 요청한 음성 수가 실제 음성 수를 넘으면 양성이 다시 선택되어 분류 손실이 두 번 합산될 수 있다. 무시 레이블 처리도 따로 정의되어 있지 않다. 이 구현을 그대로 완전한 hard negative mining이라고 가정하지 않는다.

## 5. 사전 학습된 라이브러리 모델

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

이 부분은 앞의 조각을 조립한 모델이 아니라 별도의 [torchvision SSD300](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssd300_vgg16.html)을 사용한다. 0~1 범위 C×H×W 텐서의 목록을 받아 추론 시 이미지별 boxes·scores·labels 사전을 반환한다. 가중치를 처음 사용할 때 다운로드가 필요할 수 있다. 예제의 무작위 이미지는 API의 입력 형태를 보여 줄 뿐 탐지 성능을 평가하는 자료가 아니다.

## 6. 다른 탐지기와 비교하기

[SSD 원 논문](https://arxiv.org/abs/1512.02325)의 구조와 다른 YOLO 버전을 비교할 때는 backbone·해상도·학습 데이터·평가 지표·장치·정밀도·NMS 포함 여부를 맞춘다. SSD가 모든 YOLO보다 작은 객체에 강하거나 둘의 속도가 고정 FPS라는 주장은 성립하지 않는다. 특징 해상도와 기본 상자 설계가 작은 객체의 학습·표현에 어떻게 영향을 주는지 비교하는 것이 유용하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
기본 생성기의 기본 상자 수를 계산하고 head와 맞춰야 할 순서를 설명하라.

</div>

??? success "연습문제 1 풀이"

    단계별 [4,6,6,6,4,4]개를 각 특징 위치 수에 곱해 합하면 8732개다. 단계→행→열→기본 상자 순서를 head의 펼친 결과와 일치시킨다. 수만 같고 순서가 다르면 목표 매칭이 어긋난다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
[cx,cy,w,h]를 각각 0~1로 clamp하면 상자 모서리도 0~1인가?

</div>

??? success "연습문제 2 풀이"

    아니다. 예를 들어 cx=0.1,w=0.6이면 왼쪽 모서리는 −0.2다. 중심·크기 표현을 xyxy로 바꾼 뒤 경계 처리 정책을 적용해야 한다. 생성기의 clamp는 일부 큰 너비·높이의 종횡비도 바꿀 수 있다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
SSD head의 loc 출력과 아래 NMS가 요구하는 상자의 차이를 설명하라.

</div>

??? success "연습문제 3 풀이"

    loc는 기본 상자에 대한 오프셋이므로 디코딩과 같은 좌표계로의 변환을 먼저 해야 한다. 아래 NMS는 xyxy 상자와 점수를 가정하며 compute_iou는 별도 정의가 필요하다. 클래스·이미지별 적용과 임계값 경계도 명시한다.

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


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
양성만 많거나 양성이 없는 이미지에서 본문 hard negative mining을 점검하라.

</div>

??? success "연습문제 4 풀이"

    양성이 없으면 음성 선택 수도 0이다. 반대로 양성이 많아 3배 요청 수가 실제 음성 수를 넘으면 정렬에서 점수를 0으로 만든 양성도 neg_mask에 들어갈 수 있다. 실제 음성 수로 제한하고 양성을 명시적으로 제외하는 규칙이 필요하다. 현재 코드는 그대로 보존되어 있어 이 한계를 고려해 읽는다.



## 정리하며

SSD의 다중 해상도 특징·기본 상자·head·목표 매칭은 같은 개수와 순서를 공유해야 한다. 거리 오프셋과 이미지 좌표, 중심·크기 clamp와 모서리 clipping을 구분한다. Hard negative mining의 양성 제외와 빈 양성 사례도 실제 손실을 좌우한다.

**참고 문헌**

1. Liu, W., et al. (2016). SSD: Single Shot MultiBox Detector. *ECCV*.
2. Fu, C.Y., et al. (2017). DSSD: Deconvolutional Single Shot Detector. *arXiv*.
