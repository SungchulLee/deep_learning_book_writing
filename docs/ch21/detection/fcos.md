# FCOS: 앵커 없이 위치별로 예측하는 객체 탐지

FCOS는 피처 맵의 각 위치에서 클래스와 경계 상자의 네 변까지 거리를 예측하는 단일 단계 탐지기다. 미리 정한 너비·높이의 앵커 집합은 사용하지 않지만, 피처 stride·위치 좌표·대상 크기 범위·양성 할당 규칙은 필요하다. [원 논문](https://arxiv.org/abs/1904.01355)의 anchor-free라는 표현이 하이퍼파라미터나 NMS가 없다는 뜻은 아니다.

## 1. 거리 표현과 양성 위치

이미지 좌표 (x,y)에서 왼쪽·위·오른쪽·아래 거리 l,t,r,b를 예측하면 경계 상자는 (x−l,y−t,x+r,y+b)다. 피처 격자 좌표를 이미지 좌표로 옮기는 stride와 중심 오프셋을 정하고, 거리가 이미지 픽셀 단위인지 stride로 정규화한 값인지 맞춰야 한다.

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

정답 상자 내부라는 조건만으로 모든 할당이 결정되지는 않는다. 피처 단계별 회귀 거리 범위와 여러 객체가 겹칠 때의 선택 규칙이 필요하다. 중심 주변만 양성으로 쓰는 center sampling도 구현별 선택이다. 배경과 무시할 위치의 레이블을 별도로 정의한다.

## 2. 분류·회귀·centerness head

입력 피처 목록의 각 원소가 B×C×H×W이면 아래 head는 B×K×H×W 클래스 로짓, B×4×H×W 거리 값, B×1×H×W centerness 로짓을 목록으로 반환한다. 분류·회귀 tower는 모든 단계에서 웨이트를 공유한다. `GroupNorm(32,C)`는 C가 32로 나누어져야 하며 선언된 scale이 다섯 개라 입력 피처는 최대 다섯 단계다. 필요한 torch·nn·F·math import도 별도로 준비해야 한다.

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
        
        # 클래스 매기기 가지
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
        
        # 초점 로스를 위한 치우침 첫자리매김
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)
    
    def forward(self, features: list) -> tuple:
        """
        인수:
            features: FPN에서 온 피처 지도의 목록
            
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
            
            # 가운데다움(되돌리기 피처에서 어림)
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

코드 주석에는 exp가 적혀 있지만 실제 회귀는 ReLU 뒤에 학습 가능한 scale을 곱한다. scale은 1×1 컨볼루션을 1에 적용한 웨이트+바이어스로, 양수 제약이 없다. 따라서 최종 거리가 항상 비음수라고 보장할 수 없으며 원 논문의 양수 거리 파라메터화와 동일하지 않다. 분류 bias 초기화는 prior_prob에 대응하는 초기 sigmoid 값을 만들며 0<prior_prob<1이 필요하다.

양성 위치의 centerness 목표는 다음과 같다.

$$c^*=\sqrt{\frac{\min(l,r)}{\max(l,r)}\frac{\min(t,b)}{\max(t,b)}}.$$

상자 중심이면 1이고 경계 쪽에서 작아진다. 이는 기하학적 목표이지 상자가 정답과 갖는 IoU 그 자체가 아니다. 음수 거리나 상자 밖 위치에는 이 식을 그대로 적용하지 않는다. 코드의 ε 때문에 정확한 중심에서도 계산값이 1보다 조금 작을 수 있다. 추론에서는 예측 centerness와 클래스 점수를 정한 규칙으로 결합하고 NMS를 적용한다.

## 3. 로스와 목표 텐서의 조건

일반적인 목적은 분류 focal loss, 양성 위치의 상자 회귀 로스, centerness BCE의 합이다. 계수·정규화·회귀 가중 방식은 구현에 따라 명시해야 한다. 아래 코드는 세 로스를 같은 계수로 합하며 양성 수로 나눈다.

```python
class FCOSLoss(nn.Module):
    """FCOS 로스 함수."""
    
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
        """FCOS 로스를 셈한다."""
        
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
        
        # 클래스 매기기 로스(초점 로스)
        cls_loss = sigmoid_focal_loss(
            all_cls_scores,
            labels,
            reduction='sum'
        ) / num_pos
        
        if pos_mask.sum() > 0:
            # 되돌리기 로스(겹침 비 로스)
            reg_loss = iou_loss(
                all_bbox_preds[pos_mask],
                bbox_targets[pos_mask],
                reduction='sum'
            ) / num_pos
            
            # 가운데다움 로스(두 클래스 엇갈린 엔트로피)
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

이 로스는 미리 할당한 `targets`를 요구하며 할당 과정 자체는 없다. 피처 단계→배치→공간 순서로 펼친 예측과 같은 순서로 목표를 정렬해야 한다. `labels>0`은 배경 0, 양성 클래스 1 이상을 가정한다. 하지만 일반적인 sigmoid focal loss는 N×K 로짓과 같은 모양의 이진 클래스 목표를 받는다. 정의되지 않은 `sigmoid_focal_loss`가 1차원 클래스 ID를 어떻게 처리하는지 없으므로, torchvision 함수를 그대로 대입하면 완성되는 코드가 아니다. 양성 마스크용 ID와 분류용 one-hot 목표를 구분해야 한다.

양성이 없을 때 회귀·centerness는 0으로 처리하지만 분류 로스는 여전히 num_pos=0으로 나누므로 문제가 남는다. 유효한 정규화 규칙이 필요하다. `iou_loss`도 별도 정의가 없고 거리 표현을 직접 받는지, 좌표로 디코딩한 상자를 받는지 정해야 한다. 이 예제를 완성된 학습기로 설명하지 않는다.

## 4. 앵커 기반 방식과 비교

앵커 기반 탐지는 기준 상자의 크기·비율과 정답 매칭을 설계한다. FCOS는 위치·거리 표현으로 바꾸지만 크기별 단계 할당과 겹친 정답 처리·후처리는 남는다. 둘 다 축 정렬 직사각형을 예측하는 경우 임의 모양의 물체 윤곽을 직접 출력하는 것은 아니다. 모델의 속도·정확도는 backbone·해상도·학습·후처리와 함께 비교하며 anchor-free라는 사실만으로 더 빠르거나 더 정확하다고 단정하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
위치 (40,50)과 거리 (10,20,30,20)의 상자 및 centerness를 ε 없이 계산하라.

</div>

??? success "풀이"

    상자는 (30,30,70,70)이다. centerness는 sqrt((10/30)(20/20))=1/√3이다. y축 중앙에 있지만 x축 중앙은 아니므로 값이 1보다 작다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
ReLU 뒤의 scale이 음수가 되면 어떤 문제가 생기는가?

</div>

??? success "풀이"

    ReLU 출력은 비음수지만 음수 scale을 곱하면 음수 거리가 생긴다. x−l, x+r 등의 디코딩이 의도한 상자를 보장하지 못한다. 코드에 적힌 exp 주석만으로 양수 보장을 주장할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
FCOS에도 NMS가 필요한 이유와 아래 NMS 조각의 의존 함수를 설명하라.

</div>

??? success "풀이"

    이웃 위치가 같은 객체를 예측할 수 있어 점수·겹침 기준으로 중복 후보를 정리한다. anchor-free는 NMS-free와 다르다. 아래 NumPy 조각에는 compute_iou 정의가 별도로 필요하고 임계값과 같은 IoU는 유지한다. FCOS 출력은 먼저 같은 이미지 좌표로 디코딩해야 한다.

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
FCOSLoss의 양성 없는 배치와 분류 목표 모양을 점검하라.

</div>

??? success "풀이"

    회귀·centerness의 분기와 달리 분류 항은 num_pos=0으로 나눈다. 정규화 분모의 하한 또는 다른 일관된 규칙이 필요하다. labels의 정수 ID는 양성 선택에 쓸 수 있지만 sigmoid focal loss에 맞는 N×K 이진 목표와는 다르다. 무시 레이블 처리도 정의해야 한다.



## 정리하며

FCOS는 앵커 크기 집합 대신 위치별 네 거리와 centerness로 탐지한다. 좌표·거리 단위, 양성 할당, scale의 부호, 목표 텐서 순서·모양과 양성 없는 배치 처리가 실제 구현의 핵심이다. 현재 코드는 head와 로스의 골격이며 완성된 탐지 파이프라인은 아니다.

**참고 문헌**

1. Tian, Z., et al. (2019). FCOS: Fully Convolutional One-Stage Object Detection. ICCV.
2. Lin, T.-Y., et al. (2017). Feature Pyramid Networks for Object Detection. CVPR.
