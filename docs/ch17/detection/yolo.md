# YOLO: 한 번만 본다
## 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

- YOLO의 철학과 두 단계 알아내개와의 차이를 설명한다
- 격자 바탕 알아내기와 닻 상자 어림을 이해한다
- YOLOv1에서 YOLOv8까지의 흐름을 좇는다
- YOLO 방식의 알아내기 머리와 손실 함수를 짠다
- 미리 익힌 YOLO 모델을 미룸과 곱게 다듬기에 쓴다
- 실시간 쓰임새에 맞게 YOLO 모델을 다듬는다

## YOLO의 철학

YOLO(한 번만 본다)는 물체 알아내기를 되돌리기 문제 하나로 세워, 온 그림에서 두름 상자와 갈래 확률을 한 번의 값매김으로 곧바로 어림하며 판을 뒤집었다.

### 핵심 통찰

자리를 제안하고 따로 갈래를 매기는 대신, YOLO는 그림을 격자로 나누고 모든 상자와 갈래를 한꺼번에 어림한다:

```
┌─────────────────────────────────────────────────────────────────┐
│                     YOLO Detection Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input Image (448×448)                                          │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────┐                    │
│  │        Single CNN Forward Pass           │                    │
│  │    (Feature extraction + prediction)     │                    │
│  └────────────────────┬────────────────────┘                    │
│                       │                                          │
│                       ▼                                          │
│  ┌─────────────────────────────────────────┐                    │
│  │           S × S Grid Output              │                    │
│  │    Each cell: B boxes + C classes       │                    │
│  │    Shape: (S, S, B×5 + C)               │                    │
│  └────────────────────┬────────────────────┘                    │
│                       │                                          │
│                       ▼                                          │
│              NMS + Final Detections                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 두 단계 알아내개보다 나은 점

| 갈래 | YOLO | 두 단계(더 빠른 R-CNN) |
|--------|------|--------------------------|
| **빠르기** | 초당 45~155틀 이상 | 초당 5~15틀 |
| **전체 맥락** | 온 그림을 본다 | 제안만 본다 |
| **얼개** | 더 단순하고 하나로 묶임 | 복잡하고 조각이 여럿 |
| **뒷바탕 어긋남** | 헛양성이 적다 | 뒷바탕을 더 헷갈린다 |
| **작은 물체** | 더 어렵다 | FPN을 쓰면 더 낫다 |

## 격자 바탕 알아내기

YOLO는 들임 그림을 S×S 격자로 나눈다. 격자 칸마다 가운데점이 그 칸 안에 떨어지는 물체를 알아낼 몫을 맡는다.

### 칸의 어림

격자 칸마다 YOLO는 다음을 어림한다:

- **두름 상자 B개**: 저마다 (x, y, w, h, 믿음도)를 갖는다
- **갈래 확률 C개**: P(갈래_i | 물체)

```
Grid Cell Output:
┌─────────────────────────────────────────────────────────────┐
│  Box 1: [x₁, y₁, w₁, h₁, conf₁]                            │
│  Box 2: [x₂, y₂, w₂, h₂, conf₂]                            │
│  ...                                                        │
│  Box B: [xB, yB, wB, hB, confB]                             │
│  Classes: [P(c₁|obj), P(c₂|obj), ..., P(cC|obj)]          │
└─────────────────────────────────────────────────────────────┘

Total predictions per cell: B × 5 + C
Total output tensor: S × S × (B × 5 + C)
```

### 자리표 부호화

YOLO는 격자 칸을 기준으로 고르게 맞춘 자리표를 쓴다:

- **(x, y)**: 격자 칸 모서리에서의 어긋남, [0, 1]로 맞춤
- **(w, h)**: 그림 크기에 대한 상대값, [0, 1]로 맞춤
- **믿음도**: P(물체) × IoU(어림, 참값)

```python
import torch


def decode_yolo_boxes(
    predictions: torch.Tensor,
    grid_size: int,
    num_boxes: int,
    image_size: int
) -> torch.Tensor:
    """
    YOLO의 어림을 절대 상자 자리표로 푼다.
    
    인수:
        predictions: (batch, S, S, B*5+C) 날 어림
        grid_size: S(격자 차원)
        num_boxes: B(칸마다의 상자 수)
        image_size: 들임 그림 차원
        
    반환값:
        boxes: xyxy 꼴의 (batch, S*S*B, 4)
    """
    batch_size = predictions.shape[0]
    cell_size = image_size / grid_size
    
    # 격자 어긋남 만들기
    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_size),
        torch.arange(grid_size),
        indexing='ij'
    )
    grid_x = grid_x.to(predictions.device).float()
    grid_y = grid_y.to(predictions.device).float()
    
    boxes = []
    for b in range(num_boxes):
        start_idx = b * 5
        
        # 어림 뽑아내기
        x = predictions[..., start_idx + 0]      # 칸 안의 상대 x
        y = predictions[..., start_idx + 1]      # 칸 안의 상대 y
        w = predictions[..., start_idx + 2]      # 그림에 대한 상대 너비
        h = predictions[..., start_idx + 3]      # 그림에 대한 상대 높이
        
        # 절대 자리표로 바꾸기
        x_abs = (grid_x + x) * cell_size
        y_abs = (grid_y + y) * cell_size
        w_abs = w * image_size
        h_abs = h * image_size
        
        # xyxy 꼴로 바꾸기
        x1 = x_abs - w_abs / 2
        y1 = y_abs - h_abs / 2
        x2 = x_abs + w_abs / 2
        y2 = y_abs + h_abs / 2
        
        box = torch.stack([x1, y1, x2, y2], dim=-1)
        boxes.append(box.reshape(batch_size, -1, 4))
    
    return torch.cat(boxes, dim=1)
```

## YOLO의 흐름

### YOLOv1(2015)

처음의 YOLO는 한 방 알아내기라는 틀을 들여왔다:

- 7×7 격자, 칸마다 상자 2개, 갈래 20개(PASCAL VOC)
- 내놓음: 7×7×30 텐서
- 누비기 층 24개 + 온전히 이은 층 2개
- GPU에서 초당 45틀

**한계**:

- 작은 물체와 무리 지은 물체에 약하다
- 칸마다 상자 2개로 제한된다
- 어림에 자리 제약이 있다

### YOLOv2/YOLO9000(2016)

핵심 나아진 점:

- 모든 누비기 층에 **묶음 고르게 맞추기**
- **높은 해상도 갈래 매개**: 448×448에서 곱게 다듬기
- **닻 상자**: k-평균으로 자료에서 상자의 앞선 것을 배운다
- **지나침 층**: 앞선 층에서 온 결이 고운 특징
- **여러 잣수 익히기**: 여러 해상도로 익힌다

### YOLOv3(2018)

얼개의 큰 바뀜:

- **Darknet-53 등뼈**: 53층 잔차 그물
- **여러 잣수 어림**: 3가지 잣수에서 알아낸다
- **서로 얽히지 않은 로지스틱 갈래 매개**: 여러 이름표에 더 낫다

```python
import torch.nn as nn


class DarknetBlock(nn.Module):
    """
    Darknet 잔차 덩이.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
    
    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class YOLOv3Head(nn.Module):
    """
    잣수 하나를 위한 YOLOv3 알아내기 머리.
    """
    def __init__(
        self,
        in_channels: int,
        num_anchors: int = 3,
        num_classes: int = 80
    ):
        super().__init__()
        
        # 닻마다 어림: 자리표 4 + 물체다움 1 + 갈래 수
        out_channels = num_anchors * (5 + num_classes)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels * 2, out_channels, 1)
        )
        
        self.num_anchors = num_anchors
        self.num_classes = num_classes
    
    def forward(self, x):
        """
        반환값:
            (batch, num_anchors, H, W, 5 + num_classes)
        """
        out = self.conv(x)
        batch, _, H, W = out.shape
        
        out = out.view(batch, self.num_anchors, 5 + self.num_classes, H, W)
        out = out.permute(0, 1, 3, 4, 2)
        
        return out
```

### YOLOv4(2020)

그때의 가장 앞선 재주를 아울렀다:

- **CSPDarknet53 등뼈**: 단계를 가로지르는 부분 이음
- **SPP(자리 피라미드 모으기)**: 여러 잣수 특징 모으기
- **PANet 목**: 경로 모으기 그물
- **앞선 자료 불리기**: 모자이크, CutMix, 스스로 맞서 익히기

### YOLOv5(2020)

쓰기 편함에 초점을 둔 PyTorch 다시 짜기:

- 깔끔한 PyTorch 코드 바탕
- 익히기와 펼치기가 쉽다
- 여러 모델 크기(n, s, m, l, x)
- 안에 갖춘 자료 불리기

### YOLOv6(2022)

산업에 초점을 둔 나아짐:

- EfficientRep 등뼈
- Rep-PAN 목
- 펼치기에 맞게 다듬음

### YOLOv7(2022)

익히기의 새로움:

- 넓힌 효율적 층 모으기(E-ELAN)
- 계획된 매개변수 다시 매기기 누비기
- 크기가 다를 때의 겹친 잣수 맞추기

### YOLOv8(2023)

닻 없는 알아내기를 쓰는 최신 세대:

```python
# ultralytics YOLOv8 쓰기
from ultralytics import YOLO

# 미리 익힌 모델 읽어 들이기
model = YOLO('yolov8n.pt')  # 나노 모델

# 추론
results = model('image.jpg')

# 학습
model.train(data='coco.yaml', epochs=100, imgsz=640)

# 내보내기
model.export(format='onnx')
```

## YOLOv8 얼개

### 등뼈: CSPDarknet

등뼈는 단계를 가로지르는 부분 이음(CSP)을 쓴다:

```
┌─────────────────────────────────────────────────────────────────┐
│                    CSPDarknet Backbone                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input (640×640)                                                 │
│      │                                                          │
│      ▼                                                          │
│  ┌─────────────────┐                                            │
│  │   Stem (Focus)   │ → 80×80                                   │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐                                            │
│  │  CSP Block 1    │ → 80×80                                    │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐                                            │
│  │  CSP Block 2    │ → 40×40  (P3)                              │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐                                            │
│  │  CSP Block 3    │ → 20×20  (P4)                              │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐                                            │
│  │  CSP Block 4    │ → 10×10  (P5)                              │
│  │     + SPPF      │                                            │
│  └─────────────────┘                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 목: FPN + PAN

경로 모으기를 곁들인 특징 피라미드 그물:

```python
class PANNeck(nn.Module):
    """
    여러 잣수 특징을 녹여 붙이는 경로 모으기 그물 목.
    """
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        
        # 위에서 아래로 가는 길(FPN)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, out_channels, 1)
            for ch in in_channels_list
        ])
        
        # 아래에서 위로 가는 길(PAN)
        self.downsample_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
            for _ in range(len(in_channels_list) - 1)
        ])
        
        # 녹여 붙이는 누비기
        self.fusion_convs = nn.ModuleList([
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1)
            for _ in range(len(in_channels_list) - 1)
        ])
    
    def forward(self, features):
        """
        인수:
            features: 특징 지도의 목록 [P3, P4, P5]
            
        반환값:
            잣수마다 녹여 붙인 특징의 목록
        """
        # 옆 이음
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        
        # 위에서 아래로 가는 길
        for i in range(len(laterals) - 1, 0, -1):
            upsampled = nn.functional.interpolate(
                laterals[i], scale_factor=2, mode='nearest'
            )
            laterals[i-1] = laterals[i-1] + upsampled
        
        # 아래에서 위로 가는 길
        outputs = [laterals[0]]
        for i in range(len(laterals) - 1):
            downsampled = self.downsample_convs[i](outputs[-1])
            fused = torch.cat([downsampled, laterals[i+1]], dim=1)
            outputs.append(self.fusion_convs[i](fused))
        
        return outputs
```

### 닻 없는 알아내기 머리

YOLOv8은 머리를 떼어 놓은 닻 없는 방식을 쓴다:

```python
class YOLOv8Head(nn.Module):
    """
    갈래 매기기와 되돌리기를 떼어 놓은 닻 없는 알아내기 머리.
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 80,
        reg_max: int = 16
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.reg_max = reg_max
        
        # 갈래 매기기 가지
        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        self.cls_pred = nn.Conv2d(in_channels, num_classes, 1)
        
        # 되돌리기 가지
        self.reg_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        # 분포 초점 손실: 4 × reg_max개의 값을 어림
        self.reg_pred = nn.Conv2d(in_channels, 4 * reg_max, 1)
    
    def forward(self, x):
        """
        반환값:
            cls_out: (batch, num_classes, H, W)
            reg_out: (batch, 4*reg_max, H, W)
        """
        cls_feat = self.cls_conv(x)
        reg_feat = self.reg_conv(x)
        
        cls_out = self.cls_pred(cls_feat)
        reg_out = self.reg_pred(reg_feat)
        
        return cls_out, reg_out
```

## YOLO 손실 함수

YOLO는 여러 조각으로 된 손실 함수를 쓴다:

### YOLOv1~v3의 손실

$$L = \lambda_{coord} L_{coord} + L_{conf} + L_{cls}$$

**자리표 손실**(물체가 있는 칸에만):

$$L_{coord} = \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 \right]$$

$$+ \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} \left[ (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right]$$

**믿음도 손실**:

$$L_{conf} = \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} (C_i - \hat{C}_i)^2 + \lambda_{noobj} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{noobj} (C_i - \hat{C}_i)^2$$

**갈래 매기기 손실**(물체가 있는 칸에만):

$$L_{cls} = \sum_{i=0}^{S^2} \mathbb{1}_{i}^{obj} \sum_{c \in classes} (p_i(c) - \hat{p}_i(c))^2$$

```python
import torch
import torch.nn.functional as F


def yolo_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    num_boxes: int = 2,
    lambda_coord: float = 5.0,
    lambda_noobj: float = 0.5
) -> torch.Tensor:
    """
    YOLOv1 방식의 손실 함수.
    
    인수:
        predictions: (batch, S, S, B*5 + C)
        targets: [x, y, w, h, obj, classes...]를 담은 (batch, S, S, 5 + C)
        num_classes: 갈래의 개수 C
        num_boxes: 칸마다의 상자 개수 B
        
    반환값:
        전체 손실
    """
    batch_size, S, _, _ = predictions.shape
    
    # 어림 뜯어 읽기
    pred_boxes = []
    pred_confs = []
    for b in range(num_boxes):
        start = b * 5
        pred_boxes.append(predictions[..., start:start+4])
        pred_confs.append(predictions[..., start+4:start+5])
    
    pred_classes = predictions[..., num_boxes*5:]
    
    # 목표 뜯어 읽기
    target_box = targets[..., :4]
    target_obj = targets[..., 4:5]
    target_classes = targets[..., 5:]
    
    # 물체 마스크
    obj_mask = target_obj.squeeze(-1) == 1  # (batch, S, S)
    noobj_mask = target_obj.squeeze(-1) == 0
    
    # 맡은 어림개 찾기(목표와의 겹침 비가 가장 큰 것)
    ious = []
    for pred_box in pred_boxes:
        iou = compute_iou(pred_box, target_box)  # (batch, S, S)
        ious.append(iou)
    
    ious = torch.stack(ious, dim=-1)  # (batch, S, S, B)
    best_box = ious.argmax(dim=-1)  # (batch, S, S)
    
    # 자리표 손실(맡은 어림개만)
    coord_loss = 0
    for b in range(num_boxes):
        responsible = (best_box == b) & obj_mask
        if responsible.sum() > 0:
            pred = pred_boxes[b][responsible]
            target = target_box[responsible]
            
            # xy 손실
            coord_loss += F.mse_loss(pred[:, :2], target[:, :2], reduction='sum')
            
            # 너비·높이 손실(잣수에 안 바뀌도록 제곱근)
            coord_loss += F.mse_loss(
                torch.sqrt(pred[:, 2:4].abs() + 1e-6),
                torch.sqrt(target[:, 2:4].abs() + 1e-6),
                reduction='sum'
            )
    
    coord_loss *= lambda_coord
    
    # 믿음도 손실
    conf_loss = 0
    for b in range(num_boxes):
        responsible = (best_box == b) & obj_mask
        
        # 물체 믿음도
        if responsible.sum() > 0:
            pred_conf = pred_confs[b][responsible]
            target_iou = ious[..., b][responsible]
            conf_loss += F.mse_loss(pred_conf.squeeze(-1), target_iou, reduction='sum')
        
        # 물체 없음 믿음도
        not_responsible = ~responsible & noobj_mask
        if not_responsible.sum() > 0:
            pred_conf = pred_confs[b][not_responsible]
            conf_loss += lambda_noobj * F.mse_loss(
                pred_conf.squeeze(-1),
                torch.zeros_like(pred_conf.squeeze(-1)),
                reduction='sum'
            )
    
    # 갈래 매기기 손실
    if obj_mask.sum() > 0:
        cls_loss = F.mse_loss(
            pred_classes[obj_mask],
            target_classes[obj_mask],
            reduction='sum'
        )
    else:
        cls_loss = 0
    
    # 묶음 크기로 고르게 맞춘 전체 손실
    total_loss = (coord_loss + conf_loss + cls_loss) / batch_size
    
    return total_loss
```

### 요즘 YOLO의 손실

YOLOv5 이후는 더 정교한 손실을 쓴다:

- 두름 상자 되돌리기에 **CIoU 손실**
- 물체다움과 갈래 매기기에 **두 갈래 엇갈린 엔트로피**
- 갈래 치우침을 다루는 **초점 손실**

```python
def modern_yolo_loss(
    pred_boxes: torch.Tensor,
    pred_obj: torch.Tensor,
    pred_cls: torch.Tensor,
    target_boxes: torch.Tensor,
    target_obj: torch.Tensor,
    target_cls: torch.Tensor,
    box_weight: float = 7.5,
    obj_weight: float = 1.0,
    cls_weight: float = 0.5
) -> dict:
    """
    CIoU와 두 갈래 엇갈린 엔트로피를 쓴 요즘 YOLO 손실.
    """
    # 상자 손실(CIoU)
    ciou = compute_ciou(pred_boxes, target_boxes)
    box_loss = (1 - ciou).mean()
    
    # 물체다움 손실(로짓을 쓴 두 갈래 엇갈린 엔트로피)
    obj_loss = F.binary_cross_entropy_with_logits(
        pred_obj, target_obj, reduction='mean'
    )
    
    # 갈래 매기기 손실(로짓을 쓴 두 갈래 엇갈린 엔트로피)
    cls_loss = F.binary_cross_entropy_with_logits(
        pred_cls, target_cls, reduction='mean'
    )
    
    total_loss = (
        box_weight * box_loss +
        obj_weight * obj_loss +
        cls_weight * cls_loss
    )
    
    return {
        'loss': total_loss,
        'box_loss': box_loss,
        'obj_loss': obj_loss,
        'cls_loss': cls_loss
    }
```

## 실전에서 YOLO 쓰기

### Ultralytics YOLOv8

```python
from ultralytics import YOLO
import torch

# 모델 읽어 들이기(저절로 내려받는다)
model_nano = YOLO('yolov8n.pt')    # 가장 빠름
model_small = YOLO('yolov8s.pt')
model_medium = YOLO('yolov8m.pt')
model_large = YOLO('yolov8l.pt')
model_xlarge = YOLO('yolov8x.pt')  # 가장 정확함

# 추론
results = model_nano('image.jpg')

# 결과 다루기
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    
    for box, score, cls in zip(boxes, scores, classes):
        print(f"Class {int(cls)}: {score:.2f} at {box}")

# 묶음 미룸
results = model_nano(['img1.jpg', 'img2.jpg', 'img3.jpg'])

# 고름을 준 미룸
results = model_nano(
    'image.jpg',
    conf=0.25,        # 믿음도 문턱값
    iou=0.45,         # NMS 겹침 비 문턱값
    max_det=300,      # 최대 알아냄 수
    classes=[0, 2],   # 갈래 거르기(person, car)
    device='cuda:0'
)
```

### 맞춤 모델 익히기

```python
from ultralytics import YOLO

# 미리 익힌 모델에서 시작
model = YOLO('yolov8n.pt')

# 맞춤 자료 뭉치로 익히기
results = model.train(
    data='custom_data.yaml',  # 자료 뭉치 자리매김
    epochs=100,
    imgsz=640,
    batch=16,
    workers=8,
    device='cuda',
    patience=50,         # 일찍 멈추기
    save=True,
    project='runs/detect',
    name='custom_yolo'
)

# 검증
metrics = model.val()
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")

# 내보내기
model.export(format='onnx', dynamic=True)
model.export(format='torchscript')
model.export(format='tensorrt', half=True)
```

### 자료 뭉치 자리매김(custom_data.yaml)

```yaml
# custom_data.yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test  # 있어도 되고 없어도 됨

nc: 3  # 갈래 수
names: ['class1', 'class2', 'class3']
```

## 모형 견줌

### YOLOv8 변종

| 모델 | 매개변수 | FLOPs | mAP@50 | mAP@50:95 | 빠르기(T4) |
|-------|--------|-------|--------|-----------|------------|
| YOLOv8n | 3.2M | 8.7G | 52.6% | 37.3% | 0.99ms |
| YOLOv8s | 11.2M | 28.6G | 61.8% | 44.9% | 1.20ms |
| YOLOv8m | 25.9M | 78.9G | 67.2% | 50.2% | 1.83ms |
| YOLOv8l | 43.7M | 165.2G | 69.8% | 52.9% | 2.39ms |
| YOLOv8x | 68.2M | 257.8G | 71.0% | 53.9% | 3.53ms |

### 모델 크기 고르기

```
Use Case                    Recommended Model
─────────────────────────────────────────────
Real-time video (>30 FPS)   YOLOv8n, YOLOv8s
Mobile/Edge deployment      YOLOv8n
General applications        YOLOv8m
High accuracy required      YOLOv8l, YOLOv8x
Research/Benchmarking       YOLOv8x
```

## 요약

YOLO는 한 방 방식으로 물체 알아내기의 판을 뒤집었다:

1. **그물 하나**: 앞먹임 한 번으로 모든 알아냄을 낸다
2. **격자 바탕**: 그림을 칸으로 나누고 칸마다 상자를 어림한다
3. **끝에서 끝까지**: 화소에서 상자로 곧바로 되돌린다
4. **실시간**: 모델 크기에 따라 초당 30~155틀 이상
5. **끊임없이 나아감**: YOLOv8은 요즘 익히기와 닻 없는 알아내기를 쓴다

핵심 짜기 세부 사항:

- 특징 켜마다 여러 잣수로 어림하기
- 닻 상자(v2~v7) 또는 닻 없는(v8) 어림
- 정확한 상자 되돌리기를 위한 CIoU 손실
- 센 자료 불리기(모자이크, MixUp)

YOLO 모델은 실시간 쓰임새에서 빠르기와 정확도의 맞바꿈이 가장 좋다.

## 참고 문헌

1. Redmon, J., et al. (2016). You Only Look Once: Unified, Real-Time Object Detection. *CVPR*.
2. Redmon, J., & Farhadi, A. (2017). YOLO9000: Better, Faster, Stronger. *CVPR*.
3. Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement. *arXiv*.
4. Bochkovskiy, A., et al. (2020). YOLOv4: Optimal Speed and Accuracy of Object Detection. *arXiv*.
5. Jocher, G. (2020-2023). Ultralytics YOLOv5/YOLOv8. *GitHub*.

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
    In one-stage detectors, most anchor boxes correspond to background (easy negatives), while only a few contain objects. Standard cross-entropy loss is dominated by the large number of easy negatives, drowning out the gradient signal from hard positives. **Focal Loss** adds a modulating factor: $\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$. When $\gamma > 0$, easy examples (high $p_t$) are down-weighted exponentially, focusing training on hard examples. With $\gamma = 2$ and $\alpha = 0.25$, RetinaNet achieves accuracy comparable to two-stage detectors while maintaining one-stage speed.
