# YOLO: 격자 예측에서 다중 해상도 탐지까지

YOLO는 이미지 전체에서 객체의 위치와 클래스를 예측하는 단일 단계 탐지 계열이다. 최초 YOLO와 후속 구현은 출력 표현·학습 목표·backbone이 다르므로 하나의 격자 공식이나 손실을 모든 버전에 적용하면 안 된다. 이 절은 YOLOv1·v3·Ultralytics YOLOv8을 중심으로 각 코드의 범위를 구분한다.

## 1. 단일 단계 탐지와 이미지 특징

[최초 YOLO](https://arxiv.org/abs/1506.02640)는 이미지 전체의 특징에서 상자와 클래스를 함께 예측했다. 아래 도식의 S×S×(5B+C)는 **YOLOv1의 출력 형식**이다. 후속 앵커 기반·앵커 없는 head의 공통 형식은 아니다. 네트워크 출력 이후에도 좌표 변환·점수 필터·중복 억제 등 후처리가 필요하다.

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

두 단계 탐지기도 backbone에서 전체 이미지 특징을 만들고 제안 영역별 특징을 사용한다. ‘YOLO는 전체를 보고 Faster R-CNN은 제안만 본다’는 대비는 정확하지 않다. 단일 단계는 별도 영역 제안·영역별 head를 두지 않는다는 구조상의 차이다. 속도와 작은 객체 성능·오탐률은 모델·해상도·학습·하드웨어에 따라 비교한다.

## 2. YOLOv1의 격자와 좌표

YOLOv1은 객체 중심이 포함된 격자 칸이 그 객체의 예측을 담당한다. 칸마다 B개의 상자·confidence와 공유된 C개 클래스 조건부 확률을 출력한다. VOC의 S=7, B=2, C=20이면 7×7×30이다. 클래스 예측을 칸 안의 상자들이 공유하므로 같은 칸의 여러 객체를 표현하는 데 제약이 있다.

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

중심 x,y는 칸 안의 상대 위치, 너비·높이는 전체 이미지에 대한 비율로 표현한다. YOLOv1의 confidence 목표는 객체 존재와 예측·정답 IoU를 결합한 값이다. 이를 후속 YOLO의 모든 objectness 정의와 동일시하지 않는다. 학습 목표가 이러한 좌표 범위여도 임의 raw 네트워크 출력이 자동으로 유효한 좌표 범위 안에 드는 것은 아니다.

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

`decode_yolo_boxes`는 이미 해당 의미로 표현된 N×S×S×(5B+C) 값에서 상자를 이미지 좌표 xyxy로 바꾼다. sigmoid·앵커 exp 변환을 하지 않으므로 v3 로짓을 넣는 디코더가 아니다. confidence와 클래스 점수도 반환하지 않는다. 결과는 상자 인덱스 b를 바깥 순서로 연결하므로 점수를 별도로 펼칠 때 같은 순서를 사용한다. 음수 너비·높이나 이미지 밖 좌표를 검증·보정하지 않는다.

## 3. 후속 버전에서 바뀐 표현

YOLOv2는 앵커와 크기 군집화, 다중 해상도 학습 등을 사용했고, [YOLOv3](https://arxiv.org/abs/1804.02767)는 잔차 backbone과 세 해상도의 앵커 예측·독립 클래스 sigmoid를 사용한다. 아래는 잔차 블록과 **한 해상도 head**이며 전체 Darknet-53·세 스케일 연결은 아니다.

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

입력 B×C×H×W에서 head는 B×A×H×W×(5+K)를 반환한다. 기본 A=3,K=80이면 마지막 차원 85다. 이 값은 raw 예측이므로 앵커·stride·격자와 함께 디코딩하고 objectness·클래스를 해석해야 한다.

YOLOv4 이후에는 CSP 계열 특징, 피라미드 연결·학습 기법·추론 구조 재매개변수화 등 다양한 설계가 제시되었다. YOLO라는 이름을 공유해도 개발 주체와 계보·정의가 모두 같지는 않다. 번호만으로 모든 이전 버전의 엄밀한 상위 집합이나 성능 순서를 뜻하지 않는다.

다음은 2023년에 공개된 Ultralytics YOLOv8 사용 예다. YOLOv8을 ‘최신 세대’라고 고정해 설명하지 않는다. 이 코드는 추론뿐 아니라 학습과 파일 내보내기도 수행하므로 읽을 때 각 호출의 목적을 구분한다.

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

## 4. YOLOv8 특징 결합과 분리 head

YOLOv8의 구체적인 층은 [공식 모델 설정](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/models/v8/yolov8.yaml)을 기준으로 확인한다. C2f·SPPF 및 여러 해상도의 특징 연결을 사용한다. 아래 기존 도식의 `Stem (Focus)`와 640 입력에서 P3=40·P4=20·P5=10 표기는 실제 YOLOv8 설정을 정확히 나타내지 않는다. 통상 stride8·16·32 탐지 특징은 640 입력에서 80·40·20이다. 도식은 CSP·다중 단계의 개념 그림으로만 읽는다.

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

아래 `PANNeck`은 상위 특징의 업샘플링·더하기와 아래 방향의 다운샘플링·연결을 보여 주는 단순 모듈이다. 실제 YOLOv8 neck의 모든 블록을 재현하지 않는다. 입력 특징은 고해상도부터 저해상도 순서이며 두 배씩 크기가 맞아야 한다. 홀수 크기에서는 scale_factor=2 업샘플과 stride2 결과가 맞지 않을 수 있고, 목록 길이가 채널 설정과 같아야 한다.

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

`YOLOv8Head`는 분류·회귀 가지를 분리한 예다. 입력 B×C×H×W에 분류는 B×K×H×W, 회귀는 B×(4R)×H×W를 반환한다. 기본 R=16에서 64채널은 직접적인 상자 네 좌표가 아니라 각 변 거리의 16개 bin 로짓이다. bin별 확률과 기대 거리·격자·stride로 상자를 복원해야 한다. 별도의 objectness 출력은 없고 다중 해상도 head의 조립·디코딩도 포함하지 않는다.

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

## 5. 버전별 손실을 구분하기

YOLOv1의 제곱 오차 목적은 책임 있는 상자의 중심·제곱근 크기, confidence와 칸별 클래스 확률에 대한 항으로 구성된다. 후속 버전이 모두 같은 MSE를 쓰는 것은 아니다. 아래는 v1을 설명하기 위한 축약 구현이다.

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
    
    # 배치 크기로 고르게 맞춘 전체 손실
    total_loss = (coord_loss + conf_loss + cls_loss) / batch_size
    
    return total_loss
```

`compute_iou`가 정의되어 있지 않다. 입력 좌표가 칸 상대 중심·전체 상대 크기라면 같은 표현을 올바르게 xyxy로 변환해 IoU를 계산해야 한다. 책임 상자는 정답과의 IoU로 정하고 양성 칸의 공유 클래스 손실을 계산한다. 다만 코드의 음성 confidence 항은 배경 칸에만 적용되어 객체가 있는 칸의 비책임 상자는 제외한다. 원래 목적의 비책임 예측 처리와 구분한다. IoU를 confidence의 목표로 쓸 때도 이를 detach하지 않으면 상자 좌표 쪽으로 경사가 흐를 수 있다. 절댓값 안의 제곱근은 잘못된 음수 크기를 유효하게 만드는 모델 제약이 아니다.

다음 `modern_yolo_loss`는 CIoU·objectness BCE·클래스 BCE의 합을 보여 주는 **일반적인 골격**이다. `compute_ciou` 정의, 양성 할당·마스크, 정규화와 가중치 설계가 필요하다. 모든 입력 상자에 평균을 내므로 호출자가 유효한 회귀 대상을 골라야 한다.

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

이 함수는 별도의 objectness를 출력하지 않는 위 YOLOv8Head와 그대로 연결되지 않는다. YOLOv8 계열 학습의 거리 분포 손실(DFL)·목표 할당도 없다. 공식 학습 경로는 [손실 구현](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/utils/loss.py)을 참고하며, 모든 후속 버전이 focal loss나 같은 objectness 항을 기본으로 쓴다고 설명하지 않는다.

## 6. 사전 학습 모델의 추론·파인튜닝

아래는 여러 모델 크기를 로드해 추론 결과를 읽는 예다. 모델 파일·이미지 경로·지원 장치와 패키지 버전이 필요하고 가중치를 처음 받을 때 다운로드가 생길 수 있다. 모델별 반복 로드는 메모리를 사용하므로 다섯 크기를 모두 로드해야만 추론할 수 있는 것은 아니다.

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

# 배치 미룸
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

학습·검증·내보내기는 서로 다른 작업이다. train은 가중치를 바꾸고 파일을 저장하며 val의 지표는 지정된 검증 분할에 대한 값이다. 검증에 반복 사용한 자료를 독립 최종 시험 결과로 보고하지 않는다.

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

기존 코드의 `format='tensorrt'`는 [공식 export 형식](https://docs.ultralytics.com/modes/export/)의 TensorRT 지정값 `engine`과 다르다. 지원 backend·정밀도·장치·동적 크기 옵션은 설치 버전과 환경에서 확인해야 한다. 내보냈다는 사실만으로 원래 모델과 동등한 정확도나 목표 지연시간이 검증되는 것은 아니다.

데이터 YAML은 자료 경로·분할·클래스를 정한다. 실제 이미지마다 별도의 탐지 레이블도 있어야 한다. 일반적인 Ultralytics 탐지 레이블은 클래스 ID와 정규화한 중심 x,y·너비·높이이며 [데이터 형식](https://docs.ultralytics.com/datasets/detect/)과 일치시킨다. 클래스 ID는 names의 0부터 시작하는 인덱스다. 영상·촬영 대상 등 관련 표본이 학습과 검증에 섞이지 않도록 과제에 맞게 분리한다.

```yaml
# custom_data.yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test  # 있어도 되고 없어도 됨

nc: 3  # 갈래 수
names: ['class1', 'class2', 'class3']
```

## 7. 모델 크기와 평가 조건

n·s·m·l·x는 크기 선택지이며 최선의 선택은 장치·지연시간·메모리·대상 크기·오류 비용에 달렸다. [공식 YOLOv8 지표](https://docs.ultralytics.com/models/yolov8/)에서 0.99~3.53ms로 제시된 지연시간은 A100 TensorRT 조건의 값이다. 이를 T4 속도나 이미지 읽기부터 표시까지의 전체 지연시간으로 인용하지 않는다. AP50과 AP50:95도 서로 다른 매칭 기준이다.

아래 기존 선택 표는 크기별 후보를 좁히는 참고일 뿐 ‘nano면 30FPS’ 같은 보장이 아니다. 배치·입력 크기·정밀도·후처리·측정 구간을 정하고 실제 대상 환경에서 검증해야 한다.

```
Use Case                    Recommended Model
─────────────────────────────────────────────
Real-time video (>30 FPS)   YOLOv8n, YOLOv8s
Mobile/Edge deployment      YOLOv8n
General applications        YOLOv8m
High accuracy required      YOLOv8l, YOLOv8x
Research/Benchmarking       YOLOv8x
```

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
YOLOv1의 7×7×30과 YOLOv3 head의 B×3×H×W×85가 다른 이유를 설명하라.

</div>

??? success "연습문제 1 풀이"

    v1은 칸마다 상자 2개의 다섯 값과 칸이 공유하는 클래스 20개를 출력한다. v3 head는 앵커마다 좌표·objectness·클래스 80개를 출력한다. 같은 디코더나 손실을 모양만 바꿔 재사용할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
YOLOv8Head의 기본 회귀 64채널과 objectness 출력 여부를 설명하라.

</div>

??? success "연습문제 2 풀이"

    네 변마다 16개 거리 bin 로짓이라 4×16=64다. 분포의 기대값 등을 사용해 거리를 얻어 좌표로 변환해야 하며 이 코드에는 그 절차가 없다. 반환값은 클래스·거리 분포 두 가지로 별도 objectness가 없다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
아래 NMS에 넣기 전 YOLO raw 출력에 필요한 변환과 함수 의존성을 설명하라.

</div>

??? success "연습문제 3 풀이"

    버전별 좌표 디코딩으로 같은 이미지의 xyxy 상자를 만들고 클래스·objectness 규칙에 맞게 점수를 계산한다. 클래스·이미지별 NMS와 resize 역변환을 정의한다. 아래 compute_iou는 별도 정의가 필요하며 본문의 v1 loss용 좌표 IoU 함수와 요구 입력이 같은지 확인해야 한다.

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
640 입력의 P3·P4·P5 크기와 기존 도식의 차이, TensorRT 지연시간의 측정 조건을 설명하라.

</div>

??? success "연습문제 4 풀이"

    stride8·16·32이면 특징 크기는 80·40·20이다. 기존 도식의 40·20·10 및 Focus 표기를 실제 YOLOv8 설정으로 사용하지 않는다. A100 TensorRT의 모델 추론 지연시간을 T4나 전체 애플리케이션 지연시간으로 바꿔 인용할 수 없다.



## 정리하며

YOLO의 버전별 좌표·클래스·거리 분포 표현과 손실을 구분해야 한다. 이 절의 head·neck·손실 조각은 전체 모델의 재현이 아니며, 기존 도식·export 인수의 한계도 설명과 함께 읽는다. 모델 선택과 성능 보고에는 데이터·장치·입력·후처리 조건을 명시한다.

**참고 문헌**

1. Redmon, J., et al. (2016). You Only Look Once: Unified, Real-Time Object Detection. *CVPR*.
2. Redmon, J., & Farhadi, A. (2017). YOLO9000: Better, Faster, Stronger. *CVPR*.
3. Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement. *arXiv*.
4. Bochkovskiy, A., et al. (2020). YOLOv4: Optimal Speed and Accuracy of Object Detection. *arXiv*.
5. Jocher, G. (2020-2023). Ultralytics YOLOv5/YOLOv8. *GitHub*.
