# Mask R-CNN

Mask R-CNN은 Faster R-CNN의 객체 탐지에 객체별 이진 마스크 예측을 추가한 인스턴스 분할 모델이다. 객체의 클래스와 경계 상자는 탐지 head가, 객체 내부의 전경 마스크는 mask head가 담당한다. 전체 영상의 모든 픽셀에 하나의 클래스 맵을 주는 시맨틱 분할과 구분한다.

## 1. 제안 영역과 병렬 head

backbone·FPN에서 특징을 만들고 RPN으로 제안 영역을 얻는다. 영역별 특징을 RoIAlign으로 추출한 뒤 분류·상자 회귀와 마스크 예측을 수행한다. 각 head의 기능은 분리되지만 backbone 등 가중치를 공유하므로 학습 경사가 서로 완전히 독립적인 것은 아니다.

```
Input Image
     │
     ▼
┌─────────────────────────┐
│  Backbone (ResNet + FPN) │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Region Proposal Network │
│         (RPN)            │
└──────────┬──────────────┘
           │
           ▼ (Regions of Interest)
┌─────────────────────────┐
│      RoI Align           │  ← Key improvement over RoI Pooling
└──────────┬──────────────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
┌─────────┐  ┌─────────┐
│Box Head │  │Mask Head│
│(class + │  │(binary  │
│ bbox)   │  │ mask)   │
└─────────┘  └─────────┘
```

Mask R-CNN의 두 단계 구조는 영역 제안과 영역별 예측의 구분이다. mask head를 추가했다고 모든 물체가 검출되거나 마스크의 정확도가 보장되는 것은 아니다. 상자·매칭·클래스와 영역 정렬의 오차가 마스크 결과에 영향을 준다.

## 2. RoIAlign의 좌표 정렬

RoIPool은 RoI 경계와 bin 등을 특징 격자에 양자화할 때 위치 오차를 만들 수 있다. RoIAlign은 연속 좌표에 지정한 표본점을 쌍선형 보간하고 bin별로 요약해 이런 반올림을 피한다. 특징 맵의 네 이웃 값에서 보간한 값이며 실제 영상의 ‘정확한 연속 특징’을 복원한다는 뜻은 아니다.

입력 이미지와 특징 맵의 스케일, 좌표·픽셀 중심 관례와 구현의 aligned 설정 등도 일관되어야 한다. RoIAlign이 모든 위치 오차를 없애거나 어떤 backbone에서도 고정된 AP 향상을 준다고 설명하지 않는다. 논문의 C4·C5·stride별 비교도 다른 조건의 결과다.

## 3. 마스크 head와 클래스 선택

아래 head는 영역 특징 N×256×h×w에 공간 크기를 유지하는 합성곱 네 개와 stride2 전치 합성곱, 클래스별 1×1을 적용한다. 출력은 N×K×2h×2w의 **로짓**이다. 입력이14×14이면28×28이 되지만 `mask_size` 인수는 사용하지 않으므로 어떤 입력에서도 지정 해상도를 만드는 함수는 아니다.

```python
import torch
import torch.nn as nn

class MaskHead(nn.Module):
    """
    마스크 R-CNN의 마스크 어림 머리.
    
    RoI로 맞춘 특징을 받아 갈래마다 두 갈래 마스크를 어림한다.
    
    인수:
        in_channels: 들임 특징 채널(RoI Align에서 옴)
        num_classes: 물체 갈래의 개수
        mask_size: 내놓는 마스크 해상도(붙박이: 28×28)
    """
    def __init__(self, in_channels: int = 256, num_classes: int = 80, 
                 mask_size: int = 28):
        super().__init__()
        
        # 3×3 누비기 4개(보통의 마스크 R-CNN 꾸밈)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
        )
        
        # 뒤바꾼 누비기로 2배 키우기
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        
        # 갈래마다 마스크 어림(갈래마다 하나씩 두 갈래 마스크 K개)
        self.mask_pred = nn.Conv2d(256, num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        인수:
            x: RoI로 맞춘 특징 (N, C, mask_size/2, mask_size/2)
        
        반환값:
            갈래마다의 마스크 로짓 (N, num_classes, mask_size, mask_size)
        """
        x = self.conv_layers(x)
        x = self.relu(self.deconv(x))
        return self.mask_pred(x)
```

정답 클래스 k인 양성 RoI에는 해당 채널의 픽셀별 BCE를 적용한다. 다른 클래스 마스크와 softmax 경쟁을 시키지 않고 sigmoid 기반 이진 문제로 학습한다. 추론에서는 예측한 클래스의 마스크를 선택해 경계 상자 좌표로 옮긴다. 한 채널의 class-agnostic head도 다른 설계로 가능하지만 아래 코드는 클래스별 head다.

$$L_{mask}=-\frac1{m^2}\sum_{i,j}\left[y_{ij}\log p_{ij}^{(k)}+(1-y_{ij})\log(1-p_{ij}^{(k)})\right].$$

다음 함수의 `mask_logits`에는 이미 매칭된 양성 RoI만 들어간다고 가정한다. 정답 마스크는 그 RoI에서 crop·정렬·resize한 같은 공간 크기여야 하고 `gt_labels`는 해당 head의 유효한 정수 채널 인덱스여야 한다. 영상 전체의 마스크와 라벨을 이 함수가 자동으로 매칭하지는 않는다. N=0인 경우도 별도 처리가 없어서 평균 BCE가 정의되지 않을 수 있다.

```python
def mask_rcnn_loss(mask_logits: torch.Tensor, gt_masks: torch.Tensor, 
                   gt_labels: torch.Tensor) -> torch.Tensor:
    """
    마스크 R-CNN의 마스크 손실을 셈한다.
    
    참값 갈래의 마스크만 손실에 보태진다.
    
    인수:
        mask_logits: 어림한 마스크 (N, K, m, m)
        gt_masks: 참값 두 갈래 마스크 (N, m, m)
        gt_labels: 참값 갈래 이름표 (N,)
    """
    # 참값 갈래의 마스크 고르기
    N = mask_logits.shape[0]
    indices = torch.arange(N, device=mask_logits.device)
    selected_masks = mask_logits[indices, gt_labels]  # (N, m, m)
    
    return nn.functional.binary_cross_entropy_with_logits(
        selected_masks, gt_masks.float()
    )
```

탐지·마스크 head 수준에서는 L_cls+L_box+L_mask를 합한다. RPN까지 학습하는 전체 모델은 RPN objectness와 상자 회귀 손실도 포함한다. head의 RoI 크기나 분기 구조는 구현에 따라 다를 수 있어 모든 head가 완전히 같은 형태의 RoI 특징을 그대로 쓴다고 가정하지 않는다.

## 4. 사전 학습 모델의 입력·출력

아래 함수는 legacy `pretrained=True` 인수를 사용한다. 라이브러리 버전의 가중치 API와 입력 규약을 확인해야 한다. 모델은 호출마다 새로 구성되며 필요한 가중치가 없으면 다운로드가 필요할 수 있다. CPU에 만든 모델과 입력의 장치도 맞아야 한다.

일반적인 torchvision 탐지 입력은 [C,H,W] 부동소수점 이미지의 리스트이며 값 범위는0~1이다. 전처리는 해당 모델의 내부 변환 규약에 맞추고, 분류 모델용 정규화를 중복 적용하지 않는다. 출력은 객체별 boxes·labels·scores와 N×1×H×W의 soft mask다.

코드의 `threshold`는 검출 score 필터이고 마스크의 임계값은 별도로0.5에 고정한다. 두 값은 다른 기준이다. 빈 검출이 가능하며 이 함수는 마스크 합성·시각화나 전체 장면의 중첩 충돌 해결까지 수행하지 않는다.

```python
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn

def load_and_run_maskrcnn(image: torch.Tensor, threshold: float = 0.5):
    """
    들임 그림에 미리 익힌 마스크 R-CNN을 돌린다.
    
    인수:
        image: 들임 텐서 (3, H, W), 값은 [0, 1]
        threshold: 알아냄의 믿음도 문턱값
    
    반환값:
        상자, 이름표, 점수, 마스크를 담아 거른 어림
    """
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    
    with torch.no_grad():
        predictions = model([image])[0]
    
    keep = predictions['scores'] > threshold
    
    return {
        'boxes': predictions['boxes'][keep],       # (N, 4) xyxy 꼴
        'labels': predictions['labels'][keep],     # (N,) 갈래 번호
        'scores': predictions['scores'][keep],     # (N,) 믿음도
        'masks': predictions['masks'][keep] > 0.5  # (N, 1, H, W) 두 갈래 마스크
    }
```

## 5. 사용자 클래스의 파인튜닝

다음 함수는 box predictor와 mask predictor의 출력 수를 함께 바꾼다. 여기서 `num_classes`는 torchvision 규약의 **배경을 포함한** 클래스 수다. 양성 라벨은 배경0과 구분되며 데이터 라벨과 predictor 채널을 일관되게 매핑해야 한다. COCO의 실질80개 클래스와 배경·범주 ID를 반영한 모델 출력 수를 단순히 같은 수로 취급하면 안 된다.

학습 데이터에는 이미지별 boxes, 정수 labels와 인스턴스별 masks 등이 필요하다. 이 함수는 모델 head를 구성할 뿐 데이터 매칭·학습 루프·검증·라벨 변환을 구현하지 않는다. 교체한 predictor는 새로 학습해야 한다. 입력·배경 규약은 [torchvision의 모델 설명](https://docs.pytorch.org/vision/0.12/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html)을 참고한다.

```python
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_custom_maskrcnn(num_classes: int):
    """갈래 수를 맞춘 마스크 R-CNN을 만든다."""
    model = maskrcnn_resnet50_fpn(pretrained=True)
    
    # 상자 어림개 갈음
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # 마스크 어림개 갈음
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    
    return model
```

## 6. 다른 인스턴스 분할 접근과의 비교

YOLACT의 prototype 조합, SOLOv2의 위치 기반 마스크 예측과 Mask R-CNN의 RoI별 예측은 다른 구성이다. PointRend는 거친 마스크의 일부 위치를 정교하게 다듬는 모듈이며 독립적인 동일 종류 탐지기와 같은 항목으로 비교하면 안 된다.

FPS·mask AP는 backbone·입력 해상도·학습·평가 split·장치·배치·후처리를 명시해야 비교 가능하다. 출처와 조건이 없는 수치표로 두 단계가 항상 더 정확하고 한 단계가 항상 더 빠르다고 결론 내리지 않는다. 경계 품질·작은 객체·검출 누락과 전체 비용을 과제에 맞춰 살핀다.

설계와 ablation은 [Mask R-CNN 원 논문](https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html)을 참고한다. 코드·출력·그림은 이번 편집에서 실행하거나 재생성하지 않았다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
시맨틱 분할과 Mask R-CNN의 출력 단위를 비교하라.

</div>

??? success "연습문제 1 풀이"
    시맨틱 분할은 영상의 픽셀마다 클래스 레이블을 주고 같은 클래스의 객체 ID를 구분하지 않는다. Mask R-CNN은 검출한 객체마다 클래스·상자·마스크를 반환하므로 객체별 중첩이나 검출 누락이 가능하다. 파놉틱 출력 전체를 자동으로 제공하는 것도 아니다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
mask_size=28을 지정하면 모든 입력 RoI에서28×28이 되는가?

</div>

??? success "연습문제 2 풀이"
    아니다. 이 인수는 저장·사용하지 않으며 합성곱은 공간을 유지하고 전치 합성곱이2배 키운다.14×14 입력에서28×28이고 다른 h×w에서는2h×2w다. head의 실제 연산으로 출력 크기를 추적해야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
마스크 손실에 전체 영상의 GT 마스크를 바로 넣을 수 있는가?

</div>

??? success "연습문제 3 풀이"
    이 함수는 양성 RoI가 이미 GT 객체와 매칭되고 그 영역에 맞춘 마스크가 같은 크기로 준비되었다고 가정한다. 해당 GT 클래스 채널을 선택해 BCE를 계산한다. 매칭·crop·resize와 빈 양성 RoI의 처리, 클래스 인덱스 변환을 외부에서 수행해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
RoIAlign의 보간과 정답 클래스 마스크 BCE가 각각 해결하는 문제를 구분하라.

</div>

??? success "연습문제 4 풀이"
    RoIAlign은 특징 격자의 좌표 양자화를 줄여 영역과 특징의 정렬을 돕는다. 클래스별 BCE는 정답 클래스의 이진 전경 문제를 학습하며 클래스 간 픽셀 softmax 경쟁을 분리한다. 둘은 다른 설계 요소이고 모든 오류 제거·정해진 AP 개선을 보장하지 않는다.

## 정리하며

Mask R-CNN은 영역별 클래스·상자·마스크 예측을 결합한다. 연속 좌표 보간과 클래스별 이진 마스크 손실을 구분하고, RoI 목표의 매칭·배경 인덱스·전체 RPN 손실·입출력 규약을 실제 구현에 맞춰 확인한다.

**참고 문헌**

1. He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). Mask R-CNN. ICCV.
2. Bolya, D., et al. (2019). YOLACT: Real-time Instance Segmentation. ICCV.
3. Wang, X., et al. (2020). SOLOv2: Dynamic and Fast Instance Segmentation. NeurIPS.
4. Kirillov, A., et al. (2020). PointRend: Image Segmentation as Rendering. CVPR.
