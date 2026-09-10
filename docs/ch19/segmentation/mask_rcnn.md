# 마스크 R-CNN

---

## 1. 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

- 마스크 R-CNN의 얼개를 더 빠른 R-CNN의 넓힘으로 이해한다
- 정확한 화소 수준 마스크를 내놓는 데 RoI Align이 하는 몫을 설명한다
- 마스크 어림 머리와 마스크 전용 손실 함수를 짠다
- 미리 익힌 마스크 R-CNN을 낱 물체 나누기 미룸에 쓴다
- 마스크 R-CNN과 한 단계 낱 물체 나누기 대안을 가린다

---

## 2. 더 빠른 R-CNN에서 마스크 R-CNN으로

마스크 R-CNN(He 외, 2017)은 이미 있던 두름 상자 머리와 갈래 매기기 머리 곁에 나란한 **마스크 어림 가지**를 더해 더 빠른 R-CNN을 넓힌다. 핵심 눈썰미는 낱 물체 나누기를 알아내기(두름 상자 + 갈래)와 낱 물체마다의 두 갈래 마스크 어림으로 쪼갤 수 있다는 것이다.

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

### RoI Align: 결정적인 새로움

(빠른 R-CNN의) 보통 RoI 모으기는 자리를 양자화하면서, 곧 뜬소수점 RoI 자리표를 정수 격자 자리로 반올림하면서 어긋남을 낳는다. 두름 상자 되돌리기에는 이 거친 맞춤도 봐줄 만하다. 그러나 화소 수준 마스크 어림에서는 크게 나빠진다.

**RoI Align**은 정수가 아닌 자리에서 두 줄 사이 끼움으로 정확한 특징 값을 셈해 양자화를 아예 없앤다:

$$\text{RoI Pool}: \text{round}(x / \text{stride}) \rightarrow \text{integer grid}$$

$$\text{RoI Align}: \text{bilinear\_interpolate}(x / \text{stride}) \rightarrow \text{exact position}$$

사소해 보이는 이 바뀜이 COCO에서 마스크 AP를 1~3점 올린다.

---

## 3. 마스크 머리 얼개

가림 머리는 알아낸 낱개마다 두 값 가림을 미루어 보는 작은 온통 누비기 그물이다. RoI로 맞춘 결 위에서 움직이며 갈래마다 크기가 붙박인 가림(흔히 $28 \times 28$이나 $14 \times 14$)을 미루어 본다.

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

### 마스크 손실

마스크 R-CNN은 **화소마다의 두 갈래 엇갈린 엔트로피** 손실을 쓰되 참값 갈래에 딸린 마스크에만 쓴다. 이러면 마스크 어림이 갈래 매기기에서 떨어져 나와 마스크 머리가 갈래끼리 다툴 필요가 없다:

$$\mathcal{L}_{\text{mask}} = -\frac{1}{m^2} \sum_{i,j} \left[ y_{ij} \log \hat{y}_{ij}^{(k)} + (1 - y_{ij}) \log(1 - \hat{y}_{ij}^{(k)}) \right]$$

여기서 $k$은 그 낱 물체의 참값 갈래이고 $m$은 마스크 해상도이다.

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

### 여러 일 손실

온전한 마스크 R-CNN 손실은 세 항을 아우른다:

$$\mathcal{L} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{box}} + \mathcal{L}_{\text{mask}}$$

머리마다 같은 RoI 특징 위에서 서로 얽히지 않고 돌아가며, 마스크 손실은 양성(짝지어진) 제안에만 걸린다.

---

## 4. 미리 익힌 마스크 R-CNN 쓰기

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

### 맞춤 갈래에 맞게 곱게 다듬기

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

---

## 5. 한 단계 방식과의 견줌

| 방법 | 갈래 | 빠르기(초당 틀) | 마스크 AP(COCO) | 쓰임새 |
|--------|------|-------------|----------------|----------|
| 마스크 R-CNN | 두 단계 | 약 5 | 37.1 | 정확도가 중요한 곳 |
| YOLACT | 한 단계 | 약 30 | 29.8 | 실시간 |
| SOLOv2 | 한 단계 | 약 15 | 37.8 | 균형 |
| PointRend | 다듬기 | 약 5 | 38.3 | 높은 해상도 마스크 |

마스크 R-CNN은 여전히 표준 두 단계 방식이다. YOLACT 같은 한 단계 방법은 정확도를 내주고 빠르기를 얻는 반면, SOLOv2 같은 요즘 방식은 자리 제안 없이도 견줄 만한 정확도를 낸다.

---

## 연습문제

**연습문제 1.**
뜻 나누기, 낱 물체 나누기, 온통 나누기의 차이를 설명하여라.

??? success "연습문제 1 풀이"
    **뜻 나누기**는 화소마다 갈래 이름표를 붙이되 같은 갈래의 서로 다른 낱 물체를 가리지 않는다. **낱 물체 나누기**는 낱낱의 물체를 알아내고 저마다 화소 수준 마스크를 주되 셀 수 있는 "것" 갈래에만 그렇게 한다. **온통 나누기**는 둘을 아우른다. 곧 화소마다 갈래 이름표를 붙이고 것 갈래에는 낱 물체 번호도 매긴다. 보기로 거리 장면에서 뜻 나누기는 모든 차를 "차"로 이름 붙이고, 낱 물체 나누기는 차를 하나하나 가려내며, 온통 나누기는 그 둘을 다 하면서 "길", "하늘" 따위도 이름 붙인다.

---

**연습문제 2.**
U-넷 얼개를 설명하고 나누기에서 건너뛰는 이음이 왜 중요한지 밝혀라.

??? success "연습문제 2 풀이"
    U-넷은 오그라드는 부호기 길(누비기와 모으기의 되풀이)과 부풀어 오르는 풀개 길(키우기와 누비기)로 이루어져 U 꼴을 이룬다. **건너뛰는 이음**은 부호기의 특징 지도를 그에 맞는 풀개 켜에 이어 붙인다. 부호기가 *무엇*(뜻 특징)을 담아내면서 *어디*(자리의 세밀함)를 잃기 때문에 이것이 결정적이다. 건너뛰는 이음은 정밀한 화소 수준 어림에 필요한 높은 해상도의 자리 앎을 주어, 풀개가 거친 뜻 앎과 고운 자리 세부를 아우르게 한다.

---

**연습문제 3.**
그림 나누기에는 어떤 손실 함수가 흔히 쓰이는가? 엇갈린 엔트로피 손실과 다이스 손실을 견주어라.

??? success "연습문제 3 풀이"
    **엇결 엔트로피 잃음**은 화소마다 따로 다룬다. $L_{CE} = -\sum_i y_i \log \hat{y}_i$이다. 눈금은 잘 맞지만 수가 많은 갈래에 휘둘릴 수 있다. **다이스 잃음**은 미루어 본 가림과 참 가림이 겹치는 정도를 잰다. $L_{Dice} = 1 - \frac{2|P \cap G|}{|P| + |G|}$이다. 다이스 잃음은 따짐 자(다이스 계수)를 곧바로 가장 좋게 하고, 화소 수와 상관없이 갈래마다 같은 짐을 주므로 갈래 치우침도 더 잘 다룬다. 참으로는 둘을 섞은 $L = \lambda L_{CE} + (1-\lambda) L_{Dice}$이 가장 잘 듣는 일이 잦다.

---

**연습문제 4.**
마스크 R-CNN이 낱 물체 나누기를 위해 더 빠른 R-CNN을 어떻게 넓히는지 밝히고 RoIAlign이 하는 몫을 설명하여라.

??? success "연습문제 4 풀이"
    마스크 R-CNN은 이미 있던 갈래 매기기 가지와 두름 상자 되돌리기 가지 곁에, 알아낸 물체마다 두 갈래 마스크를 어림하는 나란한 가지를 더한다. 핵심 새로움은 RoI 모으기를 갈음하는 **RoIAlign**이다. RoI 모으기는 양자화된 자리표(정수 화소 자리로 반올림)를 써서 특징 지도와 본디 그림 사이가 어긋난다. RoIAlign은 정확한 뜬소수점 자리에서 두 줄 사이 끼움을 써서 양자화 찌꺼기를 없앤다. 이 정밀한 맞춤이 화소 수준 마스크 어림에 결정적이며, RoI 모으기에 견주어 마스크 AP를 상대적으로 10~50% 올린다.

## 정리하며

마스크 R-CNN이 이바지한 핵심:

1. **단순한 넓힘**: 짐을 거의 늘리지 않고 더 빠른 R-CNN에 마스크 가지를 더한다
2. **RoI Align**: 양자화를 없애 자리를 정밀하게 맞춘다
3. **떼어 놓은 어림**: 갈래마다의 두 갈래 마스크로 갈래끼리 다투지 않는다
4. **여러 일 익히기**: 알아내기와 나누기를 함께 가장 좋게 한다

이 얼개는 낱 물체 나누기를 풀 만한 문제로 세웠으며, Cascade Mask R-CNN과 PointRend를 비롯한 요즘 방식의 바탕으로 남아 있다.

**참고 문헌**

1. He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). Mask R-CNN. ICCV.
2. Bolya, D., et al. (2019). YOLACT: Real-time Instance Segmentation. ICCV.
3. Wang, X., et al. (2020). SOLOv2: Dynamic and Fast Instance Segmentation. NeurIPS.
4. Kirillov, A., et al. (2020). PointRend: Image Segmentation as Rendering. CVPR.
