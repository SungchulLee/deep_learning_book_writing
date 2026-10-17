# 이미지 분할의 기초

이미지 분할(segmentation)은 영상의 각 픽셀이나 객체에 마스크를 부여하는 문제다. 이 절에서는 분할 과제의 종류, 픽셀별 출력과 인코더–디코더의 역할을 정리하고, IoU·Dice·픽셀 정확도의 정의와 집계 방식을 구분한다.

## 1. 분류·탐지·분할

이미지 분류는 영상 수준의 클래스를, 객체 탐지는 객체별 클래스와 경계 상자를 예측한다. **시맨틱 분할**은 픽셀마다 클래스를 부여하지만 같은 클래스의 객체를 구분하지 않는다. **인스턴스 분할**은 객체별 마스크를 구분한다. **파놉틱 분할**은 모든 유효 픽셀의 클래스와, 개별 객체를 세는 thing 클래스의 인스턴스 ID를 함께 부여한다. 도로·하늘처럼 개체를 따로 세지 않는 stuff 클래스도 포함한다.

아래 도식의 구분을 이 용어에 대응해 읽는다.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Visual Understanding Tasks                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Image Classification    Object Detection    Semantic Segmentation  │
│  ┌─────────────────┐    ┌─────────────────┐  ┌─────────────────┐   │
│  │                 │    │   ┌───┐         │  │█████░░░░░░░░░░░│   │
│  │    [Image]      │    │   │Cat│  ┌───┐  │  │█████░░░███░░░░│   │
│  │                 │    │   └───┘  │Dog│  │  │█████░░░███░░░░│   │
│  │  Label: "Cat"   │    │          └───┘  │  │░░░░░░░░███░░░░│   │
│  └─────────────────┘    └─────────────────┘  └─────────────────┘   │
│                                                                      │
│  Output: Single label   Output: Bounding     Output: Pixel-wise     │
│  per image              boxes + labels        class labels          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

입력 X∈Rᴴˣᵂˣᶜ에서 C는 **입력 채널 수**이며 RGB는 3이다. 분류할 클래스 수 K와 다르다. 상호 배타적인 K클래스 분할의 정답은 Y∈{0,…,K−1}ᴴˣᵂ이고, 모델 로짓은 보통 N×K×H×W다. 클래스 축 softmax 뒤 합이 1인 픽셀별 확률을 얻고 argmax로 레이블을 선택한다. 모델이 반환하는 원시 로짓 자체가 확률은 아니다.

이진 전경/배경은 한 채널 로짓과 sigmoid로 나타낼 수도 있다. 이 방식의 0/1 마스크와 두 클래스 softmax의 정수 클래스 맵은 로스·형태가 다르므로 섞어 사용하지 않는다.

## 2. 픽셀 문맥과 공유 계산

픽셀의 색 하나만으로 클래스를 결정하기 어려워 주변 문맥과 넓은 장면 정보가 필요하다. 패치별 분류는 이 문맥을 사용할 수 있지만 겹치는 패치를 매번 계산하면 중복이 크다. 완전 컨볼루션 모델은 공간 전체에서 피처 계산을 공유한다.

아래 `NaivePixelClassifier.forward`는 pass만 있어 실제 픽셀 예측을 수행하지 않는다. 패치 분류기의 비용을 설명하는 골격이며 완성된 분할 구현이 아니다.

```python
import torch
import torch.nn as nn

# 막무가내 화소별 클래스 매개(설명용이며 실전용 아님)
class NaivePixelClassifier(nn.Module):
    """
    화소마다 클래스를 매기는 것이 왜 실전에 맞지 않는지 보여 준다.
    화소마다 작은 이웃 조각만 본다.
    """
    def __init__(self, num_classes, patch_size=3):
        super().__init__()
        self.patch_size = patch_size
        # 조각을 펴서 클래스 매기기
        self.classifier = nn.Sequential(
            nn.Linear(3 * patch_size * patch_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # 이것은 몹시 느리고 쓸모가 없다
        # 화소마다 3x3 이웃만 보기 때문이다
        pass
```

컨볼루션의 이론적 수용영역은 stride·커널·dilation과 깊이에 따라 커진다. 병목까지 내려갔다고 반드시 전체 이미지를 참조하는 것은 아니고, 이론적 영역 안의 실제 영향도도 균일하지 않다. 다운샘플링·dilated convolution·다중 스케일 연산은 서로 다른 방식으로 문맥과 비용을 조절한다.

## 3. 인코더–디코더와 스킵 연결

인코더는 해상도를 줄이며 문맥 피처를 만들고, 디코더는 업샘플링과 피처 변환으로 촘촘한 출력을 만든다. 업샘플링만으로 소실된 세부를 반드시 복원할 수는 없다. 스킵 연결은 상대적으로 높은 해상도의 인코더 피처를 디코더에 제공한다. 원소별 덧셈과 채널 연결은 형태 조건과 후속 연산이 다르다.

```
Input Image (H×W×3)
       │
       ▼
┌──────────────────┐
│     ENCODER      │  ← Extracts hierarchical features
│  (Contracting)   │  ← Reduces spatial dimensions
│                  │  ← Increases channel dimensions
└────────┬─────────┘
         │
    ┌────▼────┐
    │Bottleneck│     ← Most compressed representation
    │ (H/16)   │     ← Largest receptive field
    └────┬────┘
         │
┌────────▼─────────┐
│     DECODER      │  ← Recovers spatial resolution
│   (Expanding)    │  ← Combines with encoder features
│                  │  ← Produces dense predictions
└────────┬─────────┘
         │
         ▼
Output Mask (H×W×K)
```

아래 예제는 세 번 풀링한 뒤 전치 컨볼루션으로 복원하고 인코더 피처를 채널로 연결한다. 최종 1×1은 K채널 로짓을 반환한다. 패딩을 유지하는 컨볼루션이라도 2배 풀링의 바닥 나눗셈 때문에 임의의 홀수 크기에서는 스킵과 업샘플 크기가 다를 수 있다. 양쪽 크기가 맞는 입력이나 명시적인 정렬 처리가 필요하다.

```python
class EncoderDecoderWithSkips(nn.Module):
    """
    건너뛰는 이음을 보여 주는 간추린 인코더-디코더.
    """
    def __init__(self, in_channels=3, num_classes=21):
        super().__init__()
        
        # 인코더 블록
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # 병목
        self.bottleneck = self._conv_block(256, 512)
        
        # 디코더 덩이(건너뛰는 이음 탓에 들임 채널이 두 배임에 유의)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(512, 256)  # 키우기에서 256 + 건너뛰기에서 256
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256, 128)  # 키우기에서 128 + 건너뛰기에서 128
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128, 64)   # 키우기에서 64 + 건너뛰기에서 64
        
        # 마지막 클래스 매기기 레이어
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 건너뛰는 이음에 쓸 피처를 갈무리하는 인코더
        e1 = self.enc1(x)           # 건너뛰기에 쓰려 갈무리
        x = self.pool(e1)
        
        e2 = self.enc2(x)           # 건너뛰기에 쓰려 갈무리
        x = self.pool(e2)
        
        e3 = self.enc3(x)           # 건너뛰기에 쓰려 갈무리
        x = self.pool(e3)
        
        # 병목
        x = self.bottleneck(x)
        
        # 건너뛰는 이음을 갖춘 디코더
        x = self.up3(x)
        x = torch.cat([x, e3], dim=1)  # 건너뛰는 이음: 이어 붙이기
        x = self.dec3(x)
        
        x = self.up2(x)
        x = torch.cat([x, e2], dim=1)  # 건너뛰는 이음
        x = self.dec2(x)
        
        x = self.up1(x)
        x = torch.cat([x, e1], dim=1)  # 건너뛰는 이음
        x = self.dec1(x)
        
        return self.final(x)
```

## 4. IoU와 클래스 집계

한 클래스의 이진 예측 영역 A와 정답 B에서 IoU=|A∩B|/|A∪B|=TP/(TP+FP+FN)다. 배경을 얼마나 정확히 맞혔는지를 세는 TN은 이 식에 없다. 모든 클래스의 IoU를 동일하게 평균하면 macro mIoU지만, 배경 포함 여부와 양쪽 모두 없는 클래스의 처리를 명시해야 한다.

아래 코드는 ignore_index인 정답 픽셀을 제외하고 입력 전체의 픽셀을 합한다. 합집합이 0인 클래스는 NaN으로 둔 뒤 mIoU에서 제외하고, 모두 제외되면 0을 반환한다. 이 0은 정의되지 않는 경우의 구현상 약속이지 실제 성능 측정값이 아니다. `np`를 사용하므로 NumPy를 별도로 import해야 한다.

```python
def calculate_iou(pred: torch.Tensor, target: torch.Tensor, 
                  num_classes: int, ignore_index: int = 255) -> dict:
    """
    클래스마다의 겹침 비와 평균 겹침 비를 셈한다.
    
    인수:
        pred: 어림한 클래스 이름표 (B, H, W)
        target: 참값 이름표 (B, H, W)
        num_classes: 클래스의 개수
        ignore_index: 무시할 번호(보기로 테두리 화소)
    
    반환값:
        클래스별 겹침 비와 평균 겹침 비를 담은 사전
    """
    ious = {}
    
    # 쓸 수 있는 화소의 마스크 만들기
    valid_mask = (target != ignore_index)
    
    for cls in range(num_classes):
        pred_cls = (pred == cls) & valid_mask
        target_cls = (target == cls) & valid_mask
        
        intersection = (pred_cls & target_cls).float().sum()
        union = (pred_cls | target_cls).float().sum()
        
        if union > 0:
            ious[cls] = (intersection / union).item()
        else:
            ious[cls] = float('nan')  # 그 클래스가 없음
    
    # 평균 겹침 비 셈하기(NaN 클래스 뺌)
    valid_ious = [v for v in ious.values() if not np.isnan(v)]
    ious['mIoU'] = np.mean(valid_ious) if valid_ious else 0.0
    
    return ious
```

데이터 전체에서 TP·FP·FN을 모은 뒤 계산하는 지표와 이미지별 지표를 평균하는 것은 일반적으로 다르다. 큰 이미지·큰 객체가 받는 웨이트가 달라지므로 어느 집계를 쓰는지 보고한다. 배치 크기를 바꿔도 같은 평가가 되려면 단순 배치 평균보다 누적 통계의 정의를 확인한다.

## 5. Dice와 soft Dice

이진 영역의 Dice는 2TP/(2TP+FP+FN)이며 같은 비어 있지 않은 영역 쌍에서 Dice=2IoU/(1+IoU)다. 이 관계를 서로 다르게 평균한 mean Dice와 mIoU 사이에 그대로 적용하면 안 된다.

아래 함수는 입력을 펼쳐 원소 곱과 합을 계산한다. 0/1 마스크이면 평활화한 이진 Dice, 확률이면 soft Dice의 한 형태다. 정수 다중 클래스 레이블을 그대로 곱하면 클래스별 겹침이 아니므로 클래스별 이진 맵이나 적절한 one-hot 표현이 필요하다.

```python
def calculate_dice(pred: torch.Tensor, target: torch.Tensor, 
                   smooth: float = 1e-6) -> float:
    """
    두 분류의 다이스 계수를 셈한다.
    
    인수:
        pred: 시그모이드를 거친 어림 확률 (B, 1, H, W)
        target: 참값 두 클래스 마스크 (B, 1, H, W)
        smooth: 0으로 나누는 것을 막는 평활 인수
    
    반환값:
        다이스 계수
    """
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    
    dice = (2. * intersection + smooth) / (
        pred_flat.sum() + target_flat.sum() + smooth
    )
    
    return dice.item()
```

smooth를 분자·분모에 넣어 양쪽 모두 빈 마스크를 1로 만드는 규칙을 사용한다. 이는 평활화 없는 0/0의 수학적 정의를 해결한 보편적 정답이 아니다. soft Dice 로스는 미분 가능한 대리 목적이며 임계값으로 만든 평가 마스크의 Dice와 정확히 같지 않다. 특정 평균 방식·배경 처리·빈 객체가 학습과 평가에 영향을 준다.

## 6. 픽셀 정확도와 클래스 불균형

픽셀 정확도는 유효 픽셀 중 맞힌 비율이다. 이진 분류에서는 (TP+TN)/(TP+TN+FP+FN)이다. 아래 함수는 모든 픽셀이 무시된 경우 0을 반환하므로 그런 표본을 일반 성능값과 구분해서 집계한다.

```python
def calculate_pixel_accuracy(pred: torch.Tensor, target: torch.Tensor,
                             ignore_index: int = 255) -> float:
    """
    화소마다의 정확도를 셈한다.
    
    인수:
        pred: 어림한 클래스 이름표 (B, H, W)
        target: 참값 이름표 (B, H, W)
        ignore_index: 셈에서 무시할 번호
    
    반환값:
        실수로 된 화소 정확도
    """
    valid_mask = (target != ignore_index)
    correct = ((pred == target) & valid_mask).float().sum()
    total = valid_mask.float().sum()
    
    return (correct / total).item() if total > 0 else 0.0
```

배경이 95%인 영상에서 모두 배경으로 예측하면 픽셀 정확도는 95%여도 전경 IoU는 0이다. 그렇다고 어떤 과제에서든 정확도가 무의미하거나 IoU 하나로 충분한 것은 아니다. 작은 객체·경계·클래스별 성능·필요한 운영 조건을 함께 평가한다.

```
Scenario: 95% background, 5% lesion

Prediction A (predicts everything as background):
- Pixel Accuracy: 95% ✓ (misleadingly high!)
- IoU for lesion: 0% ✗

Prediction B (correctly segments lesion):
- Pixel Accuracy: 98%
- IoU for lesion: 85% ✓

IoU penalizes missing small objects that pixel accuracy ignores.
```

## 7. 로스와 데이터 설계

상호 배타적인 클래스에는 픽셀별 교차 엔트로피, 독립적인 이진 마스크에는 BCE를 사용할 수 있다. 클래스 웨이트·표집·focal loss·Dice 기반 로스는 불균형을 다루는 방법이지만 각각 다른 목적과 하이퍼파라미터를 갖는다. CE를 쓴다고 유한 모델이 자동으로 캘리브레이션되는 것도, Dice가 모든 클래스에 자동으로 동일한 웨이트를 주는 것도 아니다.

클래스·객체 크기·경계의 평가가 중요하면 그 집계와 로스의 관계를 명시한다. crop·회전·resize 같은 기하 변환은 이미지와 마스크에 같은 난수·좌표를 적용하고, 정수 클래스 마스크는 레이블을 보존하는 보간을 사용한다. 색상·밝기 변화는 보통 이미지에만 적용한다.

## 8. 응용과 한계

도로 장면, 의료 영상, 위성 영상과 이미지 편집 등에서 분할 마스크를 활용한다. 하지만 픽셀 겹침이 높다는 사실만으로 전체 시스템의 안전성·임상 유효성·부피 측정 정확도가 보장되지는 않는다. 촬영 조건·센서·대상과 데이터 분할, 주석의 불확실성·경계·작은 객체를 과제에 맞게 평가해야 한다. 다중 스케일·피라미드·경계 로스·후처리는 후보 방법이며 모든 실패를 자동으로 해결하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
시맨틱·인스턴스·파놉틱 분할에서 같은 클래스의 두 객체를 어떻게 표현하는가?

</div>

??? success "풀이"
    시맨틱 분할은 두 객체를 같은 클래스 레이블로 표시하고 ID를 구분하지 않는다. 인스턴스 분할은 객체별 마스크를 구분한다. 파놉틱 분할은 모든 유효 픽셀의 클래스와 thing 객체의 ID를 함께 표현하고 stuff 영역도 포함한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
스킵 연결과 업샘플링의 역할 및 형태 조건을 설명하라.

</div>

??? success "풀이"
    업샘플링은 해상도를 키우고 스킵은 높은 해상도에서 얻은 피처를 제공한다. 로스된 정보를 반드시 복원하는 것은 아니다. 연결은 배치·공간 크기를 맞추고 채널은 합하며, 원소별 덧셈은 채널까지 같아야 한다. 풀링한 홀수 크기는 명시적 정렬이 필요할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
평가 Dice와 soft Dice 로스, 클래스 평균의 차이를 설명하라.

</div>

??? success "풀이"
    평가 Dice는 임계값 또는 argmax로 만든 클래스 영역의 겹침이다. soft Dice는 확률의 곱·합을 사용하는 미분 가능한 대리 목적이다. 클래스별 평균·배경 포함·빈 마스크 규칙에 따라 값과 웨이트가 달라져 자동으로 동일한 클래스 웨이트를 준다고 볼 수 없다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
RoIAlign이 필요한 문제와 픽셀 전체의 시맨틱 분할을 구분하라.

</div>

??? success "풀이"
    Mask R-CNN은 제안된 객체 영역마다 마스크를 예측하고 RoIAlign으로 해당 피처를 연속 좌표에서 표본화한다. 시맨틱 분할은 영상 전체의 픽셀 클래스를 예측하며 RoI가 필수는 아니다. RoIAlign의 보간은 좌표 반올림의 오차를 줄이지만 모든 마스크의 정밀도나 고정된 AP 향상을 보장하지 않는다.

## 정리하며

분할은 픽셀의 클래스와 객체 ID를 어떤 수준에서 구분하는지에 따라 과제가 달라진다. 로짓·확률·레이블의 형태와 IoU·Dice의 집계·빈 마스크 규칙을 명시한다. 인코더–디코더와 스킵의 역할을 이해하되 구조·로스 하나가 정확성을 자동으로 보장한다고 해석하지 않는다.

**더 읽을거리**


1. Long, J., Shelhamer, E., & Darrell, T. (2015). Fully Convolutional Networks for Semantic Segmentation. CVPR.
2. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.
3. Chen, L.-C., et al. (2017). Rethinking Atrous Convolution for Semantic Image Segmentation. arXiv.
4. Minaee, S., et al. (2021). Image Segmentation Using Deep Learning: A Survey. IEEE TPAMI.
