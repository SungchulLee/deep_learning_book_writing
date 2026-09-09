# 나누기의 근본

---

## 1. 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

- 그림 가르기, 물체 알아내기, 뜻 나누기의 근본 차이를 이해한다
- 화소마다 갈래 매기기와 그것이 셈에 미치는 뜻을 설명한다
- 부호기-풀개 얼개라는 틀을 설명한다
- 기본 값매김 잣대(겹침 비, 다이스 계수, 화소 정확도)를 짠다
- 뜻 나누기의 흔한 쓰임새와 어려움을 알아본다

---

## 2. 뜻 나누기 들어가기

뜻 나누기는 셈틀 보기에서 가장 결이 고운 보기 이해 가운데 하나이다. 그림 전체에 이름표 하나를 붙이는 그림 가르기나 두름 상자로 물체 자리를 잡는 물체 알아내기와 달리, 뜻 나누기는 그림의 **화소마다** 미리 정해 둔 갈래를 매긴다.

### 보기 이해의 층위

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

### 수식으로 나타내기

들임 그림 $\mathbf{X} \in \mathbb{R}^{H \times W \times C}$이 주어졌을 때(여기서 $H$은 높이, $W$은 너비, $C$은 갈래 수이며 RGB이면 흔히 3이다) 뜻 나누기는 다음 날임을 내놓는다.

$$\mathbf{Y} \in \{0, 1, 2, \ldots, K-1\}^{H \times W}$$

여기서 $K$은 뜻 갈래의 수다. 화소 $(i, j)$마다 갈래 이름표 $y_{i,j} \in \{0, 1, \ldots, K-1\}$을 받는다.

실전에서 신경망은 화소마다 갈래에 대한 확률 분포를 내놓는다:

$$\mathbf{\hat{Y}} \in [0, 1]^{H \times W \times K}$$

여기서 $\hat{y}_{i,j,k}$은 화소 $(i, j)$이 갈래 $k$에 들 확률을 나타낸다. 마지막 미루어 봄은 다음으로 얻는다.

$$y_{i,j} = \arg\max_k \hat{y}_{i,j,k}$$

---

## 3. 화소마다 갈래 매기기

### 생각의 얼거리

뜻 나누기는 화소 자리마다 그림 가르기를 하는 일로 볼 수 있다. 그러나 화소마다 갈래 매개를 따로 돌리는 이 막무가내 방식은 셈이 감당하기 어렵고 결정적인 자리 맥락을 놓친다.

크기가 $512 \times 512$인 그림을 보자.

- 전체 화소: $262,144$
- 화소마다 둘레 자리의 맥락이 필요하다
- 따로 갈래를 매기면 자리 관계를 놓친다

### 받는 자리 문제

화소마다 갈래 매기기의 핵심 어려움은 화소마다 넉넉한 맥락을 얻게 하는 것이다. 신경 세포의 **받는 자리**란 그 깨어남에 영향을 주는 들임 그림의 자리를 뜻한다.

```python
import torch
import torch.nn as nn

# 막무가내 화소별 갈래 매개(설명용이며 실전용 아님)
class NaivePixelClassifier(nn.Module):
    """
    화소마다 갈래를 매기는 것이 왜 실전에 맞지 않는지 보여 준다.
    화소마다 작은 이웃 조각만 본다.
    """
    def __init__(self, num_classes, patch_size=3):
        super().__init__()
        self.patch_size = patch_size
        # 조각을 펴서 갈래 매기기
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

요즘 나누기 그물은 이를 다음으로 푼다:

1. 받는 자리를 차츰 넓히는 **부호기 그물**
2. 자리의 세밀함을 지키는 **건너뛰는 이음**
3. 받는 자리를 효율적으로 넓히는 **벌린/구멍 뚫린 누비기**
4. 크기가 다른 물체를 담아내는 **여러 잣수 다루기**

---

## 4. 부호기-풀개 얼개

부호기-풀개 얼개는 뜻 나누기의 바탕이 되는 틀이다. 이는 뜻 이해(무엇)와 자리 잡기(어디) 사이의 근본 맞바꿈을 다룬다.

### 구조 훑어보기

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

### 앎의 흐름

**부호기(오그라드는 길):**

- 모으기나 성큼 누비기로 차츰 줄여 뽑기
- 갈수록 추상적이고 뜻이 담긴 특징을 담아낸다
- 받는 자리를 넓혀 전체 맥락을 이해한다
- 흔한 흐름: $H \times W \rightarrow H/2 \times W/2 \rightarrow H/4 \times W/4 \rightarrow \ldots$

**병목:**

- 가장 눌러 담은 나타냄
- 뜻이 풍부하게 담겨 있다
- 받는 자리가 가장 넓다. 곧 그림 전체를 "볼" 수 있다
- 자리의 세밀함은 적다

**풀개(부풀어 오르는 길):**

- 뒤바꾼 누비기나 사이 끼움으로 차츰 키우기
- 자리 해상도를 되찾는다
- 높은 수준의 뜻과 낮은 수준의 세부를 아우른다
- 촘촘한 화소마다의 어림을 내놓는다

### 건너뛰는 이음: 앎의 틈 잇기

나누기 얼개의 결정적인 새로움은 부호기 층을 그에 맞는 풀개 층에 곧바로 잇는 **건너뛰는 이음**을 쓰는 것이다.

```python
class EncoderDecoderWithSkips(nn.Module):
    """
    건너뛰는 이음을 보여 주는 간추린 부호기-풀개.
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
        
        # 풀개 덩이(건너뛰는 이음 탓에 들임 채널이 두 배임에 유의)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(512, 256)  # 키우기에서 256 + 건너뛰기에서 256
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256, 128)  # 키우기에서 128 + 건너뛰기에서 128
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128, 64)   # 키우기에서 64 + 건너뛰기에서 64
        
        # 마지막 갈래 매기기 층
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
        # 건너뛰는 이음에 쓸 특징을 갈무리하는 부호기
        e1 = self.enc1(x)           # 건너뛰기에 쓰려 갈무리
        x = self.pool(e1)
        
        e2 = self.enc2(x)           # 건너뛰기에 쓰려 갈무리
        x = self.pool(e2)
        
        e3 = self.enc3(x)           # 건너뛰기에 쓰려 갈무리
        x = self.pool(e3)
        
        # 병목
        x = self.bottleneck(x)
        
        # 건너뛰는 이음을 갖춘 풀개
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

**건너뛰는 이음이 중요한 까닭:**

1. **기울기 흐름**: 뒤로 퍼뜨리기 동안 기울기가 지나는 곧은 길
2. **세부 지키기**: 낮은 수준의 특징(테두리, 결)이 지켜진다
3. **여러 잣수 앎**: 해상도가 다른 특징을 아우른다
4. **익히기의 든든함**: 깊은 그물을 가장 좋게 하기가 쉬워진다

---

## 5. 평가 지표

### 겹침 비(IoU / 자카드 지수)

겹침 비는 나누기 값매김의 표준 잣대이다. 어림한 자리와 참값 자리 사이의 겹침을 잰다.

$$\text{IoU} = \frac{|A \cap B|}{|A \cup B|} = \frac{TP}{TP + FP + FN}$$

여기서 각 기호는 다음과 같다.

- $TP$(참양성): 올바로 어림한 앞바탕 화소
- $FP$(헛양성): 앞바탕으로 잘못 어림한 뒷바탕 화소
- $FN$(헛음성): 뒷바탕으로 잘못 어림한 앞바탕 화소

```python
def calculate_iou(pred: torch.Tensor, target: torch.Tensor, 
                  num_classes: int, ignore_index: int = 255) -> dict:
    """
    갈래마다의 겹침 비와 평균 겹침 비를 셈한다.
    
    인수:
        pred: 어림한 갈래 이름표 (B, H, W)
        target: 참값 이름표 (B, H, W)
        num_classes: 갈래의 개수
        ignore_index: 무시할 번호(보기로 테두리 화소)
    
    반환값:
        갈래별 겹침 비와 평균 겹침 비를 담은 사전
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
            ious[cls] = float('nan')  # 그 갈래가 없음
    
    # 평균 겹침 비 셈하기(NaN 갈래 뺌)
    valid_ious = [v for v in ious.values() if not np.isnan(v)]
    ious['mIoU'] = np.mean(valid_ious) if valid_ious else 0.0
    
    return ious
```

### 다이스 계수(F1 점수)

다이스 계수는 겹침 비와 가깝게 이어져 있으며 특히 의료 영상에서 널리 쓰인다.

$$\text{Dice} = \frac{2|A \cap B|}{|A| + |B|} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

겹침 비와의 관계:

$$\text{Dice} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}}$$

```python
def calculate_dice(pred: torch.Tensor, target: torch.Tensor, 
                   smooth: float = 1e-6) -> float:
    """
    두 갈래 나누기의 다이스 계수를 셈한다.
    
    인수:
        pred: 시그모이드를 거친 어림 확률 (B, 1, H, W)
        target: 참값 두 갈래 마스크 (B, 1, H, W)
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

### 화소 정확도

직관적이기는 하나 갈래가 치우치면 화소 정확도는 오해를 부를 수 있다.

$$\text{Pixel Accuracy} = \frac{\text{Correct Pixels}}{\text{Total Pixels}} = \frac{TP + TN}{TP + TN + FP + FN}$$

```python
def calculate_pixel_accuracy(pred: torch.Tensor, target: torch.Tensor,
                             ignore_index: int = 255) -> float:
    """
    화소마다의 정확도를 셈한다.
    
    인수:
        pred: 어림한 갈래 이름표 (B, H, W)
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

### 왜 화소 정확도가 아니라 겹침 비인가?

병터가 화소의 5%만 덮는 의료 그림을 보자:

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

---

## 6. 뜻 나누기의 쓰임새

### 스스로 몰기

나누기 덕분에 차가 길 장면을 화소 수준에서 이해할 수 있다:

| 갈래 | 쓰임 |
|-------|---------|
| 길 | 달릴 수 있는 바닥 가려내기 |
| 인도 | 걷는 이의 자리 |
| 차 | 움직이는 걸림돌 알아내기 |
| 걷는 이 | 안전에 결정적인 알아내기 |
| 길 표지 | 길잡이와 규칙 |
| 건물 | 장면 이해 |

### 의료 영상

임상 쓰임을 위한 정밀한 테두리 그리기:

- **종양 나누기**: 치료 계획을 위한 정확한 부피 재기
- **장기 나누기**: 수술 계획과 방사선 치료
- **망막 혈관 나누기**: 당뇨 망막병증 가려내기
- **세포 나누기**: 병리와 약 찾기

### 인공위성과 항공 영상

큰 잣수의 지구 살피기:

- 땅 쓰임 가르기
- 도시 계획과 개발
- 재해 살피기
- 농사 지켜보기
- 환경 바뀜 알아내기

### 그림 고치기와 늘린 현실

소비자 쓰임새:

- 인물 모드(뒷바탕 흐리기)
- 가상 입어 보기(옷, 화장)
- 뒷바탕 갈아 끼우기
- 늘린 현실 물체 놓기

---

## 7. 뜻 나누기의 핵심 어려움

### 갈래 치우침

실제 자료 뭉치는 갈래가 심하게 치우친 것이 많다. 스스로 몰기에서 "길"은 넘쳐나지만 "신호등"은 드물다.

**풀이:**

- 무게를 준 손실 함수
- 어려운 보기 캐기를 위한 초점 손실
- 작은 물체를 도드라지게 하는 다이스 손실
- 적은 갈래를 더 많이 뽑기

### 테두리 정밀도

물체 테두리는 정확히 나누기가 어렵기로 이름났다.

**풀이:**

- 테두리를 헤아리는 손실 함수
- 여러 잣수 다루기
- CRF로 뒷손질하기
- 테두리 알아내기로 이끌기

### 잣수 흩어짐

같은 갈래의 물체도 크기가 크게 다를 수 있다(가까운 차와 먼 차).

**풀이:**

- 여러 잣수 특징 녹여 붙이기
- 피라미드 모으기 단원
- 구멍 뚫린 자리 피라미드 모으기(ASPP)
- 특징 피라미드 그물(FPN)

### 계산 효율

촘촘한 어림은 화소마다 다뤄야 한다.

**풀이:**

- 효율적인 등뼈 그물(MobileNet, EfficientNet)
- 깊이별로 갈라지는 누비기
- 앎 내리기
- 신경망 얼개 찾기

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

뜻 나누기는 그림 이해를 화소 수준까지 넓혀 결이 고운 장면 살피기를 가능하게 한다. 건너뛰는 이음을 갖춘 부호기-풀개 얼개는 뜻 이해와 자리 정밀도의 균형을 잡는 바탕 틀이 되었다. 특히 치우친 자료 뭉치에서는 겹침 비와 다이스 잣대로 제대로 값매김하는 일이 꼭 필요하다.

이 분야는 빠르게 나아가고 있으며, 변환기 바탕 얼개와 스스로 살피는 배움이 할 수 있는 것의 한계를 밀어내고 있다. 이어지는 절에서는 요즘 뜻 나누기를 빚어낸 구체적인 얼개, 곧 FCN, U-넷, DeepLab을 깊이 파고든다.

**더 읽을거리**

1. Long, J., Shelhamer, E., & Darrell, T. (2015). Fully Convolutional Networks for Semantic Segmentation. CVPR.
2. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.
3. Chen, L.-C., et al. (2017). Rethinking Atrous Convolution for Semantic Image Segmentation. arXiv.
4. Minaee, S., et al. (2021). Image Segmentation Using Deep Learning: A Survey. IEEE TPAMI.
