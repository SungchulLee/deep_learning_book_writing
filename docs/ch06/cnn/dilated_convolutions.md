# 팽창 합성곱
## 들어가며

표준 합성곱은 핵 크기가 정하는 좁은 수용 영역을 갖는다. 더 넓은 공간 맥락을 붙잡으려면 더 큰 핵을 쓰거나(매개변수가 $O(K^2)$으로 늘어난다), 층을 더 쌓거나(계산이 늘고 기울기가 깊어진다), **팽창 합성곱**을 쓰면 된다(매개변수는 그대로이고 수용 영역이 넓어진다).

**팽창 합성곱**("구멍을 낸"이라는 뜻의 프랑스어 *à trous*에서 온 **아트루스 합성곱**이라고도 한다)은 핵의 원소 사이에 틈을 넣어 매개변수를 늘리지 않고도 핵이 덮는 공간을 넓힌다. 이 기법은 의미 분할, 음향 생성처럼 높은 해상도에서 넓은 수용 영역이 필요한 과제의 요즘 구조에서 바탕이 된다.

---

## 수학적 정의

### 실효 핵 크기

팽창은 핵의 원소 사이에 "구멍"(0)을 넣는다. 팽창률이 $d$일 때 **실효 핵 크기**는 다음과 같다.

$$K_{eff} = K + (K - 1)(d - 1) = d(K - 1) + 1$$

$K=3$일 때는 다음과 같다.

- $d=1$: $K_{eff} = 3$ (표준 합성곱)
- $d=2$: $K_{eff} = 5$
- $d=4$: $K_{eff} = 9$
- $d=8$: $K_{eff} = 17$

### 형식적 정의

2차원 입력 $X$과 핵 $W$에 팽창률 $d$을 쓰면 다음과 같다.

$$Y[i, j] = \sum_{m=0}^{K-1} \sum_{n=0}^{K-1} X[i + d \cdot m, j + d \cdot n] \cdot W[m, n]$$

핵의 가중치는 표준 합성곱과 같다. 팽창은 *어느 입력 자리*를 표본으로 삼을지만 바꾼다.

### 출력 크기 공식

$$H_{out} = \left\lfloor \frac{H_{in} + 2p - d(K - 1) - 1}{s} \right\rfloor + 1$$

(보폭이 1일 때) 출력 크기를 입력과 같게 하려면 덧대기를 다음과 같이 둔다.

$$p = \frac{d(K-1)}{2}$$

$K=3$, $d=2$이면 $p = 2$이고, $K=3$, $d=4$이면 $p = 4$이다.

---

## 그림으로 보기

```
Standard (d=1)      Dilated (d=2)       Dilated (d=3)
K=3, K_eff=3        K=3, K_eff=5        K=3, K_eff=7

[●][●][●]           [●][ ][●][ ][●]     [●][ ][ ][●][ ][ ][●]
[●][●][●]           [ ][ ][ ][ ][ ]     [ ][ ][ ][ ][ ][ ][ ]
[●][●][●]           [●][ ][●][ ][●]     [ ][ ][ ][ ][ ][ ][ ]
                    [ ][ ][ ][ ][ ]     [●][ ][ ][●][ ][ ][●]
                    [●][ ][●][ ][●]     [ ][ ][ ][ ][ ][ ][ ]
                                        [ ][ ][ ][ ][ ][ ][ ]
                                        [●][ ][ ][●][ ][ ][●]

[●] = kernel weight position
[ ] = skipped (implicit zero)
```

---

## 수용 영역의 증가

### 방법 견주기

| 방법 | 수용 영역의 증가 | 매개변수의 증가 |
|--------|------------------------|------------------|
| 더 큰 핵 | K에 선형 | 제곱: $O(K^2)$ |
| 더 깊은 신경망 | 깊이에 선형 | 깊이에 선형 |
| 팽창 합성곱 | 팽창을 쌓은 수에 지수적 | 일정 |

### 팽창을 쌓을 때의 지수적 증가

팽창률 1, 2, 4, 8인 팽창 합성곱을 쌓아 보자.

```
Layer 1 (d=1): RF = 3
Layer 2 (d=2): RF = 3 + 2×2 = 7  
Layer 3 (d=4): RF = 7 + 2×4 = 15
Layer 4 (d=8): RF = 15 + 2×8 = 31
```

층 4개에 층마다 매개변수 9개(모두 36개)만으로 31×31 수용 영역을 얻는다. 표준 합성곱 하나로 하려면 매개변수가 961개인 31×31 핵이 필요하다!

### 효율 견주기

```python
import torch.nn as nn

def compute_receptive_field(layers):
    """합성곱과 풀링 층의 나열에 대해 수용 영역을 계산한다."""
    rf, jump = 1, 1
    for layer in layers:
        k = layer.get('kernel', 1)
        s = layer.get('stride', 1)
        d = layer.get('dilation', 1)
        k_eff = d * (k - 1) + 1
        rf = rf + (k_eff - 1) * jump
        jump = jump * s
    return rf

# 표준 3×3 합성곱 (5층)
standard = [{'kernel': 3, 'stride': 1} for _ in range(5)]

# 팽창률을 키워 가는 팽창 합성곱
dilated = [
    {'kernel': 3, 'stride': 1, 'dilation': 1},
    {'kernel': 3, 'stride': 1, 'dilation': 2},
    {'kernel': 3, 'stride': 1, 'dilation': 4},
    {'kernel': 3, 'stride': 1, 'dilation': 8},
    {'kernel': 3, 'stride': 1, 'dilation': 16},
]

rf_standard = compute_receptive_field(standard)
rf_dilated = compute_receptive_field(dilated)

print(f"Standard 5 layers (3×3): RF = {rf_standard}")    # 11
print(f"Dilated 5 layers (d=1,2,4,8,16): RF = {rf_dilated}")  # 63
print(f"Ratio: {rf_dilated / rf_standard:.1f}× larger with same parameters!")
```

---

## PyTorch 구현

### 기본 팽창 합성곱

```python
import torch
import torch.nn as nn

# 표준 3×3 합성곱
conv_standard = nn.Conv2d(64, 128, kernel_size=3, dilation=1, padding=1)

# 팽창 3×3 합성곱 (d=2): 실효 수용 영역 5×5
conv_d2 = nn.Conv2d(64, 128, kernel_size=3, dilation=2, padding=2)

# 팽창 3×3 합성곱 (d=4): 실효 수용 영역 9×9
conv_d4 = nn.Conv2d(64, 128, kernel_size=3, dilation=4, padding=4)

x = torch.randn(1, 64, 56, 56)

# 알맞게 덧대면 모두 같은 출력 크기를 낸다
print(f"Standard: {conv_standard(x).shape}")  # [1, 128, 56, 56]
print(f"Dilation 2: {conv_d2(x).shape}")      # [1, 128, 56, 56]
print(f"Dilation 4: {conv_d4(x).shape}")      # [1, 128, 56, 56]

# 매개변수 수가 같다!
for name, conv in [("Standard", conv_standard), ("d=2", conv_d2), ("d=4", conv_d4)]:
    params = sum(p.numel() for p in conv.parameters())
    k_eff = conv.dilation[0] * (conv.kernel_size[0] - 1) + 1
    print(f"{name}: params={params:,}, effective RF={k_eff}×{k_eff}")
```

**핵심 관찰**: 팽창을 준 3×3은 표준 5×5과 같은 5×5 수용 영역을 **매개변수 64%를 덜 쓰고** 이룬다!

### 온전한 비교

```python
def analyze_conv(name, conv, input_shape):
    """합성곱 층의 성질을 분석한다."""
    x = torch.randn(*input_shape)
    y = conv(x)
    params = sum(p.numel() for p in conv.parameters())
    k = conv.kernel_size[0]
    d = conv.dilation[0]
    receptive_field = d * (k - 1) + 1
    
    print(f"{name}:")
    print(f"  Input shape:  {list(x.shape)}")
    print(f"  Output shape: {list(y.shape)}")
    print(f"  Parameters:   {params:,}")
    print(f"  Receptive field: {receptive_field}×{receptive_field}")
    print()

input_shape = (1, 64, 56, 56)

# 팽창 3×3 합성곱
conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=2, stride=1, dilation=2)
analyze_conv("Dilated 3×3 (d=2)", conv3, input_shape)

# 표준 5×5 합성곱 (팽창 3×3 d=2와 수용 영역이 같다)
conv4 = nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=1, dilation=1)
analyze_conv("Standard 5×5", conv4, input_shape)
```

**출력:**
```
Dilated 3×3 (d=2):
  Parameters:   73,856
  Receptive field: 5×5

Standard 5×5:
  Parameters:   204,928
  Receptive field: 5×5
```

---

## 격자 무늬 문제

### 무엇이 문제인가

**같은 팽창률**의 합성곱을 쌓으면 어떤 입력 자리는 아예 표본이 되지 않는 "격자 무늬" 흠이 생긴다.

```
Dilation=2, two layers:

Layer 1 samples:    Layer 2 samples:    Combined coverage:
[●][ ][●][ ][●]     [●][ ][●][ ][●]     [●][ ][●][ ][●]
[ ][ ][ ][ ][ ]     [ ][ ][ ][ ][ ]     [ ][ ][ ][ ][ ]
[●][ ][●][ ][●]  →  [●][ ][●][ ][●]  =  [●][ ][●][ ][●]
[ ][ ][ ][ ][ ]     [ ][ ][ ][ ][ ]     [ ][ ][ ][ ][ ]
[●][ ][●][ ][●]     [●][ ][●][ ][●]     [●][ ][●][ ][●]

Problem: The [ ] positions are NEVER sampled!
```

### 해결책

**서로 배수가 아닌 팽창률을 쓰거나** "톱니" 방식을 쓴다.

- ✅ 좋음: \$1, 2, 5, 1, 2, 5$ ("Understanding Convolution for Semantic Segmentation"의 HDC 방식)
- ✅ 좋음: \$1, 2, 4, 8$ 뒤에 되풀이
- ❌ 나쁨: \$2, 2, 2, 2$ (격자 무늬)

핵심은 잇따른 팽창률이 1보다 큰 공약수를 가지면 안 된다는 것이다. HDC(혼합 팽창 합성곱) 방식은 모든 입력 자리가 덮이도록 해 준다.

---

## 조밀 예측을 위한 팽창

### 분할의 어려움

의미 분할에는 다음이 필요하다.

1. **넓은 수용 영역**: 맥락을 알기 위해 (이 화소는 고양이의 일부인가 개의 일부인가?)
2. **높은 해상도의 출력**: 섬세한 경계를 지키기 위해

표준 CNN은 딜레마에 놓인다.

- (보폭이나 풀링으로) 하향 표본화하면 수용 영역은 넓어지지만 해상도를 잃는다
- 상향 표본화로 해상도는 되찾지만 정보는 이미 사라졌다

### 팽창 합성곱이라는 해법

팽창 합성곱은 **하향 표본화 없이 넓은 수용 영역**을 준다.

```
Standard CNN path:
Input (224×224) → Conv/Pool → (112×112) → Conv/Pool → (56×56) → ... → (7×7)
                                                                        ↓
                                                                   Upsample
                                                                        ↓
                                                              Output (224×224)
                                                           [Information lost!]

Dilated CNN path:
Input (224×224) → Conv d=1 → (224×224) → Conv d=2 → (224×224) → Conv d=4 → ...
                                                                        ↓
                                                              Output (224×224)
                                                        [Full resolution preserved!]
```

### 여러 규모의 특징 추출: ASPP

아트루스 공간 피라미드 풀링(ASPP) 모듈은 서로 다른 팽창률의 합성곱을 나란히 적용하여 여러 규모의 맥락을 붙잡는다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AtrousSpatialPyramidPooling(nn.Module):
    """
    여러 규모의 특징을 뽑는 DeepLab의 ASPP 모듈.
    팽창률이 서로 다른 합성곱을 나란히 쓴다.
    """
    def __init__(self, in_channels, out_channels, rates=[6, 12, 18]):
        super().__init__()
        
        # 1x1 합성곱 (전역 특징)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 팽창률이 서로 다른 팽창 합성곱들
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 
                         padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for rate in rates
        ])
        
        # 전역 평균 풀링 (이미지 수준 특징)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 특징을 엮는 마지막 1x1 합성곱
        num_features = 1 + len(rates) + 1  # 1x1과 팽창 합성곱들과 전역 풀링
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * num_features, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        
        # 모든 가지 적용
        features = [self.conv1x1(x)]
        features += [conv(x) for conv in self.dilated_convs]
        
        # 전역 풀링 가지 (크기를 맞추려고 상향 표본화)
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=size, mode='bilinear', 
                                   align_corners=False)
        features.append(global_feat)
        
        # 이어 붙인 뒤 사영
        x = torch.cat(features, dim=1)
        x = self.project(x)
        
        return x

# ASPP 모듈 시험
aspp = AtrousSpatialPyramidPooling(256, 256, rates=[6, 12, 18])
x = torch.randn(2, 256, 28, 28)
out = aspp(x)
print(f"ASPP: {x.shape} → {out.shape}")  # 공간 차원이 같다
print(f"Parameters: {sum(p.numel() for p in aspp.parameters()):,}")
```

---

## WaveNet과 시간 방향 팽창

### 음향의 어려움

음향 신호에는 아주 긴 수용 영역이 필요하다(16kHz에서 몇 초의 음향은 표본 수만 개이다). 표준 합성곱으로 하려면 쓸 수 없을 만큼 큰 핵이나 수백 개의 층이 필요하다.

### 시계열 데이터를 위한 지수 팽창

WaveNet은 인과적인 1차원 합성곱에 **지수적으로 커지는 팽창률**을 쓴다.

```python
import torch.nn as nn

class DilatedCausalConv1d(nn.Module):
    """순차열 모형을 위한 팽창 인과 합성곱."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding
        )
    
    def forward(self, x):
        out = self.conv(x)
        # 인과성을 지키려고 오른쪽 덧대기 제거
        return out[:, :, :-self.padding] if self.padding > 0 else out

# 팽창률을 지수적으로 키우며 쌓기
def build_wavenet_stack(channels, kernel_size=2, num_layers=10):
    layers = []
    for i in range(num_layers):
        dilation = 2 ** i  # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
        layers.append(DilatedCausalConv1d(channels, channels, kernel_size, dilation))
    return nn.Sequential(*layers)

# 수용 영역 계산:
# K=2이고 팽창률이 [1, 2, 4, ..., 512]일 때:
# RF = 1 + sum(d * (K-1)) = 1 + (1+2+4+...+512) = 표본 1024개
```

이렇게 하면 층 10개와 층마다 일정한 매개변수만으로 표본 1024개의 수용 영역을 얻는다. 시간 방향 합성곱 구조는 [1차원 합성곱](conv1d.md)을 보라.

---

## 실무 지침

### 팽창률 고르기

| 목표 | 팽창률 | 참고 |
|------|----------|-------|
| 표준 합성곱 | 1 | 대부분의 층에서 기본값 |
| 더 넓은 수용 영역 | 2, 4, 8, … | 지수적으로 키운다 |
| 여러 규모의 특징 | 여러 팽창률 (ASPP) | 분할용 |

### 흔히 쓰는 팽창 방식

```python
# 방식 1: 지수적으로 키우기 (WaveNet 방식)
# 팽창률: 1, 2, 4, 8, 16
# 알맞은 곳: 시계열 데이터, 넓은 수용 영역

# 방식 2: ASPP의 나란한 팽창률
# 팽창률: 6, 12, 18 (나란히 적용)
# 알맞은 곳: 여러 규모의 분할 맥락

# 방식 3: HDC (격자 무늬 없음)
# 팽창률: 1, 2, 5, 1, 2, 5, …
# 알맞은 곳: 흠 없는 조밀 예측

# 방식 4: 톱니처럼 되돌리기
# 팽창률: 1, 2, 4, 8, 1, 2, 4, 8
# 알맞은 곳: 되풀이되는 블록에서 매번 새로 덮기
```

### 조밀 예측을 위한 구조 설계

분할에서는 (ResNet 같은) 분류용 뼈대를 가져와 다음과 같이 고치는 것이 표준적인 방법이다.

- 마지막 두 하향 표본화 단계를 없앤다
- 해상도를 지키려고 팽창 합성곱으로 갈아 끼운다
- 여러 규모의 맥락을 위해 ASPP를 더한다

```python
# 분할을 위한 ResNet 뼈대 수정
# 원래:  4단계에서 stride=2, 5단계에서 stride=2
# 수정:  4단계에서 dilation=2, 5단계에서 dilation=4

# 이렇게 하면 해상도가 1/32이 아니라 1/8로 남는다:
# 원래:  224 → 112 → 56 → 28 → 14 → 7    (보폭 32)
# 수정:  224 → 112 → 56 → 28 → 28 → 28    (보폭 8)
```

---

## 요약

| 항목 | 설명 |
|--------|-------------|
| **연산** | 핵의 원소 사이에 틈을 둔 표준 합성곱 |
| **실효 핵** | $K_{eff} = d(K-1) + 1$ |
| **매개변수** | 표준과 같음 (팽창은 공짜다!) |
| **핵심 이점** | 매개변수는 그대로 두고 수용 영역이 지수적으로 넓어진다 |
| **주된 함정** | 같은 팽창률을 되풀이할 때 생기는 격자 무늬 흠 |
| **주된 쓰임** | 의미 분할, 음향, 온전한 해상도에서 넓은 수용 영역이 필요한 과제 |

## 핵심 정리

1. **팽창은 핵의 원소 사이에 틈을 넣어 수용 영역을 넓힌다.** 매개변수는 더 들지 않는다
2. **지수적으로 쌓으면**($d = 1, 2, 4, 8, \dots$) 아주 적은 층으로 거대한 수용 영역을 얻는다
3. **격자 무늬 문제**는 같은 팽창률을 쌓을 때 생긴다. 팽창률을 바꾸어 가며 써서 빠짐없이 덮어라
4. **ASPP**는 서로 다른 팽창률의 합성곱을 나란히 적용하여 여러 규모의 맥락을 붙잡는다
5. **조밀 예측에서는** 하향 표본화를 팽창으로 갈아 끼우면 넓은 수용 영역을 지키면서 공간 해상도도 지킬 수 있다
6. **WaveNet 방식의 구조**는 인과적인 팽창 합성곱으로 시간 방향을 효율적으로 다룬다

## 참고 문헌

1. Yu, F., & Koltun, V. (2016). Multi-scale context aggregation by dilated convolutions. *ICLR 2016*.

2. Chen, L.-C., Papandreou, G., Kokkinos, I., Murphy, K., & Yuille, A. L. (2018). DeepLab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected CRFs. *IEEE TPAMI*.

3. van den Oord, A., et al. (2016). WaveNet: A generative model for raw audio. *arXiv preprint arXiv:1609.03499*.

4. Wang, P., et al. (2018). Understanding convolution for semantic segmentation. *WACV 2018*.

5. Dumoulin, V., & Visin, F. (2016). A guide to convolution arithmetic for deep learning. *arXiv preprint arXiv:1603.07285*.

## 연습문제

**연습문제 1.**
핵 크기가 3이고 팽창률이 1, 2, 4인 팽창 합성곱 세 층의 수용 영역을 계산하라.

??? success "연습문제 1 풀이"
    RF $= 1 + \sum_{l} (k-1) \cdot d_l = 1 + 2(1) + 2(2) + 2(4) = 1 + 2 + 4 + 8 = 15$이다. 팽창 층 세 개는 수용 영역 15를 이루지만 표준 층 세 개는 7에 그친다.

---

**연습문제 2.**
팽창 합성곱의 '격자 무늬 흠' 문제와 그것을 누그러뜨리는 방법을 설명하라.

??? success "연습문제 2 풀이"
    팽창률이 크면 핵이 멀찍이 떨어진 자리에서만 표본을 얻어 지역 정보를 놓친다(바둑판 무늬). 누그러뜨리는 방법: (1) 1,2,5,1,2,5처럼 고르지 않은 팽창률을 쓰는 혼합 팽창 합성곱(HDC)을 쓴다, (2) 팽창 층 사이에 표준 합성곱을 넣는다, (3) 변형 가능 합성곱을 쓴다.

---

**연습문제 3.**
팽창 합성곱을 PyTorch로 구현하고 출력마다 어느 입력 자리가 이바지하는지 그려 보라.

??? success "연습문제 3 풀이"
    ```python
    conv = nn.Conv2d(1, 1, kernel_size=3, dilation=2, padding=2)
    # dilation=2이면 3x3 핵이 5x5 넓이를 걸친다
    # (한 칸 걸러 표본을 얻는다)
    ```

---

**연습문제 4.**
수용 영역을 넓히는 방법으로 팽창 합성곱과 풀링을 견주어라. 팽창의 이점은 무엇인가?

??? success "연습문제 4 풀이"
    풀링은 하향 표본화로 수용 영역을 넓히지만 공간 해상도를 잃는다. 팽창은 하향 표본화 없이 수용 영역을 넓혀 온전한 해상도를 지킨다. 출력이 입력 해상도와 같아야 하는 조밀 예측 과제(분할, 깊이 추정)에는 팽창이 꼭 필요하다.
