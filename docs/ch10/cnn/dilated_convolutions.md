# 팽창 합성곱

팽창 합성곱(Dilated Convolution)은 커널 원소가 참조하는 입력 위치 사이의 간격을 늘린다. 채널 수와 커널 크기가 같으면 가중치 수를 유지하면서 더 넓은 범위의 특징을 계산할 수 있다. 아트루스 합성곱이라고도 한다.

이 페이지에서 수용 영역은 출력이 참조할 수 있는 입력 위치의 경계 범위를 뜻하며, 그 안의 모든 위치가 연결된다는 뜻은 아니다. 일반적인 점화식 $r_l=r_{l-1}+(K_l-1)d_l\prod_{i<l}s_i$는 [수용 영역](receptive_field.md)에서 설명한다.

## 1. 수학적 정의

### 실효 커널 크기

커널 사이에 0을 넣은 더 큰 커널로 생각하면 계산을 이해하기 쉽다. 실제 구현은 0 가중치를 저장하지 않고 간격을 두고 입력에 접근할 수 있다.

$$K_{\text{eff}} = K + (K - 1)(d - 1) = d(K - 1) + 1$$

$K=3$일 때는 다음과 같다.

- $d=1$: $K_{\text{eff}} = 3$ (표준 합성곱)
- $d=2$: $K_{\text{eff}} = 5$
- $d=4$: $K_{\text{eff}} = 9$
- $d=8$: $K_{\text{eff}} = 17$

### 형식적 정의

2차원 입력 $X$과 커널 $W$에 팽창률 $d$을 쓰면 다음과 같다.

$$Y[i, j] = \sum_{m=0}^{K-1} \sum_{n=0}^{K-1} X[i + d \cdot m, j + d \cdot n] \cdot W[m, n]$$

커널의 가중치는 표준 합성곱과 같다. 팽창은 *어느 입력 자리*를 표본으로 삼을지만 바꾼다.

### 출력 크기 공식

$$H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2p - d(K - 1) - 1}{s} \right\rfloor + 1$$

(보폭이 1일 때) 출력 크기를 입력과 같게 하려면 패딩를 다음과 같이 둔다.

$$p = \frac{d(K-1)}{2}$$

좌우 대칭인 정수 패딩을 사용하려면 $d(K-1)$이 짝수여야 한다. 홀수 K이면 항상 만족하지만, 짝수 K에서도 d가 짝수이면 가능하다. 총 패딩 수가 홀수이면 양쪽의 패딩 수를 다르게 두어 길이를 유지할 수 있다.

---

## 2. 그림으로 보기

팽창률을 키워도 $\bullet$ 으로 찍은 가중치 자리의 **개수**는 늘 9개로 그대로이고, 그 9개가 걸치는 **너비**만 $3 \to 5 \to 7$로 늘어난다. 매개변수를 한 개도 더 쓰지 않고 수용 영역을 넓힌다는 말이 그림으로는 이것이다.

```text
Standard (d=1)      Dilated (d=2)       Dilated (d=3)
K=3, K_eff=3        K=3, K_eff=5        K=3, K_eff=7

[●][●][●]           [●][ ][●][ ][●]     [●][ ][ ][●][ ][ ][●]
[●][●][●]           [ ][ ][ ][ ][ ]     [ ][ ][ ][ ][ ][ ][ ]
[●][●][●]           [●][ ][●][ ][●]     [ ][ ][ ][ ][ ][ ][ ]
                    [ ][ ][ ][ ][ ]     [●][ ][ ][●][ ][ ][●]
                    [●][ ][●][ ][●]     [ ][ ][ ][ ][ ][ ][ ]
                                        [ ][ ][ ][ ][ ][ ][ ]
                                        [●][ ][ ][●][ ][ ][●]

[●] = kernel weight position  (9 of them in every case)
[ ] = skipped (implicit zero)
```

세 그림 모두 $\bullet$ 이 9개다. 오른쪽으로 갈수록 $\bullet$ 사이가 벌어질 뿐이며, 벌어진 만큼 건너뛴 자리는 **이 층에서는** 보이지 않는다. 5절에서 보듯 이 건너뜀이 쌓이면 흠이 된다.

---

## 3. 수용 영역의 증가

### 방법 비교

| 방법 | 수용 영역의 증가 | 매개변수의 증가 |
|--------|------------------------|------------------|
| 더 큰 커널 | K에 선형 | 제곱: $O(K^2)$ |
| 더 깊은 신경망 | 깊이에 선형 | 깊이에 선형 |
| 팽창 합성곱 | 팽창률의 합에 비례하며 지수적 팽창률에서 지수적 | 고정 채널 수라면 깊이에 선형 |

매개변수 수가 같은 경우에도 참조하는 위치의 간격은 다르다. 지수적 증가는 팽창률 자체를 지수적으로 키운 설정의 성질이며, 일정한 팽창률의 층을 늘리면 수용 영역의 경계 범위는 선형으로 증가한다.

### 팽창을 쌓을 때의 지수적 증가

팽창률 1, 2, 4, 8인 팽창 합성곱을 쌓아 보자. 보폭이 모두 1이므로 점화식은 $r_l = r_{l-1} + (K-1) d_l = r_{l-1} + 2 d_l$으로 줄어든다.

```text
Layer 1 (d=1): RF = 1 + 2×1 = 3
Layer 2 (d=2): RF = 3 + 2×2 = 7
Layer 3 (d=4): RF = 7 + 2×4 = 15
Layer 4 (d=8): RF = 15 + 2×8 = 31
```

커널 3·보폭 1·팽창률 $2^{l-1}$인 L개 층의 수용 영역은 $1+2(2^L-1)=2^{L+1}-1$이다. 단일 입력·출력 채널에 편향을 제외하면 4개 층의 가중치는 36개이다. 같은 경계 범위를 갖는 31×31 단일 합성곱의 961개보다 적지만 임의의 큰 커널을 동일하게 표현하는 것은 아니다.

### 효율 비교

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

print(f"Standard 5 layers (3×3): RF = {rf_standard}")    # 1 + 5×2 = 11
print(f"Dilated 5 layers (d=1,2,4,8,16): RF = {rf_dilated}")  # 1 + 2×31 = 63
print(f"Ratio: {rf_dilated / rf_standard:.1f}× larger with same parameters!")
```

**출력:**

```
Standard 5 layers (3×3): RF = 11
Dilated 5 layers (d=1,2,4,8,16): RF = 63
Ratio: 5.7× larger with same parameters!
```

같은 수의 층, 같은 수의 매개변수인데 수용 영역은 11에서 63으로 5.7배가 된다. [수용 영역](receptive_field.md) 쪽도 같은 두 배열을 같은 식으로 재어 같은 11과 63을 얻는다.

---

## 4. PyTorch 구현

### 기본 팽창 합성곱

```python
import torch
import torch.nn as nn

# 표준 3×3 합성곱
conv_standard = nn.Conv2d(64, 128, kernel_size=3, dilation=1, padding=1)

# 팽창 3×3 합성곱 (d=2): 실효 핵 2×(3-1)+1 = 5, 곧 수용 영역 5×5
conv_d2 = nn.Conv2d(64, 128, kernel_size=3, dilation=2, padding=2)

# 팽창 3×3 합성곱 (d=4): 실효 핵 4×(3-1)+1 = 9, 곧 수용 영역 9×9
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

**출력:**

```
Standard: torch.Size([1, 128, 56, 56])
Dilation 2: torch.Size([1, 128, 56, 56])
Dilation 4: torch.Size([1, 128, 56, 56])
Standard: params=73,856, effective RF=3×3
d=2: params=73,856, effective RF=5×5
d=4: params=73,856, effective RF=9×9
```

**비교 기준**: 팽창률 2인 3×3 합성곱의 유효 커널 크기는 5이다. 표준 5×5보다 가중치를 적게 쓰지만 25개 입력 위치를 모두 독립적으로 결합하지는 않는다. 두 연산의 표현력과 표본 위치도 다르다.

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
  Input shape:  [1, 64, 56, 56]
  Output shape: [1, 128, 56, 56]
  Parameters:   73,856
  Receptive field: 5×5

Standard 5×5:
  Input shape:  [1, 64, 56, 56]
  Output shape: [1, 128, 56, 56]
  Parameters:   204,928
  Receptive field: 5×5
```

이 예제는 두 층 모두 패딩 2로 출력 크기를 56×56으로 유지한다. 입력·출력 모양과 수용 영역의 경계 범위는 같지만, 팽창 3×3은 그 안의 9개 위치만 참조하고 표준 5×5는 25개를 참조한다.

---

## 5. 격자 무늬 문제

### 무엇이 문제인가

**같은 팽창률**의 합성곱을 쌓으면 어떤 입력 자리는 아예 표본이 되지 않는 "격자 무늬" 흠이 생긴다.

$d=2$인 3×3 층 둘을 쌓으면 점화식이 주는 수용 영역은 $1 + 2 \cdot 2 + 2 \cdot 2 = 9$, 곧 9×9이다. 그런데 그 9×9 안에서 실제로 표본이 되는 자리는 짝수 자리뿐이다.

```text
Layer 1 samples (RF 5x5)     Two d=2 layers stacked (RF 9x9)
[●][ ][●][ ][●]              [●][ ][●][ ][●][ ][●][ ][●]
[ ][ ][ ][ ][ ]              [ ][ ][ ][ ][ ][ ][ ][ ][ ]
[●][ ][●][ ][●]              [●][ ][●][ ][●][ ][●][ ][●]
[ ][ ][ ][ ][ ]              [ ][ ][ ][ ][ ][ ][ ][ ][ ]
[●][ ][●][ ][●]              [●][ ][●][ ][●][ ][●][ ][●]
                             [ ][ ][ ][ ][ ][ ][ ][ ][ ]
                             [●][ ][●][ ][●][ ][●][ ][●]
                             [ ][ ][ ][ ][ ][ ][ ][ ][ ]
                             [●][ ][●][ ][●][ ][●][ ][●]

Problem: 81 positions in range, only 25 ever sampled.
         The other 56 are NEVER seen, no matter how deep the stack goes.
```

눈여겨볼 것은 층을 더 쌓아도 빈칸이 메워지지 않는다는 점이다. 수용 영역의 너비는 $9 \to 13 \to 17$로 계속 넓어지지만, 팽창률이 2로 같은 한 홀수 자리는 영영 표본이 되지 않는다. 넓어지는 것은 격자 자체일 뿐이다.

### 팽창률 설계

서로 다른 팽창률과 표준 합성곱을 조합해 격자 간격과 빈 위치를 줄일 수 있다. 그러나 팽창률의 최대공약수가 1이라는 조건만으로 유한한 수용 영역의 모든 위치가 연결되지는 않는다. 예를 들어 커널 3인 두 층의 팽창률을 1과 100으로 두면 최대공약수는 1이지만 넓은 빈 구간이 남는다.

$1,2,4,8$이나 $1,2,5$ 같은 조합의 연결 범위는 커널 크기와 층 순서까지 포함해 계산한다. 같은 $d>1$만 반복하는 경우에는 한 출력이 접근하는 인덱스가 특정 나머지류에 제한된다. 여러 분기를 결합하거나 더 조밀한 연산을 넣는 설계도 가능하지만, 단순히 팽창률을 바꾸었다는 이유로 빈 위치가 모두 사라진다고 가정하지 않는다.

---

## 6. 조밀 예측을 위한 팽창

### 분할의 어려움

시맨틱 세그멘테이션에는 다음이 필요하다.

1. **넓은 수용 영역**: 맥락을 알기 위해 (이 픽셀는 고양이의 일부인가 개의 일부인가?)
2. **높은 해상도의 출력**: 섬세한 경계를 지키기 위해

표준 CNN은 딜레마에 놓인다.

- (보폭이나 풀링으로) 다운샘플링하면 수용 영역은 넓어지지만 해상도를 잃는다
- 업샘플링로 해상도는 되찾지만 정보는 이미 사라졌다

### 팽창 합성곱이라는 해법

팽창 합성곱은 **다운샘플링 없이 넓은 수용 영역**을 준다.

```text
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

도식의 위쪽은 다운샘플링으로 특징 맵의 해상도를 낮추고 아래쪽은 해상도를 유지한다. 낮은 해상도 특징의 한 위치는 원본 격자에서 더 큰 간격에 대응하지만 특정 개수의 픽셀을 완전히 대표한다는 뜻은 아니다. 해상도를 유지하면 이후 층의 계산·메모리 비용도 더 크게 든다.

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
    def __init__(self, in_channels, out_channels, rates=(6, 12, 18)):
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

**출력:**

```
ASPP: torch.Size([2, 256, 28, 28]) → torch.Size([2, 256, 28, 28])
Parameters: 2,231,296
```

들고 난 크기가 둘 다 28×28인 것이 요점이다. 팽창률 6, 12, 18인 가지는 각각 $K_{\text{eff}} = 13, 25, 37$을 보면서도 `padding=rate` 덕분에 크기를 그대로 돌려준다. 매개변수 223만 개 가운데 177만 개가 세 팽창 가지의 3×3 합성곱에서 온다.

---

## 7. WaveNet과 시간 방향 팽창

### 음향의 어려움

음향 생성은 수만 개 표본에 걸친 문맥이 필요할 수 있다. 팽창과 다운샘플링, 순환 연산이나 어텐션 등으로 넓은 문맥을 처리할 수 있으며, 인과적인 원시 파형 생성에서는 팽창 합성곱이 한 가지 효율적인 선택이다.

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
# RF = 1 + sum(d × (K-1)) = 1 + (1+2+4+...+512) = 1 + 1023 = 표본 1024개
```

이렇게 하면 층 10개와 층마다 일정한 매개변수만으로 표본 1024개의 수용 영역을 얻는다. [수용 영역](receptive_field.md) 쪽은 이 블록 세 벌을 쌓아 층 30개로 $3 \times 1023 + 1 = 3070$을 얻는 것을 보인다. 시간 방향 합성곱 구조는 [1차원 합성곱](conv1d.md)을 보라.

---

## 8. 실무 지침

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

분할에서는 (ResNet 같은) 분류용 백본를 가져와 다음과 같이 고치는 것이 표준적인 방법이다.

- 마지막 두 다운샘플링 단계를 없앤다
- 해상도를 지키려고 팽창 합성곱으로 갈아 끼운다
- 여러 규모의 맥락을 위해 ASPP를 더한다

```python
# 분할을 위한 ResNet 뼈대 수정
# 원래:  4단계에서 stride=2, 5단계에서 stride=2
# 수정:  4단계에서 dilation=2, 5단계에서 dilation=4

# 이렇게 하면 해상도가 1/32이 아니라 1/8로 남는다:
# 원래:  224 → 112 → 56 → 28 → 14 → 7    (보폭 32, 224/32 = 7)
# 수정:  224 → 112 → 56 → 28 → 28 → 28    (보폭 8,  224/8 = 28)
```

---

## 9. 핵심 정리

1. **팽창은 커널의 원소 사이에 틈을 넣어 수용 영역을 넓힌다.** 매개변수는 더 들지 않는다
2. **지수적으로 쌓으면**($d = 1, 2, 4, 8, \dots$) 아주 적은 층으로 거대한 수용 영역을 얻는다
3. **격자 무늬 문제**는 같은 팽창률을 쌓을 때 생긴다. 팽창률을 바꾸어 가며 써서 빠짐없이 덮어라
4. **ASPP**는 서로 다른 팽창률의 합성곱을 나란히 적용하여 여러 규모의 맥락을 붙잡는다
5. **조밀 예측에서는** 다운샘플링를 팽창으로 갈아 끼우면 넓은 수용 영역을 지키면서 공간 해상도도 지킬 수 있다
6. **WaveNet 방식의 구조**는 인과적인 팽창 합성곱으로 시간 방향을 효율적으로 다룬다

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
커널 크기가 3이고 팽창률이 1, 2, 4인 팽창 합성곱 세 층의 수용 영역을 계산하라.

</div>

??? success "연습문제 1 풀이"
    보폭이 모두 1이므로 점화식은 $r_l = r_{l-1} + (k-1) d_l$이다. $r_0 = 1$에서 시작하면 다음과 같다.

    $$r_3 = 1 + 2(1) + 2(2) + 2(4) = 1 + 2 + 4 + 8 = 15$$

    팽창 층 세 개는 수용 영역 15를 이루지만, 같은 세 층을 팽창 없이($d = 1$) 쓰면 $1 + 3 \cdot 2 = 7$에 그친다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
팽창 합성곱의 격자 아티팩트를 설명하고 최대공약수 1 조건만으로 충분한지 논하라.

</div>

??? success "연습문제 2 풀이"
    큰 간격으로 입력을 참조하면 수용 영역의 경계 안에도 연결되지 않은 위치가 남는다. 같은 d=2만 반복하면 접근 가능한 인덱스의 홀짝이 제한된다. 서로 다른 팽창률이나 표준 합성곱을 조합해 완화할 수 있지만, 커널 3·팽창률 1과 100의 반례처럼 최대공약수 1만으로 모든 위치가 덮이지는 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
팽창 합성곱을 PyTorch로 구현하고 출력마다 어느 입력 자리가 이바지하는지 그려 보라.

</div>

??? success "연습문제 3 풀이"
    아래 단일 선형 합성곱에서는 가중치를 모두 1로 두었기 때문에 출력 한 위치에서의 입력 기울기로 연결된 위치를 확인할 수 있다. 비선형 모델 일반에서는 비활성화·상쇄·수치 오차로 기울기가 0이 될 수 있어, 0 기울기만으로 구조적 연결이 없다고 단정하지 않는다.

    ```python
    import torch
    import torch.nn as nn

    conv = nn.Conv2d(1, 1, kernel_size=3, dilation=2, padding=2, bias=False)
    nn.init.constant_(conv.weight, 1.0)   # 가중치가 0이면 기울기도 0이 되므로 1로 고정

    x = torch.zeros(1, 1, 9, 9, requires_grad=True)
    y = conv(x)
    y[0, 0, 4, 4].backward()              # 가운데 출력 한 자리만 역전파

    contrib = (x.grad[0, 0] != 0)
    for row in contrib.tolist():
        print(" ".join("O" if v else "." for v in row))
    print(f"contributing positions: {int(contrib.sum())} / 81")
    print(f"k_eff = 2*(3-1)+1 = {2 * (3 - 1) + 1}")
    ```

    출력은 다음과 같다.

    ```
    . . . . . . . . .
    . . . . . . . . .
    . . O . O . O . .
    . . . . . . . . .
    . . O . O . O . .
    . . . . . . . . .
    . . O . O . O . .
    . . . . . . . . .
    . . . . . . . . .
    contributing positions: 9 / 81
    k_eff = 2*(3-1)+1 = 5
    ```

    `O`가 찍힌 자리는 행과 열이 모두 $\{2, 4, 6\}$이다. 곧 $3 \times 3 = 9$자리가 이바지하고(커널의 가중치 수와 같다), 그 자리들이 걸치는 너비는 $6 - 2 + 1 = 5$으로 $K_{\text{eff}} = d(K-1)+1 = 5$과 맞아떨어진다. 5×5 넓이를 보면서 가중치는 9개만 쓴다는 말이 이 그림이다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
팽창 합성곱과 다운샘플링의 해상도·계산 비용 절충을 설명하라.

</div>

??? success "연습문제 4 풀이"
    팽창은 출력 격자 간격을 늘리지 않고 수용 영역을 넓힐 수 있지만 높은 해상도의 특징 맵을 계속 처리하는 비용이 든다. 다운샘플링은 후속 층의 공간 크기를 줄이는 대신 위치 정보의 해상도를 낮춘다. 세그멘테이션에 팽창이 반드시 필요한 것은 아니며 인코더–디코더, 스킵 연결 등도 사용할 수 있다. 2차원에서 보폭 2는 각 축을 대략 절반으로 줄이므로 다른 조건이 같다면 후속 해당 특징 맵의 위치 수가 대략 4분의 1이다.

---

## 정리하며

팽창률 d인 커널 K의 유효 크기는 $d(K-1)+1$이다. 가중치 수는 같은 채널·커널 크기의 표준 합성곱과 같지만 연결 위치와 표현력은 다르다. 수용 영역의 지수적 증가는 지수적으로 늘린 팽창률의 결과이며, 격자 빈틈은 실제 연결 인덱스로 확인한다. 해상도를 유지하는 효과와 후속 계산·메모리 비용도 함께 평가한다.

**참고 문헌**

1. Yu, F., & Koltun, V. (2016). Multi-scale context aggregation by dilated convolutions. *ICLR 2016*.

2. Chen, L.-C., Papandreou, G., Kokkinos, I., Murphy, K., & Yuille, A. L. (2018). DeepLab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected CRFs. *IEEE TPAMI*.

3. van den Oord, A., et al. (2016). WaveNet: A generative model for raw audio. *arXiv preprint arXiv:1609.03499*.

4. Wang, P., et al. (2018). Understanding convolution for semantic segmentation. *WACV 2018*.

5. Dumoulin, V., & Visin, F. (2016). A guide to convolution arithmetic for deep learning. *arXiv preprint arXiv:1603.07285*.
