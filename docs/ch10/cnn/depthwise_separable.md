# 묶음 합성곱과 깊이별 분리 합성곱

표준 합성곱은 입력 채널과 출력 채널이 조밀하게 이어져 있어 계산 비용이 크다. **묶음 합성곱**(grouped convolution)과 **깊이별 분리 합성곱**(depthwise separable convolution)은 합성곱 연산을 쪼개는 구조적 혁신으로, 같은 입출력 모양을 지키면서 매개변수와 곱셈·덧셈 횟수를 크게 줄인다.

정확도까지 함께 올라가는 것은 아니다. 아낀 매개변수를 층을 더 쌓는 데 되돌려 쓰면 정확도가 올라갈 수 있지만, 같은 깊이·같은 너비에서 표준 합성곱을 깊이별 분리로 바꾸면 채널을 섞는 자유도가 줄어 표현력이 조금 떨어진다. 이 맞바꿈은 연습문제 4에서 다시 다룬다.

이 기법들은 MobileNet, EfficientNet, ShuffleNet, ResNeXt 같은 효율적인 CNN 구조의 바탕이며, 휴대 기기와 말단 장치에 모델을 올릴 수 있게 해 준다.

!!! note "용어: 묶음(group)과 배치(batch)는 다르다"

    이 쪽에서 $G$는 **채널을 나눈 묶음의 개수**이지, 한 번에 밀어 넣는 표본의 개수인 배치 크기가 아니다. 두 낱말이 헷갈리기 쉬워 이 책은 grouped convolution을 **묶음 합성곱**으로 적는다. 다른 글에서는 "그룹 합성곱"이라고도 한다. 같은 쪽에 나오는 `nn.BatchNorm2d`(배치 정규화)의 "배치"는 여전히 표본 묶음을 뜻한다.

---

## 1. 표준 합성곱 되짚기

입력 채널이 $C_{in}$개, 출력 채널이 $C_{out}$개이고 핵 크기가 $K$인 표준 합성곱에 대해 다음과 같다.

- **가중치**: $C_{out} \times C_{in} \times K \times K$
- **편향**: $C_{out}$ (`bias=True`일 때)
- **곱셈·덧셈(MAC) 횟수**: $C_{out} \times C_{in} \times K^2 \times H_{out} \times W_{out}$

필터마다 모든 입력 채널과 **완전히 이어져** 있는데, 계산 비용이 여기서 나온다.

편향을 따로 적어 둔 까닭이 있다. 아래에서 줄어드는 비를 따질 때 **가중치는 $G$배·$C$배로 정확히 줄지만 편향은 전혀 줄지 않기 때문**에, 층 전체의 매개변수 비는 언제나 가중치만 볼 때보다 조금 작다. 이 쪽의 수치는 모두 `sum(p.numel() for p in m.parameters())`로 실제 모듈에서 센 값이다.

!!! warning "MAC과 FLOP은 두 배 차이가 난다"

    곱셈 한 번과 덧셈 한 번을 묶어 1 MAC이라 부르고, 이를 2 FLOP으로 세는 것이 흔한 관례다. 위 식은 **MAC**을 센 것이고, 8절의 코드는 `2 *`를 곱해 **FLOP**을 센다. 그래서 8절이 찍는 462,422,016은 위 식이 주는 $128 \cdot 64 \cdot 9 \cdot 56 \cdot 56 = 231{,}211{,}008$의 정확히 두 배다. 논문에서 "MobileNet은 569M MAdds"처럼 쓰인 숫자를 이 쪽의 FLOP과 곧바로 견주면 안 된다. 다행히 **비**는 어느 쪽으로 세든 같다.

---

## 2. 묶음 합성곱

### 개념

**묶음 합성곱**은 입력 채널과 출력 채널을 각각 $G$개의 묶음으로 나누고 묶음마다 따로 처리한다.

```text
Standard Convolution:           Grouped Convolution (G=2):

C_in ────────────→ C_out        C_in/2 ──→ C_out/2  (Group 1)
(all channels      (all
connected)         outputs)     C_in/2 ──→ C_out/2  (Group 2)
```

- 입력을 채널 $C_{in}/G$개씩 $G$개의 묶음으로 나눈다
- 묶음마다 제 필터로 채널 $C_{out}/G$개를 낸다
- 출력을 채널 차원으로 이어 붙인다

### 수식으로 나타내기

묶음 $g$($g = 0, 1, \dots, G-1$)에 대해 다음과 같다.

$$Y_{g}[o, i, j] = \sum_{c=0}^{C_{in}/G - 1} \sum_{m,n} X_g[c, i+m, j+n] \cdot K_g[o, c, m, n]$$

여기서 각 기호는 다음과 같다.

- $X_g$: 입력 채널 $[g \cdot C_{in}/G, (g+1) \cdot C_{in}/G)$
- $Y_g$: 출력 채널 $[g \cdot C_{out}/G, (g+1) \cdot C_{out}/G)$
- $K_g$: 묶음 $g$의 핵

### 계산량 절약

| 지표 | 표준 | 묶음 ($G$개) | 줄어드는 비 |
|--------|----------|-------------------|-----------|
| 가중치 | $C_{out} \times C_{in} \times K^2$ | $C_{out} \times \frac{C_{in}}{G} \times K^2$ | 정확히 $G\times$ |
| 편향 | $C_{out}$ | $C_{out}$ | $1\times$ (줄지 않는다) |
| MAC 횟수 | $C_{out} \times C_{in} \times K^2 \times H \times W$ | 표준의 $\frac{1}{G}$ | 정확히 $G\times$ |

편향이 그대로 남으므로 **층 전체의 매개변수는 정확히 $G$배 줄지 않는다**. 아래 코드가 그 차이를 찍는다.

### PyTorch 구현

```python
import torch
import torch.nn as nn

# 표준 합성곱
conv_standard = nn.Conv2d(64, 128, kernel_size=3, padding=1)
conv_grouped_2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, groups=2)
conv_grouped_4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, groups=4)

# 가중치와 편향을 따로 센다 — 편향은 묶음 수와 무관하게 C_out = 128개로 고정이다
for name, conv in [("Standard", conv_standard),
                   ("Grouped (G=2)", conv_grouped_2),
                   ("Grouped (G=4)", conv_grouped_4)]:
    w = conv.weight.numel()
    b = conv.bias.numel()
    print(f"{name:15s}: weight={w:>7,}  bias={b:>4,}  total={w + b:>7,}")

w_std = conv_standard.weight.numel()
p_std = sum(p.numel() for p in conv_standard.parameters())
print()
for name, conv in [("G=2", conv_grouped_2), ("G=4", conv_grouped_4)]:
    w = conv.weight.numel()
    p = sum(q.numel() for q in conv.parameters())
    print(f"{name}: weight ratio = {w_std / w:.3f}x,  total ratio = {p_std / p:.3f}x")

# 모양 확인
x = torch.randn(1, 64, 32, 32)
print(f"\nInput shape: {x.shape}")
print(f"Standard output: {conv_standard(x).shape}")
print(f"Grouped (G=2) output: {conv_grouped_2(x).shape}")
print(f"Grouped (G=4) output: {conv_grouped_4(x).shape}")
```

**출력:**

```
Standard       : weight= 73,728  bias= 128  total= 73,856
Grouped (G=2)  : weight= 36,864  bias= 128  total= 36,992
Grouped (G=4)  : weight= 18,432  bias= 128  total= 18,560

G=2: weight ratio = 2.000x,  total ratio = 1.997x
G=4: weight ratio = 4.000x,  total ratio = 3.979x

Input shape: torch.Size([1, 64, 32, 32])
Standard output: torch.Size([1, 128, 32, 32])
Grouped (G=2) output: torch.Size([1, 128, 32, 32])
Grouped (G=4) output: torch.Size([1, 128, 32, 32])
```

가중치 비는 2.000과 4.000으로 딱 떨어지지만 층 전체로는 1.997과 3.979다. 편향 128개가 세 경우 모두 그대로 남기 때문이다. $G$가 커질수록 이 어긋남은 커진다. 가중치가 줄어드는 만큼 줄지 않는 편향의 비중이 커지기 때문이며, $G = 64$까지 밀면 3절에서 보듯 64배가 아니라 57.7배가 된다.

### 제약

- $C_{in}$이 $G$으로 나누어떨어져야 한다
- $C_{out}$이 $G$으로 나누어떨어져야 한다
- $G = C_{in} = C_{out}$이면 깊이별 합성곱이 된다

---

## 3. 깊이별 합성곱

### 개념

**깊이별 합성곱**은 $G = C_{in}$인 묶음 합성곱의 극단이다. 입력 채널마다 제 필터를 따로 갖는다.

```text
Depthwise Convolution:

Channel 1 ──[Filter 1]──→ Output Channel 1
Channel 2 ──[Filter 2]──→ Output Channel 2
Channel 3 ──[Filter 3]──→ Output Channel 3
    ⋮            ⋮              ⋮
Channel C ──[Filter C]──→ Output Channel C
```

### 수식으로 나타내기

$$Y[c, i, j] = \sum_{m,n} X[c, i+m, j+n] \cdot K[c, m, n]$$

채널마다 제 $K \times K$ 필터로 따로 합성곱한다.

### 계산량 분석

| 지표 | 표준 | 깊이별 | 줄어드는 비 |
|--------|----------|-----------|------------------|
| 가중치 | $C \times C \times K^2$ | $C \times K^2$ | 정확히 $C\times$ |
| 편향 | $C$ | $C$ | $1\times$ (줄지 않는다) |
| MAC 횟수 | $C^2 \times K^2 \times H \times W$ | $C \times K^2 \times H \times W$ | 정확히 $C\times$ |

$C = 64$에서 가중치는 36,864에서 576으로 정확히 64배 줄지만, 편향 64개가 양쪽에 그대로 남으므로 층 전체는 36,928에서 640으로 **57.7배** 줄어든다.

### PyTorch 구현

```python
import torch
import torch.nn as nn

# 깊이별 합성곱: groups = in_channels = out_channels
depthwise_conv = nn.Conv2d(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    padding=1,
    groups=64  # 핵심: groups가 채널 수와 같다
)

# 표준과 견주기
standard_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)

# 깊이별 가중치 64 × 3 × 3 = 576, 편향 64 → 합계 640
# 표준   가중치 64 × 64 × 3 × 3 = 36,864, 편향 64 → 합계 36,928
for name, conv in [("Depthwise", depthwise_conv), ("Standard", standard_conv)]:
    w = conv.weight.numel()
    b = conv.bias.numel()
    print(f"{name:10s}: weight={w:>6,}  bias={b:>3,}  total={w + b:>6,}")

w_ratio = standard_conv.weight.numel() / depthwise_conv.weight.numel()
p_ratio = (sum(p.numel() for p in standard_conv.parameters())
           / sum(p.numel() for p in depthwise_conv.parameters()))
print(f"\nWeight reduction: {w_ratio:.1f}x   (= C = 64)")
print(f"Total reduction:  {p_ratio:.1f}x   (biases do not shrink)")
```

**출력:**

```
Depthwise : weight=   576  bias= 64  total=   640
Standard  : weight=36,864  bias= 64  total=36,928

Weight reduction: 64.0x   (= C = 64)
Total reduction:  57.7x   (biases do not shrink)
```

---

## 4. 깊이별 분리 합성곱

### 개념

깊이별 합성곱만으로는 채널 사이에 정보가 오가지 않는다. 그래서 공간을 거르는 일과 채널을 섞는 일을 따로 떼어 이어 붙인다. 깊이별 분리 합성곱은 표준 합성곱을 이렇게 두 단계로 쪼갠다.

1. **깊이별**: 공간적인 거르기 (채널마다 공간 무늬를 잡는다)
2. **점별 (1×1)**: 채널 섞기 (채널에 걸친 정보를 엮는다)

```text
Standard Convolution:
Input (C_in, H, W) ──[K×K×C_in×C_out]──→ Output (C_out, H, W)

Depthwise Separable Convolution:
Input (C_in, H, W) ──[Depthwise K×K×C_in]──→ (C_in, H, W) ──[Pointwise 1×1×C_in×C_out]──→ Output (C_out, H, W)
```

### 수식으로 나타내기

**1단계 — 깊이별**:

$$M[c, i, j] = \sum_{m,n} X[c, i+m, j+n] \cdot K_{dw}[c, m, n]$$

**2단계 — 점별**:

$$Y[o, i, j] = \sum_{c=0}^{C_{in}-1} M[c, i, j] \cdot K_{pw}[o, c]$$

### 계산량 견주기

입력이 $(C_{in}, H, W)$, 출력이 $(C_{out}, H', W')$, 핵이 $K$일 때 다음과 같다(편향 없이 가중치만 센다).

| 부분 | 가중치 | MAC 횟수 |
|-----------|-----------|-------|
| 표준 | $C_{in} \times C_{out} \times K^2$ | $C_{in} \times C_{out} \times K^2 \times H' \times W'$ |
| 깊이별 | $C_{in} \times K^2$ | $C_{in} \times K^2 \times H' \times W'$ |
| 점별 | $C_{in} \times C_{out}$ | $C_{in} \times C_{out} \times H' \times W'$ |
| **깊이별 분리 합계** | $C_{in}(K^2 + C_{out})$ | $C_{in}(K^2 + C_{out}) \times H' \times W'$ |

### 줄어드는 비

$$\frac{\text{Standard}}{\text{Depthwise Separable}} = \frac{C_{in} \times C_{out} \times K^2}{C_{in} \times K^2 + C_{in} \times C_{out}} = \frac{1}{\frac{1}{C_{out}} + \frac{1}{K^2}}$$

$C_{in}$이 약분되어 사라지는 것에 주목하자. 줄어드는 비는 **입력 채널 수와 무관하고** 오직 $K$와 $C_{out}$으로 정해진다. $K = 3$이면 $1/K^2 = 1/9$이 항상 남으므로, $C_{out}$을 아무리 키워도 비는 $K^2 = 9$를 넘지 못한다. 그래서 흔히 쓰는 값에서 이 비는 늘 8과 9 사이에 놓인다.

- $K=3$, $C_{out}=128$: $1/(1/128 + 1/9) = 8.41$배
- $K=3$, $C_{out}=256$: $1/(1/256 + 1/9) = 8.69$배

아래 코드는 첫째 줄의 8.41배를 실제 모듈로 확인한다.

### PyTorch 구현

```python
import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    """
    깊이별 분리 합성곱 블록.

    다음으로 이루어진다:
    1. 깊이별 합성곱: 채널마다 공간적 거르기
    2. 점별 합성곱: 채널을 섞는 1×1 합성곱
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, bias=False):
        super().__init__()

        # 깊이별: groups = in_channels
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias
        )

        # 점별: 1×1 합성곱
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# 표준 합성곱과 견주기 — 양쪽 모두 bias=False이므로 여기서는 편향 문제가 없고
# 비가 앞서 유도한 8.41배와 정확히 맞아떨어진다
in_ch, out_ch = 64, 128
kernel_size = 3

standard = nn.Conv2d(in_ch, out_ch, kernel_size, padding=1, bias=False)
ds_conv = DepthwiseSeparableConv(in_ch, out_ch, kernel_size, padding=1, bias=False)

standard_params = sum(p.numel() for p in standard.parameters())
ds_params = sum(p.numel() for p in ds_conv.parameters())

print(f"Standard conv params: {standard_params:,}")     # 64 × 128 × 9 = 73,728
print(f"  depthwise: {sum(p.numel() for p in ds_conv.depthwise.parameters()):,}")
print(f"  pointwise: {sum(p.numel() for p in ds_conv.pointwise.parameters()):,}")
print(f"Depthwise separable params: {ds_params:,}")     # 576 + 8,192 = 8,768
print(f"Reduction: {standard_params / ds_params:.2f}×")  # 8.41

# 출력 모양이 맞는지 확인
x = torch.randn(1, in_ch, 32, 32)
print(f"\nStandard output: {standard(x).shape}")
print(f"DS conv output: {ds_conv(x).shape}")
```

**출력:**

```
Standard conv params: 73,728
  depthwise: 576
  pointwise: 8,192
Depthwise separable params: 8,768
Reduction: 8.41×

Standard output: torch.Size([1, 128, 32, 32])
DS conv output: torch.Size([1, 128, 32, 32])
```

8절에서 같은 입출력 채널로 같은 비교를 다시 하는데 거기서는 8,768이 아니라 8,960이 나온다. 차이는 편향 192개(깊이별 64 + 점별 128)뿐이다. 8절의 모듈은 `bias=True`(PyTorch 기본값)로 만들었기 때문이며, 비도 8.41배가 아니라 8.24배로 조금 낮아진다.

!!! warning "매개변수가 8배 줄었다고 8배 빨라지는 것은 아니다"

    이 쪽이 세는 것은 매개변수와 곱셈·덧셈 횟수이지 실제 걸린 시간이 아니다. 깊이별 합성곱은 가중치 하나당 하는 일이 적어 연산 집약도가 낮고, 그래서 GPU에서는 계산이 아니라 메모리 대역폭에 발목이 잡히기 쉽다. 8.4배 줄어든 FLOP이 8.4배의 벽시계 시간 단축으로 이어지지 않으며, 커널 구현과 하드웨어에 따라 2~3배에 그치거나 배치가 작을 때는 오히려 느려지기도 한다. **이 쪽에는 측정한 시간이 한 줄도 실려 있지 않다.** 자기 하드웨어에서 실제 시간이 궁금하면 직접 재야 한다.

---

## 5. MobileNet V1 블록

MobileNet은 깊이별 분리 합성곱에 배치 정규화와 ReLU를 붙여 쓴다. 두 단계 사이에도 정규화와 활성화가 들어가므로, 이 블록은 표준 합성곱 하나가 아니라 합성곱 두 개짜리 작은 탑이다.

```python
import torch
import torch.nn as nn

class MobileNetV1Block(nn.Module):
    """
    MobileNet V1 방식의 깊이별 분리 블록.

    짜임:
    깊이별 합성곱 → BN → ReLU → 점별 합성곱 → BN → ReLU
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride,
                      padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# 같은 입출력 채널을 갖는 표준 합성곱 탑과 견주기
block = MobileNetV1Block(64, 128)
standard_block = nn.Sequential(
    nn.Conv2d(64, 128, 3, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True)
)

x = torch.randn(1, 64, 32, 32)
print(f"MobileNetV1Block: {tuple(x.shape)} -> {tuple(block(x).shape)}")
print(f"  depthwise part: {sum(p.numel() for p in block.depthwise.parameters()):,}")
print(f"  pointwise part: {sum(p.numel() for p in block.pointwise.parameters()):,}")
print(f"  total:          {sum(p.numel() for p in block.parameters()):,}")
print(f"Standard conv + BN + ReLU: {sum(p.numel() for p in standard_block.parameters()):,}")
print(f"Reduction: {sum(p.numel() for p in standard_block.parameters()) / sum(p.numel() for p in block.parameters()):.2f}x")
```

**출력:**

```
MobileNetV1Block: (1, 64, 32, 32) -> (1, 128, 32, 32)
  depthwise part: 704
  pointwise part: 8,448
  total:          9,152
Standard conv + BN + ReLU: 73,984
Reduction: 8.08x
```

4절의 맨 합성곱끼리는 8.41배였는데 여기서는 8.08배다. 배치 정규화가 채널마다 $\gamma$와 $\beta$ 두 개씩 붙는데, 깊이별 분리 쪽에는 BN이 **두 번**(64채널과 128채널) 들어가 384개가 늘고 표준 쪽에는 **한 번**(128채널)만 들어가 256개가 늘기 때문이다. 정규화 층은 합성곱을 얇게 만들수록 상대적으로 비싸진다.

---

## 6. MobileNet V2: 뒤집은 잔차

MobileNet V2는 선형 병목을 갖춘 **뒤집은 잔차 블록**을 들여온다.

```text
Standard Residual:          Inverted Residual:
wide → narrow → wide        narrow → wide → narrow

Input (C)                   Input (C)
    ↓                           ↓
Conv 1×1 (C→C/4)           Conv 1×1 (C→C×t)  [Expansion]
    ↓                           ↓
Conv 3×3 (C/4)             DWConv 3×3 (C×t)  [Depthwise]
    ↓                           ↓
Conv 1×1 (C/4→C)           Conv 1×1 (C×t→C') [Projection]
    ↓                           ↓
Add residual               Add residual (if stride=1 and C=C')
```

넓은 가운데 층을 둘 수 있는 까닭이 깊이별 합성곱에 있다. 표준 3×3으로 채널을 6배 늘렸다면 가중치가 채널 수의 제곱으로 불어나지만, 깊이별은 채널 수에 비례할 뿐이어서 $t = 6$까지 부풀려도 감당이 된다.

### PyTorch 구현

```python
import torch
import torch.nn as nn

class InvertedResidual(nn.Module):
    """
    MobileNet V2의 뒤집은 잔차 블록.

    인수:
        in_channels: 입력 채널
        out_channels: 출력 채널
        stride: 깊이별 합성곱의 보폭
        expand_ratio: 숨은 채널의 확장 배수
    """
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super().__init__()

        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels

        hidden_channels = in_channels * expand_ratio

        layers = []

        # 확장 (1×1 합성곱) - expand_ratio > 1일 때만
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6(inplace=True)
            ])

        # 깊이별 합성곱
        layers.extend([
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride=stride,
                      padding=1, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU6(inplace=True)
        ])

        # 사영 (1×1 합성곱) - 선형 (활성화 없음!)
        layers.extend([
            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

# 사용 예: 숨은 채널은 32 × 6 = 192개
# 확장 32×192 = 6,144 · BN 384 | 깊이별 192×9 = 1,728 · BN 384
# 사영 192×32 = 6,144 · BN 64  → 합계 14,848
block = InvertedResidual(32, 32, stride=1, expand_ratio=6)
x = torch.randn(1, 32, 56, 56)
out = block(x)
print(f"Input: {x.shape}, Output: {out.shape}")
print(f"Parameters: {sum(p.numel() for p in block.parameters()):,}")
```

**출력:**

```
Input: torch.Size([1, 32, 56, 56]), Output: torch.Size([1, 32, 56, 56])
Parameters: 14,848
```

---

## 7. 채널 섞기 (ShuffleNet)

묶음 합성곱은 묶음 사이에 정보가 흐르지 못하게 한다. 1×1 묶음 합성곱을 여러 층 쌓으면 어떤 출력 채널은 끝까지 같은 묶음의 입력만 보게 된다. **채널 섞기**가 이 한계를 푼다.

```text
Before Shuffle:                After Shuffle:
Group 1: [a₁, a₂, a₃]         [a₁, b₁, c₁]
Group 2: [b₁, b₂, b₃]    →    [a₂, b₂, c₂]
Group 3: [c₁, c₂, c₃]         [a₃, b₃, c₃]
```

### PyTorch 구현

```python
import torch
import torch.nn as nn

def channel_shuffle(x, groups):
    """
    채널 섞기 연산.

    텐서를 (N, C, H, W)에서 (N, G, C//G, H, W)로 바꾸고
    묶음 축과 묶음 안 채널 축을 전치한 뒤 다시 펼친다.
    """
    N, C, H, W = x.shape

    # 모양 바꾸기: (N, C, H, W) → (N, G, C//G, H, W)
    x = x.view(N, groups, C // groups, H, W)

    # 전치: (N, G, C//G, H, W) → (N, C//G, G, H, W)
    x = x.transpose(1, 2).contiguous()

    # 펼치기: (N, C//G, G, H, W) → (N, C, H, W)
    x = x.view(N, C, H, W)

    return x

class ShuffleNetBlock(nn.Module):
    """묶음 합성곱과 채널 섞기를 쓰는 ShuffleNet V1 단위."""

    def __init__(self, in_channels, out_channels, groups=3, stride=1):
        super().__init__()

        self.stride = stride
        self.groups = groups

        mid_channels = out_channels // 4

        if stride == 2:
            out_channels = out_channels - in_channels

        # 묶음 합성곱 1×1
        self.gconv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, groups=groups, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # 깊이별 3×3
        self.dwconv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, stride=stride,
                      padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels)
        )

        # 묶음 합성곱 1×1
        self.gconv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # 지름길
        if stride == 2:
            self.shortcut = nn.AvgPool2d(3, stride=2, padding=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.gconv1(x)
        out = channel_shuffle(out, self.groups)
        out = self.dwconv(out)
        out = self.gconv2(out)

        shortcut = self.shortcut(x)

        if self.stride == 2:
            out = torch.cat([out, shortcut], dim=1)
        else:
            out = out + shortcut

        return nn.functional.relu(out)

# 섞기가 하는 일을 직접 보기: 채널 9개를 묶음 3개 [0,1,2] [3,4,5] [6,7,8]로 볼 때
# 새 묶음은 [0,3,6] [1,4,7] [2,5,8]이 되어 묶음마다 옛 묶음 셋을 하나씩 물려받는다
demo = torch.arange(9).float().view(1, 9, 1, 1)
print(f"Before shuffle: {[int(v) for v in demo.flatten()]}")
print(f"After  shuffle: {[int(v) for v in channel_shuffle(demo, 3).flatten()]}")

# 시험: mid_channels = 24 // 4 = 6
block = ShuffleNetBlock(24, 24, groups=3, stride=1)
x = torch.randn(1, 24, 56, 56)
out = block(x)
print(f"\nShuffleNet block: {tuple(x.shape)} -> {tuple(out.shape)}")
print(f"Parameters: {sum(p.numel() for p in block.parameters()):,}")
```

**출력:**

```
Before shuffle: [0, 1, 2, 3, 4, 5, 6, 7, 8]
After  shuffle: [0, 3, 6, 1, 4, 7, 2, 5, 8]

ShuffleNet block: (1, 24, 56, 56) -> (1, 24, 56, 56)
Parameters: 222
```

찍힌 순열 `[0, 3, 6, 1, 4, 7, 2, 5, 8]`이 위 그림 그대로다. 채널 0·3·6은 원래 서로 다른 묶음에 있었으므로, 다음 묶음 합성곱의 첫 묶음은 이제 세 묶음 모두에서 정보를 받는다. 매개변수 222개는 이 블록이 얼마나 얇은지를 보여 준다. 같은 24채널을 잇는 표준 3×3 합성곱 하나만 해도 가중치가 $24 \times 24 \times 9 = 5{,}184$개다.

---

## 8. 효율 견주기

```python
import torch
import torch.nn as nn

def count_ops_and_params(model, input_shape):
    """매개변수를 세고 합성곱의 부동소수점 연산 수를 어림한다.

    곱셈과 덧셈 한 쌍을 2 FLOP으로 센다. 논문에서는 이 쌍을 1 MAdd로
    세는 일이 많은데, 그러면 아래 숫자의 절반이 된다.
    합성곱만 세고 BatchNorm과 ReLU는 빼므로 어림값이다.
    """
    params = sum(p.numel() for p in model.parameters())

    flops = 0
    x = torch.randn(*input_shape)

    def hook(module, inputs, output):
        nonlocal flops
        if isinstance(module, nn.Conv2d):
            out_h, out_w = output.shape[2:]
            flops += (2 * module.kernel_size[0] * module.kernel_size[1] *
                      module.in_channels * module.out_channels *
                      out_h * out_w // module.groups)

    hooks = []
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(hook))

    _ = model(x)

    for h in hooks:
        h.remove()

    return params, flops

# 여러 합성곱 종류 견주기 — 여기서는 모두 bias=True(PyTorch 기본값)이다
in_ch, out_ch = 64, 128
H, W = 56, 56

standard = nn.Conv2d(in_ch, out_ch, 3, padding=1)
ds_conv = nn.Sequential(
    nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch),
    nn.Conv2d(in_ch, out_ch, 1)
)
grouped = nn.Conv2d(in_ch, out_ch, 3, padding=1, groups=4)

input_shape = (1, in_ch, H, W)

base_params, base_flops = count_ops_and_params(standard, input_shape)

print("Comparison of Convolution Types:")
print("-" * 72)
for name, model in [("Standard", standard),
                    ("Depthwise Sep", ds_conv),
                    ("Grouped (G=4)", grouped)]:
    params, flops = count_ops_and_params(model, input_shape)
    print(f"{name:15s}: Params={params:>10,}, FLOPs={flops:>15,}"
          f"  ({base_params / params:>5.2f}x, {base_flops / flops:>5.2f}x)")
```

**출력:**

```
Comparison of Convolution Types:
------------------------------------------------------------------------
Standard       : Params=    73,856, FLOPs=    462,422,016  ( 1.00x,  1.00x)
Depthwise Sep  : Params=     8,960, FLOPs=     54,992,896  ( 8.24x,  8.41x)
Grouped (G=4)  : Params=    18,560, FLOPs=    115,605,504  ( 3.98x,  4.00x)
```

마지막 줄이 이 쪽 전체의 요점을 한 줄로 보여 준다. 묶음 합성곱($G=4$)의 FLOP 비는 **정확히 4.00배**인데 매개변수 비는 **3.98배**다. 편향은 계산량에 거의 기여하지 않아 FLOP 쪽 나눗셈에서는 빠지지만, 매개변수 쪽에서는 128개가 줄지 않은 채 남아 비를 끌어내린다. 깊이별 분리도 같은 이유로 8.41(FLOP)과 8.24(매개변수)로 갈린다.

---

## 9. 구조 견주기

| 구조 | 핵심 혁신 | 흔한 쓰임새 |
|--------------|----------------|------------------|
| **MobileNetV1** | 깊이별 분리 합성곱 | 휴대 기기 배포 |
| **MobileNetV2** | 뒤집은 잔차와 선형 병목 | 휴대 기기·말단 장치 |
| **ShuffleNet** | 채널 섞기와 묶음 합성곱 | 매우 효율적 |
| **EfficientNet** | 복합 규모 조정과 MBConv | 최상급 효율 |
| **ResNeXt** | 잔차 블록 속 묶음 합성곱 | 높은 정확도 |

---

## 10. 핵심 정리

1. **묶음 합성곱**은 채널을 서로 독립인 묶음으로 나누어 가중치를 정확히 $G$배 줄인다. 편향은 줄지 않으므로 층 전체는 그보다 조금 덜 줄어든다 ($G=4$에서 4.000배 대 3.979배)
2. **깊이별 합성곱**은 $G = C_{in}$인 묶음 합성곱으로, 채널마다 필터 하나를 쓴다 ($C=64$에서 가중치 64배, 층 전체 57.7배)
3. **깊이별 분리** = 깊이별 + 점별이며, 줄어드는 비는 $1/(1/C_{out} + 1/K^2)$이다. $K=3$이면 $C_{out}$을 아무리 키워도 9배를 넘지 못하고, 흔한 값에서는 8~9배다
4. **채널 섞기**는 ShuffleNet에서 묶음 사이에 정보가 흐르게 해 준다
5. **뒤집은 잔차**(MobileNetV2)는 선형 병목과 함께 좁음 → 넓음 → 좁음의 짜임을 쓴다
6. 매개변수와 FLOP이 8배 줄었다는 것이 벽시계 시간이 8배 빨라졌다는 뜻은 아니다. 깊이별 합성곱은 연산 집약도가 낮아 메모리 대역폭에 묶이기 쉽고, 이 쪽에는 측정한 시간이 없다

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
깊이별 분리 합성곱과 표준 합성곱의 가중치 수와 MAC 횟수를 유도하고, 두 비가 서로 같음을 보여라. 이어서 편향을 함께 세면 왜 매개변수 비만 달라지는지 설명하라.

</div>

??? success "연습문제 1 풀이"
    **가중치.** 표준 합성곱은 출력 채널마다 $C_{\text{in}} \times K \times K$짜리 필터를 하나씩 가지므로 가중치가 $K^2 \cdot C_{\text{in}} \cdot C_{\text{out}}$개다. 깊이별 단계는 입력 채널마다 $K \times K$ 필터 하나씩이므로 $K^2 \cdot C_{\text{in}}$개, 점별 단계는 $1 \times 1$ 필터를 $C_{\text{out}}$개 쓰되 각각 $C_{\text{in}}$개의 입력을 받으므로 $C_{\text{in}} \cdot C_{\text{out}}$개다. 합쳐서 $C_{\text{in}}(K^2 + C_{\text{out}})$개다.

    **MAC 횟수.** 출력 자리 $H' \times W'$마다 위 가중치를 한 번씩 쓰므로, 두 경우 모두 가중치 수에 $H'W'$을 곱한 값이다. 표준은 $K^2 C_{\text{in}} C_{\text{out}} H' W'$, 깊이별 분리는 $C_{\text{in}}(K^2 + C_{\text{out}}) H' W'$이다.

    **비.** $H'W'$이 분모와 분자에서 함께 약분되므로 가중치 비와 MAC 비는 같다.

    $$\frac{K^2 C_{\text{in}} C_{\text{out}}}{C_{\text{in}}(K^2 + C_{\text{out}})} = \frac{K^2 C_{\text{out}}}{K^2 + C_{\text{out}}} = \frac{1}{\frac{1}{C_{\text{out}}} + \frac{1}{K^2}}$$

    $C_{\text{in}}$도 약분되어 사라지므로 비는 입력 채널 수와 무관하다. $C_{\text{out}} = 256$, $K = 3$이면 $1/(1/256 + 1/9) = 8.69$배다.

    **편향.** 표준 합성곱의 편향은 $C_{\text{out}}$개, 깊이별 분리는 깊이별 $C_{\text{in}}$개와 점별 $C_{\text{out}}$개를 합쳐 $C_{\text{in}} + C_{\text{out}}$개다. 즉 편향은 줄기는커녕 **늘어난다**. 게다가 편향은 출력 자리마다 덧셈 한 번만 더할 뿐이라 MAC 횟수에는 사실상 기여하지 않는다. 그래서 MAC 비는 위 식대로지만 매개변수 비는 그보다 낮아진다. $C_{\text{in}} = 64$, $C_{\text{out}} = 128$, $K = 3$에서 8절이 찍는 대로 MAC 비는 8.41배, 매개변수 비는 $73{,}856 / 8{,}960 = 8.24$배다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
`groups` 매개변수를 써서 깊이별 분리 합성곱을 PyTorch로 구현하고, 매개변수 수를 실제로 세어 표준 합성곱과 견주어라.

</div>

??? success "연습문제 2 풀이"
    ```python
    import torch
    import torch.nn as nn

    class DepthwiseSeparable(nn.Module):
        def __init__(self, c_in, c_out, k=3):
            super().__init__()
            # groups=c_in이 깊이별을 만든다: 채널마다 제 k×k 필터 하나
            self.depthwise = nn.Conv2d(c_in, c_in, k, padding=k // 2, groups=c_in)
            # 1×1이 채널을 섞는다
            self.pointwise = nn.Conv2d(c_in, c_out, 1)

        def forward(self, x):
            return self.pointwise(self.depthwise(x))

    ds = DepthwiseSeparable(64, 128)
    std = nn.Conv2d(64, 128, 3, padding=1)
    p_ds = sum(p.numel() for p in ds.parameters())
    p_std = sum(p.numel() for p in std.parameters())
    print(f"DS: {p_ds:,}  Standard: {p_std:,}  Reduction: {p_std / p_ds:.2f}x")
    print(f"Output: {tuple(ds(torch.randn(1, 64, 32, 32)).shape)}")
    ```

    출력은 다음과 같다.

    ```
    DS: 8,960  Standard: 73,856  Reduction: 8.24x
    Output: (1, 128, 32, 32)
    ```

    $8{,}960 = (576 + 64) + (8{,}192 + 128)$이다. `padding=k//2`는 $k$가 홀수일 때 공간 크기를 그대로 지킨다. 4절의 8,768과 어긋나는 까닭은 오직 편향 192개인데, 4절은 `bias=False`로 만들었기 때문이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
MobileNet과 EfficientNet의 설계에서 묶음 합성곱이 하는 구실을 설명하라.

</div>

??? success "연습문제 3 풀이"
    MobileNet은 묶음 합성곱을 극단까지 민 깊이별 합성곱($G = C_{\text{in}}$)에 1×1 점별 합성곱을 붙여 쓴다. 3절과 4절의 계산대로 곱셈·덧셈 횟수를 $1/(1/C_{\text{out}} + 1/K^2)$로, $K=3$일 때 8~9분의 1로 줄인다. EfficientNet은 깊이와 너비와 해상도의 균형을 잡는 복합 규모 조정을 쓰며, 그 구성 블록인 MBConv는 MobileNetV2의 뒤집은 잔차(6절)를 그대로 가져온 것이다.

    묶음 합성곱이 맡는 구실은 **공간 거르기와 채널 섞기를 떼어 놓는 것**이다. 묶음만 쓰면 묶음 사이에 정보가 오가지 않아 표현력이 깎이는데, 뒤이은 점별 1×1이 모든 채널을 다시 잇기 때문에 그 손실이 메워진다. ShuffleNet은 이 1×1마저 묶음으로 만들어 더 아끼고, 대신 채널 섞기(7절)로 정보를 통하게 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
깊이별 분리 합성곱을 쓸 때의 맞바꿈은 무엇인가? 표준 합성곱이 나을 때는 언제인가?

</div>

??? success "연습문제 4 풀이"
    깊이별 분리 합성곱은 매개변수와 계산에서 효율적이지만, 깊이별 단계가 채널을 섞지 않으므로 모델의 용량이 줄어들 수 있다. 다음 경우에는 표준 합성곱이 낫다.

    1. 모델의 용량이 중요하고 계산이 병목이 아닐 때
    2. 채널 수가 적어 아끼는 양이 얼마 안 될 때. 줄어드는 비 $1/(1/C_{\text{out}} + 1/K^2)$는 $C_{\text{out}}$이 작을수록 작아진다. $C_{\text{out}} = 16$, $K = 3$이면 5.76배에 그친다
    3. 하드웨어가 조밀한 행렬 곱에 맞추어져 있을 때. 4절의 경고대로 FLOP이 8배 줄어도 벽시계 시간은 그만큼 줄지 않으며, 깊이별 커널이 잘 최적화되지 않은 환경에서는 오히려 느려지기도 한다

---

## 정리하며

| 종류 | 가중치 | 가중치가 줄어드는 비 | 쓰임새 |
|------|------------|-----------|----------|
| 표준 | $C_{out} \times C_{in} \times K^2$ | — | 기준선 |
| 묶음 ($G$) | $\div G$ | 정확히 $G\times$ | ResNeXt |
| 깊이별 | $C \times K^2$ | 정확히 $C\times$ | 공간적 거르기 |
| 깊이별 분리 | $C_{in}(K^2 + C_{out})$ | $1/(1/C_{out} + 1/K^2)$, $K=3$에서 8~9배 | MobileNet, EfficientNet |
| 뒤집은 잔차 | 확장과 깊이별 | 효율적 | MobileNetV2 이후 |

편향까지 세면 층 전체의 비는 위 값보다 늘 조금 작다. 이 쪽이 찍은 세 쌍 — 4.000 대 3.979, 64.0 대 57.7, 8.41 대 8.24 — 이 그 차이다.

**참고 문헌**

1. Chollet, F. (2017). "Xception: Deep Learning with Depthwise Separable Convolutions."
2. Howard, A. G., et al. (2017). "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications."
3. Sandler, M., et al. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks."
4. Zhang, X., et al. (2018). "ShuffleNet: An Extremely Computation-Efficient CNN for Mobile Devices."
5. Tan, M., & Le, Q. V. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks."
