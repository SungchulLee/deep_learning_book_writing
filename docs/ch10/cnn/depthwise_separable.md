# 그룹 컨볼루션과 깊이별 분리 컨볼루션

그룹 컨볼루션은 채널을 여러 그룹으로 나누어 독립적으로 처리한다. 깊이별 컨볼루션은 그 특수한 경우이며, 뒤에 1×1 점별 컨볼루션을 연결한 깊이별 분리 컨볼루션은 공간 필터링과 채널 혼합을 나눈다.

이 구조는 웨이트와 연산량을 줄일 수 있지만 표현력과 실행 효율의 절충이 있다. 같은 입출력 모양이 같더라도 임의의 표준 컨볼루션을 똑같이 표현하는 것은 아니며 실제 속도는 목표 하드웨어에서 확인해야 한다.

여기서 그룹 수 G는 채널을 나누는 수이며 배치 크기와 다르다. 아래 깊이별 컨볼루션의 계산은 별도 표시가 없으면 채널 배수(depth multiplier)가 1인 경우를 기준으로 한다.

## 1. 표준 컨볼루션 되짚기

입력 채널이 $C_{in}$개, 출력 채널이 $C_{out}$개이고 커널 크기가 $K$인 표준 컨볼루션에 대해 다음과 같다.

- **웨이트**: $C_{out} \times C_{in} \times K \times K$
- **바이어스**: $C_{out}$ (`bias=True`일 때)
- **곱셈·덧셈(MAC) 횟수**: $C_{out} \times C_{in} \times K^2 \times H_{out} \times W_{out}$

필터마다 모든 입력 채널과 **완전히 이어져** 있는데, 계산 비용이 여기서 나온다.

바이어스를 따로 적어 둔 까닭이 있다. 아래에서 줄어드는 비를 따질 때 **웨이트는 $G$배·$C$배로 정확히 줄지만 바이어스는 전혀 줄지 않기 때문**에, 레이어 전체의 감소 비율은 바이어스 유무와 그룹 수에 따라 웨이트의 감소 비율과 달라질 수 있다. 이 페이지의 수치는 모두 `sum(p.numel() for p in m.parameters())`로 실제 모듈에서 센 값이다.

!!! warning "MAC과 FLOP은 두 배 차이가 난다"

    곱셈 한 번과 덧셈 한 번을 묶어 1 MAC이라 부르고, 이를 2 FLOP으로 세는 것이 흔한 관례다. 위 식은 **MAC**을 센 것이고, 8절의 코드는 `2 *`를 곱해 **FLOP**을 센다. 그래서 8절이 찍는 462,422,016은 위 식이 주는 $128 \cdot 64 \cdot 9 \cdot 56 \cdot 56 = 231{,}211{,}008$의 정확히 두 배다. 논문에서 "MobileNet은 569M MAdds"처럼 쓰인 숫자를 이 페이지의 FLOP과 곧바로 견주면 안 된다. 다행히 **비**는 어느 쪽으로 세든 같다.

---

## 2. 그룹 컨볼루션

### 개념

**그룹 컨볼루션**은 입력 채널과 출력 채널을 각각 $G$개의 묶음으로 나누고 묶음마다 따로 처리한다.

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
- $K_g$: 묶음 $g$의 커널

### 계산량 절약

| 지표 | 표준 | 묶음 ($G$개) | 줄어드는 비 |
|--------|----------|-------------------|-----------|
| 웨이트 | $C_{out} \times C_{in} \times K^2$ | $C_{out} \times \frac{C_{in}}{G} \times K^2$ | 정확히 $G\times$ |
| 바이어스 | $C_{out}$ | $C_{out}$ | $1\times$ (줄지 않는다) |
| MAC 횟수 | $C_{out} \times C_{in} \times K^2 \times H \times W$ | 표준의 $\frac{1}{G}$ | 정확히 $G\times$ |

바이어스가 그대로 남으므로 **레이어 전체의 파라메터는 일반적으로 정확히 $G$배 줄지 않는다**. 아래 코드가 그 차이를 찍는다.

### PyTorch 구현

```python
import torch
import torch.nn as nn

# 표준 컨볼루션
conv_standard = nn.Conv2d(64, 128, kernel_size=3, padding=1)
conv_grouped_2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, groups=2)
conv_grouped_4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, groups=4)

# 웨이트와 바이어스를 따로 센다 — 바이어스는 묶음 수와 무관하게 C_out = 128개로 고정이다
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

웨이트의 감소 비율은 그룹 수와 같지만 바이어스를 포함하면 작아진다. 이 예의 표준 73856개와 G=2의 36992개를 비교하면 약 1.997배이고, G=4의 18560개와 비교하면 약 3.979배이다. 입력·출력 채널 수가 같아지는 깊이별 예제의 수치는 다음 절에서 별도로 비교한다.

### 제약

- $C_{in}$이 $G$으로 나누어떨어져야 한다
- $C_{out}$이 $G$으로 나누어떨어져야 한다
- $G = C_{in} = C_{out}$이면 깊이별 컨볼루션이 된다

---

## 3. 깊이별 컨볼루션

### 개념

**깊이별 컨볼루션**은 $G = C_{in}$인 그룹 컨볼루션의 극단이다. 입력 채널마다 제 필터를 따로 갖는다.

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

채널마다 제 $K \times K$ 필터로 따로 컨볼루션한다.

### 계산량 분석

| 지표 | 표준 | 깊이별 | 줄어드는 비 |
|--------|----------|-----------|------------------|
| 웨이트 | $C \times C \times K^2$ | $C \times K^2$ | 정확히 $C\times$ |
| 바이어스 | $C$ | $C$ | $1\times$ (줄지 않는다) |
| MAC 횟수 | $C^2 \times K^2 \times H \times W$ | $C \times K^2 \times H \times W$ | 정확히 $C\times$ |

$C = 64$에서 웨이트는 36,864에서 576으로 정확히 64배 줄지만, 바이어스 64개가 양쪽에 그대로 남으므로 레이어 전체는 36,928에서 640으로 **57.7배** 줄어든다.

### PyTorch 구현

```python
import torch
import torch.nn as nn

# 깊이별 컨볼루션: groups = in_channels = out_channels
depthwise_conv = nn.Conv2d(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    padding=1,
    groups=64  # 핵심: groups가 채널 수와 같다
)

# 표준과 견주기
standard_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)

# 깊이별 웨이트 64 × 3 × 3 = 576, 바이어스 64 → 합계 640
# 표준   웨이트 64 × 64 × 3 × 3 = 36,864, 바이어스 64 → 합계 36,928
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

## 4. 깊이별 분리 컨볼루션

### 개념

깊이별 컨볼루션만으로는 채널 사이에 정보가 오가지 않는다. 그래서 공간을 거르는 일과 채널을 섞는 일을 따로 떼어 이어 붙인다. 깊이별 분리 컨볼루션은 표준 컨볼루션을 이렇게 두 단계로 쪼갠다.

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

### 계산량 비교

입력이 $(C_{in}, H, W)$, 출력이 $(C_{out}, H', W')$, 커널이 $K$일 때 다음과 같다(바이어스 없이 웨이트만 센다).

| 부분 | 웨이트 | MAC 횟수 |
|-----------|-----------|-------|
| 표준 | $C_{in} \times C_{out} \times K^2$ | $C_{in} \times C_{out} \times K^2 \times H' \times W'$ |
| 깊이별 | $C_{in} \times K^2$ | $C_{in} \times K^2 \times H' \times W'$ |
| 점별 | $C_{in} \times C_{out}$ | $C_{in} \times C_{out} \times H' \times W'$ |
| **깊이별 분리 합계** | $C_{in}(K^2 + C_{out})$ | $C_{in}(K^2 + C_{out}) \times H' \times W'$ |

### 줄어드는 비

$$\frac{\text{Standard}}{\text{Depthwise Separable}} = \frac{C_{in} \times C_{out} \times K^2}{C_{in} \times K^2 + C_{in} \times C_{out}} = \frac{1}{\frac{1}{C_{out}} + \frac{1}{K^2}}$$

$C_{in}$이 약분되어 사라지는 것에 주목하자. 줄어드는 비는 **입력 채널 수와 무관하고** 오직 $K$와 $C_{out}$으로 정해진다. $K = 3$이면 $1/K^2 = 1/9$이 항상 남으므로, $C_{out}$을 아무리 키워도 비는 $K^2 = 9$를 넘지 못한다. 다만 출력 채널 수가 작으면 8보다 작으며, K=1에서는 분리 단계가 오히려 웨이트를 추가한다.

- $K=3$, $C_{out}=128$: $1/(1/128 + 1/9) = 8.41$배
- $K=3$, $C_{out}=256$: $1/(1/256 + 1/9) = 8.69$배

아래 코드는 첫째 줄의 8.41배를 실제 모듈로 확인한다.

### PyTorch 구현

```python
import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    """
    깊이별 분리 컨볼루션 블록.

    다음으로 이루어진다:
    1. 깊이별 컨볼루션: 채널마다 공간적 거르기
    2. 점별 컨볼루션: 채널을 섞는 1×1 컨볼루션
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

        # 점별: 1×1 컨볼루션
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# 표준 컨볼루션과 견주기 — 양쪽 모두 bias=False이므로 여기서는 바이어스 문제가 없고
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

8절에서 같은 입출력 채널로 같은 비교를 다시 하는데 거기서는 8,768이 아니라 8,960이 나온다. 차이는 바이어스 192개(깊이별 64 + 점별 128)뿐이다. 8절의 모듈은 `bias=True`(PyTorch 기본값)로 만들었기 때문이며, 비도 8.41배가 아니라 8.24배로 조금 낮아진다.

!!! warning "연산량 감소와 실행 시간 감소는 다르다"
    깊이별 컨볼루션은 연산 집약도가 낮아 메모리 이동이나 커널 실행 비용의 영향이 커질 수 있다. 같은 FLOP 감소라도 장치·자료형·배치 크기와 구현에 따라 속도 효과가 다르다. 이 페이지는 파라메터와 연산량을 비교하며 실행 시간을 측정한 결과는 제공하지 않는다.

---

## 5. MobileNet V1 블록

MobileNet은 깊이별 분리 컨볼루션에 배치 정규화와 ReLU를 붙여 쓴다. 두 단계 사이에도 정규화와 활성화가 들어가므로, 이 블록은 표준 컨볼루션 하나가 아니라 컨볼루션 두 개짜리 작은 탑이다.

```python
import torch
import torch.nn as nn

class MobileNetV1Block(nn.Module):
    """
    MobileNet V1 방식의 깊이별 분리 블록.

    짜임:
    깊이별 컨볼루션 → BN → ReLU → 점별 컨볼루션 → BN → ReLU
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

# 같은 입출력 채널을 갖는 표준 컨볼루션 탑과 견주기
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

4절의 맨 컨볼루션끼리는 8.41배였는데 여기서는 8.08배다. 배치 정규화가 채널마다 $\gamma$와 $\beta$ 두 개씩 붙는데, 깊이별 분리 쪽에는 BN이 **두 번**(64채널과 128채널) 들어가 384개가 늘고 표준 쪽에는 **한 번**(128채널)만 들어가 256개가 늘기 때문이다. 정규화 레이어는 컨볼루션을 얇게 만들수록 상대적으로 비싸진다.

---

## 6. MobileNet V2: 역잔차

MobileNet V2는 선형 병목을 갖춘 **역잔차 블록**을 들여온다.

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

역잔차 블록은 1×1 컨볼루션으로 채널을 확장한 뒤 깊이별 컨볼루션으로 공간 피처를 계산하고 선형 투영으로 채널을 줄인다. 깊이별 공간 연산은 확장 채널 수에 비례하지만, 앞뒤의 점별 컨볼루션 비용도 포함해야 한다. 채널 확장을 공짜로 처리하는 구조는 아니다.

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
        stride: 깊이별 컨볼루션의 스트라이드
        expand_ratio: 숨은 채널의 확장 배수
    """
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super().__init__()

        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels

        hidden_channels = in_channels * expand_ratio

        layers = []

        # 확장 (1×1 컨볼루션) - expand_ratio > 1일 때만
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6(inplace=True)
            ])

        # 깊이별 컨볼루션
        layers.extend([
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride=stride,
                      padding=1, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU6(inplace=True)
        ])

        # 사영 (1×1 컨볼루션) - 선형 (활성화 없음!)
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

그룹 컨볼루션은 묶음 사이에 정보가 흐르지 못하게 한다. 1×1 그룹 컨볼루션을 여러 레이어 쌓으면 어떤 출력 채널은 끝까지 같은 묶음의 입력만 보게 된다. **채널 섞기**가 이 한계를 푼다.

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
    """묶음 컨볼루션과 채널 섞기를 쓰는 ShuffleNet V1 단위."""

    def __init__(self, in_channels, out_channels, groups=3, stride=1):
        super().__init__()

        self.stride = stride
        self.groups = groups

        mid_channels = out_channels // 4

        if stride == 2:
            out_channels = out_channels - in_channels

        # 묶음 컨볼루션 1×1
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

        # 묶음 컨볼루션 1×1
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

예시 순열은 다른 그룹의 채널을 재배치하여 다음 그룹 컨볼루션이 서로 다른 원래 그룹의 정보를 받을 수 있게 한다. 이것이 모든 가능한 채널 상호작용을 즉시 복원한다는 뜻은 아니다. 표준 컨볼루션과 비교할 때에는 입출력뿐 아니라 블록 안의 연산 구성도 함께 확인한다.

---

## 8. 효율 비교

```python
import torch
import torch.nn as nn

def count_ops_and_params(model, input_shape):
    """파라메터를 세고 컨볼루션의 부동소수점 연산 수를 어림한다.

    곱셈과 덧셈 한 쌍을 2 FLOP으로 센다. 논문에서는 이 쌍을 1 MAdd로
    세는 일이 많은데, 그러면 아래 숫자의 절반이 된다.
    컨볼루션만 세고 BatchNorm과 ReLU는 빼므로 어림값이다.
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

# 여러 컨볼루션 종류 견주기 — 여기서는 모두 bias=True(PyTorch 기본값)이다
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

비교 코드의 FLOP 계산은 곱셈·누적을 2 FLOP으로 세고 바이어스 덧셈과 다른 레이어의 연산을 제외한 근사이다. 이 규약에서 G=4의 비율은 4이고 파라메터 비율은 약 3.98이다. 깊이별 분리에서도 같은 이유로 연산량과 파라메터의 감소 비율이 다르다. 두 수치 모두 실제 지연의 비율은 아니다.

---

## 9. 구조 비교

| 구조 | 핵심 혁신 | 흔한 쓰임새 |
|--------------|----------------|------------------|
| **MobileNetV1** | 깊이별 분리 컨볼루션 | 휴대 기기 배포 |
| **MobileNetV2** | 역잔차와 선형 병목 | 휴대 기기·엣지 장치 |
| **ShuffleNet** | 채널 섞기와 그룹 컨볼루션 | 매우 효율적 |
| **EfficientNet** | 복합 스케일링과 MBConv | 최상급 효율 |
| **ResNeXt** | 잔차 블록 속 그룹 컨볼루션 | 높은 정확도 |

---

## 10. 핵심 정리

1. **그룹 컨볼루션**은 채널을 서로 독립인 묶음으로 나누어 웨이트를 정확히 $G$배 줄인다. 바이어스는 줄지 않으므로 레이어 전체는 그보다 조금 덜 줄어든다 ($G=4$에서 4.000배 대 3.979배)
2. **깊이별 컨볼루션**은 $G = C_{in}$인 그룹 컨볼루션으로, 채널마다 필터 하나를 쓴다 ($C=64$에서 웨이트 64배, 레이어 전체 57.7배)
3. **깊이별 분리** = 깊이별 + 점별이며, 줄어드는 비는 $1/(1/C_{out} + 1/K^2)$이다. $K=3$이면 $C_{out}$을 아무리 키워도 9배를 넘지 못하고, 흔한 값에서는 8~9배다
4. **채널 섞기**는 ShuffleNet에서 묶음 사이에 정보가 흐르게 해 준다
5. **역잔차**(MobileNetV2)는 선형 병목과 함께 좁음 → 넓음 → 좁음의 구조를 쓴다
6. 파라메터와 FLOP이 8배 줄었다는 것이 벽시계 시간이 8배 빨라졌다는 뜻은 아니다. 깊이별 컨볼루션은 연산 집약도가 낮아 메모리 대역폭에 묶이기 쉽고, 이 페이지에는 측정한 시간이 없다

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
깊이별 분리 컨볼루션과 표준 컨볼루션의 웨이트 수와 MAC 횟수를 유도하고, 두 비가 서로 같음을 보여라. 이어서 바이어스를 함께 세면 왜 파라메터 비만 달라지는지 설명하라.

</div>

??? success "풀이"
    **웨이트.** 표준 컨볼루션은 출력 채널마다 $C_{\text{in}} \times K \times K$짜리 필터를 하나씩 가지므로 웨이트가 $K^2 \cdot C_{\text{in}} \cdot C_{\text{out}}$개다. 깊이별 단계는 입력 채널마다 $K \times K$ 필터 하나씩이므로 $K^2 \cdot C_{\text{in}}$개, 점별 단계는 $1 \times 1$ 필터를 $C_{\text{out}}$개 쓰되 각각 $C_{\text{in}}$개의 입력을 받으므로 $C_{\text{in}} \cdot C_{\text{out}}$개다. 합쳐서 $C_{\text{in}}(K^2 + C_{\text{out}})$개다.

    **MAC 횟수.** 출력 자리 $H' \times W'$마다 위 웨이트를 한 번씩 쓰므로, 두 경우 모두 웨이트 수에 $H'W'$을 곱한 값이다. 표준은 $K^2 C_{\text{in}} C_{\text{out}} H' W'$, 깊이별 분리는 $C_{\text{in}}(K^2 + C_{\text{out}}) H' W'$이다.

    **비.** $H'W'$이 분모와 분자에서 함께 약분되므로 웨이트 비와 MAC 비는 같다.

    $$\frac{K^2 C_{\text{in}} C_{\text{out}}}{C_{\text{in}}(K^2 + C_{\text{out}})} = \frac{K^2 C_{\text{out}}}{K^2 + C_{\text{out}}} = \frac{1}{\frac{1}{C_{\text{out}}} + \frac{1}{K^2}}$$

    $C_{\text{in}}$도 약분되어 사라지므로 비는 입력 채널 수와 무관하다. $C_{\text{out}} = 256$, $K = 3$이면 $1/(1/256 + 1/9) = 8.69$배다.

    **바이어스.** 표준 컨볼루션의 바이어스는 $C_{\text{out}}$개, 깊이별 분리는 깊이별 $C_{\text{in}}$개와 점별 $C_{\text{out}}$개를 합쳐 $C_{\text{in}} + C_{\text{out}}$개다. 즉 바이어스는 줄기는커녕 **늘어난다**. 게다가 바이어스는 출력 자리마다 덧셈 한 번만 더할 뿐이라 MAC 횟수에는 사실상 기여하지 않는다. 그래서 MAC 비는 위 식대로지만 파라메터 비는 그보다 낮아진다. $C_{\text{in}} = 64$, $C_{\text{out}} = 128$, $K = 3$에서 8절이 찍는 대로 MAC 비는 8.41배, 파라메터 비는 $73{,}856 / 8{,}960 = 8.24$배다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
`groups` 파라메터를 써서 깊이별 분리 컨볼루션을 PyTorch로 구현하고, 파라메터 수를 실제로 세어 표준 컨볼루션과 비교하여라.

</div>

??? success "풀이"
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

    $8{,}960 = (576 + 64) + (8{,}192 + 128)$이다. `padding=k//2`는 $k$가 홀수일 때 공간 크기를 그대로 지킨다. 4절의 8,768과 어긋나는 까닭은 오직 바이어스 192개인데, 4절은 `bias=False`로 만들었기 때문이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
깊이별 컨볼루션과 점별 컨볼루션의 역할을 구분하고 MobileNet 계열의 설계와 연결하라.

</div>

??? success "풀이"
    깊이별 컨볼루션은 입력 채널마다 공간 필터링을 수행하고 점별 컨볼루션은 채널을 혼합한다. 채널 배수 1이고 중간 비선형성이 없는 두 단계를 합치면 유효 커널은 $W_{o,c,m,n}=P_{o,c}D_{c,m,n}$로 제한된다. 점별 혼합이 있어도 임의의 표준 커널의 자유도가 모두 복원되지는 않는다. MobileNetV2는 확장·깊이별·선형 투영을 결합하고 EfficientNet의 MBConv도 이에 기반하되 SE 등의 구성을 더한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
깊이별 분리 컨볼루션을 쓸 때의 트레이드오프는 무엇인가? 표준 컨볼루션이 나을 때는 언제인가?

</div>

??? success "풀이"
    깊이별 분리 컨볼루션은 파라메터와 계산에서 효율적이지만, 깊이별 단계가 채널을 섞지 않으므로 모델의 용량이 줄어들 수 있다. 다음 경우에는 표준 컨볼루션이 낫다.

    1. 모델의 용량이 중요하고 계산이 병목이 아닐 때
    2. 채널 수가 적어 아끼는 양이 얼마 안 될 때. 줄어드는 비 $1/(1/C_{\text{out}} + 1/K^2)$는 $C_{\text{out}}$이 작을수록 작아진다. $C_{\text{out}} = 16$, $K = 3$이면 5.76배에 그친다
    3. 하드웨어가 조밀한 행렬 곱에 맞추어져 있을 때. 4절의 경고대로 FLOP이 8배 줄어도 벽시계 시간은 그만큼 줄지 않으며, 깊이별 커널이 잘 최적화되지 않은 환경에서는 오히려 느려지기도 한다

---

## 정리하며

그룹 수 G인 컨볼루션은 웨이트를 표준의 1/G로 줄인다. 채널 배수 1인 깊이별 분리 컨볼루션은 $C_{in}(K^2+C_{out})$개의 웨이트를 사용하고, 표준 대비 감소 비율은 $1/(1/C_{out}+1/K^2)$이다. 바이어스·정규화·활성화의 비용은 별도로 세며, 표현력과 실제 지연은 이 비율만으로 판단하지 않는다.

**참고 문헌**

1. Chollet, F. (2017). "Xception: Deep Learning with Depthwise Separable Convolutions."
2. Howard, A. G., et al. (2017). "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications."
3. Sandler, M., et al. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks."
4. Zhang, X., et al. (2018). "ShuffleNet: An Extremely Computation-Efficient CNN for Mobile Devices."
5. Tan, M., & Le, Q. V. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks."
