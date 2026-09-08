# 묶음 합성곱과 깊이별 분리 합성곱

표준 합성곱은 입력 채널과 출력 채널이 조밀하게 이어져 있어 계산 비용이 크다. **묶음 합성곱**과 **깊이별 분리 합성곱**은 합성곱 연산을 쪼개는 구조적 혁신으로, 성능은 지키거나 오히려 높이면서 매개변수와 계산량을 크게 줄인다.

이 기법들은 MobileNet, EfficientNet, ShuffleNet, ResNeXt 같은 효율적인 CNN 구조의 바탕이며, 휴대 기기와 말단 장치에 모델을 올릴 수 있게 해 준다.

---

## 1. 표준 합성곱 되짚기

입력 채널이 $C_{in}$개, 출력 채널이 $C_{out}$개이고 핵 크기가 $K$인 표준 합성곱에 대해 다음과 같다.

- **매개변수**: $C_{out} \times C_{in} \times K \times K$
- **부동소수점 연산 수**: $C_{out} \times C_{in} \times K^2 \times H_{out} \times W_{out}$

필터마다 모든 입력 채널과 **완전히 이어져** 있는데, 계산 비용이 여기서 나온다.

---

## 2. 묶음 합성곱

### 개념

**묶음 합성곱**은 입력 채널과 출력 채널을 각각 $G$개의 묶음으로 나누고 묶음마다 따로 처리한다.

```
Standard Convolution:           Grouped Convolution (G=2):
                                
C_in ────────────→ C_out        C_in/2 ──→ C_out/2  (Group 1)
(all channels      (all           
connected)         outputs)      C_in/2 ──→ C_out/2  (Group 2)
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

| 지표 | 표준 | 묶음 (묶음 G개) | 줄어드는 비 |
|--------|----------|-------------------|-----------|
| 매개변수 | $C_{out} \times C_{in} \times K^2$ | $C_{out} \times \frac{C_{in}}{G} \times K^2$ | $G\times$ |
| 부동소수점 연산 수 | $C_{out} \times C_{in} \times K^2 \times H \times W$ | 표준의 $\frac{1}{G}$ | $G\times$ |

### PyTorch 구현

```python
import torch
import torch.nn as nn

# 표준 합성곱
conv_standard = nn.Conv2d(64, 128, kernel_size=3, padding=1)
params_standard = sum(p.numel() for p in conv_standard.parameters())
print(f"Standard conv params: {params_standard:,}")  # 73,856

# 묶음 합성곱 (G=2)
conv_grouped_2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, groups=2)
params_grouped_2 = sum(p.numel() for p in conv_grouped_2.parameters())
print(f"Grouped conv (G=2) params: {params_grouped_2:,}")  # 36,992 (2배 줄어듦)

# 묶음 합성곱 (G=4)
conv_grouped_4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, groups=4)
params_grouped_4 = sum(p.numel() for p in conv_grouped_4.parameters())
print(f"Grouped conv (G=4) params: {params_grouped_4:,}")  # 18,560 (4배 줄어듦)

# 모양 확인
x = torch.randn(1, 64, 32, 32)
print(f"\nInput shape: {x.shape}")
print(f"Standard output: {conv_standard(x).shape}")
print(f"Grouped (G=2) output: {conv_grouped_2(x).shape}")
print(f"Grouped (G=4) output: {conv_grouped_4(x).shape}")
```

### 제약

- $C_{in}$이 $G$으로 나누어떨어져야 한다
- $C_{out}$이 $G$으로 나누어떨어져야 한다
- $G = C_{in} = C_{out}$이면 깊이별 합성곱이 된다

---

## 3. 깊이별 합성곱

### 개념

**깊이별 합성곱**은 $G = C_{in}$인 묶음 합성곱의 극단이다. 입력 채널마다 제 필터를 따로 갖는다.

```
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
| 매개변수 | $C \times C \times K^2$ | $C \times K^2$ | $C\times$ |
| 부동소수점 연산 수 | $C^2 \times K^2 \times H \times W$ | $C \times K^2 \times H \times W$ | $C\times$ |

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

params = sum(p.numel() for p in depthwise_conv.parameters())
print(f"Depthwise conv params: {params:,}")  # 640 (64 × 3 × 3 + 편향 64)

# 표준과 견주기
standard_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
standard_params = sum(p.numel() for p in standard_conv.parameters())
print(f"Standard conv params: {standard_params:,}")  # 36,928

print(f"Reduction: {standard_params / params:.1f}×")  # 약 58배 줄어듦
```

---

## 4. 깊이별 분리 합성곱

### 개념

깊이별 분리 합성곱은 표준 합성곱을 두 단계로 쪼갠다.

1. **깊이별**: 공간적인 거르기 (채널마다 공간 무늬를 잡는다)
2. **점별 (1×1)**: 채널 섞기 (채널에 걸친 정보를 엮는다)

```
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

입력이 $(C_{in}, H, W)$, 출력이 $(C_{out}, H', W')$, 핵이 $K$일 때 다음과 같다.

| 부분 | 매개변수 | 부동소수점 연산 수 |
|-----------|-----------|-------|
| 표준 | $C_{in} \times C_{out} \times K^2$ | $C_{in} \times C_{out} \times K^2 \times H' \times W'$ |
| 깊이별 | $C_{in} \times K^2$ | $C_{in} \times K^2 \times H' \times W'$ |
| 점별 | $C_{in} \times C_{out}$ | $C_{in} \times C_{out} \times H' \times W'$ |
| **깊이별 분리 합계** | $C_{in}(K^2 + C_{out})$ | $C_{in}(K^2 + C_{out}) \times H' \times W'$ |

### 줄어드는 비

$$\frac{\text{Standard}}{\text{Depthwise Separable}} = \frac{C_{in} \times C_{out} \times K^2}{C_{in} \times K^2 + C_{in} \times C_{out}} = \frac{1}{\frac{1}{C_{out}} + \frac{1}{K^2}}$$

$K=3$, $C_{out}=256$이면 **약 8~9배 줄어든다**.

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

# 표준 합성곱과 견주기
in_ch, out_ch = 64, 128
kernel_size = 3

standard = nn.Conv2d(in_ch, out_ch, kernel_size, padding=1, bias=False)
ds_conv = DepthwiseSeparableConv(in_ch, out_ch, kernel_size, padding=1, bias=False)

standard_params = sum(p.numel() for p in standard.parameters())
ds_params = sum(p.numel() for p in ds_conv.parameters())

print(f"Standard conv params: {standard_params:,}")     # 73,728
print(f"Depthwise separable params: {ds_params:,}")     # 8,768
print(f"Reduction: {standard_params / ds_params:.1f}×")  # ~8.4×

# 출력 모양이 맞는지 확인
x = torch.randn(1, in_ch, 32, 32)
print(f"\nStandard output: {standard(x).shape}")
print(f"DS conv output: {ds_conv(x).shape}")
```

---

## 5. MobileNet V1 블록

MobileNet은 깊이별 분리 합성곱에 배치 정규화와 ReLU를 붙여 쓴다.

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
```

---

## 6. MobileNet V2: 뒤집은 잔차

MobileNet V2는 선형 병목을 갖춘 **뒤집은 잔차 블록**을 들여온다.

```
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

# 사용 예
block = InvertedResidual(32, 32, stride=1, expand_ratio=6)
x = torch.randn(1, 32, 56, 56)
out = block(x)
print(f"Input: {x.shape}, Output: {out.shape}")
print(f"Parameters: {sum(p.numel() for p in block.parameters()):,}")
```

---

## 7. 채널 섞기 (ShuffleNet)

묶음 합성곱은 묶음 사이에 정보가 흐르지 못하게 한다. **채널 섞기**가 이 한계를 푼다.

```
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
    묶음과 채널을 전치한 뒤 다시 펼친다.
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

# 시험
block = ShuffleNetBlock(24, 24, groups=3, stride=1)
x = torch.randn(1, 24, 56, 56)
out = block(x)
print(f"ShuffleNet block: {x.shape} → {out.shape}")
```

---

## 8. 효율 견주기

```python
import torch
import torch.nn as nn

def count_ops_and_params(model, input_shape):
    """매개변수를 세고 부동소수점 연산 수를 어림한다."""
    params = sum(p.numel() for p in model.parameters())
    
    flops = 0
    x = torch.randn(*input_shape)
    
    def hook(module, input, output):
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

# 여러 합성곱 종류 견주기
in_ch, out_ch = 64, 128
H, W = 56, 56

standard = nn.Conv2d(in_ch, out_ch, 3, padding=1)
ds_conv = nn.Sequential(
    nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch),
    nn.Conv2d(in_ch, out_ch, 1)
)
grouped = nn.Conv2d(in_ch, out_ch, 3, padding=1, groups=4)

input_shape = (1, in_ch, H, W)

print("Comparison of Convolution Types:")
print("-" * 50)
for name, model in [("Standard", standard), 
                    ("Depthwise Sep", ds_conv), 
                    ("Grouped (G=4)", grouped)]:
    params, flops = count_ops_and_params(model, input_shape)
    print(f"{name:15s}: Params={params:>10,}, FLOPs={flops:>15,}")
```

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

1. **묶음 합성곱**은 채널을 서로 독립인 묶음으로 나누어 매개변수를 $G$배 줄인다
2. **깊이별 합성곱**은 $G = C_{in}$인 묶음 합성곱으로, 채널마다 필터 하나를 쓴다
3. **깊이별 분리** = 깊이별 + 점별이며, 매개변수를 약 8~9배 줄인다
4. **채널 섞기**는 ShuffleNet에서 묶음 사이에 정보가 흐르게 해 준다
5. **뒤집은 잔차**(MobileNetV2)는 선형 병목과 함께 좁음 → 넓음 → 좁음의 짜임을 쓴다
6. 이 기법들 덕분에 정확도를 크게 잃지 않고도 휴대 기기와 말단 장치에 올릴 효율적인 모델을 만들 수 있다

---

## 연습문제

**연습문제 1.**
깊이별 분리 합성곱과 표준 합성곱의 매개변수 수와 부동소수점 연산 수를 유도하라.

??? success "연습문제 1 풀이"
    표준 합성곱은 매개변수가 $K^2 \cdot C_{\text{in}} \cdot C_{\text{out}}$개, 연산이 $K^2 \cdot C_{\text{in}} \cdot C_{\text{out}} \cdot H \cdot W$번이다. 깊이별 분리 합성곱은 매개변수가 $K^2 \cdot C_{\text{in}} + C_{\text{in}} \cdot C_{\text{out}}$개이다. 비는 $1/C_{\text{out}} + 1/K^2$이다. $C_{\text{out}}=256, K=3$이면 매개변수가 약 $8{\sim}9\times$ 적다.

---

**연습문제 2.**
`groups` 매개변수를 써서 깊이별 분리 합성곱을 PyTorch로 구현하라.

??? success "연습문제 2 풀이"
    ```python
    class DepthwiseSeparable(nn.Module):
        def __init__(self, c_in, c_out, k=3):
            super().__init__()
            self.depthwise = nn.Conv2d(c_in, c_in, k, padding=k//2, groups=c_in)
            self.pointwise = nn.Conv2d(c_in, c_out, 1)
        def forward(self, x):
            return self.pointwise(self.depthwise(x))
    ```

---

**연습문제 3.**
MobileNet과 EfficientNet의 설계에서 묶음 합성곱이 하는 구실을 설명하라.

??? success "연습문제 3 풀이"
    MobileNet은 깊이별 분리 합성곱(groups = $C_{\text{in}}$)으로 계산을 약 9분의 1로 줄인다. EfficientNet은 깊이와 너비와 해상도의 균형을 잡는 복합 규모 조정을 쓰며 깊이별 분리 합성곱을 구성 블록으로 삼는다. 묶음 합성곱은 연산량을 줄이면서도 점별로 섞는 단계 덕분에 표현력을 지킨다.

---

**연습문제 4.**
깊이별 분리 합성곱을 쓸 때의 맞바꿈은 무엇인가? 표준 합성곱이 나을 때는 언제인가?

??? success "연습문제 4 풀이"
    깊이별 분리 합성곱은 매개변수와 계산에서 효율적이지만, 깊이별 단계가 채널을 섞지 않으므로 모델의 용량이 줄어들 수 있다. (1) 모델의 용량이 중요하고 계산이 병목이 아닐 때, (2) 채널 수가 적어 아끼는 양이 얼마 안 될 때, (3) 하드웨어가 조밀한 행렬 곱에 맞추어져 있을 때는 표준 합성곱이 낫다.

## 정리하며

| 종류 | 매개변수 | 줄어드는 비 | 쓰임새 |
|------|------------|-----------|----------|
| 표준 | $C_{out} \times C_{in} \times K^2$ | — | 기준선 |
| 묶음 (G) | $\div G$ | $G\times$ | ResNeXt |
| 깊이별 | $C \times K^2$ | $C\times$ | 공간적 거르기 |
| 깊이별 분리 | $C_{in}(K^2 + C_{out})$ | 약 8~9배 | MobileNet, EfficientNet |
| 뒤집은 잔차 | 확장과 깊이별 | 효율적 | MobileNetV2 이후 |

**참고 문헌**

1. Chollet, F. (2017). "Xception: Deep Learning with Depthwise Separable Convolutions."
2. Howard, A. G., et al. (2017). "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications."
3. Sandler, M., et al. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks."
4. Zhang, X., et al. (2018). "ShuffleNet: An Extremely Computation-Efficient CNN for Mobile Devices."
5. Tan, M., & Le, Q. V. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks."
