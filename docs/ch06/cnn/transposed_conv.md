# 전치 합성곱
## 들어가며

**전치 합성곱**(**분수 보폭 합성곱**, 또는 흔히 **역합성곱**이라고도 한다)은 보통 합성곱을 입력에 대해 미분한 연산이다. 표준 합성곱이 대체로 공간 차원을 줄이는 데 반해 전치 합성곱은 이를 **늘리므로**, 신경망에서 학습 가능한 상향 표본화의 대표적인 방법이 된다.

전치 합성곱은 다음에서 꼭 필요한 부품이다.

- 의미 분할을 위한 **부호기-복호기 구조** (U-Net, SegNet)
- 이미지 합성을 위한 **생성 모델** (GAN, VAE)
- 이미지를 키우는 **초해상도 신경망**
- 여러 규모의 물체 탐지를 위한 **특징 피라미드 신경망**

> **용어에 대한 참고**: "역합성곱"이라는 이름은 엄밀히 말하면 틀렸다(역합성곱은 합성곱을 되돌리는 신호 처리의 특정 연산이다). 수학적으로 정확한 말은 "전치 합성곱"이며, 합성곱의 퇴플리츠 행렬을 전치한 것을 곱한다는 뜻이다.

---

## 수학적 바탕

### 행렬 곱으로 본 합성곱

전치 합성곱을 이해하려면 먼저 표준 합성곱을 행렬 곱으로 나타내야 한다. 1차원 입력 $\mathbf{x} \in \mathbb{R}^5$과 핵 $\mathbf{k} = [k_0, k_1, k_2]$에 대해 유효 합성곱 $\mathbf{y} = \mathbf{C}\mathbf{x}$은 다음을 쓴다.

$$\mathbf{C} = \begin{bmatrix}
k_0 & k_1 & k_2 & 0 & 0 \\
0 & k_0 & k_1 & k_2 & 0 \\
0 & 0 & k_0 & k_1 & k_2
\end{bmatrix}$$

이는 $\mathbb{R}^5 \to \mathbb{R}^3$으로 보낸다(하향 표본화).

### 전치한 연산

**전치 합성곱**은 $\mathbf{C}^\top$을 쓴다.

$$\mathbf{C}^\top = \begin{bmatrix}
k_0 & 0 & 0 \\
k_1 & k_0 & 0 \\
k_2 & k_1 & k_0 \\
0 & k_2 & k_1 \\
0 & 0 & k_2
\end{bmatrix}$$

이는 $\mathbb{R}^3 \to \mathbb{R}^5$으로 보낸다(상향 표본화). 핵의 가중치는 같지만 이어지는 방식이 뒤집힌다.

### 핵심 착상: 합성곱의 기울기

입력에 대한 합성곱의 역전파가 바로 전치 합성곱이다.

$$\frac{\partial L}{\partial \mathbf{x}} = \mathbf{C}^\top \frac{\partial L}{\partial \mathbf{y}}$$

그래서 역전파에서 전치 합성곱이 자연스럽게 나타난다.

```python
import torch
import torch.nn.functional as F

# 순전파: conv2d
x = torch.randn(1, 3, 8, 8, requires_grad=True)
w = torch.randn(16, 3, 3, 3, requires_grad=True)
y = F.conv2d(x, w, padding=1)

# 역전파: conv_transpose2d와 같다
grad_output = torch.randn_like(y)
y.backward(grad_output)

# 손수 확인
grad_input_manual = F.conv_transpose2d(grad_output, w, padding=1)
print(f"Gradient match: {torch.allclose(x.grad, grad_input_manual, atol=1e-5)}")
```

---

## 전치 합성곱이 움직이는 방식

### 상향 표본화 장치

전치 합성곱은 다음과 같이 이해할 수 있다.

1. (보폭이 1보다 크면) 입력 원소 사이에 **0을 끼워 넣는다**
2. 입력을 **덧댄다**
3. 같은 핵으로 **표준 합성곱**을 적용한다

3×3 핵을 쓰는 보폭 2 전치 합성곱에서는 다음과 같다.

```
Input (2×2):       Insert zeros (3×3):      Pad (5×5):           Convolve (4×4 output):
┌───┬───┐          ┌───┬───┬───┐           ┌───┬───┬───┬───┬───┐
│ a │ b │    →      │ a │ 0 │ b │     →     │ 0 │ 0 │ 0 │ 0 │ 0 │  →  4×4 output
├───┼───┤          ├───┼───┼───┤           ├───┼───┼───┼───┼───┤
│ c │ d │          │ 0 │ 0 │ 0 │           │ 0 │ a │ 0 │ b │ 0 │
└───┴───┘          ├───┼───┼───┤           ├───┼───┼───┼───┼───┤
                   │ c │ 0 │ d │           │ 0 │ 0 │ 0 │ 0 │ 0 │
                   └───┴───┴───┘           ├───┼───┼───┼───┼───┤
                                           │ 0 │ c │ 0 │ d │ 0 │
                                           ├───┼───┼───┼───┼───┤
                                           │ 0 │ 0 │ 0 │ 0 │ 0 │
                                           └───┴───┴───┴───┴───┘
```

### 출력 크기 공식

전치 합성곱에 대해 다음과 같다.

$$H_{out} = (H_{in} - 1) \times s - 2p + d(K - 1) + p_{out} + 1$$

여기서 각 기호는 다음과 같다.

- $H_{in}$: 입력의 높이
- $s$: 보폭
- $p$: 덧대기
- $d$: 팽창률
- $K$: 핵 크기
- $p_{out}$: 출력 덧대기 (모호함을 없앤다)

3×3 핵으로 보폭 2 상향 표본화를 하는 흔한 경우에는 다음과 같다.

$$H_{out} = (H_{in} - 1) \times 2 - 2 \times 1 + 3 + 1 = 2 \times H_{in}$$

---

## PyTorch 구현

### 기본 사용법

```python
import torch
import torch.nn as nn

# 보통 합성곱 (하향 표본화)
conv = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)

# 전치 합성곱 (상향 표본화)
conv_transpose = nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, 
                                     padding=1, output_padding=1)

x = torch.randn(1, 64, 32, 32)

# 하향 표본화
y = conv(x)
print(f"Conv: {x.shape} → {y.shape}")  # [1, 64, 32, 32] → [1, 32, 16, 16]

# 상향 표본화
z = conv_transpose(y)
print(f"ConvT: {y.shape} → {z.shape}")  # [1, 32, 16, 16] → [1, 64, 32, 32]
```

### `output_padding` 매개변수

보폭이 1보다 크면 보통 합성곱에서 여러 입력 크기가 같은 출력 크기를 낼 수 있다. 이를테면 보폭 2에서는 31×31 입력과 32×32 입력이 모두 16×16 출력을 낸다. `output_padding`이 이 모호함을 없앤다.

```python
# output_padding이 없으면 32×32가 아니라 31×31이 될 수 있다
conv_t_no_op = nn.ConvTranspose2d(32, 64, 3, stride=2, padding=1)
# output_padding=1이면 32×32가 보장된다
conv_t_with_op = nn.ConvTranspose2d(32, 64, 3, stride=2, padding=1, output_padding=1)

y = torch.randn(1, 32, 16, 16)
print(f"Without output_padding: {conv_t_no_op(y).shape}")     # [1, 64, 31, 31]
print(f"With output_padding=1: {conv_t_with_op(y).shape}")    # [1, 64, 32, 32]
```

### 매개변수의 수

전치 합성곱은 입력 채널과 출력 채널을 맞바꾼 보통 합성곱과 매개변수 수가 같다.

```python
# 보통: 채널 64개 → 32개
conv = nn.Conv2d(64, 32, 3, bias=False)
print(f"Conv2d params: {sum(p.numel() for p in conv.parameters()):,}")
# 32 × 64 × 3 × 3 = 18,432

# 전치: 채널 32개 → 64개
conv_t = nn.ConvTranspose2d(32, 64, 3, bias=False)
print(f"ConvTranspose2d params: {sum(p.numel() for p in conv_t.parameters()):,}")
# 32 × 64 × 3 × 3 = 18,432 (같다!)
```

---

## 바둑판 무늬 흠 문제

### 무엇이 문제인가

보폭이 1보다 큰 전치 합성곱은 **바둑판 무늬 흠**을 만드는 것으로 악명 높다. 이는 핵이 고르지 않게 겹쳐 출력에 격자 같은 무늬가 생기는 것이다.

```
Stride-2 ConvTranspose with 3×3 kernel:

Contribution count at each output position:
┌───┬───┬───┬───┬───┬───┐
│ 1 │ 1 │ 2 │ 1 │ 2 │ 1 │    Uneven overlap creates
├───┼───┼───┼───┼───┼───┤    a checkerboard pattern
│ 1 │ 1 │ 2 │ 1 │ 2 │ 1 │    where some positions
├───┼───┼───┼───┼───┼───┤    receive more contributions
│ 2 │ 2 │ 4 │ 2 │ 4 │ 2 │    than others
├───┼───┼───┼───┼───┼───┤
│ 1 │ 1 │ 2 │ 1 │ 2 │ 1 │
└───┴───┴───┴───┴───┴───┘
```

### 해법 1: 핵 크기를 보폭의 배수로

보폭으로 딱 나누어떨어지는 핵 크기를 쓴다.

```python
# 나쁨: stride=2, kernel=3 → 고르지 않은 겹침
bad = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)

# 나음: stride=2, kernel=4 → 고른 겹침
better = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)

# 이것도 좋음: stride=2, kernel=2 → 겹침 없음
good = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
```

### 해법 2: 크기 조정 뒤 합성곱 (권장)

상향 표본화와 합성곱을 떼어 놓아 전치 합성곱을 아예 쓰지 않는 더 깔끔한 방법이다.

```python
class UpsampleConv(nn.Module):
    """
    보간 뒤 합성곱으로 상향 표본화한다.
    전치 합성곱에서 오는 바둑판 무늬 흠을 피한다.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 scale_factor=2, mode='bilinear'):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode,
                                     align_corners=False if mode != 'nearest' else None)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              padding=kernel_size // 2)
    
    def forward(self, x):
        x = self.upsample(x)
        return self.conv(x)

# 비교
x = torch.randn(1, 64, 16, 16)

# 전치 합성곱
conv_t = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)

# 크기 조정 뒤 합성곱 (흠 없음)
resize_conv = UpsampleConv(64, 32, scale_factor=2)

print(f"ConvTranspose: {conv_t(x).shape}")     # [1, 32, 32, 32]
print(f"Resize+Conv:   {resize_conv(x).shape}") # [1, 32, 32, 32]
```

### 해법 3: 부화소 합성곱 (PixelShuffle)

초해상도에서 쓰는 방법으로, 채널을 공간 차원으로 다시 늘어놓는다.

```python
class SubPixelUpsample(nn.Module):
    """
    효율적인 상향 표본화를 위한 부화소 합성곱(PixelShuffle).
    
    보통 합성곱으로 채널을 r²개 만든 뒤 다시 늘어놓아
    공간 해상도를 r배로 키운다.
    """
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * upscale_factor**2, 
                              kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(upscale_factor)
    
    def forward(self, x):
        x = self.conv(x)
        return self.shuffle(x)

# 예
sub_pixel = SubPixelUpsample(64, 32, upscale_factor=2)
x = torch.randn(1, 64, 16, 16)
out = sub_pixel(x)
print(f"Sub-pixel: {x.shape} → {out.shape}")  # [1, 64, 16, 16] → [1, 32, 32, 32]
```

---

## 부호기-복호기 구조

### 단순한 자기부호기

```python
import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    """
    복호에 전치 합성곱을 쓰는 합성곱 자기부호기.
    """
    def __init__(self):
        super().__init__()
        
        # 부호기: 차츰 하향 표본화
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),    # 224 → 112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),   # 112 → 56
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 56 → 28
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # 복호기: 차츰 상향 표본화
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 28 → 56
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 56 → 112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),     # 112 → 224
            nn.Sigmoid(),  # 출력은 [0, 1]
        )
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

model = ConvAutoencoder()
x = torch.randn(2, 3, 224, 224)
reconstruction = model(x)
print(f"Input: {x.shape}")
print(f"Reconstruction: {reconstruction.shape}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### U-Net 방식 (건너뛰기 연결이 있는)

```python
class UNetDecoder(nn.Module):
    """
    부호기에서 오는 건너뛰기 연결이 있는 U-Net 복호기 블록.
    
    특징 맵을 상향 표본화하여 짝이 되는 부호기 특징과 이어 붙인 뒤
    합성곱을 적용한다.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        
        # 상향 표본화 (전치 합성곱 또는 크기 조정 뒤 합성곱)
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                      kernel_size=4, stride=2, padding=1)
        
        # 이어 붙인 뒤: (in_channels//2 + skip_channels) → out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x, skip):
        x = self.up(x)
        
        # 크기가 어긋나는 경우 처리 (차원이 홀수일 때 생길 수 있다)
        if x.shape != skip.shape:
            x = nn.functional.interpolate(x, size=skip.shape[2:])
        
        x = torch.cat([x, skip], dim=1)  # 채널을 따라 이어 붙이기
        return self.conv(x)
```

---

## 상향 표본화 방법 견주기

| 방법 | 학습 가능 | 흠 | 매개변수 | 속도 |
|--------|-----------|-----------|------------|-------|
| **ConvTranspose2d** | 그렇다 | 바둑판 무늬 (K % s ≠ 0일 때) | $C_{in} \times C_{out} \times K^2$ | 빠름 |
| **쌍선형 보간 뒤 합성곱** | 일부 | 깨끗함 | $C_{in} \times C_{out} \times K^2$ | 보통 |
| **최근접 보간 뒤 합성곱** | 일부 | 네모진 무늬 | $C_{in} \times C_{out} \times K^2$ | 보통 |
| **PixelShuffle** | 그렇다 | 깨끗함 | $C_{in} \times C_{out} \times r^2 \times K^2$ | 빠름 |
| **쌍선형 보간만** | 아니다 | 매끈함 (흐릿함) | 0 | 매우 빠름 |

```python
import torch
import torch.nn as nn

# 모든 방법: 채널 64개, 16×16 → 32×32

x = torch.randn(1, 64, 16, 16)

methods = {
    'ConvTranspose (K=4, s=2)': nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
    'ConvTranspose (K=2, s=2)': nn.ConvTranspose2d(64, 32, 2, stride=2),
    'PixelShuffle': nn.Sequential(
        nn.Conv2d(64, 32 * 4, 3, padding=1),
        nn.PixelShuffle(2)
    ),
    'Bilinear + Conv': nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(64, 32, 3, padding=1)
    ),
}

for name, module in methods.items():
    out = module(x)
    params = sum(p.numel() for p in module.parameters())
    print(f"{name:<30}: {x.shape} → {out.shape}, params: {params:,}")
```

---

## 1차원 전치 합성곱

전치 합성곱은 시간 방향 상향 표본화를 위해 1차원에서도 쓸 수 있다.

```python
# 시간 방향 상향 표본화를 위한 1차원 전치 합성곱
conv_t1d = nn.ConvTranspose1d(
    in_channels=64,
    out_channels=32,
    kernel_size=4,
    stride=2,
    padding=1
)

x = torch.randn(1, 64, 50)  # 시각 50개
out = conv_t1d(x)
print(f"1D ConvTranspose: {x.shape} → {out.shape}")  # [1, 32, 100]
```

---

## 요약

| 항목 | 설명 |
|--------|-------------|
| **연산** | 합성곱 행렬의 전치로, 낮은 해상도를 높은 해상도로 보낸다 |
| **합성곱과의 관계** | 입력에 대한 conv2d의 기울기 |
| **출력 크기** | $(H_{in}-1) \times s - 2p + d(K-1) + p_{out} + 1$ |
| **흔한 쓰임** | 복호기 신경망, GAN, 분할, 초해상도 |
| **주된 함정** | $K \% s \neq 0$일 때의 바둑판 무늬 흠 |
| **모범 관행** | $s$으로 나누어떨어지는 $K$을 쓰거나 크기 조정 뒤 합성곱을 쓴다 |

## 핵심 정리

1. **전치 합성곱은 합성곱 행렬의 역행렬이 아니라 전치**이다. 합성곱을 되돌리지 않는다
2. 역전파에서 입력에 대한 보통 합성곱의 **기울기로 자연스럽게 나타난다**
3. **바둑판 무늬 흠**은 핵 크기가 보폭으로 나누어떨어지지 않아 겹침이 고르지 않을 때 생긴다. $K = 2s$이나 $K = s$을 쓰면 피할 수 있다
4. 흠 없는 상향 표본화에는 **크기 조정 뒤 합성곱**(쌍선형 보간 뒤 보통 합성곱)이 나을 때가 많다
5. **PixelShuffle**(부화소 합성곱)은 채널을 다시 늘어놓아 효율적이고 흠 없는 상향 표본화를 준다
6. **output_padding**은 보통 합성곱에서 여러 입력 크기가 같은 출력 크기로 갈 때 생기는 모호함을 없앤다

## 참고 문헌

1. Dumoulin, V., & Visin, F. (2016). "A guide to convolution arithmetic for deep learning." *arXiv preprint arXiv:1603.07285*.

2. Long, J., Shelhamer, E., & Darrell, T. (2015). "Fully Convolutional Networks for Semantic Segmentation." *CVPR*.

3. Odena, A., Dumoulin, V., & Olah, C. (2016). "Deconvolution and Checkerboard Artifacts." *Distill*. https://distill.pub/2016/deconv-checkerboard/

4. Shi, W., et al. (2016). "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network." *CVPR*.

5. Ronneberger, O., Fischer, P., & Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." *MICCAI*.

## 연습문제

**연습문제 1.**
전치 합성곱의 출력 크기 공식 $o = (i-1) \times s - 2p + k + \text{output\_padding}$을 유도하라.

??? success "연습문제 1 풀이"
    전치 합성곱은 보통 합성곱의 공간 변환을 되짚는다. 보통 합성곱이 크기 $i$을 $o = (i+2p-k)/s + 1$으로 보낸다면 전치 합성곱은 $o$을 다시 $i$으로 보낸다. $i$에 대해 풀면 $i = (o-1)s - 2p + k$이다. 출력 덧대기는 순방향 합성곱에서 여러 입력 크기가 같은 출력으로 갈 때의 모호함을 처리한다.

---

**연습문제 2.**
전치 합성곱이 바둑판 무늬 흠을 만들 수 있는 까닭과 그것을 피하는 방법을 설명하라.

??? success "연습문제 2 풀이"
    보폭이 1보다 크면 전치 합성곱이 보폭만큼 떨어진 자리에 값을 놓고 틈을 메우므로 겹침이 고르지 않은 무늬(바둑판)가 생긴다. 해법: (1) 대신 최근접 이웃 상향 표본화 뒤 보통 합성곱을 쓴다, (2) 핵 크기가 보폭으로 나누어떨어지게 한다, (3) 쌍선형 상향 표본화 뒤 합성곱을 쓴다.

---

**연습문제 3.**
(가) 전치 합성곱과 (나) 쌍선형 보간 뒤 합성곱으로 상향 표본화 모듈을 각각 구현하고 결과를 견주어라.

??? success "연습문제 3 풀이"
    ```python
    # 방법 (가)
    up_a = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
    # 방법 (나)
    up_b = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(64, 32, kernel_size=3, padding=1)
    )
    ```

---

**연습문제 4.**
전치 합성곱은 어떤 구조에서 흔히 쓰이는가?

??? success "연습문제 4 풀이"
    복호기 신경망에서 쓰인다. U-Net(분할), 자기부호기, GAN(생성기의 상향 표본화), 초해상도 신경망이 그 예이다. 고정된 상향 표본화 방법의 학습 가능한 짝으로, 신경망이 알맞은 상향 표본화 필터를 배우게 해 준다.
