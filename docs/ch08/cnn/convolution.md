# 합성곱 연산

합성곱 신경망(CNN)이라는 이름은 함수 둘을 엮어 셋째 함수를 만드는 수학 연산인 **합성곱**에서 왔다. 딥러닝에서 합성곱은 신경망이 입력 데이터로부터 특징의 공간적 위계를 스스로 배우게 해 주며, 그 덕분에 이미지 인식과 컴퓨터 비전, 신호 처리 과제에서 아주 강력하다.

이 절은 이산 합성곱과 상호상관을 엄밀하게 다루어, CNN이 이미지에서 특징을 뽑아내는 방식을 이해하는 데 필요한 이론적 바탕을 세운다.

> **용어에 대한 중요한 참고**: 딥러닝 문헌과 PyTorch 같은 프레임워크에서 "합성곱"이라 부르는 것은 엄밀히 말하면 **상호상관**이다. 핵이 학습되는 매개변수이므로 뒤집기는 문제가 되지 않는다. 신경망은 핵이든 그것을 뒤집은 것이든 배울 수 있다. 이 책에서도 그 관행을 따라, 따로 말하지 않는 한 "합성곱"으로 상호상관을 가리킨다.

---

## 1. 수학적 바탕

### 연속 합성곱

이산 합성곱을 보기에 앞서 연속인 경우를 짧게 되짚는다. 연속 함수 $f$과 $g$에 대해 합성곱 $(f * g)$을 다음과 같이 정의한다.

$$(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) \cdot g(t - \tau) \, d\tau$$

이 연산은 함수 $g$을 함수 $f$ 위로 "미끄러뜨리며" 자리마다 두 함수의 점별 곱의 적분을 계산한다.

### 이산 합성곱

이산 순차열에서는 적분이 합으로 바뀐다. 입력 신호 $x[n]$과 핵(필터) $h[m]$이 주어질 때 **이산 합성곱**은 다음과 같다.

$$(x * h)[n] = \sum_{m=-\infty}^{\infty} x[m] \cdot h[n - m]$$

핵심은 핵 $h$을 입력 위로 미끄러뜨리기 전에 **뒤집는다**(거꾸로 놓는다)는 점이다. 이 뒤집기가 참된 합성곱과 상호상관을 가른다.

### 상호상관

**상호상관**은 합성곱과 비슷하되 핵을 **뒤집지 않는다**.

$$(x \star h)[n] = \sum_{m=-\infty}^{\infty} x[m] \cdot h[n + m]$$

또는 같은 말로, 크기가 $k$인 유한한 핵에 대해 다음과 같다.

$$y[i] = \sum_{j=0}^{k-1} x[i + j] \cdot h[j]$$

CNN이 실제로 계산하는 것이 바로 이것이다.

### 비교표

| 성질 | 참된 합성곱 | 상호상관 (CNN) |
|----------|------------------|-------------------------|
| 핵을 뒤집는가 | 그렇다 (180°) | 아니다 |
| 수학 기호 | $f * g$ | $f \star g$ |
| 교환법칙 | 성립: $f * g = g * f$ | 성립하지 않음 |
| 결합법칙 | 성립 | 성립하지 않음 |
| 신호 처리 | 표준 정의 | 유사도 척도 |
| 딥러닝 | 거의 쓰지 않음 | 표준 |

### CNN이 상호상관을 쓰는 까닭

1. **학습되는 핵**: 핵은 학습 중에 배우는 것이므로 최적인 핵을 뒤집은 것도 어차피 배우게 된다
2. **계산의 간결함**: 상호상관은 뒤집는 연산을 따로 하지 않아도 된다
3. **같은 결과**: (고전적인 이미지 처리에서 흔한) 대칭인 핵에서는 합성곱과 상호상관이 똑같다

---

## 2. 이미지를 위한 2차원 합성곱

### 수학적 정의

크기가 $H \times W$인 2차원 입력 이미지 $I$과 크기가 $M \times N$인 핵 $K$에 대해, 위치 $(i, j)$에서의 이산 2차원 합성곱(상호상관)은 다음과 같다.

$$(I * K)[i, j] = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} I[i + m, j + n] \cdot K[m, n]$$

### 출력의 차원

크기가 $H \times W$인 입력과 크기가 $K \times K$인 핵에 대해 (덧대기 없이) 출력의 크기는 다음과 같다.

$$H_{out} = H - K + 1$$

$$W_{out} = W - K + 1$$

공간 차원이 이렇게 줄어드는 것은 출력마다 둘레의 화소가 있어야 계산되는 핵의 성질에서 곧바로 따라 나온다.

### 시각적 해석

합성곱 연산은 다음과 같이 그려 볼 수 있다.

1. 핵을 이미지의 왼쪽 위 모서리에 놓는다
2. 겹치는 원소끼리 성분별 곱을 계산한다
3. 모든 곱을 더하여 출력 값 하나를 얻는다
4. 핵을 다음 자리로 미끄러뜨리고 되풀이한다

```
Input Image (5×5)          Kernel (3×3)
┌───┬───┬───┬───┬───┐     ┌───┬───┬───┐
│ 1 │ 2 │ 3 │ 0 │ 1 │     │ 1 │ 0 │-1 │
├───┼───┼───┼───┼───┤     ├───┼───┼───┤
│ 0 │ 1 │ 2 │ 3 │ 1 │     │ 1 │ 0 │-1 │
├───┼───┼───┼───┼───┤     ├───┼───┼───┤
│ 1 │ 2 │ 1 │ 0 │ 0 │     │ 1 │ 0 │-1 │
├───┼───┼───┼───┼───┤     └───┴───┴───┘
│ 0 │ 0 │ 1 │ 2 │ 1 │
├───┼───┼───┼───┼───┤
│ 1 │ 1 │ 0 │ 1 │ 0 │
└───┴───┴───┴───┴───┘

Output at position (0,0):
= 1×1 + 2×0 + 3×(-1) + 0×1 + 1×0 + 2×(-1) + 1×1 + 2×0 + 1×(-1)
= 1 + 0 - 3 + 0 + 0 - 2 + 1 + 0 - 1
= -4
```

### PyTorch 구현

```python
import torch
import torch.nn as nn

# 2차원 합성곱 예제
# 입력: (배치 크기, 채널, 높이, 너비)
x = torch.arange(1, 17, dtype=torch.float32).reshape(1, 1, 4, 4)
print("Input:")
print(x.squeeze())

# 2차원 합성곱 층 만들기
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=False)

# 소벨 비슷한 모서리 검출 핵
with torch.no_grad():
    kernel = torch.tensor([[[[1., 0., -1.],
                             [2., 0., -2.],
                             [1., 0., -1.]]]])
    conv2d.weight = nn.Parameter(kernel)

output = conv2d(x)
print(f"\nOutput shape: {output.shape}")  # torch.Size([1, 1, 2, 2])
print("Output:")
print(output.squeeze())
```

**출력:**

```
Input:
tensor([[ 1.,  2.,  3.,  4.],
        [ 5.,  6.,  7.,  8.],
        [ 9., 10., 11., 12.],
        [13., 14., 15., 16.]])

Output shape: torch.Size([1, 1, 2, 2])
Output:
tensor([[-8., -8.],
        [-8., -8.]], grad_fn=<SqueezeBackward0>)
```

---

## 3. 다채널 합성곱

### RGB 이미지와 특징 맵

실제 이미지에는 채널이 여럿 있다(예를 들어 RGB는 채널이 3개이다). CNN에서 중간 층은 채널이 많은 **특징 맵**을 내놓는다. 다채널 합성곱이 이를 자연스럽게 다룬다.

### 수식으로 나타내기

채널이 $C_{in}$개인 입력과 채널이 $C_{in}$개인 핵에 대해 위치 $(i, j)$의 출력은 다음과 같다.

$$(I * K)[i, j] = \sum_{c=0}^{C_{in}-1} \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} I[c, i + m, j + n] \cdot K[c, m, n]$$

이제 핵은 3차원이다: $K \in \mathbb{R}^{C_{in} \times M \times N}$.

### 여러 개의 출력 채널

출력 채널을 $C_{out}$개 만들려면 서로 다른 핵 $C_{out}$개를 쓰며, 핵마다 출력 채널 하나를 낸다.

$$Y[o, i, j] = \sum_{c=0}^{C_{in}-1} \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} X[c, i + m, j + n] \cdot K[o, c, m, n] + b[o]$$

여기서 각 기호는 다음과 같다.

- $K \in \mathbb{R}^{C_{out} \times C_{in} \times M \times N}$은 전체 핵 텐서이다
- $b \in \mathbb{R}^{C_{out}}$은 편향 벡터이다

### 매개변수의 수

합성곱 층의 매개변수 수는 다음과 같다.

$$\text{Parameters} = C_{out} \times C_{in} \times K_H \times K_W + C_{out}$$

여기서 마지막 항이 편향을 나타낸다.

**예:** 입력 채널이 32개, 출력 채널이 64개이고 핵이 $3 \times 3$인 층은 다음과 같다.

$$64 \times 32 \times 3 \times 3 + 64 = 18{,}496 \text{개의 매개변수}$$

### PyTorch 구현

```python
import torch
import torch.nn as nn

# 다채널 합성곱
# RGB 이미지: 채널 3개, 32x32
batch_size = 4
x = torch.randn(batch_size, 3, 32, 32)

# 합성곱 층: 입력 채널 3개 → 출력 채널 64개
conv = nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    padding=1,  # 같은 크기 덧대기
    bias=True
)

output = conv(x)
print(f"Input shape: {x.shape}")       # torch.Size([4, 3, 32, 32])
print(f"Output shape: {output.shape}") # torch.Size([4, 64, 32, 32])

# 매개변수 개수
num_params = sum(p.numel() for p in conv.parameters())
print(f"Parameters: {num_params}")     # 64 × 3 × 3 × 3 + 64 = 1,792
```

**출력:**

```
Input shape: torch.Size([4, 3, 32, 32])
Output shape: torch.Size([4, 64, 32, 32])
Parameters: 1792
```

---

## 4. 합성곱의 성질

### 평행 이동 동변성

합성곱의 근본적인 성질 하나가 **평행 이동 동변성**이다. 입력을 옮기면 출력도 그만큼 옮겨진다.

엄밀히 말해 $T_{\Delta}$을 평행 이동 연산자라 하면 다음이 성립한다.

$$T_{\Delta}(f * g) = (T_{\Delta} f) * g = f * (T_{\Delta} g)$$

이 성질은 CNN에 매우 중요하다. 특징 검출기(핵)는 그 특징이 이미지의 어디에 나타나든 똑같이 잡아낸다.

```python
import torch
import torch.nn.functional as F

# 평행 이동 동변성 보이기
kernel = torch.randn(1, 1, 3, 3)

# 원래 이미지
img = torch.zeros(1, 1, 10, 10)
img[0, 0, 2:5, 2:5] = 1.0  # (2,2) 자리의 정사각형

# 평행 이동한 이미지
img_shifted = torch.zeros(1, 1, 10, 10)
img_shifted[0, 0, 4:7, 4:7] = 1.0  # (4,4) 자리의 같은 정사각형

# 합성곱 적용
out1 = F.conv2d(img, kernel, padding=1)
out2 = F.conv2d(img_shifted, kernel, padding=1)

# 두 출력은 서로 옮겨진 것이다
# (경계 효과는 빼고)
```

### 지역성 (지역 수용 영역)

출력 값 하나하나는 핵의 크기가 정하는 입력의 **지역 영역**에만 기댄다. 이는 이미지를 이해하는 데 모서리나 질감 같은 지역적인 무늬가 중요하다는 귀납 편향을 심어 준다.

### 매개변수 공유

같은 핵을 모든 공간 위치에 적용한다.

- **메모리 효율**: 입력 전체에 같은 매개변수를 쓴다
- **통계적 효율**: 모든 위치에서 한꺼번에 배운다
- **과적합 감소**: 완전 연결층보다 매개변수가 적다

### 성긴 연결

출력마다 모든 입력에 기대는 완전 연결층과 달리 합성곱 층은 **성긴 연결**을 갖는다. 출력마다 입력의 작은 부분집합에만 기댄다.

---

## 5. 흔히 쓰는 핵과 그 효과

### 모서리 검출 핵

**소벨 연산자**는 가로와 세로 모서리를 잡는다.

$$G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad
G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$

**라플라스** (이계 도함수로 모든 방향의 모서리를 잡는다):

$$K_{laplacian} = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 0 \end{bmatrix}$$

### 그 밖에 흔히 쓰는 핵

**날카롭게 하기**:

$$K_{sharpen} = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 5 & -1 \\ 0 & -1 & 0 \end{bmatrix}$$

**가우스 흐리기**:

$$K_{blur} = \frac{1}{16} \begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix}$$

**항등** (바뀌지 않음):

$$K_{identity} = \begin{bmatrix} 0 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{bmatrix}$$

### CNN이 배우는 것

학습된 CNN에서 핵은 사람이 설계한 것이 아니라 역전파로 **배운** 것이다. 신경망이 과제에 알맞은 핵을 스스로 찾아낸다.

- **앞쪽 층**: 소벨 연산자와 비슷한 모서리 검출기, 색 검출기, 질감 검출기
- **중간 층**: 부분 검출기 (눈, 바퀴, 창문)
- **깊은 층**: 물체 검출기, 복잡한 무늬 인식기

---

## 6. 행렬 곱으로 본 합성곱

창을 미끄러뜨리는 관점은 직관적이지만, 합성곱은 **행렬 곱**으로도 나타낼 수 있고 GPU가 실제로 그렇게 계산한다.

### 퇴플리츠 행렬로 나타내기

1차원 입력 $\mathbf{x} = [x_0, x_1, x_2, x_3, x_4]^\top$과 핵 $\mathbf{k} = [k_0, k_1, k_2]^\top$에 대해 합성곱 $\mathbf{y} = \mathbf{T}\mathbf{x}$은 다음 퇴플리츠 행렬을 쓴다.

$$\mathbf{T} = \begin{bmatrix}
k_0 & k_1 & k_2 & 0 & 0 \\
0 & k_0 & k_1 & k_2 & 0 \\
0 & 0 & k_0 & k_1 & k_2
\end{bmatrix}$$

### im2col: 실제 구현

실제로 2차원 합성곱은 **im2col**(이미지를 열로) 변환으로 구현한다. 입력 조각들을 행렬로 다시 늘어놓아 합성곱이 행렬 곱 하나가 되게 한다.

1. $K \times K$ 입력 조각을 하나씩 뽑아 열로 펼친다
2. 모든 조각을 쌓아 행렬 $\mathbf{X}_{col} \in \mathbb{R}^{(C_{in} \cdot K^2) \times (H_{out} \cdot W_{out})}$을 만든다
3. 핵의 모양을 $\mathbf{W}_{row} \in \mathbb{R}^{C_{out} \times (C_{in} \cdot K^2)}$으로 바꾼다
4. $\mathbf{Y} = \mathbf{W}_{row} \cdot \mathbf{X}_{col}$을 계산한다

```python
import torch
import torch.nn.functional as F

def conv2d_via_im2col(x, weight, bias=None, stride=1, padding=0):
    """
    im2col과 GEMM으로 구현한 2차원 합성곱.
    
    인수:
        x: 입력 텐서 (N, C_in, H, W)
        weight: 핵 텐서 (C_out, C_in, kH, kW)
        bias: 선택적인 편향 (C_out,)
    """
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = weight.shape
    
    # 덧대기 적용
    if padding > 0:
        x = F.pad(x, [padding]*4)
        _, _, H, W = x.shape
    
    H_out = (H - kH) // stride + 1
    W_out = (W - kW) // stride + 1
    
    # im2col: 조각 뽑기
    # unfold는 한 차원을 따라 미끄러지는 창을 뽑는다
    cols = x.unfold(2, kH, stride).unfold(3, kW, stride)  # (N, C_in, H_out, W_out, kH, kW)
    cols = cols.contiguous().view(N, C_in * kH * kW, H_out * W_out)  # (N, C_in*kH*kW, L)
    
    # 가중치 모양 바꾸기: (C_out, C_in*kH*kW)
    W_row = weight.view(C_out, -1)
    
    # 행렬 곱: (C_out, C_in*kH*kW) × (N, C_in*kH*kW, L) → (N, C_out, L)
    out = torch.bmm(W_row.unsqueeze(0).expand(N, -1, -1), cols)
    
    # 공간 모양으로 되돌리기
    out = out.view(N, C_out, H_out, W_out)
    
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    
    return out

# PyTorch와 견주어 확인
x = torch.randn(2, 3, 8, 8)
w = torch.randn(16, 3, 3, 3)
b = torch.randn(16)

out_custom = conv2d_via_im2col(x, w, b, padding=1)
out_pytorch = F.conv2d(x, w, b, padding=1)

print(f"Max difference: {(out_custom - out_pytorch).abs().max().item():.2e}")
```

이 행렬 곱 관점은 입력에 대한 합성곱의 **역전파**가 왜 전치 합성곱인지도 밝혀 준다. 그것은 $\mathbf{T}^\top$을 곱하는 것에 해당한다.

---

## 7. 구현 예제

### 손수 만드는 2차원 합성곱

```python
import torch
import torch.nn as nn

def manual_conv2d(input_tensor, kernel, bias=None):
    """
    배움을 위해 2차원 합성곱을 손수 구현한다.
    
    인수:
        input_tensor: (배치, in_channels, H, W)
        kernel: (out_channels, in_channels, kH, kW)
        bias: (out_channels,) 또는 None
    
    반환값:
        output: (배치, out_channels, H_out, W_out)
    """
    batch_size, in_channels, H, W = input_tensor.shape
    out_channels, _, kH, kW = kernel.shape
    
    # 출력 차원 계산
    H_out = H - kH + 1
    W_out = W - kW + 1
    
    # 출력 초기화
    output = torch.zeros(batch_size, out_channels, H_out, W_out)
    
    # 합성곱 수행
    for b in range(batch_size):           # 배치의 표본마다
        for oc in range(out_channels):    # 출력 채널마다
            for i in range(H_out):        # 출력 행마다
                for j in range(W_out):    # 출력 열마다
                    # 수용 영역 뽑기
                    receptive_field = input_tensor[b, :, i:i+kH, j:j+kW]
                    # 성분별 곱한 뒤 더하기
                    output[b, oc, i, j] = (receptive_field * kernel[oc]).sum()
                    # 편향이 있으면 더하기
                    if bias is not None:
                        output[b, oc, i, j] += bias[oc]
    
    return output

# 손수 만든 구현 시험
x = torch.randn(2, 3, 8, 8)
kernel = torch.randn(16, 3, 3, 3)
bias = torch.randn(16)

output_manual = manual_conv2d(x, kernel, bias)

# PyTorch와 견주어 확인
conv = nn.Conv2d(3, 16, 3, bias=True)
conv.weight.data = kernel
conv.bias.data = bias
output_pytorch = conv(x)

print(f"Max difference: {(output_manual - output_pytorch).abs().max().item():.2e}")
# 아주 작아야 한다 (수치 정밀도)
```

### 모서리 검출 시각화

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def create_test_image():
    """모서리가 뚜렷한 간단한 이미지를 만든다."""
    img = np.zeros((64, 64), dtype=np.float32)
    img[16:48, 16:48] = 1.0  # 검은 배경 위의 흰 정사각형
    return torch.tensor(img).unsqueeze(0).unsqueeze(0)

# 소벨 핵 정의
sobel_x = torch.tensor([[-1., 0., 1.],
                        [-2., 0., 2.],
                        [-1., 0., 1.]]).view(1, 1, 3, 3)

sobel_y = torch.tensor([[-1., -2., -1.],
                        [ 0.,  0.,  0.],
                        [ 1.,  2.,  1.]]).view(1, 1, 3, 3)

# 합성곱 적용
image = create_test_image()
edge_x = F.conv2d(image, sobel_x, padding=1)
edge_y = F.conv2d(image, sobel_y, padding=1)

# 모서리의 크기 계산
edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)

# 결과: edge_x는 세로 모서리를, edge_y는 가로 모서리를 잡는다
# edge_magnitude는 모든 모서리를 보인다
```

### 합성곱과 상호상관 견주기

```python
import torch
import torch.nn.functional as F

def true_convolution(x, kernel):
    """참된 수학적 합성곱을 한다 (핵을 뒤집는다)."""
    flipped_kernel = torch.flip(kernel, dims=[2, 3])
    return F.conv2d(x, flipped_kernel)

def cross_correlation(x, kernel):
    """상호상관을 한다 (PyTorch가 conv2d라 부르는 것)."""
    return F.conv2d(x, kernel)

# 차이를 보려고 비대칭 핵 사용
kernel = torch.tensor([[1., 2., 3.],
                       [4., 5., 6.],
                       [7., 8., 9.]]).view(1, 1, 3, 3)

x = torch.randn(1, 1, 5, 5)

conv_result = true_convolution(x, kernel)
xcorr_result = cross_correlation(x, kernel)

print("For asymmetric kernels, convolution ≠ cross-correlation")
print(f"Difference: {(conv_result - xcorr_result).abs().max().item():.4f}")

# 대칭인 핵에서는 둘이 똑같다
symmetric_kernel = torch.tensor([[1., 2., 1.],
                                 [2., 4., 2.],
                                 [1., 2., 1.]]).view(1, 1, 3, 3)

conv_symmetric = true_convolution(x, symmetric_kernel)
xcorr_symmetric = cross_correlation(x, symmetric_kernel)

print("\nFor symmetric kernels, convolution = cross-correlation")
print(f"Difference: {(conv_symmetric - xcorr_symmetric).abs().max().item():.2e}")
```

---

## 8. 계산 복잡도

### 직접 계산하는 합성곱

입력 크기가 $H \times W$이고 채널이 $C_{in}$개, 핵 크기가 $K \times K$, 출력 채널이 $C_{out}$개일 때 다음과 같다.

$$O(H \times W \times C_{in} \times C_{out} \times K^2)$$

### FFT 기반 합성곱

핵이 크면 FFT 기반 합성곱이 더 효율적일 수 있다.

$$O(H \times W \times \log(HW) \times C_{in} \times C_{out})$$

실제로 CNN에서 흔한 핵 크기($3 \times 3$, $5 \times 5$)에서는 메모리 접근 방식이 낫고 구현이 잘 다듬어져 있어(cuDNN, 위노그라드) 직접 계산하는 합성곱이 더 빠르다.

### 실제 성능

요즘 딥러닝 프레임워크는 여러 최적화를 쓴다.

- **im2col과 GEMM**: 합성곱을 행렬 곱으로 바꾸어 잘 다듬어진 BLAS 라이브러리를 쓴다
- **위노그라드 합성곱**: 작은 핵($3 \times 3$)에서 곱셈 횟수를 약 2.25배 줄인다
- **FFT 합성곱**: 핵 크기가 대략 11×11을 넘을 때 이롭다
- **cuDNN 자동 조정**: 주어진 텐서 모양에 가장 빠른 알고리즘을 저절로 고른다

---

## 9. 합성곱을 지나는 역전파

기울기 계산을 이해하는 일은 층을 직접 만들고 학습의 벌레를 잡는 데 꼭 필요하다.

**입력에 대한 기울기** (기울기를 뒤로 전파하기 위한 것):

$$\frac{\partial L}{\partial X_{c,i,j}} = \sum_{k,m,n} \frac{\partial L}{\partial Y_{k, i-m, j-n}} \cdot W_{k,c,m,n}$$

이는 출력 기울기와 **뒤집은** 가중치의 **온전한 합성곱**이며, **전치 합성곱**과 같다([전치 합성곱](transposed_conv.md) 참고).

**가중치에 대한 기울기** (매개변수를 갱신하기 위한 것):

$$\frac{\partial L}{\partial W_{k,c,m,n}} = \sum_{i,j} \frac{\partial L}{\partial Y_{k,i,j}} \cdot X_{c, i+m, j+n}$$

이는 입력과 출력 기울기 사이의 **상호상관**이다.

**편향에 대한 기울기**:

$$\frac{\partial L}{\partial b_k} = \sum_{i,j} \frac{\partial L}{\partial Y_{k,i,j}}$$

---

## 10. 핵심 정리

1. **CNN은 참된 합성곱이 아니라 상호상관을 쓴다.** 다만 두 말을 자주 섞어 쓴다
2. **핵이 입력 위를 미끄러지며** 자리마다 내적을 계산한다
3. **다채널 합성곱**은 모든 입력 채널의 몫을 더한다
4. **평행 이동 동변성** 덕분에 CNN이 물체의 위치에 흔들리지 않는다
5. **매개변수 공유**와 **지역 연결**이 합성곱을 효율적으로 만든다
6. **앞쪽 층**은 단순한 특징을, **깊은 층**은 복잡한 무늬를 잡는다
7. **합성곱은 퇴플리츠/im2col을 거친 행렬 곱**이며, 그 덕분에 GPU 가속이 가능하다

---

## 연습문제

**연습문제 1.**
단일 채널 입력과 핵에 대해 2차원 상호상관을 (라이브러리 함수 없이) 밑바닥부터 구현하라. `F.conv2d`과 견주어 확인하라.

??? success "연습문제 1 풀이"
    ```python
    import torch
    import torch.nn.functional as F

    def manual_conv2d(input_2d, kernel):
        H, W = input_2d.shape
        kH, kW = kernel.shape
        oH, oW = H - kH + 1, W - kW + 1
        output = torch.zeros(oH, oW)
        for i in range(oH):
            for j in range(oW):
                patch = input_2d[i:i+kH, j:j+kW]
                output[i, j] = (patch * kernel).sum()
        return output

    x = torch.randn(5, 5)
    k = torch.randn(3, 3)
    manual = manual_conv2d(x, k)
    pytorch = F.conv2d(x[None, None], k[None, None]).squeeze()
    assert torch.allclose(manual, pytorch, atol=1e-5)
    ```

---

**연습문제 2.**
입력 채널이 3개, 출력 채널이 64개이고 핵이 $3 \times 3$인 (편향이 있는) 신경망의 첫 합성곱 층에서 매개변수의 수를 계산하라.

??? success "연습문제 2 풀이"
    출력 채널마다 입력 채널당 $3 \times 3$ 핵 하나와 편향 하나가 있다.

    필터당 매개변수: $3 \times 3 \times 3 = 27$(가중치) $+ 1$(편향) $= 28$.

    모두 $64 \times 28 = 1{,}792$개의 매개변수이다.

    편향이 없으면 $64 \times 27 = 1{,}728$개이다.

---

**연습문제 3.**
합성곱이 평행 이동에 동변임을 보여라. 곧 $T_a$이 입력을 화소 $a$개만큼 옮길 때 $\text{conv}(T_a(x)) = T_a(\text{conv}(x))$임을 보여라.

??? success "연습문제 3 풀이"
    $(f * g)(t) = \sum_\tau f(\tau)g(t - \tau)$이라 하자. 이동 연산자를 $(T_a f)(t) = f(t - a)$으로 정의한다.

    $$
    (T_a f * g)(t) = \sum_\tau f(\tau - a)g(t-\tau)
    $$

    $\tau' = \tau - a$으로 바꾸어 놓으면 다음과 같다.

    $$
    = \sum_{\tau'} f(\tau')g(t - a - \tau') = (f * g)(t - a) = T_a(f * g)(t)
    $$

    따라서 $T_a(f) * g = T_a(f * g)$이며 평행 이동 동변성이 증명된다. $\square$

---

**연습문제 4.**
합성곱과 상호상관의 차이를 설명하라. 핵이 어떤 조건을 만족할 때 둘이 같아지는가?

??? success "연습문제 4 풀이"
    참된 합성곱은 미끄러뜨리기 전에 핵을 뒤집는다: $(f * g)(t) = \sum_\tau f(\tau)g(t - \tau)$. 상호상관은 뒤집지 않는다: $(f \star g)(t) = \sum_\tau f(\tau)g(t + \tau)$.

    핵이 대칭일 때, 곧 모든 $\tau$에 대해 $g(\tau) = g(-\tau)$일 때 둘이 같아진다. 가우스 핵과 항등 핵이 그 예이다.

    실제로 딥러닝 프레임워크는 상호상관을 쓰면서 그것을 "합성곱"이라 부른다. 핵을 학습하므로 이 구별은 중요하지 않다. 필요하면 신경망이 뒤집힌 것을 배운다.

## 정리하며

| 항목 | 설명 |
|--------|-------------|
| **연산** | 지역 영역에서 성분별로 곱한 뒤 더하기 |
| **CNN의 관행** | 상호상관을 쓴다 (핵을 뒤집지 않는다) |
| **다채널** | 모든 입력 채널에 대해 더하고, 출력 채널만큼 쌓는다 |
| **핵심 성질** | 평행 이동 동변성, 지역성, 매개변수 공유 |
| **매개변수** | $C_{out} \times C_{in} \times K^2 + C_{out}$ |
| **출력 크기** | 덧대기가 없으면 $(H - K + 1) \times (W - K + 1)$ |
| **구현** | 효율을 위해 GPU에서 im2col과 GEMM을 쓴다 |

**참고 문헌**

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). "Gradient-based learning applied to document recognition." *Proceedings of the IEEE*, 86(11), 2278-2324.

2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning." MIT Press. Chapter 9: Convolutional Networks.

3. Dumoulin, V., & Visin, F. (2016). "A guide to convolution arithmetic for deep learning." *arXiv preprint arXiv:1603.07285*.

4. Chellapilla, K., Puri, S., & Simard, P. (2006). "High Performance Convolutional Neural Networks for Document Processing." *Tenth International Workshop on Frontiers in Handwriting Recognition*.
