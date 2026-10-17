# 컨볼루션 연산

컨볼루션 뉴럴 넷트웍은 작은 커널을 여러 위치에 공유해 국소 피처를 계산한다. 이 절에서는 수학적 컨볼루션과 CNN에서 사용하는 상호상관의 차이, 다채널 계산과 행렬 표현, 역전파를 연결한다.

딥러닝 프레임워크의 컨볼루션은 일반적으로 커널을 뒤집지 않는 상호상관이다. 학습 가능한 커널에서는 뒤집기를 파라메터의 재배열로 흡수할 수 있지만, 수식이나 고정 필터를 비교할 때에는 두 정의를 구분해야 한다.

## 1. 수학적 기초

### 연속 컨볼루션

이산 컨볼루션을 보기에 앞서 연속인 경우를 짧게 되짚는다. 연속 함수 $f$과 $g$에 대해 컨볼루션 $(f * g)$을 다음과 같이 정의한다.

$$(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) \cdot g(t - \tau) \, d\tau$$

이 연산은 함수 $g$을 함수 $f$ 위로 "미끄러뜨리며" 자리마다 두 함수의 점별 곱의 적분을 계산한다.

### 이산 컨볼루션

이산 시퀀스에서는 적분이 합으로 바뀐다. 입력 신호 $x[n]$과 커널(필터) $h[m]$이 주어질 때 **이산 컨볼루션**은 다음과 같다.

$$(x * h)[n] = \sum_{m=-\infty}^{\infty} x[m] \cdot h[n - m]$$

핵심은 커널 $h$을 입력 위로 미끄러뜨리기 전에 **뒤집는다**(거꾸로 놓는다)는 점이다. 이 뒤집기가 참된 컨볼루션과 상호상관을 가른다.

### 상호상관

**상호상관**은 컨볼루션과 비슷하되 커널을 **뒤집지 않는다**.

$$(x \star h)[n] = \sum_{m=-\infty}^{\infty} x[n + m] \cdot h[m]$$

또는 같은 말로, 크기가 $k$인 유한한 커널에 대해 다음과 같다.

$$y[i] = \sum_{j=0}^{k-1} x[i + j] \cdot h[j]$$

CNN이 실제로 계산하는 것이 바로 이것이다.

### 비교표

| 성질 | 참된 컨볼루션 | 상호상관 (CNN) |
|----------|------------------|-------------------------|
| 커널을 뒤집는가 | 그렇다 (180°) | 아니다 |
| 수학 기호 | $f * g$ | $f \star g$ |
| 교환법칙 | 성립: $f * g = g * f$ | 성립하지 않음 |
| 결합법칙 | 성립 | 성립하지 않음 |
| 신호 처리 | 표준 정의 | 유사도 척도 |
| 딥러닝 | 거의 쓰지 않음 | 표준 |

### CNN이 상호상관을 쓰는 까닭

1. **학습되는 커널**: 커널은 학습 중에 배우는 것이므로 최적인 커널을 뒤집은 것도 어차피 배우게 된다
2. **계산의 간결함**: 상호상관은 뒤집는 연산을 따로 하지 않아도 된다
3. **같은 결과**: (고전적인 이미지 처리에서 흔한) 대칭인 커널에서는 컨볼루션과 상호상관이 똑같다

---

## 2. 이미지를 위한 2차원 컨볼루션

### 수학적 정의

크기가 $H\times W$인 단일 채널 입력과 $M\times N$ 커널에 대해, 다음은 스트라이드 1·팽창률 1·패딩 없음인 상호상관이다. 수학적 컨볼루션 기호 $*$와 구분해 $\star$로 표시한다.

$$(I \star K)[i, j] = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} I[i + m, j + n] \cdot K[m, n]$$

### 출력의 차원

입력이 $H\times W$, 커널이 $K\times K$이고 스트라이드와 팽창률이 1이며 패딩이 없을 때 출력 크기는 다음과 같다.

$$H_{out} = H - K + 1$$

$$W_{out} = W - K + 1$$

공간 차원이 이렇게 줄어드는 것은 출력마다 둘레의 픽셀가 있어야 계산되는 커널의 성질에서 곧바로 따라 나온다.

### 시각적 해석

컨볼루션 연산은 다음과 같이 그려 볼 수 있다.

1. 커널을 이미지의 왼쪽 위 모서리에 놓는다
2. 겹치는 원소끼리 성분별 곱을 계산한다
3. 모든 곱을 더하여 출력 값 하나를 얻는다
4. 커널을 다음 자리로 미끄러뜨리고 되풀이한다

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

# 2차원 컨볼루션 예제
# 입력: (배치 크기, 채널, 높이, 너비)
x = torch.arange(1, 17, dtype=torch.float32).reshape(1, 1, 4, 4)
print("Input:")
print(x.squeeze())

# 2차원 컨볼루션 레이어 만들기
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=False)

# 소벨 비슷한 모서리 검출 커널
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

## 3. 다채널 컨볼루션

### RGB 이미지와 피처 맵

실제 이미지에는 채널이 여럿 있다(예를 들어 RGB는 채널이 3개이다). CNN에서 중간 레이어는 채널이 많은 **피처 맵**을 내놓는다. 다채널 컨볼루션이 이를 자연스럽게 다룬다.

### 수식으로 나타내기

채널이 $C_{in}$개인 입력과 채널이 $C_{in}$개인 커널에 대해 위치 $(i, j)$의 출력은 다음과 같다.

$$(I \star K)[i, j] = \sum_{c=0}^{C_{in}-1} \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} I[c, i + m, j + n] \cdot K[c, m, n]$$

이제 커널은 3차원이다: $K \in \mathbb{R}^{C_{in} \times M \times N}$.

### 여러 개의 출력 채널

출력 채널을 $C_{out}$개 만들려면 서로 다른 커널 $C_{out}$개를 쓰며, 커널마다 출력 채널 하나를 낸다.

$$Y[o, i, j] = \sum_{c=0}^{C_{in}-1} \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} X[c, i + m, j + n] \cdot K[o, c, m, n] + b[o]$$

여기서 각 기호는 다음과 같다.

- $K \in \mathbb{R}^{C_{out} \times C_{in} \times M \times N}$은 전체 커널 텐서이다
- $b \in \mathbb{R}^{C_{out}}$은 바이어스 벡터이다

### 파라메터의 수

컨볼루션 레이어의 파라메터 수는 다음과 같다.

$$\text{Parameters} = C_{out} \times C_{in} \times K_H \times K_W + C_{out}$$

여기서 마지막 항이 바이어스를 나타낸다.

**예:** 입력 채널이 32개, 출력 채널이 64개이고 커널이 $3 \times 3$인 레이어는 다음과 같다.

$$64 \times 32 \times 3 \times 3 + 64 = 18{,}496 \text{개의 파라메터}$$

### PyTorch 구현

```python
import torch
import torch.nn as nn

# 다채널 컨볼루션
# RGB 이미지: 채널 3개, 32x32
batch_size = 4
x = torch.randn(batch_size, 3, 32, 32)

# 컨볼루션 레이어: 입력 채널 3개 → 출력 채널 64개
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

# 파라메터 개수
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

## 4. 컨볼루션의 성질

### 평행 이동 등변성

무한 격자에서 스트라이드 1로 같은 커널을 공유하는 컨볼루션·상호상관은 평행 이동 등변성을 갖는다. 입력이 이동하면 출력도 그에 맞춰 이동한다. 유한 이미지의 경계 처리와 스트라이드에 따른 부분표집은 이 성질의 적용 범위를 제한한다.

엄밀히 말해 $T_{\Delta}$을 평행 이동 연산자라 하면 다음이 성립한다.

$$T_{\Delta}(f * g) = (T_{\Delta} f) * g = f * (T_{\Delta} g)$$

같은 검출기를 위치마다 공유하므로 동일한 국소 패턴을 같은 방식으로 처리한다. 출력이 이동하는 등변성과 출력이 바뀌지 않는 불변성은 다르다. CNN 전체가 물체 위치에 완전히 불변이라는 뜻은 아니다.

```python
import torch
import torch.nn.functional as F

# 평행 이동 동변성 보이기
torch.manual_seed(0)
kernel = torch.randn(1, 1, 3, 3)

# 원래 이미지
img = torch.zeros(1, 1, 10, 10)
img[0, 0, 2:5, 2:5] = 1.0  # (2,2) 자리의 정사각형

# 평행 이동한 이미지
img_shifted = torch.zeros(1, 1, 10, 10)
img_shifted[0, 0, 4:7, 4:7] = 1.0  # (4,4) 자리의 같은 정사각형

# 컨볼루션 적용
out1 = F.conv2d(img, kernel, padding=1)
out2 = F.conv2d(img_shifted, kernel, padding=1)

# 동변이라는 말은 "출력도 꼭 그만큼 옮겨진다"는 뜻이다. 말로 두지 말고 재 보자.
# out2를 두 칸씩 되돌려 out1과 겹쳐 본다. 가장자리는 덧대기 때문에 다르므로
# 두 쪽 모두 안쪽만 잘라 견준다.
shift = 2
a = out1[0, 0, :-shift, :-shift]      # 원래 것의 안쪽
b = out2[0, 0, shift:, shift:]        # 옮긴 것을 되돌린 안쪽

print(f"겹친 자리 크기   : {tuple(a.shape)}")
print(f"되돌린 뒤 최대 차이: {(a - b).abs().max().item():.2e}")
print(f"출력이 통째로 같은가: {torch.allclose(out1, out2)}")
```

**출력:**

```
겹친 자리 크기   : (8, 8)
되돌린 뒤 최대 차이: 0.00e+00
출력이 통째로 같은가: False
```

출력을 그대로 비교한 값과 이동량을 보정해 비교한 값을 구분한다. 이동이 출력의 위치를 바꾸는 등변성을 보여 주는 예이며, 여기서 관찰한 결과를 모든 패딩·스트라이드와 경계에 그대로 적용하지 않는다.

### 지역성 (지역 리셉티브 필드)

출력 값 하나하나는 커널의 크기가 정하는 입력의 **국소 영역**에만 기댄다. 이는 이미지를 이해하는 데 모서리나 질감 같은 지역적인 무늬가 중요하다는 귀납 바이어스를 심어 준다.

### 파라메터 공유

같은 커널을 모든 공간 위치에 적용한다.

- **메모리 효율**: 입력 전체에 같은 파라메터를 쓴다
- **통계적 효율**: 모든 위치에서 한꺼번에 배운다
- **파라메터 제약**: 같은 입력·출력 격자를 연결하는 완전 연결 레이어보다 적은 계수를 사용하는 경우가 많다. 이것만으로 일반화 성능을 보장하지는 않는다

### 희소 연결

출력마다 모든 입력에 기대는 완전 연결 레이어와 달리 컨볼루션 레이어는 **희소 연결**을 갖는다. 출력마다 입력의 작은 부분집합에만 기댄다.

---

## 5. 흔히 쓰는 커널과 그 효과

### 모서리 검출 커널

**소벨 연산자**는 가로와 세로 모서리를 잡는다.

$$G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad
G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$

**라플라스** (이계 도함수로 모든 방향의 모서리를 잡는다):

$$K_{laplacian} = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 0 \end{bmatrix}$$

### 그 밖에 흔히 쓰는 커널

**날카롭게 하기**:

$$K_{sharpen} = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 5 & -1 \\ 0 & -1 & 0 \end{bmatrix}$$

**가우스 흐리기**:

$$K_{blur} = \frac{1}{16} \begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix}$$

**항등** (바뀌지 않음):

$$K_{identity} = \begin{bmatrix} 0 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{bmatrix}$$

### CNN이 배우는 것

학습되는 커널은 로스의 그래디언트를 통해 갱신된다. 앞쪽 레이어의 국소 피처와 뒤쪽 레이어의 조합 피처가라는 해석은 유용하지만, 모든 채널이 사람이 이름 붙일 수 있는 독립적인 검출기라는 뜻은 아니다.

- **앞쪽 레이어**: 소벨 연산자와 비슷한 모서리 검출기, 색 검출기, 질감 검출기
- **중간 레이어**: 부분 검출기 (눈, 바퀴, 창문)
- **깊은 레이어**: 물체 검출기, 복잡한 무늬 인식기

---

## 6. 행렬 곱으로 본 컨볼루션

컨볼루션은 입력 패치와 커널의 행렬 곱으로 표현할 수 있다. 이는 유용한 계산 관점이지만 모든 GPU 구현이 거대한 패치 행렬을 실제로 생성한다는 뜻은 아니다.

### 퇴플리츠 행렬로 나타내기

1차원 입력 $\mathbf{x} = [x_0, x_1, x_2, x_3, x_4]^\top$과 커널 $\mathbf{k} = [k_0, k_1, k_2]^\top$에 대해 컨볼루션 $\mathbf{y} = \mathbf{T}\mathbf{x}$은 다음 퇴플리츠 행렬을 쓴다.

$$\mathbf{T} = \begin{bmatrix}
k_0 & k_1 & k_2 & 0 & 0 \\
0 & k_0 & k_1 & k_2 & 0 \\
0 & 0 & k_0 & k_1 & k_2
\end{bmatrix}$$

### im2col: 실제 구현

im2col은 입력 패치를 열로 배열해 행렬 곱으로 바꾸는 구현 방식 중 하나이다. 실제 라이브러리는 모양·자료형·장치에 따라 패치를 명시적으로 만들지 않는 알고리즘이나 다른 변환을 선택할 수 있다.

1. $K \times K$ 입력 조각을 하나씩 뽑아 열로 펼친다
2. 모든 조각을 쌓아 행렬 $\mathbf{X}_{col} \in \mathbb{R}^{(C_{in} \cdot K^2) \times (H_{out} \cdot W_{out})}$을 만든다
3. 커널의 모양을 $\mathbf{W}_{row} \in \mathbb{R}^{C_{out} \times (C_{in} \cdot K^2)}$으로 바꾼다
4. $\mathbf{Y} = \mathbf{W}_{row} \cdot \mathbf{X}_{col}$을 계산한다

```python
import torch
import torch.nn.functional as F

def conv2d_via_im2col(x, weight, bias=None, stride=1, padding=0):
    """
    im2col과 GEMM으로 구현한 2차원 컨볼루션.
    
    인수:
        x: 입력 텐서 (N, C_in, H, W)
        weight: 커널 텐서 (C_out, C_in, kH, kW)
        bias: 선택적인 바이어스 (C_out,)
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

    # 차원 차례를 먼저 바꾸어야 한다. 위의 꼴은 자리(H_out, W_out)가 커널 안쪽
    # (kH, kW)보다 **앞**에 있는데, 우리가 필요한 것은 커널 한 조각이 한 열에
    # 모인 (C_in*kH*kW, L) 이다. 차례를 바꾸지 않고 view 로 접으면 같은 메모리를
    # 다른 뜻으로 읽어 조각이 뒤섞인다 — 그래도 모양은 맞으므로 조용히 틀린다.
    cols = cols.permute(0, 1, 4, 5, 2, 3).contiguous()    # (N, C_in, kH, kW, H_out, W_out)
    cols = cols.view(N, C_in * kH * kW, H_out * W_out)    # (N, C_in*kH*kW, L)
    
    # 웨이트 모양 바꾸기: (C_out, C_in*kH*kW)
    W_row = weight.view(C_out, -1)
    
    # 행렬 곱: (C_out, C_in*kH*kW) × (N, C_in*kH*kW, L) → (N, C_out, L)
    out = torch.bmm(W_row.unsqueeze(0).expand(N, -1, -1), cols)
    
    # 공간 모양으로 되돌리기
    out = out.view(N, C_out, H_out, W_out)
    
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    
    return out

# PyTorch와 견주어 확인
torch.manual_seed(0)
x = torch.randn(2, 3, 8, 8)
w = torch.randn(16, 3, 3, 3)
b = torch.randn(16)

out_custom = conv2d_via_im2col(x, w, b, padding=1)
out_pytorch = F.conv2d(x, w, b, padding=1)

print(f"Max difference: {(out_custom - out_pytorch).abs().max().item():.2e}")
```

**출력:**

```
Max difference: 2.38e-06
```

이 예제의 작은 차이는 행렬 표현과 라이브러리 결과가 수치적으로 가깝다는 점을 보여 준다. 차이가 크면 reshape·permute, 채널 순서와 바이어스 처리 등을 점검한다. 특정 오차 크기만으로 원인을 permute 누락 하나로 확정하지 않는다.

이 행렬 곱 관점은 입력에 대한 컨볼루션의 **역전파**가 왜 전치 컨볼루션인지도 밝혀 준다. 그것은 $\mathbf{T}^\top$을 곱하는 것에 해당한다.

---

## 7. 구현 예제

### 손수 만드는 2차원 컨볼루션

```python
import torch
import torch.nn as nn

def manual_conv2d(input_tensor, kernel, bias=None):
    """
    러닝을 위해 2차원 컨볼루션을 손수 구현한다.
    
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
    
    # 컨볼루션 수행
    for b in range(batch_size):           # 배치의 표본마다
        for oc in range(out_channels):    # 출력 채널마다
            for i in range(H_out):        # 출력 행마다
                for j in range(W_out):    # 출력 열마다
                    # 리셉티브 필드 뽑기
                    receptive_field = input_tensor[b, :, i:i+kH, j:j+kW]
                    # 성분별 곱한 뒤 더하기
                    output[b, oc, i, j] = (receptive_field * kernel[oc]).sum()
                    # 바이어스가 있으면 더하기
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

**출력:**

```
Max difference: 3.81e-06
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

# 소벨 커널 정의
sobel_x = torch.tensor([[-1., 0., 1.],
                        [-2., 0., 2.],
                        [-1., 0., 1.]]).view(1, 1, 3, 3)

sobel_y = torch.tensor([[-1., -2., -1.],
                        [ 0.,  0.,  0.],
                        [ 1.,  2.,  1.]]).view(1, 1, 3, 3)

# 컨볼루션 적용
image = create_test_image()
edge_x = F.conv2d(image, sobel_x, padding=1)
edge_y = F.conv2d(image, sobel_y, padding=1)

# 모서리의 크기 계산
edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)

# 어느 커널이 어느 모서리를 잡는지 자리를 집어 재 본다.
# 정사각형은 16~47 행·열에 있으므로 (32, 16)은 **세로** 모서리 위의 한 점이고
# (16, 32)는 **가로** 모서리 위의 한 점이다.
v = (32, 16)   # 세로 모서리
h = (16, 32)   # 가로 모서리

print(f"{'자리':<12} {'sobel_x':>9} {'sobel_y':>9}")
print(f"{'세로 모서리':<12} {edge_x[0,0][v]:>9.1f} {edge_y[0,0][v]:>9.1f}")
print(f"{'가로 모서리':<12} {edge_x[0,0][h]:>9.1f} {edge_y[0,0][h]:>9.1f}")
print(f"\n크기의 최댓값: {edge_magnitude.max().item():.1f}")
```

**출력:**

```
자리             sobel_x   sobel_y
세로 모서리             4.0       0.0
가로 모서리             0.0       4.0

크기의 최댓값: 4.2
```

커널마다 **자기 방향에서만** 4.0이 서고 다른 방향에서는 정확히 0이다. 두 커널이 서로 직교한 방향을 본다는 뜻이다.

크기의 최댓값 4.2는 모서리 한가운데가 아니라 **모퉁이**에서 나온다. 모퉁이에서는 두 커널이 함께 반응하되 어느 쪽도 온전한 4.0을 내지 못해 $\sqrt{3^2+3^2} \approx 4.24$가 된다. 모서리 한가운데의 4.0보다 조금 크다.

### 컨볼루션과 상호상관 비교

```python
import torch
import torch.nn.functional as F

def true_convolution(x, kernel):
    """참된 수학적 컨볼루션을 한다 (커널을 뒤집는다)."""
    flipped_kernel = torch.flip(kernel, dims=[2, 3])
    return F.conv2d(x, flipped_kernel)

def cross_correlation(x, kernel):
    """상호상관을 한다 (PyTorch가 conv2d라 부르는 것)."""
    return F.conv2d(x, kernel)

# 차이를 보려고 비대칭 커널 사용
torch.manual_seed(0)
kernel = torch.tensor([[1., 2., 3.],
                       [4., 5., 6.],
                       [7., 8., 9.]]).view(1, 1, 3, 3)

x = torch.randn(1, 1, 5, 5)

conv_result = true_convolution(x, kernel)
xcorr_result = cross_correlation(x, kernel)

print("For asymmetric kernels, convolution ≠ cross-correlation")
print(f"Difference: {(conv_result - xcorr_result).abs().max().item():.4f}")

# 대칭인 커널에서는 둘이 똑같다
symmetric_kernel = torch.tensor([[1., 2., 1.],
                                 [2., 4., 2.],
                                 [1., 2., 1.]]).view(1, 1, 3, 3)

conv_symmetric = true_convolution(x, symmetric_kernel)
xcorr_symmetric = cross_correlation(x, symmetric_kernel)

print("\nFor symmetric kernels, convolution = cross-correlation")
print(f"Difference: {(conv_symmetric - xcorr_symmetric).abs().max().item():.2e}")
```

**출력:**

```
For asymmetric kernels, convolution ≠ cross-correlation
Difference: 36.2866

For symmetric kernels, convolution = cross-correlation
Difference: 0.00e+00
```

---

## 8. 계산 복잡도

일반적인 밀집 컨볼루션의 직접 계산에서 곱셈·누적 수는

$$O(H_{out}W_{out}C_{in}C_{out}K_HK_W)$$

이다. 배치 크기가 N이면 N배가 되고, 그룹 컨볼루션은 groups에 따라 채널 연결 수가 줄어든다. FLOPs를 셀 때 곱셈과 덧셈을 각각 한 연산으로 보는지 명시해야 한다.

im2col과 GEMM, 암시적 GEMM, Winograd, FFT 등은 다른 계산·메모리 절충을 가진다. FFT는 큰 커널에서 유리할 수 있지만 변환과 메모리 비용을 포함해야 한다. 작은 커널에서의 Winograd 곱셈 감소 비율도 타일과 커널 구성에 따라 달라진다.

따라서 커널 크기 하나로 가장 빠른 방식을 정하거나 고정된 속도 향상을 보장할 수 없다. 라이브러리의 알고리즘 선택과 자동 튜닝 여부, 목표 장치에서의 측정이 필요하다.

---

## 9. 컨볼루션을 지나는 역전파

그래디언트 계산을 이해하는 일은 레이어를 직접 만들고 학습의 버그를 잡는 데 꼭 필요하다.

**입력에 대한 그래디언트** (그래디언트를 뒤로 전파하기 위한 것):

$$\frac{\partial L}{\partial X_{c,i,j}} = \sum_{k,m,n} \frac{\partial L}{\partial Y_{k, i-m, j-n}} \cdot W_{k,c,m,n}$$

위 식은 스트라이드·팽창률 1인 순전파 상호상관에 대해, 출력 그래디언트를 범위 밖에서 0으로 두었을 때의 입력 그래디언트이다. 수학적 정의로는 출력 그래디언트와 뒤집지 않은 웨이트의 full 컨볼루션이다. 이를 상호상관 함수로 구현할 때 커널을 뒤집는다. 일반적인 스트라이드·패딩에서는 해당 선형 연산의 전치인 전치 컨볼루션과 연결된다([전치 컨볼루션](transposed_conv.md) 참고).

**웨이트에 대한 그래디언트** (파라메터를 갱신하기 위한 것):

$$\frac{\partial L}{\partial W_{k,c,m,n}} = \sum_{i,j} \frac{\partial L}{\partial Y_{k,i,j}} \cdot X_{c, i+m, j+n}$$

이는 입력과 출력 그래디언트 사이의 **상호상관**이다.

**바이어스에 대한 그래디언트**:

$$\frac{\partial L}{\partial b_k} = \sum_{i,j} \frac{\partial L}{\partial Y_{k,i,j}}$$

---

## 10. 핵심 정리

1. **CNN은 참된 컨볼루션이 아니라 상호상관을 쓴다.** 다만 두 말을 자주 섞어 쓴다
2. **커널이 입력 위를 미끄러지며** 자리마다 내적을 계산한다
3. **다채널 컨볼루션**은 모든 입력 채널의 몫을 더한다
4. **평행 이동 등변성**은 입력의 이동에 출력 위치가 대응한다는 성질이며 경계·스트라이드의 영향을 받는다
5. **파라메터 공유**와 **지역 연결**이 컨볼루션을 효율적으로 만든다
6. **앞쪽 레이어**는 단순한 피처를, **깊은 레이어**는 복잡한 무늬를 잡는다
7. **행렬 표현**은 계산을 이해하는 관점이며 실제 가속 알고리즘은 구현과 장치에 따라 달라진다

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
단일 채널 입력과 커널에 대해 2차원 상호상관을 (라이브러리 함수 없이) 밑바닥부터 구현하라. `F.conv2d`과 비교하여 확인하라.

</div>

??? success "풀이"
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

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
입력 채널이 3개, 출력 채널이 64개이고 커널이 $3 \times 3$인 (바이어스가 있는) 뉴럴 넷트웍의 첫 컨볼루션 레이어에서 파라메터의 수를 계산하라.

</div>

??? success "풀이"
    출력 채널마다 입력 채널당 $3 \times 3$ 커널 하나와 바이어스 하나가 있다.

    필터당 파라메터: $3 \times 3 \times 3 = 27$(웨이트) $+ 1$(바이어스) $= 28$.

    모두 $64 \times 28 = 1{,}792$개의 파라메터이다.

    편향이 없으면 $64 \times 27 = 1{,}728$개이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
무한 정수 격자에서 스트라이드 1인 상호상관이 입력의 평행 이동에 등변임을 보여라.

</div>

??? success "풀이"
    $(T_ax)[i]=x[i-a]$로 두면 $((T_ax)\star h)[i]=\sum_jx[i+j-a]h[j]=(x\star h)[i-a]=T_a(x\star h)[i]$이다. 유한한 이미지에서는 경계와 자르기 방식에 따라 이 등식의 적용 범위가 달라지며, 스트라이드가 있으면 격자에 정렬되는 이동만 대응할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
컨볼루션과 상호상관의 차이를 설명하고 두 결과를 비교할 때 필요한 조건을 적어라.

</div>

??? success "풀이"
    상호상관은 $y[i]=\sum_jx[i+j]h[j]$로 계산하고 같은 정렬의 컨볼루션은 커널의 인덱스를 뒤집어 적용한다. 커널이 중심을 기준으로 대칭이고 출력의 정렬·경계 규약도 같으면 두 결과가 같다. 학습 가능한 커널의 뒤집기는 리파라메리제이션할 수 있지만 고정 필터나 역전파 식에서는 정의를 명시해야 한다.

---

## 정리하며

CNN의 컨볼루션은 일반적으로 상호상관이며, 커널 공유와 국소 연결을 사용한다. groups=1이고 바이어스가 있는 레이어의 파라메터는 $C_{out}C_{in}K_HK_W+C_{out}$개이다. 출력 크기에는 스트라이드·패딩·팽창률이 영향을 주고, 이동 등변성은 불변성과 구분해야 한다. 행렬 표현의 전치는 입력 그래디언트와 전치 컨볼루션을 이해하는 기준이다.

**참고 문헌**

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). "Gradient-based learning applied to document recognition." *Proceedings of the IEEE*, 86(11), 2278-2324.

2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning." MIT Press. Chapter 9: Convolutional Networks.

3. Dumoulin, V., & Visin, F. (2016). "A guide to convolution arithmetic for deep learning." *arXiv preprint arXiv:1603.07285*.

4. Chellapilla, K., Puri, S., & Simard, P. (2006). "High Performance Convolutional Neural Networks for Document Processing." *Tenth International Workshop on Frontiers in Handwriting Recognition*.
