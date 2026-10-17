# 1차원 컨볼루션

1차원 컨볼루션은 시퀀스의 한 축을 따라 같은 커널을 공유해 국소 피처를 계산한다. 시계열·음향·텍스트 등에 사용할 수 있으며, 다채널 입력과 인과 패딩, 팽창률을 조합해 시간 컨볼루션 뉴럴 넷트웍(TCN)을 구성한다.

컨볼루션이라는 이름과 달리 nn.Conv1d는 커널을 뒤집지 않는 상호상관을 계산한다. 수식과 역전파를 해석할 때에는 수학적 컨볼루션의 정의와 구분한다.

## 1. 수학적 정식화

### 단일 채널 1차원 컨볼루션

1차원 입력 $\mathbf{x} \in \mathbb{R}^{n}$과 커널 $\mathbf{w} \in \mathbb{R}^{k}$에 대해 다음과 같다.

$$y[i] = \sum_{j=0}^{k-1} x[i + j] \cdot w[j]$$

(패딩가 없을 때) 출력 크기는 $n - k + 1$이다.

!!! warning "컨볼루션이라 부르지만 커널을 뒤집지 않는다"
    위 식은 엄밀히 말하면 컨볼루션이 아니라 **상호상관**이다. 참된 컨볼루션이라면
    커널을 뒤집어 $y[i] = \sum_{j=0}^{k-1} x[i+j] \cdot w[k-1-j]$로 적어야 한다.
    부호가 뒤집힌 커널에서는 이 차이가 곧바로 드러난다. 아래 보기의
    $\mathbf{w} = [1, 0, -1]$을 뒤집으면 $[-1, 0, 1]$이 되어 출력이
    $[-2, -2, -2]$가 아니라 $[2, 2, 2]$가 된다.

    `nn.Conv1d`도, 이 페이지의 모든 식도 뒤집지 않는 쪽을 쓴다. 커널을 학습으로 얻는
    이상 뒤집기란 파라메터를 다시 이름 붙이는 일에 지나지 않기 때문이다. 다만
    수식을 손으로 옮겨 적을 때는 어느 쪽인지 반드시 정해 두어야 한다 —
    [컨볼루션](convolution.md)에서 두 연산을 나란히 놓고 다룬다.

### 예

입력 $\mathbf{x} = [1, 2, 3, 4, 5]$과 커널 $\mathbf{w} = [1, 0, -1]$을 생각해 보자.

```
Position 0: 1×1 + 2×0 + 3×(-1) = 1 - 3 = -2
Position 1: 2×1 + 3×0 + 4×(-1) = 2 - 4 = -2
Position 2: 3×1 + 4×0 + 5×(-1) = 3 - 5 = -2

Output: [-2, -2, -2]
```

이 커널은 두 칸 떨어진 값의 차이를 계산하는 차분 필터이다. 중앙 차분의 미분 근사로 해석하려면 표본 간격과 정규화 계수, 부호 규약을 함께 정해야 한다.

### 다채널로 나타내기

입력 채널이 $C_{in}$개이고 출력 채널을 $C_{out}$개 낼 때 다음과 같다.

$$Y[o, i] = \sum_{c=0}^{C_{in}-1} \sum_{j=0}^{k-1} X[c, i+j] \cdot W[o, c, j] + b[o]$$

웨이트 텐서의 모양은 $W \in \mathbb{R}^{C_{out} \times C_{in} \times k}$이다.

---

## 2. PyTorch의 `nn.Conv1d`

### 인터페이스

`nn.Conv1d`의 인자를 한자리에 모아 보자. 아래는 실제로 돌아가는 코드이며,
값은 뒤의 다채널 예제에서 다시 쓰는 것과 같게 잡았다.

```python
import torch
import torch.nn as nn

# Conv1d의 인자 — 값은 뒤의 다채널 예제와 같다
conv1d = nn.Conv1d(
    in_channels=8,        # 입력 채널의 수
    out_channels=32,      # 출력 채널(필터)의 수
    kernel_size=5,        # 컨볼루션 커널의 크기
    stride=1,             # 컨볼루션의 스트라이드
    padding=2,            # 양쪽에 더하는 0 덧대기
    dilation=1,           # 커널 원소 사이의 간격
    groups=1,             # 막힌 연결의 수
    bias=True,            # 학습 가능한 바이어스 더하기
    padding_mode='zeros'  # 'zeros', 'reflect', 'replicate', 'circular'
)
print(conv1d)
```

**출력:**

```
Conv1d(8, 32, kernel_size=(5,), stride=(1,), padding=(2,))
```

기본값과 같은 인자는 출력에서 빠진다. `dilation=1`, `groups=1`, `bias=True`,
`padding_mode='zeros'`가 보이지 않는 까닭이 그것이다.

**입력 모양**: $(N, C_{in}, L)$ — 배치 크기, 입력 채널, 시퀀스 길이

**출력 모양**: $(N, C_{out}, L_{out})$이며 $L_{out} = \left\lfloor \frac{L + 2p - d(k-1) - 1}{s} \right\rfloor + 1$이다

### 기본 예제

```python
import torch
import torch.nn as nn

# 1차원 컨볼루션 예제
# 입력: (배치 크기, 입력 채널, 길이)
x = torch.tensor([[[1., 2., 3., 4., 5.]]])  # 모양: (1, 1, 5)

# 1차원 컨볼루션 레이어 만들기: 입력 채널 1개, 출력 채널 1개, 커널 크기 3
conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, bias=False)

# 웨이트를 [1, 0, -1]로 직접 지정 (모서리 검출 커널)
with torch.no_grad():
    conv1d.weight = nn.Parameter(torch.tensor([[[1., 0., -1.]]]))

output = conv1d(x)
print(f"Input shape: {x.shape}")      # torch.Size([1, 1, 5])
print(f"Output shape: {output.shape}") # torch.Size([1, 1, 3])
print(f"Output: {output}")             # tensor([[[-2., -2., -2.]]])
```

**출력:**

```
Input shape: torch.Size([1, 1, 5])
Output shape: torch.Size([1, 1, 3])
Output: tensor([[[-2., -2., -2.]]], grad_fn=<ConvolutionBackward0>)
```

웨이트를 직접 지정했으므로 이 출력은 1절의 손 계산과 정확히 같다. 커널을 뒤집는
참된 컨볼루션이었다면 $[2, 2, 2]$가 나왔을 자리이다.

### 다채널 예제

```python
torch.manual_seed(0)

# 피처 8개(예: OHLCV와 지표), 길이 100인 시계열
batch_size = 32
x = torch.randn(batch_size, 8, 100)  # (N, C_in, L)

# 커널 크기 5로 시간 피처 32개 뽑기
conv = nn.Conv1d(in_channels=8, out_channels=32, kernel_size=5, padding=2)

output = conv(x)
print(f"Input: {x.shape}")    # [32, 8, 100]
print(f"Output: {output.shape}")  # [32, 32, 100] (padding=2이면 길이가 같다)

params = sum(p.numel() for p in conv.parameters())
print(f"Parameters: {params:,}")  # 8 × 32 × 5 + 32 = 1,312
```

**출력:**

```
Input: torch.Size([32, 8, 100])
Output: torch.Size([32, 32, 100])
Parameters: 1,312
```

---

## 3. 행렬 곱으로 본 1차원 컨볼루션

1차원 컨볼루션을 창을 미끄러뜨리는 것으로 보는 관점은 **퇴플리츠 행렬**을 곱하는 것으로도 나타낼 수 있다. 입력 $\mathbf{x} = [x_0, x_1, x_2, x_3, x_4]^\top$과 커널 $\mathbf{k} = [k_0, k_1, k_2]^\top$에 대해 다음과 같다.

$$\mathbf{y} = \mathbf{T}\mathbf{x} = \begin{bmatrix}
k_0 & k_1 & k_2 & 0 & 0 \\
0 & k_0 & k_1 & k_2 & 0 \\
0 & 0 & k_0 & k_1 & k_2
\end{bmatrix}
\begin{bmatrix}
x_0 \\ x_1 \\ x_2 \\ x_3 \\ x_4
\end{bmatrix}$$

$i$번째 행이 $i$번째 열부터 $k_0, k_1, k_2$를 늘어놓는다는 점을 눈여겨보라. 이것이 1절의 $y[i] = \sum_j x[i+j] k[j]$를 그대로 옮긴 것이다.

이 행렬은 희소하고 대각선을 따라 값이 반복되는 구조를 갖는다. 이를 활용하면 행렬의 모든 원소를 저장하지 않고 커널만 저장할 수 있다. 속도 차이는 입력 크기와 구현에 따라 달라진다.

### $\mathbf{T}^\top$으로 본 전치 컨볼루션

입력에 대한 컨볼루션의 기울기는 $\mathbf{T}^\top$을 곱하는 것에 해당하며, 이것이 **전치 컨볼루션**이다([전치 컨볼루션](transposed_conv.md) 참고).

$$\mathbf{T}^\top = \begin{bmatrix}
k_0 & 0 & 0 \\
k_1 & k_0 & 0 \\
k_2 & k_1 & k_0 \\
0 & k_2 & k_1 \\
0 & 0 & k_2
\end{bmatrix}$$

이 예의 전치 연산은 길이 3인 벡터를 길이 5로 보낸다. 원래 입력을 복원하는 역행렬은 아니며, 스트라이드 1에서도 경계 규약에 따라 길이가 늘어날 수 있다.

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

# 확인: Conv1d의 역전파 = ConvTranspose1d의 순전파
x = torch.randn(1, 1, 5, requires_grad=True)
w = torch.randn(1, 1, 3)

# 순전파
y = torch.nn.functional.conv1d(x, w)
# y의 모양은 (1, 1, 3)

# 역전파가 x에 대한 그래디언트를 준다
grad_output = torch.randn(1, 1, 3)
y.backward(grad_output)

# 이는 전치 컨볼루션과 같다
grad_manual = torch.nn.functional.conv_transpose1d(grad_output, w)
print(f"Gradient match: {torch.allclose(x.grad, grad_manual, atol=1e-5)}")
```

**출력:**

```
Gradient match: True
```

---

## 4. 1차원 컨볼루션의 역전파

### 순전파

$$y_i = \sum_{j=0}^{k-1} x_{i+j} \cdot w_j$$

### 입력에 대한 그래디언트

$$\frac{\partial L}{\partial x_i} = \sum_{j=\max(0, i-k+1)}^{\min(i, n-k)} \frac{\partial L}{\partial y_j} \cdot w_{i-j}$$

$w$의 첨자가 $i - j$라는 점이 요점이다. 이는 출력 그래디언트와 커널의 **full 컨볼루션**이지, 뒤집은 커널과의 컨볼루션이 아니다.

$$\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{y}} *_{\text{full}} \mathbf{w} = \text{pad}_{k-1}\!\left(\frac{\partial L}{\partial \mathbf{y}}\right) \star \text{flip}(\mathbf{w})$$

여기서 $*_{\text{full}}$은 full 컨볼루션, $\star$은 상호상관이다. 뒤집기는 **식이 아니라 구현에서** 나온다 — 컨볼루션을 상호상관 고리로 계산하려면 커널을 뒤집어야 하므로, 아래 코드가 `w_flip`을 쓴다. 두 자리 모두에서 뒤집으면 부호가 어긋나고, 그 오차는 그래디언트 확인을 통과하지 못한다.

### 커널에 대한 그래디언트

$$\frac{\partial L}{\partial w_j} = \sum_{i} \frac{\partial L}{\partial y_i} \cdot x_{i+j}$$

이는 입력과 출력 그래디언트의 **상호상관**이다.

### NumPy 구현

```python
import numpy as np

def conv1d_forward(x, w):
    """1차원 컨볼루션(상호상관) 순전파."""
    n, k = len(x), len(w)
    out_len = n - k + 1
    y = np.zeros(out_len)
    for i in range(out_len):
        y[i] = np.sum(x[i:i+k] * w)
    return y

def conv1d_backward(x, w, grad_output):
    """
    1차원 컨볼루션 역전파.

    반환값:
        grad_x: 입력에 대한 그래디언트 (dL/dx)
        grad_w: 웨이트에 대한 그래디언트 (dL/dw)
    """
    n, k = len(x), len(w)
    out_len = len(grad_output)

    # 입력에 대한 그래디언트: grad_output과 w의 온전한 컨볼루션.
    # 상호상관 고리로 계산하므로 여기서 커널을 뒤집는다.
    grad_x = np.zeros(n)
    w_flip = w[::-1]
    grad_padded = np.pad(grad_output, (k-1, k-1), mode='constant')
    for i in range(n):
        grad_x[i] = np.sum(grad_padded[i:i+k] * w_flip)

    # 웨이트에 대한 그래디언트: 입력과 grad_output의 상관
    grad_w = np.zeros(k)
    for j in range(k):
        grad_w[j] = np.sum(x[j:j+out_len] * grad_output)

    return grad_x, grad_w

# 수치적 그래디언트 확인
np.random.seed(42)
x = np.random.randn(8)
w = np.random.randn(3)

y = conv1d_forward(x, w)
grad_output = np.random.randn(len(y))

grad_x, grad_w = conv1d_backward(x, w, grad_output)

# 수치적 확인
eps = 1e-5
grad_w_numerical = np.zeros_like(w)
for i in range(len(w)):
    w_plus, w_minus = w.copy(), w.copy()
    w_plus[i] += eps
    w_minus[i] -= eps
    grad_w_numerical[i] = (np.sum(conv1d_forward(x, w_plus) * grad_output) -
                            np.sum(conv1d_forward(x, w_minus) * grad_output)) / (2 * eps)

print("Analytical grad_w:", grad_w)
print("Numerical grad_w: ", grad_w_numerical)
print("Match:", np.allclose(grad_w, grad_w_numerical))

# grad_x가 어느 쪽 컨볼루션인지도 확인한다.
# np.convolve는 참된 컨볼루션이므로 커널을 뒤집어 받지 않는다.
print("grad_x == full conv(dL/dy, w):", np.allclose(grad_x, np.convolve(grad_output, w)))
print("grad_x == full conv(dL/dy, flip(w)):",
      np.allclose(grad_x, np.convolve(grad_output, w[::-1])))
```

**출력:**

```
Analytical grad_w: [-3.76229763 -3.75680121 -0.74651747]
Numerical grad_w:  [-3.76229763 -3.75680121 -0.74651747]
Match: True
grad_x == full conv(dL/dy, w): True
grad_x == full conv(dL/dy, flip(w)): False
```

출력 그래디언트와 원래 커널의 full 컨볼루션이 입력 그래디언트와 일치하는지 비교한다. 상호상관으로 구현할 때에는 커널을 뒤집지만 수학적 컨볼루션 식에 다시 뒤집기를 넣으면 다른 연산이 된다.

---

## 5. 인과 컨볼루션

많은 시계열 응용에서 모델은 미래를 들여다보아서는 안 된다. 시각 $t$의 출력은 시각이 $t$ 이하인 입력에만 기대야 한다. 이를 위해 **인과 컨볼루션**이 필요하다.

### 왼쪽에만 패딩

커널 크기가 $k$, 팽창률이 $d$일 때 양쪽에 $p = d(k-1)$만큼 덧대면 출력 길이가 $L + d(k-1)$이 된다. 뒤쪽 $p$개를 잘라 내면 길이가 $L$로 돌아오고, 남은 각 출력은 자기 시각과 그 이전만 본다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    """
    인과적 1차원 컨볼루션: 시각 t의 출력은 시각이 t 이하인 입력에만 기댄다.

    왼쪽에만 덧대고 뒤쪽 원소를 잘라 내어 이룬다.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
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

torch.manual_seed(0)

# 길이 확인
causal = CausalConv1d(1, 1, kernel_size=3)
x = torch.randn(1, 1, 10)
y = causal(x)
print(f"Input length: {x.shape[2]}, Output length: {y.shape[2]}")

# 인과성 확인: 시각 5부터 입력을 크게 흔들어도 그 앞의 출력은 꿈쩍하지 않아야 한다
x_future = x.clone()
x_future[0, 0, 5:] += 100.0
diff = (causal(x_future) - y).abs()[0, 0]
print(f"Max |change| at t < 5: {diff[:5].max():.2e}")
print(f"First changed output index: {int((diff > 1e-6).nonzero().min())}")
```

**출력:**

```
Input length: 10, Output length: 10
Max |change| at t < 5: 0.00e+00
First changed output index: 5
```

출력 길이만으로 인과성을 판단할 수 없다. 이 예제는 시각 5부터 입력을 바꾼 뒤 그보다 앞선 출력이 바뀌지 않는지 확인한다. 한 입력에 대한 수치 비교는 점검이며, 일반적인 인과성은 각 출력이 참조하는 인덱스가 현재 이하라는 구조로 설명해야 한다.

### 팽창 인과 컨볼루션 (WaveNet 방식)

팽창률을 지수적으로 키우며 인과 컨볼루션을 쌓으면 인과성을 지키면서도 아주 넓은 리셉티브 필드를 얻는다.

```python
class DilatedCausalConv1d(nn.Module):
    """순차열 모형을 위한 팽창 인과 컨볼루션."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding
        )

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.padding] if self.padding > 0 else out

def build_wavenet_stack(channels, kernel_size=2, num_layers=10):
    """팽창률을 지수적으로 키우며 쌓는다."""
    layers = []
    for i in range(num_layers):
        dilation = 2 ** i  # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
        layers.append(DilatedCausalConv1d(channels, channels, kernel_size, dilation))
    return nn.Sequential(*layers)

# 리셉티브 필드는 레이어마다 d * (K - 1)씩 늘어난다. 값으로 적지 말고 세어 보자.
kernel_size, num_layers = 2, 10
stack = build_wavenet_stack(64, kernel_size=kernel_size, num_layers=num_layers)

dilations = [2 ** i for i in range(num_layers)]
receptive_field = 1 + sum(d * (kernel_size - 1) for d in dilations)

print(f"Dilations: {dilations}")
print(f"Total layers: {len(stack)}, Receptive field: {receptive_field} samples")
print(f"At 16kHz audio: {receptive_field/16000:.3f}s of context")
```

**출력:**

```
Dilations: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
Total layers: 10, Receptive field: 1024 samples
At 16kHz audio: 0.064s of context
```

이 예제는 커널 크기 2, 스트라이드 1과 팽창률 $1,2,\ldots,512$를 사용하므로 이론적 리셉티브 필드는 $1+\sum_{i=0}^{9}2^i=1024$이다. 팽창률을 지수적으로 늘렸기 때문에 레이어 수에 대해 지수적으로 커진다. 고정된 채널 수에서 파라메터 수는 레이어 수에 비례하며, 연산량은 시퀀스 길이에도 의존한다.

---

## 6. 시간 컨볼루션 뉴럴 넷트웍 (TCN)

TCN은 인과 컨볼루션, 팽창, 잔차 연결을 엮어 널리 쓸 수 있는 시퀀스 모델을 만든다.

```python
import torch
import torch.nn as nn

class TCNBlock(nn.Module):
    """
    시간 컨볼루션 뉴럴 넷트웍 블록.

    다음을 엮는다:
    - 팽창 인과 컨볼루션 **둘** (리셉티브 필드 계산에서 자주 놓치는 대목이다)
    - 웨이트 정규화
    - ReLU 활성화
    - 규제를 위한 드롭아웃
    - 잔차 연결
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()

        self.padding = (kernel_size - 1) * dilation

        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      dilation=dilation, padding=self.padding)
        )
        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      dilation=dilation, padding=self.padding)
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 잔차 연결 (채널이 바뀌면 1×1 컨볼루션)
        self.residual = (nn.Conv1d(in_channels, out_channels, 1)
                        if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        # 첫 컨볼루션 블록
        out = self.conv1(x)
        out = out[:, :, :-self.padding]  # 인과적으로 잘라내기
        out = self.relu(out)
        out = self.dropout(out)

        # 둘째 컨볼루션 블록 — 팽창률이 같으므로 리셉티브 필드가 한 번 더 늘어난다
        out = self.conv2(out)
        out = out[:, :, :-self.padding]  # 인과적으로 잘라내기
        out = self.relu(out)
        out = self.dropout(out)

        # 잔차
        return self.relu(out + self.residual(x))

class TCN(nn.Module):
    """완전한 시간 컨볼루션 뉴럴 넷트웍."""
    def __init__(self, input_channels, hidden_channels, output_size,
                 kernel_size=3, num_layers=6, dropout=0.2):
        super().__init__()

        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            in_ch = input_channels if i == 0 else hidden_channels
            layers.append(TCNBlock(in_ch, hidden_channels, kernel_size, dilation, dropout))

        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_channels, output_size)

    def forward(self, x):
        # x: (배치, 채널, 순차열 길이)
        out = self.network(x)
        # 분류/회귀를 위해 마지막 시각을 쓴다
        out = out[:, :, -1]
        return self.output_layer(out)

torch.manual_seed(0)

# 예: 피처 8개짜리 가격 이력에서 다음 시점의 수익률 예측
kernel_size, num_layers = 3, 8
model = TCN(input_channels=8, hidden_channels=64, output_size=1,
            kernel_size=kernel_size, num_layers=num_layers)

x = torch.randn(32, 8, 256)  # 표본 32개, 피처 8개, 시각 256개
pred = model(x)
print(f"Input: {x.shape}, Prediction: {pred.shape}")  # [32, 1]

# 리셉티브 필드: 블록마다 팽창 컨볼루션이 둘이므로 (K-1)*d 가 블록마다 두 번 더해진다.
# RF = 1 + 2 * sum_{i=0}^{7} 2^i * (3-1) = 1 + 4 * 255 = 1021
convs_per_block = 2
receptive_field = 1 + convs_per_block * sum(
    2 ** i * (kernel_size - 1) for i in range(num_layers))
print(f"Receptive field (formula): {receptive_field} time steps")

# 식을 믿지 말고 재 본다. 마지막 출력에 그래디언트를 흘려 0이 아닌 입력 자리를 센다.
model.eval()
probe = torch.zeros(1, 8, 2048, requires_grad=True)
model.network(probe)[0, :, -1].sum().backward()
touched = (probe.grad.abs().sum(0).sum(0) != 0).nonzero()
print(f"Receptive field (measured): {int(touched.max() - touched.min()) + 1} time steps")
```

**출력:**

```
Input: torch.Size([32, 8, 256]), Prediction: torch.Size([32, 1])
Receptive field (formula): 1021 time steps
Receptive field (measured): 1021 time steps
```

!!! warning "TCN 블록에는 컨볼루션이 두 개 있다"
    같은 팽창률의 컨볼루션이 블록마다 두 번 있으므로, 커널 3·블록 8개의 이론적 리셉티브 필드는 $1+2(3-1)\sum_{i=0}^{7}2^i=1021$이다. 블록당 하나만 세면 511로 잘못 계산한다.

    기울기가 0이 아닌 위치를 조사하는 예제는 해당 초기 웨이트와 입력에서 실제로 영향을 준 위치를 확인한다. ReLU 비활성화, 상쇄와 수치적 언더플로 때문에 일반적으로 이론적 리셉티브 필드 전체가 검출된다고 보장할 수 없다. 입력 길이가 256이면 이론적 범위 중 존재하는 과거 입력은 최대 256개이고 나머지는 패딩이다. 추가 레이어는 존재하는 과거 범위를 더 넓힐 수 없지만 표현 자체는 바꿀 수 있다.

---

## 7. 금융 시계열을 위한 1차원 컨볼루션

### 가격 데이터에서 피처 샘플링

```python
import torch
import torch.nn as nn

class FinancialFeatureExtractor(nn.Module):
    """
    금융 시계열에서 여러 시간 지평의 무늬를 뽑아내는
    여러 규모의 1차원 컨볼루션.
    """
    def __init__(self, input_features, hidden_dim=64):
        super().__init__()

        # 단기 무늬 (3일 창)
        self.short_conv = nn.Sequential(
            nn.Conv1d(input_features, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # 중기 무늬 (5일 / 주간)
        self.medium_conv = nn.Sequential(
            nn.Conv1d(input_features, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # 장기 무늬 (21일 / 월간)
        self.long_conv = nn.Sequential(
            nn.Conv1d(input_features, hidden_dim, kernel_size=21, padding=10),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        """
        인수:
            x: (배치, 피처, 시각)
               예: features = [시가, 고가, 저가, 종가, 거래량, 수익률, …]
        반환값:
            여러 규모의 피처: (배치, 3*hidden_dim, 시각)
        """
        short = self.short_conv(x)
        medium = self.medium_conv(x)
        long_term = self.long_conv(x)

        return torch.cat([short, medium, long_term], dim=1)

torch.manual_seed(0)

# 사용 예
extractor = FinancialFeatureExtractor(input_features=6, hidden_dim=32)
# 피처 6개: OHLCV와 수익률, 거래일 252일
x = torch.randn(16, 6, 252)
features = extractor(x)
print(f"Multi-scale features: {features.shape}")  # [16, 96, 252]
```

**출력:**

```
Multi-scale features: torch.Size([16, 96, 252])
```

세 클래스는 홀수 커널 크기 k에 대해 양쪽 패딩을 $(k-1)/2$로 두어 길이 252를 유지한다. 따라서 채널 축으로 결합하면 $3\times32=96$채널이 된다. 각 시각의 출력은 같은 창 안의 이후 시각도 참조하므로 시각별 온라인 예측에는 인과적이지 않다. 다만 관측이 끝난 과거 창 전체를 사용해 창 이후를 예측하는 경우에는 대칭 컨볼루션 자체가 미래 데이터 누출을 뜻하지 않는다. 예측 시점에 무엇이 관측되었는지를 기준으로 판단한다.

---

## 8. Conv1d와 Conv2d: 언제 무엇을 쓸까

| 기준 | Conv1d | Conv2d |
|-----------|--------|--------|
| 데이터의 구조 | 시퀀스, 시계열 | 이미지, 공간 격자 |
| 입력 모양 | $(N, C, L)$ | $(N, C, H, W)$ |
| 커널이 미끄러지는 방향 | 1차원 (시간/위치) | 2차원 (높이 × 너비) |
| 흔한 커널 크기 | 3, 5, 7, 21 | 3×3, 5×5, 7×7 |
| 응용 예 | 음향, 자연어 처리, 금융 | 이미지, 영상 프레임, 변동성 곡면 |
| 파라메터 수 (바이어스 제외) | $C_{out} \times C_{in} \times K$ | $C_{out} \times C_{in} \times K^2$ |

---

## 9. 핵심 정리

1. **Conv1d**는 모양이 $(N, C, L)$인 시퀀스를 다루며 시간 차원을 따라 커널을 미끄러뜨린다. 이름과 달리 커널을 뒤집지 않는 **상호상관**이다
2. **인과 컨볼루션**(왼쪽 패딩 뒤 잘라내기)은 출력이 과거와 현재의 입력에만 기대게 한다. 길이가 보존된다는 사실만으로는 인과성이 증명되지 않으니 미래를 흔들어 확인하라
3. **팽창률을 지수적으로 늘려 인과 컨볼루션을 쌓으면** 이론적 리셉티브 필드가 지수적으로 넓어진다. 커널 크기 2인 팽창 컨볼루션 10레이어이면 리셉티브 필드가 1024가 된다
4. **TCN**은 팽창 인과 컨볼루션과 잔차 연결을 결합한다. 블록당 두 컨볼루션의 기여를 각각 세면 커널 3·블록 8에서 리셉티브 필드는 1021이다.
5. 커널 크기가 서로 다른 **여러 규모의 컨볼루션**은 서로 다른 시간 지평의 무늬를 붙잡는다
6. 1차원 컨볼루션의 **역전파**는 출력 그래디언트와 커널의 full 컨볼루션(= 전치 컨볼루션)이다. 커널을 뒤집는 것은 그 컨볼루션을 상호상관으로 **구현할 때**이지 식 자체가 아니다

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
1차원 컨볼루션이 데이터 구조에 적합한 사례 세 가지를 설명하라.

</div>

??? success "풀이"
    시간에 따라 배열된 신호, 음향 파형, 토큰 임베딩 시퀀스에서 한 축의 국소 관계를 처리할 수 있다. 어떤 구조가 더 나은지는 입력 표현과 과제에 달려 있으며 차원 수만으로 속도나 정확도를 확정할 수 없다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
입력 길이가 100, 커널 크기가 5, 스트라이드가 2, 패딩가 1인 1차원 컨볼루션의 출력 길이를 계산하라.

</div>

??? success "풀이"
    팽창률이 $d = 1$이므로 2절의 식 $L_{out} = \lfloor (L + 2p - d(k-1) - 1)/s \rfloor + 1$이 $\lfloor (L + 2p - k)/s \rfloor + 1$로 줄어든다.

    $$L_{out} = \left\lfloor \frac{100 + 2 - 5}{2} \right\rfloor + 1 = \left\lfloor \frac{97}{2} \right\rfloor + 1 = 48 + 1 = 49$$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
시계열 분류를 위한 1차원 CNN을 PyTorch로 구현하라. 길이가 제각각인 시퀀스에도 쓸 수 있어야 한다.

</div>

??? success "풀이"
    `AdaptiveAvgPool1d(1)`은 허용되는 입력 길이가 바뀌어도 출력의 길이 축을 1로 만든다. 그러나 하나의 배치 텐서에 서로 다른 길이를 그대로 쌓을 수 있다는 뜻은 아니다. 패딩을 쓴다면 컨볼루션과 풀링에서 패딩 값의 영향을 어떻게 처리할지 정해야 한다.

    ```python
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    num_classes = 5

    model = nn.Sequential(
        nn.Conv1d(1, 32, kernel_size=5, padding=2), nn.ReLU(),
        nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, num_classes)
    )

    x = torch.randn(16, 1, 300)
    print("Logits:", model(x).shape)
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    ```

    출력:

    ```
    Logits: torch.Size([16, 5])
    Parameters: 10821
    ```

    파라메터를 손으로 세면 $(1 \cdot 32 \cdot 5 + 32) + (32 \cdot 64 \cdot 5 + 64) + (64 \cdot 5 + 5) = 192 + 10304 + 325 = 10821$로 맞는다. 길이 300은 어디에도 들어가지 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
Conv1d의 출력을 평탄화한 것과 같은 크기의 출력을 내는 완전 연결 레이어의 웨이트 수를 비교하라.

</div>

??? success "풀이"
    입력이 $LC_{in}$차원이고 출력이 $L_{out}C_{out}$차원이면 완전 연결 레이어의 웨이트는 $LC_{in}L_{out}C_{out}$개이다. groups=1인 Conv1d의 웨이트는 $kC_{in}C_{out}$개이므로 바이어스를 제외한 비는 $LL_{out}/k$이다. 완전 연결 레이어의 출력이 위치 없이 $C_{out}$개뿐이면 비가 $L/k$이지만 출력 크기가 다른 비교이다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
스트라이드 1인 CausalConv1d의 최종 출력 길이를 구하고 인과성을 인덱스로 설명하라.

</div>

??? success "풀이"
    $p=d(k-1)$를 양쪽에 패딩하면 길이는 $L+p$가 되고 뒤의 p개를 제거하면 L이다. 남긴 출력 i가 참조하는 입력 인덱스는 $i-p+jd$이며 $0\le j<k$에서 최대가 i이므로 미래를 보지 않는다. $k=1$이면 p=0이고 코드가 자르기를 생략한다. 뒤에서 제거한 값은 단순히 0 출력이 아니라 원래 입력과 패딩이 섞여 계산된 값일 수 있다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
블록마다 같은 팽창률의 컨볼루션을 두 개 쓰고 팽창률을 2의 거듭제곱으로 늘리는 TCN의 이론적 리셉티브 필드를 구하라.

</div>

??? success "풀이"
    스트라이드 1인 컨볼루션 하나가 $d(k-1)$을 더하므로 L개 블록의 이론적 리셉티브 필드는 $1+2(k-1)\sum_{i=0}^{L-1}2^i=1+2(k-1)(2^L-1)$이다. $k=3,L=8$이면 1021이고 블록당 하나만 세면 511이다. 비선형 모델에서 한 입력의 0이 아닌 그래디언트 범위가 언제나 이 값과 같아야 하는 것은 아니다.

---

## 정리하며

Conv1d는 주로 $(N,C_{in},L)$ 입력에 상호상관을 적용한다. 출력 길이는 스트라이드·패딩·팽창률로 정해진다. 인과성은 길이 보존이 아니라 참조하는 시간 인덱스로 판단하고, 이론적 리셉티브 필드와 특정 입력에서의 실제 그래디언트 범위를 구분한다. [컨볼루션](convolution.md), [팽창 컨볼루션](dilated_convolutions.md)과 [리셉티브 필드](receptive_field.md)에서 관련 계산을 이어 간다.

**참고 문헌**

1. van den Oord, A., et al. (2016). "WaveNet: A Generative Model for Raw Audio." *arXiv preprint arXiv:1609.03499*.

2. Bai, S., Kolter, J. Z., & Koltun, V. (2018). "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling." *arXiv preprint arXiv:1803.01271*.

3. Lea, C., et al. (2017). "Temporal Convolutional Networks for Action Segmentation and Detection." *CVPR*.

4. Dumoulin, V., & Visin, F. (2016). "A guide to convolution arithmetic for deep learning." *arXiv preprint arXiv:1603.07285*.
