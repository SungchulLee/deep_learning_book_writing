# 1차원 합성곱

2차원 합성곱이 이미지 처리의 일꾼이라면, **1차원 합성곱**은 시계열, 음향 신호, 텍스트 순차열, 금융 데이터 같은 순차 데이터를 다룬다. 학습 가능한 핵을 하나의 공간(시간) 차원을 따라 미끄러뜨리며 자리마다 지역적인 무늬를 뽑아낸다.

1차원 합성곱은 WaveNet이나 시간 합성곱 신경망(TCN) 같은 구조의 바탕이며, 퀀트 금융에서 가격 순차열, 호가창 스냅숏을 비롯한 시간 신호를 다루는 데 널리 쓰인다.

---

## 1. 수학적 정식화

### 단일 채널 1차원 합성곱

1차원 입력 $\mathbf{x} \in \mathbb{R}^{n}$과 핵 $\mathbf{w} \in \mathbb{R}^{k}$에 대해 다음과 같다.

$$y[i] = \sum_{j=0}^{k-1} x[i + j] \cdot w[j]$$

(덧대기가 없을 때) 출력 크기는 $n - k + 1$이다.

!!! warning "합성곱이라 부르지만 핵을 뒤집지 않는다"
    위 식은 엄밀히 말하면 합성곱이 아니라 **상호상관**이다. 참된 합성곱이라면
    핵을 뒤집어 $y[i] = \sum_{j=0}^{k-1} x[i+j] \cdot w[k-1-j]$로 적어야 한다.
    부호가 뒤집힌 핵에서는 이 차이가 곧바로 드러난다. 아래 보기의
    $\mathbf{w} = [1, 0, -1]$을 뒤집으면 $[-1, 0, 1]$이 되어 출력이
    $[-2, -2, -2]$가 아니라 $[2, 2, 2]$가 된다.

    `nn.Conv1d`도, 이 쪽의 모든 식도 뒤집지 않는 쪽을 쓴다. 핵을 학습으로 얻는
    이상 뒤집기란 매개변수를 다시 이름 붙이는 일에 지나지 않기 때문이다. 다만
    수식을 손으로 옮겨 적을 때는 어느 쪽인지 반드시 정해 두어야 한다 —
    [합성곱](convolution.md)에서 두 연산을 나란히 놓고 다룬다.

### 예

입력 $\mathbf{x} = [1, 2, 3, 4, 5]$과 핵 $\mathbf{w} = [1, 0, -1]$을 생각해 보자.

```
Position 0: 1×1 + 2×0 + 3×(-1) = 1 - 3 = -2
Position 1: 2×1 + 3×0 + 4×(-1) = 2 - 4 = -2
Position 2: 3×1 + 4×0 + 5×(-1) = 3 - 5 = -2

Output: [-2, -2, -2]
```

이 핵은 이산 도함수(차분)를 계산하여 신호의 변화를 잡아낸다.

### 다채널로 나타내기

입력 채널이 $C_{in}$개이고 출력 채널을 $C_{out}$개 낼 때 다음과 같다.

$$Y[o, i] = \sum_{c=0}^{C_{in}-1} \sum_{j=0}^{k-1} X[c, i+j] \cdot W[o, c, j] + b[o]$$

가중치 텐서의 모양은 $W \in \mathbb{R}^{C_{out} \times C_{in} \times k}$이다.

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
    kernel_size=5,        # 합성곱 핵의 크기
    stride=1,             # 합성곱의 보폭
    padding=2,            # 양쪽에 더하는 0 덧대기
    dilation=1,           # 핵 원소 사이의 간격
    groups=1,             # 막힌 연결의 수
    bias=True,            # 학습 가능한 편향 더하기
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

**입력 모양**: $(N, C_{in}, L)$ — 배치 크기, 입력 채널, 순차열 길이

**출력 모양**: $(N, C_{out}, L_{out})$이며 $L_{out} = \left\lfloor \frac{L + 2p - d(k-1) - 1}{s} \right\rfloor + 1$이다

### 기본 예제

```python
import torch
import torch.nn as nn

# 1차원 합성곱 예제
# 입력: (배치 크기, 입력 채널, 길이)
x = torch.tensor([[[1., 2., 3., 4., 5.]]])  # 모양: (1, 1, 5)

# 1차원 합성곱 층 만들기: 입력 채널 1개, 출력 채널 1개, 핵 크기 3
conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, bias=False)

# 가중치를 [1, 0, -1]로 직접 지정 (모서리 검출 핵)
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

가중치를 직접 지정했으므로 이 출력은 1절의 손 계산과 정확히 같다. 핵을 뒤집는
참된 합성곱이었다면 $[2, 2, 2]$가 나왔을 자리이다.

### 다채널 예제

```python
torch.manual_seed(0)

# 특징 8개(예: OHLCV와 지표), 길이 100인 시계열
batch_size = 32
x = torch.randn(batch_size, 8, 100)  # (N, C_in, L)

# 핵 크기 5로 시간 특징 32개 뽑기
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

## 3. 행렬 곱으로 본 1차원 합성곱

1차원 합성곱을 창을 미끄러뜨리는 것으로 보는 관점은 **퇴플리츠 행렬**을 곱하는 것으로도 나타낼 수 있다. 입력 $\mathbf{x} = [x_0, x_1, x_2, x_3, x_4]^\top$과 핵 $\mathbf{k} = [k_0, k_1, k_2]^\top$에 대해 다음과 같다.

$$\mathbf{y} = \mathbf{T}\mathbf{x} = \begin{bmatrix}
k_0 & k_1 & k_2 & 0 & 0 \\
0 & k_0 & k_1 & k_2 & 0 \\
0 & 0 & k_0 & k_1 & k_2
\end{bmatrix}
\begin{bmatrix}
x_0 \\ x_1 \\ x_2 \\ x_3 \\ x_4
\end{bmatrix}$$

$i$번째 행이 $i$번째 열부터 $k_0, k_1, k_2$를 늘어놓는다는 점을 눈여겨보라. 이것이 1절의 $y[i] = \sum_j x[i+j] k[j]$를 그대로 옮긴 것이다.

이 행렬은 **성기고** (대각선을 따라 값이 같은) **짜임이 있다**. 그래서 합성곱이 일반적인 행렬 곱보다 훨씬 효율적이다. 서로 다른 값을 $k$개만 저장하면 된다.

### $\mathbf{T}^\top$으로 본 전치 합성곱

입력에 대한 합성곱의 기울기는 $\mathbf{T}^\top$을 곱하는 것에 해당하며, 이것이 **전치 합성곱**이다([전치 합성곱](transposed_conv.md) 참고).

$$\mathbf{T}^\top = \begin{bmatrix}
k_0 & 0 & 0 \\
k_1 & k_0 & 0 \\
k_2 & k_1 & k_0 \\
0 & k_2 & k_1 \\
0 & 0 & k_2
\end{bmatrix}$$

이는 길이 3인 벡터를 다시 길이 5로 보내며 상향 표본화를 한다.

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

# 역전파가 x에 대한 기울기를 준다
grad_output = torch.randn(1, 1, 3)
y.backward(grad_output)

# 이는 전치 합성곱과 같다
grad_manual = torch.nn.functional.conv_transpose1d(grad_output, w)
print(f"Gradient match: {torch.allclose(x.grad, grad_manual, atol=1e-5)}")
```

**출력:**

```
Gradient match: True
```

---

## 4. 1차원 합성곱의 역전파

### 순전파

$$y_i = \sum_{j=0}^{k-1} x_{i+j} \cdot w_j$$

### 입력에 대한 기울기

$$\frac{\partial L}{\partial x_i} = \sum_{j=\max(0, i-k+1)}^{\min(i, n-k)} \frac{\partial L}{\partial y_j} \cdot w_{i-j}$$

$w$의 첨자가 $i - j$라는 점이 요점이다. 이는 출력 기울기와 핵의 **온전한 합성곱**이지, 뒤집은 핵과의 합성곱이 아니다.

$$\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{y}} *_{\text{full}} \mathbf{w} = \text{pad}_{k-1}\!\left(\frac{\partial L}{\partial \mathbf{y}}\right) \star \text{flip}(\mathbf{w})$$

여기서 $*_{\text{full}}$은 온전한 합성곱, $\star$은 상호상관이다. 뒤집기는 **식이 아니라 구현에서** 나온다 — 합성곱을 상호상관 고리로 계산하려면 핵을 뒤집어야 하므로, 아래 코드가 `w_flip`을 쓴다. 두 자리 모두에서 뒤집으면 부호가 어긋나고, 그 오차는 기울기 확인을 통과하지 못한다.

### 핵에 대한 기울기

$$\frac{\partial L}{\partial w_j} = \sum_{i} \frac{\partial L}{\partial y_i} \cdot x_{i+j}$$

이는 입력과 출력 기울기의 **상호상관**이다.

### NumPy 구현

```python
import numpy as np

def conv1d_forward(x, w):
    """1차원 합성곱(상호상관) 순전파."""
    n, k = len(x), len(w)
    out_len = n - k + 1
    y = np.zeros(out_len)
    for i in range(out_len):
        y[i] = np.sum(x[i:i+k] * w)
    return y

def conv1d_backward(x, w, grad_output):
    """
    1차원 합성곱 역전파.

    반환값:
        grad_x: 입력에 대한 기울기 (dL/dx)
        grad_w: 가중치에 대한 기울기 (dL/dw)
    """
    n, k = len(x), len(w)
    out_len = len(grad_output)

    # 입력에 대한 기울기: grad_output과 w의 온전한 합성곱.
    # 상호상관 고리로 계산하므로 여기서 핵을 뒤집는다.
    grad_x = np.zeros(n)
    w_flip = w[::-1]
    grad_padded = np.pad(grad_output, (k-1, k-1), mode='constant')
    for i in range(n):
        grad_x[i] = np.sum(grad_padded[i:i+k] * w_flip)

    # 가중치에 대한 기울기: 입력과 grad_output의 상관
    grad_w = np.zeros(k)
    for j in range(k):
        grad_w[j] = np.sum(x[j:j+out_len] * grad_output)

    return grad_x, grad_w

# 수치적 기울기 확인
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

# grad_x가 어느 쪽 합성곱인지도 확인한다.
# np.convolve는 참된 합성곱이므로 핵을 뒤집어 받지 않는다.
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

마지막 두 줄이 이 절의 요점을 값으로 못박는다. 기울기는 핵 **그대로**와의 온전한 합성곱이며, 뒤집은 핵으로 쓰면 틀린다.

---

## 5. 인과 합성곱

많은 시계열 응용에서 모델은 미래를 들여다보아서는 안 된다. 시각 $t$의 출력은 시각이 $t$ 이하인 입력에만 기대야 한다. 이를 위해 **인과 합성곱**이 필요하다.

### 왼쪽에만 덧대기

핵 크기가 $k$, 팽창률이 $d$일 때 양쪽에 $p = d(k-1)$만큼 덧대면 출력 길이가 $L + d(k-1)$이 된다. 뒤쪽 $p$개를 잘라 내면 길이가 $L$로 돌아오고, 남은 각 출력은 자기 시각과 그 이전만 본다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    """
    인과적 1차원 합성곱: 시각 t의 출력은 시각이 t 이하인 입력에만 기댄다.

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

길이가 그대로인 것만으로는 인과성의 증거가 되지 못한다. 양쪽에 똑같이 덧대도 길이는 보존되기 때문이다. 뒤쪽 셋째 줄이 진짜 증거다. 시각 5 이후를 100만큼 흔들었는데 시각 0–4의 출력이 정확히 0만큼 바뀌었고, 바뀐 첫 자리가 정확히 시각 5이다.

### 팽창 인과 합성곱 (WaveNet 방식)

팽창률을 지수적으로 키우며 인과 합성곱을 쌓으면 인과성을 지키면서도 아주 넓은 수용 영역을 얻는다.

```python
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
        return out[:, :, :-self.padding] if self.padding > 0 else out

def build_wavenet_stack(channels, kernel_size=2, num_layers=10):
    """팽창률을 지수적으로 키우며 쌓는다."""
    layers = []
    for i in range(num_layers):
        dilation = 2 ** i  # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
        layers.append(DilatedCausalConv1d(channels, channels, kernel_size, dilation))
    return nn.Sequential(*layers)

# 수용 영역은 층마다 d * (K - 1)씩 늘어난다. 값으로 적지 말고 세어 보자.
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

팽창률의 합이 $1 + 2 + \cdots + 512 = 1023$이므로 수용 영역은 $1 + 1023 = 1024$이다. 층 수에 **선형**으로 드는 값으로 문맥을 **지수적으로** 넓힌 셈이다.

---

## 6. 시간 합성곱 신경망 (TCN)

TCN은 인과 합성곱, 팽창, 잔차 연결을 엮어 널리 쓸 수 있는 순차열 모형을 만든다.

```python
import torch
import torch.nn as nn

class TCNBlock(nn.Module):
    """
    시간 합성곱 신경망 블록.

    다음을 엮는다:
    - 팽창 인과 합성곱 **둘** (수용 영역 계산에서 자주 놓치는 대목이다)
    - 가중치 정규화
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

        # 잔차 연결 (채널이 바뀌면 1×1 합성곱)
        self.residual = (nn.Conv1d(in_channels, out_channels, 1)
                        if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        # 첫 합성곱 블록
        out = self.conv1(x)
        out = out[:, :, :-self.padding]  # 인과적으로 잘라내기
        out = self.relu(out)
        out = self.dropout(out)

        # 둘째 합성곱 블록 — 팽창률이 같으므로 수용 영역이 한 번 더 늘어난다
        out = self.conv2(out)
        out = out[:, :, :-self.padding]  # 인과적으로 잘라내기
        out = self.relu(out)
        out = self.dropout(out)

        # 잔차
        return self.relu(out + self.residual(x))

class TCN(nn.Module):
    """완전한 시간 합성곱 신경망."""
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

# 예: 특징 8개짜리 가격 이력에서 다음 시점의 수익률 예측
kernel_size, num_layers = 3, 8
model = TCN(input_channels=8, hidden_channels=64, output_size=1,
            kernel_size=kernel_size, num_layers=num_layers)

x = torch.randn(32, 8, 256)  # 표본 32개, 특징 8개, 시각 256개
pred = model(x)
print(f"Input: {x.shape}, Prediction: {pred.shape}")  # [32, 1]

# 수용 영역: 블록마다 팽창 합성곱이 둘이므로 (K-1)*d 가 블록마다 두 번 더해진다.
# RF = 1 + 2 * sum_{i=0}^{7} 2^i * (3-1) = 1 + 4 * 255 = 1021
convs_per_block = 2
receptive_field = 1 + convs_per_block * sum(
    2 ** i * (kernel_size - 1) for i in range(num_layers))
print(f"Receptive field (formula): {receptive_field} time steps")

# 식을 믿지 말고 재 본다. 마지막 출력에 기울기를 흘려 0이 아닌 입력 자리를 센다.
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

!!! warning "블록당 합성곱이 둘이라는 점을 빼먹기 쉽다"
    블록마다 팽창 합성곱이 하나라고 세면 $1 + 2\sum_{i=0}^{7} 2^i = 511$이 나온다.
    실제로는 `conv1`과 `conv2`가 같은 팽창률로 두 번 쌓이므로
    $1 + 2 \cdot 2 \sum_{i=0}^{7} 2^i = 1 + 4 \cdot 255 = 1021$이다. 위 코드의
    마지막 줄은 이 값을 식과 따로, 기울기를 흘려 **재서** 얻은 것이라 둘이
    맞는다는 사실이 곧 확인이 된다.

    수용 영역 1021은 입력 길이 256보다 넓다. 즉 이 설정에서 마지막 출력은 이미
    순차열 전체를 본다. 팽창 층을 더 쌓아도 문맥은 더 늘지 않는다.

---

## 7. 금융 시계열을 위한 1차원 합성곱

### 가격 데이터에서 특징 뽑기

```python
import torch
import torch.nn as nn

class FinancialFeatureExtractor(nn.Module):
    """
    금융 시계열에서 여러 시간 지평의 무늬를 뽑아내는
    여러 규모의 1차원 합성곱.
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
            x: (배치, 특징, 시각)
               예: features = [시가, 고가, 저가, 종가, 거래량, 수익률, …]
        반환값:
            여러 규모의 특징: (배치, 3*hidden_dim, 시각)
        """
        short = self.short_conv(x)
        medium = self.medium_conv(x)
        long_term = self.long_conv(x)

        return torch.cat([short, medium, long_term], dim=1)

torch.manual_seed(0)

# 사용 예
extractor = FinancialFeatureExtractor(input_features=6, hidden_dim=32)
# 특징 6개: OHLCV와 수익률, 거래일 252일
x = torch.randn(16, 6, 252)
features = extractor(x)
print(f"Multi-scale features: {features.shape}")  # [16, 96, 252]
```

**출력:**

```
Multi-scale features: torch.Size([16, 96, 252])
```

세 갈래 모두 `padding = (k-1)/2`로 잡아 길이를 252로 맞추었기에 `torch.cat`이 채널 축에서 이어 붙을 수 있다. 채널은 $3 \times 32 = 96$이 된다. 이 세 갈래는 인과적이지 **않다** — 양쪽에 덧대므로 시각 $t$의 출력이 미래를 본다. 예측 모형에 쓰려면 5절의 인과 덧대기로 바꾸어야 한다.

---

## 8. Conv1d와 Conv2d: 언제 무엇을 쓸까

| 기준 | Conv1d | Conv2d |
|-----------|--------|--------|
| 데이터의 짜임 | 순차열, 시계열 | 이미지, 공간 격자 |
| 입력 모양 | $(N, C, L)$ | $(N, C, H, W)$ |
| 핵이 미끄러지는 방향 | 1차원 (시간/위치) | 2차원 (높이 × 너비) |
| 흔한 핵 크기 | 3, 5, 7, 21 | 3×3, 5×5, 7×7 |
| 응용 예 | 음향, 자연어 처리, 금융 | 이미지, 영상 프레임, 변동성 곡면 |
| 매개변수 수 (편향 제외) | $C_{out} \times C_{in} \times K$ | $C_{out} \times C_{in} \times K^2$ |

---

## 9. 핵심 정리

1. **Conv1d**는 모양이 $(N, C, L)$인 순차열을 다루며 시간 차원을 따라 핵을 미끄러뜨린다. 이름과 달리 핵을 뒤집지 않는 **상호상관**이다
2. **인과 합성곱**(왼쪽 덧대기 뒤 잘라내기)은 출력이 과거와 현재의 입력에만 기대게 한다. 길이가 보존된다는 사실만으로는 인과성이 증명되지 않으니 미래를 흔들어 확인하라
3. **팽창 인과 합성곱을 쌓으면** 수용 영역이 지수적으로 넓어진다. 핵 크기 2인 팽창 합성곱 10층이면 수용 영역이 1024가 된다
4. **TCN**은 팽창 인과 합성곱에 잔차 연결을 엮어 경쟁력 있는 순차열 모형을 만든다. 블록마다 합성곱이 **둘**이므로 수용 영역도 두 배로 늘어난다(핵 3, 블록 8이면 511이 아니라 1021)
5. 핵 크기가 서로 다른 **여러 규모의 합성곱**은 서로 다른 시간 지평의 무늬를 붙잡는다
6. 1차원 합성곱의 **역전파**는 출력 기울기와 핵의 온전한 합성곱(= 전치 합성곱)이다. 핵을 뒤집는 것은 그 합성곱을 상호상관으로 **구현할 때**이지 식 자체가 아니다

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
2차원 합성곱보다 1차원 합성곱이 나은 때를 설명하고 응용 예 세 가지를 들어라.

</div>

??? success "연습문제 1 풀이"
    공간 구조가 1차원인 순차 데이터에는 1차원 합성곱이 낫다. (1) 시계열 예측, (2) 음향·음성 처리, (3) 자연어 텍스트 분류(글자 단위 또는 낱말 단위)가 그 예이다. 핵을 한 방향으로만 미끄러뜨리므로 2차원보다 계산이 싸다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
입력 길이가 100, 핵 크기가 5, 보폭이 2, 덧대기가 1인 1차원 합성곱의 출력 길이를 계산하라.

</div>

??? success "연습문제 2 풀이"
    팽창률이 $d = 1$이므로 2절의 식 $L_{out} = \lfloor (L + 2p - d(k-1) - 1)/s \rfloor + 1$이 $\lfloor (L + 2p - k)/s \rfloor + 1$로 줄어든다.

    $$L_{out} = \left\lfloor \frac{100 + 2 - 5}{2} \right\rfloor + 1 = \left\lfloor \frac{97}{2} \right\rfloor + 1 = 48 + 1 = 49$$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
시계열 분류를 위한 1차원 CNN을 PyTorch로 구현하라. 길이가 제각각인 순차열에도 쓸 수 있어야 한다.

</div>

??? success "연습문제 3 풀이"
    `AdaptiveAvgPool1d(1)`이 길이 축을 1로 눌러 주므로, 어떤 길이의 순차열이 들어와도 분류기가 받는 벡터의 크기는 채널 수로 고정된다. 이것이 길이에 얽매이지 않는 열쇠이다.

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

    매개변수를 손으로 세면 $(1 \cdot 32 \cdot 5 + 32) + (32 \cdot 64 \cdot 5 + 64) + (64 \cdot 5 + 5) = 192 + 10304 + 325 = 10821$로 맞는다. 길이 300은 어디에도 들어가지 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
핵 크기가 $k$인 `nn.Conv1d`를 같은 입력에 적용한 완전 연결층과 견주어라. 매개변수는 얼마나 줄어드는가?

</div>

??? success "연습문제 4 풀이"
    길이가 $L$이고 채널이 $C_{\text{in}}$개인 입력을 받아 자리마다 $C_{\text{out}}$개의 값을 내는 경우로 견주자. 완전 연결층은 입력 $L \cdot C_{\text{in}}$개를 모두 보므로 자리 하나당 $L \cdot C_{\text{in}} \cdot C_{\text{out}}$개의 가중치가 필요하다. Conv1d는 창 하나만 보고 그 창을 모든 자리에서 함께 쓰므로 $k \cdot C_{\text{in}} \cdot C_{\text{out}}$개면 된다. 줄어드는 비는 $L/k$이다. $L=1000, k=5$이면 가중치 공유 덕분에 매개변수가 200분의 1이 된다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
`CausalConv1d`가 핵 크기 $k$, 팽창률 $d$에 대해 언제나 입력과 같은 길이를 낸다는 것을 2절의 출력 길이 식으로 보여라. 보폭은 $s = 1$이다.

</div>

??? success "연습문제 5 풀이"
    덧대기가 $p = d(k-1)$이므로 $s = 1$일 때

    $$L_{out} = \frac{L + 2d(k-1) - d(k-1) - 1}{1} + 1 = L + d(k-1)$$

    이다. 즉 합성곱이 오른쪽으로 $d(k-1)$만큼 길어진 것을 내놓는다. `forward`가 `out[:, :, :-self.padding]`으로 뒤쪽 $d(k-1)$개를 잘라 내므로 최종 길이는 다시 $L$이 된다.

    5절의 $k = 3$, $d = 1$이면 $p = 2$이고 $L_{out} = 10 + 2 = 12$, 잘라 낸 뒤 10이다. 잘라 낸 자리는 오른쪽 덧대기로 만들어진 0에 기댄 출력, 곧 미래를 보게 될 뻔한 자리들이다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
`TCNBlock`을 $L$개 쌓고 블록마다 팽창률을 $2^i$로 둘 때(핵 크기 $k$, 보폭 1) 수용 영역의 닫힌 식을 유도하라. $k=3$, $L=8$에 대해 값을 구하고, 블록당 합성곱을 하나로 잘못 세면 어떤 값이 나오는지도 적어라.

</div>

??? success "연습문제 6 풀이"
    보폭이 1인 층을 쌓을 때 수용 영역은 층마다 $d(k-1)$씩 **더해진다**. `TCNBlock` 하나에는 팽창률이 같은 합성곱이 둘($\texttt{conv1}$, $\texttt{conv2}$) 들어 있으므로 블록 하나가 $2 d (k-1)$을 더한다. 따라서

    $$\text{RF} = 1 + 2(k-1)\sum_{i=0}^{L-1} 2^{i} = 1 + 2(k-1)\left(2^{L} - 1\right)$$

    이다. $k = 3$, $L = 8$이면 $\sum_{i=0}^{7} 2^i = 2^8 - 1 = 255$이므로

    $$\text{RF} = 1 + 2 \cdot 2 \cdot 255 = 1021$$

    이다. 블록당 합성곱을 하나로 세면 $1 + (k-1)(2^L - 1) = 1 + 2 \cdot 255 = 511$이 나온다. 6절의 코드가 기울기를 흘려 잰 값이 1021이므로 511은 실제의 절반쯤을 말하는 셈이고, 이 차이는 모형이 볼 수 있다고 믿는 문맥의 길이를 두 배로 잘못 잡게 만든다.

---

## 정리하며

| 항목 | 설명 |
|--------|-------------|
| **연산** | 한 공간 차원을 따라 미끄러지는 내적 (핵을 뒤집지 않는 상호상관) |
| **입력 모양** | $(N, C_{in}, L)$: 배치, 채널, 순차열 길이 |
| **출력 크기** | $\lfloor (L + 2p - d(k-1) - 1) / s \rfloor + 1$ |
| **인과성** | 왼쪽에만 덧대어 미래 정보가 새지 않게 한다 |
| **팽창** | 매개변수는 그대로 두고 수용 영역이 지수적으로 넓어진다 |
| **행렬 형태** | 퇴플리츠 행렬이며, 전치하면 전치 합성곱이 된다 |

더 넓은 맥락은 [합성곱](convolution.md)(2차원과 상호상관의 구분), [팽창 합성곱](dilated_convolutions.md)(팽창률 설계), [수용 영역](receptive_field.md)(층을 쌓을 때의 일반 공식)에서 이어 볼 수 있다.

**참고 문헌**

1. van den Oord, A., et al. (2016). "WaveNet: A Generative Model for Raw Audio." *arXiv preprint arXiv:1609.03499*.

2. Bai, S., Kolter, J. Z., & Koltun, V. (2018). "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling." *arXiv preprint arXiv:1803.01271*.

3. Lea, C., et al. (2017). "Temporal Convolutional Networks for Action Segmentation and Detection." *CVPR*.

4. Dumoulin, V., & Visin, F. (2016). "A guide to convolution arithmetic for deep learning." *arXiv preprint arXiv:1603.07285*.
