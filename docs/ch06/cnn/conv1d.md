# 1차원 합성곱

2차원 합성곱이 이미지 처리의 일꾼이라면, **1차원 합성곱**은 시계열, 음향 신호, 텍스트 순차열, 금융 데이터 같은 순차 데이터를 다룬다. 학습 가능한 핵을 하나의 공간(시간) 차원을 따라 미끄러뜨리며 자리마다 지역적인 무늬를 뽑아낸다.

1차원 합성곱은 WaveNet이나 시간 합성곱 신경망(TCN) 같은 구조의 바탕이며, 퀀트 금융에서 가격 순차열, 호가창 스냅숏을 비롯한 시간 신호를 다루는 데 널리 쓰인다.

---

## 1. 수학적 정식화

### 단일 채널 1차원 합성곱

1차원 입력 $\mathbf{x} \in \mathbb{R}^{n}$과 핵 $\mathbf{w} \in \mathbb{R}^{k}$에 대해 다음과 같다.

$$y[i] = \sum_{j=0}^{k-1} x[i + j] \cdot w[j]$$

(덧대기가 없을 때) 출력 크기는 $n - k + 1$이다.

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

```python
import torch
import torch.nn as nn

# Conv1d의 서명
conv1d = nn.Conv1d(
    in_channels,    # 입력 채널의 수
    out_channels,   # 출력 채널(필터)의 수
    kernel_size,    # 합성곱 핵의 크기
    stride=1,       # 합성곱의 보폭
    padding=0,      # 양쪽에 더하는 0 덧대기
    dilation=1,     # 핵 원소 사이의 간격
    groups=1,       # 막힌 연결의 수
    bias=True,      # 학습 가능한 편향 더하기
    padding_mode='zeros'  # 'zeros', 'reflect', 'replicate', 'circular'
)
```

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

### 다채널 예제

```python
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

이는 기울기와 **뒤집은 핵**의 **온전한 합성곱**과 같다.

$$\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{y}} *_{full} \text{flip}(\mathbf{w})$$

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
    
    # 입력에 대한 기울기: 뒤집은 핵과의 온전한 합성곱
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
```

**출력:**

```
Analytical grad_w: [-3.76229763 -3.75680121 -0.74651747]
Numerical grad_w:  [-3.76229763 -3.75680121 -0.74651747]
Match: True
```

---

## 5. 인과 합성곱

많은 시계열 응용에서 모델은 미래를 들여다보아서는 안 된다. 시각 $t$의 출력은 시각이 $t$ 이하인 입력에만 기대야 한다. 이를 위해 **인과 합성곱**이 필요하다.

### 왼쪽에만 덧대기

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

# 인과성 확인
causal = CausalConv1d(1, 1, kernel_size=3)
x = torch.randn(1, 1, 10)
y = causal(x)
print(f"Input length: {x.shape[2]}, Output length: {y.shape[2]}")
# 둘 다 10: output[t]는 input[t-2], input[t-1], input[t]에 기댄다
```

**출력:**

```
Input length: 10, Output length: 10
```

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

# 수용 영역 계산:
# K=2이고 팽창률이 [1, 2, 4, ..., 512]일 때:
# RF = 1 + sum(d * (K-1)) = 1 + (1+2+4+...+512) = 표본 1024개
stack = build_wavenet_stack(64, kernel_size=2, num_layers=10)
print(f"Total layers: 10, Receptive field: 1024 samples")
print(f"At 16kHz audio: {1024/16000:.3f}s of context")
```

**출력:**

```
Total layers: 10, Receptive field: 1024 samples
At 16kHz audio: 0.064s of context
```

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
    - 팽창 인과 합성곱
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
        
        # 둘째 합성곱 블록
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

# 예: 특징 8개짜리 가격 이력에서 다음 시점의 수익률 예측
model = TCN(input_channels=8, hidden_channels=64, output_size=1,
            kernel_size=3, num_layers=8)

x = torch.randn(32, 8, 256)  # 표본 32개, 특징 8개, 시각 256개
pred = model(x)
print(f"Input: {x.shape}, Prediction: {pred.shape}")  # [32, 1]

# 수용 영역: i=0..7에 대해 1 + sum(2^i * (3-1)) = 1 + 2*(1+2+...+128) = 511
print(f"Receptive field: 511 time steps")
```

**출력:**

```
Input: torch.Size([32, 8, 256]), Prediction: torch.Size([32, 1])
Receptive field: 511 time steps
```

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

---

## 8. Conv1d와 Conv2d: 언제 무엇을 쓸까

| 기준 | Conv1d | Conv2d |
|-----------|--------|--------|
| 데이터의 짜임 | 순차열, 시계열 | 이미지, 공간 격자 |
| 입력 모양 | $(N, C, L)$ | $(N, C, H, W)$ |
| 핵이 미끄러지는 방향 | 1차원 (시간/위치) | 2차원 (높이 × 너비) |
| 흔한 핵 크기 | 3, 5, 7, 21 | 3×3, 5×5, 7×7 |
| 응용 예 | 음향, 자연어 처리, 금융 | 이미지, 영상 프레임, 변동성 곡면 |
| 매개변수 수 | $C_{out} \times C_{in} \times K$ | $C_{out} \times C_{in} \times K^2$ |

---

## 9. 핵심 정리

1. **Conv1d**는 모양이 $(N, C, L)$인 순차열을 다루며 시간 차원을 따라 핵을 미끄러뜨린다
2. **인과 합성곱**(왼쪽 덧대기 뒤 잘라내기)은 출력이 과거와 현재의 입력에만 기대게 한다
3. **팽창 인과 합성곱을 쌓으면** 수용 영역이 지수적으로 넓어진다. 핵 크기 2인 팽창 합성곱 10층이면 수용 영역이 1024가 된다
4. **TCN**은 팽창 인과 합성곱에 잔차 연결을 엮어 경쟁력 있는 순차열 모형을 만든다
5. 핵 크기가 서로 다른 **여러 규모의 합성곱**은 서로 다른 시간 지평의 무늬를 붙잡는다
6. 1차원 합성곱의 **역전파**는 뒤집은 핵과의 온전한 합성곱(= 전치 합성곱)이 된다

---

## 연습문제

**연습문제 1.**
2차원 합성곱보다 1차원 합성곱이 나은 때를 설명하고 응용 예 세 가지를 들어라.

??? success "연습문제 1 풀이"
    공간 구조가 1차원인 순차 데이터에는 1차원 합성곱이 낫다. (1) 시계열 예측, (2) 음향·음성 처리, (3) 자연어 텍스트 분류(글자 단위 또는 낱말 단위)가 그 예이다. 핵을 한 방향으로만 미끄러뜨리므로 2차원보다 계산이 싸다.

---

**연습문제 2.**
입력 길이가 100, 핵 크기가 5, 보폭이 2, 덧대기가 1인 1차원 합성곱의 출력 길이를 계산하라.

??? success "연습문제 2 풀이"
    출력 길이 $= \lfloor(L_{\text{in}} + 2P - K) / S\rfloor + 1 = \lfloor(100 + 2 - 5)/2\rfloor + 1 = \lfloor 97/2 \rfloor + 1 = 48 + 1 = 49$.

---

**연습문제 3.**
시계열 분류를 위한 1차원 CNN을 PyTorch로 구현하라.

??? success "연습문제 3 풀이"
    ```python
    model = nn.Sequential(
        nn.Conv1d(1, 32, kernel_size=5, padding=2), nn.ReLU(),
        nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, num_classes)
    )
    ```

---

**연습문제 4.**
핵 크기가 $k$인 `nn.Conv1d`을 같은 입력에 적용한 완전 연결층과 견주어라. 매개변수는 얼마나 줄어드는가?

??? success "연습문제 4 풀이"
    길이가 $L$이고 채널이 $C_{\text{in}}$개인 입력에 대한 완전 연결층은 매개변수가 $L \cdot C_{\text{in}} \cdot C_{\text{out}}$개이다. Conv1d은 $k \cdot C_{\text{in}} \cdot C_{\text{out}}$개이다. 줄어드는 비는 $L/k$이다. $L=1000, k=5$이면 가중치 공유 덕분에 매개변수가 200분의 1이 된다.

## 정리하며

| 항목 | 설명 |
|--------|-------------|
| **연산** | 한 공간 차원을 따라 미끄러지는 내적 |
| **입력 모양** | $(N, C_{in}, L)$: 배치, 채널, 순차열 길이 |
| **출력 크기** | $\lfloor (L + 2p - d(k-1) - 1) / s \rfloor + 1$ |
| **인과성** | 왼쪽에만 덧대어 미래 정보가 새지 않게 한다 |
| **팽창** | 매개변수는 그대로 두고 수용 영역이 지수적으로 넓어진다 |
| **행렬 형태** | 퇴플리츠 행렬이며, 전치하면 전치 합성곱이 된다 |

**참고 문헌**

1. van den Oord, A., et al. (2016). "WaveNet: A Generative Model for Raw Audio." *arXiv preprint arXiv:1609.03499*.

2. Bai, S., Kolter, J. Z., & Koltun, V. (2018). "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling." *arXiv preprint arXiv:1803.01271*.

3. Lea, C., et al. (2017). "Temporal Convolutional Networks for Action Segmentation and Detection." *CVPR*.

4. Dumoulin, V., & Visin, F. (2016). "A guide to convolution arithmetic for deep learning." *arXiv preprint arXiv:1603.07285*.
