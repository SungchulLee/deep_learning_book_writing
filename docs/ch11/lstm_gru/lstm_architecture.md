# LSTM 구조

LSTM은 셀 상태와 게이트를 이용해 기본 RNN의 장기 의존성 학습을 돕는 순환 뉴럴 넷트웍이다. 그래디언트 소실을 완전히 제거하는 구조는 아니며, 현재 흔히 사용하는 형태에는 초기 LSTM 이후 도입된 망각 게이트가 포함된다.

셀 상태에는 이전 값을 성분별로 보존하고 새 값을 더하는 경로가 있다. 게이트는 이 경로의 보존·갱신 비율을 학습한다. 어느 길이까지 정보를 유지하는지는 과제와 파라메터에 따라 달라지며 수백·수천 시점을 보장하지 않는다.

**구조의 핵심 착상:** LSTM은 상태 벡터 두 개를 지닌다.

- **은닉 상태** $h_t$: 셀의 출력으로 쓰이는 상태다
- **셀 상태** $c_t$: 게이트로 보존·갱신하는 내부 상태다

학습 가능한 게이트 세 개가 정보의 흐름을 다스린다.

- **망각 게이트** $f_t$: 셀 상태에서 무엇을 버릴지
- **입력 게이트** $i_t$: 어떤 새 정보를 담을지
- **출력 게이트** $o_t$: 셀 상태에서 무엇을 내보낼지

---

## 1. 그래디언트 소실 문제

기본 tanh RNN에서 $J_k=\operatorname{diag}(1-h_k^2)W_{hh}$라 두면 다음과 같다.

$$\frac{\partial h_t}{\partial h_0}=J_tJ_{t-1}\cdots J_1$$

마지막 로스의 열벡터 그래디언트는 이 곱의 전치로 앞쪽 상태에 전달된다. 각 야코비안의 노름이 공통 상수 $q<1$ 이하이면 장거리 기여는 지수적으로 줄어든다. 반면 순환 행렬의 최대 특잇값이 1보다 크다는 사실만으로 실제 그래디언트 폭발이 보장되는 것은 아니다.

LSTM의 셀 상태는 다음처럼 갱신된다.

$$c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t$$

게이트와 후보를 고정한 직접 경로의 야코비안은 $\operatorname{diag}(f_t)$다. 이 경로에는 기본 RNN처럼 매 시점 순환 행렬과 tanh 도함수를 반드시 곱하는 과정이 없다. 망각 게이트가 1에 가까우면 전달을 유지하기 쉬워진다. 다만 전체 미분에는 은닉 상태를 거쳐 게이트와 후보에 영향을 주는 경로도 포함되므로, 전체 그래디언트가 망각 게이트의 곱과 항상 같지는 않다. 1보다 작은 값의 곱은 긴 시간에 걸쳐 여전히 줄어들 수 있다.

## 2. LSTM 셀의 구조

LSTM 셀에는 주요 부품이 네 가지 있다. 게이트 세 개(망각, 입력, 출력)와 셀 상태 갱신 장치이다.

### 온전한 수식

시각 $t$에서 입력 $x_t \in \mathbb{R}^{d}$, 이전 은닉 상태 $h_{t-1} \in \mathbb{R}^{n}$, 이전 셀 상태 $c_{t-1} \in \mathbb{R}^{n}$이 주어졌을 때 다음과 같다.

**망각 게이트** — 셀 상태에서 어떤 정보를 버릴지 정한다.

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

$f_t \approx 0$이면 잊고, $f_t \approx 1$이면 기억한다.

**입력 게이트** — 어떤 새 정보를 담을지 정한다.

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

**셀 후보** — 셀 상태에 더할 후보 값을 만든다.

$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

**셀 상태 갱신** — (망각 게이트가 조절한) 옛 기억과 (입력 게이트가 조절한) 새 정보를 엮는다.

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

**출력 게이트** — 셀 상태를 기반으로 무엇을 내보낼지 정한다.

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

**은닉 상태** — 셀 상태를 걸러 낸 것이다.

$$h_t = o_t \odot \tanh(c_t)$$

여기서:

- $\sigma$은 시그모이드 함수 $\sigma(x) = \frac{1}{1 + e^{-x}}$이다
- $\odot$은 성분별(아다마르) 곱을 뜻한다
- $[h_{t-1}, x_t]$은 피처 차원을 따라 이어 붙인 것을 뜻한다
- $W_f, W_i, W_c, W_o \in \mathbb{R}^{n \times (n+d)}$은 웨이트 행렬이다
- $b_f, b_i, b_c, b_o \in \mathbb{R}^{n}$은 바이어스 벡터이다

### 왜 하필 이 비선형인가

엑티베이션 함수의 선택에는 뜻이 있다.

| 부품 | 활성화 | 범위 | 근거 |
|-----------|------------|-------|-----------|
| 게이트 ($f_t, i_t, o_t$) | 시그모이드 | $(0, 1)$ | 부드러운 이진 스위치. "꺼짐"과 "켜짐" 사이를 매끄럽게 잇는다 |
| 셀 후보 ($\tilde{c}_t$) | tanh | $(-1, 1)$ | 0을 중심으로 하며 양수와 음수 갱신을 모두 허용한다 |
| 출력 변환 | tanh | $(-1, 1)$ | 셀 상태를 유계 값으로 변환한다 |

**게이트에 시그모이드를 쓰는 이유:** 각 성분의 계수를 0과 1 사이로 제한하기 위해서다. 매끄러운 함수이지만 포화하면 도함수가 매우 작아질 수 있다. 실제로 완전한 이진 스위치를 계산하는 것은 아니다.

**후보와 출력에 tanh를 쓰는 이유:** 후보를 양수·음수의 유계 값으로 만들고 셀 상태를 유계 출력으로 변환한다. 후보의 크기가 제한되어도 값을 반복해서 더하는 셀 상태 자체의 크기가 모든 시간에 걸쳐 일정 범위로 제한된다는 보장은 없다.

---

## 3. 정보 흐름 그려 보기

LSTM에는 나란한 통로가 두 개 있다고 볼 수 있다. 셀 하나를 통째로 펼치면 그 둘이 한 그림에 들어온다.

![LSTM 셀 하나를 펼친 그림. 맨 위에 셀 상태가 지나는 가로줄이 띠로 강조되어 있고 그 위에는 성분별 곱 마디와 덧셈 마디만 놓여 있다. 아래쪽에는 이전 은닉 상태와 입력을 이어 붙인 회색 버스가 지나가고 거기에서 망각, 입력, 후보, 출력의 네 묶음이 갈라져 나온다. 망각 묶음은 곱 마디로 올라가고, 입력과 후보 묶음은 서로 곱해진 뒤 덧셈 마디로 올라가며, 셀 상태는 덧셈 마디보다 오른쪽에서 갈라져 tanh를 지나 출력 묶음과 곱해져 은닉 상태가 된다](figures/lstm_cell_flow.svg)

**어느 마디가 띠 위에 올라와 있는지**를 먼저 보면 된다. 띠 위에 있는 것은 곱 마디와 덧셈 마디 둘뿐이고, 네 묶음이 하는 일은 모두 띠 아래에서 끝난다. 망각 묶음만 곱 마디로 곧장 올라가 **옛 값에 직접 손을 대고**, 입력과 후보는 아래에서 먼저 서로 곱해진 다음 그 결과 하나만 덧셈 마디로 올라간다. 출력 묶음은 띠에 닿지도 않는다 — 셀 상태를 바꾸는 것이 아니라 내보낼 때 거르기만 하기 때문이다.

갈라지는 자리도 눈여겨볼 만하다. $h_t$로 가는 가지는 덧셈 마디보다 **오른쪽**에서 나오므로, 은닉 상태가 보는 것은 이미 갱신이 끝난 $c_t$다. 그리고 그 $h_t$는 다음 시각에 회색 버스로 되돌아와 네 묶음을 다시 계산한다. 띠가 곧 전체 그래디언트가 아닌 까닭이 이 되돌아오는 길이다 — 띠는 1절에서 말한 **직접 경로 하나**를 그린 것이고, 전체 미분에는 이 가지를 거치는 항이 함께 들어온다.

### 셀 상태 통로 (기억의 고속도로)

셀 상태의 직접 갱신 경로는 성분별 곱과 덧셈이다. 게이트를 계산하는 경로에는 행렬 곱과 활성화가 있으므로 LSTM 전체에 그런 연산이 없다는 뜻은 아니다.

### 은닉 상태 통로 (출력)

은닉 상태는 셀 상태를 걸러 변형한 모습이다. 다음에 쓰이므로 비선형을 거친다.

1. 지금의 출력이나 예측을 계산하기
2. 다음 시각의 문 계산에 정보를 주기

**설계의 착상:** 셀 상태는 날 정보를 담고, 은닉 상태는 그 정보를 과제에 맞게 보여 준다.

---

## 4. 차원 분석

입력 차원이 $d$이고 은닉 차원이 $n$인 LSTM에 대해 다음과 같다.

| 부품 | 모양 | 설명 |
|-----------|-------|-------------|
| $x_t$ | $(d,)$ | 입력 벡터 |
| $h_{t-1}, h_t$ | $(n,)$ | 은닉 상태 |
| $c_{t-1}, c_t$ | $(n,)$ | 셀 상태 |
| $[h_{t-1}, x_t]$ | $(n+d,)$ | 이어 붙인 입력 |
| $W_f, W_i, W_c, W_o$ | $(n, n+d)$ | 웨이트 행렬 |
| $b_f, b_i, b_c, b_o$ | $(n,)$ | 바이어스 벡터 |
| $f_t, i_t, o_t$ | $(n,)$ | 문의 엑티베이션 |
| $\tilde{c}_t$ | $(n,)$ | 셀 후보 |

### 파라메터의 수

**파라메터 총수:**

$$\text{Parameters} = 4 \times [n \times (n + d) + n] = 4n^2 + 4nd + 4n = 4n(n + d + 1)$$

**예:** $d = 100$(입력)이고 $n = 256$(은닉 차원)이면 다음과 같다.

$$\text{Parameters} = 4 \times 256 \times (256 + 100 + 1) = 4 \times 256 \times 357 = 365{,}568$$

### 다른 구조와 비교

| 구조 | 파라메터 | RNN에 대한 비 |
|--------------|------------|-----------------|
| 기본 RNN | $n^2 + nd + n$ | 1배 |
| GRU | $3(n^2 + nd + n)$ | 3배 |
| LSTM | $4(n^2 + nd + n)$ | 4배 |

이 식과 표는 네 계산 묶음마다 바이어스 하나를 두는 규약이다. PyTorch 기본 LSTM은 입력·순환 바이어스를 따로 두므로 단일 레이어·방향의 파라메터는 `4n(n+d+2)`개다. 예시 d=100,n=256에서는 366592개이며, 위 식의 365568개와 구분해야 한다. 셀 후보를 포함한 네 계산 묶음을 게이트 네 개라고 부르는 것은 부정확하다.

!!! tip "계산의 트레이드오프"
    같은 입력·은닉 차원과 바이어스 규약에서 LSTM의 순환 부분은 기본 RNN의 4배다. 성능 향상이나 더 작은 은닉 차원으로의 대체 가능성은 과제에서 확인해야 하며, 모델 전체나 실행 시간의 4배를 뜻하지는 않는다.

---

## 5. PyTorch로 밑바닥부터 구현하기

### LSTM 셀

```python
import torch
import torch.nn as nn
import numpy as np

class LSTMCell(nn.Module):
    """
    러닝을 위해 손수 만든 LSTM 세포 구현.
    
    계산 효율을 위해 웨이트 행렬을 합쳐 표준 LSTM 식을 구현한다.
    
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 효율을 위해 웨이트를 합침: [W_i, W_f, W_c, W_o]를 쌓았다
        # 그러면 모든 문을 행렬 곱 한 번으로 계산할 수 있다
        # 모양: 입력 웨이트는 (4 * hidden_size, input_size)
        # 모양: 숨은 웨이트는 (4 * hidden_size, hidden_size)
        self.weight_ih = nn.Parameter(
            torch.randn(4 * hidden_size, input_size) / np.sqrt(input_size)
        )
        self.weight_hh = nn.Parameter(
            torch.randn(4 * hidden_size, hidden_size) / np.sqrt(hidden_size)
        )
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))
        
        # 망각 문의 바이어스를 1로 초기화 (그래디언트의 흐름에 매우 중요)
        self._init_forget_gate_bias()
    
    def _init_forget_gate_bias(self):
        """
        망각 문의 바이어스를 1로 초기화한다.
        
        그러면 학습 초반에 모델이 기본적으로 기억하게 되어, 무엇을 잊어야
        할지 배우기 전에 그래디언트가 일찍 사라지는 것을 막는다.
        
        """
        with torch.no_grad():
            # 문의 순서: [입력, 망각, 세포, 출력]
            # 망각 문의 바이어스는 두 번째 사분면에 있다
            n = self.hidden_size
            self.bias[n:2*n].fill_(1.0)
    
    def forward(
        self, 
        x: torch.Tensor, 
        state: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        한 시각의 순전파.
        
        인수:
            x: 모양이 (배치 크기, input_size)인 입력 텐서
            state: (h_prev, c_prev)의 쌍이며 각각 (배치 크기, hidden_size)
                   None이면 0으로 초기화한다.
        
        반환값:
            h_new: 새 숨은 상태 (배치 크기, hidden_size)
            c_new: 새 세포 상태 (배치 크기, hidden_size)
        """
        batch_size = x.size(0)
        
        # 상태가 주어지지 않았으면 초기화
        if state is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_prev, c_prev = state
        
        # 효율을 위해 행렬 곱 한 번으로 모든 문 계산
        # gates = W_ih @ x + W_hh @ h_prev + bias
        gates = (x @ self.weight_ih.t() + 
                 h_prev @ self.weight_hh.t() + 
                 self.bias)
        
        # 문마다 나누기
        n = self.hidden_size
        i_gate = torch.sigmoid(gates[:, 0*n:1*n])      # 입력 문
        f_gate = torch.sigmoid(gates[:, 1*n:2*n])      # 망각 문
        c_tilde = torch.tanh(gates[:, 2*n:3*n])        # 세포 후보
        o_gate = torch.sigmoid(gates[:, 3*n:4*n])      # 출력 문
        
        # 세포 상태 갱신: c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
        c_new = f_gate * c_prev + i_gate * c_tilde
        
        # 숨은 상태: h_t = o_t ⊙ tanh(c_t)
        h_new = o_gate * torch.tanh(c_new)
        
        return h_new, c_new

class LSTM(nn.Module):
    """
    순차열을 처리하는 온전한 LSTM 레이어.
    
    여러 레이어를 쌓는 것을 지원하며 뒤따르는 처리를 위해 모든 숨은 상태를
    돌려준다.
    """
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 레이어마다 세포 만들기
        # 첫 레이어는 input_size를, 뒤 레이어는 hidden_size를 받는다
        self.cells = nn.ModuleList([
            LSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
        # 레이어 사이의 드롭아웃 (마지막 레이어 뒤에는 없다)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(
        self, 
        x: torch.Tensor, 
        state: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        순차열 전체를 LSTM으로 처리한다.
        
        인수:
            x: 입력 순차열 (배치 크기, seq_len, input_size)
            state: 처음 상태의 쌍:
                   - h_0: (num_layers, 배치 크기, hidden_size)
                   - c_0: (num_layers, 배치 크기, hidden_size)
        
        반환값:
            output: 맨 위 레이어의 모든 숨은 상태 (배치 크기, seq_len, hidden_size)
            (h_n, c_n): 모든 레이어의 마지막 상태이며 입력 상태와 모양이 같다
        """
        batch_size, seq_len, _ = x.size()
        
        # 모든 레이어의 상태 초기화
        if state is None:
            h = [torch.zeros(batch_size, self.hidden_size, device=x.device) 
                 for _ in range(self.num_layers)]
            c = [torch.zeros(batch_size, self.hidden_size, device=x.device) 
                 for _ in range(self.num_layers)]
        else:
            h = [state[0][i] for i in range(self.num_layers)]
            c = [state[1][i] for i in range(self.num_layers)]
        
        # 순차열을 시각마다 처리
        outputs = []
        for t in range(seq_len):
            layer_input = x[:, t, :]
            
            # 레이어마다 통과
            for layer_idx, cell in enumerate(self.cells):
                h[layer_idx], c[layer_idx] = cell(
                    layer_input, (h[layer_idx], c[layer_idx])
                )
                layer_input = h[layer_idx]
                
                # 레이어 사이에 드롭아웃 적용 (마지막 레이어 뒤에는 없다)
                if self.dropout is not None and layer_idx < self.num_layers - 1:
                    layer_input = self.dropout(layer_input)
            
            outputs.append(h[-1])  # 맨 위 레이어의 출력 모으기
        
        # 출력과 상태 쌓기
        output = torch.stack(outputs, dim=1)  # (배치, seq_len, hidden)
        h_n = torch.stack(h, dim=0)           # (num_layers, 배치, hidden)
        c_n = torch.stack(c, dim=0)           # (num_layers, 배치, hidden)
        
        return output, (h_n, c_n)
```

### PyTorch와 비교하여 확인하기

직접 구현은 바이어스 하나를 쓰므로 내장 모듈의 두 바이어스 합을 맞춰 비교한다. 아래 검증 함수는 정의되어 있지만 호출은 주석 처리되어 있다. 단일 레이어·드롭아웃 없는 순전파 비교이며, 모든 설정과 역전파의 동등성을 검증하는 코드는 아니다.

```python
def verify_implementation():
    """우리 구현이 PyTorch와 맞는지 확인한다."""
    torch.manual_seed(42)
    
    batch_size, seq_len, input_size, hidden_size = 4, 10, 8, 16
    
    # 우리 구현
    our_lstm = LSTM(input_size, hidden_size, num_layers=1)
    
    # PyTorch의 구현
    torch_lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
    
    # 웨이트 옮기기 (순서가 다른 점을 감안)
    with torch.no_grad():
        # PyTorch는 문의 순서를 [i, f, g, o]로 두며 우리도 같게 쓴다
        torch_lstm.weight_ih_l0.copy_(our_lstm.cells[0].weight_ih)
        torch_lstm.weight_hh_l0.copy_(our_lstm.cells[0].weight_hh)
        torch_lstm.bias_ih_l0.copy_(our_lstm.cells[0].bias)
        torch_lstm.bias_hh_l0.zero_()  # 우리는 바이어스를 합치고 PyTorch는 나눈다
    
    # 시험 입력
    x = torch.randn(batch_size, seq_len, input_size)
    
    # 순전파
    our_output, (our_h, our_c) = our_lstm(x)
    torch_output, (torch_h, torch_c) = torch_lstm(x)
    
    # 비교
    print(f"Output close: {torch.allclose(our_output, torch_output, atol=1e-5)}")
    print(f"Hidden close: {torch.allclose(our_h, torch_h, atol=1e-5)}")
    print(f"Cell close: {torch.allclose(our_c, torch_c, atol=1e-5)}")

# verify_implementation()
```

---

## 6. PyTorch 내장 LSTM 쓰기

```python
import torch
import torch.nn as nn

# LSTM 레이어 만들기
lstm = nn.LSTM(
    input_size=100,      # 입력 피처의 차원
    hidden_size=256,     # 숨은 상태와 세포 상태의 차원
    num_layers=2,        # 쌓은 LSTM 레이어의 수
    batch_first=True,    # 입력의 모양: (배치, seq, 피처)
    dropout=0.3,         # 레이어 사이의 드롭아웃 (num_layers > 1일 때만)
    bidirectional=False  # 순방향만 처리
)

# 입력: (배치 크기, sequence_length, input_size)
x = torch.randn(32, 50, 100)

# 순전파
output, (h_n, c_n) = lstm(x)

print(f"Output shape: {output.shape}")     # (32, 50, 256)
print(f"Final hidden: {h_n.shape}")        # (2, 32, 256) — 레이어마다 하나
print(f"Final cell: {c_n.shape}")          # (2, 32, 256) — 레이어마다 하나

# 뒤따르는 과제를 위해 출력에 접근
last_output = output[:, -1, :]    # 마지막 시각: (32, 256)
last_hidden = h_n[-1]             # 마지막 레이어의 숨은 상태: (32, 256)
# 참고: 단방향 LSTM에서는 last_output == last_hidden이다
```

**출력:**

```
Output shape: torch.Size([32, 50, 256])
Final hidden: torch.Size([2, 32, 256])
Final cell: torch.Size([2, 32, 256])
```

---

## 7. 실전 응용

### 시퀀스 분류

`padding_idx`는 임베딩의 PAD 행을 처리하는 설정이다. LSTM의 패딩 시점 상태 갱신까지 생략하지 않으므로 실제 길이를 반영한 최종 상태 선택이나 패킹이 필요하다.

```python
class LSTMClassifier(nn.Module):
    """
    순차열 분류를 위한 LSTM (감성 분석, 스팸 판정 등).
    
    마지막 숨은 상태를 순차열 전체의 차원 고정 표현으로 삼아
    분류한다.
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        embed_dim: int, 
        hidden_size: int, 
        num_classes: int,
        num_layers: int = 2, 
        dropout: float = 0.5,
        padding_idx: int = 0
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        인수:
            x: 토큰 색인 (배치 크기, seq_len)
        
        반환값:
            logits: 분류 로짓 (배치 크기, num_classes)
        """
        # 토큰 임베딩
        embedded = self.embedding(x)  # (배치, seq_len, embed_dim)
        embedded = self.dropout(embedded)
        
        # LSTM 순전파
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        
        # 분류에 마지막 숨은 상태 쓰기
        # h_n의 모양: (num_layers, 배치, hidden_size)
        # 마지막 레이어의 숨은 상태 가져오기
        final_hidden = h_n[-1]  # (배치, hidden_size)
        final_hidden = self.dropout(final_hidden)
        
        # 분류
        logits = self.fc(final_hidden)  # (배치, num_classes)
        
        return logits

# 사용 예
model = LSTMClassifier(
    vocab_size=30000,
    embed_dim=128,
    hidden_size=256,
    num_classes=2,  # 이진 분류
    num_layers=2,
    dropout=0.5
)

# 입력: 토큰 100개짜리 순차열 32개의 배치
x = torch.randint(0, 30000, (32, 100))
logits = model(x)  # (32, 2)
probs = torch.softmax(logits, dim=-1)
```

### 시퀀스 생성 (언어 모델)

웨이트 공유는 임베딩과 은닉 차원이 같을 때만 적용된다. 아래 `generate`는 프롬프트 전체를 처리한 후 마지막 프롬프트 토큰을 다시 입력하므로 첫 생성 분포가 의도한 문맥과 다르다. 올바른 순서는 프롬프트의 마지막 로짓에서 첫 새 토큰을 고르고, 이후 선택한 새 토큰을 한 번씩 입력하는 것이다. 빈 프롬프트, 양수 온도와 top_k 범위, 종료 토큰 처리도 별도로 필요하다.

```python
class LSTMLanguageModel(nn.Module):
    """
    자기회귀 텍스트 생성을 위한 LSTM.
    
    앞선 토큰이 주어졌을 때 다음 토큰을 예측하여, 되풀이 표본 추출로
    텍스트를 만들 수 있게 한다.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        tie_weights: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 숨은 상태를 어휘로 사영
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # 선택적으로 임베딩과 출력 웨이트를 묶기
        if tie_weights and embed_dim == hidden_size:
            self.fc.weight = self.embedding.weight
    
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        인수:
            x: 토큰 색인 (배치 크기, seq_len)
            hidden: 이전의 (h, c) 상태
        
        반환값:
            logits: 다음 토큰의 로짓 (배치 크기, seq_len, vocab_size)
            hidden: 갱신된 (h, c) 상태
        """
        embedded = self.dropout(self.embedding(x))
        lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)
        
        return logits, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device):
        """숨은 상태를 0으로 초기화한다."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h, c)
    
    @torch.no_grad()
    def generate(
        self,
        start_tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None
    ) -> list[int]:
        """
        자기회귀적으로 토큰을 만든다.
        
        인수:
            start_tokens: 처음 토큰의 색인 (seq_len,)
            max_new_tokens: 만들 새 토큰의 수
            temperature: 표본 추출의 온도 (높을수록 더 무작위)
            top_k: 정하면 가장 그럴듯한 상위 k개 토큰에서만 뽑는다
        
        반환값:
            만들어진 토큰 색인의 목록
        """
        self.eval()
        device = next(self.parameters()).device
        
        tokens = start_tokens.tolist()
        hidden = self.init_hidden(1, device)
        
        # 프롬프트 처리
        x = torch.tensor([tokens], device=device)
        _, hidden = self.forward(x, hidden)
        
        # 새 토큰 만들기
        current_token = torch.tensor([[tokens[-1]]], device=device)
        
        for _ in range(max_new_tokens):
            logits, hidden = self.forward(current_token, hidden)
            logits = logits[0, -1, :] / temperature  # (vocab_size,)
            
            # 선택적인 상위 k개 거르기
            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                probs = torch.softmax(top_k_logits, dim=-1)
                idx = torch.multinomial(probs, 1)
                next_token = top_k_indices[idx].item()
            else:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            
            tokens.append(next_token)
            current_token = torch.tensor([[next_token]], device=device)
        
        return tokens
```

### 시계열 예측

```python
class LSTMForecaster(nn.Module):
    """
    다변량 시계열 예측을 위한 LSTM.
    
    관측의 순차열이 주어지면 정해진 예측 지평의 미래 값을
    예측한다.
    """
    
    def __init__(
        self,
        input_size: int,     # 입력 피처의 수
        hidden_size: int,
        num_layers: int,
        output_size: int,    # 출력 피처의 수
        horizon: int = 1,    # 예측 지평 (앞을 내다보는 스텝 수)
        dropout: float = 0.2
    ):
        super().__init__()
        self.horizon = horizon
        self.output_size = output_size
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 예측 지평의 모든 스텝을 한꺼번에 예측
        self.fc = nn.Linear(hidden_size, output_size * horizon)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        인수:
            x: 입력 순차열 (배치 크기, seq_len, input_size)
        
        반환값:
            predictions: 예측 (배치 크기, horizon, output_size)
        """
        # 순차열 처리
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 예측에 마지막 숨은 상태 쓰기
        final_hidden = h_n[-1]  # (배치, hidden_size)
        
        # 모든 스텝 예측
        predictions = self.fc(final_hidden)  # (배치, output_size * horizon)
        predictions = predictions.view(-1, self.horizon, self.output_size)
        
        return predictions

# 예: 지난 20스텝으로 피처 3개의 다음 스텝5 예측
model = LSTMForecaster(
    input_size=3,
    hidden_size=128,
    num_layers=2,
    output_size=3,
    horizon=5
)

x = torch.randn(32, 20, 3)  # 표본 32개, 시각 20개, 피처 3개
predictions = model(x)       # (32, 5, 3)
```

---

## 8. 초기화와 학습의 모범 관행

### 웨이트 초기화

```python
def init_lstm_weights(lstm: nn.LSTM):
    """
    모범 관행에 따라 LSTM의 웨이트를 초기화한다.
    
    - 입력에서 숨은 상태로 가는 웨이트: 자비에르(글로럿) 균등
    - 숨은 상태끼리의 웨이트: 직교 (그래디언트의 노름을 지킨다)
    - 바이어스: 0, 다만 망각 문의 바이어스는 1
    """
    for name, param in lstm.named_parameters():
        if 'weight_ih' in name:
            # 입력에서 숨은 상태로 가는 웨이트: 자비에르 균등
            nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:
            # 숨은 상태끼리의 웨이트: 직교
            # 그러면 순환을 지나도 그래디언트의 크기가 지켜진다
            nn.init.orthogonal_(param)
        elif 'bias' in name:
            # 바이어스: 대체로 0
            nn.init.zeros_(param)
            # 다만 망각 문의 바이어스는 1로 둔다
            n = param.size(0) // 4
            param.data[n:2*n].fill_(1.0)
```

### 망각 게이트의 바이어스를 왜 1로 두는가

게이트의 활성화 전 값이 0 부근이면 시그모이드 출력은 0.5 부근이다. 망각 게이트의 유효 바이어스를 양수로 두면 초기 보존 계수를 키울 수 있다. 하지만 무작위 입력·웨이트가 항상 작은 값이라는 가정이나 모든 셀이 절반의 정보를 잊는다는 해석은 성립하지 않는다.

$b_f = 1$으로 두면 망각 게이트 시그모이드의 입력이 치우쳐 다음과 같이 된다.

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + 1) \approx \sigma(1) \approx 0.73$$

이 근사는 가중 입력이 0 부근일 때 유효하다. 위 초기화 함수는 PyTorch의 두 바이어스 모두에서 망각 부분을 1로 채우므로 유효 바이어스는 2다. 이 경우 0 입력 근사값은 sigmoid(2) 약 0.881이며 위 sigmoid(1) 예시와 다르다. 또한 쌓인 순환 웨이트 전체를 직교 초기화하는 것은 각 게이트 블록을 각각 직교 초기화하는 것과 다르며 전체 그래디언트를 보장하지 않는다.

### 그래디언트 자르기

```python
# 학습 중에는 폭발을 막으려고 그래디언트를 자른다
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
optimizer.step()
```

LSTM은 그래디언트 소실을 다루지만 특히 학습 초반에는 그래디언트 폭발을 겪을 수 있다. 그래디언트 자르기가 표준 관행이다.

### 러닝레이트 스케줄링

```python
# 검증 로스가 정체되면 러닝레이트를 줄인다
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=5
)

# 세대마다
scheduler.step(val_loss)
```

---

## 9. LSTM의 변형

### 망각 게이트와 입력 게이트 묶기

관찰: 옛 정보를 잊을 때는 대개 새 정보를 넣고, 그 반대도 마찬가지이다. 두 문을 묶으면 다음과 같다.

$$c_t = f_t \odot c_{t-1} + (1 - f_t) \odot \tilde{c}_t$$

이렇게 하면 파라메터가 줄고 이전 값과 후보에 대한 성분별 계수의 합이 1이라는 제약이 걸린다. 벡터의 노름이나 정보량이 보존된다는 뜻은 아니다.

### GRU (게이트가 있는 순환 단위)

GRU는 다음과 같이 LSTM을 간소화한다.

1. 셀 상태와 은닉 상태를 하나로 합친다
2. 망각 게이트와 입력 게이트를 "갱신 문" 하나로 엮는다
3. 후보 계산에 이전 상태를 얼마나 반영할지 조절하는 리셋 게이트를 둔다. 출력 게이트와 일대일로 대응하지는 않는다

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$$

$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$$

$$\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t])$$

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

**트레이드오프:** 파라메터가 적고(4배가 아니라 3배) 많은 과제에서 성능이 비슷하지만 기억을 다스리는 유연함은 덜하다.

---

## 10. 언제 LSTM을 쓸까

LSTM은 고정 차원의 상태를 유지하며 입력을 한 시점씩 처리할 수 있어 스트림 처리의 후보가 된다. 가변 길이 배치에는 길이 처리가 필요하며 양방향 모델은 전체 입력이 있어야 한다.

GRU, 1차원 CNN이나 트랜스포머와 비교할 때는 데이터, 필요한 문맥, 지연·메모리 제한과 학습 예산을 맞춘다. 시퀀스가 1000개를 넘는다는 이유만으로 특정 구조를 선택할 수는 없다. 모델 크기가 작은 데이터에서 과적합을 덜 일으키는지도 검증해야 한다.

## 11. 흔한 문제 잡기

### 증상과 점검

로스 정체, 같은 출력과 장거리 성능 저하는 데이터·타깃, 로스 규약, 초기화와 러닝레이트 등 여러 원인이 가능하다. 특정 증상만으로 원인을 확정하지 말고 입력 범위와 분할, 유한한 로스·그래디언트와 길이별 검증 성능을 확인한다.

### 진단 코드

아래 함수는 `torch.no_grad()` 안에서 backward를 호출하므로 의도한 그래디언트 검사를 수행할 수 없다. `model.train()`은 그래디언트 기록을 다시 켜지 않는다. 기록이 켜진 문맥에서 모델 반환값과 로스를 맞추고 기존 그래디언트를 초기화해야 한다. 평가 모드 복원도 필요하다.

```python
def diagnose_lstm(model, dataloader, device):
    """LSTM 학습을 진단한다."""
    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            
            # 그래디언트의 크기 확인
            model.train()
            output, _ = model(x)
            loss = output.sum()
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    print(f"{name}: grad_norm = {grad_norm:.6f}")
            
            break
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
LSTM의 식을 적고 각 문의 구실을 설명하라.

</div>

??? success "풀이"
    망각 게이트: $f_t = \sigma(W_f[h_{t-1}, x_t] + b_f)$ — 셀 상태에서 무엇을 버릴지.
    입력 게이트: $i_t = \sigma(W_i[h_{t-1}, x_t] + b_i)$ — 어떤 새 정보를 담을지.
    후보: $\tilde{c}_t = \tanh(W_c[h_{t-1}, x_t] + b_c)$ — 새 후보 값.
    세포 갱신: $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$.
    출력 게이트: $o_t = \sigma(W_o[h_{t-1}, x_t] + b_o)$ — 무엇을 내보낼지.
    은닉 상태: $h_t = o_t \odot \tanh(c_t)$.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
셀 상태의 직접 경로와 전체 그래디언트의 차이를 설명하라.

</div>

??? success "풀이"
    게이트·후보를 고정한 직접 경로는 $\operatorname{diag}(f_t)$다. 시간에 따른 직접 경로에는 그 대각 행렬들이 곱해진다. 전체 미분에는 이전 은닉 상태가 게이트와 후보에 미치는 경로도 있으므로 이 곱만으로 전체 그래디언트를 설명할 수 없다. 망각 값이 1에 가까워도 장거리 전달은 과제에서 확인해야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
입력 차원 300, 은닉 차원 512인 LSTM 셀의 파라메터 수를 바이어스 규약별로 구하라.

</div>

??? success "풀이"
    계산 묶음마다 바이어스 하나면 $4\times512\times(300+512+1)=1665024$개다. PyTorch 기본처럼 입력·순환 바이어스를 각각 두면 $4\times512\times814=1667072$개다. 후보를 포함한 네 묶음이며 출력 레이어와 임베딩은 제외한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
망각 게이트의 양수 바이어스 초기화가 하는 일과 코드에서 두 바이어스를 설정할 때의 주의점을 설명하라.

</div>

??? success "풀이"
    가중 입력이 0 부근일 때 유효 바이어스 1은 보존 계수를 sigmoid(1) 약 0.731로 만든다. 이는 초기 보존을 돕는 선택이지 그래디언트나 정보 유지를 보장하지 않는다. 내장 LSTM의 두 바이어스에 각각 1을 넣으면 합은 2가 되므로 하나에만 넣거나 합이 원하는 값이 되도록 나눠야 한다.

## 정리하며

LSTM은 셀 상태와 세 게이트로 보존·갱신·출력을 조절한다. 직접 셀 경로의 이점과 전체 그래디언트의 보장은 구분해야 한다. 내장 구현과 비교할 때 바이어스의 합과 상태 형태를 맞추고, 생성·패딩·진단 코드의 제약을 함께 확인한다.

**참고 문헌**

1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.

2. Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). Learning to Forget: Continual Prediction with LSTM. *Neural Computation*, 12(10), 2451-2471.

3. Greff, K., Srivastava, R. K., Koutník, J., Steunebrink, B. R., & Schmidhuber, J. (2017). LSTM: A Search Space Odyssey. *IEEE Transactions on Neural Networks and Learning Systems*, 28(10), 2222-2232.

4. Jozefowicz, R., Zaremba, W., & Sutskever, I. (2015). An Empirical Exploration of Recurrent Network Architectures. *ICML*.

5. Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *EMNLP*.
