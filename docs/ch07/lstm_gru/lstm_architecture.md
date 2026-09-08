# LSTM 구조

장단기 기억(LSTM) 신경망은 순차 데이터의 먼 거리 의존을 붙잡으려고 만든 특수한 순환 신경망이다. 1997년 Hochreiter와 Schmidhuber가 내놓았으며, 먼 시각에서 배우지 못하게 하는 기울기 소실 문제라는 기본 RNN의 근본적인 한계를 푼다.

LSTM의 핵심 혁신은 **세포 상태**에 있다. 정보가 거의 변형되지 않고 여러 시각을 가로질러 흐르는 전용 기억 통로이다. 이 "기억의 고속도로" 덕분에 LSTM은 수백, 때로는 수천 시각 동안 정보를 골라 기억할 수 있다.

**구조의 핵심 착상:** LSTM은 상태 벡터 두 개를 지닌다.

- **숨은 상태** $h_t$: 단기 기억이며 세포의 출력이다
- **세포 상태** $c_t$: 장기 기억이며 정보의 고속도로이다

학습 가능한 문 세 개가 정보의 흐름을 다스린다.

- **망각 문** $f_t$: 세포 상태에서 무엇을 버릴지
- **입력 문** $i_t$: 어떤 새 정보를 담을지
- **출력 문** $o_t$: 세포 상태에서 무엇을 내보낼지

---

## 1. 기울기 소실 문제

LSTM의 구조를 이해하기에 앞서 그것이 푸는 문제를 알아야 한다.

### 기본 RNN의 문제

기본 RNN에서 숨은 상태는 다음과 같이 갱신된다.

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

시간을 거슬러 가는 역전파(BPTT)에서 기울기는 이 순환을 타고 흐른다.

$$\frac{\partial h_t}{\partial h_0} = \prod_{k=1}^{t} \frac{\partial h_k}{\partial h_{k-1}} = \prod_{k=1}^{t} W_{hh}^\top \cdot \text{diag}(\tanh'(z_k))$$

**겹치는 문제가 둘 있다.**

1. **활성화의 포화:** $|\tanh'(x)| \leq 1$이고 등호는 $x=0$에서만 성립하므로, 포화된 활성화를 거듭 곱하면 기울기가 0으로 간다.

2. **$W_{hh}$의 스펙트럼 성질:** 가장 큰 특잇값이 $\sigma_{\max}(W_{hh}) < 1$이면 기울기가 지수적으로 사라지고, $\sigma_{\max}(W_{hh}) > 1$이면 폭발한다.

**결과:** 길이가 $T$인 순차열에서 시각 $t$이 시각 $T$의 손실에 이바지하는 기울기는 $\lambda < 1$일 때 $O(\lambda^{T-t})$으로 줄어든다. 먼 과거의 정보는 학습 중에 사실상 보이지 않는다.

### LSTM의 해법: 덧셈 갱신

LSTM은 곱셈 순환을 **덧셈** 세포 상태 갱신으로 바꾼다.

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

세포 상태 통로를 지나는 기울기는 다음과 같다.

$$\frac{\partial c_t}{\partial c_{t-1}} = f_t$$

$f_t \approx 1$이면(망각 문이 열리면) 기울기가 **그대로** 흐른다.

$$\frac{\partial \mathcal{L}}{\partial c_{t-k}} = \frac{\partial \mathcal{L}}{\partial c_t} \cdot \prod_{i=t-k}^{t-1} f_{i+1}$$

기울기가 가중치 행렬과 포화하는 활성화를 지나는 기본 RNN과 달리, LSTM에서는 1 가까이 머무를 수 있는 학습된 문을 지난다. 이것이 "일정 오차 회전목마"이며, 기울기가 수백 걸음을 사라지지 않고 흐를 수 있다.

!!! note "핵심 착상"
    LSTM이 기울기 문제를 아예 없애는 것은 아니다. 그것을 에두르는 **학습 가능한 우회로**를 줄 뿐이다. 신경망은 언제 기억할지($f_t \approx 1$)와 언제 잊을지($f_t \approx 0$)를 배워, 쓸모없는 정보는 버리면서 과제에 필요한 기울기는 흐르게 한다.

---

## 2. LSTM 세포의 구조

LSTM 세포에는 주요 부품이 네 가지 있다. 문 세 개(망각, 입력, 출력)와 세포 상태 갱신 장치이다.

### 온전한 수식

시각 $t$에서 입력 $x_t \in \mathbb{R}^{d}$, 이전 숨은 상태 $h_{t-1} \in \mathbb{R}^{n}$, 이전 세포 상태 $c_{t-1} \in \mathbb{R}^{n}$이 주어졌을 때 다음과 같다.

**망각 문** — 세포 상태에서 어떤 정보를 버릴지 정한다.

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

$f_t \approx 0$이면 잊고, $f_t \approx 1$이면 기억한다.

**입력 문** — 어떤 새 정보를 담을지 정한다.

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

**세포 후보** — 세포 상태에 더할 후보 값을 만든다.

$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

**세포 상태 갱신** — (망각 문이 조절한) 옛 기억과 (입력 문이 조절한) 새 정보를 엮는다.

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

**출력 문** — 세포 상태를 바탕으로 무엇을 내보낼지 정한다.

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

**숨은 상태** — 세포 상태를 걸러 낸 것이다.

$$h_t = o_t \odot \tanh(c_t)$$

여기서:

- $\sigma$은 시그모이드 함수 $\sigma(x) = \frac{1}{1 + e^{-x}}$이다
- $\odot$은 성분별(아다마르) 곱을 뜻한다
- $[h_{t-1}, x_t]$은 특징 차원을 따라 이어 붙인 것을 뜻한다
- $W_f, W_i, W_c, W_o \in \mathbb{R}^{n \times (n+d)}$은 가중치 행렬이다
- $b_f, b_i, b_c, b_o \in \mathbb{R}^{n}$은 편향 벡터이다

### 왜 하필 이 비선형인가

활성화 함수의 선택에는 뜻이 있다.

| 부품 | 활성화 | 범위 | 근거 |
|-----------|------------|-------|-----------|
| 문 ($f_t, i_t, o_t$) | 시그모이드 | $(0, 1)$ | 부드러운 이진 스위치. "꺼짐"과 "켜짐" 사이를 매끄럽게 잇는다 |
| 세포 후보 ($\tilde{c}_t$) | tanh | $(-1, 1)$ | 0을 중심으로 하며 양수와 음수 갱신을 모두 허용한다 |
| 출력 변환 | tanh | $(-1, 1)$ | 문을 지나기 전에 세포 상태를 정규화한다 |

**왜 문에 시그모이드를 쓰는가?** 문은 부드러운 이진 결정을 내려야 한다. 시그모이드는 스위치를 흉내 내면서도 어디서나 매끄러운 기울기를 준다.

**왜 내용에 tanh를 쓰는가?** 세포 상태는 시간이 지나며 쌓일 수 있다. tanh는 새로 더해지는 몫을 $[-1, 1]$으로 묶어 끝없이 커지는 것을 막으면서도 더하고 빼는 갱신을 모두 허용한다.

---

## 3. 정보 흐름 그려 보기

LSTM에는 나란한 통로가 두 개 있다고 볼 수 있다.

### 세포 상태 통로 (기억의 고속도로)

```
c_{t-1} ──[×f_t]──[+]── c_t ──→
                   ↑
            [×i_t]─┘
               ↑
           c̃_t (candidate)
```

세포 상태는 성분별 연산만 거친다. 주 통로에는 행렬 곱도 활성화 함수도 없다. 이것이 기울기를 지키는 "정보의 고속도로"이다.

### 숨은 상태 통로 (출력)

```
c_t ──[tanh]──[×o_t]── h_t ──→ output
```

숨은 상태는 세포 상태를 걸러 변형한 모습이다. 다음에 쓰이므로 비선형을 거친다.

1. 지금의 출력이나 예측을 계산하기
2. 다음 시각의 문 계산에 정보를 주기

**설계의 착상:** 세포 상태는 날 정보를 담고, 숨은 상태는 그 정보를 과제에 맞게 보여 준다.

---

## 4. 차원 분석

입력 차원이 $d$이고 숨은 차원이 $n$인 LSTM에 대해 다음과 같다.

| 부품 | 모양 | 설명 |
|-----------|-------|-------------|
| $x_t$ | $(d,)$ | 입력 벡터 |
| $h_{t-1}, h_t$ | $(n,)$ | 숨은 상태 |
| $c_{t-1}, c_t$ | $(n,)$ | 세포 상태 |
| $[h_{t-1}, x_t]$ | $(n+d,)$ | 이어 붙인 입력 |
| $W_f, W_i, W_c, W_o$ | $(n, n+d)$ | 가중치 행렬 |
| $b_f, b_i, b_c, b_o$ | $(n,)$ | 편향 벡터 |
| $f_t, i_t, o_t$ | $(n,)$ | 문의 활성값 |
| $\tilde{c}_t$ | $(n,)$ | 세포 후보 |

### 매개변수의 수

**매개변수 총수:**

$$\text{Parameters} = 4 \times [n \times (n + d) + n] = 4n^2 + 4nd + 4n = 4n(n + d + 1)$$

**예:** $d = 100$(입력)이고 $n = 256$(숨은 차원)이면 다음과 같다.

$$\text{Parameters} = 4 \times 256 \times (256 + 100 + 1) = 4 \times 256 \times 357 = 365{,}568$$

### 다른 구조와 견주기

| 구조 | 매개변수 | RNN에 대한 비 |
|--------------|------------|-----------------|
| 기본 RNN | $n^2 + nd + n$ | 1배 |
| GRU | $3(n^2 + nd + n)$ | 3배 |
| LSTM | $4(n^2 + nd + n)$ | 4배 |

4배로 늘어나는 까닭은 가중치 행렬이 네 개(망각, 입력, 세포, 출력)이고 저마다 RNN의 가중치 행렬 하나와 크기가 같기 때문이다.

!!! tip "계산의 맞바꿈"
    LSTM은 매개변수가 4배로 늘어나는 대신 기울기의 흐름이 훨씬 좋아진다. 실제로 기본 RNN보다 작은 숨은 차원으로도 더 나은 성능을 낼 때가 많아 매개변수의 부담을 얼마간 덜어 준다.

---

## 5. PyTorch로 밑바닥부터 구현하기

### LSTM 세포

```python
import torch
import torch.nn as nn
import numpy as np

class LSTMCell(nn.Module):
    """
    배움을 위해 손수 만든 LSTM 세포 구현.
    
    계산 효율을 위해 가중치 행렬을 합쳐 표준 LSTM 식을 구현한다.
    
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 효율을 위해 가중치를 합침: [W_i, W_f, W_c, W_o]를 쌓았다
        # 그러면 모든 문을 행렬 곱 한 번으로 계산할 수 있다
        # 모양: 입력 가중치는 (4 * hidden_size, input_size)
        # 모양: 숨은 가중치는 (4 * hidden_size, hidden_size)
        self.weight_ih = nn.Parameter(
            torch.randn(4 * hidden_size, input_size) / np.sqrt(input_size)
        )
        self.weight_hh = nn.Parameter(
            torch.randn(4 * hidden_size, hidden_size) / np.sqrt(hidden_size)
        )
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))
        
        # 망각 문의 편향을 1로 초기화 (기울기의 흐름에 매우 중요)
        self._init_forget_gate_bias()
    
    def _init_forget_gate_bias(self):
        """
        망각 문의 편향을 1로 초기화한다.
        
        그러면 학습 초반에 모델이 기본적으로 기억하게 되어, 무엇을 잊어야
        할지 배우기 전에 기울기가 일찍 사라지는 것을 막는다.
        
        """
        with torch.no_grad():
            # 문의 순서: [입력, 망각, 세포, 출력]
            # 망각 문의 편향은 두 번째 사분면에 있다
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
    순차열을 처리하는 온전한 LSTM 층.
    
    여러 층을 쌓는 것을 지원하며 뒤따르는 처리를 위해 모든 숨은 상태를
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
        
        # 층마다 세포 만들기
        # 첫 층은 input_size를, 뒤 층은 hidden_size를 받는다
        self.cells = nn.ModuleList([
            LSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
        # 층 사이의 드롭아웃 (마지막 층 뒤에는 없다)
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
            output: 맨 위 층의 모든 숨은 상태 (배치 크기, seq_len, hidden_size)
            (h_n, c_n): 모든 층의 마지막 상태이며 입력 상태와 모양이 같다
        """
        batch_size, seq_len, _ = x.size()
        
        # 모든 층의 상태 초기화
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
            
            # 층마다 통과
            for layer_idx, cell in enumerate(self.cells):
                h[layer_idx], c[layer_idx] = cell(
                    layer_input, (h[layer_idx], c[layer_idx])
                )
                layer_input = h[layer_idx]
                
                # 층 사이에 드롭아웃 적용 (마지막 층 뒤에는 없다)
                if self.dropout is not None and layer_idx < self.num_layers - 1:
                    layer_input = self.dropout(layer_input)
            
            outputs.append(h[-1])  # 맨 위 층의 출력 모으기
        
        # 출력과 상태 쌓기
        output = torch.stack(outputs, dim=1)  # (배치, seq_len, hidden)
        h_n = torch.stack(h, dim=0)           # (num_layers, 배치, hidden)
        c_n = torch.stack(c, dim=0)           # (num_layers, 배치, hidden)
        
        return output, (h_n, c_n)
```

### PyTorch와 견주어 확인하기

```python
def verify_implementation():
    """우리 구현이 PyTorch와 맞는지 확인한다."""
    torch.manual_seed(42)
    
    batch_size, seq_len, input_size, hidden_size = 4, 10, 8, 16
    
    # 우리 구현
    our_lstm = LSTM(input_size, hidden_size, num_layers=1)
    
    # PyTorch의 구현
    torch_lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
    
    # 가중치 옮기기 (순서가 다른 점을 감안)
    with torch.no_grad():
        # PyTorch는 문의 순서를 [i, f, g, o]로 두며 우리도 같게 쓴다
        torch_lstm.weight_ih_l0.copy_(our_lstm.cells[0].weight_ih)
        torch_lstm.weight_hh_l0.copy_(our_lstm.cells[0].weight_hh)
        torch_lstm.bias_ih_l0.copy_(our_lstm.cells[0].bias)
        torch_lstm.bias_hh_l0.zero_()  # 우리는 편향을 합치고 PyTorch는 나눈다
    
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

# LSTM 층 만들기
lstm = nn.LSTM(
    input_size=100,      # 입력 특징의 차원
    hidden_size=256,     # 숨은 상태와 세포 상태의 차원
    num_layers=2,        # 쌓은 LSTM 층의 수
    batch_first=True,    # 입력의 모양: (배치, seq, 특징)
    dropout=0.3,         # 층 사이의 드롭아웃 (num_layers > 1일 때만)
    bidirectional=False  # 순방향만 처리
)

# 입력: (배치 크기, sequence_length, input_size)
x = torch.randn(32, 50, 100)

# 순전파
output, (h_n, c_n) = lstm(x)

print(f"Output shape: {output.shape}")     # (32, 50, 256)
print(f"Final hidden: {h_n.shape}")        # (2, 32, 256) — 층마다 하나
print(f"Final cell: {c_n.shape}")          # (2, 32, 256) — 층마다 하나

# 뒤따르는 과제를 위해 출력에 접근
last_output = output[:, -1, :]    # 마지막 시각: (32, 256)
last_hidden = h_n[-1]             # 마지막 층의 숨은 상태: (32, 256)
# 참고: 단방향 LSTM에서는 last_output == last_hidden이다
```

---

## 7. 실전 응용

### 순차열 분류

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
        # 마지막 층의 숨은 상태 가져오기
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

### 순차열 생성 (언어 모형)

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
        
        # 선택적으로 임베딩과 출력 가중치를 묶기
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
        input_size: int,     # 입력 특징의 수
        hidden_size: int,
        num_layers: int,
        output_size: int,    # 출력 특징의 수
        horizon: int = 1,    # 예측 지평 (앞을 내다보는 걸음 수)
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
        
        # 예측 지평의 모든 걸음을 한꺼번에 예측
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
        
        # 모든 걸음 예측
        predictions = self.fc(final_hidden)  # (배치, output_size * horizon)
        predictions = predictions.view(-1, self.horizon, self.output_size)
        
        return predictions

# 예: 지난 20걸음으로 특징 3개의 다음 5걸음 예측
model = LSTMForecaster(
    input_size=3,
    hidden_size=128,
    num_layers=2,
    output_size=3,
    horizon=5
)

x = torch.randn(32, 20, 3)  # 표본 32개, 시각 20개, 특징 3개
predictions = model(x)       # (32, 5, 3)
```

---

## 8. 초기화와 학습의 모범 관행

### 가중치 초기화

```python
def init_lstm_weights(lstm: nn.LSTM):
    """
    모범 관행에 따라 LSTM의 가중치를 초기화한다.
    
    - 입력에서 숨은 상태로 가는 가중치: 자비에르(글로럿) 균등
    - 숨은 상태끼리의 가중치: 직교 (기울기의 노름을 지킨다)
    - 편향: 0, 다만 망각 문의 편향은 1
    """
    for name, param in lstm.named_parameters():
        if 'weight_ih' in name:
            # 입력에서 숨은 상태로 가는 가중치: 자비에르 균등
            nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:
            # 숨은 상태끼리의 가중치: 직교
            # 그러면 순환을 지나도 기울기의 크기가 지켜진다
            nn.init.orthogonal_(param)
        elif 'bias' in name:
            # 편향: 대체로 0
            nn.init.zeros_(param)
            # 다만 망각 문의 편향은 1로 둔다
            n = param.size(0) // 4
            param.data[n:2*n].fill_(1.0)
```

### 망각 문의 편향을 왜 1로 두는가

무작위 가중치로 초기화하면 문의 활성값이 (작은 무작위 값의 시그모이드라) 약 0.5이다. 곧 모델이 시각마다 정보의 절반쯤을 잊으며 시작하는데, 먼 거리 의존을 배우기에 좋지 않다.

$b_f = 1$으로 두면 망각 문 시그모이드의 입력이 치우쳐 다음과 같이 된다.

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + 1) \approx \sigma(1) \approx 0.73$$

이 "기본은 기억" 초기화 덕분에 학습 초반에도 기울기가 흐른다. 그다음에 신경망이 필요에 따라 특정 정보를 잊는 법을 배운다.

### 기울기 자르기

```python
# 학습 중에는 폭발을 막으려고 기울기를 자른다
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
optimizer.step()
```

LSTM은 기울기 소실을 다루지만 특히 학습 초반에는 기울기 폭발을 겪을 수 있다. 기울기 자르기가 표준 관행이다.

### 학습률 스케줄링

```python
# 검증 손실이 정체되면 학습률을 줄인다
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

### 망각 문과 입력 문 묶기

관찰: 옛 정보를 잊을 때는 대개 새 정보를 넣고, 그 반대도 마찬가지이다. 두 문을 묶으면 다음과 같다.

$$c_t = f_t \odot c_{t-1} + (1 - f_t) \odot \tilde{c}_t$$

이렇게 하면 매개변수가 줄고 전체 "정보의 양"이 보존된다는 제약이 걸린다.

### GRU (문 달린 순환 단위)

GRU는 다음과 같이 LSTM을 간소화한다.

1. 세포 상태와 숨은 상태를 하나로 합친다
2. 망각 문과 입력 문을 "갱신 문" 하나로 엮는다
3. 출력 문 대신 "재설정 문"을 쓴다

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$$

$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$$

$$\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t])$$

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

**맞바꿈:** 매개변수가 적고(4배가 아니라 3배) 많은 과제에서 성능이 비슷하지만 기억을 다스리는 유연함은 덜하다.

---

## 10. 언제 LSTM을 쓸까

### LSTM의 강점

- **긴 순차열**: 시각 수백 개까지 효과적이다
- **길이가 다양한 순차열**: 순환으로 자연스럽게 다룬다
- **실시간·흐름 처리**: 한 시각씩 처리할 수 있다
- **순차적 귀납 편향**: 시간 순서를 명시적으로 다룬다

### 다른 것을 생각해 볼 때

| 상황 | 대안 | 까닭 |
|-----------|-------------|--------|
| 아주 긴 순차열 (1000 이상) | 트랜스포머 | 병렬 처리가 낫고 문맥 전체에 어텐션을 준다 |
| 단순하고 짧은 순차열 | GRU | 매개변수가 적고 빠르다 |
| 대규모 병렬 처리가 필요할 때 | 1차원 CNN, 트랜스포머 | LSTM은 본디 순차적이다 |
| 전역적인 의존 | 트랜스포머 | 자기 어텐션이 먼 거리를 곧바로 붙잡는다 |
| 말단 장치 배포 | GRU나 맞춤 구조 | 메모리를 덜 쓴다 |

### 요즘의 자리

자연어 처리에서는 트랜스포머가 LSTM을 거의 대체했지만, 다음에서는 LSTM이 여전히 겨룰 만하거나 낫다.

- 순차 구조가 뚜렷한 시계열
- 지연이 적어야 하는 흐름 처리 응용
- 데이터가 적은 상황 (과적합에 덜 빠진다)
- 말단 장치 배포 (메모리 사용이 예측 가능하다)

---

## 11. 흔한 문제 잡기

### 증상과 해법

| 징후 | 짐작되는 원인 | 해결책 |
|---------|--------------|----------|
| 손실이 줄지 않는다 | 학습률이 너무 높거나 낮다 | 1e-3으로 해 보고 조정한다 |
| 손실이 크게 요동친다 | 기울기 폭발 | 기울기 자르기를 더한다 |
| 모델의 출력이 늘 같다 | 죽은 뉴런이나 기울기 소실 | 망각 문의 편향이 1인지 확인한다 |
| 먼 거리 성능이 나쁘다 | 숨은 차원이 모자라다 | hidden_size를 키운다 |
| 과적합 | 모델이 너무 크거나 규제가 없다 | 드롭아웃을 더하고 크기를 줄인다 |
| 학습이 느리다 | 순차열이 너무 길다 | 잘라 낸 BPTT를 쓴다 |

### 진단 코드

```python
def diagnose_lstm(model, dataloader, device):
    """LSTM 학습을 진단한다."""
    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            
            # 기울기의 크기 확인
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

**연습문제 1.**
LSTM의 식을 적고 각 문의 구실을 설명하라.

??? success "연습문제 1 풀이"
    망각 문: $f_t = \sigma(W_f[h_{t-1}, x_t] + b_f)$ — 세포 상태에서 무엇을 버릴지.
    입력 문: $i_t = \sigma(W_i[h_{t-1}, x_t] + b_i)$ — 어떤 새 정보를 담을지.
    후보: $\tilde{c}_t = \tanh(W_c[h_{t-1}, x_t] + b_c)$ — 새 후보 값.
    세포 갱신: $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$.
    출력 문: $o_t = \sigma(W_o[h_{t-1}, x_t] + b_o)$ — 무엇을 내보낼지.
    숨은 상태: $h_t = o_t \odot \tanh(c_t)$.

---

**연습문제 2.**
세포 상태 $c_t$이 기울기 소실을 덜어 주는 '컨베이어 벨트' 노릇을 하는 방식을 설명하라.

??? success "연습문제 2 풀이"
    세포 상태 갱신 $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$은 (잔차 연결처럼) 덧셈이다. 기울기는 $\frac{\partial c_T}{\partial c_t} = \prod_{k=t+1}^T f_k$으로 흐른다. 망각 문이 1에 가까우면 기울기가 그대로 지나가므로 시각 수백 개에 걸쳐 배울 수 있다.

---

**연습문제 3.**
입력 차원이 300이고 숨은 차원이 512인 LSTM의 매개변수 수를 계산하라.

??? success "연습문제 3 풀이"
    LSTM에는 문이 넷이고 저마다 입력용과 숨은 상태용 가중치 행렬이 있다.
    $4 \times (300 \times 512 + 512 \times 512 + 512) = 4 \times (153{,}600 + 262{,}144 + 512) = 4 \times 416{,}256 = 1{,}665{,}024$개의 매개변수이다.

---

**연습문제 4.**
망각 문의 편향을 0이 아니라 1로 초기화하는 일이 잦은 까닭은 무엇인가?

??? success "연습문제 4 풀이"
    편향이 0이면 망각 문이 $\sigma(0) = 0.5$이어서 세포 상태가 걸음마다 절반으로 준다. 편향을 1로 두면 $\sigma(1) \approx 0.73$이 되어 대체로 '기억하는' 상태에서 출발한다. 그러면 LSTM이 무엇을 잊어야 할지 배우기도 전에 중요한 정보를 잊는 일을 막는다(Jozefowicz 등, 2015).

## 정리하며

LSTM 신경망은 다음으로 기울기 소실 문제를 푼다.

1. **세포 상태**: 기울기를 지키는 덧셈 방식 정보 고속도로
2. **망각 문**: 어떤 정보를 버릴지 배운다
3. **입력 문**: 어떤 새 정보를 담을지 배운다
4. **출력 문**: 어떤 정보를 내보낼지 배운다

**핵심 성질:**

- 기울기가 곱셈으로 줄어들지 않고 세포 상태를 타고 흐른다
- 문이 과제에 맞는 정보 흐름의 방식을 배운다
- 시각 수백 개에 걸친 의존을 붙잡는다
- 기본 RNN보다 매개변수가 4배지만 기울기의 흐름이 훨씬 낫다

**모범 관행:**

- 망각 문의 편향을 1로 초기화한다
- 기울기 자르기를 쓴다
- 층 사이에 드롭아웃을 적용한다
- 매개변수에 제약이 있으면 GRU를 생각해 본다

이 구조는 계산 비용을 더하지만 훨씬 긴 순차열에서 배울 수 있게 해 주며, 이는 실제 순차열 모형 과제 대부분에 꼭 필요하다.

---

**참고 문헌**

1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.

2. Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). Learning to Forget: Continual Prediction with LSTM. *Neural Computation*, 12(10), 2451-2471.

3. Greff, K., Srivastava, R. K., Koutník, J., Steunebrink, B. R., & Schmidhuber, J. (2017). LSTM: A Search Space Odyssey. *IEEE Transactions on Neural Networks and Learning Systems*, 28(10), 2222-2232.

4. Jozefowicz, R., Zaremba, W., & Sutskever, I. (2015). An Empirical Exploration of Recurrent Network Architectures. *ICML*.

5. Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *EMNLP*.
