# GRU 구조

GRU는 리셋·업데이트 게이트로 은닉 상태를 조절하는 순환 구조다. LSTM과 달리 별도 셀 상태나 출력 게이트가 없으며, 같은 입력·은닉 차원에서는 순환 부분의 파라메터가 적다.

적은 파라메터는 모델 크기를 줄이지만 실제 속도, 과적합과 정확도의 우열을 보장하지 않는다. 이 페이지에서는 수식과 직접 구현의 차이까지 살펴본다.

**핵심 착상:** LSTM이 "무엇을 기억할지"(셀 상태)와 "무엇을 내보낼지"(은닉 상태)에 서로 다른 통로를 쓰는 데 반해, GRU는 이를 하나의 은닉 상태로 합치고 문 두 개로 정보의 흐름을 다스린다.

---

## 1. 수학적 정식화

시각 $t$에서 입력 $x_t \in \mathbb{R}^d$과 이전 은닉 상태 $h_{t-1} \in \mathbb{R}^n$이 주어졌을 때 다음과 같다.

### 리셋 게이트

후보를 계산할 때 이전 상태를 얼마나 잊을지 다스린다.

$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$

- $r_t \approx 0$이면 이전 상태가 "되돌려진다"(후보 계산에서 무시된다)
- $r_t \approx 1$이면 이전 상태가 후보에 온전히 영향을 준다

### 업데이트 게이트

옛 상태와 새 상태 사이의 보간을 다스린다.

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$

- $z_t \approx 0$이면 이전 상태를 지킨다(LSTM의 망각 게이트이 1에 가까운 것과 같다)
- $z_t \approx 1$이면 새 후보를 쓴다(LSTM의 입력 게이트이 1에 가까운 것과 같다)

### 후보 은닉 상태

리셋 게이트의 조절을 받아 계산한 새 정보이다.

$$\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)$$

리셋 게이트 $r_t$은 tanh *안*에 나타나며, $x_t$과 이어 붙이기 전에 $h_{t-1}$에 곱해진다. 덕분에 모델이 새 내용을 계산할 때 이전 상태를 아예 무시할 수 있다.

### 은닉 상태 갱신

이전 상태와 후보 사이의 보간이다.

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

여기서:

- $\sigma$은 시그모이드 함수 $\sigma(x) = \frac{1}{1+e^{-x}}$이다
- $\odot$은 성분별(아다마르) 곱을 뜻한다
- $[a, b]$은 피처 차원을 따라 이어 붙인 것을 뜻한다
- $W_r, W_z, W_h \in \mathbb{R}^{n \times (n+d)}$은 웨이트 행렬이다
- $b_r, b_z, b_h \in \mathbb{R}^n$은 바이어스 벡터이다

---

## 2. 문 이해하기

### 망각과 입력을 합친 업데이트 게이트

갱신식에는 우아한 제약이 담겨 있다.

$$h_t = \underbrace{(1 - z_t) \odot h_{t-1}}_{\text{retained information}} + \underbrace{z_t \odot \tilde{h}_t}_{\text{new information}}$$

이는 **볼록 결합**이다(웨이트의 합이 1이다).

| $z_t$ 값 | 효과 | LSTM에서의 대응 |
|-------------|--------|-----------------|
| $z_t \approx 0$ | 옛것을 지킴: $h_t \approx h_{t-1}$ | 망각 게이트 ≈ 1, 입력 게이트 ≈ 0 |
| $z_t \approx 1$ | 새것을 씀: $h_t \approx \tilde{h}_t$ | 망각 게이트 ≈ 0, 입력 게이트 ≈ 1 |
| $z_t = 0.5$ | 반반 섞음 | 부분 갱신 |

**보간의 의미:** 계수의 합이 1이라는 것은 각 성분이 이전 값과 후보 사이에 놓인다는 뜻이다. 벡터 전체를 단일 스칼라로 혼합하는 볼록 결합과 구분해야 한다. 후보가 0이면 새 정보 없이 이전 값을 줄일 수도 있으므로 정보량 보존을 강제하지 않는다.

### 리셋 게이트: 급격한 전환을 가능하게 하다

리셋 게이트은 LSTM의 어느 문과도 다른 구실을 한다. $r_t \approx 0$이면 다음과 같다.

$$\tilde{h}_t \approx \tanh(W_h \cdot [\mathbf{0}, x_t] + b_h)$$

이전 상태가 없는 것처럼 후보를 계산한다. 오로지 지금 입력만으로 "새로 시작"하는 계산이다. 문장 경계나 화제 전환, 이상 탐지에 매우 중요하다.

**문 사이의 상호작용:** 리셋 게이트은 *어떤 후보를 내놓을지*에 영향을 주고, 업데이트 게이트은 *그 후보를 얼마나 쓸지*를 정한다.

1. 리셋 게이트: "새 후보가 이력을 고려해야 하는가?"
2. 업데이트 게이트: "이 새 후보를 쓰기는 해야 하는가?"

---

## 3. 그래디언트 흐름 분석

이 페이지의 규약 $h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t$를 미분하면 다음과 같다.

$$\frac{\partial h_t}{\partial h_{t-1}}=\operatorname{diag}(1-z_t)+\operatorname{diag}(z_t)\frac{\partial\tilde h_t}{\partial h_{t-1}}+\operatorname{diag}(\tilde h_t-h_{t-1})\frac{\partial z_t}{\partial h_{t-1}}$$

첫 항이 직접 보존 경로다. z가 작으면 이 항은 항등에 가까워지지만 나머지 항까지 자동으로 작다고 보장할 수는 없다. 후보 미분에는 리셋 게이트의 의존성도 포함된다. LSTM의 직접 셀 경로와 마찬가지로 보존 경로의 존재는 전체 그래디언트 소실의 제거를 뜻하지 않는다.

## 4. PyTorch로 밑바닥부터 구현하기

### GRU 셀 (학습을 위한 판본)

```python
import torch
import torch.nn as nn
import numpy as np

class GRUCell(nn.Module):
    """
    러닝을 위해 손수 만든 GRU 세포 구현.
    
    알아보기 쉽도록 웨이트 행렬을 따로 두고 표준 GRU 식을 구현한다.
    실전 구현은 효율을 위해 행렬을 합친다.
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 재설정 문의 파라메터
        self.W_r = nn.Parameter(
            torch.randn(hidden_size, input_size + hidden_size) / np.sqrt(input_size + hidden_size)
        )
        self.b_r = nn.Parameter(torch.zeros(hidden_size))
        
        # 갱신 문의 파라메터
        self.W_z = nn.Parameter(
            torch.randn(hidden_size, input_size + hidden_size) / np.sqrt(input_size + hidden_size)
        )
        self.b_z = nn.Parameter(torch.zeros(hidden_size))
        
        # 후보의 파라메터
        self.W_h = nn.Parameter(
            torch.randn(hidden_size, input_size + hidden_size) / np.sqrt(input_size + hidden_size)
        )
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(
        self, 
        x: torch.Tensor, 
        h_prev: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        한 시각의 순전파.
        
        인수:
            x: 입력 텐서 (배치 크기, input_size)
            h_prev: 이전 숨은 상태 (배치 크기, hidden_size)
        
        반환값:
            h: 새 숨은 상태 (배치 크기, hidden_size)
        """
        batch_size = x.size(0)
        
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # 문 계산을 위해 이어 붙이기
        combined = torch.cat([h_prev, x], dim=1)
        
        # 재설정 문
        r = torch.sigmoid(combined @ self.W_r.t() + self.b_r)
        
        # 갱신 문
        z = torch.sigmoid(combined @ self.W_z.t() + self.b_z)
        
        # 재설정 조절을 받은 후보 숨은 상태
        combined_reset = torch.cat([r * h_prev, x], dim=1)
        h_tilde = torch.tanh(combined_reset @ self.W_h.t() + self.b_h)
        
        # 마지막 숨은 상태: 보간
        h = (1 - z) * h_prev + z * h_tilde
        
        return h
```

### GRU 셀 (효율적인 판본)

이 코드는 위 셀과 단순한 최적화 관계가 아니다. 위 셀은 리셋을 이전 상태에 먼저 곱하고, 아래 셀은 순환 선형 변환 결과에 곱하므로 일반적으로 계산이 다르다. 아래 셀도 이전 상태의 계수로 `1-z`를 쓰며, PyTorch 내장 GRU는 그 계수로 `z`를 쓴다. [내장 GRU의 공식 식](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)과 비교할 때 리셋 위치, 업데이트 규약과 두 바이어스를 모두 맞춰야 한다.

```python
class GRUCellEfficient(nn.Module):
    """
    웨이트 행렬을 합친 효율적인 GRU 세포.
    
    모든 문의 계산을 하나의 행렬 곱으로 합쳐 GPU를 더 잘 쓴다.
    
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.weight_ih = nn.Parameter(
            torch.randn(3 * hidden_size, input_size) / np.sqrt(input_size)
        )
        self.weight_hh = nn.Parameter(
            torch.randn(3 * hidden_size, hidden_size) / np.sqrt(hidden_size)
        )
        self.bias_ih = nn.Parameter(torch.zeros(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.zeros(3 * hidden_size))
    
    def forward(self, x: torch.Tensor, h_prev: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = x.size(0)
        
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        gi = x @ self.weight_ih.t() + self.bias_ih
        gh = h_prev @ self.weight_hh.t() + self.bias_hh
        
        n = self.hidden_size
        r_i, z_i, n_i = gi[:, :n], gi[:, n:2*n], gi[:, 2*n:]
        r_h, z_h, n_h = gh[:, :n], gh[:, n:2*n], gh[:, 2*n:]
        
        r = torch.sigmoid(r_i + r_h)
        z = torch.sigmoid(z_i + z_h)
        
        # 후보: 숨은 상태의 몫에만 재설정 문 적용
        h_tilde = torch.tanh(n_i + r * n_h)
        
        h = (1 - z) * h_prev + z * h_tilde
        
        return h
```

### 온전한 GRU 레이어

각 시점·레이어를 반복하며 마지막 레이어의 전체 출력과 레이어별 최종 상태를 반환한다. 초기 상태의 자료형은 입력과 맞추어야 하며, 길이 0인 입력은 마지막 stack에서 실패한다.

```python
class GRU(nn.Module):
    """
    순차열을 처리하는 온전한 GRU 레이어.
    
    여러 레이어를 쌓는 것과 레이어 사이의 드롭아웃을 지원한다.
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
        
        self.cells = nn.ModuleList([
            GRUCellEfficient(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(
        self, 
        x: torch.Tensor, 
        h_0: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        순차열 전체를 GRU로 처리한다.
        
        인수:
            x: 입력 순차열 (배치 크기, seq_len, input_size)
            h_0: 처음 숨은 상태 (num_layers, 배치 크기, hidden_size)
        
        반환값:
            output: 맨 위 레이어의 모든 숨은 상태 (배치 크기, seq_len, hidden_size)
            h_n: 모든 레이어의 마지막 숨은 상태 (num_layers, 배치 크기, hidden_size)
        """
        batch_size, seq_len, _ = x.size()
        
        if h_0 is None:
            h = [torch.zeros(batch_size, self.hidden_size, device=x.device)
                 for _ in range(self.num_layers)]
        else:
            h = [h_0[i] for i in range(self.num_layers)]
        
        outputs = []
        for t in range(seq_len):
            layer_input = x[:, t, :]
            
            for layer_idx, cell in enumerate(self.cells):
                h[layer_idx] = cell(layer_input, h[layer_idx])
                layer_input = h[layer_idx]
                
                if self.dropout is not None and layer_idx < self.num_layers - 1:
                    layer_input = self.dropout(layer_input)
            
            outputs.append(h[-1])
        
        output = torch.stack(outputs, dim=1)
        h_n = torch.stack(h, dim=0)
        
        return output, h_n
```

---

## 5. PyTorch 내장 GRU 쓰기

```python
import torch
import torch.nn as nn

# GRU 레이어 만들기
gru = nn.GRU(
    input_size=100,      # 입력 피처의 차원
    hidden_size=256,     # 숨은 상태의 차원
    num_layers=2,        # 쌓은 GRU 레이어의 수
    batch_first=True,    # 입력의 모양: (배치, seq, 피처)
    dropout=0.3,         # 레이어 사이의 드롭아웃 (num_layers > 1일 때만)
    bidirectional=False  # 기본은 단방향
)

# 입력: (배치 크기, sequence_length, input_size)
x = torch.randn(32, 50, 100)

# 순전파
output, h_n = gru(x)

print(f"Output shape: {output.shape}")     # (32, 50, 256)
print(f"Final hidden: {h_n.shape}")        # (2, 32, 256) — 레이어마다 하나

# LSTM과의 핵심 차이: 세포 상태가 없다!
# LSTM이 돌려주는 것: output, (h_n, c_n)
# GRU가 돌려주는 것:  output, h_n
```

**출력:**

```
Output shape: torch.Size([32, 50, 256])
Final hidden: torch.Size([2, 32, 256])
```

---

## 6. 실전 응용

### 텍스트 분류

```python
class GRUClassifier(nn.Module):
    """순차열 분류를 위한 GRU."""
    
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes,
                 num_layers=2, dropout=0.5, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.gru = nn.GRU(embed_dim, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        _, h_n = self.gru(embedded)
        logits = self.fc(self.dropout(h_n[-1]))
        return logits
```

### 시계열 예측

```python
class GRUForecaster(nn.Module):
    """여러 걸음을 내다보는 시계열 예측을 위한 GRU."""
    
    def __init__(self, input_size, hidden_size, num_layers, 
                 output_size, horizon=1, dropout=0.2):
        super().__init__()
        self.horizon = horizon
        self.output_size = output_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size * horizon)
    
    def forward(self, x):
        _, h_n = self.gru(x)
        predictions = self.fc(h_n[-1])
        return predictions.view(-1, self.horizon, self.output_size)

# 예: 지난 20걸음으로 피처 3개의 다음 5걸음 예측
model = GRUForecaster(input_size=3, hidden_size=128, num_layers=2,
                      output_size=3, horizon=5)

x = torch.randn(32, 20, 3)
predictions = model(x)  # (32, 5, 3)
```

### 양방향 GRU

```python
class BiGRUEncoder(nn.Module):
    """
    순차열 부호화를 위한 양방향 GRU.
    
    순차열을 양쪽 방향으로 처리하여 자리마다 과거와 미래의 문맥을
    모두 붙잡는다.
    """
    
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0,
                          bidirectional=True)
    
    def forward(self, x):
        outputs, h_n = self.gru(x)
        # outputs: (배치, seq, hidden * 2) — 이미 이어 붙어 있다
        
        # h_n: (num_layers * 2, 배치, hidden) → (num_layers, 배치, hidden * 2)
        batch_size = x.size(0)
        h_n = h_n.view(self.num_layers, 2, batch_size, self.hidden_size)
        h_n = h_n.permute(0, 2, 1, 3).reshape(self.num_layers, batch_size, -1)
        
        return outputs, h_n
```

---

## 7. 초기화의 모범 관행

```python
def init_gru_weights(gru: nn.GRU):
    """
    안정된 학습을 위해 GRU의 웨이트를 초기화한다.
    
    - 입력에서 숨은 상태로 가는 웨이트: 자비에르 균등
    - 숨은 상태끼리의 웨이트: 직교 (그래디언트의 노름을 지킨다)
    - 바이어스: 0 (또는 선택적으로 재설정 문의 바이어스를 1 쪽으로)
    """
    for name, param in gru.named_parameters():
        if 'weight_ih' in name:
            nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:
            nn.init.orthogonal_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
            # 선택: 재설정 문을 "기억" 쪽으로 치우치게 하기
            # n = param.size(0) // 3
            # param.data[:n].fill_(1.0)  # Reset gate bias
```

**직교 초기화의 범위:** 쌓인 `weight_hh` 전체에 직교 초기화를 적용하는 것과 게이트별 정사각 블록을 각각 직교 초기화하는 것은 다르다. 어느 방식이든 게이트와 비선형성이 포함된 전체 순환 야코비안의 노름까지 보존한다고 보장하지 않는다.

---

## 8. GRU의 변형

### 최소 GRU

문 하나만 쓰는 판본으로 파라메터를 더 줄인다.

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

$$\tilde{h}_t = \tanh(W_h \cdot [h_{t-1}, x_t] + b_h)$$

$$h_t = f_t \odot h_{t-1} + (1 - f_t) \odot \tilde{h}_t$$

리셋 게이트을 아예 없애고 갱신·망각 게이트 하나만 쓴다. 더 간단하지만 표현력은 떨어진다.

```python
class MinimalGRUCell(nn.Module):
    """문 하나짜리 최소 GRU."""
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        combined_size = input_size + hidden_size
        self.W_f = nn.Linear(combined_size, hidden_size)
        self.W_h = nn.Linear(combined_size, hidden_size)
    
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([h, x], dim=-1)
        f = torch.sigmoid(self.W_f(combined))
        h_tilde = torch.tanh(self.W_h(combined))
        return f * h + (1 - f) * h_tilde
```

### 문을 묶은 GRU

리셋 게이트과 업데이트 게이트을 묶어 더 간소화한 것이다.

$$r_t = 1 - z_t$$

이렇게 하면 많이 갱신할 때($z_t$이 클 때) 더 많이 되돌리게 되고($r_t$이 작아지고) 그 반대도 마찬가지가 된다.

---

## 9. 파라메터 효율

```python
def compare_parameters(input_size: int = 100, hidden_size: int = 256):
    """구조 사이의 파라메터 수를 견준다."""
    rnn_params = hidden_size * (input_size + hidden_size) + hidden_size
    gru_params = 3 * rnn_params
    lstm_params = 4 * rnn_params
    
    print(f"Parameter comparison (input={input_size}, hidden={hidden_size}):")
    print(f"  Vanilla RNN: {rnn_params:,} (1.0×)")
    print(f"  GRU:         {gru_params:,} (3.0×)")
    print(f"  LSTM:        {lstm_params:,} (4.0×)")
    print(f"\n  GRU savings vs LSTM: {(1 - gru_params/lstm_params)*100:.1f}%")

# compare_parameters()
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
GRU의 식을 적고 LSTM과 비교하여라.

</div>

??? success "연습문제 1 풀이"
    재설정: $r_t = \sigma(W_r[h_{t-1}, x_t])$. 갱신: $z_t = \sigma(W_z[h_{t-1}, x_t])$. 후보: $\tilde{h}_t = \tanh(W[r_t \odot h_{t-1}, x_t])$. 출력: $h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$.
    GRU는 망각 게이트과 입력 게이트을 업데이트 게이트 $z_t$ 하나로 합치고 셀 상태를 따로 두지 않아 LSTM보다 간단하다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
같은 입력·은닉 차원의 GRU와 LSTM 파라메터 수를 비교할 때 명시해야 할 조건을 설명하라.

</div>

??? success "연습문제 2 풀이"
    단일 레이어·방향과 같은 바이어스 규약을 맞춘다. 묶음마다 바이어스 하나면 각각 $3h(x+h+1)$, $4h(x+h+1)$개이고 PyTorch 기본 두 바이어스이면 $3h(x+h+2)$, $4h(x+h+2)$개다. 후보를 포함한 세·네 계산 묶음이며, 임베딩이나 출력 레이어가 추가된 전체 모델에서는 3:4 비율이 유지되지 않을 수 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
LSTM과 GRU를 선택하는 비교 실험의 조건을 설명하라.

</div>

??? success "연습문제 3 풀이"
    같은 분할, 전처리와 평가 기준을 사용하고 각 모델의 러닝레이트와 은닉 크기를 검증셋에서 조정한다. 같은 은닉 크기 비교와 같은 파라메터·시간 예산 비교는 질문이 다르므로 구분한다. 여러 초기값의 변동과 실제 장치에서의 지연·메모리를 함께 확인한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
GRU 셀를 PyTorch로 밑바닥부터 구현하라.

</div>

??? success "연습문제 4 풀이"
    ```python
    class GRUCell(nn.Module):
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.Wr = nn.Linear(input_size + hidden_size, hidden_size)
            self.Wz = nn.Linear(input_size + hidden_size, hidden_size)
            self.Wh = nn.Linear(input_size + hidden_size, hidden_size)
        def forward(self, x, h):
            xh = torch.cat([x, h], dim=-1)
            r = torch.sigmoid(self.Wr(xh))
            z = torch.sigmoid(self.Wz(xh))
            h_tilde = torch.tanh(self.Wh(torch.cat([x, r * h], -1)))
            return (1 - z) * h + z * h_tilde
    ```

## 정리하며

GRU는 하나의 상태를 게이트로 보존·갱신한다. 직접 구현과 내장 모듈은 업데이트 계수와 리셋 적용 위치가 다를 수 있으므로 식을 먼저 맞춰야 한다. 적은 파라메터는 구조적 차이이며 실제 성능·속도의 우열은 평가 조건에 따라 확인한다.

**참고 문헌**

1. Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *EMNLP*.

2. Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling. *NIPS Workshop*.

3. Jozefowicz, R., Zaremba, W., & Sutskever, I. (2015). An Empirical Exploration of Recurrent Network Architectures. *ICML*.

4. Greff, K., Srivastava, R. K., Koutník, J., Steunebrink, B. R., & Schmidhuber, J. (2017). LSTM: A Search Space Odyssey. *IEEE Transactions on Neural Networks and Learning Systems*, 28(10), 2222-2232.

5. Heck, J., & Salem, F. M. (2017). Simplified Minimal Gated Unit Variations for Recurrent Neural Networks. *MWSCAS*.
