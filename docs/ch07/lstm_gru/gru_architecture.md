# GRU 구조
## 들어가며

2014년 Cho 등이 내놓은 문 달린 순환 단위(GRU)는 LSTM을 간소화한 판본으로, 매개변수를 덜 쓰면서도 비슷한 성능을 낸다. GRU는 세포 상태와 숨은 상태를 하나의 상태 벡터로 합치고 문의 수를 셋에서 둘로 줄여 LSTM 구조를 간단하게 만든다.

이렇게 구조를 간소화하면 실용적인 이점이 있다. 학습이 빠르고, 메모리를 덜 쓰고, 작은 데이터셋에서 과적합의 위험이 줄어든다. 그러면서도 기본 RNN이 배우지 못하는 먼 거리 의존을 붙잡는 능력은 지킨다.

**핵심 착상:** LSTM이 "무엇을 기억할지"(세포 상태)와 "무엇을 내보낼지"(숨은 상태)에 서로 다른 통로를 쓰는 데 반해, GRU는 이를 하나의 숨은 상태로 합치고 문 두 개로 정보의 흐름을 다스린다.

---

## 수학적 정식화

시각 $t$에서 입력 $x_t \in \mathbb{R}^d$과 이전 숨은 상태 $h_{t-1} \in \mathbb{R}^n$이 주어졌을 때 다음과 같다.

### 재설정 문

후보를 계산할 때 이전 상태를 얼마나 잊을지 다스린다.

$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$

- $r_t \approx 0$이면 이전 상태가 "되돌려진다"(후보 계산에서 무시된다)
- $r_t \approx 1$이면 이전 상태가 후보에 온전히 영향을 준다

### 갱신 문

옛 상태와 새 상태 사이의 보간을 다스린다.

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$

- $z_t \approx 0$이면 이전 상태를 지킨다(LSTM의 망각 문이 1에 가까운 것과 같다)
- $z_t \approx 1$이면 새 후보를 쓴다(LSTM의 입력 문이 1에 가까운 것과 같다)

### 후보 숨은 상태

재설정 문의 조절을 받아 계산한 새 정보이다.

$$\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)$$

재설정 문 $r_t$은 tanh *안*에 나타나며, $x_t$과 이어 붙이기 전에 $h_{t-1}$에 곱해진다. 덕분에 모델이 새 내용을 계산할 때 이전 상태를 아예 무시할 수 있다.

### 숨은 상태 갱신

이전 상태와 후보 사이의 보간이다.

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

여기서:

- $\sigma$은 시그모이드 함수 $\sigma(x) = \frac{1}{1+e^{-x}}$이다
- $\odot$은 성분별(아다마르) 곱을 뜻한다
- $[a, b]$은 특징 차원을 따라 이어 붙인 것을 뜻한다
- $W_r, W_z, W_h \in \mathbb{R}^{n \times (n+d)}$은 가중치 행렬이다
- $b_r, b_z, b_h \in \mathbb{R}^n$은 편향 벡터이다

---

## 문 이해하기

### 망각과 입력을 합친 갱신 문

갱신식에는 우아한 제약이 담겨 있다.

$$h_t = \underbrace{(1 - z_t) \odot h_{t-1}}_{\text{retained information}} + \underbrace{z_t \odot \tilde{h}_t}_{\text{new information}}$$

이는 **볼록 결합**이다(가중치의 합이 1이다).

| $z_t$ 값 | 효과 | LSTM에서의 대응 |
|-------------|--------|-----------------|
| $z_t \approx 0$ | 옛것을 지킴: $h_t \approx h_{t-1}$ | 망각 문 ≈ 1, 입력 문 ≈ 0 |
| $z_t \approx 1$ | 새것을 씀: $h_t \approx \tilde{h}_t$ | 망각 문 ≈ 0, 입력 문 ≈ 1 |
| $z_t = 0.5$ | 반반 섞음 | 부분 갱신 |

**결정적인 착상:** LSTM에서는 망각 문 $f_t$과 입력 문 $i_t$이 서로 독립이라 모두 잊으면서($f_t = 0$) 아무것도 더하지 않을($i_t = 0$) 수 있다. GRU에서는 $(1-z_t)$과 $z_t$이라는 서로 보완하는 가중치가 **보존을 강제한다**. 옛 정보를 많이 버릴수록 새 정보로 그만큼 채워야 한다.

### 재설정 문: 급격한 전환을 가능하게 하다

재설정 문은 LSTM의 어느 문과도 다른 구실을 한다. $r_t \approx 0$이면 다음과 같다.

$$\tilde{h}_t \approx \tanh(W_h \cdot [\mathbf{0}, x_t] + b_h)$$

이전 상태가 없는 것처럼 후보를 계산한다. 오로지 지금 입력만으로 "새로 시작"하는 셈이다. 문장 경계나 화제 전환, 이상 탐지에 매우 중요하다.

**문 사이의 상호작용:** 재설정 문은 *어떤 후보를 내놓을지*에 영향을 주고, 갱신 문은 *그 후보를 얼마나 쓸지*를 정한다.

1. 재설정 문: "새 후보가 이력을 고려해야 하는가?"
2. 갱신 문: "이 새 후보를 쓰기는 해야 하는가?"

---

## 기울기 흐름 분석

### GRU의 기울기 경로

숨은 상태를 지나는 기울기는 다음과 같다.

$$\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(1 - z_t) + z_t \odot \frac{\partial \tilde{h}_t}{\partial h_{t-1}}$$

첫 항 $(1 - z_t)$이 **곧바른 기울기 경로**를 준다. $z_t \approx 0$이면(옛 상태를 지키면) 다음과 같다.

$$\frac{\partial h_t}{\partial h_{t-1}} \approx I$$

이 경로로 기울기가 그대로 흘러 기울기 소실을 막는다.

### LSTM과 견주기

| 항목 | LSTM | GRU |
|--------|------|-----|
| 기울기 경로 | 세포 상태를 지남: $\frac{\partial c_t}{\partial c_{t-1}} = f_t$ | 보간을 지남: $(1-z_t)$ |
| 장치 | 덧셈 갱신 | 보간 갱신 |
| 문이 1에 가까울 때 | $f_t \approx 1$이면 기울기가 지켜진다 | $z_t \approx 0$이면 기울기가 지켜진다 |
| 독립성 | 세포와 숨은 상태의 기울기가 따로 있다 | 기울기 경로가 하나이다 |

---

## PyTorch로 밑바닥부터 구현하기

### GRU 세포 (배움을 위한 판본)

```python
import torch
import torch.nn as nn
import numpy as np

class GRUCell(nn.Module):
    """
    배움을 위해 손수 만든 GRU 세포 구현.
    
    알아보기 쉽도록 가중치 행렬을 따로 두고 표준 GRU 식을 구현한다.
    실전 구현은 효율을 위해 행렬을 합친다.
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 재설정 문의 매개변수
        self.W_r = nn.Parameter(
            torch.randn(hidden_size, input_size + hidden_size) / np.sqrt(input_size + hidden_size)
        )
        self.b_r = nn.Parameter(torch.zeros(hidden_size))
        
        # 갱신 문의 매개변수
        self.W_z = nn.Parameter(
            torch.randn(hidden_size, input_size + hidden_size) / np.sqrt(input_size + hidden_size)
        )
        self.b_z = nn.Parameter(torch.zeros(hidden_size))
        
        # 후보의 매개변수
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

### GRU 세포 (효율적인 판본)

```python
class GRUCellEfficient(nn.Module):
    """
    가중치 행렬을 합친 효율적인 GRU 세포.
    
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

### 온전한 GRU 층

```python
class GRU(nn.Module):
    """
    순차열을 처리하는 온전한 GRU 층.
    
    여러 층을 쌓는 것과 층 사이의 드롭아웃을 지원한다.
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
            output: 맨 위 층의 모든 숨은 상태 (배치 크기, seq_len, hidden_size)
            h_n: 모든 층의 마지막 숨은 상태 (num_layers, 배치 크기, hidden_size)
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

## PyTorch 내장 GRU 쓰기

```python
import torch
import torch.nn as nn

# GRU 층 만들기
gru = nn.GRU(
    input_size=100,      # 입력 특징의 차원
    hidden_size=256,     # 숨은 상태의 차원
    num_layers=2,        # 쌓은 GRU 층의 수
    batch_first=True,    # 입력의 모양: (배치, seq, 특징)
    dropout=0.3,         # 층 사이의 드롭아웃 (num_layers > 1일 때만)
    bidirectional=False  # 기본은 단방향
)

# 입력: (배치 크기, sequence_length, input_size)
x = torch.randn(32, 50, 100)

# 순전파
output, h_n = gru(x)

print(f"Output shape: {output.shape}")     # (32, 50, 256)
print(f"Final hidden: {h_n.shape}")        # (2, 32, 256) — 층마다 하나

# LSTM과의 핵심 차이: 세포 상태가 없다!
# LSTM이 돌려주는 것: output, (h_n, c_n)
# GRU가 돌려주는 것:  output, h_n
```

---

## 실전 응용

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

# 예: 지난 20걸음으로 특징 3개의 다음 5걸음 예측
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

## 초기화의 모범 관행

```python
def init_gru_weights(gru: nn.GRU):
    """
    안정된 학습을 위해 GRU의 가중치를 초기화한다.
    
    - 입력에서 숨은 상태로 가는 가중치: 자비에르 균등
    - 숨은 상태끼리의 가중치: 직교 (기울기의 노름을 지킨다)
    - 편향: 0 (또는 선택적으로 재설정 문의 편향을 1 쪽으로)
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

**숨은 가중치에 왜 직교 초기화를 쓰는가?** 직교 행렬은 특잇값이 모두 1이므로 행렬 곱을 지나도 기울기의 크기가 지켜진다. 순환을 지나는 기울기의 흐름에 도움이 된다.

---

## GRU의 변형

### 최소 GRU

문 하나만 쓰는 판본으로 매개변수를 더 줄인다.

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

$$\tilde{h}_t = \tanh(W_h \cdot [h_{t-1}, x_t] + b_h)$$

$$h_t = f_t \odot h_{t-1} + (1 - f_t) \odot \tilde{h}_t$$

재설정 문을 아예 없애고 갱신·망각 문 하나만 쓴다. 더 간단하지만 표현력은 떨어진다.

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

재설정 문과 갱신 문을 묶어 더 간소화한 것이다.

$$r_t = 1 - z_t$$

이렇게 하면 많이 갱신할 때($z_t$이 클 때) 더 많이 되돌리게 되고($r_t$이 작아지고) 그 반대도 마찬가지가 된다.

---

## 매개변수 효율

```python
def compare_parameters(input_size: int = 100, hidden_size: int = 256):
    """구조 사이의 매개변수 수를 견준다."""
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

## 요약

GRU는 LSTM을 우아하게 간소화한 것이다.

| 기능 | GRU에서의 구현 |
|---------|-------------------|
| 기억 관리 | 숨은 상태 하나 (세포 상태를 따로 두지 않음) |
| 망각 장치 | 갱신 문의 $(1-z_t)$ 인수 |
| 입력 장치 | 갱신 문의 $z_t$ 인수 (서로 보완) |
| 전환을 위한 되돌리기 | 재설정 문 $r_t$이 후보를 조절 |
| 기울기의 흐름 | $(1-z_t)$이라는 곧바른 경로 |

**핵심 이점:**

- LSTM보다 매개변수가 25% 적다
- 학습과 추론이 빠르다
- 대부분의 과제에서 성능이 비슷하다
- 이해하고 벌레잡기가 더 쉽다

**핵심 맞바꿈:**

- LSTM의 독립적인 문보다 유연하지 않다
- 망각과 입력이 묶여 있어 어떤 쓰임에는 제약이 될 수 있다

**요컨대** GRU는 순차열 모형의 기본 선택으로 훌륭하다. 효율이 좋아 계산 자원이 넉넉하지 않거나 빠르게 되풀이해 보아야 할 때 낫다. LSTM은 그 복잡함이 분명히 이득이 되는 과제에 아껴 두라.

---

## 참고 문헌

1. Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *EMNLP*.

2. Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling. *NIPS Workshop*.

3. Jozefowicz, R., Zaremba, W., & Sutskever, I. (2015). An Empirical Exploration of Recurrent Network Architectures. *ICML*.

4. Greff, K., Srivastava, R. K., Koutník, J., Steunebrink, B. R., & Schmidhuber, J. (2017). LSTM: A Search Space Odyssey. *IEEE Transactions on Neural Networks and Learning Systems*, 28(10), 2222-2232.

5. Heck, J., & Salem, F. M. (2017). Simplified Minimal Gated Unit Variations for Recurrent Neural Networks. *MWSCAS*.

## 연습문제

**연습문제 1.**
GRU의 식을 적고 LSTM과 견주어라.

??? success "연습문제 1 풀이"
    재설정: $r_t = \sigma(W_r[h_{t-1}, x_t])$. 갱신: $z_t = \sigma(W_z[h_{t-1}, x_t])$. 후보: $\tilde{h}_t = \tanh(W[r_t \odot h_{t-1}, x_t])$. 출력: $h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$.
    GRU는 망각 문과 입력 문을 갱신 문 $z_t$ 하나로 합치고 세포 상태를 따로 두지 않아 LSTM보다 간단하다.

---

**연습문제 2.**
숨은 차원이 같을 때 GRU의 매개변수가 LSTM보다 적은 까닭을 설명하라.

??? success "연습문제 2 풀이"
    GRU에는 문 행렬이 3개, LSTM에는 4개 있고 GRU에는 세포 상태가 따로 없다. 매개변수는 GRU가 $3(d_x + d_h)d_h + 3d_h$, LSTM이 $4(d_x + d_h)d_h + 4d_h$이다. GRU는 LSTM의 75%를 쓴다.

---

**연습문제 3.**
언제 LSTM 대신 GRU를 고르고, 반대로는 언제 그러한가?

??? success "연습문제 3 풀이"
    GRU는 데이터셋이 작을 때(과적합할 매개변수가 적다), 학습을 빨리 해야 할 때, 먼 거리 기억이 덜 중요한 과제에 알맞다. LSTM은 먼 거리 기억이 정확해야 하는 과제(언어 모형 따위)와 계산이 병목이 아닐 때 알맞다. 실험적으로 성능 차이는 작을 때가 많다.

---

**연습문제 4.**
GRU 세포를 PyTorch로 밑바닥부터 구현하라.

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
