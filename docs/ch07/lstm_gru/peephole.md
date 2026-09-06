# 엿보기 연결
## 들어가며

표준 LSTM의 문은 이전 숨은 상태 $h_{t-1}$과 지금 입력 $x_t$으로 활성값을 계산한다. 그런데 그러면 문이 정작 자기가 다스려야 할 기억인 세포 상태 $c_{t-1}$을 곧바로 볼 수 없다. Gers와 Schmidhuber(2000)가 내놓은 **엿보기 연결**은 문이 세포 상태에 곧바로 닿게 하여 이 한계를 푼다.

---

## 왜 필요한가

표준 LSTM에서 문은 세포 상태를 실제로 보지 않고 그에 대한 결정을 내린다.

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

숨은 상태 $h_t = o_t \odot \tanh(c_t)$은 $c_t$을 걸러 변형한 모습이다. 출력 문이 닫힌 채로($o_t \approx 0$) 세포 상태에 정보가 담길 수 있는데, 그러면 다음 시각의 문이 $h_{t-1}$을 통해서는 그것을 볼 수 없다.

**문제:** $t-1$번째 걸음의 출력 문이 어떤 세포 차원을 눌러 버리면, $t$번째 걸음의 망각 문과 입력 문은 무엇을 지키고 더할지 정할 때 그 담긴 값을 볼 수 없다.

**엿보기 연결은** 세포 상태에서 각 문으로 가는 곧바른 성분별 연결을 더하여 이를 푼다.

---

## 수학적 정식화

### 표준 LSTM (엿보기 없음)

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

$$h_t = o_t \odot \tanh(c_t)$$

### 엿보기 LSTM

망각 문과 입력 문은 **이전** 세포 상태 $c_{t-1}$에서 엿보기 연결을 받고, 출력 문은 (세포 갱신 뒤에 계산되므로) **지금** 세포 상태 $c_t$에서 엿보기 연결을 받는다.

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + \mathbf{w_f \odot c_{t-1}} + b_f)$$

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + \mathbf{w_i \odot c_{t-1}} + b_i)$$

$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + \mathbf{w_o \odot c_t} + b_o)$$

$$h_t = o_t \odot \tanh(c_t)$$

여기서 $w_f, w_i, w_o \in \mathbb{R}^n$은 (온전한 행렬이 아니라) **대각** 엿보기 가중치 벡터이다.

!!! note "왜 대각 가중치인가"
    엿보기 연결은 온전한 행렬 곱($W \cdot c$)이 아니라 성분별 곱($w \odot c$)을 쓴다. 곧 문의 차원마다 그에 대응하는 세포 차원만 본다. 이 대각 제약 덕분에 늘어나는 매개변수가 $3n$개로 적게 유지되면서도 문이 "제" 세포 상태 값을 살필 수 있다.

---

## 매개변수 수에 미치는 영향

| 부품 | 표준 LSTM | 엿보기 LSTM |
|-----------|---------------|---------------|
| 가중치 행렬 | $4n(n+d)$ | $4n(n+d)$ |
| 편향 | $4n$ | $4n$ |
| 엿보기 벡터 | — | $3n$ |
| **합계** | $4n(n+d+1)$ | $4n(n+d+1) + 3n$ |

흔한 숨은 차원($n = 256$)에서 엿보기는 매개변수를 $3 \times 256 = 768$개만 더한다. 전체 $4 \times 256 \times (256 + d + 1)$에 견주면 무시할 만하다.

---

## PyTorch 구현

```python
import torch
import torch.nn as nn

class PeepholeLSTMCell(nn.Module):
    """엿보기 연결이 있는 LSTM 세포."""
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 표준 가중치
        combined_size = input_size + hidden_size
        self.W_f = nn.Linear(combined_size, hidden_size)
        self.W_i = nn.Linear(combined_size, hidden_size)
        self.W_c = nn.Linear(combined_size, hidden_size)
        self.W_o = nn.Linear(combined_size, hidden_size)
        
        # 엿보기 가중치 (대각이라 그냥 벡터이다)
        self.p_f = nn.Parameter(torch.randn(hidden_size) * 0.1)
        self.p_i = nn.Parameter(torch.randn(hidden_size) * 0.1)
        self.p_o = nn.Parameter(torch.randn(hidden_size) * 0.1)
        
        # 망각 편향을 1로 초기화
        nn.init.constant_(self.W_f.bias, 1.0)
    
    def forward(
        self, 
        x: torch.Tensor, 
        states: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        인수:
            x: 입력 (배치 크기, input_size)
            states: (h_prev, c_prev)의 쌍
        
        반환값:
            h: 새 숨은 상태
            (h, c): 새 상태의 쌍
        """
        batch_size = x.size(0)
        
        if states is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_prev, c_prev = states
        
        combined = torch.cat([x, h_prev], dim=-1)
        
        # 세포 상태로 엿보기 연결이 있는 문들
        # 망각 문과 입력 문은 이전 세포 상태를 엿본다
        f = torch.sigmoid(self.W_f(combined) + self.p_f * c_prev)
        i = torch.sigmoid(self.W_i(combined) + self.p_i * c_prev)
        c_tilde = torch.tanh(self.W_c(combined))
        
        # 세포 상태 갱신 (표준 LSTM과 같다)
        c = f * c_prev + i * c_tilde
        
        # 출력 문의 엿보기는 새 세포 상태를 쓴다
        o = torch.sigmoid(self.W_o(combined) + self.p_o * c)
        h = o * torch.tanh(c)
        
        return h, (h, c)

class PeepholeLSTM(nn.Module):
    """순차열을 처리하는 온전한 엿보기 LSTM 층."""
    
    def __init__(self, input_size: int, hidden_size: int, 
                 num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.cells = nn.ModuleList([
            PeepholeLSTMCell(
                input_size if layer == 0 else hidden_size, 
                hidden_size
            )
            for layer in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x, states=None):
        """
        인수:
            x: 입력 순차열 (배치 크기, seq_len, input_size)
            states: 모든 층의 처음 (h_0, c_0)
        
        반환값:
            output: 맨 위 층의 숨은 상태 (배치 크기, seq_len, hidden_size)
            (h_n, c_n): 모든 층의 마지막 상태
        """
        batch_size, seq_len, _ = x.size()
        
        if states is None:
            h = [torch.zeros(batch_size, self.hidden_size, device=x.device)
                 for _ in range(self.num_layers)]
            c = [torch.zeros(batch_size, self.hidden_size, device=x.device)
                 for _ in range(self.num_layers)]
        else:
            h = [states[0][i] for i in range(self.num_layers)]
            c = [states[1][i] for i in range(self.num_layers)]
        
        outputs = []
        for t in range(seq_len):
            layer_input = x[:, t, :]
            
            for layer_idx, cell in enumerate(self.cells):
                h[layer_idx], (_, c[layer_idx]) = cell(
                    layer_input, (h[layer_idx], c[layer_idx])
                )
                layer_input = h[layer_idx]
                
                if self.dropout and layer_idx < self.num_layers - 1:
                    layer_input = self.dropout(layer_input)
            
            outputs.append(h[-1])
        
        output = torch.stack(outputs, dim=1)
        h_n = torch.stack(h, dim=0)
        c_n = torch.stack(c, dim=0)
        
        return output, (h_n, c_n)
```

---

## 언제 엿보기 연결을 쓸까

### 이득을 보는 과제

엿보기 연결은 **정확한 타이밍**이나 **크기를 아는 문 조절**이 필요한 과제에 가장 쓸모가 있다.

| 과제의 종류 | 엿보기가 돕는 까닭 |
|-----------|-------------------|
| **리듬·박자 배우기** | 문이 세포 상태에 쌓인 타이밍을 좇을 수 있다 |
| **정확한 간격 예측** | 세포 상태의 크기가 흐른 시간을 담는다 |
| **문턱값 기반 결정** | 세포 값이 문턱을 넘을 때 문이 반응할 수 있다 |
| **세는 과제** | 세포 상태가 계수기 노릇을 하고 문이 그 수를 읽을 수 있다 |

### 표준 LSTM으로 충분할 때

대부분의 실제 과제에서는 엿보기가 없는 표준 LSTM도 비슷한 성능을 낸다.

- **언어 모형**: 토큰 수준의 무늬는 $h_{t-1}$으로도 잘 잡힌다
- **감성 분석**: 전체적인 감성에는 정확한 타이밍이 필요 없다
- **일반적인 시계열**: 대부분의 무늬는 세포 상태를 알 필요가 없다

### 실험적 증거

Greff 등(2017)은 LSTM 변형을 폭넓게 살핀 연구에서 다음을 알아냈다.

- 엿보기 연결은 대부분의 표준 자료에서 개선이 미미하다
- 망각 문과 출력 활성화가 훨씬 중요한 부품이다
- 정확한 타이밍이 필요한 특정 과제에서는 엿보기가 도움이 될 수 있다

---

## 기울기 흐름에 미치는 영향

엿보기 연결은 세포 상태를 지나는 기울기 경로를 더한다.

$$\frac{\partial f_t}{\partial c_{t-1}} = \sigma'(\cdot) \cdot \text{diag}(w_f)$$

이는 곧바른 되먹임 고리를 만든다. 세포 상태가 망각 문에 영향을 주고, 그것이 다음 세포 상태에 영향을 준다. 타이밍에 민감한 과제에서는 학습이 나아질 수 있지만 최적화가 더 까다로워질 수도 있다.

---

## 요약

엿보기 연결은 문이 세포 상태를 곧바로 볼 수 있게 하여 표준 LSTM을 넓힌다.

| 문 | 표준 LSTM의 입력 | 엿보기로 더해지는 것 |
|------|--------------------|--------------------|
| 망각 $f_t$ | $[h_{t-1}, x_t]$ | $+ w_f \odot c_{t-1}$ |
| 입력 $i_t$ | $[h_{t-1}, x_t]$ | $+ w_i \odot c_{t-1}$ |
| 출력 $o_t$ | $[h_{t-1}, x_t]$ | $+ w_o \odot c_t$ |

**핵심 정리:**

- 엿보기가 더하는 매개변수($3n$개)는 LSTM 전체에 견주면 무시할 만하다
- 정확한 타이밍이나 크기를 아는 문 조절이 필요한 과제에 가장 쓸모가 있다
- 두루 쓰는 순차열 모형에는 표준 LSTM으로 대체로 충분하다
- 요즘은 널리 쓰이지 않지만 타이밍이 중요한 응용에서는 해 볼 만하다

---

## 참고 문헌

1. Gers, F. A., & Schmidhuber, J. (2000). Recurrent Nets that Time and Count. *IJCNN*.

2. Gers, F. A., Schraudolph, N. N., & Schmidhuber, J. (2003). Learning Precise Timing with LSTM Recurrent Networks. *Journal of Machine Learning Research*, 3, 115-143.

3. Greff, K., Srivastava, R. K., Koutník, J., Steunebrink, B. R., & Schmidhuber, J. (2017). LSTM: A Search Space Odyssey. *IEEE Transactions on Neural Networks and Learning Systems*, 28(10), 2222-2232.

## 연습문제

**연습문제 1.**
엿보기 연결이 무엇이며 표준 LSTM의 식을 어떻게 바꾸는지 설명하라.

??? success "연습문제 1 풀이"
    엿보기 연결은 문이 세포 상태를 '엿보게' 해 준다. $f_t = \sigma(W_f[h_{t-1}, x_t] + V_f c_{t-1})$, $i_t = \sigma(W_i[h_{t-1}, x_t] + V_i c_{t-1})$, $o_t = \sigma(W_o[h_{t-1}, x_t] + V_o c_t)$이다. 대각 행렬 $V_f, V_i, V_o$이 매개변수 $3d_h$개를 더한다.

---

**연습문제 2.**
엿보기 연결이 가장 큰 이득을 주는 때는 언제인가?

??? success "연습문제 2 풀이"
    음악의 리듬 검출, 정확한 간격 세기, 타이밍이 중요한 제어 과제처럼 정확한 타이밍이 필요할 때이다. 세포 상태는 (tanh와 출력 문에 눌린) 숨은 상태보다 더 정확한 수치 정보를 지니므로, 곧바로 닿을 수 있으면 타이밍에 도움이 된다.

---

**연습문제 3.**
엿보기 연결을 더하면 매개변수가 얼마나 늘어나는가?

??? success "연습문제 3 풀이"
    크기가 $d_h$인 대각 행렬 세 개, 곧 매개변수 $3d_h$개가 는다. $d_h = 512$이면 기본 LSTM의 약 160만 개에 견주어 1536개만 늘어난다. 부담은 무시할 만하다(0.1% 미만).

---

**연습문제 4.**
엿보기 연결이 있는 LSTM 세포를 PyTorch로 구현하라.

??? success "연습문제 4 풀이"
    ```python
    class PeepholeLSTMCell(nn.Module):
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size)
            self.Vf = nn.Parameter(torch.zeros(hidden_size))
            self.Vi = nn.Parameter(torch.zeros(hidden_size))
            self.Vo = nn.Parameter(torch.zeros(hidden_size))
        def forward(self, x, h, c):
            gates = self.W(torch.cat([x, h], -1))
            f, i, g, o = gates.chunk(4, -1)
            f = torch.sigmoid(f + self.Vf * c)
            i = torch.sigmoid(i + self.Vi * c)
            c_new = f * c + i * torch.tanh(g)
            o = torch.sigmoid(o + self.Vo * c_new)
            h_new = o * torch.tanh(c_new)
            return h_new, c_new
    ```
