# Peephole 연결

표준 LSTM의 문은 이전 은닉 상태 $h_{t-1}$과 지금 입력 $x_t$으로 엑티베이션을 계산한다. 그런데 그러면 문이 정작 자기가 다스려야 할 기억인 셀 상태 $c_{t-1}$을 곧바로 볼 수 없다. Gers와 Schmidhuber(2000)가 내놓은 **Peephole 연결**은 문이 셀 상태에 곧바로 닿게 하여 이 한계를 푼다.

---

## 1. 왜 필요한가

표준 LSTM에서 문은 셀 상태를 실제로 보지 않고 그에 대한 결정을 내린다.

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

은닉 상태 $h_t = o_t \odot \tanh(c_t)$은 $c_t$을 걸러 변형한 모습이다. 출력 게이트이 닫힌 채로($o_t \approx 0$) 셀 상태에 정보가 담길 수 있는데, 그러면 다음 시각의 문이 $h_{t-1}$을 통해서는 그것을 볼 수 없다.

**문제:** $t-1$번째 스텝의 출력 게이트이 어떤 세포 차원을 눌러 버리면, $t$번째 스텝의 망각 게이트과 입력 게이트은 무엇을 지키고 더할지 정할 때 그 담긴 값을 볼 수 없다.

**Peephole 연결은** 셀 상태에서 각 문으로 가는 곧바른 성분별 연결을 더하여 이를 푼다.

---

## 2. 수학적 정식화

### 표준 LSTM (peephole 없음)

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

$$h_t = o_t \odot \tanh(c_t)$$

### peephole LSTM

망각 게이트과 입력 게이트은 **이전** 셀 상태 $c_{t-1}$에서 Peephole 연결을 받고, 출력 게이트은 (세포 갱신 뒤에 계산되므로) **지금** 셀 상태 $c_t$에서 Peephole 연결을 받는다.

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + \mathbf{w_f \odot c_{t-1}} + b_f)$$

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + \mathbf{w_i \odot c_{t-1}} + b_i)$$

$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + \mathbf{w_o \odot c_t} + b_o)$$

$$h_t = o_t \odot \tanh(c_t)$$

여기서 $w_f, w_i, w_o \in \mathbb{R}^n$은 (온전한 행렬이 아니라) **대각** peephole 웨이트 벡터이다.

!!! note "왜 대각 웨이트인가"
    Peephole 연결은 온전한 행렬 곱($W \cdot c$)이 아니라 성분별 곱($w \odot c$)을 쓴다. 곧 문의 차원마다 그에 대응하는 세포 차원만 본다. 이 대각 제약 덕분에 늘어나는 파라메터가 $3n$개로 적게 유지되면서도 문이 "제" 셀 상태 값을 살필 수 있다.

---

## 3. 파라메터 수에 미치는 영향

| 부품 | 표준 LSTM | peephole LSTM |
|-----------|---------------|---------------|
| 웨이트 행렬 | $4n(n+d)$ | $4n(n+d)$ |
| 바이어스 | $4n$ | $4n$ |
| peephole 벡터 | — | $3n$ |
| **합계** | $4n(n+d+1)$ | $4n(n+d+1) + 3n$ |

흔한 은닉 차원($n = 256$)에서 peephole는 파라메터를 $3 \times 256 = 768$개만 더한다. 전체 $4 \times 256 \times (256 + d + 1)$에 견주면 무시할 만하다.

---

## 4. PyTorch 구현

```python
import torch
import torch.nn as nn

class PeepholeLSTMCell(nn.Module):
    """엿보기 연결이 있는 LSTM 세포."""
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 표준 웨이트
        combined_size = input_size + hidden_size
        self.W_f = nn.Linear(combined_size, hidden_size)
        self.W_i = nn.Linear(combined_size, hidden_size)
        self.W_c = nn.Linear(combined_size, hidden_size)
        self.W_o = nn.Linear(combined_size, hidden_size)
        
        # 엿보기 웨이트 (대각이라 그냥 벡터이다)
        self.p_f = nn.Parameter(torch.randn(hidden_size) * 0.1)
        self.p_i = nn.Parameter(torch.randn(hidden_size) * 0.1)
        self.p_o = nn.Parameter(torch.randn(hidden_size) * 0.1)
        
        # 망각 바이어스를 1로 초기화
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
    """순차열을 처리하는 온전한 엿보기 LSTM 레이어."""
    
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
            states: 모든 레이어의 처음 (h_0, c_0)
        
        반환값:
            output: 맨 위 레이어의 숨은 상태 (배치 크기, seq_len, hidden_size)
            (h_n, c_n): 모든 레이어의 마지막 상태
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

## 5. 언제 Peephole 연결을 쓸까

Peephole은 게이트가 셀 상태의 크기를 직접 참조하게 하므로 타이밍이나 계수처럼 수치 상태가 중요한 과제의 후보가 될 수 있다. 그렇다고 표준 LSTM보다 항상 정확하거나 특정 과제에서 반드시 유리한 것은 아니다.

같은 데이터와 초기 조건, 학습 예산에서 peephole 유무를 비교한다. 추가 파라메터는 적더라도 게이트의 피드백 경로가 바뀌므로 최적화와 일반화는 별도로 평가해야 한다. 연구 결과는 해당 데이터와 설정의 범위에서 해석한다.

## 6. 그래디언트 흐름에 미치는 영향

Peephole 연결은 셀 상태를 지나는 그래디언트 경로를 더한다.

$$\left.\frac{\partial f_t}{\partial c_{t-1}}\right|_{h_{t-1}\ \mathrm{fixed}} = \operatorname{diag}(f_t\odot(1-f_t))\operatorname{diag}(w_f)$$

이는 곧바른 피드백 고리를 만든다. 셀 상태가 망각 게이트에 영향을 주고, 그것이 다음 셀 상태에 영향을 준다. 타이밍에 민감한 과제에서는 학습이 나아질 수 있지만 최적화가 더 까다로워질 수도 있다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
Peephole 연결에서 각 게이트가 어느 셀 상태를 참조하는지 설명하라.

</div>

??? success "연습문제 1 풀이"
    망각·입력 게이트에는 각각 $w_f\odot c_{t-1}$와 $w_i\odot c_{t-1}$를 더한다. 출력 게이트에는 갱신된 $c_t$로 만든 $w_o\odot c_t$를 더한다. 나머지 입력·은닉 선형 변환과 바이어스는 유지한다. 성분별 연결이면 추가 파라메터는 $3h$개다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
Peephole 연결이 유용할 수 있는 이유와 검증할 항목을 설명하라.

</div>

??? success "연습문제 2 풀이"
    출력 게이트와 tanh를 거치지 않은 셀 값에 직접 접근하여 상태 크기에 따른 게이트 결정을 학습할 수 있다. 타이밍·계수 과제에서 도움이 될 가능성이 있지만 이득은 보장되지 않는다. 같은 학습 조건의 표준 LSTM과 검증 성능·비용을 비교한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
대각 peephole이 추가하는 파라메터와 상대 증가율을 구하라.

</div>

??? success "연습문제 3 풀이"
    은닉 크기 h이면 $3h$개다. 바이어스 하나인 기본 셀의 $4h(h+x+1)$개에 대한 증가율은 $3/[4(h+x+1)]$이다. h=512,x=300이면 추가 1536개로 약 0.0923%다. 입력 차원과 기본 바이어스 규약을 명시해야 정확한 비율을 말할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
Peephole 연결이 있는 LSTM 세포를 PyTorch로 구현하라.

</div>

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

## 정리하며

Peephole은 셀 상태에서 게이트로 직접 연결을 더한다. 망각·입력 게이트는 이전 셀 상태, 출력 게이트는 새 셀 상태를 참조한다. 성분별 추가 웨이트는 3h개이며, 효과와 최적화의 변화는 실제 과제에서 비교해야 한다.

**참고 문헌**

1. Gers, F. A., & Schmidhuber, J. (2000). Recurrent Nets that Time and Count. *IJCNN*.

2. Gers, F. A., Schraudolph, N. N., & Schmidhuber, J. (2003). Learning Precise Timing with LSTM Recurrent Networks. *Journal of Machine Learning Research*, 3, 115-143.

3. Greff, K., Srivastava, R. K., Koutník, J., Steunebrink, B. R., & Schmidhuber, J. (2017). LSTM: A Search Space Odyssey. *IEEE Transactions on Neural Networks and Learning Systems*, 28(10), 2222-2232.
