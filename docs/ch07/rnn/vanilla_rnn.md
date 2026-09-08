# 기본 RNN

---

## 1. 구조 개관

기본(엘만) RNN은 가장 단순한 순환 구조로, 시각마다 선형 변환 뒤 $\tanh$ 비선형을 거쳐 갱신되는 숨은 상태 하나로 정의된다. 단순하지만 기본 RNN을 제대로 이해하는 일은 꼭 필요하다. 더 정교한 모든 순환 구조가 딛고 서는 계산 방식을 여기서 세우기 때문이다.

갱신식은 다음과 같다.

$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$

$$y_t = W_{hy} h_t + b_y$$

$\tanh$ 활성화는 숨은 상태 값을 $[-1, 1]$으로 묶어 끝없이 커지는 것을 막으면서도 양수와 음수 활성값을 모두 허용한다. 이 선택은 기울기의 흐름에 중요한 결과를 낳는데, BPTT 절에서 살펴본다.

---

## 2. 밑바닥부터 구현하기

### RNN 세포

세포는 순환의 한 시각을 계산한다.

```python
import torch
import torch.nn as nn

class VanillaRNNCell(nn.Module):
    """한 시각을 계산하는 RNN 세포 하나."""
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 알아보기 쉽도록 선형층을 따로 둠
        self.W_xh = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        
    def forward(self, x, h_prev):
        """
        인수:
            x: 입력 텐서 (배치 크기, input_size)
            h_prev: 이전 숨은 상태 (배치 크기, hidden_size)
        반환값:
            h: 새 숨은 상태 (배치 크기, hidden_size)
        """
        h = torch.tanh(self.W_xh(x) + self.W_hh(h_prev))
        return h
```

편향은 `W_xh`에만 둔다. 둘 다에 두면 편향 벡터 둘이 그냥 더해질 뿐이라 군더더기이다.

### 완전한 RNN 모듈

전체 모듈은 순차열에 걸쳐 세포를 되풀이한다.

```python
class VanillaRNN(nn.Module):
    """순차열 전체를 처리하는 온전한 RNN."""
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = VanillaRNNCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, h_0=None):
        """
        인수:
            x: 입력 순차열 (배치 크기, seq_len, input_size)
            h_0: 처음 숨은 상태 (배치 크기, hidden_size)
        반환값:
            outputs: 모든 은닉 상태 (batch_size, seq_len, hidden_size)
            h_n: 마지막 숨은 상태 (배치 크기, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        
        if h_0 is None:
            h_0 = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        h = h_0
        outputs = []
        
        for t in range(seq_len):
            h = self.rnn_cell(x[:, t, :], h)
            outputs.append(h)
        
        outputs = torch.stack(outputs, dim=1)  # (배치, seq_len, hidden)
        return outputs, h
```

시각을 도는 명시적인 반복문이 순차적인 성질을 뚜렷이 보여 준다. 숨은 상태마다 앞의 것에 기대므로 병렬 처리가 막힌다.

---

## 3. PyTorch 내장 RNN 쓰기

PyTorch는 CUDA 핵을 합친 최적화된 `nn.RNN`을 제공한다.

```python
rnn = nn.RNN(
    input_size=10,      # 입력의 특징 차원
    hidden_size=20,     # 숨은 상태의 차원
    num_layers=1,       # 쌓은 RNN 층의 수
    batch_first=True,   # 입력의 모양: (배치, seq, 특징)
    nonlinearity='tanh' # 활성화 함수 ('tanh' 또는 'relu')
)

# 입력: (batch_size=32, seq_len=15, input_size=10)
x = torch.randn(32, 15, 10)

# 순전파
outputs, h_n = rnn(x)

print(f"Outputs shape: {outputs.shape}")  # (32, 15, 20)
print(f"Final hidden: {h_n.shape}")       # (1, 32, 20)
```

`outputs` 텐서는 시각마다의 숨은 상태를 담고 `h_n`은 마지막 숨은 상태만 담는다. `h_n`의 첫 차원은 `num_layers`에 대응하며 한 층짜리 RNN에서는 1이다.

### 가중치에 접근하기

```python
# PyTorch는 우리 표기와 다르게 가중치의 이름을 붙인다
print(rnn.weight_ih_l0.shape)  # (20, 10) — W_xh (입력에서 숨은 상태로)
print(rnn.weight_hh_l0.shape)  # (20, 20) — W_hh (숨은 상태에서 숨은 상태로)
print(rnn.bias_ih_l0.shape)    # (20,)    — b_xh
print(rnn.bias_hh_l0.shape)    # (20,)    — b_hh
```

---

## 4. 순차열 분류

순차열 전체를 하나의 범주로 분류할 때는 대체로 마지막 숨은 상태를 순차열의 표현으로 쓴다.

```python
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x: (배치, seq_len) — 토큰 색인
        embedded = self.embedding(x)        # (배치, seq_len, embed_dim)
        outputs, h_n = self.rnn(embedded)   # h_n: (1, 배치, hidden)
        
        h_final = h_n.squeeze(0)            # (배치, hidden)
        logits = self.fc(h_final)           # (배치, num_classes)
        return logits
```

이론적으로 마지막 숨은 상태 $h_T$이 순차열의 모든 정보를 담지만, 실제로 기본 RNN은 앞쪽 시각의 정보를 지키는 데 애를 먹는다.

---

## 5. 자기회귀 생성

텍스트를 만들 때 모델은 시각마다 출력을 내고 그 예측을 다시 입력으로 넣는다.

```python
class RNNGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        outputs, hidden = self.rnn(embedded, hidden)
        logits = self.fc(outputs)  # (배치, seq_len, vocab_size)
        return logits, hidden
    
    def generate(self, start_token, max_length, temperature=1.0):
        """자기회귀 생성."""
        tokens = [start_token]
        hidden = None
        
        for _ in range(max_length - 1):
            x = torch.tensor([[tokens[-1]]])
            logits, hidden = self.forward(x, hidden)
            
            # 온도를 적용한 분포에서 뽑기
            probs = torch.softmax(logits[0, -1] / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            tokens.append(next_token)
            
        return tokens
```

생성하는 동안 숨은 상태가 호출 사이에 이어진다. 새 토큰마다 $h_t$에 눌러 담긴 온전한 이력에 조건이 걸린다.

---

## 6. 활성화 함수의 선택

기본 RNN은 활성화 함수 두 가지를 지원한다.

**$\tanh$** (기본값): 출력이 $[-1, 1]$이고 도함수가 $1 - \tanh^2(x) \in [0, 1]$이다. 도함수가 묶여 있어 기울기 소실에 한몫하지만 활성값이 안정된다.

**ReLU**: 출력이 $[0, \infty)$이고 도함수가 0 아니면 1이다. 기울기 소실을 누그러뜨릴 수 있지만 숨은 상태가 더는 묶여 있지 않아 활성값이 폭발할 위험이 있다. ReLU를 쓰는 RNN은 초기화를 신경 써야 한다. Le 등(2015)은 ReLU 활성화와 함께 $W_{hh}$을 항등 행렬로 초기화하면 어떤 과제에서는 LSTM에 맞먹는 성능이 나옴을 보였다.

```python
# ReLU 판본
rnn_relu = nn.RNN(
    input_size=10, hidden_size=20,
    batch_first=True, nonlinearity='relu'
)
```

---

## 7. 초매개변수 길잡이

| 매개변수 | 흔한 범위 | 고려할 점 |
|-----------|--------------|----------------|
| `hidden_size` | 64~512 | 복잡한 과제일수록 크게. 입력의 복잡함을 넘어서면 수확이 준다 |
| `num_layers` | 1~3 | 깊은 RNN 절을 보라 |
| `dropout` | 0.0~0.5 | 층 사이에만. 순환 안에는 두지 않는다 |
| `nonlinearity` | `tanh` | `relu`을 쓰려면 $W_{hh}$을 항등 행렬로 초기화해야 한다 |

---

## 8. 기본 RNN의 한계

기본 RNN의 단순함은 근본적인 한계를 낳는다.

**짧은 실효 기억**: $W_{hh}$을 거듭 곱하면 기울기가 사라지거나 폭발하여 실제 기억의 지평이 시각 10~20개쯤으로 제한된다.

**골라서 기억하지 못함**: 시각마다 같은 $\tanh$ 변환으로 숨은 상태를 덮어쓴다. 정보를 골라서 지키거나 고치거나 잊을 장치가 없는데, 문 달린 구조(LSTM, GRU)는 바로 그 능력을 명시적으로 준다.

**순차적인 계산**: $h_t \to h_{t+1}$ 의존 때문에 시각에 걸친 병렬 처리가 막혀, 모든 자리를 한꺼번에 처리하는 트랜스포머 같은 구조보다 학습이 느리다.

이런 한계 때문에 기본 RNN에서 LSTM과 GRU 구조로 나아가게 되었고, 이들은 숨은 상태를 지나는 정보의 흐름을 다스리는 문 장치를 들여왔다.

---

## 연습문제

**연습문제 1.**
긴 순차열에서 기본 RNN의 병목을 설명하라.

??? success "연습문제 1 풀이"
    숨은 상태 $h_t$이 이력 전체를 크기가 고정된 벡터로 눌러 담아야 한다. 긴 순차열에서는 행렬 곱과 비선형이 거듭되며 앞쪽 정보가 '씻겨 나간다'. 기울기도 지수적으로 사라진다. $\frac{\partial h_T}{\partial h_t} = \prod_{k=t}^{T-1} W_h \text{diag}(\tanh'(\cdot))$이고 스펙트럼 반지름이 1보다 작으면 지수적으로 줄어든다.

---

**연습문제 2.**
tanh 활성화가 기울기 소실 문제에 한몫하는 까닭은 무엇인가?

??? success "연습문제 2 풀이"
    $\tanh'(z) \in (0, 1]$이고 $z=0$에서 최댓값 1을 갖는다. $T$단계를 지나면 기울기에 $\tanh'$ 값이 $T$번 곱해져 작아진다. 여기에 ($\|W_h\| < 1$인) $W_h$이 겹치면 기울기가 지수적으로 사라진다. 그래서 기본 RNN은 10~20단계쯤의 짧은 의존만 배울 수 있다.

---

**연습문제 3.**
순차열 분류를 위한 기본 RNN을 PyTorch로 구현하라.

??? success "연습문제 3 풀이"
    ```python
    rnn = nn.RNN(input_size=50, hidden_size=128, batch_first=True)
    fc = nn.Linear(128, num_classes)
    x = rnn(input_seq)[0][:, -1, :]  # 마지막 숨은 상태
    logits = fc(x)
    ```

---

**연습문제 4.**
기본 RNN과 그에 맞먹는 LSTM의 매개변수 수를 견주어라. LSTM에는 매개변수가 얼마나 더 많은가?

??? success "연습문제 4 풀이"
    기본 RNN은 매개변수가 $(d_x + d_h + 1) \cdot d_h$개이다. LSTM은 (문이 넷이므로) $4 \times (d_x + d_h + 1) \cdot d_h$개이다. 숨은 차원이 같을 때 LSTM의 매개변수는 기본 RNN의 정확히 4배이다.

## 정리하며

기본 RNN은 순환 계산의 바탕을 세운다. 지금 입력과 이전 기억을 $\tanh$ 비선형으로 엮어 숨은 상태를 갱신하는 것이다. 밑바닥부터의 구현은 모든 RNN 변형이 함께 쓰는 핵심 반복문을 드러낸다. PyTorch의 `nn.RNN`이 최적화된 실행을 주지만, 가중치 공유와 순차적 처리, 눌러 담긴 이력으로서의 숨은 상태 같은 밑바탕 원리를 이해해야 학습 문제를 짚어 내고 더 나아간 구조의 필요를 알 수 있다.
