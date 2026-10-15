# 기본 RNN

---

## 1. 구조 개관

기본(엘만) RNN은 가장 단순한 순환 구조로, 시각마다 선형 변환 뒤 $\tanh$ 비선형을 거쳐 갱신되는 은닉 상태 하나로 정의된다. 단순하지만 기본 RNN을 제대로 이해하는 일은 꼭 필요하다. 더 정교한 모든 순환 구조가 딛고 서는 계산 방식을 여기서 세우기 때문이다.

갱신식은 다음과 같다.

$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$

$$y_t = W_{hy} h_t + b_y$$

$\tanh$ 활성화는 은닉 상태 값을 $[-1, 1]$으로 묶어 끝없이 커지는 것을 막으면서도 양수와 음수 활성값을 모두 허용한다. 이 선택은 기울기의 흐름에 중요한 결과를 낳는데, BPTT 절에서 살펴본다.

---

## 2. 밑바닥부터 구현하기

### RNN 셀

RNN 셀은 한 시점의 상태 갱신을 계산한다.

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

사용자 정의 셀은 `W_xh`에만 편향을 둔다. 같은 순전파 함수는 두 편향의 합으로도 표현할 수 있지만, 편향을 하나 또는 둘로 두면 옵티마이저가 갱신하는 매개변수 수와 갱신 과정은 달라질 수 있다. 내장 셀과 비교할 때는 두 편향의 합을 맞춘다.

### 완전한 RNN 모듈

전체 모듈은 시퀀스에 걸쳐 셀을 반복 적용한다.

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

시간 축의 반복문은 이전 은닉 상태에 대한 의존성을 보여 준다. 이 사용자 정의 모듈은 `fc`를 등록하지만 순전파에서 사용하지 않으므로 반환값은 출력층의 예측이 아니라 은닉 상태다. `output_size`는 반환 형태에 영향을 주지 않고 `fc`의 매개변수는 이 순전파의 손실에 기여하지 않는다. 빈 시퀀스는 `stack`에서 처리할 원소가 없으며, 초기 상태를 만들 때 입력의 자료형도 맞춰야 한다.

---

## 3. PyTorch 내장 RNN 쓰기

PyTorch는 장치에 맞게 최적화된 최적화된 `nn.RNN`을 제공한다.

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

**출력:**

```
Outputs shape: torch.Size([32, 15, 20])
Final hidden: torch.Size([1, 32, 20])
```

단방향 예제의 `outputs`는 마지막 층의 시점별 은닉 출력을, `h_n`은 각 층의 최종 상태를 담는다. 일반적인 `h_n`의 첫 축은 층 수와 방향 수의 곱이다. `batch_first`는 입력·출력의 배치 축 위치를 바꾸지만 최종 상태의 축 순서는 바꾸지 않는다.

### 가중치에 접근하기

```python
# PyTorch는 우리 표기와 다르게 가중치의 이름을 붙인다
print(rnn.weight_ih_l0.shape)  # (20, 10) — W_xh (입력에서 숨은 상태로)
print(rnn.weight_hh_l0.shape)  # (20, 20) — W_hh (숨은 상태에서 숨은 상태로)
print(rnn.bias_ih_l0.shape)    # (20,)    — b_xh
print(rnn.bias_hh_l0.shape)    # (20,)    — b_hh
```

**출력:**

```
torch.Size([20, 10])
torch.Size([20, 20])
torch.Size([20])
torch.Size([20])
```

---

## 4. 시퀀스 분류

시퀀스 전체를 하나의 범주로 분류할 때는 대체로 마지막 은닉 상태를 시퀀스의 표현으로 쓴다.

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

마지막 상태 $h_T$는 전체 입력에 의존하지만 모든 정보를 손실 없이 담는 것은 아니다. 패딩이 있는 입력을 그대로 처리하면 PAD에서도 상태가 갱신되므로, 실제 길이를 반영한 상태 선택이나 패킹이 필요하다.

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

생성 중에는 은닉 상태가 호출 사이에 이어져 이전 토큰의 정보를 전달한다. 예제의 생성 함수는 시작 토큰을 포함해 최대 길이를 맞추지만 종료 토큰을 검사하지 않는다. 입력 텐서는 CPU에 생성되므로 GPU 모델에는 장치를 맞춰야 한다. 온도는 양수여야 하고, 실제 추론에서는 평가 모드와 기울기 기록 중지를 설정해야 한다. 이 함수 자체에는 그 처리가 없다.

---

## 6. 활성화 함수의 선택

기본 RNN은 활성화 함수 두 가지를 지원한다.

**$\tanh$** (기본값): 출력이 $[-1, 1]$이고 도함수가 $1 - \tanh^2(x) \in [0, 1]$이다. 도함수가 묶여 있어 기울기 소실에 한몫하지만 활성값이 안정된다.

**ReLU**: 출력이 $[0, \infty)$이고 도함수가 0 아니면 1이다. 기울기 소실을 누그러뜨릴 수 있지만 은닉 상태가 더는 묶여 있지 않아 활성값이 폭발할 위험이 있다. ReLU를 쓰는 RNN은 초기화를 신경 써야 한다. Le 등(2015)은 ReLU 활성화와 함께 $W_{hh}$을 항등 행렬로 초기화하면 어떤 과제에서는 LSTM에 맞먹는 성능이 나옴을 보였다.

```python
# ReLU 판본
rnn_relu = nn.RNN(
    input_size=10, hidden_size=20,
    batch_first=True, nonlinearity='relu'
)
```

---

## 7. 하이퍼파라미터 길잡이

| 매개변수 | 흔한 범위 | 고려할 점 |
|-----------|--------------|----------------|
| `hidden_size` | 64~512 | 복잡한 과제일수록 크게. 입력의 복잡함을 넘어서면 수확이 준다 |
| `num_layers` | 1~3 | 깊이와 계산량, 검증 성능을 함께 비교한다 |
| `dropout` | 0.0~0.5 | 내장 dropout은 마지막 층을 제외한 층 사이에 적용된다. 층이 하나면 적용되지 않는다 |
| `nonlinearity` | `tanh` | `relu`에서는 초기화와 상태 크기를 점검한다. 항등 초기화는 가능한 선택 중 하나다 |

---

## 8. 기본 RNN의 한계

기본 RNN의 단순함은 근본적인 한계를 낳는다.

**장기 의존성 학습의 어려움**: 시간에 따른 야코비안의 곱이 작아지거나 커지면 앞쪽 입력에 관한 학습 신호가 약해지거나 불안정해진다. 기억 길이가 항상 10~20개 시점으로 제한되는 것은 아니며 가중치와 활성화, 데이터와 학습 설정에 따라 달라진다.

**명시적인 게이트의 부재**: 기본 RNN도 학습한 가중치와 비선형성으로 정보를 유지하거나 버릴 수 있다. 다만 LSTM·GRU처럼 보존과 갱신을 조절하는 별도 게이트는 없다.

**시간 축의 의존성**: 다음 상태 계산에 이전 상태가 필요하다. 이것이 시간 축 병렬화를 제한하지만, 실제 실행 시간의 우열은 길이, 배치 크기, 모델 크기와 하드웨어에 따라 달라진다.

이런 한계 때문에 기본 RNN에서 LSTM과 GRU 구조로 나아가게 되었고, 이들은 은닉 상태를 지나는 정보의 흐름을 다스리는 게이트를 들여왔다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
긴 시퀀스에서 기본 RNN의 학습이 어려워질 수 있는 이유를 야코비안으로 설명하라.

</div>

??? success "연습문제 1 풀이"
    $J_k=\operatorname{diag}(1-h_k^2)W_{hh}$이면 열벡터 기준 $\partial h_T/\partial h_t=J_TJ_{T-1}\cdots J_{t+1}$이다. 각 연산자 노름이 공통 상수 $q<1$로 제한되면 곱의 노름은 $q^{T-t}$ 이하로 줄어든다. 일반적인 비선형 RNN에서는 고정 가중치 행렬의 스펙트럼 반지름 하나만으로 실제 기울기를 모두 판정할 수 없다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
tanh의 포화가 기울기에 주는 영향과 그 설명의 한계를 서술하라.

</div>

??? success "연습문제 2 풀이"
    도함수는 $1-\tanh^2(z)$이며 큰 절댓값의 입력에서는 0에 가까워진다. 이런 인자가 반복되면 기울기 소실에 기여한다. 다만 순환 가중치도 함께 곱해지므로 tanh를 쓴다는 사실만으로 모든 기울기가 감소한다고 단정할 수 없다. 은닉 상태가 유계라는 것과 기울기가 유계라는 것도 다르다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
시퀀스 분류를 위한 기본 RNN을 PyTorch로 구현하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    rnn = nn.RNN(input_size=50, hidden_size=128, batch_first=True)
    fc = nn.Linear(128, num_classes)
    x = rnn(input_seq)[0][:, -1, :]  # 마지막 숨은 상태
    logits = fc(x)
    ```

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
같은 입력·은닉 차원의 단일 기본 RNN과 투영 없는 LSTM의 매개변수 수를 비교하라. 편향 규약을 명시하라.

</div>

??? success "연습문제 4 풀이"
    편향 하나인 식에서는 기본 RNN이 $h(x+h+1)$개, LSTM이 $4h(x+h+1)$개다. PyTorch 기본 bias=True에서는 입력·순환 편향이 별도여서 각각 $h(x+h+2)$개와 $4h(x+h+2)$개다. 두 경우 모두 동일한 규약에서는 4배이며 출력층과 임베딩은 제외한다. LSTM의 네 묶음은 입력·망각·출력 게이트와 셀 후보 계산이다.

## 정리하며

기본 RNN은 입력과 이전 은닉 상태를 결합하는 셀을 반복한다. 내장 모듈과 직접 구현을 비교할 때는 편향, 반환 상태와 실제 사용되는 층을 맞춰야 한다. 긴 시퀀스의 어려움은 야코비안의 곱과 정보 요약에서 생기며, 고정된 기억 길이로 설명할 수 없다.
