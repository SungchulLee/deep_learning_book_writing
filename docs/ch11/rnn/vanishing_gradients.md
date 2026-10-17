# 그래디언트 소실

---

## 1. 핵심 문제

시간을 거슬러 역전파할 때 그래디언트는 많은 시각을 거슬러 흘러야 한다. 순환 관계를 지나며 곱셈이 거듭되면 그래디언트가 지수적으로 작아져, 시퀀스에서 멀리 떨어진 원소 사이의 의존을 뉴럴 넷트웍이 배울 수 없게 된다. 이것이 기본 RNN의 가장 큰 한계인 **그래디언트 소실 문제**이다.

---

## 2. 수학적 분석

마지막 시점 로스 $L_T$만 있을 때, 열벡터 그래디언트는 다음과 같이 전달된다.

$$\nabla_{h_t}L_T=J_{t+1}^{\top}J_{t+2}^{\top}\cdots J_T^{\top}\nabla_{h_T}L_T$$

$$J_k=\frac{\partial h_k}{\partial h_{k-1}}=\operatorname{diag}(\sigma'(z_k))W_{hh}$$

여기서 $z_k$는 시점 $k$의 활성화 전 값이다. 여러 시점에 로스가 있으면 각 로스에서 오는 기여를 합한다.

### 노름으로 보는 충분조건

모든 시점에서 $\|J_k\|_2\le q<1$이고 마지막 그래디언트가 유계이면 다음 상한을 얻는다.

$$\|\nabla_{h_t}L_T\|_2\le q^{T-t}\|\nabla_{h_T}L_T\|_2$$

이는 긴 거리의 기여가 지수적으로 줄어드는 충분조건이다. tanh의 도함수는 1 이하이고 포화 구간에서는 0에 가까워진다. 따라서 $\|W_{hh}\|_2<1$은 tanh RNN에서 이런 감소를 보이는 충분조건이지만, 모든 RNN이 이 조건을 만족하는 것은 아니다.

### 고윳값과 특잇값의 구분

선형 순환에서는 $W_{hh}^n$의 장기 거동을 고윳값으로 분석할 수 있다. 다만 고유 분해는 대각화 가능한 경우에 적용되며, 비정규 행렬에서는 일시적인 증폭도 생길 수 있다. 비선형 RNN의 실제 전달은 시점마다 달라지는 $J_k$의 곱이므로 $W_{hh}$의 고윳값만으로 그래디언트의 크기를 판정할 수 없다. 최대 특잇값이 1보다 크다는 사실도 실제 경로의 폭발을 보장하지 않는다.

## 3. 결과

먼 시점의 로스에서 앞쪽 상태로 오는 그래디언트가 작아지면 장기 의존성을 학습하기 어려워진다. 다만 시점별로 별도의 웨이트를 쓰는 것은 아니다. 공유 웨이트는 최근 시점이나 다른 예제의 기여로 계속 갱신될 수 있다.

실제 기억 길이는 과제, 웨이트, 활성화와 학습 설정에 달려 있다. 모든 기본 RNN이 10~20개 시점만 기억한다는 고정 한계로 해석해서는 안 된다. 그래디언트 소실이 강하다는 것과 과거 정보가 상태에서 완전히 사라졌다는 것도 같은 주장은 아니다.

## 4. 실험으로 보이기

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def demonstrate_vanishing_gradients():
    """시각을 지나며 그래디언트가 줄어드는 모습을 보인다."""
    hidden_size = 100
    seq_length = 50
    
    rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
    
    # 웨이트를 작게 초기화하면 그래디언트가 더 잘 사라진다
    for name, param in rnn.named_parameters():
        if 'weight' in name:
            nn.init.uniform_(param, -0.1, 0.1)
    
    x = torch.randn(1, seq_length, hidden_size, requires_grad=True)
    outputs, h_n = rnn(x)
    
    # 마지막 출력에서 입력 자리마다의 그래디언트 재기
    gradient_norms = []
    for t in range(seq_length):
        if x.grad is not None:
            x.grad.zero_()
        outputs[0, t].sum().backward(retain_graph=True)
        gradient_norms.append(x.grad[0, 0].norm().item())
    
    plt.figure(figsize=(10, 4))
    plt.plot(range(seq_length), gradient_norms[::-1])
    plt.xlabel('Distance from Output (timesteps)')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Vanishing in Vanilla RNN')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.show()

demonstrate_vanishing_gradients()
```

이 코드는 마지막 출력 하나를 고정한 채 입력 위치를 바꾸지 않는다. 첫 입력 위치를 고정하고 출력 시점을 바꾸어 그 출력 합의 입력 그래디언트를 계산한다. 저장한 값을 역순으로 그리므로 가로축의 값은 실제 시간 간격과 반대로 대응한다. 따라서 축 제목 그대로 거리별 그래디언트라고 읽으면 안 된다. 감소 정도와 수치는 초기값과 입력에 달려 있으며 특정 배수로 보장되지 않는다.

---

## 5. 진단

### 그래디언트 노름 살피기

```python
class VanishingGradientDetector:
    """학습 중에 레이어에 걸친 그래디언트의 노름을 좇는다."""
    
    def __init__(self, model):
        self.model = model
        self.gradient_history = []
    
    def log_gradients(self):
        grad_stats = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_stats[name] = {
                    'norm': param.grad.norm().item(),
                    'mean_abs': param.grad.abs().mean().item(),
                    'max_abs': param.grad.abs().max().item()
                }
        self.gradient_history.append(grad_stats)
    
    def check_vanishing(self, threshold=1e-7):
        """순환 웨이트의 그래디언트가 사라지고 있는지 확인한다."""
        if not self.gradient_history:
            return False
        latest = self.gradient_history[-1]
        for name, stats in latest.items():
            if 'weight_hh' in name and stats['norm'] < threshold:
                print(f"⚠️  Vanishing gradient: {name} norm = {stats['norm']:.2e}")
                return True
        return False
```

### 징후

| 징후 | 무엇을 볼 것인가 |
|-----------|---------------|
| 로스가 일찍 정체됨 | 처음에 떨어진 뒤 학습 로스가 더 줄지 않는다 |
| 긴 시퀀스에서 정확도가 나쁨 | 짧은 시퀀스에서는 잘하는데 긴 것에서는 실패한다 |
| `weight_hh`의 그래디언트가 거의 0 | 순환 웨이트의 그래디언트 노름이 입력 웨이트보다 몇 자릿수 작다 |
| 은닉 상태가 무너짐 | 앞쪽 입력이 무엇이든 마지막 은닉 상태가 비슷해진다 |

---

## 6. 누그러뜨리는 방법

### 웨이트 초기화

$W_{hh}$를 직교 행렬로 초기화하면 선형 변환 자체의 특잇값은 1이 된다. 순환 선형 변환이 그래디언트를 줄이거나 키우는 요인을 완화할 수 있지만, 비선형성과 실제 상태 궤적까지 포함한 최적 초기화를 보장하지는 않는다.

```python
def init_for_gradient_preservation(rnn):
    """그래디언트 소실을 누그러뜨리려고 RNN의 웨이트를 초기화한다."""
    for name, param in rnn.named_parameters():
        if 'weight_ih' in name:
            nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:
            nn.init.orthogonal_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
```

직교 행렬 $Q$는 $\|Qv\|_2=\|v\|_2$를 만족한다. 하지만 RNN의 야코비안에는 활성화 도함수도 포함되므로 초기화 직후라도 전체 그래디언트 노름이 보존되는 것은 아니다. 학습으로 웨이트가 바뀌면 직교성도 유지된다고 가정할 수 없다.

### 건너뛰기 연결

잔차 연결의 효과는 어느 축을 건너뛰는지에 따라 다르다. 아래 코드는 현재 입력의 투영을 현재 출력에 더한다.

```python
class ResidualRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn_cell = nn.RNNCell(input_size, hidden_size)
        self.projection = nn.Linear(input_size, hidden_size)
    
    def forward(self, x, h):
        h_new = self.rnn_cell(x, h)
        return h_new + self.projection(x)  # 잔차 경로
```

입력 투영 경로는 현재 입력에서 출력으로 가는 직접 경로다. 이전 은닉 상태에 대한 야코비안에는 항등 경로가 추가되지 않으므로, 이 코드만으로 시간 축의 순환 곱을 건너뛴다고 설명할 수 없다. 시간 축에서 이전 상태를 직접 더하는 연결과 구분해야 한다.

### 레이어 정규화

아래 코드는 활성화 전 결합값에 LayerNorm을 적용한다. 값의 스케일을 조절할 수 있지만 tanh 포화를 항상 막거나 큰 도함수를 보장하지는 않는다. 정규화 자체의 야코비안과 학습 가능한 배율도 그래디언트에 영향을 준다.

```python
class LayerNormRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_xh = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x, h):
        combined = self.W_xh(x) + self.W_hh(h)
        normalized = self.layer_norm(combined)
        return torch.tanh(normalized)
```

### 게이트를 이용한 완화

LSTM과 GRU는 게이트와 덧셈 갱신 경로를 이용해 장기 정보 전달을 돕는다. 모든 그래디언트 소실을 제거하는 것은 아니다. LSTM의 셀 상태 갱신식은 다음과 같다.

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

게이트와 후보 값을 고정한 직접 경로에서는 셀 상태 미분이 $\operatorname{diag}(f_t)$다. 망각 게이트가 1에 가까우면 그 경로의 그래디언트를 유지하기 쉽다. 다만 전체 미분에는 이전 은닉 상태가 게이트·후보 계산에 미치는 간접 경로도 포함된다. 망각 게이트가 지속해서 1보다 작으면 직접 경로의 곱도 줄어들 수 있다.

```python
# 그대로 갈아 끼우기: LSTM은 그래디언트의 흐름이 안정적이다
lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

# GRU: 그래디언트의 이점은 비슷하면서 더 간단한 대안
gru = nn.GRU(input_size, hidden_size, batch_first=True)
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
마지막 로스에서 과거 상태로 오는 그래디언트를 쓰고, tanh RNN에서 순환 웨이트의 스펙트럼 노름이 1 미만일 때의 상한을 유도하라.

</div>

??? success "풀이"
    $g_t=J_{t+1}^{\top}\cdots J_T^{\top}g_T$다. tanh 도함수의 절댓값이 1 이하이므로 $\|J_k\|_2\le\|W_h\|_2=q<1$이면 $\|g_t\|_2\le q^{T-t}\|g_T\|_2$다. 이는 마지막 로스의 기여에 대한 상한이며 중간 로스의 국소 기여까지 모두 사라진다는 뜻은 아니다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
그래디언트 소실을 완화하는 설계와 그래디언트 폭발을 제한하는 방법을 구분하라.

</div>

??? success "풀이"
    게이트 상태 경로, 적절한 초기화, 연결 경로 변경과 정규화는 장거리 그래디언트 전달을 도울 수 있지만 각각 한계가 있다. 노름 클리핑은 큰 그래디언트를 줄이는 방법으로, 이미 사라진 그래디언트를 되살리지는 않는다. 절단 BPTT는 역전파 경로를 제한하므로 장거리 그래디언트 소실의 근본적인 해결책과도 다르다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
직교 초기화가 선형 변환의 노름을 보존해도 전체 RNN 그래디언트를 보장하지 않는 이유를 설명하라.

</div>

??? success "풀이"
    직교 순환 행렬은 벡터의 2-노름을 보존하지만 실제 야코비안은 활성화 도함수 행렬과의 곱이다. tanh 포화가 그래디언트를 줄일 수 있고, 학습 후에는 직교성도 보장되지 않는다. 초기화의 도움과 전체 시간 경로의 보존을 구분해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
그래디언트 폭발과 노름 클리핑의 동작을 설명하라.

</div>

??? success "풀이"
    야코비안 곱 등의 영향으로 그래디언트가 매우 커져 큰 업데이트가 발생할 수 있다. 유한한 그래디언트 $g$의 노름 클리핑은 $g\leftarrow g\min(1,\theta/\|g\|)$로 방향을 유지하면서 전체 노름을 제한한다. 순환 행렬의 최대 특잇값이 1보다 크다는 조건만으로 실제 폭발이 반드시 일어나는 것은 아니다. 이미 생긴 NaN이나 무한대를 자동으로 복구하는 방법도 아니다.

## 정리하며

그래디언트 소실은 긴 경로의 야코비안 곱이 작은 전달량을 만드는 현상이다. 공유 웨이트의 갱신, 상태의 정보 보존과 시간 경로의 그래디언트를 구분해야 한다. 초기화·정규화·게이트는 완화책이며, 클리핑은 폭발을 제한하는 별도의 도구다.
