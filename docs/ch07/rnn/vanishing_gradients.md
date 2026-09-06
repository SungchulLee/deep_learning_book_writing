# 기울기 소실
## 핵심 문제

시간을 거슬러 역전파할 때 기울기는 많은 시각을 거슬러 흘러야 한다. 순환 관계를 지나며 곱셈이 거듭되면 기울기가 지수적으로 작아져, 순차열에서 멀리 떨어진 원소 사이의 의존을 신경망이 배울 수 없게 된다. 이것이 기본 RNN의 가장 큰 한계인 **기울기 소실 문제**이다.

## 수학적 분석

### 시간을 거스르는 기울기의 흐름

앞쪽 숨은 상태 $h_t$에 대한 손실의 기울기에는 야코비 행렬의 곱이 들어 있다.

$$\frac{\partial \mathcal{L}}{\partial h_t} = \frac{\partial \mathcal{L}}{\partial h_T} \cdot \prod_{k=t}^{T-1} \frac{\partial h_{k+1}}{\partial h_k}$$

야코비 항 하나하나는 다음과 같다.

$$\frac{\partial h_{k+1}}{\partial h_k} = \text{diag}(\sigma'(z_k)) \cdot W_{hh}$$

여기서 $\sigma'$은 활성화 함수의 도함수이고 $z_k$은 활성화 전 값이다.

### 기울기 노름의 한계

야코비 곱의 노름은 다음을 만족한다.

$$\left\| \prod_{k=t}^{T-1} \frac{\partial h_{k+1}}{\partial h_k} \right\| \leq \prod_{k=t}^{T-1} \left\| \frac{\partial h_{k+1}}{\partial h_k} \right\|$$

$\gamma = \max_k \left\| \frac{\partial h_{k+1}}{\partial h_k} \right\|$이라 하자. $\gamma < 1$이면 다음이 성립한다.

$$\left\| \frac{\partial \mathcal{L}}{\partial h_t} \right\| \leq \left\| \frac{\partial \mathcal{L}}{\partial h_T} \right\| \cdot \gamma^{T-t} \xrightarrow{T-t \to \infty} 0$$

기울기는 시간 거리 $T - t$에 따라 지수적으로 줄어든다.

### 왜 $\gamma < 1$이 흔한 경우인가

두 가지 요인이 겹쳐 $\gamma < 1$이 기본 거동이 된다.

**활성화 함수의 포화.** $\tanh$의 도함수는 다음과 같다.

$$\tanh'(x) = 1 - \tanh^2(x)$$

이 도함수는 $x = 0$에서만 최댓값 1을 갖고 $|x|$이 커질수록 0에 가까워진다. 숨은 상태의 활성값이 0 둘레에 모여 있지 않으면(학습이 진행될수록 흔해진다) $\text{diag}(\sigma'(z_k))$ 인수가 기울기를 줄인다.

**순환 가중치의 스펙트럼 노름.** $\tanh'(x) \approx 1$이더라도 $\|W_{hh}\| < 1$이면(곧 $W_{hh}$의 가장 큰 특잇값이 1보다 작으면) 곱셈이 거듭되며 기울기가 0으로 간다.

$$\|W_{hh}^{T-t}\| \to 0 \quad \text{as } T-t \to \infty$$

### 고윳값의 관점

$W_{hh}$의 고유 분해가 더 깊은 통찰을 준다. $W_{hh} = U \Lambda U^{-1}$이면 $W_{hh}^n = U \Lambda^n U^{-1}$이다. $|\lambda_i| < 1$인 고유벡터 방향의 기울기 성분은 $|\lambda_i|^n$으로 줄고, $|\lambda_i| > 1$인 성분은 커진다. 흔한 무작위 초기화에서는 대부분의 고윳값이 $|\lambda_i| < 1$이므로 앞쪽 시각의 기울기 정보가 체계적으로 파괴된다.

## 결과

**먼 거리 의존을 배우지 못한다.** 모델이 앞쪽 입력과 뒤쪽 출력을 이어 주지 못한다. 자리 5의 낱말이 자리 50의 올바른 이름표를 정한다면, 자리 5에 닿는 기울기 신호는 무시할 만큼 작다.

**앞쪽 시각은 거의 갱신되지 않는다.** 순차열의 앞부분을 다루는 가중치는 사실상 얼어붙는다. 뜻있는 변화를 낼 만큼의 기울기를 받지 못한다.

**실효 기억이 짧다.** 이론적으로는 이력 전체에 닿을 수 있지만, 실제로 모델은 가장 최근 시각 10~20개만 보는 것처럼 움직인다.

## 실험으로 보이기

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def demonstrate_vanishing_gradients():
    """시각을 지나며 기울기가 줄어드는 모습을 보인다."""
    hidden_size = 100
    seq_length = 50
    
    rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
    
    # 가중치를 작게 초기화하면 기울기가 더 잘 사라진다
    for name, param in rnn.named_parameters():
        if 'weight' in name:
            nn.init.uniform_(param, -0.1, 0.1)
    
    x = torch.randn(1, seq_length, hidden_size, requires_grad=True)
    outputs, h_n = rnn(x)
    
    # 마지막 출력에서 입력 자리마다의 기울기 재기
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

로그 눈금 그림이 지수적인 감소를 드러낸다. 시각 0의 기울기는 마지막 시각보다 $10^{6}$배에서 $10^{10}$배까지 작을 수 있다.

## 진단

### 기울기 노름 살피기

```python
class VanishingGradientDetector:
    """학습 중에 층에 걸친 기울기의 노름을 좇는다."""
    
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
        """순환 가중치의 기울기가 사라지고 있는지 확인한다."""
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
| 손실이 일찍 정체됨 | 처음에 떨어진 뒤 학습 손실이 더 줄지 않는다 |
| 긴 순차열에서 정확도가 나쁨 | 짧은 순차열에서는 잘하는데 긴 것에서는 실패한다 |
| `weight_hh`의 기울기가 거의 0 | 순환 가중치의 기울기 노름이 입력 가중치보다 몇 자릿수 작다 |
| 숨은 상태가 무너짐 | 앞쪽 입력이 무엇이든 마지막 숨은 상태가 비슷해진다 |

## 누그러뜨리는 방법

### 가중치 초기화

$W_{hh}$을 직교 행렬로 초기화하면 처음에 모든 특잇값이 1이 되어 기울기 흐름에 가장 좋은 출발점이 된다.

```python
def init_for_gradient_preservation(rnn):
    """기울기 소실을 누그러뜨리려고 RNN의 가중치를 초기화한다."""
    for name, param in rnn.named_parameters():
        if 'weight_ih' in name:
            nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:
            nn.init.orthogonal_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
```

직교 행렬은 모든 $x$에 대해 $\|Qx\| = \|x\|$을 만족하므로 곱셈이 거듭되어도 기울기의 노름이 지켜진다. 적어도 학습이 가중치를 바꾸기 전인 초기화 시점에는 그렇다.

### 건너뛰기 연결

잔차 연결은 순환을 건너뛰는 곧바른 기울기 경로를 준다.

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

건너뛰기 연결 덕분에 기울기가 $W_{hh}$ 곱셈의 사슬을 모두 지나지 않고도 출력에서 입력으로 곧바로 흐를 수 있다.

### 층 정규화

숨은 상태의 활성값을 정규화하면 포화를 막고 $\tanh'$이 큰 영역에 값을 붙들어 둘 수 있다.

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

### 문 달린 구조 (근본적인 해법)

LSTM과 GRU 구조는 덧셈으로 갱신하는 경로로 기울기 소실 문제를 푼다. LSTM의 세포 상태 갱신식은 다음과 같다.

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

이는 기울기의 고속도로를 낸다. $\frac{\partial c_t}{\partial c_{t-1}} = f_t$이며 이는 $[0, 1]$의 값을 성분별로 곱하는 간단한 연산이다. $f_t \approx 1$(망각 문이 열림)이면 기울기가 그대로 흐르고 $W_{hh}$을 거듭 곱하는 일이 없다.

```python
# 그대로 갈아 끼우기: LSTM은 기울기의 흐름이 안정적이다
lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

# GRU: 기울기의 이점은 비슷하면서 더 간단한 대안
gru = nn.GRU(input_size, hidden_size, batch_first=True)
```

## 요약

기울기 소실 문제는 순환을 지나며 곱셈이 거듭되는 데서 생긴다. 시각마다 기울기에 $\text{diag}(\tanh') \cdot W_{hh}$이 곱해지고, 이 곱의 스펙트럼 노름이 1보다 작으면 기울기가 시간 거리에 따라 지수적으로 줄어든다. 그래서 먼 거리 의존을 배우지 못하고 기본 RNN이 사실상 기억이 짧은 모델이 된다. 초기화와 건너뛰기 연결과 정규화가 얼마간 도움이 되지만, 근본적인 해법은 덧셈 기울기 경로를 주는 문 달린 구조(LSTM, GRU)이다.

## 연습문제

**연습문제 1.**
시각 $T$개를 지나는 기울기의 흐름을 유도하고 $\|W_h\| < 1$일 때 지수적으로 줄어듦을 보여라.

??? success "연습문제 1 풀이"
    $\frac{\partial L}{\partial h_t} = \frac{\partial L}{\partial h_T} \prod_{k=t+1}^T \text{diag}(\tanh'(z_k)) W_h$이다. $\sigma_{\max}(W_h) < 1$이고 $|\tanh'| \leq 1$이면 $\|\frac{\partial L}{\partial h_t}\| \leq \sigma_{\max}(W_h)^{T-t} \|\frac{\partial L}{\partial h_T}\|$이고, $T - t$이 커질수록 지수적으로 0에 간다.

---

**연습문제 2.**
RNN의 기울기 소실을 누그러뜨리는 기법 네 가지를 열거하라.

??? success "연습문제 2 풀이"
    (1) 문 장치를 갖춘 LSTM과 GRU. (2) 기울기 자르기(폭발하는 기울기용). (3) 건너뛰기 연결과 잔차 연결. (4) 알맞은 초기화($W_h$의 직교 초기화). 그 밖에 더 짧은 순차열, 어텐션 장치, 잘라 낸 BPTT도 있다.

---

**연습문제 3.**
$W_h$의 직교 초기화가 기울기 소실에 도움이 되는 까닭을 설명하라.

??? success "연습문제 3 풀이"
    직교 행렬은 모든 특잇값이 1이므로 모든 $T$에 대해 $\|W_h^T\| = 1$이다. 곧 기울기가 시간을 거슬러 흐르는 동안 커지지도 작아지지도 않는다. 실제로는 비선형 때문에 얼마간 줄어들지만, 직교 초기화가 무작위 가우스 초기화보다 훨씬 나은 출발점을 준다.

---

**연습문제 4.**
경사 폭발 문제란 무엇이며 경사 자르기가 이를 어떻게 다루는가?

??? success "연습문제 4 풀이"
    $\sigma_{\max}(W_h) > 1$이면 기울기가 지수적으로 커진다. $\|g\| \propto \sigma_{\max}^T$이다. 그러면 매개변수가 엄청나게 갱신되어 학습이 불안정해진다. 기울기 자르기는 $g \leftarrow g \cdot \min(1, \theta/\|g\|)$으로 크기를 다시 맞추어 방향은 지키면서 노름을 $\theta$으로 묶는다.
