# LSTM의 세포 상태와 기울기의 흐름

LSTM 구조는 순환 신경망의 기울기 소실 문제를 풀려고 일부러 만든 것이다. 그 한가운데에 **세포 상태**가 있다. 정보와 기울기가 거의 변형되지 않고 여러 시각을 가로질러 흐르는 전용 기억 통로이다. LSTM에서 기울기가 어떻게 흐르는지 이해하면 기본 RNN이 실패하는 곳에서 LSTM이 성공하는 까닭을 알 수 있고, LSTM 자신이 애먹는 때도 짚어 낼 수 있다.

---

## 1. 근본 문제: 기본 RNN이 실패하는 까닭

### 기본 RNN의 기울기 분석

숨은 상태를 $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$으로 갱신하는 기본 RNN에서, 시각 $t-k$의 숨은 상태에 대한 손실의 기울기에는 다음이 들어 있다.

$$\frac{\partial \mathcal{L}}{\partial h_{t-k}} = \frac{\partial \mathcal{L}}{\partial h_t} \cdot \prod_{j=t-k}^{t-1} \frac{\partial h_{j+1}}{\partial h_j}$$

야코비 항 하나하나는 다음과 같다.

$$\frac{\partial h_{j+1}}{\partial h_j} = \text{diag}(\tanh'(z_{j+1})) \cdot W_{hh}$$

여기서 $z_{j+1} = W_{hh} h_j + W_{xh} x_{j+1} + b$이다.

### 고윳값의 관점

야코비 행렬의 곱은 다음처럼 움직인다.

$$\prod_{j=t-k}^{t-1} \frac{\partial h_{j+1}}{\partial h_j} \approx (D \cdot W_{hh})^k$$

여기서 $D$은 성분이 $(0, 1)$에 있는 대각 행렬이다($\tanh'(x) \in (0, 1]$이므로).

$\lambda_{\max}$을 $W_{hh}$의 가장 큰 특잇값이라 하자.

- $\lambda_{\max} < 1$이면 $\lambda_{\max}^k \to 0$이므로 기울기가 지수적으로 사라진다
- $\lambda_{\max} > 1$이면 기울기가 지수적으로 폭발한다
- $\lambda_{\max} \approx 1$이라는 "딱 알맞은 구간"은 불안정하여 지키기 어렵다

이것이 LSTM이 푸는 **근본적인 딜레마**이다.

---

## 2. 기울기의 고속도로: 세포 상태

### 구조의 핵심 혁신

LSTM은 덧셈 갱신 규칙을 갖춘 별도의 **세포 상태** $c_t$을 들여온다.

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

이는 기본 RNN의 곱셈 갱신과 근본적으로 다르다. 세포 상태는 **기울기의 고속도로**, 곧 기울기가 거의 변형되지 않고 흐르는 길 노릇을 한다.

### 세포 상태를 지나는 기울기: 엄밀한 유도

$c_{t-1}$에 대해 미분하면 다음과 같다.

$$\frac{\partial c_t}{\partial c_{t-1}} = \frac{\partial}{\partial c_{t-1}}\left[f_t \odot c_{t-1} + i_t \odot \tilde{c}_t\right]$$

곱의 미분법을 적용하면 다음과 같다.

$$\frac{\partial c_t}{\partial c_{t-1}} = \text{diag}(f_t) + \underbrace{c_{t-1} \odot \frac{\partial f_t}{\partial c_{t-1}} + \tilde{c}_t \odot \frac{\partial i_t}{\partial c_{t-1}} + i_t \odot \frac{\partial \tilde{c}_t}{\partial c_{t-1}}}_{\text{indirect paths through } h_{t-1}}$$

**결정적인 관찰**: 첫 항 $\text{diag}(f_t)$이 막힘없는 곧바른 기울기 경로를 준다. $f_t \approx 1$이면 다음과 같다.

$$\frac{\partial c_t}{\partial c_{t-1}} \approx I + \text{(small indirect terms)}$$

곧 야코비 행렬이 항등 행렬에 가까워 기울기가 그대로 흐른다.

### 일정 오차 회전목마

Hochreiter와 Schmidhuber는 이를 본디 **일정 오차 회전목마(CEC)**라 불렀다. 핵심 성질은 다음과 같다.

$$\frac{\partial c_t}{\partial c_{t-k}} = \prod_{j=t-k}^{t-1} \frac{\partial c_{j+1}}{\partial c_j} \approx \prod_{j=t-k}^{t-1} \text{diag}(f_{j+1})$$

망각 문이 한결같이 1에 가까우면 이 곱이 항등에 가깝게 머문다.

$$\prod_{j=t-k}^{t-1} \text{diag}(f_{j+1}) \approx I$$

**LSTM이 통하는 까닭의 수학적 알맹이가 바로 이것이다.**

---

## 3. 기본 RNN과의 대비

| 항목 | 기본 RNN | LSTM |
|--------|-------------|------|
| 상태 갱신 | $h_t = \tanh(W_{hh}h_{t-1} + \dots)$ | $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ |
| 기울기 경로 | 곱셈 ($W_{hh}$을 지남) | 덧셈 ($f_t$을 지남) |
| 야코비 행렬 | $\text{diag}(\tanh') \cdot W_{hh}$ | $\text{diag}(f_t)$에 간접 항 |
| 고윳값 | $W_{hh}$을 꼼꼼히 맞추어야 한다 | $f_t$을 시각마다 배운다 |
| 먼 거리 | 지수적으로 줄거나 는다 | $\approx 1$을 지킬 수 있다 |

### 기울기 감소의 수학적 비교

**기본 RNN** ($\|\text{diag}(\tanh') \cdot W_{hh}\| \approx \gamma$이라 하면):

$$\left\|\frac{\partial h_t}{\partial h_{t-k}}\right\| \approx \gamma^k$$

**LSTM** (망각 문의 평균이 $\bar{f}$이라 하면):

$$\left\|\frac{\partial c_t}{\partial c_{t-k}}\right\| \approx \bar{f}^k + O(\text{indirect paths})$$

차이는 이렇다. $\gamma$은 고정된 가중치와 포화하는 활성화가 정하지만 $\bar{f}$은 **배우는 것**이라 과제에 맞추어 달라질 수 있다.

---

## 4. 온전한 기울기 경로 분석

### 온전한 역전파 식

손실 $\mathcal{L}$에서 시각 $t-k$의 매개변수까지 기울기가 흐르는 길을 따라가 보자.

**1단계**: 출력까지의 기울기

$$\frac{\partial \mathcal{L}}{\partial h_t}$$

**2단계**: 숨은 상태에서 세포 상태로

$$\frac{\partial h_t}{\partial c_t} = \text{diag}(o_t) \cdot \text{diag}(\tanh'(c_t))$$

**3단계**: 세포 상태를 시간을 거슬러

$$\frac{\partial c_t}{\partial c_{t-1}} = \text{diag}(f_t) + \text{indirect terms}$$

**4단계**: 시각마다 기울기 쌓기

$c_{t-k}$까지의 전체 기울기는 여러 몫이 쌓인 것이다.

$$\frac{\partial \mathcal{L}}{\partial c_{t-k}} = \sum_{j=t-k}^{T} \frac{\partial \mathcal{L}}{\partial h_j} \cdot \frac{\partial h_j}{\partial c_j} \cdot \frac{\partial c_j}{\partial c_{t-k}}$$

### 기울기 경로 그려 보기

```
Loss at time T
      ↓
    ∂L/∂h_T ←──────────────────────┐
      ↓                             │
    ∂h_T/∂c_T = o_T ⊙ tanh'(c_T)   │  (output gate modulates)
      ↓                             │
    ∂L/∂c_T                         │
      ↓ ×f_T (DIRECT PATH)          │
    ∂L/∂c_{T-1} ←───── ∂L/∂h_{T-1} ─┘  (gradient also flows via h)
      ↓ ×f_{T-1}
    ∂L/∂c_{T-2}
      ↓
     ...
      ↓
    ∂L/∂c_1

LEGEND:
─── Direct path (gradient highway, scaled by forget gates)
←── Indirect path (through hidden states and gates)
```

### 간접 경로의 몫

간접 경로는 작기는 해도 중요한 기울기 신호를 준다.

$$\frac{\partial c_t}{\partial c_{t-1}}\bigg|_{\text{indirect}} = \frac{\partial c_t}{\partial h_{t-1}} \cdot \frac{\partial h_{t-1}}{\partial c_{t-1}}$$

펼치면 다음과 같다.

$$= \left(\frac{\partial f_t}{\partial h_{t-1}} \odot c_{t-1} + \frac{\partial i_t}{\partial h_{t-1}} \odot \tilde{c}_t + \frac{\partial \tilde{c}_t}{\partial h_{t-1}} \odot i_t\right) \cdot \text{diag}(o_{t-1}) \cdot \text{diag}(\tanh'(c_{t-1}))$$

이 간접 경로 덕분에 기울기가 문의 계산에 영향을 주어, 신경망이 **언제** 기억하고 잊을지 배울 수 있다.

---

## 5. 기울기의 흐름에서 각 문의 구실

### 망각 문: 고속도로의 관리자

망각 문 $f_t$이 기울기 흐름을 **가장 크게 좌우한다**.

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def analyze_forget_gate_impact():
    """
    망각 문의 값이 기울기 전파에 어떤 영향을 주는지 보인다.
    """
    scenarios = {
        'Always forget (f=0.0)': 0.0,
        'Half remember (f=0.5)': 0.5,
        'Mostly remember (f=0.9)': 0.9,
        'Almost always remember (f=0.95)': 0.95,
        'Always remember (f=1.0)': 1.0
    }
    
    sequence_lengths = [10, 50, 100, 200, 500]
    
    print("Gradient Magnitude After k Timesteps")
    print("=" * 70)
    print(f"{'Scenario':<30} " + " ".join(f"k={k:<5}" for k in sequence_lengths))
    print("-" * 70)
    
    for name, f_val in scenarios.items():
        gradients = [f_val ** k for k in sequence_lengths]
        grad_strs = [f"{g:.2e}" if g < 0.01 else f"{g:.4f}" for g in gradients]
        print(f"{name:<30} " + " ".join(f"{s:<6}" for s in grad_strs))
    
    print("\nKey Insight: Even f=0.95 leads to 0.95^100 ≈ 0.006 gradient magnitude")

def visualize_forget_gate_gradient_decay():
    """
    망각 문의 값에 따른 기울기의 감소를 그려 본다.
    """
    forget_values = [0.5, 0.8, 0.9, 0.95, 0.99, 1.0]
    max_length = 200
    
    plt.figure(figsize=(12, 5))
    
    for f_val in forget_values:
        gradients = [f_val ** k for k in range(max_length)]
        plt.semilogy(gradients, label=f'f={f_val}', linewidth=2)
    
    plt.axhline(y=0.01, color='r', linestyle='--', alpha=0.7, label='1% threshold')
    plt.xlabel('Timesteps Back')
    plt.ylabel('Gradient Magnitude (log scale)')
    plt.title('Gradient Decay Through Cell State for Different Forget Gate Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max_length)
    plt.show()
```

### 입력 문: 새 정보 조절하기

입력 문 $i_t$은 기울기에 간접적으로 영향을 준다.

$$\frac{\partial c_t}{\partial \tilde{c}_t} = i_t$$

$i_t \approx 0$이면 신경망이 새 정보를 무시하고 기울기가 후보 계산으로 흐르지 않는다. 신경망이 늘 입력을 무시하는 법을 배우면 문제가 될 수 있다.

### 출력 문: 보임을 다스리는 문

출력 문 $o_t$은 $h_t$에서 $c_t$으로 기울기가 흐르는 방식에 영향을 준다.

$$\frac{\partial h_t}{\partial c_t} = o_t \odot \tanh'(c_t)$$

$o_t \approx 0$이면 세포 상태가 출력에서 "가려져" 그 시각에 손실에서 오는 기울기가 세포 상태에 미치는 영향이 줄어든다.

---

## 6. 실험적 기울기 분석

```python
def analyze_gradient_flow(model, seq_length=100, input_size=64, 
                          num_samples=50, model_type='lstm'):
    """
    여러 RNN 판본의 기울기 흐름을 두루 분석한다.
    
    시각마다 기울기의 노름을 돌려주어, 마지막 출력에서 시간을 거슬러
    기울기가 얼마나 잘 전파되는지 잰다.
    """
    gradient_norms = torch.zeros(seq_length)
    
    for _ in range(num_samples):
        # 무작위 입력
        x = torch.randn(1, seq_length, input_size, requires_grad=True)
        
        # 순전파
        outputs, _ = model(x)
        
        # 마지막 출력에서 역전파
        loss = outputs[0, -1, :].sum()
        loss.backward()
        
        # 시각마다 기울기 재기
        for t in range(seq_length):
            gradient_norms[t] += x.grad[0, t, :].norm().item()
        
        # 다음 표본을 위해 초기화
        x.grad.zero_()
    
    gradient_norms /= num_samples
    return gradient_norms

def comprehensive_gradient_comparison():
    """
    RNN과 LSTM과 GRU의 기울기 흐름을 견준다.
    """
    seq_length = 100
    hidden_size = 128
    input_size = 64
    
    # 모델들
    models = {
        'RNN': nn.RNN(input_size, hidden_size, batch_first=True),
        'LSTM': nn.LSTM(input_size, hidden_size, batch_first=True),
        'GRU': nn.GRU(input_size, hidden_size, batch_first=True)
    }
    
    results = {}
    for name, model in models.items():
        results[name] = analyze_gradient_flow(model, seq_length, input_size)
    
    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    timesteps = np.arange(seq_length)
    
    # 선형 눈금
    ax = axes[0]
    for name, grads in results.items():
        ax.plot(timesteps, grads.numpy(), label=name, linewidth=2)
    ax.set_xlabel('Timestep (distance from output)')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Flow (Linear Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 로그 눈금
    ax = axes[1]
    for name, grads in results.items():
        ax.semilogy(timesteps, grads.numpy(), label=name, linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Gradient Norm (log)')
    ax.set_title('Gradient Flow (Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 감소율 분석
    ax = axes[2]
    for name, grads in results.items():
        decay_rates = grads[:-1] / (grads[1:] + 1e-10)
        ax.plot(timesteps[:-1], decay_rates.numpy(), label=name, linewidth=2)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='No decay')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Decay Rate (t/t+1)')
    ax.set_title('Local Gradient Decay Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 2)
    
    plt.tight_layout()
    plt.show()
    
    # 수치 요약
    print("\nGradient Flow Summary:")
    print("=" * 60)
    for name, grads in results.items():
        first_grad = grads[0].item()
        last_grad = grads[-1].item()
        decay_ratio = first_grad / (last_grad + 1e-10)
        half_life = np.argmax(grads.numpy() < grads[0].item() / 2)
        
        print(f"{name}:")
        print(f"  First timestep gradient: {first_grad:.6f}")
        print(f"  Last timestep gradient:  {last_grad:.6f}")
        print(f"  Total decay ratio:       {decay_ratio:.2e}")
        print(f"  Half-life (timesteps):   {half_life if half_life > 0 else '>100'}")
        print()
    
    return results
```

---

## 7. LSTM에서도 기울기가 사라질 수 있는 까닭

기울기의 고속도로가 있어도 LSTM이 기울기 소실에서 완전히 자유롭지는 않다. 언제 왜 그런지 알면 더 나은 학습 절차를 설계할 수 있다.

### 조건 1: 망각 문의 포화

망각 문이 (세게 잊으려고) 0에 가까운 값을 내는 법을 배우면 기울기의 고속도로가 막힌다.

$$f_t \approx 0 \implies \frac{\partial c_t}{\partial c_{t-1}} \approx 0$$

**해법**: 기본적으로 기억하도록 망각 문의 편향을 1~2로 초기화한다.

```python
def initialize_forget_gate_bias(lstm, bias_value=1.0):
    """
    기억하도록 이끌려고 망각 문의 편향을 초기화한다.
    
    PyTorch의 LSTM에서 편향의 순서는 [입력, 망각, 세포, 출력]이다.
    """
    for name, param in lstm.named_parameters():
        if 'bias_ih' in name or 'bias_hh' in name:
            n = param.size(0)
            # 망각 문은 두 번째 사분면에 있다
            start = n // 4
            end = n // 2
            param.data[start:end].fill_(bias_value)
    
    print(f"Initialized forget gate biases to {bias_value}")
```

### 조건 2: 아주 긴 순차열

$f_t = 0.99$이더라도 기울기는 지수적으로 줄어든다.

$$0.99^{100} \approx 0.366$$

$$0.99^{500} \approx 0.007$$

$$0.99^{1000} \approx 0.00004$$

**이것은 근본적인 한계이다.** 토큰이 1000개를 넘는 순차열에서는 LSTM도 애를 먹는다.

```python
def demonstrate_long_sequence_decay():
    """현실적인 망각 문 값에 대한 기울기의 감소를 보인다."""
    
    # 학습된 모델에서 나온 현실적인 망각 문 값
    realistic_f_values = [0.9, 0.95, 0.99]
    lengths = [50, 100, 200, 500, 1000, 2000]
    
    print("Gradient Magnitude After k Steps (assuming constant forget gate)")
    print("=" * 70)
    
    for f_val in realistic_f_values:
        print(f"\nForget gate f = {f_val}:")
        for k in lengths:
            gradient = f_val ** k
            status = "✓" if gradient > 0.01 else "⚠️" if gradient > 0.001 else "❌"
            print(f"  k={k:4d}: gradient = {gradient:.2e} {status}")
```

### 조건 3: 간접 경로가 주도할 때

어떤 과제에서는 중요한 기울기 신호가 곧바른 세포 상태 경로가 아니라 (문을 거치는) 간접 경로로 흐른다. 그 간접 경로가 사라지면 세포 상태의 기울기가 멀쩡해도 학습이 나빠진다.

### 조건 4: 출력 문이 막을 때

출력 문이 $o_t \approx 0$이면 손실에서 오는 기울기가 세포 상태에 닿기 어렵다.

$$\frac{\partial h_t}{\partial c_t} = o_t \odot \tanh'(c_t) \approx 0$$

---

## 8. 기울기 흐름 진단

### 종합 진단 도구

```python
class LSTMGradientDiagnostics:
    """
    LSTM 신경망의 기울기 흐름을 두루 진단한다.
    
    기울기의 건강을 살피고 있을 법한 문제를 짚어 낸다.
    - 기울기 소실 (세포의 기울기가 너무 작다)
    - 기울기 폭발 (세포의 기울기가 너무 크다)
    - 죽은 문 (망각 문이나 입력 문이 0이나 1에 붙박였다)
    - 실효 기억 길이
    """
    
    def __init__(self, model):
        self.model = model
        self.gradient_history = []
        self.gate_history = []
        self.hooks = []
    
    def register_hooks(self):
        """순전파와 역전파 훅을 등록한다."""
        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                hidden, (h_n, c_n) = output
                self.gate_history.append({
                    'hidden_mean': hidden.abs().mean().item(),
                    'hidden_std': hidden.std().item(),
                    'cell_mean': c_n.abs().mean().item(),
                    'cell_std': c_n.std().item()
                })
        
        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradient_history.append({
                    'output_grad_norm': grad_output[0].norm().item(),
                    'output_grad_mean': grad_output[0].abs().mean().item(),
                    'output_grad_max': grad_output[0].abs().max().item()
                })
        
        for module in self.model.modules():
            if isinstance(module, nn.LSTM):
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_full_backward_hook(backward_hook))
    
    def analyze(self, verbose=True):
        """모은 기울기 통계를 분석한다."""
        if not self.gradient_history:
            print("No gradients captured. Run training steps first.")
            return None
        
        stats = {
            'grad_norms': [g['output_grad_norm'] for g in self.gradient_history],
            'grad_means': [g['output_grad_mean'] for g in self.gradient_history],
            'grad_maxes': [g['output_grad_max'] for g in self.gradient_history]
        }
        
        analysis = {
            'mean_grad_norm': np.mean(stats['grad_norms']),
            'std_grad_norm': np.std(stats['grad_norms']),
            'max_grad_norm': np.max(stats['grad_norms']),
            'min_grad_norm': np.min(stats['grad_norms']),
            'vanishing_risk': np.mean(stats['grad_norms']) < 1e-6,
            'exploding_risk': np.max(stats['grad_norms']) > 1e3
        }
        
        if verbose:
            print("LSTM Gradient Diagnostics Report")
            print("=" * 50)
            print(f"Samples analyzed: {len(self.gradient_history)}")
            print(f"\nGradient Norm Statistics:")
            print(f"  Mean: {analysis['mean_grad_norm']:.6f}")
            print(f"  Std:  {analysis['std_grad_norm']:.6f}")
            print(f"  Min:  {analysis['min_grad_norm']:.6f}")
            print(f"  Max:  {analysis['max_grad_norm']:.6f}")
            print(f"\nRisk Assessment:")
            print(f"  Vanishing gradient risk: {'HIGH ⚠️' if analysis['vanishing_risk'] else 'Low ✓'}")
            print(f"  Exploding gradient risk: {'HIGH ⚠️' if analysis['exploding_risk'] else 'Low ✓'}")
        
        return analysis
    
    def cleanup(self):
        """훅을 없앤다."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
```

### 실효 기억 길이 재기

```python
def measure_effective_memory_length(model, input_size, max_length=500, 
                                     threshold=0.01, num_trials=20):
    """
    기울기가 실제로 얼마나 멀리 거슬러 전파되는지 잰다.
    
    실효 기억 길이는 기울기의 크기가 최댓값의 `threshold` 아래로
    떨어지는 지점이다.
    
    인수:
        model: LSTM 모델
        input_size: 입력 특징의 차원
        max_length: 시험할 순차열의 최대 길이
        threshold: 기울기의 문턱값 (최댓값에 대한 비율)
        num_trials: 평균을 낼 시행 횟수
    
    반환값:
        effective_length: 어림한 실효 기억
        gradient_profile: 기울기의 전체 프로파일
    """
    gradient_profile = torch.zeros(max_length)
    
    model.eval()
    
    for _ in range(num_trials):
        x = torch.randn(1, max_length, input_size, requires_grad=True)
        
        outputs, _ = model(x)
        
        # 마지막 시각에서 역전파
        loss = outputs[0, -1, :].sum()
        loss.backward()
        
        # 기울기 기록
        for t in range(max_length):
            gradient_profile[t] += x.grad[0, t, :].norm().item()
        
        # 기울기 지우기
        model.zero_grad()
    
    gradient_profile /= num_trials
    
    # 정규화
    max_grad = gradient_profile.max()
    normalized = gradient_profile / max_grad
    
    # 실효 길이 찾기
    effective_length = (normalized > threshold).sum().item()
    
    # 반감기도 계산
    half_life = (normalized > 0.5).sum().item()
    
    print(f"Effective Memory Analysis (threshold={threshold}):")
    print(f"  Effective memory length: {effective_length} timesteps")
    print(f"  Gradient half-life: {half_life} timesteps")
    print(f"  Max gradient at timestep: {gradient_profile.argmax().item()}")
    
    return effective_length, gradient_profile
```

---

## 9. 요즘 구조와 견주기

### 트랜스포머: 궁극의 고속도로인 자기 어텐션

트랜스포머는 먼 거리 기울기 문제를 더 곧바로 푼다.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

자리 $t$에서 자리 $t-k$까지의 기울기는 어텐션 가중치를 타고 흐른다. 거리에 상관없이 곱셈으로 줄어들지 않는 **곧바른 경로**이다.

| 항목 | LSTM | 트랜스포머 |
|--------|------|-------------|
| 기울기 경로 | 망각 문을 지남 | 어텐션을 지남 |
| 거리 의존 | 지수적 감소 | (이론적으로) 감소 없음 |
| 계산 | 순차적 | 병렬 |
| 메모리 | O(hidden_size) | O(sequence_length²) |

### 하이웨이 신경망과 건너뛰기 연결

LSTM의 세포 상태는 하이웨이 신경망과 ResNet에 영감을 주었다.

$$y = T(x) \cdot H(x) + (1 - T(x)) \cdot x$$

문을 묶은 LSTM과 견주어 보라.

$$c_t = f_t \odot \tilde{c}_t + (1 - f_t) \odot c_{t-1}$$

원리는 똑같다. **덧셈 갱신이 기울기를 지킨다.**

---

## 연습문제

**연습문제 1.**
LSTM 세포 상태를 지나는 기울기의 흐름을 유도하고 그것이 기울기 소실 문제를 피함을 보여라.

??? success "연습문제 1 풀이"
    세포 상태의 기울기는 $\frac{\partial c_T}{\partial c_t} = \prod_{k=t+1}^T f_k$이다. 망각 문이 1에 가까우면(기억하면) 이 곱이 1 가까이 머물러 기울기가 여러 시각에 걸쳐 막힘없이 흐른다. 지수적으로 줄어드는 기본 RNN의 $\prod_k \sigma_{\max}(W_h)\tanh'(z_k)$과 견주어 보라.

---

**연습문제 2.**
세포 상태와 잔차 연결 사이의 비유를 설명하라.

??? success "연습문제 2 풀이"
    세포 갱신 $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$은 ResNet의 $y = x + F(x)$과 비슷한 덧셈 갱신이다. 세포 상태가 기울기의 '고속도로' 노릇을 하고 망각 문이 기울기가 얼마나 지나갈지 다스린다. 두 장치 모두 덧셈 건너뛰기 연결로 기울기 소실을 푼다.

---

**연습문제 3.**
실제로 세포 상태는 대체로 어떤 정보를 담는 법을 배우는가?

??? success "연습문제 3 풀이"
    언어 모형에서는 문법적 상태(열린 괄호, 따옴표), 먼 거리의 일치(주어와 동사의 수), 감성의 극성 따위이다. 시계열에서는 추세의 방향과 계절의 위상이다. 망각 문은 맥락이 바뀔 때 쓸모없어진 정보를 골라 지우는 법을 배운다.

---

**연습문제 4.**
감성 분석 LSTM의 망각 문 활성값을 시간에 따라 그려 보고 그 무늬를 해석하라.

??? success "연습문제 4 풀이"
    망각 문이 1에 가까우면(흰색) 기억하는 것이고 0에 가까우면(검은색) 잊는 것이다. 흔한 무늬는 이렇다. 문장 경계 부근에서 문이 닫히고, 부정어가 앞선 감성을 잊게 만들며, 접속어('but', 'however')가 뒤 절을 위해 상태를 되돌리게 한다.

## 정리하며

LSTM에서 기울기가 흐를 수 있는 것은 **덧셈 세포 상태 갱신** 덕분이다.

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

핵심 착상은 다음과 같다.

1. **곧바른 기울기 경로**: $\frac{\partial c_t}{\partial c_{t-1}} = f_t$이 1 가까이 머무를 수 있다
2. **학습되는 문**: 신경망이 정보를 지킬 때와 갱신할 때를 배운다
3. **망각 문의 초기화**: 기울기가 흐르도록 편향을 1로 둔다
4. **한계는 남는다**: 아주 긴 순차열(1000 이상)은 여전히 LSTM에 벅차다

진단할 때의 관행은 다음과 같다.

- 학습 중에 기울기의 노름을 살핀다
- 실효 기억 길이를 잰다
- 포화된 문이 있는지 확인한다
- 안전망으로 기울기 자르기를 쓴다

이 분석은 LSTM이 거둔 역사적 성공을 설명하고, 트랜스포머(곧바른 어텐션), 하이웨이 신경망(건너뛰기 연결), 그리고 기울기 흐름 문제를 더 파고든 여러 구조적 개선이 나오게 된 까닭을 밝혀 준다.

---

**참고 문헌**

1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.

2. Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). Learning to Forget: Continual Prediction with LSTM. *Neural Computation*, 12(10), 2451-2471.

3. Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the Difficulty of Training Recurrent Neural Networks. *ICML*.

4. Greff, K., Srivastava, R. K., Koutník, J., Steunebrink, B. R., & Schmidhuber, J. (2017). LSTM: A Search Space Odyssey. *IEEE Transactions on Neural Networks and Learning Systems*, 28(10), 2222-2232.
