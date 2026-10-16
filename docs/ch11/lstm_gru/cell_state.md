# LSTM의 셀 상태와 그래디언트의 흐름

LSTM의 셀 상태에는 정보를 성분별로 보존하고 갱신하는 직접 경로가 있다. 이 경로는 장기 의존성 학습을 도울 수 있지만 전체 그래디언트의 보존을 보장하지 않는다. 이 절에서는 직접 경로, 은닉 상태를 거치는 간접 경로와 실제 진단 코드가 측정하는 양을 구분한다.

---

## 1. 근본 문제: 기본 RNN이 실패하는 까닭

열벡터 기준 tanh RNN의 시점별 야코비안은 $J_j=\operatorname{diag}(1-h_j^2)W_{hh}$다. 마지막 로스의 그래디언트는 다음과 같이 전달된다.

$$\nabla_{h_{t-k}}L_t=J_{t-k+1}^{\top}\cdots J_t^{\top}\nabla_{h_t}L_t$$

모든 $\|J_j\|_2$가 공통 상수 $q<1$ 이하이면 이 기여의 노름은 $q^k$ 배 이하로 줄어든다. 시점마다 활성화가 다르므로 야코비안 곱을 고정 행렬의 거듭제곱으로 항상 바꿀 수는 없다. 순환 웨이트의 최대 특잇값이 1보다 크다는 사실만으로 폭발이 보장되지도 않는다. 중간 시점의 로스가 있으면 그 기여도 더한다.

## 2. 그래디언트의 고속도로: 셀 상태

셀 상태 갱신은 다음과 같다.

$$c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t$$

계산 그래프에서 이전 은닉 상태 $h_{t-1}$를 고정한 직접 셀 경로의 미분은 $\operatorname{diag}(f_t)$다. 그러나 $c_{t-1}$가 $h_{t-1}$를 통해 게이트와 후보에 미치는 간접 경로도 있다. 그 경로를 포함해 한 단계의 셀 의존성을 표현하면 다음 형태가 된다.

$$D_t=\operatorname{diag}(f_t)+A_tB_{t-1}$$

$$A_t=\operatorname{diag}(c_{t-1})\frac{\partial f_t}{\partial h_{t-1}}+\operatorname{diag}(\tilde c_t)\frac{\partial i_t}{\partial h_{t-1}}+\operatorname{diag}(i_t)\frac{\partial\tilde c_t}{\partial h_{t-1}}$$

$$B_{t-1}=\operatorname{diag}(o_{t-1})\operatorname{diag}(1-\tanh^2(c_{t-1}))$$

이 식은 peephole 없는 표준 LSTM에서 이전 시점의 다른 직접 입력을 고정한 경로별 설명이다. 전체 시간 미분은 은닉·셀 상태를 합친 상태의 블록 야코비안으로 계산하는 것이 명확하다. 망각 게이트가 1에 가깝다는 조건만으로 간접 항도 작다고 볼 수 없다.

일정 오차 순환(CEC)은 원래 LSTM의 오차 전달 설계에서 나온 용어다. 망각 게이트가 있는 현대적 셀의 직접 경로에는 게이트의 곱이 생긴다. 이 곱이 1에 가까울 수 있다는 것과 전체 오차가 항상 일정하다는 것은 다르다.

## 3. 기본 RNN과의 대비

기본 RNN의 시간 경로에는 순환 선형 변환과 활성화 도함수가 반복된다. LSTM은 별도로 직접 셀 보존 경로를 둔다. 이 경로의 성분별 전달량은 망각 게이트 값의 곱이다.

$$\prod_{j=t-k+1}^{t}f_{j,a}$$

이는 해당 성분의 직접 경로에 대한 값이며, 전체 야코비안 노름을 단순히 산술 평균 망각 값의 k제곱으로 계산할 수는 없다. 게이트 값도 학습된 웨이트와 현재 입력·상태에 따라 달라지며 RNN의 야코비안 역시 고정된 상수가 아니다.

## 4. 온전한 그래디언트 경로 분석

### 전체 상태를 통한 역전파

$s_t=(h_t,c_t)$로 묶고 $K_t=\partial s_t/\partial s_{t-1}$라 하자. 현재 시점의 국소 로스 그래디언트를 $q_t$라 쓰면 전체 상태 그래디언트는 다음과 같다.

$$g_t=q_t+K_{t+1}^{\top}g_{t+1}$$

이 재귀식은 셀과 은닉 경로를 함께 포함한다. 각 시점의 전체 그래디언트를 다시 모든 로스에 대해 합하면 경로를 중복 계산할 수 있으므로 국소 기여와 누적 기여를 구분해야 한다.

### 그래디언트 경로 그려 보기

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

### 간접 경로의 역할

간접 경로는 게이트와 후보를 학습시키는 신호를 전달한다. 그 크기가 항상 작지는 않으며 직접 경로와 더해지거나 상쇄될 수 있다. 직접 보존 경로만 살펴서 전체 학습 상태를 판정하지 않는다.

---

## 5. 그래디언트의 흐름에서 각 문의 구실

### 망각 게이트: 고속도로의 관리자

망각 게이트 $f_t$이 그래디언트 흐름을 **가장 크게 좌우한다**.

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def analyze_forget_gate_impact():
    """
    망각 문의 값이 그래디언트 전파에 어떤 영향을 주는지 보인다.
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
    망각 문의 값에 따른 그래디언트의 감소를 그려 본다.
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

### 입력 게이트: 새 정보 조절하기

입력 게이트 $i_t$은 그래디언트에 간접적으로 영향을 준다.

$$\frac{\partial c_t}{\partial \tilde{c}_t} = \operatorname{diag}(i_t)$$

$i_t \approx 0$이면 뉴럴 넷트웍이 새 정보를 무시하고 그래디언트가 후보 계산으로 흐르지 않는다. 뉴럴 넷트웍이 늘 입력을 무시하는 법을 배우면 문제가 될 수 있다.

### 출력 게이트: 보임을 다스리는 문

출력 게이트 $o_t$은 $h_t$에서 $c_t$으로 그래디언트가 흐르는 방식에 영향을 준다.

$$\frac{\partial h_t}{\partial c_t} = \operatorname{diag}(o_t)\operatorname{diag}(1-\tanh^2(c_t))$$

$o_t \approx 0$이면 셀 상태가 출력에서 "가려져" 그 시각에 로스에서 오는 그래디언트가 셀 상태에 미치는 영향이 줄어든다.

---

## 6. 실험적 그래디언트 분석

아래 함수는 무작위 입력에서 마지막 은닉 출력 합의 입력 위치별 그래디언트를 평균한다. 학습한 모델의 기억 과제를 평가하는 코드는 아니다. 가로축은 입력 위치이며 출력에서의 거리는 반대 방향이다. 반감기 계산은 첫 위치의 작은 값을 기준으로 앞으로 검색하므로 “출력에서 거슬러 간 반감기”가 아니다. 모델별 입력과 초기값도 맞추지 않았고 파라메터 그래디언트를 시행마다 지우지 않으므로, 결과의 비교 범위를 제한해서 해석해야 한다.

```python
def analyze_gradient_flow(model, seq_length=100, input_size=64, 
                          num_samples=50, model_type='lstm'):
    """
    여러 RNN 판본의 그래디언트 흐름을 두루 분석한다.
    
    시각마다 그래디언트의 노름을 돌려주어, 마지막 출력에서 시간을 거슬러
    그래디언트가 얼마나 잘 전파되는지 잰다.
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
        
        # 시각마다 그래디언트 재기
        for t in range(seq_length):
            gradient_norms[t] += x.grad[0, t, :].norm().item()
        
        # 다음 표본을 위해 초기화
        x.grad.zero_()
    
    gradient_norms /= num_samples
    return gradient_norms

def comprehensive_gradient_comparison():
    """
    RNN과 LSTM과 GRU의 그래디언트 흐름을 견준다.
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

## 7. LSTM에서도 그래디언트가 사라질 수 있는 까닭

그래디언트의 고속도로가 있어도 LSTM이 그래디언트 소실에서 완전히 자유롭지는 않다. 언제 왜 그런지 알면 더 나은 학습 절차를 설계할 수 있다.

### 조건 1: 망각 게이트의 포화

망각 게이트이 (세게 잊으려고) 0에 가까운 값을 내는 법을 배우면 그래디언트의 고속도로가 막힌다.

$$f_t \approx 0 \implies \frac{\partial c_t}{\partial c_{t-1}} \approx 0$$

**초기화의 범위:** 양수 망각 바이어스는 초기 보존을 도울 수 있지만 이후 포화를 없애지는 않는다. 아래 함수는 입력·순환 바이어스 모두를 같은 값으로 채우므로 유효 바이어스는 인수의 두 배가 된다.

```python
def initialize_forget_gate_bias(lstm, bias_value=1.0):
    """
    기억하도록 이끌려고 망각 문의 바이어스를 초기화한다.
    
    PyTorch의 LSTM에서 바이어스의 순서는 [입력, 망각, 세포, 출력]이다.
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

### 조건 2: 아주 긴 시퀀스

$f_t = 0.99$이더라도 그래디언트는 지수적으로 줄어든다.

$$0.99^{100} \approx 0.366$$

$$0.99^{500} \approx 0.007$$

$$0.99^{1000} \approx 0.00004$$

이 수치는 일정한 망각 값의 직접 경로를 가정한 계산이다. 게이트가 변하거나 다른 경로가 기여하는 전체 LSTM의 정확한 그래디언트는 아니다. 모든 LSTM에 공통인 1000토큰 한계를 뜻하지 않는다.

```python
def demonstrate_long_sequence_decay():
    """현실적인 망각 문 값에 대한 그래디언트의 감소를 보인다."""
    
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

어떤 과제에서는 중요한 그래디언트 신호가 곧바른 셀 상태 경로가 아니라 (문을 거치는) 간접 경로로 흐른다. 그 간접 경로가 사라지면 셀 상태의 그래디언트가 멀쩡해도 학습이 나빠진다.

### 조건 4: 출력 게이트이 막을 때

출력 게이트이 $o_t \approx 0$이면 로스에서 오는 그래디언트가 셀 상태에 닿기 어렵다.

$$\frac{\partial h_t}{\partial c_t} = o_t \odot \tanh'(c_t) \approx 0$$

---

## 8. 그래디언트 흐름 진단

### 종합 진단 도구

아래 도구의 `gate_history`는 실제 게이트 값을 기록하지 않고 은닉 출력과 최종 셀 상태의 통계만 저장한다. 역전파 훅은 출력 그래디언트를 기록하므로 내부 셀 상태의 그래디언트나 시간별 전달량과 다르다. 고정 임계값에 따른 위험 표시는 로스 배율·모델 크기에도 영향을 받으며 확정 진단이 아니다. 실제 게이트 기록에는 셀 계산을 드러내는 별도 계측이 필요하다.

```python
class LSTMGradientDiagnostics:
    """
    LSTM 뉴럴 넷트웍의 그래디언트 흐름을 두루 진단한다.
    
    그래디언트의 건강을 살피고 있을 법한 문제를 짚어 낸다.
    - 그래디언트 소실 (세포의 그래디언트가 너무 작다)
    - 그래디언트 폭발 (세포의 그래디언트가 너무 크다)
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
        """모은 그래디언트 통계를 분석한다."""
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

아래 함수가 반환하는 길이는 정규화한 입력 그래디언트가 임계값보다 큰 위치의 개수다. 출력에서 연속해서 도달하는 최대 거리나 실제 기억 과제의 성능은 아니다. “반감기”도 절반 이상인 위치 개수이며, 그래디언트가 모두 0이면 정규화가 유효하지 않다. CPU·실수 입력을 가정하고 호출 뒤 평가 모드를 유지하는 점도 고려한다.

```python
def measure_effective_memory_length(model, input_size, max_length=500, 
                                     threshold=0.01, num_trials=20):
    """
    그래디언트가 실제로 얼마나 멀리 거슬러 전파되는지 잰다.
    
    실효 기억 길이는 그래디언트의 크기가 최댓값의 `threshold` 아래로
    떨어지는 지점이다.
    
    인수:
        model: LSTM 모델
        input_size: 입력 피처의 차원
        max_length: 시험할 순차열의 최대 길이
        threshold: 그래디언트의 임계값 (최댓값에 대한 비율)
        num_trials: 평균을 낼 시행 횟수
    
    반환값:
        effective_length: 어림한 실효 기억
        gradient_profile: 그래디언트의 전체 프로파일
    """
    gradient_profile = torch.zeros(max_length)
    
    model.eval()
    
    for _ in range(num_trials):
        x = torch.randn(1, max_length, input_size, requires_grad=True)
        
        outputs, _ = model(x)
        
        # 마지막 시각에서 역전파
        loss = outputs[0, -1, :].sum()
        loss.backward()
        
        # 그래디언트 기록
        for t in range(max_length):
            gradient_profile[t] += x.grad[0, t, :].norm().item()
        
        # 그래디언트 지우기
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

## 9. 요즘 구조와 비교

어텐션은 허용된 먼 위치 사이에 직접 정보 전달 경로를 만들 수 있다. 그러나 웨이트와 투영, 소프트맥스가 그래디언트에 영향을 주므로 거리와 무관하게 그래디언트가 항상 보존된다고 할 수는 없다. 학습 때의 병렬 처리와 자기회귀 생성도 구분한다.

LSTM의 스트림 추론 상태는 길이와 무관한 고정 크기로 유지할 수 있지만 전체 BPTT의 저장량은 길이에 비례한다. 트랜스포머의 메모리도 학습·추론, KV 캐시와 어텐션 구현에 따라 달라진다. 서로 다른 실행 조건을 한 표에서 단순 비교하면 안 된다.

하이웨이·잔차 구조와 셀 갱신은 덧셈 경로라는 공통점이 있다. 그렇더라도 게이트나 잔차 분기의 전체 미분을 포함해야 하며 덧셈 자체가 모든 그래디언트를 보존하는 것은 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
LSTM의 직접 셀 경로가 장기 그래디언트 전달을 도울 수 있는 이유와 한계를 설명하라.

</div>

??? success "연습문제 1 풀이"
    게이트와 후보를 고정한 직접 경로는 망각 게이트의 대각 행렬이며 긴 경로에는 그 곱이 생긴다. 값이 1에 가까우면 전달을 유지하기 쉽지만 1 미만의 값이 반복되면 여전히 감소한다. 전체 미분은 은닉 상태를 거치는 간접 경로도 포함하므로 이 곱과 같다고 단정할 수 없다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
LSTM 셀 상태와 잔차 연결의 공통점과 차이를 설명하라.

</div>

??? success "연습문제 2 풀이"
    둘 다 덧셈을 통해 직접 경로를 만든다. 잔차 항등 경로의 계수는 1인 반면 LSTM의 직접 셀 경로는 망각 게이트로 조절된다. 전체 그래디언트는 다른 경로의 미분도 포함하므로 직접 경로가 있다고 해서 소실·폭발이 모두 사라지는 것은 아니다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
셀 상태가 어떤 정보를 담았는지 알아보려면 어떤 근거가 필요한가?

</div>

??? success "연습문제 3 풀이"
    게이트나 상태의 패턴만 보고 문법·감성·추세를 저장했다고 확정할 수 없다. 통제된 입력, 탐침 과제와 상태 성분의 변화에 따른 예측 변화를 함께 살펴본다. 여러 성분에 정보가 분산될 수 있고 상관관계가 곧 인과적 역할은 아니라는 점도 고려한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
감성 분석 LSTM의 게이트 히트맵을 해석할 때 확인할 조건을 설명하라.

</div>

??? success "연습문제 4 풀이"
    실제 게이트를 기록했는지, 색 범위와 단어·시점 대응이 맞는지 확인한다. 큰 망각 값은 직접 보존 항의 계수가 크다는 뜻이지 전체 정보나 그래디언트가 그대로 유지된다는 뜻은 아니다. 특정 접속어나 부정어에서 반드시 같은 패턴이 나타난다고 가정하지 않는다.

## 정리하며

셀 상태의 직접 경로는 장기 정보 전달을 돕는 구조다. 전체 상태의 블록 야코비안과 직접 경로를 구분하고, 진단 함수가 기록하는 출력 그래디언트·게이트·입력 민감도를 혼동하지 않아야 한다. 고정 기억 길이와 보존 보장은 피한다.

**참고 문헌**

1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.

2. Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). Learning to Forget: Continual Prediction with LSTM. *Neural Computation*, 12(10), 2451-2471.

3. Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the Difficulty of Training Recurrent Neural Networks. *ICML*.

4. Greff, K., Srivastava, R. K., Koutník, J., Steunebrink, B. R., & Schmidhuber, J. (2017). LSTM: A Search Space Odyssey. *IEEE Transactions on Neural Networks and Learning Systems*, 28(10), 2222-2232.
