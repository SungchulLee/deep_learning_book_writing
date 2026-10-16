# 경사의 흐름

---

## 1. 학습 목표

!!! abstract "배울 내용"

    - 야코비안 행렬 곱의 관점에서 깊은 신경망을 지나는 경사의 크기 분석하기
    - 시그모이드가 경사를 지수적으로 줄이는 이유를 유도하고 경사 소실 문제를 수치로 재기
    - ReLU, 알맞은 초기화, 배치 정규화, 잔차 연결이 경사의 병리를 각각 어떻게 다루는지 설명하기
    - 분산 보존 논증에서 Xavier와 He 초기화 방식을 유도하기
    - 경사 감시 도구를 구현하고 경사 통계량으로 학습 문제 진단하기
    - 20층 이상에서도 안정적으로 학습되는 잔차 신경망 만들기

---

## 2. 미리 알아야 할 것

| 주제 | 왜 중요한가 |
|-------|---------------|
| 역전파 (9.2절) | 경사의 흐름은 여러 층을 지나는 역전파의 거동이다 |
| 연쇄 법칙과 야코비안 행렬 | 경사의 크기는 야코비안 행렬들의 곱에 달려 있다 |
| 활성화 함수 (9.1절) | 활성화마다 경사의 성질이 다르다 |

---

## 3. 개요

연쇄 법칙에 따르면 깊은 신경망의 경사는 층마다 하나씩인 **여러 인수의 곱**이다. 이 인수들이 계속 1보다 작으면 경사가 지수적으로 0에 가까워지고(**경사 소실**), 계속 1보다 크면 지수적으로 커진다(**경사 폭발**). 이 곱셈적인 움직임을 이해하고 다스리는 일이 깊은 신경망 학습에 필수적이다.

---

## 4. 야코비안 행렬 곱의 관점

### 활성화에 대한 경사

층의 사상을 $\mathbf{a}^{[k]}=\sigma^{[k]}(W^{[k]}\mathbf{a}^{[k-1]}+b^{[k]})$로 쓰자. 원소별 활성화이면 야코비안은 $J^{[k]}=D^{[k]}W^{[k]}$이며 $D^{[k]}=\operatorname{diag}((\sigma^{[k]})'(z^{[k]}))$이다. 여기서는 활성화 후 값에 대한 경사 $g^{[l]}=\nabla_{a^{[l]}}\mathcal{L}$을 사용한다.

$$
g^{[l]}=(J^{[l+1]})^\top\cdots(J^{[L]})^\top g^{[L]}
$$

활성화 전 값에 대한 오차 신호와 구별해야 한다. 그 신호는 $\delta^{[l]}=D^{[l]}(W^{[l+1]})^\top\delta^{[l+1]}$로 전파된다.

### 노름의 상한과 실제 경사

$$
\|g^{[l]}\|_2\leq\left(\prod_{k=l+1}^{L}\|D^{[k]}\|_2\|W^{[k]}\|_2\right)\|g^{[L]}\|_2
$$

이 식은 **상한**이다. 곱이 깊이에 따라 작아지면 경사도 작아지지만, 상한이 커진다고 실제 경사가 반드시 폭발하지는 않는다. 행렬이 각 방향을 얼마나 줄이거나 늘리는지와 연속된 층의 방향 정렬도 중요하다. 각 노름이 1에 가깝다는 사실만으로 모든 경사가 보존되는 것도 아니다.

## 5. 경사 소실 문제

### 시그모이드: 수치적 분석

시그모이드 활성화 $\sigma(z) = 1/(1 + e^{-z})$에 대해 다음이 성립한다.

$$
\sigma'(z) = \sigma(z)(1 - \sigma(z)) \leq \frac{1}{4}
$$

최대 도함수는 $z = 0$에서 얻는 $\frac{1}{4}$이다. 따라서 $\|\mathbf{D}^{[k]}\| \leq \frac{1}{4}$이고 다음이 성립한다.

$$
\|g^{[l]}\|_2 \leq \left(\frac{1}{4}\right)^{L-l} \prod_{k=l+1}^{L} \|W^{[k]}\|_2\,\|g^{[L]}\|_2
$$

$\|\mathbf{W}^{[k]}\| = 1$이라 해도 20층 신경망은 경사를 다음만큼 줄인다.

$$
\left(\frac{1}{4}\right)^{19} \approx 3.6 \times 10^{-12}
$$

이 크기는 매우 작아 일반적인 갱신에 영향을 거의 주지 못할 수 있다. 다만 float32가 이 수 자체를 표현하지 못해 0이 되는 것은 아니다. 실제 경사의 규모에는 출력 경사와 가중치도 함께 영향을 준다.

### Tanh: 조금 나은 편

$\tanh(z)$의 최대 도함수는 $\tanh'(0) = 1$이지만, 0에서 멀어진 입력에서는 $\tanh'(z) = 1 - \tanh^2(z) < 1$이다. 실무에서 tanh도 여전히 경사 소실을 겪지만 시그모이드보다는 덜하다.

### 경사 소실의 징후

- 앞쪽 층이 극도로 느리게 배운다 (가중치가 거의 바뀌지 않는다)
- 성능이 수렴하지 않았는데도 학습 손실이 정체한다
- 경사의 노름이 층 번호에 따라 기하급수적으로 줄어든다

---

## 6. 경사 폭발 문제

### 경사가 커질 때

가중치와 활성화의 야코비안 곱이 경사가 향하는 방향을 반복해서 늘리면 경사가 폭발할 수 있다. ReLU의 활성 구간에서도 가중치 행렬이 크기를 키울 수 있다. 단, $\|W^{[k]}\|_2>1$은 가능한 확장의 상한에 관한 조건이며 경사 폭발의 충분조건은 아니다.

### 징후

- 손실이 갑자기 `NaN`이나 `Inf`가 된다
- 가중치가 아주 큰 값으로 자란다
- 학습이 불안정해진다 (손실이 진동하거나 발산한다)

### 해결: 경사 클리핑

경사 클리핑는 매개변수를 갱신하기 전에 경사의 노름을 제약한다.

**노름으로 자르기** (너무 크면 배율을 줄인다):

$$
\mathbf{g} \leftarrow \begin{cases} \mathbf{g} & \text{if } \|\mathbf{g}\| \leq \tau \\ \tau \cdot \frac{\mathbf{g}}{\|\mathbf{g}\|} & \text{if } \|\mathbf{g}\| > \tau \end{cases}
$$

```python
# 전역 기울기 노름을 max_norm으로 자르기
import torch

torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**값으로 자르기** (각 성분을 범위 안에 가둔다):

```python
# 기울기의 각 원소를 [-0.5, 0.5]로 죄기
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

노름으로 자르는 편이 경사의 방향을 보존하므로 대체로 낫다.

---

## 7. 경사 소실의 해결책

### 1. ReLU 활성화

$$
\text{ReLU}(z) = \max(0, z), \qquad \text{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases}
$$

활성화된 ReLU의 국소 도함수는 1이다. 시그모이드처럼 이 단계에서 작은 도함수를 곱하는 효과를 피하지만, 가중치와 활성 패턴 때문에 전체 경사가 작아지거나 커질 가능성은 남는다.

**주의 — 죽은 뉴런:** 모든 학습 입력에서 출력이 계속 0인 ReLU는 해당 경로의 국소 경사를 전달하지 못한다. 입력 표현이나 다른 갱신이 변하면 상태가 바뀔 수도 있으므로 영원히 회복하지 못한다고 단정하지 않는다. Leaky ReLU는 음수 영역에도 기울기를 둔다.

### 2. 알맞은 가중치 초기화

이 분산 분석의 목표는 초기 활성화와 경사의 대표적인 크기를 유지하는 것이다. 각 행렬의 작용소 노름을 1로 맞추는 조건과는 다르다.

#### Xavier (Glorot) 초기화

**상황:** 입력이 $n_{\text{in}}$개, 출력이 $n_{\text{out}}$개이며 tanh나 시그모이드 활성화를 쓰는 층.

**유도:** 순전파에서 분산이 보존되도록 $\text{Var}(\mathbf{a}^{[l]}) = \text{Var}(\mathbf{a}^{[l-1]})$을 원한다.

$z_j^{[l]} = \sum_{i=1}^{n_\text{in}} W_{ji} a_i^{[l-1]}$에서 독립성과 평균 0을 가정하면 다음과 같다.

$$
\text{Var}(z_j^{[l]}) = n_{\text{in}} \cdot \text{Var}(W_{ji}) \cdot \text{Var}(a_i^{[l-1]})
$$

$\text{Var}(z_j^{[l]}) = \text{Var}(a_i^{[l-1]})$으로 두려면 다음이 필요하다.

$$
\text{Var}(W_{ji}) = \frac{1}{n_{\text{in}}}
$$

역전파에 대해서도 대칭적으로 논하면 $\text{Var}(W_{ji}) = 1/n_{\text{out}}$을 얻는다. Xavier는 팬인과 팬아웃의 평균을 분모에 두어 절충한다. 이는 두 역수의 산술평균과 다르다.

$$
\boxed{W_{ji} \sim \mathcal{U}\!\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}},\; \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)}
$$

같은 분산을 갖는 정규분포 버전은 $W_{ji} \sim \mathcal{N}\!\left(0,\; \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)$이다.

```python
nn.init.xavier_uniform_(layer.weight)   # 균등분포 판본
nn.init.xavier_normal_(layer.weight)    # 정규분포 판본
```

#### He (Kaiming) 초기화

**상황:** ReLU 활성화.

ReLU가 활성화의 대략 절반을 0으로 만들므로 실효 팬인이 절반이 된다. 그래서 다음을 얻는다.

$$
\text{Var}(W_{ji}) = \frac{2}{n_{\text{in}}}
$$

$$
\boxed{W_{ji} \sim \mathcal{N}\!\left(0,\; \frac{2}{n_{\text{in}}}\right)}
$$

```python
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
```

!!! tip "경험칙"
    Xavier는 tanh와 선형층의 출발점으로, He는 ReLU 계열의 출발점으로 쓸 수 있다. gain은 활성화에 맞게 선택한다. PyTorch `nn.Linear`의 기본 가중치는 대략 `[-1/sqrt(fan_in), 1/sqrt(fan_in)]`의 균등분포이며, ReLU용 He 분산 `2/fan_in`과 다르다.

### 3. 배치 정규화

배치 정규화(Ioffe & Szegedy, 2015)는 각 미니배치 안에서 활성화 전 값이 평균 0, 분산 1이 되도록 정규화한 뒤 학습된 아핀 변환을 적용한다.

$$
\hat{z}_i = \frac{z_i - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}, \qquad \tilde{z}_i = \gamma \hat{z}_i + \beta
$$

여기서 $\mu_B, \sigma_B^2$은 배치 통계량이고 $\gamma, \beta$은 학습 가능한 매개변수이다.

**경사의 흐름에 미치는 영향:** 배치 정규화는 층의 스케일과 최적화 조건을 바꿀 수 있다. 학습 가능한 배율과 이동 항 때문에 활성화가 포화하지 않는다고 보장하지는 않는다. 입력 분포, 배치 크기와 학습·평가 통계량 차이도 고려한다.

### 4. 잔차(건너뛰기) 연결

잔차 연결(He 등, 2016)은 경사를 위한 **직접적인 덧셈 경로**를 제공한다.

$$
\mathbf{a}^{[l]} = \underbrace{f(\mathbf{a}^{[l-1]})}_{\text{learned residual}} + \underbrace{\mathbf{a}^{[l-1]}}_{\text{identity shortcut}}
$$

이 연결을 지나는 경사는 다음과 같다.

$$
\frac{\partial \mathbf{a}^{[l]}}{\partial \mathbf{a}^{[l-1]}} = \underbrace{\frac{\partial f}{\partial \mathbf{a}^{[l-1]}}}_{\text{can vanish}} + \underbrace{\mathbf{I}}_{\text{always = 1}}
$$

항등 항은 잔차 분기를 통과하지 않는 경로를 추가한다. 잔차 야코비안이 작으면 전체 야코비안이 항등에 가까워질 수 있다. 하지만 $J_f=-I$이면 합이 0이 되는 것처럼 상쇄도 가능하므로, 경사 크기에 양의 하한을 보장하지 않는다.

잔차 블록이 $L$개인 신경망에서 임의의 층 $l$의 경사는 곱이 아니라 **경로들의 합**을 받는다.

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[l]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[L]}} \left(\mathbf{I} + \sum_{\text{paths}} \prod_{\text{residual Jacobians}}\right)
$$

위 식은 항등 경로와 잔차 경로들의 기여를 전개한 개략적인 표현이다. 경로의 합에는 상쇄가 일어날 수 있다. 활성화나 차원 변경이 추가된 블록은 그 연산의 야코비안도 포함해야 한다.

---

## 8. PyTorch 구현

### 경사 감시

```python
import torch
import torch.nn as nn

def monitor_gradients(model: nn.Module) -> dict[str, dict]:
    """역전파 뒤에 기울기 통계를 모은다."""
    # 반드시 loss.backward() 뒤, optimizer.step() 앞에서 불러야 한다.
    # step()이 끝나면 기울기가 다음 걸음을 위해 지워지기 때문이다.
    stats = {}
    for name, param in model.named_parameters():
        # grad가 None인 경우가 있다. 아직 역전파를 하지 않았거나,
        # requires_grad=False로 얼려 둔 층이거나, 손실까지 이어지지
        # 않은 층이다. 마지막 경우가 곧 배선이 끊긴 버그다
        if param.grad is not None:
            g = param.grad
            stats[name] = {
                # 노름이 가장 중요하다. 이 하나로 소실과 폭발을 다 잡아낸다
                'norm': g.norm().item(),
                'mean': g.mean().item(),
                'std':  g.std().item(),
                # 최댓값은 노름이 멀쩡해도 몇몇 성분만 튀는 경우를 잡는다
                'max':  g.abs().max().item(),
            }
    return stats

def print_gradient_report(stats: dict):
    """경고와 함께 기울기 건강 보고서를 출력한다."""
    print(f"{'Parameter':<35s} {'Norm':>10s} {'Max':>10s} {'Status':>8s}")
    print("-" * 68)
    for name, s in stats.items():
        # 문턱값은 눈대중이지만 실무에서 쓸 만하다.
        # 층 이름 차례로 보아 입력 쪽으로 갈수록 노름이 급히 줄면
        # 기울기 소실이고, 특정 층에서만 튀면 그 층이 말썽이다
        if s['norm'] < 1e-7:
            status = "⚠️ VANISH"    # 사실상 학습이 멎은 층
        elif s['norm'] > 1e3:
            status = "⚠️ EXPLOD"    # 기울기 절단이 필요한 신호
        else:
            status = "✓"
        # 지수 표기(2e)로 찍는 까닭은 층 사이에 값의 자릿수가
        # 몇 배씩 차이 나기 때문이다
        print(f"{name:<35s} {s['norm']:>10.2e} {s['max']:>10.2e} {status:>8s}")
```

### 활성화 함수 비교

```python
import matplotlib.pyplot as plt

def gradient_flow_experiment():
    """10층 신경망에서 활성화 함수에 따른 기울기 흐름을 비교한다."""
    activations = {
        'ReLU':    nn.ReLU,
        'Sigmoid': nn.Sigmoid,
        'Tanh':    nn.Tanh,
    }
    
    results = {}

    for act_name, act_cls in activations.items():
        # 10층 신경망 만들기.
        # 활성화 함수만 바꾸고 나머지는 똑같이 두어야 차이를 활성 탓으로
        # 돌릴 수 있다
        layers = []
        for _ in range(10):
            layers.extend([nn.Linear(64, 64), act_cls()])
        layers.append(nn.Linear(64, 1))
        model = nn.Sequential(*layers)

        # 순전파 + 역전파.
        # 주의: manual_seed가 모델을 만든 "뒤"에 있다. 그래서 입력 x, y는
        # 세 경우에 같지만 가중치 초기값은 서로 다르다. 엄밀히 견주려면
        # 이 줄이 nn.Sequential 앞에 와야 한다
        torch.manual_seed(42)
        x = torch.randn(32, 64)
        y = torch.randn(32, 1)
        # 목표가 무작위라 배울 것이 없다. 역전파를 한 번 돌려
        # 기울기의 크기만 보려는 것이다
        loss = nn.MSELoss()(model(x), y)
        loss.backward()

        # 층마다 가중치 기울기의 노름 모으기.
        # 시그모이드는 도함수의 최댓값이 0.25라, 층을 지날 때마다
        # 기울기가 최소 4분의 1로 줄어든다. 10층이면 4^10 배이니
        # 입력 쪽 층은 사실상 학습되지 않는다.
        # tanh는 최댓값이 1이라 덜하고, ReLU는 양수 쪽에서 정확히 1이라
        # 가장 잘 흐른다
        norms = []
        for layer in model:
            if isinstance(layer, nn.Linear):
                norms.append(layer.weight.grad.norm().item())
        results[act_name] = norms
    
    # 그래프 그리기
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = {'ReLU': 'o-', 'Sigmoid': 's-', 'Tanh': '^-'}
    for act_name, norms in results.items():
        ax.semilogy(range(1, len(norms) + 1), norms, markers[act_name],
                     label=act_name, lw=2, ms=8)
    
    ax.set_xlabel('Layer (1 = closest to output)', fontsize=12)
    ax.set_ylabel('Gradient Norm (log scale)', fontsize=12)
    ax.set_title('Gradient Flow: Effect of Activation Function (10-layer MLP)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('gradient_flow_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

gradient_flow_experiment()
```

### 초기화 비교

```python
def initialization_experiment():
    """초기화가 기울기 흐름에 미치는 영향을 보인다."""
    inits = {
        'Default (Kaiming Uniform)': lambda m: None,  # PyTorch 기본값
        'Xavier Normal':             lambda m: nn.init.xavier_normal_(m.weight),
        'He/Kaiming Normal':         lambda m: nn.init.kaiming_normal_(m.weight, nonlinearity='relu'),
        'Too Small (σ=0.01)':        lambda m: nn.init.normal_(m.weight, std=0.01),
        'Too Large (σ=1.0)':         lambda m: nn.init.normal_(m.weight, std=1.0),
    }
    
    results = {}
    for init_name, init_fn in inits.items():
        # 초기화 방식마다 씨앗을 같은 값으로 되돌린다. 그래야 차이가
        # 초기화 때문이지 뽑기 운 때문이 아니라고 말할 수 있다
        torch.manual_seed(42)
        layers = []
        for _ in range(10):
            lin = nn.Linear(64, 64)
            # 기본값 항목은 lambda m: None 이라 아무것도 하지 않는다.
            # 곧 PyTorch가 nn.Linear를 만들 때 넣어 준 값을 그대로 쓴다
            init_fn(lin)
            layers.extend([lin, nn.ReLU()])
        layers.append(nn.Linear(64, 1))
        model = nn.Sequential(*layers)

        x = torch.randn(32, 64)
        y = torch.randn(32, 1)
        # 목표가 무작위라 배울 것이 없다. 학습이 목적이 아니라
        # 역전파를 한 번 돌려 기울기의 크기만 보려는 것이다
        loss = nn.MSELoss()(model(x), y)
        loss.backward()

        # 앞의 실험이 활성의 기울기를 보았다면 여기서는 가중치의
        # 기울기를 본다. 실제로 갱신되는 것이 이쪽이다
        norms = []
        for layer in model:
            if isinstance(layer, nn.Linear):
                norms.append(layer.weight.grad.norm().item())
        results[init_name] = norms
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, norms in results.items():
        # semilogy: y축만 로그. 층마다 일정 배수로 줄거나 늘면
        # 로그 축에서 직선이 되고 그 기울기가 층당 변화율이 된다.
        # 선형 축이면 "너무 작음"과 "알맞음"이 둘 다 0에 붙어 보인다
        ax.semilogy(range(1, len(norms) + 1), norms, 'o-', label=name, lw=2, ms=6)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Gradient Norm (log scale)', fontsize=12)
    ax.set_title('Effect of Weight Initialization on Gradient Flow', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('initialization_gradient_flow.png', dpi=150, bbox_inches='tight')
    plt.show()

initialization_experiment()
```

### 잔차 MLP

```python
class ResidualBlock(nn.Module):
    """배치 정규화를 갖춘 2층 잔차 블록."""
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
    
    def forward(self, x):
        # 이 페이지의 답이 이 한 줄에 있다. x + f(x)를 x로 미분하면
        # 1 + f'(x)이므로, f 쪽 기울기가 아무리 작아져도 1이 남는다.
        # 층을 스무 번 지나도 기울기가 0으로 사그라들지 않는 까닭이다.
        # 블록 끝에 활성화를 두지 않고 더한 "뒤"에 거는 순서도 중요하다.
        # 그래야 항등 경로가 활성화를 거치지 않고 곧장 흐른다
        return torch.relu(x + self.block(x))   # 건너뛰기 연결

class ResidualMLP(nn.Module):
    """잔차 연결을 갖는 깊은 MLP."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks):
        super().__init__()
        # 잔차 블록은 입력과 출력의 크기가 같아야 더할 수 있다. 그래서
        # 먼저 784차원을 hidden_dim으로 옮겨 놓고 블록을 쌓는다
        self.proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.head = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        return self.head(self.blocks(self.proj(x)))

# ── 20층에서 기울기의 건강 상태 확인 ──
torch.manual_seed(42)
model = ResidualMLP(784, 128, 10, num_blocks=10)   # 실효 층 20개

# 배치 크기가 32인 것은 BatchNorm1d 때문이기도 하다. 배치가 1이면
# 배치 안의 분산이 정의되지 않아 학습 모드에서 오류가 난다.
# 모델을 eval()로 바꾸지 않았으므로 BN은 이 배치의 통계를 쓴다
x = torch.randn(32, 784)
y = torch.randint(0, 10, (32,))
loss = nn.CrossEntropyLoss()(model(x), y)
loss.backward()

print("Residual MLP Gradient Report (20 layers):")
stats = monitor_gradients(model)
print_gradient_report(stats)
```

---

## 9. 진단 요약

| 징후 | 짐작되는 원인 | 해결책 |
|---------|-------------|----------|
| 앞쪽 층의 경사가 $\approx 0$ | 경사 소실 | ReLU, He 초기화, 배치 정규화, 잔차 연결을 쓴다 |
| 손실이 `NaN`/`Inf`이 됨 | 경사 폭발 | 경사 클리핑, 학습률 낮추기, 더 나은 초기화 |
| 많은 뉴런이 늘 0을 냄 | 죽은 ReLU 뉴런 | Leaky ReLU를 쓰거나 학습률을 낮춘다 |
| 층마다 경사의 노름이 들쭉날쭉함 | 나쁜 초기화 | Xavier/He 초기화를 적용하고 배치 정규화를 더한다 |
| 처음엔 나아가다 학습이 멈춤 | 포화된 활성화 | 활성화의 분포를 확인하고 배치 정규화를 더한다 |

---

## 10. 핵심 정리

경사의 크기는 층별 야코비안의 곱과 방향 정렬에 달려 있다. ReLU, 초기화, 정규화 층과 잔차 연결은 안정적인 학습을 돕는 수단이며 보장 조건은 아니다. 클리핑은 계산된 유한 경사의 크기를 제한하지만 NaN의 원인을 해결하지는 않는다. 층별 경사와 활성화 통계량을 손실 변화와 함께 살핀다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
ReLU가 시그모이드의 경사 소실을 완화하는 이유와 남는 한계를 설명하라.

</div>

??? success "연습문제 1 풀이"
    시그모이드의 국소 도함수는 최대 $1/4$이지만 ReLU는 양수 입력에서 1이다. 따라서 활성 경로에서는 작은 활성화 도함수를 반복해서 곱하는 효과를 피한다. 전체 경사는 가중치 행렬도 통과하므로 ReLU를 쓴다고 경사 보존이 보장되지는 않는다. 음수 입력에서는 국소 도함수가 0이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
잔차 블록 $y=x+F(x)$의 경사를 구하라. 항등 경로가 경사의 크기를 1 이상으로 보장하는가?

</div>

??? success "연습문제 2 풀이"
    $\nabla_x L=(I+J_F)^\top\nabla_y L$이다. $J_F$가 작으면 항등에 가까운 전파를 얻을 수 있다. 그러나 $F(x)=-x$이면 $I+J_F=0$이므로 경사는 0이다. 항등 경로는 작은 경사를 완화하는 구조적 이점이며 크기의 보편적 하한은 아니다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
경사 폭발 문제란 무엇이며 경사 클리핑가 이를 어떻게 다루는가?

</div>

??? success "연습문제 3 풀이"
    (RNN 등에서) 경사의 크기가 층을 지나며 지수적으로 커지면 갱신이 너무 커져 학습이 발산한다. 경사 클리핑는 경사의 노름을 제한한다. $g \leftarrow g \cdot \min(1, \text{max\_norm}/\|g\|)$이다. 이는 방향을 보존하면서 크기만 묶어 둔다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
학습 중에 경사의 노름을 감시하도록 구현하고 에포크에 걸쳐 그려라.

</div>

??? success "연습문제 4 풀이"
    ```python
    for epoch in range(100):
        loss.backward()
        total_norm = sum(p.grad.norm()**2 for p in model.parameters())**0.5
        norms.append(total_norm.item())
        optimizer.step()
    ```

## 정리하며

경사 소실과 폭발은 야코비안 곱으로 이해할 수 있다. 노름의 상한을 실제 경사와 구별하고, 층별 통계량을 관찰해 문제를 진단한다. 초기화와 잔차 연결은 전파를 돕고 클리핑은 과도한 갱신을 제한하지만, 안정적인 학습 여부는 모델과 데이터 조건에서 확인해야 한다.

**참고 문헌**

- Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *AISTATS*.
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. *ICCV*.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.
- Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *ICML*.
- Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the difficulty of training recurrent neural networks. *ICML*.
