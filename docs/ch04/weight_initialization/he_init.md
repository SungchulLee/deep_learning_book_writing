# He (Kaiming) 초기화
He 초기화(He 등, 2015)는 ReLU와 그 변형들이 입력의 일부를 0으로 만들어 다음 층으로 넘어가는 실효 분산을 줄인다는 점을 반영하여 가중치의 분산을 정한다. (표준 ReLU에서) 인수 2로 이를 보정함으로써, He 초기화는 ReLU 계열 비선형성을 쓰는 깊은 신경망에서 활성화와 경사의 크기를 안정적으로 유지한다. 현대의 합성곱 구조와 순방향 구조의 표준 초기화이다.

---

## 1. 유도

### ReLU의 분산 인수

활성화 전 값이 $z \sim \mathcal{N}(0, \sigma_z^2)$일 때 ReLU의 출력은 다음과 같다.

$$h = \text{ReLU}(z) = \max(0, z)$$

ReLU가 대칭 분포의 음수 쪽 절반을 0으로 만들므로 다음이 성립한다.

$$\mathbb{E}[h] = \frac{1}{2}\,\mathbb{E}[|z|] = \frac{\sigma_z}{\sqrt{2\pi}}$$

$$\mathbb{E}[h^2] = \frac{1}{2}\,\mathbb{E}[z^2] = \frac{\sigma_z^2}{2}$$

따라서 다음이 성립한다.

$$\text{Var}(h) = \mathbb{E}[h^2] - (\mathbb{E}[h])^2 = \frac{\sigma_z^2}{2} - \frac{\sigma_z^2}{2\pi} = \frac{\sigma_z^2}{2}\left(1 - \frac{1}{\pi}\right)$$

지배적인 항은 $\sigma_z^2/2$이다. He 등은 다음 근사를 쓴다.

$$\text{Var}(\text{ReLU}(z)) \approx \frac{1}{2}\,\text{Var}(z)$$

이는 음수 값이 0이 되는 것을 반영하면서 더 작은 $1/\pi$ 보정을 흡수한다. 이 근사는 2차 적률에 대해서는 정확하기도 하다. $\mathbb{E}[\text{ReLU}(z)^2] = \frac{1}{2}\text{Var}(z)$이다.

### 순전파 조건

앞 절의 활성화 전 분산과 합치면 다음과 같다.

$$\text{Var}(z^{(l)}) = n_{\text{in}}\,\sigma^2\,\mathbb{E}\bigl[(h^{(l-1)})^2\bigr]$$

ReLU에서 $\mathbb{E}[h^2] = \frac{1}{2}\text{Var}(z^{(l-1)})$이므로 다음을 얻는다.

$$\text{Var}(z^{(l)}) = n_{\text{in}}\,\sigma^2 \cdot \frac{1}{2}\,\text{Var}(z^{(l-1)})$$

층에서 층으로 안정적이려면 $\text{Var}(z^{(l)}) = \text{Var}(z^{(l-1)})$이어야 하므로 다음이 성립한다.

$$n_{\text{in}}\,\sigma^2 \cdot \frac{1}{2} = 1$$

$$\boxed{\sigma^2 = \frac{2}{n_{\text{in}}}}$$

이것이 **He 초기화**의 분산 공식이다(팬인 방식).

### 역전파 조건

경사 신호에 같은 분석을 적용하면 다음을 얻는다.

$$\sigma^2 = \frac{2}{n_{\text{out}}}$$

역전파 경사의 안정성이 더 중요할 때(예: 건너뛰기 연결이 없는 아주 깊은 신경망) 팬아웃 방식을 쓰기도 한다.

### Xavier와의 비교

| | Xavier | He |
|---|--------|-----|
| 순전파 조건 | $n_{\text{in}}\,\sigma^2 = 1$ | $n_{\text{in}}\,\sigma^2 = 2$ |
| 활성화에 대한 가정 | $\text{Var}(f(z)) \approx \text{Var}(z)$ | $\text{Var}(\text{ReLU}(z)) = \frac{1}{2}\text{Var}(z)$ |
| 분산 | $\frac{2}{n_{\text{in}} + n_{\text{out}}}$ | $\frac{2}{n_{\text{in}}}$ |
| 효과 | tanh/시그모이드에 알맞다 | ReLU에 알맞다 |

He 초기화는 Xavier의 팬인 분산의 정확히 2배이며, ReLU가 들여오는 \$1/2$ 인수를 보정한다.

---

## 2. 분포의 변형

### He 정규 (Kaiming 정규)

$$W \sim \mathcal{N}\!\left(0,\;\frac{2}{n_{\text{in}}}\right)$$

```python
import torch.nn as nn
import torch.nn.init as init

linear = nn.Linear(256, 128)
init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='relu')
```

### He 균등 (Kaiming 균등)

$a^2/3 = 2/n_{\text{in}}$으로 두면 다음을 얻는다.

$$W \sim \mathcal{U}\!\left(-\sqrt{\frac{6}{n_{\text{in}}}},\;\sqrt{\frac{6}{n_{\text{in}}}}\right)$$

```python
init.kaiming_uniform_(linear.weight, mode='fan_in', nonlinearity='relu')
```

### 팬인 방식과 팬아웃 방식

```python
# 팬인: 순전파 활성화의 크기를 보존한다 (기본값)
init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='relu')

# 팬아웃: 역전파 기울기의 크기를 보존한다
init.kaiming_normal_(linear.weight, mode='fan_out', nonlinearity='relu')
```

팬인이 기본값이며 대부분의 구조에 알맞다. 신경망이 넓다가 좁아지는 모양일 때(예: 압축 병목) 팬아웃이 나을 수 있다.

---

## 3. Leaky ReLU와 PReLU로의 확장

음의 기울기가 $a$인 Leaky ReLU에 대해 다음과 같다.

$$\text{LeakyReLU}(z) = \begin{cases} z & z \geq 0 \\ az & z < 0 \end{cases}$$

2차 적률은 다음이 된다.

$$\mathbb{E}[\text{LeakyReLU}(z)^2] = \frac{1}{2}(1 + a^2)\,\text{Var}(z)$$

안정성 조건을 세우면 다음과 같다.

$$n_{\text{in}}\,\sigma^2 \cdot \frac{1 + a^2}{2} = 1$$

$$\boxed{\sigma^2 = \frac{2}{(1 + a^2)\,n_{\text{in}}}}$$

표준 ReLU($a = 0$)에서는 $2/n_{\text{in}}$을 되찾는다. $a = 0.01$인 Leaky ReLU에서는 보정이 무시할 만하다($1 + 0.0001 \approx 1$). PReLU에서는 $a$을 학습하지만 0.25로 초기화하므로 $\sigma^2 \approx 2/(1.0625 \cdot n_{\text{in}})$이 된다.

```python
# 기울기가 0.2인 Leaky ReLU
init.kaiming_normal_(linear.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
```

---

## 4. 실험적 확인

```python
import torch
import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt

def compare_init_relu(n_layers=50, hidden=512, n_samples=1024):
    """여러 층에 걸쳐 ReLU와 함께 Xavier 초기화와 He 초기화를 비교한다."""
    # 학습은 하지 않는다. 무작위로 초기화한 층 50개에 신호를 한 번
    # 통과시켜, 층을 지날수록 활성의 표준편차가 어떻게 되는지만 본다.
    # 이 값이 0으로 가면 기울기도 함께 사라져 학습 자체가 시작되지 못한다.
    x = torch.randn(n_samples, hidden)

    results = {}
    for name, init_fn in [
        # Xavier는 분산을 2/(fan_in+fan_out)으로 잡는다. tanh처럼 원점에서
        # 기울기가 1인 활성을 가정한 값이라, ReLU와 함께 쓰면 층마다
        # 절반씩 죽는 몫을 메우지 못해 신호가 사그라든다
        ('Xavier Normal', init.xavier_normal_),
        # He는 2/fan_in 으로 잡아 ReLU가 죽이는 절반을 미리 메운다.
        # fan_in 방식은 순전파의 분산을 지키는 쪽이다
        ('He Normal (fan_in)', lambda w: init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu')),
        # fan_out 방식은 역전파의 분산을 지키는 쪽이다. 정사각 층에서는
        # fan_in과 fan_out이 같아 둘이 겹치지만, 너비가 바뀌는 망에서는 갈린다
        ('He Normal (fan_out)', lambda w: init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')),
    ]:
        h = x.clone()   # 세 방식 모두 같은 입력에서 시작해야 견줄 수 있다
        stds = []
        for _ in range(n_layers):
            # bias=False 로 두는 까닭은 편향이 분산에 끼어들어
            # 초기화 방식의 차이를 흐리기 때문이다
            linear = nn.Linear(hidden, hidden, bias=False)
            init_fn(linear.weight)
            h = torch.relu(linear(h))
            stds.append(h.std().item())
        results[name] = stds

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, stds in results.items():
        ax.plot(stds, linewidth=2, label=name)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Activation Std')
    ax.set_title('Activation Standard Deviation Through 50 ReLU Layers')
    ax.legend()
    ax.set_ylim(0, 3)
    # 목표선 1.0. He 초기화의 두 곡선은 이 선 언저리에 머무르고
    # Xavier 곡선은 층이 깊어질수록 0으로 내려간다.
    # 주의: 이 axhline은 legend() 뒤에 있어 범례에 나오지 않는다
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4, label='Target')
    plt.tight_layout()
    plt.savefig('he_vs_xavier_relu.png', dpi=150, bbox_inches='tight')
    plt.show()

compare_init_relu()
```

### 경사 크기 비교

```python
def compare_gradient_flow(n_layers=30, hidden=256, n_samples=256):
    """각 층에서 기울기의 크기를 비교한다."""
    results = {}

    for name, init_fn in [
        ('Xavier', init.xavier_normal_),
        ('He', lambda w: init.kaiming_normal_(w, nonlinearity='relu')),
    ]:
        # 신경망 만들기
        layers = []
        for _ in range(n_layers):
            linear = nn.Linear(hidden, hidden, bias=False)
            init_fn(linear.weight)
            layers.append(linear)

        # ReLU를 쓰는 순전파
        x = torch.randn(n_samples, hidden, requires_grad=True)
        h = x
        activations = []
        for linear in layers:
            h = torch.relu(linear(h))
            h.retain_grad()
            activations.append(h)

        # 역전파
        loss = h.sum()
        loss.backward()

        grad_norms = [a.grad.norm().item() for a in activations]
        results[name] = grad_norms

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, norms in results.items():
        ax.plot(range(n_layers, 0, -1), norms, linewidth=2, label=name)
    ax.set_xlabel('Distance from Output (layers)')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Magnitude vs Distance from Output')
    ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    plt.savefig('gradient_flow_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

compare_gradient_flow()
```

---

## 5. 특수한 경우와 확장

### 잔차 신경망

ResNet에서 각 블록의 출력은 $h + F(h)$이며 여기서 $F$은 잔차 함수이다. $F$을 표준 He 가중치로 초기화하면 잔차 연결마다 분산이 두 배가 된다. 흔히 쓰는 완화책이 둘 있다.

**마지막 층의 0 초기화.** 마지막 배치 정규화의 $\gamma = 0$으로 두거나 마지막 합성곱의 가중치를 0으로 두어 처음에 $F(h) = 0$이 되게 한다.

```python
# 마지막 배치 정규화를 0으로 초기화한 ResNet 블록
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        # 합성곱 층에 He 초기화
        init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')

        # 잔차 블록이 항등함수로 시작하도록 마지막 배치 정규화를 0으로 초기화
        init.zeros_(self.bn2.weight)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(x + out)
```

**깊이에 따른 분산 배율 조정.** GPT-2를 비롯한 트랜스포머 모델들은 층의 수를 $L$이라 할 때 잔차 경로의 배율을 $1/\sqrt{2L}$으로 조정한다.

```python
# 트랜스포머식 잔차 배율 조정
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_layers):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        # 출력 사영을 1/√(2L)으로 배율 조정
        init.normal_(self.ffn[2].weight, std=0.02 / (2 * n_layers) ** 0.5)

    def forward(self, x):
        return x + self.ffn(x)
```

### 직교 초기화

He의 대안으로, $W$을 무작위 직교행렬(또는 그 부분 블록)으로 초기화할 수 있다. 직교행렬은 노름을 정확히 보존한다. $\|Wx\|_2 = \|x\|_2$이다. 이는 활성화 함수와 무관하게 순전파 신호를 완벽히 보존해 주지만, ReLU가 첫 층 이후로 직교성을 깨뜨린다.

```python
linear = nn.Linear(256, 256, bias=False)
init.orthogonal_(linear.weight, gain=2 ** 0.5)  # ReLU에는 gain=√2
```

`gain` 매개변수는 He 초기화의 인수 2와 마찬가지로 활성화 함수의 효과를 반영한다.

### LSUV (층 순차 단위 분산)

실제 데이터의 배치에서 잰 출력 분산이 1이 될 때까지 각 층의 가중치를 되풀이해 조정하는, 데이터 기반 초기화이다. 분포에 대한 가정에 기대지 않아도 된다.

```python
def lsuv_init(model, data_batch, target_std=1.0, max_iter=10, tol=0.05):
    """층 순차 단위 분산(LSUV) 초기화."""
    # He 초기화는 "이런 분포에서 뽑으면 분산이 유지될 것"이라는 이론값이다.
    # LSUV는 여기서 한 걸음 더 나아가, 실제 데이터를 흘려 보고 활성의
    # 표준편차를 재어 1이 될 때까지 가중치를 직접 다시 재는 방식이다.
    # 이론이 가정한 조건이 어긋나는 구조에서도 통한다는 것이 장점이다.
    model.eval()   # 드롭아웃과 배치 정규화를 꺼야 잰 값이 흔들리지 않는다
    hooks = []
    activation_stds = {}

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 출발점으로 He 초기화
            init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                init.zeros_(module.bias)

            # 출력 표준편차를 잡아내려고 훅 등록.
            # name=name 으로 기본 인자를 묶는 것이 중요하다. 이렇게 하지
            # 않으면 모든 훅이 반복문의 마지막 name을 함께 보게 되어
            # 통계가 한 자리에만 쌓인다(파이썬 클로저의 늦은 묶기)
            def hook_fn(mod, inp, out, name=name):
                activation_stds[name] = out.detach().std().item()
            hooks.append(module.register_forward_hook(hook_fn))

    # 배율을 반복적으로 조정.
    # 한 번에 끝나지 않는 까닭은 앞 층의 가중치를 고치면 뒤 층이 받는
    # 입력이 달라져 다시 재야 하기 때문이다. 그래서 앞에서 뒤로 훑는
    # 일을 여러 번 되풀이한다
    for iteration in range(max_iter):
        with torch.no_grad():   # 초기화 단계이므로 기울기가 필요 없다
            model(data_batch)   # 훅이 이때 활성 표준편차를 채운다

        all_close = True
        for name, module in model.named_modules():
            if name in activation_stds:
                std = activation_stds[name]
                # std > 1e-8: 활성이 완전히 죽은 층은 건드리지 않는다.
                # 0에 가까운 값으로 나누면 가중치가 폭발한다
                if abs(std - target_std) > tol and std > 1e-8:
                    # 가중치를 target/현재 배로 늘이면 출력 표준편차도
                    # 같은 배로 바뀐다. 선형층이라 이 비례가 성립한다.
                    # .data를 쓰는 것은 autograd를 건너뛰고 값만 고치기 위해서다
                    module.weight.data *= target_std / std
                    all_close = False

        if all_close:
            break   # 모든 층이 tol 안에 들면 끝난다

    # 훅은 반드시 떼어 낸다. 그대로 두면 학습 내내 순전파마다
    # 통계를 셈해 느려지고 메모리도 샌다
    for h in hooks:
        h.remove()

    model.train()   # 학습 결로 되돌린다
```

---

## 6. 계량 금융에서의 응용

He 초기화는 금융에서 쓰이는 대부분의 현대적 구조에서 기본이다.

**딥 헤징 신경망.** 동적 헤징 전략을 배우는 순환 신경망이나 순방향 신경망은 은닉층 전반에 ReLU나 Leaky ReLU 활성화를 쓴다. He 초기화는 (재조정 단계에 대응하는 많은 층에 걸친) 긴 시간 지평에서도 경사가 나빠지지 않고 학습할 수 있게 해 준다.

**신경망 특징 추출기를 쓰는 요인 모형.** 신경망이 자산 수익률의 횡단면에서 비선형 요인을 뽑아낼 때, 처음 순전파가 모든 자산에 대해 의미 있는 활성화를 내야 한다. He 초기화는 일부 자산이 활성화 0을, 따라서 경사 0을 받는 죽은 뉴런 문제를 막아 준다.

```python
import torch
import torch.nn as nn
import torch.nn.init as init

class DeepHedgingNetwork(nn.Module):
    """알맞은 초기화로 헤지 전략을 배우는 신경망."""

    def __init__(self, n_features, n_instruments, hidden_dim=128, n_layers=6):
        super().__init__()

        layers = []
        for i in range(n_layers):
            in_dim = n_features if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.01))
        layers.append(nn.Linear(hidden_dim, n_instruments))
        self.net = nn.Sequential(*layers)

        # Leaky ReLU 층에 He 초기화
        for m in self.net:
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0.01, nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

        # 출력층: 처음에 보수적으로 시작하도록 작게 초기화
        init.xavier_normal_(self.net[-1].weight)
        init.zeros_(self.net[-1].bias)

    def forward(self, market_state):
        """각 상품에 대한 헤지 비율을 돌려준다."""
        return self.net(market_state)
```

---

## 연습문제

**연습문제 1.**
ReLU 신경망을 위한 He 초기화 $W \sim \mathcal{N}(0, 2/n_{\text{in}})$을 유도하라.

??? success "연습문제 1 풀이"
    ReLU는 평균적으로 입력의 절반을 0으로 만든다. $\text{Var}(\text{ReLU}(x)) = \frac{1}{2}\text{Var}(x)$이다. 분산을 보존하려면 $\text{Var}(y) = \frac{n_{\text{in}}}{2} \cdot \text{Var}(w) \cdot \text{Var}(x) = \text{Var}(x)$이어야 하고, 따라서 $\text{Var}(w) = 2/n_{\text{in}}$이 필요하다.

---

**연습문제 2.**
Xavier 초기화가 깊은 ReLU 신경망에서 실패하는 이유는 무엇인가?

??? success "연습문제 2 풀이"
    Xavier는 활성화가 선형이라고 가정한다($\text{Var}(\text{act}(x)) = \text{Var}(x)$). ReLU는 분산을 절반으로 줄이므로 $L$개 층을 지나면 $\text{Var}(y) = (1/2)^L \text{Var}(x) \to 0$이 된다. He 초기화는 인수 2로 이를 보정한다.

---

**연습문제 3.**
`fan_in`과 `fan_out` 두 방식으로 He 초기화를 구현하고 각각을 언제 쓸지 설명하라.

??? success "연습문제 3 풀이"
    ```python
    # fan_in (기본값): 순전파의 분산을 보존한다
    nn.init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu')
    # fan_out: 역전파의 분산을 보존한다
    nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
    ```
    순전파의 안정성에는 `fan_in`을, 역전파 경사의 안정성이 더 중요할 때는 `fan_out`을 쓴다.

---

**연습문제 4.**
20층 ReLU 신경망을 무작위 균등, Xavier, He 초기화로 각각 학습시켜 비교하라. 층별 활성화 통계량을 그려라.

??? success "연습문제 4 풀이"
    무작위 균등에서는 몇 층 안에 활성화가 폭발하거나 주저앉는다. Xavier에서는 활성화가 지수적으로 줄어든다(층마다 절반). He에서는 20개 층 전체에 걸쳐 활성화의 크기가 한결같이 유지되어 안정적인 학습이 가능하다.

## 정리하며

| 항목 | 내용 |
|--------|--------|
| **분산 공식** | $\sigma^2 = \frac{2}{n_{\text{in}}}$ (팬인) 또는 $\frac{2}{n_{\text{out}}}$ (팬아웃) |
| **핵심 통찰** | ReLU가 분산을 절반으로 줄인다 → 인수 2로 보정한다 |
| **알맞은 대상** | ReLU, Leaky ReLU, PReLU, ELU |
| **Leaky ReLU로의 확장** | $\sigma^2 = \frac{2}{(1+a^2)\,n_{\text{in}}}$ |
| **PyTorch** | `init.kaiming_normal_`, `init.kaiming_uniform_` |
| **잔차 신경망** | 0 초기화나 $1/\sqrt{2L}$ 배율 조정과 함께 쓴다 |
| **다른 이름** | Kaiming 초기화, MSRA 초기화 |

**참고 문헌**

1. He, K., Zhang, X., Ren, S., & Sun, J. (2015). "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification." *ICCV*.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Identity Mappings in Deep Residual Networks." *ECCV*.
3. Mishkin, D., & Matas, J. (2016). "All You Need is a Good Init." *ICLR*.
4. Zhang, H., Dauphin, Y. N., & Ma, T. (2019). "Fixup Initialization: Residual Learning Without Normalization." *ICLR*.
