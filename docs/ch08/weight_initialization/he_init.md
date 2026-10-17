# He (Kaiming) 초기화
He(Kaiming) 초기화는 ReLU 계열을 쓰는 레이어의 초기 신호 크기를 조절하는 방법이다. 웨이트와 입력에 관한 독립성 가정 아래, ReLU 전후의 2차 적률을 고려해 웨이트 분산을 정한다. 분산 유지에 대한 분석이지 모든 깊이에서 안정적인 학습을 보장하는 정리는 아니다.

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

정규분포에서는 ReLU 출력의 분산이 입력 분산의 정확히 절반이 아니다. He 유도의 핵심은 평균을 뺀 분산 대신 **2차 적률**을 사용한다는 점이다. 평균이 0인 대칭 입력이면 다음 식이 성립한다.

$$
\mathbb{E}[\operatorname{ReLU}(z)^2]=\frac12\mathbb{E}[z^2]
$$

평균 0의 독립 웨이트를 곱해 다음 레이어에서 더할 때 필요한 양이 이 2차 적률이므로, 평균이 양수인 ReLU 출력도 분석할 수 있다.

### 순전파 조건

앞 절의 활성화 전 분산과 합치면 다음과 같다.

$$\text{Var}(z^{(l)}) = n_{\text{in}}\,\sigma^2\,\mathbb{E}\bigl[(h^{(l-1)})^2\bigr]$$

ReLU에서 $\mathbb{E}[h^2] = \frac{1}{2}\text{Var}(z^{(l-1)})$이므로 다음을 얻는다.

$$\text{Var}(z^{(l)}) = n_{\text{in}}\,\sigma^2 \cdot \frac{1}{2}\,\text{Var}(z^{(l-1)})$$

레이어에서 레이어로 안정적이려면 $\text{Var}(z^{(l)}) = \text{Var}(z^{(l-1)})$이어야 하므로 다음이 성립한다.

$$n_{\text{in}}\,\sigma^2 \cdot \frac{1}{2} = 1$$

$$\boxed{\sigma^2 = \frac{2}{n_{\text{in}}}}$$

이것이 **He 초기화**의 분산 공식이다(팬인 방식).

### 역전파 조건

경사 신호에 같은 분석을 적용하면 다음을 얻는다.

$$\sigma^2 = \frac{2}{n_{\text{out}}}$$

역전파 경사의 안정성이 더 중요할 때(예: 건너뛰기 연결이 없는 아주 깊은 뉴럴 넷트웍) 팬아웃 방식을 쓰기도 한다.

### Xavier와의 비교

| | Xavier | He |
|---|--------|-----|
| 순전파 조건 | $n_{\text{in}}\,\sigma^2 = 1$ | $n_{\text{in}}\,\sigma^2 = 2$ |
| 활성화에 대한 가정 | $\text{Var}(f(z)) \approx \text{Var}(z)$ | $\mathbb{E}[\operatorname{ReLU}(z)^2]=\frac12\mathbb{E}[z^2]$ (대칭 입력) |
| 분산 | $\frac{2}{n_{\text{in}} + n_{\text{out}}}$ | $\frac{2}{n_{\text{in}}}$ |
| 효과 | tanh/시그모이드에 알맞다 | ReLU에 알맞다 |

팬인 기준의 선형 레이어 분산 $1/n_{\text{in}}$에 비해 He 분산은 두 배이다. Xavier의 팬인·팬아웃 절충식과의 비율은 두 차원의 값에 따라 달라진다.

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

# 팬아웃: 역전파 그래디언트의 크기를 보존한다
init.kaiming_normal_(linear.weight, mode='fan_out', nonlinearity='relu')
```

팬인이 기본값이며 대부분의 구조에 알맞다. 뉴럴 넷트웍이 넓다가 좁아지는 모양일 때(예: 압축 병목) 팬아웃이 나을 수 있다.

---

## 3. Leaky ReLU와 PReLU로의 확장

음의 기울기가 $a$인 Leaky ReLU에 대해 다음과 같다.

$$\text{LeakyReLU}(z) = \begin{cases} z & z \geq 0 \\ az & z < 0 \end{cases}$$

2차 적률은 다음이 된다.

$$\mathbb{E}[\text{LeakyReLU}(z)^2] = \frac{1}{2}(1 + a^2)\,\text{Var}(z)$$

안정성 조건을 세우면 다음과 같다.

$$n_{\text{in}}\,\sigma^2 \cdot \frac{1 + a^2}{2} = 1$$

$$\boxed{\sigma^2 = \frac{2}{(1 + a^2)\,n_{\text{in}}}}$$

표준 ReLU($a = 0$)에서는 $2/n_{\text{in}}$을 되찾는다. $a = 0.01$인 Leaky ReLU에서는 보정이 무시할 만하다($1 + 0.0001 \approx 1$). PReLU의 음수 그래디언트를 $a=0.25$로 초기화한다면 $\sigma^2 \approx 2/(1.0625 \cdot n_{\text{in}})$이 된다.

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
    """여러 레이어에 걸쳐 ReLU와 함께 Xavier 초기화와 He 초기화를 비교한다."""
    # 학습은 하지 않는다. 무작위로 초기화한 레이어 50개에 신호를 한 번
    # 통과시켜, 레이어를 지날수록 활성의 표준편차가 어떻게 되는지만 본다.
    # 이 값이 0으로 가면 그래디언트도 함께 사라져 학습 자체가 시작되지 못한다.
    x = torch.randn(n_samples, hidden)

    results = {}
    for name, init_fn in [
        # Xavier는 분산을 2/(fan_in+fan_out)으로 잡는다. tanh처럼 원점에서
        # 기울기가 1인 활성을 가정한 값이라, ReLU와 함께 쓰면 레이어마다
        # 절반씩 죽는 몫을 메우지 못해 신호가 사그라든다
        ('Xavier Normal', init.xavier_normal_),
        # He는 2/fan_in 으로 잡아 ReLU가 죽이는 절반을 미리 메운다.
        # fan_in 방식은 순전파의 분산을 지키는 쪽이다
        ('He Normal (fan_in)', lambda w: init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu')),
        # fan_out 방식은 역전파의 분산을 지키는 쪽이다. 정사각 레이어에서는
        # fan_in과 fan_out이 같아 둘이 겹치지만, 너비가 바뀌는 망에서는 갈린다
        ('He Normal (fan_out)', lambda w: init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')),
    ]:
        h = x.clone()   # 세 방식 모두 같은 입력에서 시작해야 견줄 수 있다
        stds = []
        for _ in range(n_layers):
            # bias=False 로 두는 까닭은 바이어스가 분산에 끼어들어
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
    # Xavier 곡선은 레이어가 깊어질수록 0으로 내려간다.
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
    """각 레이어에서 그래디언트의 크기를 비교한다."""
    results = {}

    for name, init_fn in [
        ('Xavier', init.xavier_normal_),
        ('He', lambda w: init.kaiming_normal_(w, nonlinearity='relu')),
    ]:
        # 앞의 실험이 순전파에서 활성이 사그라드는지 보았다면,
        # 여기서는 역전파에서 그래디언트가 사그라드는지 본다.
        # 학습이 멈추는 진짜 원인은 이쪽이다.
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
            # retain_grad()가 없으면 중간 텐서의 .grad는 채워지지 않는다.
            # PyTorch는 잎 노드(파라메터와 입력)의 그래디언트만 남기고
            # 중간값은 메모리를 아끼려 버리기 때문이다
            h.retain_grad()
            activations.append(h)

        # 역전파.
        # 로스를 sum()으로 잡는 까닭은 실제 학습이 목적이 아니라
        # 마지막 레이어에 크기 1짜리 그래디언트를 넣어 그것이 앞으로 가며
        # 얼마나 줄어드는지만 보려는 것이기 때문이다
        loss = h.sum()
        loss.backward()

        grad_norms = [a.grad.norm().item() for a in activations]
        results[name] = grad_norms

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, norms in results.items():
        # x축을 거꾸로(n_layers -> 1) 매긴다. 그래디언트는 출력에서 입력
        # 쪽으로 흐르므로, "출력에서 얼마나 멀어졌는가"로 읽는 편이 자연스럽다
        ax.plot(range(n_layers, 0, -1), norms, linewidth=2, label=name)
    ax.set_xlabel('Distance from Output (layers)')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Magnitude vs Distance from Output')
    # 로그 축이라야 한다. 레이어마다 일정 배수로 줄어들면 로그 축에서
    # 직선이 되고, 그 그래디언트가 곧 레이어당 감쇠율이다.
    # 선형 축으로 그리면 앞쪽 몇 레이어 말고는 모두 0에 붙어 보인다
    ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    plt.savefig('gradient_flow_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

compare_gradient_flow()
```

---

## 5. 특수한 경우와 확장

### 잔차 뉴럴 넷트웍

잔차 블록 $h+F(h)$의 분산에는 두 분기의 공분산도 포함된다. 두 분기의 분산이 같고 공분산이 0이라는 근사 아래에서만 합의 분산이 두 배가 된다. 실제 구조의 정규화 레이어와 분기 스케일에 맞춰 초기화를 선택한다.

**마지막 레이어의 0 초기화.** 마지막 배치 정규화의 $\gamma = 0$으로 두거나 마지막 컨볼루션의 웨이트를 0으로 두어 처음에 $F(h) = 0$이 되게 한다.

```python
# 마지막 배치 정규화를 0으로 초기화한 ResNet 블록
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        # 컨볼루션 레이어에 He 초기화
        init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')

        # 잔차 블록이 항등함수로 시작하도록 마지막 배치 정규화를 0으로 초기화
        init.zeros_(self.bn2.weight)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(x + out)
```

**깊이에 따른 분산 배율 조정.** GPT-2를 비롯한 트랜스포머 모델들은 레이어의 수를 $L$이라 할 때 잔차 경로의 배율을 $1/\sqrt{2L}$으로 조정한다.

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

직교 초기화도 사용할 수 있다. 정방 직교행렬은 선형 변환에서 노름을 보존하지만, 직사각형 행렬은 모든 방향을 보존하지 않을 수 있다. gain을 곱하면 크기도 함께 바뀐다. 비선형 활성화를 통과한 뒤까지 노름이 보존된다고 볼 수는 없다.

```python
linear = nn.Linear(256, 256, bias=False)
init.orthogonal_(linear.weight, gain=2 ** 0.5)  # ReLU에는 gain=√2
```

`gain` 파라메터는 He 초기화의 인수 2와 마찬가지로 엑티베이션 함수의 효과를 반영한다.

### LSUV (레이어 순차 단위 분산)

실제 데이터의 배치에서 잰 출력 분산이 1이 될 때까지 각 레이어의 웨이트를 되풀이해 조정하는, 데이터 기반 초기화이다. 분포에 대한 가정에 기대지 않아도 된다.

```python
def lsuv_init(model, data_batch, target_std=1.0, max_iter=10, tol=0.05):
    """레이어 순차 단위 분산(LSUV) 초기화."""
    # He 초기화는 "이런 분포에서 뽑으면 분산이 유지될 것"이라는 이론값이다.
    # LSUV는 여기서 한 스텝 더 나아가, 실제 데이터를 흘려 보고 활성의
    # 표준편차를 재어 1이 될 때까지 웨이트를 직접 다시 재는 방식이다.
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
    # 한 번에 끝나지 않는 까닭은 앞 레이어의 웨이트를 고치면 뒤 레이어가 받는
    # 입력이 달라져 다시 재야 하기 때문이다. 그래서 앞에서 뒤로 훑는
    # 일을 여러 번 되풀이한다
    for iteration in range(max_iter):
        with torch.no_grad():   # 초기화 단계이므로 그래디언트가 필요 없다
            model(data_batch)   # 훅이 이때 활성 표준편차를 채운다

        all_close = True
        for name, module in model.named_modules():
            if name in activation_stds:
                std = activation_stds[name]
                # std > 1e-8: 활성이 완전히 죽은 레이어는 건드리지 않는다.
                # 0에 가까운 값으로 나누면 웨이트가 폭발한다
                if abs(std - target_std) > tol and std > 1e-8:
                    # 웨이트를 target/현재 배로 늘이면 출력 표준편차도
                    # 같은 배로 바뀐다. 선형 레이어라 이 비례가 성립한다.
                    # .data를 쓰는 것은 autograd를 건너뛰고 값만 고치기 위해서다
                    module.weight.data *= target_std / std
                    all_close = False

        if all_close:
            break   # 모든 레이어가 tol 안에 들면 끝난다

    # 훅은 반드시 떼어 낸다. 그대로 두면 학습 내내 순전파마다
    # 통계를 셈해 느려지고 메모리도 샌다
    for h in hooks:
        h.remove()

    model.train()   # 학습 결로 되돌린다
```

---

## 6. 계량 금융에서의 응용

금융 데이터를 다루는 ReLU 기반 MLP에도 같은 초기화 원리를 적용할 수 있다. 특정 분야라는 이유로 초기화의 안정성 조건이 달라지지는 않는다.

**시계열과 순환 구조.** 시간에 따라 반복 사용하는 웨이트는 단순한 레이어별 독립성 분석과 다르다. He 초기화만으로 긴 시계열의 경사 소실을 막을 수 없으므로 순환 웨이트의 구조와 별도 초기화도 고려한다.

**피처 추출기.** ReLU의 활성 비율과 레이어별 신호 크기를 점검한다. He 초기화가 특정 자산의 의미 있는 피처가나 죽은 뉴런 방지를 보장하는 것은 아니다.

```python
import torch
import torch.nn as nn
import torch.nn.init as init

class DeepHedgingNetwork(nn.Module):
    """알맞은 초기화로 헤지 전략을 배우는 뉴럴 넷트웍."""

    def __init__(self, n_features, n_instruments, hidden_dim=128, n_layers=6):
        super().__init__()

        layers = []
        for i in range(n_layers):
            in_dim = n_features if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.01))
        layers.append(nn.Linear(hidden_dim, n_instruments))
        self.net = nn.Sequential(*layers)

        # Leaky ReLU 레이어에 He 초기화
        for m in self.net:
            if isinstance(m, nn.Linear):
                # a는 Leaky ReLU의 음수 쪽 그래디언트이며, 위에서 준
                # LeakyReLU(0.01)과 같은 값이어야 한다. He 초기화의
                # 배율이 sqrt(2 / (1 + a^2))로 이 값에 따라 달라지기
                # 때문이다. 둘이 어긋나면 레이어마다 분산이 조금씩 밀린다
                init.kaiming_normal_(m.weight, a=0.01, nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

        # 출력 레이어: 처음에 보수적으로 시작하도록 작게 초기화.
        # 위 반복문이 이미 이 레이어까지 He로 초기화했으므로 여기서 덮어쓴다.
        # 출력 레이어 뒤에는 활성화가 없어 He가 메우려던 ReLU의 로스가
        # 애초에 없고, 자비에르가 더 좁아 헤지 비율이 0 가까이에서
        # 출발한다. 곧 학습 초반에 큰 포지션을 잡지 않는다
        init.xavier_normal_(self.net[-1].weight)
        init.zeros_(self.net[-1].bias)

    def forward(self, market_state):
        """각 상품에 대한 헤지 비율을 돌려준다."""
        return self.net(market_state)
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
평균 0의 대칭 활성화 전 값과 독립 웨이트를 가정하고 ReLU용 팬인 초기화 $\operatorname{Var}(W)=2/n_{\mathrm{in}}$을 유도하라.

</div>

??? success "풀이"
    대칭 입력에서 $\mathbb{E}[\operatorname{ReLU}(z)^2]=\mathbb{E}[z^2]/2$이다. 평균 0의 웨이트가 독립이면 다음 레이어의 활성화 전 분산은 $n_{\mathrm{in}}\operatorname{Var}(W)\mathbb{E}[\operatorname{ReLU}(z)^2]$이다. 이전 레이어의 활성화 전 분산과 같게 두면 $n_{\mathrm{in}}\operatorname{Var}(W)/2=1$을 얻는다. ReLU 출력의 분산이 정확히 절반이라고 가정할 필요는 없다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
너비가 같은 깊은 ReLU 뉴럴 넷트웍에서 gain=1인 Xavier 초기화와 He 초기화의 신호 크기 예측은 어떻게 다른가?

</div>

??? success "풀이"
    팬인과 팬아웃이 같으면 Xavier 분산은 $1/n$, He 분산은 $2/n$이다. 대칭성·독립성 근사 아래 Xavier는 다음 레이어의 활성화 전 분산을 약 절반으로 줄이고 He는 유지하도록 설계된다. 정규화 레이어, 잔차 연결과 학습 이후의 분포 변화가 있으면 이 예측이 달라질 수 있으므로 Xavier가 항상 학습에 실패한다고 단정할 수 없다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
`fan_in`과 `fan_out` 두 방식으로 He 초기화를 구현하고 각각을 언제 쓸지 설명하라.

</div>

??? success "풀이"
    ```python
    # fan_in (기본값): 순전파의 분산을 보존한다
    nn.init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu')
    # fan_out: 역전파의 분산을 보존한다
    nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
    ```
    순전파의 안정성에는 `fan_in`을, 역전파 경사의 안정성이 더 중요할 때는 `fan_out`을 쓴다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
20레이어 ReLU 모델의 초기화를 비교하는 실험을 설계하라. 어떤 통계량과 조건을 기록해야 하는가?

</div>

??? success "풀이"
    웨이트의 분포와 실제 분산, 입력 스케일, 레이어별 활성화의 평균·분산·2차 적률, 0인 활성화 비율, 경사 노름을 기록한다. 같은 구조·데이터 분할·학습 예산에서 여러 시드로 비교한다. 균등분포 자체가 문제인 것은 아니므로 초기 범위를 명시한다. He 초기화의 분산 유지 예측과 실제 학습 성능은 따로 평가한다.

## 정리하며

| 항목 | 내용 |
|--------|--------|
| **분산 공식** | $\sigma^2 = \frac{2}{n_{\text{in}}}$ (팬인) 또는 $\frac{2}{n_{\text{out}}}$ (팬아웃) |
| **핵심 통찰** | 대칭 입력에서 ReLU의 2차 적률이 절반이 되는 효과를 반영한다 |
| **알맞은 대상** | ReLU 및 음수 그래디언트를 반영한 Leaky ReLU·PReLU |
| **Leaky ReLU로의 확장** | $\sigma^2 = \frac{2}{(1+a^2)\,n_{\text{in}}}$ |
| **PyTorch** | `init.kaiming_normal_`, `init.kaiming_uniform_` |
| **잔차 뉴럴 넷트웍** | 0 초기화나 $1/\sqrt{2L}$ 배율 조정과 함께 쓴다 |
| **다른 이름** | Kaiming 초기화, MSRA 초기화 |

**참고 문헌**

1. He, K., Zhang, X., Ren, S., & Sun, J. (2015). "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification." *ICCV*.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Identity Mappings in Deep Residual Networks." *ECCV*.
3. Mishkin, D., & Matas, J. (2016). "All You Need is a Good Init." *ICLR*.
4. Zhang, H., Dauphin, Y. N., & Ma, T. (2019). "Fixup Initialization: Residual Learning Without Normalization." *ICLR*.
