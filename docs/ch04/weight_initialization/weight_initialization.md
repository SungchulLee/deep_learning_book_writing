# 가중치 초기화

첫 경사 갱신이 있기 전, 신경망의 가중치는 순전파의 활성화와 역전파의 경사를 온전히 결정한다. 처음 가중치의 크기가 너무 크면 활성화와 경사가 깊이에 따라 지수적으로 폭발하고, 너무 작으면 사라진다. 어느 쪽이든 학습이 발산하거나 멈춰 선다. 가중치 초기화 전략은 처음 순전파와 역전파 동안 신호의 크기가 층에 걸쳐 대체로 일정하게 유지되도록 각 층 매개변수의 분산을 정한다.

이 절은 분산 조건을 제일원리에서 유도하고, 널리 쓰이는 두 방식인 Xavier(Glorot)와 He(Kaiming)의 동기를 밝히며, 둘 중 무엇을 고를지에 대한 실무 지침을 제시한다.

---

## 1. 학습 목표

이 절을 마치면 다음을 이해하게 된다.

1. 소박한 무작위 초기화가 깊은 신경망에서 실패하는 이유
2. 순전파와 역전파의 분산 전파 분석
3. 활성화 함수의 선택이 올바른 분산 공식을 어떻게 정하는지
4. 초기화와 정규화 층의 관계
5. PyTorch에 내장된 초기화 도구와 그것을 쓰는 때

---

## 2. 초기화 문제

### 깊은 신경망에서의 신호 전파

층이 $L$개인 순방향 신경망을 생각하자. 층 $l$에서 활성화 전 값은 다음과 같다.

$$z^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}$$

여기서 $h^{(l-1)}$은 앞 층의 활성화이다(입력에 대해서는 $h^{(0)} = x$). 활성화는 다음과 같다.

$$h^{(l)} = f\bigl(z^{(l)}\bigr)$$

여기서 $f$은 어떤 비선형 함수이다.

가중치를 $W_{ij}^{(l)} \sim \mathcal{N}(0, \sigma^2)$에서 뽑고 편향을 0으로 두면, (입력이 평균 0의 i.i.d.이고 가중치와 독립이라는 가정 아래) 활성화 전 단위 하나의 분산은 다음과 같다.

$$\text{Var}\bigl(z_j^{(l)}\bigr) = n_{l-1}\,\sigma^2\,\text{Var}\bigl(h^{(l-1)}\bigr)$$

여기서 $n_{l-1}$은 팬인, 즉 층 $l$로 들어오는 입력 단위의 수이다.

### 활성화의 폭발과 소실

층 $L$개에 걸쳐 분산은 곱셈적으로 쌓인다.

$$\text{Var}\bigl(z^{(L)}\bigr) \propto \prod_{l=1}^{L} \bigl(n_{l-1}\,\sigma^2 \cdot c_l\bigr) \cdot \text{Var}(x)$$

여기서 $c_l$은 활성화 함수가 분산에 미치는 영향을 나타낸다. 각 층에서 $n_{l-1}\,\sigma^2\,c_l > 1$이면 분산이 지수적으로 커지고, $< 1$이면 지수적으로 줄어든다. 50층 신경망이라면 층마다 10%만 어긋나도 $(1.1)^{50} \approx 117$이나 $(0.9)^{50} \approx 0.005$의 배율이 생긴다.

### 경사의 전파

같은 분석이 거꾸로도 적용된다. 역전파 중에 층 $l$의 활성화 전 값에 대한 경사는 다음을 포함한다.

$$\frac{\partial \mathcal{L}}{\partial z^{(l)}} = \bigl(W^{(l+1)}\bigr)^\top \frac{\partial \mathcal{L}}{\partial z^{(l+1)}} \odot f'\bigl(z^{(l)}\bigr)$$

경사 신호의 분산은 (팬아웃인) $n_{l+1}\,\sigma^2$과 활성화 함수의 도함수에 달려 있다. 역전파가 안정적이려면 다음이 필요하다.

$$n_{l+1}\,\sigma^2\,\mathbb{E}\bigl[f'(z)^2\bigr] \approx 1$$

### 핵심 설계 원리

알맞은 초기화는 두 조건이 대체로 함께 만족되도록 $\sigma^2$을 고르는 일이다.

$$n_{\text{in}}\,\sigma^2\,c_{\text{fwd}} \approx 1 \qquad \text{(forward stability)}$$

$$n_{\text{out}}\,\sigma^2\,c_{\text{bwd}} \approx 1 \qquad \text{(backward stability)}$$

여기서 $c_{\text{fwd}}$과 $c_{\text{bwd}}$은 활성화 함수에 따라 달라진다.

---

## 3. 소박한 초기화: 무엇이 잘못되는가

### 상수 초기화

모든 가중치를 (0을 포함해) 같은 값으로 두는 것은 치명적이다. 모든 뉴런이 같은 출력을 계산하고 같은 경사를 받아 똑같이 갱신된다. 너비와 무관하게 신경망이 사실상 층마다 뉴런 하나인 셈이 되며, 이 실패를 **대칭성 문제**라 부른다.

### 표준정규분포 (sigma = 1)

$n_{\text{in}} = 512$인 층에서 활성화 전 분산은 $512 \cdot 1 \cdot \text{Var}(h) = 512\,\text{Var}(h)$이다. 몇 층만 지나면 활성화가 `float32`의 범위를 넘친다.

### 너무 작은 분산 (sigma = 0.001)

활성화 전 분산이 $512 \times 10^{-6}\,\text{Var}(h) \approx 5 \times 10^{-4}\,\text{Var}(h)$이다. 활성화가 0 가까이로 주저앉고 경사가 사라진다.

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def visualize_initialization_problem(n_layers=20, hidden_dim=256, n_samples=512):
    """초기화의 척도가 활성화의 크기에 미치는 영향을 보인다."""
    # 학습은 전혀 하지 않는다. 무작위로 초기화한 층 20개에 신호를 한 번
    # 통과시키기만 해도 초기화가 잘못되면 신호가 죽거나 포화된다는 것을
    # 보이려는 것이다. 곧 이것은 "학습 전에 이미 결정되는" 문제다.
    x = torch.randn(n_samples, hidden_dim)

    scales = {
        'Too small (0.001)': 0.001,   # 층마다 신호가 줄어 결국 0으로 사그라든다
        'Too large (1.0)': 1.0,       # 층마다 신호가 커져 tanh가 +-1로 포화된다
        # 자비에르(글로로) 초기화: 분산을 2/(fan_in + fan_out)으로 잡는다.
        # 순전파와 역전파 양쪽에서 분산이 유지되도록 두 값의 평균을 쓴다.
        # tanh처럼 원점 부근에서 기울기가 1인 활성에 알맞다
        'Proper (Xavier)': (2.0 / (hidden_dim + hidden_dim)) ** 0.5,
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (label, scale) in zip(axes, scales.items()):
        h = x.clone()   # 세 경우 모두 같은 입력에서 시작해야 견줄 수 있다
        means, stds = [], []

        for _ in range(n_layers):
            # 층마다 새 가중치를 뽑는다. scale이 표준편차를 정한다
            W = torch.randn(hidden_dim, hidden_dim) * scale
            h = torch.tanh(h @ W)
            means.append(h.mean().item())
            # 표준편차가 이 실험의 핵심 값이다. 이것이 0으로 가면
            # 기울기도 함께 사라지고, 1 가까이 붙박이면 포화된 것이다
            stds.append(h.std().item())

        ax.plot(stds, 'b-', linewidth=2)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Activation Std')
        ax.set_title(label)
        # 세 그림의 y축을 같은 범위로 맞춰야 눈으로 견줄 수 있다
        ax.set_ylim(0, 1.5)

    plt.tight_layout()
    plt.savefig('init_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

visualize_initialization_problem()
```

---

## 4. 분산 분석의 틀

### 가정

아래 유도는 다음의 표준적인 가정을 쓴다.

1. 가중치 $W_{ij}^{(l)}$은 평균이 0이고 분산이 $\sigma_l^2$인 i.i.d.이다
2. 편향은 0으로 초기화한다
3. 각 층의 입력은 평균이 0이다 (중심화나 정규화 후에는 대체로 참이다)
4. 가중치와 활성화는 독립이다 (엄밀히는 초기화 시점에만 참이다)

이 가정들은 첫 순전파에서는 정확히 성립하며 처음 몇 번의 경사 단계에서도 좋은 근사가 된다.

### 순전파의 분산

층 $l$의 활성화 전 단위 하나에 대해 다음이 성립한다.

$$z_j^{(l)} = \sum_{i=1}^{n_{l-1}} W_{ij}^{(l)}\,h_i^{(l-1)}$$

분산을 취하면 다음과 같다.

$$\text{Var}(z_j^{(l)}) = n_{l-1}\,\text{Var}(W_{ij}^{(l)})\,\mathbb{E}\bigl[(h_i^{(l-1)})^2\bigr]$$

여기서 $\text{Var}(XY) = \text{Var}(X)\,\text{Var}(Y) + \text{Var}(X)\,[\mathbb{E}(Y)]^2 + \text{Var}(Y)\,[\mathbb{E}(X)]^2$을 썼으며, $\mathbb{E}(X) = 0$일 때 이는 $\text{Var}(X)\,\mathbb{E}(Y^2)$으로 간단해진다.

활성화 $h^{(l)} = f(z^{(l)})$에 대해서는 $\mathbb{E}[f(z)^2]$을 $\text{Var}(z)$으로 나타내야 한다. 이 인수가 활성화 함수에 따라 달라지며, 바로 여기서 Xavier와 He 초기화가 갈린다.

### 역전파의 분산

경사 신호에 같은 논리를 적용하면 다음을 얻는다.

$$\text{Var}\!\left(\frac{\partial \mathcal{L}}{\partial h_i^{(l-1)}}\right) = n_l\,\text{Var}(W_{ij}^{(l)})\,\mathbb{E}\bigl[f'(z_j^{(l)})^2\bigr]\,\text{Var}\!\left(\frac{\partial \mathcal{L}}{\partial z_j^{(l)}}\right)$$

$n_{\text{in}} = n_{\text{out}}$이 아니라면 순전파와 역전파의 안정성을 동시에 이루기는 대체로 불가능하므로, 실용적인 방식들은 둘 사이에서 절충한다.

---

## 5. PyTorch의 초기화 도구

PyTorch는 `torch.nn.init`에 초기화 함수를 제공한다.

```python
import torch.nn as nn
import torch.nn.init as init

linear = nn.Linear(256, 128)

# Xavier(Glorot) 초기화
init.xavier_uniform_(linear.weight)
init.xavier_normal_(linear.weight)

# He(Kaiming) 초기화
init.kaiming_uniform_(linear.weight, nonlinearity='relu')
init.kaiming_normal_(linear.weight, nonlinearity='relu')

# 다른 방식들
init.orthogonal_(linear.weight)          # 노름을 정확히 보존한다
init.sparse_(linear.weight, sparsity=0.1)  # 희소 초기화

# 편향 초기화 (보통 0)
init.zeros_(linear.bias)
```

### 신경망 전체에 대한 사용자 정의 초기화

```python
def init_weights(module):
    """모든 선형층과 합성곱 층에 He 초기화를 적용한다."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        # He(카이밍) 초기화: 분산을 2/fan_in 으로 잡는다.
        # ReLU가 입력의 절반(음수 쪽)을 0으로 죽이므로 분산이 절반으로
        # 줄어든다. 그 손실을 미리 2배로 메워 두어야 층을 지나며
        # 신호가 사그라들지 않는다. nonlinearity='relu'가 이 2배를 정한다
        init.kaiming_normal_(module.weight, nonlinearity='relu')

        # 편향은 0에서 시작한다. 편향에까지 무작위를 넣으면 대칭을 깨는
        # 데 도움이 되지 않으면서 초기 출력만 흔들린다
        if module.bias is not None:
            init.zeros_(module.bias)

    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
        # 정규화 층의 weight(감마)와 bias(베타)는 항등으로 시작해야 한다.
        # 감마=1, 베타=0 이면 처음에는 정규화 결과를 그대로 통과시키고,
        # 학습이 진행되며 필요한 만큼만 늘이고 옮기게 된다
        init.ones_(module.weight)
        init.zeros_(module.bias)

model = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 10),
)

# apply는 모든 하위 모듈에 함수를 재귀적으로 건다. 그래서 위 함수가
# isinstance로 갈래를 나누어 층 종류마다 다른 초기화를 하도록 짜여 있다
model.apply(init_weights)
```

---

## 6. 초기화 선택 안내

| 활성화 함수 | 권장 초기화 | 분산 공식 |
|---------------------|-----------------|-----------------|
| 시그모이드, Tanh | Xavier (Glorot) | $\sigma^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}}$ |
| ReLU | 허(카이밍) | $\sigma^2 = \frac{2}{n_{\text{in}}}$ |
| Leaky ReLU (기울기 $a$) | He (수정판) | $\sigma^2 = \frac{2}{(1 + a^2)\,n_{\text{in}}}$ |
| SELU | LeCun 정규 | $\sigma^2 = \frac{1}{n_{\text{in}}}$ |
| GELU, Swish, Mish | He 또는 Xavier | 경험적으로 어느 쪽이든 무난하다 |

### 실무 규칙

1. **정규화가 없는 ReLU 신경망**: He 초기화를 강하게 권한다. Xavier를 쓰면 깊은 ReLU 신경망에서 활성화가 주저앉는다.
2. **배치 정규화나 층 정규화가 있는 신경망**: 정규화가 층마다 활성화의 배율을 다시 맞추므로 초기화의 중요도가 낮아진다. Xavier와 He 모두 통한다.
3. **트랜스포머**: 흔히 Xavier나 배율을 조정한 정규분포 $\mathcal{N}(0, 1/\sqrt{d_{\text{model}}})$을 쓰며, 잔차 연결은 따로 다룬다($1/\sqrt{2L}$으로 배율 조정).
4. **잔차 신경망**: 각 잔차 블록의 마지막 층을 0으로 초기화하여 처음에는 블록이 항등함수를 계산하게 만들기도 한다.

---

## 7. 정규화와의 상호작용

정규화 층(배치 정규화, 층 정규화)은 층의 경계마다 정해진 통계량을 강제하여 신호 전파 문제를 크게 누그러뜨린다. 정규화가 있으면 다음과 같다.

- 신경망이 초기화 분산에 덜 민감해진다. 정규화가 층마다 규모를 "바로잡기" 때문이다.
- 그래도 알맞게 초기화하면 학습이 더 빨리 시작된다. 이동 통계량이 안정되기 전 처음 몇 걸음에서 더 유익한 경사가 나오기 때문이다.
- 아주 깊은 신경망(100층 이상)에서는 He 초기화와 정규화를 함께 쓰는 것이 여전히 모범 사례이다.

정규화가 없다면 알맞은 초기화는 **결정적**이다. 그러지 않으면 신경망이 아예 학습되지 않을 수 있다.

---

## 8. 계량 금융에서의 응용

계량 금융에서 초기화는 특히 눈여겨볼 만하다.

**온라인 학습과 웜스타트.** 모델 가중치를 실시간으로 갱신하는 실전 시스템(예: 온라인 시장 조성 모델)에서는 새 층이나 확장된 구성 요소를 어떻게 초기화하느냐가 모델이 국면 변화에 얼마나 빨리 적응하는지를 좌우한다.

**데이터가 적을 때의 전이 학습.** 작은 금융 데이터셋으로 사전학습 모델을 미세 조정할 때, 과제에 특화된 새 갈래를 어떻게 초기화하느냐가 수렴 속도와 사전학습된 표현을 파국적으로 잊을 위험 둘 다에 영향을 준다.

**앙상블의 다양성.** 위험 관리를 위해 모델 앙상블을 만들 때 서로 다른 무작위 초기화는 서로 다른 함수를 배우게 한다. 이 다양성의 질은 초기화 분포가 손실 지형에서 서로 구별되는 끌림 골짜기들에 걸쳐 있는지에 달려 있다.

```python
# 예: 출력에 제약이 있는 가격 예측 신경망 초기화
class PricingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_layers=4):
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_d = input_dim if i == 0 else hidden_dim
            layers.extend([nn.Linear(in_d, hidden_dim), nn.ReLU()])
        self.backbone = nn.Sequential(*layers)
        self.output_head = nn.Linear(hidden_dim, 1)

        # ReLU 뼈대에 He 초기화
        for m in self.backbone:
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
                init.zeros_(m.bias)

        # 출력 머리는 작게 초기화한다 — 예측이 0 근처에서 시작한다
        init.xavier_normal_(self.output_head.weight)
        init.zeros_(self.output_head.bias)

    def forward(self, x):
        h = self.backbone(x)
        return torch.softplus(self.output_head(h))  # 양수임을 강제한다
```

---

## 연습문제

**연습문제 1.**
모든 가중치를 0으로 초기화하면 신경망이 학습하지 못하는 이유를 설명하라.

??? success "연습문제 1 풀이"
    가중치가 0이면 모든 뉴런이 같은 출력(0)을 계산하고 같은 경사를 받아 똑같이 갱신된다. 이 대칭이 결코 깨지지 않으므로 너비와 무관하게 신경망이 뉴런 하나처럼 행동한다. 무작위 초기화가 이 대칭을 깨뜨린다.

---

**연습문제 2.**
Xavier(Glorot) 초기화 $W \sim \mathcal{U}(-\sqrt{6/(n_{\text{in}}+n_{\text{out}})}, \sqrt{6/(n_{\text{in}}+n_{\text{out}})})$을 유도하라.

??? success "연습문제 2 풀이"
    분산을 보존하려면 $\text{Var}(y) = n_{\text{in}} \cdot \text{Var}(w) \cdot \text{Var}(x)$이다. $\text{Var}(y) = \text{Var}(x)$으로 두면 $\text{Var}(w) = 1/n_{\text{in}}$이다. 순전파와 역전파를 평균하면 $\text{Var}(w) = 2/(n_{\text{in}}+n_{\text{out}})$이다. 균등분포 $[-a, a]$에서는 $\text{Var} = a^2/3$이므로 $a = \sqrt{6/(n_{\text{in}}+n_{\text{out}})}$을 얻는다.

---

**연습문제 3.**
가중치를 너무 크게 초기화하면 어떻게 되는가? 너무 작게 하면?

??? success "연습문제 3 풀이"
    너무 크면 활성화가 (시그모이드/tanh에서) 포화하거나 (ReLU에서) 폭발하고, 경사가 사라지거나 폭발하며, 학습이 발산한다. 너무 작으면 활성화가 0으로 주저앉고 경사가 사라져 학습이 아주 느리거나 아예 되지 않는다. 알맞은 초기화는 층에 걸쳐 활성화와 경사의 크기를 유지한다.

---

**연습문제 4.**
Xavier와 He 초기화를 PyTorch로 구현하고 10층 신경망에서 학습의 움직임을 비교하라.

??? success "연습문제 4 풀이"
    ```python
    for m in model.modules():
        if isinstance(m, nn.Linear):
            # Xavier: nn.init.xavier_uniform_(m.weight)
            # He: nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            pass
    ```
    He 초기화는 ReLU가 분산을 절반으로 줄이는 것을 반영하므로 ReLU 신경망에서 더 잘 통한다.

## 정리하며

| 항목 | 핵심 통찰 |
|--------|-------------|
| **핵심 문제** | 가중치의 분산이 활성화와 경사가 쓸 만한 범위에 머무는지를 정한다 |
| **순전파 조건** | $n_{\text{in}} \sigma^2 c_{\text{fwd}} \approx 1$이 활성화의 분산을 보존한다 |
| **역전파 조건** | $n_{\text{out}} \sigma^2 c_{\text{bwd}} \approx 1$이 경사의 분산을 보존한다 |
| **Xavier** | 대칭적인 활성화에 대해 순전파와 역전파 사이에서 절충한다 |
| **He** | ReLU가 분포의 절반을 0으로 만드는 것을 반영한다 |
| **정규화가 있을 때** | 초기화의 중요도는 낮아지지만 여전히 이롭다 |

**참고 문헌**

1. Glorot, X., & Bengio, Y. (2010). "Understanding the difficulty of training deep feedforward neural networks." *AISTATS*.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2015). "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification." *ICCV*.
3. Saxe, A. M., McClelland, J. L., & Ganguli, S. (2014). "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks." *ICLR*.
4. Mishkin, D., & Matas, J. (2016). "All You Need is a Good Init." *ICLR*.
