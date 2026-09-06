# RealNVP과 짝지음 층
짝지음 층은 요즘 고르게 하는 흐름의 등뼈이다. 생각은 놀랄 만큼 단순하다. 곧 들임을 둘로 쪼개어 한쪽은 그대로 지나게 하고 다른 쪽은 첫째에서 셈한 매개변수로 바꾼다. 이러면 세우는 방식 자체로 뒤집을 수 있음이 보장되고 행렬식을 $O(d)$에 셈하는 세모 야코비 행렬이 나온다. **RealNVP**(실수 부피 보존 안 하는 흐름, Dinh et al., 2017)는 가림막을 번갈아 쓴 아핀 짝지음 층을 쌓으면 차원 높은 밀도 어림에 쓸 만하고 규모를 키울 수 있는 얼개가 나옴을 보였다.

## 짝지음의 원리

### 쪼개고 바꾸고 합치기

들임 $z \in \mathbb{R}^D$을 $z = (z_A, z_B)$으로 나눈다:

$$x_A = z_A \qquad\text{(항등)}$$

$$x_B = g(z_B;\;\theta(z_A)) \qquad\text{(매개변수화한 바꿈)}$$

함수 $\theta(\cdot)$(*조건개*)은 **아무** 신경망이나 될 수 있다. 뒤집을 수 있을 필요도 없고 야코비 행렬식에 보태지도 않는다. 유일한 요건은 $\theta$을 고정할 때 $g$이 $z_B$에 대해 뒤집을 수 있어야 한다는 것이다.

### 이것이 왜 되나

**뒤집을 수 있음.** $x_A = z_A$이므로 내놓기에서 조건 앎을 되찾는다. 그러면 $z_B = g^{-1}(x_B;\;\theta(x_A))$이다.

**효율 좋은 야코비.** 야코비 행렬은 덩이 세모꼴이다:

$$J = \begin{pmatrix} I & 0 \\ \partial x_B / \partial z_A & \partial g / \partial z_B \end{pmatrix}$$

그래서 $\det J = \det(\partial g / \partial z_B)$이며, 이는 (아주 복잡할 수 있는) 조건개 $\theta$이 아니라 $g$이 $z_B$을 어떻게 바꾸는지에만 달렸다.

## 아핀 짝지음

가장 널리 쓰는 짝지음 바꿈은 *아핀* 짝지음이다:

$$x_B = z_B \odot \exp\!\bigl(s(z_A)\bigr) + t(z_A)$$

여기서 $s(\cdot)$과 $t(\cdot)$은 잣수 그물과 옮김 그물이다.

### 역

$$z_B = \bigl(x_B - t(x_A)\bigr) \odot \exp\!\bigl(-s(x_A)\bigr)$$

역은 앞먹임만큼 값싸며 되풀이 절차가 없다.

### 로그 행렬식

$z_B$에 대한 야코비 행렬이 성분 $\exp(s_i)$의 대각 행렬이므로:

$$\log|\det J| = \sum_i s_i(z_A)$$

잣수 그물이 내놓는 것의 합일 뿐이다.

### 구현

```python
import torch
import torch.nn as nn
import numpy as np


class AffineCouplingLayer(nn.Module):
    """아핀 짝지음 층 하나."""

    def __init__(self, dim, mask, hidden_dims=(256, 256)):
        super().__init__()
        self.dim = dim
        self.register_buffer("mask", mask.float())
        d_cond = int(mask.sum().item())
        d_trans = dim - d_cond

        layers = []
        prev = d_cond
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, d_trans * 2))
        self.net = nn.Sequential(*layers)
        # 거의 항등인 첫자리매김
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _st(self, x_cond):
        params = self.net(x_cond)
        s, t = params.chunk(2, dim=-1)
        s = torch.tanh(s) * 2          # 안정을 위해 로그 잣수를 가둔다
        return s, t

    def forward(self, x):
        x_cond = x[:, self.mask.bool()]
        s, t = self._st(x_cond)
        y = x.clone()
        y[:, ~self.mask.bool()] = x[:, ~self.mask.bool()] * torch.exp(s) + t
        return y, s.sum(dim=-1)

    def inverse(self, y):
        y_cond = y[:, self.mask.bool()]   # x_cond과 같다
        s, t = self._st(y_cond)
        x = y.clone()
        x[:, ~self.mask.bool()] = (y[:, ~self.mask.bool()] - t) * torch.exp(-s)
        return x, -s.sum(dim=-1)
```

### 잣수 매개변수화

$\exp(s)$을 쓰면 잣수가 양수임이 보장된다. 실전에서는 극단 값을 막으려 날 잣수 내놓기를 가두거나 $\tanh$으로 눌러 준다:

```python
# 가둔 로그 잣수
s = torch.tanh(s_raw) * scale_bound      # 예컨대 scale_bound = 2

# 다른 길: 매끄럽게 양수로 만드는 부드러운 정류 선형
scale = torch.nn.functional.softplus(s_raw)
```

## 쪼개기와 가리기 전략

짝지음 층 하나는 차원의 절반을 그대로 둔다. 층마다 **가림막을 번갈아** 쓰면 모든 차원이 결국 바뀐다.

### 흔한 결

**반쪽 쪼개기(벡터):** 짝수 층에서는 앞 절반을, 홀수 층에서는 뒤 절반을 그대로 둔다.

**바둑판(그림):** 층마다 번갈아 쓰는 공간 바둑판 결.

**채널마다(그림):** 채널 차원을 따라 쪼갠다.

```python
def alternating_mask(dim, layer_idx):
    mask = torch.zeros(dim)
    if layer_idx % 2 == 0:
        mask[:dim // 2] = 1
    else:
        mask[dim // 2:] = 1
    return mask
```

## RealNVP 얼개

RealNVP는 가림막을 번갈아 쓴 아핀 짝지음 층을 쌓고 익히기 안정을 위해 묶음 고르게 맞추기를 더한다. 그림 자료에는 여러 잣수 얼개를 쓴다.

### 핵심 설계

```
For each scale:
    For L coupling blocks:
        Affine coupling (checkerboard mask)
        Batch normalisation
    Affine coupling (channel mask)
    Factor out half channels → base distribution
```

### 온전한 1차원 RealNVP

```python
class RealNVP(nn.Module):
    """벡터 자료를 위한 RealNVP."""

    def __init__(self, dim, n_layers=8, hidden_dims=(256, 256)):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            mask = alternating_mask(dim, i)
            self.layers.append(AffineCouplingLayer(dim, mask, hidden_dims))

    def forward(self, x):
        ld = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            x, log_det = layer.forward(x)
            ld += log_det
        return x, ld

    def inverse(self, z):
        ld = torch.zeros(z.shape[0], device=z.device)
        for layer in reversed(self.layers):
            z, log_det = layer.inverse(z)
            ld += log_det
        return z, ld

    def log_prob(self, x):
        z, ld = self.forward(x)
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
        return log_pz + ld

    def sample(self, n, device="cpu"):
        z = torch.randn(n, self.dim, device=device)
        x, _ = self.inverse(z)
        return x
```

### 여러 잣수 얼개(그림)

그림 자료에서 RealNVP는 공간 해상도를 채널 깊이와 맞바꾸는 **짜내기** 연산을 쓰고, 이어 채널의 절반을 곧바로 바탕 분포로 보내는 **떼어 내기** 걸음을 쓴다. 그러면 뒤쪽 잣수에서 셈이 줄고 앞선 층은 잔 공간 짜임을, 깊은 층은 뜻을 나타내게 된다.

```
Input (3, 32, 32)
  → Squeeze → (12, 16, 16) → Coupling blocks → Factor out (6, 16, 16)
  → Squeeze → (24, 8, 8)   → Coupling blocks → Factor out (12, 8, 8)
  → Squeeze → (48, 4, 4)   → Coupling blocks → Base distribution
```

## 조건개 그물

조건개 $\theta(z_A)$은 어떤 신경망이든 될 수 있다. 그 얼개가 층마다의 표현력을 다스린다:

**여러 층 인식개**는 벡터 자료에 쓴다. 정류 선형을 곁들인 숨은 층 두셋과 0으로 첫자리매김한 내놓기.

**누비기 신경망**은 그림 자료에 쓴다. $3 \times 3$ 누비기 층 여럿, 더 깊은 조건개에는 ResNet 덩이.

**핵심 설계 원칙:** 조건개의 내놓기를 0으로 첫자리매김해 흐름이 항등 바꿈으로(또는 그에 가깝게) 시작하도록 하여 초반 익히기를 안정되게 한다.

## 덧셈 짝지음과 아핀 짝지음

덧셈 짝지음($x_B = z_B + t(z_A)$)은 $s = 0$인 특별한 경우이다. 야코비 행렬식이 늘 1이라 부피를 지킨다. NICE(Dinh et al., 2015)가 이를 썼다. 배울 수 있는 잣수를 가진 아핀 짝지음이 딱 잘라 더 표현력이 좋아 표준이 되었다.

## 학습

```python
def train_realnvp(model, data, epochs=100, batch_size=256, lr=1e-3):
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data),
        batch_size=batch_size, shuffle=True,
    )
    for epoch in range(epochs):
        for (batch,) in loader:
            loss = -model.log_prob(batch).mean()
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
```

## 요약

짝지음 층은 뒤집을 수 있음의 보장, $O(d)$ 야코비 셈하기, 조건개를 아무렇게나 고를 수 있는 융통성을 준다. RealNVP는 아핀 짝지음에 번갈아 쓰는 가림막과 여러 잣수 처리를 아울러 이 설계가 복잡하고 차원 높은 분포로 규모를 키움을 보였다. 이 얼개는 여느 바탕으로 남아 있으며 Glow와 신경 스플라인 흐름의 밑바탕이다.

## 참고 문헌

1. Dinh, L., Krueger, D. & Bengio, Y. (2015). NICE: Non-linear Independent Components Estimation. *ICLR Workshop*.
2. Dinh, L., Sohl-Dickstein, J. & Bengio, S. (2017). Density Estimation Using Real-NVP. *ICLR*.
3. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.

## 연습문제

**연습문제 1.**
변수 바꿈 식을 설명하고 그것이 고르게 하는 흐름 모델의 고갱이인 까닭을 밝혀라.

??? success "연습문제 1 풀이"
    $z = f(x)$이 $x = f^{-1}(z)$인 뒤집을 수 있는 바꿈이면 변수 바꿈 식이 밀도를 이어 준다. 곧 $p_x(x) = p_z(f(x)) \left|\det \frac{\partial f}{\partial x}\right|$이다. 고르게 하는 흐름이 단순한 바탕 분포 $p_z$(예컨대 정규 분포)에 뒤집을 수 있는 바꿈의 사슬을 써서 $p_x$을 넌지시 정하므로 이것이 고갱이이다. 이 식 덕분에 (최대 가능도로 익히는 데 필요한) 정확한 밀도 값매김과 ($f^{-1}$으로 바탕 표본을 바꾸는) 뽑기가 된다. 핵심 과제는 역과 야코비 행렬식을 모두 효율 좋게 셈할 수 있는 바꿈을 설계하는 것이다.

---

**연습문제 2.**
RealNVP의 짝지음 층 얼개를 적고 그것이 왜 효율 좋은 셈하기를 되게 하는지 설명하라.

??? success "연습문제 2 풀이"
    RealNVP는 들임 $x$을 반쪽 둘 $(x_1, x_2)$으로 쪼갠다. 짝지음 층은 $y_1 = x_1$(항등), $y_2 = x_2 \odot \exp(s(x_1)) + t(x_1)$으로 바꾸며 $s$과 $t$은 신경망이다. 효율이 좋은 까닭은 이렇다. (1) **역**이 뻔하다. 곧 $x_1 = y_1$, $x_2 = (y_2 - t(y_1)) \odot \exp(-s(y_1))$이다. (2) **야코비 행렬**이 대각이 $[1, \ldots, 1, \exp(s(x_1))]$인 아래 세모꼴이라 $\log|\det J| = \sum s(x_1)_i$을 $O(d)$ 시간에 셈한다. 그물 $s, t$은 뒤집을 수 있음이나 야코비 셈하기에 영향을 주지 않고 아무리 복잡해도 된다.

---

**연습문제 3.**
고르게 하는 흐름은 이어지는 짝지음 층에서 왜 고정할 차원을 번갈아 바꾸는가?

??? success "연습문제 3 풀이"
    짝지음 층 하나는 차원의 절반($x_2$)만 바꾸고 나머지 절반($x_1$)은 그대로 둔다. 늘 같은 방식으로 쪼개면 $x_1$ 차원은 결코 바뀌지 않는다. 쪼개기를 번갈아 하면(짝수 층은 $x_1$을, 홀수 층은 $x_2$을 고정) 모든 차원이 결국 바뀐다. $L$개 층을 지나면 바꿈의 사슬을 거쳐 모든 차원 사이로 앎이 흐른다. 더 정교한 전략에는 짝지음 층 사이에 아무 자리 바꿈이나 (Glow처럼) 배운 1x1 누비기를 두는 것이 있다.

---

**연습문제 4.**
만들어 내는 모델에서 고르게 하는 흐름을 변분 자기 부호기, 맞겨루는 그물과 견주어라. 흐름만의 이점은 무엇인가?

??? success "연습문제 4 풀이"
    | 성질 | 흐름 | 변분 자기 부호기 | 맞겨루는 그물 |
    |----------|-------|------|------|
    | **정확한 가능도** | 예 | 하한(증거 하한) | 아니오 |
    | **정확한 추론** | 예($z = f(x)$) | 어림($q_\phi$) | 부호기 없음 |
    | **표본 품질** | 좋음 | 흔히 흐릿함 | 가장 좋음 |
    | **익히기 안정** | 안정(최대 가능도) | 안정(증거 하한) | 불안정(맞겨룸) |
    | **얼개 제약** | 뒤집을 수 있어야 함 | 자유로움 | 자유로움 |

    흐름만의 이점: (1) 정확한 로그 가능도 값매김으로 원칙 있는 모델 견줌이 된다. (2) 정확한 숨은 추론으로 정밀한 사이 끼움이 된다. (3) 뽑기와 밀도 값매김이 모두 효율 좋다. 주된 한계는 뒤집을 수 있어야 한다는 제약이며 이것이 얼개 고름을 옭아맨다.
