# 되짚음으로 하는 베이즈
## 개요

**되짚음으로 하는 베이즈**(블런델 외, 2015)는 변분 추론으로 베이즈 신경망을 익히는 실전 알고리즘이다. **매개변수 바꾸기 재주** 덕분에 표준 되짚음으로 증거 아래 경계(ELBO)를 최적화하여 가중값에 대한 분포를 배운다.

!!! note "온전히 다루기"
    그 자리 매개변수 바꾸기, 가중값에 대한 고르게 하는 흐름, 두루 다룬 잣대 실험 같은 앞선 주제는 **[39장: 변분 베이즈 신경망](../../ch39/bayesian_methods/variational_bnn.md)**을 보아라.

---

## 변분 목표

다룰 수 없는 뒤확률을 다룰 수 있는 변분 분포로 어림한다:

$$
p(\theta \mid \mathcal{D}) \approx q_\phi(\theta)
$$

ELBO 목표([ELBO](../variational_inference/elbo.md) 참고):

$$
\boxed{\mathcal{L}(\phi) = \mathbb{E}_{q_\phi(\theta)}[\log p(\mathcal{D} \mid \theta)] - \text{KL}(q_\phi(\theta) \| p(\theta))}
$$

첫 항은 **자료에 맞음**(기댓값 로그 가능도)을 재고, 둘째 항은 **복잡함에 대한 벌**(앞확률에서 벌어진 정도)이다.

---

## 평균장 가우스 변분 집안

표준 고름은 가중값마다 따로 가우스로 매개변수화하는 것이다:

$$
q_\phi(\theta) = \prod_{j=1}^D \mathcal{N}(\theta_j \mid \mu_j, \sigma_j^2)
$$

변분 매개변수는 $\phi = \{\mu_j, \rho_j\}_{j=1}^D$이며 여기서 $\sigma_j = \log(1 + \exp(\rho_j))$이다(소프트플러스가 양수임을 보장한다).

---

## 매개변수 바꾸기 재주

확률로 흔들리는 표집을 거쳐 되짚으려면 매개변수를 바꾼다:

$$
\theta = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

이러면 확률 요소가 붙박인 분포 $\epsilon$으로 옮겨 가 목표를 $\mu$과 $\rho$에 대해 미분할 수 있게 된다:

$$
\nabla_\phi \mathcal{L} = \nabla_\phi \mathbb{E}_{\epsilon}[\log p(\mathcal{D} \mid \mu + \sigma \odot \epsilon) - \log q_\phi(\mu + \sigma \odot \epsilon) + \log p(\mu + \sigma \odot \epsilon)]
$$

---

## 알고리즘

```
Algorithm: Bayes by Backprop
─────────────────────────────
Input: Dataset D, prior p(θ), learning rate η
Initialize: μ, ρ (variational parameters)

For each epoch:
    For each minibatch (x, y) of size m:
        1. Sample ε ~ N(0, I)
        2. Compute θ = μ + softplus(ρ) ⊙ ε
        3. Compute loss:
           L = (1/m) Σ -log p(y_i | x_i, θ)     [NLL]
             + (β/N) KL(q_φ(θ) || p(θ))          [complexity cost]
        4. Compute gradients ∂L/∂μ, ∂L/∂ρ
        5. Update: μ ← μ - η ∂L/∂μ
                   ρ ← ρ - η ∂L/∂ρ
```

KL 항이 일찍부터 판쳐 뒤확률이 앞확률로 찌부러지는 것을 막으려고 익히는 동안 **KL 무게** $\beta$을 담금질하듯 올릴 수 있다(달굼).

---

## 가우스 앞확률과 뒤확률의 KL 벌어짐

앞확률과 어림 뒤확률이 모두 가우스이면 KL은 닫힌 꼴이 된다:

$$
\text{KL}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, \sigma_p^2)) = \log\frac{\sigma_p}{\sigma} + \frac{\sigma^2 + \mu^2}{2\sigma_p^2} - \frac{1}{2}
$$

가중값이 $D$개인 온전한 망에서는:

$$
\text{KL}(q_\phi \| p) = \sum_{j=1}^D \left[\log\frac{\sigma_p}{\sigma_j} + \frac{\sigma_j^2 + \mu_j^2}{2\sigma_p^2} - \frac{1}{2}\right]
$$

---

## PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BayesLinear(nn.Module):
    """
    배울 수 있는 가중값 분포를 갖는 베이즈 선형 층.
    
    가중값: w ~ N(mu_w, softplus(rho_w)^2)
    치우침:  b ~ N(mu_b, softplus(rho_b)^2)
    """
    
    def __init__(self, in_features: int, out_features: int,
                 prior_sigma: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma
        
        # 가중값의 변분 매개변수
        self.mu_w = nn.Parameter(torch.empty(out_features, in_features))
        self.rho_w = nn.Parameter(torch.empty(out_features, in_features))
        
        # 치우침의 변분 매개변수
        self.mu_b = nn.Parameter(torch.empty(out_features))
        self.rho_b = nn.Parameter(torch.empty(out_features))
        
        self.reset_parameters()
        self.kl = 0.0
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.mu_w, a=math.sqrt(5))
        nn.init.constant_(self.rho_w, -3.0)  # 작은 첫 흩어짐
        nn.init.zeros_(self.mu_b)
        nn.init.constant_(self.rho_b, -3.0)
    
    @property
    def sigma_w(self):
        return F.softplus(self.rho_w)
    
    @property
    def sigma_b(self):
        return F.softplus(self.rho_b)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 매개변수 바꾸기로 가중값 표집
        eps_w = torch.randn_like(self.mu_w)
        eps_b = torch.randn_like(self.mu_b)
        
        w = self.mu_w + self.sigma_w * eps_w
        b = self.mu_b + self.sigma_b * eps_b
        
        # KL 벌어짐 셈하기
        self.kl = self._kl_divergence(
            self.mu_w, self.sigma_w, self.mu_b, self.sigma_b)
        
        return F.linear(x, w, b)
    
    def _kl_divergence(self, mu_w, sigma_w, mu_b, sigma_b):
        """가우스 분포의 닫힌 꼴 KL(q || 앞확률)."""
        sp = self.prior_sigma
        
        kl_w = (torch.log(sp / sigma_w) + 
                (sigma_w**2 + mu_w**2) / (2 * sp**2) - 0.5).sum()
        kl_b = (torch.log(sp / sigma_b) + 
                (sigma_b**2 + mu_b**2) / (2 * sp**2) - 0.5).sum()
        
        return kl_w + kl_b


class BayesianMLP(nn.Module):
    """회귀나 가르기를 위한 베이즈 다층 퍼셉트론."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 prior_sigma: float = 1.0):
        super().__init__()
        self.fc1 = BayesLinear(input_dim, hidden_dim, prior_sigma)
        self.fc2 = BayesLinear(hidden_dim, hidden_dim, prior_sigma)
        self.fc3 = BayesLinear(hidden_dim, output_dim, prior_sigma)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def kl_loss(self):
        """모든 베이즈 층에 걸친 전체 KL 벌어짐."""
        return sum(m.kl for m in self.modules() if isinstance(m, BayesLinear))
    
    def predict(self, x: torch.Tensor, n_samples: int = 50):
        """몬테카를로 미리봄 분포."""
        self.eval()
        preds = torch.stack([self(x) for _ in range(n_samples)])
        return preds.mean(dim=0), preds.var(dim=0)


def train_bnn(model, train_loader, n_epochs=100, lr=1e-3, 
              n_train=None, kl_weight=1.0):
    """
    KL 담금질로 베이즈 신경망 익히기.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(n_epochs):
        # KL 담금질: KL 무게를 차츰 올리기
        beta = min(1.0, (epoch + 1) / (n_epochs * 0.3)) * kl_weight
        
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            output = model(x_batch)
            nll = F.mse_loss(output, y_batch, reduction='sum')
            kl = model.kl_loss()
            
            # ELBO 손실: 음의 로그 가능도에 크기 맞춘 KL 더하기
            loss = nll + beta * kl / (n_train or len(train_loader.dataset))
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    
    return model
```

---

## 어림 베이즈 신경망으로 본 몬테카를로 떨구기

몬테카를로 떨구기(갈과 가라마니, 2016)는 표준 떨구기를 어림 변분 추론으로 다시 풀이하는 더 단순한 길을 준다:

$$
p(y^* \mid x^*, \mathcal{D}) \approx \frac{1}{T} \sum_{t=1}^T p(y^* \mid x^*, \hat{w}_t)
$$

여기서 $\hat{w}_t$은 시험할 때 무작위 떨구기 가리개를 씌운 가중값이다.

```python
class MCDropoutModel(nn.Module):
    """몬테카를로 떨구기로 어림 베이즈 신경망처럼 쓸 수 있는 표준 망."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def mc_predict(self, x, n_samples=100):
        """떨구기를 켠 채로 앞먹임을 T번 돌리기."""
        self.train()  # 떨구기를 켠 채로 둔다
        preds = torch.stack([self(x) for _ in range(n_samples)])
        self.eval()
        return preds.mean(dim=0), preds.var(dim=0)
```

| 살필 점 | 되짚음으로 하는 베이즈 | 몬테카를로 떨구기 |
|--------|-------------------|------------|
| 매개변수 | $2D$(가중값마다 평균과 흩어짐) | $D$(표준 가중값) |
| 익힘 값 | 표준 익힘의 2배쯤 | 표준 익힘 |
| 추론 값 | 앞먹임 $T$번 | 앞먹임 $T$번 |
| 불확실함의 질 | 눈금이 더 잘 맞음 | 낮춰 잡을 수 있음 |
| 구현 | 맞춤 층 | 표준에 시험 때 떨구기를 더함 |

---

## 요약

| 개념 | 핵심 |
|---------|-----------|
| **ELBO 목표** | 자료에 맞음 - KL 복잡함 벌 |
| **매개변수 바꾸기 재주** | 표집을 거쳐 기울기로 최적화할 수 있게 한다 |
| **평균장 가우스** | 가중값마다 독립인 가우스, 매개변수 $2D$개 |
| **KL 담금질** | 뒤확률이 찌부러지지 않도록 KL 무게를 서서히 올린다 |
| **몬테카를로 떨구기** | 더 단순한 길이다. 떨구기를 변분 추론으로 다시 풀이한다 |

---

## 참고 문헌

- Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015). Weight Uncertainty in Neural Networks. *ICML*.
- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation. *ICML*.
- Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. *ICLR*.
- Graves, A. (2011). Practical Variational Inference for Neural Networks. *NeurIPS*.

## 연습문제

**연습문제 1.**
베이즈 신경망이 앎 불확실함을 어떻게 재는지 설명하여라. 이는 우연 불확실함과 어떻게 다른가?

??? success "연습문제 1 풀이"
    베이즈 신경망은 점 어림값 하나가 아니라 가중값에 대한 뒤확률 분포 $p(w | \mathcal{D})$을 지닌다. **앎 불확실함**(모형의 불확실함)은 자료가 모자라서 생기며 뒤확률의 퍼짐으로 담긴다. 곧 자료가 성긴 구역에서는 뒤확률이 넓어 다양한 미리봄이 나온다. **우연 불확실함**(자료의 잡음)은 자료에 본디 있으며 내임 분포 $p(y|x,w)$으로 담긴다. 앎 불확실함은 자료가 늘면 줄지만 우연 불확실함은 줄지 않는다. 베이즈 신경망은 미리봄 흩어짐으로 앎 불확실함을 잰다: $\text{Var}[y|x] = \underbrace{\mathbb{E}_w[\text{Var}[y|x,w]]}_{\text{aleatoric}} + \underbrace{\text{Var}_w[\mathbb{E}[y|x,w]]}_{\text{epistemic}}$.

---

**연습문제 2.**
ELBO 목표에 대한, 되짚음으로 하는 베이즈의 기울기 어림꼴을 이끌어 내어라.

??? success "연습문제 2 풀이"
    ELBO은 $\mathcal{L}(\theta) = \mathbb{E}_{q_\theta(w)}[\log p(\mathcal{D}|w)] - D_{\text{KL}}(q_\theta(w) \| p(w))$이다. $w = \mu + \sigma \odot \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$인 매개변수 바꾸기 재주를 쓰면:

    $$\nabla_\theta \mathcal{L} = \nabla_\theta \left[ \log p(\mathcal{D}|w) + \log p(w) - \log q_\theta(w) \right]_{w = \mu + \sigma \odot \epsilon}$$

    이는 $\epsilon$을 표집하고 정해진 바꿈을 거쳐 기울기를 셈해 어림한다. $q$과 $p$이 가우스이면 KL 항은 흔히 닫힌 꼴로 셈할 수 있다.

---

**연습문제 3.**
베이즈 신경망에서 몬테카를로 떨구기와 드러난 변분 추론을 구현의 복잡함과 눈금 맞추기의 질로 견주어라.

??? success "연습문제 3 풀이"
    **몬테카를로 떨구기**는 시험할 때도 떨구기를 쓰고 앞먹임 $T$번의 평균을 낸다. 구현이 단순하지만(떨구기를 넣고 앞먹임을 여러 번 돌린다) 좁은 변분 집안(베르누이 곱 잡음)에 맞대응되어 눈금이 잘 맞지 않는 불확실함이 나오는 일이 잦다. **드러난 변분 추론**(이를테면 되짚음으로 하는 베이즈)은 매개변수를 곱절로 늘리고(가중값마다 $\mu$과 $\sigma$) 꼼꼼한 최적화가 필요하지만 더 풍성한 뒤확률 어림과 대체로 눈금이 더 잘 맞는 불확실함 어림값을 준다. 불확실함을 빠르게 어림하려면 몬테카를로 떨구기가, 눈금 맞추기가 중요하면 드러난 변분 추론이 낫다.

---

**연습문제 4.**
베이즈 신경망에서 앞확률 $p(w)$을 고르는 일이 왜 중요하며, 표준 가우스 앞확률과 크기 섞음 앞확률 사이의 주고받음은 무엇인가?

??? success "연습문제 4 풀이"
    앞확률은 뒤확률에 벌을 주며 익힘의 움직임과 미리봄의 불확실함에 모두 영향을 준다. **표준 가우스** $\mathcal{N}(0, \sigma^2 I)$은 단순하고 L2 벌주기에 맞대응되지만 모든 가중값에 똑같이 벌을 주어 너무 옭아맬 수 있다. **크기 섞음** 앞확률(이를테면 $\pi \mathcal{N}(0, \sigma_1^2) + (1-\pi) \mathcal{N}(0, \sigma_2^2)$)은 어떤 가중값은 크게(신호) 두면서 다른 가중값은 0에 가깝게(잡음) 몰아 맞춰 가는 성김을 준다. 주고받음은 이렇다. 크기 섞음은 표현력이 더 좋지만 최적화가 더 어렵고 웃매개변수가 늘어난다.
