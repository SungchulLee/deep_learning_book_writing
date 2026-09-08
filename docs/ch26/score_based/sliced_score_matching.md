# 저민 점수 맞추기

**저민 점수 맞추기(SSM)**는 아무 쏘기를 써서 온전한 야코비 셈을 피하는, 드러난 점수 맞추기의 대안이다. 잡음 없애는 점수 맞추기의 잡음 흔들기를 받아들일 수 없을 때 점수 신경망을 익히는, 키울 수 있고 **치우치지 않은** 길을 준다.

!!! info "핵심 생각"
    비싼 온전 야코비 대각합(뒤먹임 $D$번)을 셈하는 대신 모델 점수와 자료 점수를 아무 방향에 쏘고 허친슨 대각합 어림개로 이 1차원 쏘기를 맞춘다.

---

## 1. 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

1. 드러난 점수 맞추기의 셈 병목을 이해한다
2. 허친슨 대각합 어림개로 저민 점수 맞추기를 이끌어 낸다
3. 벡터-야코비 곱으로 저민 점수 맞추기를 효율 좋게 짠다
4. 저민 점수 맞추기를 드러난 점수 맞추기, 잡음 없애는 점수 맞추기와 견주어 언제 무엇을 쓸지 안다
5. 알맞은 쏘기 분포와 쏘기 횟수를 고른다

---

## 2. 미리 알아야 할 것

- 드러난 점수 맞추기의 목표
- 벡터 미적분(야코비, 대각합)
- PyTorch 자동 미분의 얼개

---

## 3. 대각합 어림 문제

### 1.1 드러난 점수 맞추기는 비싼 야코비 셈이 필요하다

드러난 점수 맞추기의 목표:

$$
\mathcal{L}_{\text{ESM}}(\theta) = \mathbb{E}_{p_{\text{data}}}\left[\frac{1}{2}\|\mathbf{s}_\theta(\mathbf{x})\|^2 + \text{tr}(\nabla_{\mathbf{x}} \mathbf{s}_\theta(\mathbf{x}))\right]
$$

대각합 항은 야코비의 대각 낱개를 셈해야 한다.

$$
\text{tr}(\nabla_{\mathbf{x}} \mathbf{s}_\theta) = \sum_{i=1}^D \frac{\partial s_{\theta,i}}{\partial x_i}
$$

이는 **뒤먹임 $D$번**이 필요해 차원 높은 자료에서는 감당할 수 없이 비싸다!

| 자료 갈래 | 차원 $D$ | 드러난 점수 맞추기의 뒤먹임 횟수 |
|-----------|---------------|---------------------|
| 2차원 장난감 | 2 | 2 |
| 표 자료 | 100 | 100 |
| MNIST | 784 | 784 |
| CIFAR-10 | 3,072 | 3,072 |

### 1.2 풀이: 아무 쏘기

대각 낱개 $D$개를 모두 셈하는 대신 아무 쏘기로 **대각합을 어림한다**. 이것이 저민 점수 맞추기의 핵심 통찰이다.

---

## 4. 허친슨 대각합 어림개

### 2.1 정리

아무 정사각 행렬 $\mathbf{A}$과 $\mathbb{E}[\mathbf{v}\mathbf{v}^\top] = \mathbf{I}$인 아무 벡터 $\mathbf{v}$에 대해:

$$
\boxed{\text{tr}(\mathbf{A}) = \mathbb{E}_{\mathbf{v}}[\mathbf{v}^\top \mathbf{A} \mathbf{v}]}
$$

**증명:**

$$
\mathbb{E}[\mathbf{v}^\top \mathbf{A} \mathbf{v}] = \mathbb{E}\left[\sum_{i,j} v_i A_{ij} v_j\right] = \sum_{i,j} A_{ij} \mathbb{E}[v_i v_j] = \sum_{i,j} A_{ij} \delta_{ij} = \sum_i A_{ii} = \text{tr}(\mathbf{A})
$$

### 2.2 올바른 쏘기 분포

$\mathbb{E}[\mathbf{v}\mathbf{v}^\top] = \mathbf{I}$을 채우는 아무 분포:

| 분포 | 공식 | 흩어짐 | 권함 |
|--------------|---------|----------|----------------|
| **라데마허** | $v_i \in \{-1, +1\}$을 고르게 | 더 작다 | ✅ 낫다 |
| **정규 분포** | $\mathbf{v} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ | 더 크다 | 좋은 기본값 |
| **구면에 고르게** | $\mathbf{v} \sim \text{Uniform}(\mathbb{S}^{D-1})$ | 가장 작다 | 더 복잡하다 |

라데마허 벡터는 흔히 흩어짐이 더 작은 어림을 주고 셈도 더 싸다.

---

## 5. 저민 점수 맞추기의 목표

### 3.1 이끌어 내기

드러난 점수 맞추기의 대각합 항에 허친슨 어림개를 쓰면:

$$
\text{tr}(\nabla_{\mathbf{x}} \mathbf{s}_\theta) \approx \mathbf{v}^\top \nabla_{\mathbf{x}} \mathbf{s}_\theta(\mathbf{x}) \mathbf{v}
$$

드러난 점수 맞추기에 넣으면:

$$
\boxed{\mathcal{L}_{\text{SSM}}(\theta) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} \mathbb{E}_{\mathbf{v}}\left[\frac{1}{2}(\mathbf{v}^\top \mathbf{s}_\theta(\mathbf{x}))^2 + \mathbf{v}^\top \nabla_{\mathbf{x}} \mathbf{s}_\theta(\mathbf{x}) \, \mathbf{v}\right]}
$$

### 3.2 벡터-야코비 곱으로 효율 좋게 셈하기

셈의 핵심 통찰:

$$
\mathbf{v}^\top \nabla_{\mathbf{x}} \mathbf{s}_\theta(\mathbf{x}) \, \mathbf{v} = \mathbf{v}^\top \nabla_{\mathbf{x}}(\mathbf{v}^\top \mathbf{s}_\theta(\mathbf{x}))
$$

오른쪽은 **벡터-야코비 곱(VJP)**이다.

1. 낱값 $\mathbf{v}^\top \mathbf{s}_\theta(\mathbf{x})$을 셈한다
2. $\mathbf{x}$에 대한 기울기를 구한다
3. $\mathbf{v}$과 점곱한다

이는 쏘기 벡터마다 **뒤먹임 한 번**이면 된다!

### 3.3 여러 번 쏘기

쏘기 벡터 $M$개를 쓰면 흩어짐이 줄어든다.

$$
\mathcal{L}_{\text{SSM}}(\theta) \approx \frac{1}{M} \sum_{m=1}^M \left[\frac{1}{2}(\mathbf{v}_m^\top \mathbf{s}_\theta)^2 + \mathbf{v}_m^\top \nabla_{\mathbf{x}} \mathbf{s}_\theta \, \mathbf{v}_m\right]
$$

| 쏘기 횟수 $M$ | 흩어짐 | 뒤먹임 횟수 | 흔한 쓰임 |
|-----------------|----------|-----------------|-------------|
| 1 | 크다 | 1 | 빠른 익히기 |
| 2-4 | 보통 | 2-4 | 좋은 균형 |
| 8-16 | 작다 | 8-16 | 높은 정확도 |

---

## 6. PyTorch 짜기

### 4.1 기본 저민 점수 맞추기 손실

```python
import torch
import torch.nn as nn
from typing import Literal

def sample_rademacher(shape: tuple, device: torch.device) -> torch.Tensor:
    """라데마허 아무 벡터를 뽑는다(+1이나 -1이 같은 확률)."""
    return torch.randint(0, 2, shape, device=device).float() * 2 - 1

def ssm_loss(
    score_model: nn.Module,
    samples: torch.Tensor,
    n_projections: int = 1,
    projection_type: Literal['rademacher', 'gaussian'] = 'rademacher'
) -> torch.Tensor:
    """
    저민 점수 맞추기 손실.
    
    L_SSM = E_x,v [(v·s(x))²/2 + v·∇_x(v·s(x))]
    
    인수:
        score_model: 점수 신경망 s_θ: R^D → R^D
        samples: 자료 표본, 꼴 (N, D)
        n_projections: 아무 쏘기 횟수 M
        projection_type: 'rademacher'(권함) 또는 'gaussian'
    
    반환값:
        낱값 손실
    """
    samples = samples.requires_grad_(True)
    N, D = samples.shape
    device = samples.device
    
    # 점수를 한 번 셈한다
    scores = score_model(samples)  # (N, D)
    
    total_loss = 0.0
    
    for _ in range(n_projections):
        # 아무 쏘기 벡터를 뽑는다
        if projection_type == 'rademacher':
            v = sample_rademacher((N, D), device)
        else:
            v = torch.randn(N, D, device=device)
        
        # 항 1: (v·s(x))² / 2
        score_proj = torch.sum(v * scores, dim=1)  # (N,)
        squared_term = 0.5 * score_proj ** 2
        
        # 항 2: 벡터-야코비 곱으로 구한 v·∇_x(v·s(x))
        # 낱값 (v·s)의 x에 대한 기울기가 v·∇s을 준다
        vjp = torch.autograd.grad(
            outputs=score_proj.sum(),
            inputs=samples,
            create_graph=True,
            retain_graph=True
        )[0]  # (N, D)
        
        # v·(v·∇s) = v·∇_x(v·s)
        trace_term = torch.sum(vjp * v, dim=1)  # (N,)
        
        total_loss += torch.mean(squared_term + trace_term)
    
    return total_loss / n_projections
```

### 4.2 기억을 아끼는 판

```python
def ssm_loss_memory_efficient(
    score_model: nn.Module,
    samples: torch.Tensor,
    n_projections: int = 1
) -> torch.Tensor:
    """
    쏘기마다 점수를 다시 셈해 기억을 아끼는 저민 점수 맞추기.
    GPU 기억이 모자랄 때 쓴다.
    """
    samples = samples.requires_grad_(True)
    N, D = samples.shape
    device = samples.device
    
    total_loss = 0.0
    
    for _ in range(n_projections):
        # 점수를 다시 셈한다(기억 대신 셈을 쓴다)
        scores = score_model(samples)
        
        # 라데마허 쏘기
        v = sample_rademacher((N, D), device)
        
        # 쏜 점수
        score_proj = torch.sum(v * scores, dim=1)
        
        # 저민 점수 맞추기 항
        squared_term = 0.5 * score_proj ** 2
        
        vjp = torch.autograd.grad(
            score_proj.sum(), samples,
            create_graph=True
        )[0]
        trace_term = torch.sum(vjp * v, dim=1)
        
        total_loss += torch.mean(squared_term + trace_term)
        
        # 중간 텐서를 비운다
        del scores, score_proj, vjp
    
    return total_loss / n_projections
```

### 4.3 저민 점수 맞추기 익히개 갈래

```python
class SlicedScoreMatchingTrainer:
    """저민 점수 맞추기의 온전한 익히개."""
    
    def __init__(
        self,
        score_net: nn.Module,
        lr: float = 1e-3,
        n_projections: int = 1,
        projection_type: str = 'rademacher'
    ):
        self.score_net = score_net
        self.optimizer = torch.optim.Adam(score_net.parameters(), lr=lr)
        self.n_projections = n_projections
        self.projection_type = projection_type
        
    def train_step(self, x: torch.Tensor) -> dict:
        """익히기 걸음 하나."""
        self.score_net.train()
        self.optimizer.zero_grad()
        
        loss = ssm_loss(
            self.score_net, x, 
            self.n_projections, 
            self.projection_type
        )
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}
    
    def train(
        self, 
        data: torch.Tensor, 
        n_epochs: int = 1000,
        batch_size: int = 256
    ) -> list:
        """온전한 익히기 되풀이."""
        losses = []
        N = len(data)
        
        for epoch in range(n_epochs):
            idx = torch.randperm(N)[:batch_size]
            batch = data[idx]
            
            metrics = self.train_step(batch)
            losses.append(metrics['loss'])
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}: Loss = {metrics['loss']:.6f}")
        
        return losses
```

---

## 7. 다른 방법과 견주기

### 5.1 셈 견주기

| 방법 | 야코비 비용 | 뒤먹임 횟수 | 아무 벡터 |
|--------|---------------|-----------------|----------------|
| **드러난 점수 맞추기** | 온전한 대각 | $O(D)$ | 아니다 |
| **저민 점수 맞추기** | 쏜 것 | $O(M)$ | 그렇다(벡터 $M$개) |
| **잡음 없애는 점수 맞추기** | 없음 | $O(1)$ | 아니다 |

### 5.2 통계의 성질

| 갈래 | 드러난 점수 맞추기 | 저민 점수 맞추기 | 잡음 없애는 점수 맞추기 |
|--------|-----|-----|-----|
| **치우침** | 없음 | 없음 | $O(\sigma^2)$ |
| **흩어짐** | 해당 없음 | 더 크다($M$에 달렸다) | 더 작다 |
| **한결같음** | 그렇다 | 그렇다 | 그렇다($\sigma \to 0$일 때) |

### 5.3 언제 어느 방법을 쓸까

| 장면 | 권하는 방법 |
|----------|-------------------|
| 차원 낮음($D < 10$) | 드러난 점수 맞추기(정확) |
| 차원 보통($D \sim 10$-$100$) | 저민 점수 맞추기나 잡음 없애는 점수 맞추기 |
| 차원 높음(그림) | 잡음 없애는 점수 맞추기 ✅ |
| 잡음 흔들기를 받아들일 수 없음 | 저민 점수 맞추기 |
| 정확한 점수 맞추기 보장이 필요함 | 저민 점수 맞추기 |
| 익히기 효율이 결정적임 | 잡음 없애는 점수 맞추기 |
| 두루 쓰기 | 잡음 없애는 점수 맞추기 ✅ |

!!! tip "실무적인 권고"
    **거의 모든 쓰임새에는 잡음 없애는 점수 맞추기를 쓰라.** 저민 점수 맞추기는 주로 자료에 잡음을 더할 수 없거나(예컨대 띄엄띄엄한 자료, 어떤 물리 쓰임새) 치우치지 않은 어림이 필요할 때 쓸모 있다.

---

## 8. 흩어짐 줄이기 재주

### 6.1 더 많이 쏘기

$M$을 늘리면 흩어짐이 $1/M$에 비례해 줄어든다.

$$
\text{Var}[\hat{\mathcal{L}}_{\text{SSM}}] \propto \frac{1}{M}
$$

### 6.2 맞짝 뽑기

흩어짐을 줄이려 짝지은 쏘기 $(\mathbf{v}, -\mathbf{v})$을 쓴다.

```python
def ssm_loss_antithetic(score_model, samples, n_projections=1):
    """흩어짐을 줄이려 맞짝 뽑기를 쓴 저민 점수 맞추기."""
    samples = samples.requires_grad_(True)
    scores = score_model(samples)
    
    total_loss = 0.0
    
    for _ in range(n_projections):
        v = sample_rademacher(samples.shape, samples.device)
        
        # 앞 쏘기
        score_proj_pos = torch.sum(v * scores, dim=1)
        vjp_pos = torch.autograd.grad(score_proj_pos.sum(), samples, 
                                       create_graph=True, retain_graph=True)[0]
        loss_pos = 0.5 * score_proj_pos**2 + torch.sum(vjp_pos * v, dim=1)
        
        # 맞짝 쏘기(-v)
        score_proj_neg = torch.sum(-v * scores, dim=1)
        vjp_neg = torch.autograd.grad(score_proj_neg.sum(), samples,
                                       create_graph=True, retain_graph=True)[0]
        loss_neg = 0.5 * score_proj_neg**2 + torch.sum(vjp_neg * (-v), dim=1)
        
        # 평균이 흩어짐을 줄인다
        total_loss += torch.mean((loss_pos + loss_neg) / 2)
    
    return total_loss / n_projections
```

### 6.3 다스림 변량

흩어짐을 아주 작게 하려면 아는 점수 함수에 바탕한 다스림 변량을 쓴다(앞선 재주).

---

## 9. 보기: 2차원 자료로 익히기

```python
import matplotlib.pyplot as plt
import numpy as np

# 정규 분포 섞기를 만든다
def sample_mog(n, n_components=4):
    """정규 분포 섞기에서 뽑는다."""
    angles = np.linspace(0, 2*np.pi, n_components, endpoint=False)
    centers = 2.0 * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    
    idx = np.random.randint(0, n_components, n)
    samples = centers[idx] + 0.3 * np.random.randn(n, 2)
    return torch.tensor(samples, dtype=torch.float32)

# 단순한 점수 신경망
class ScoreNet(nn.Module):
    def __init__(self, dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim)
        )
    
    def forward(self, x):
        return self.net(x)

# 저민 점수 맞추기로 익힌다
data = sample_mog(5000)
model = ScoreNet()
trainer = SlicedScoreMatchingTrainer(model, lr=1e-3, n_projections=4)
losses = trainer.train(data, n_epochs=2000, batch_size=256)

# 배운 점수 마당을 그려 본다
def plot_scores(model, data, title="Learned Score Field"):
    x = torch.linspace(-4, 4, 20)
    y = torch.linspace(-4, 4, 20)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    grid = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    with torch.no_grad():
        scores = model(grid)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.3, s=1, label='Data')
    plt.quiver(X.numpy(), Y.numpy(), 
               scores[:, 0].reshape(20, 20).numpy(),
               scores[:, 1].reshape(20, 20).numpy(),
               alpha=0.7)
    plt.title(title)
    plt.axis('equal')
    plt.legend()

plot_scores(model, data)
```

---

## 10. 간추리기

| 항목 | 설명 |
|--------|-------------|
| **목표** | $\mathcal{L}_{\text{SSM}} = \mathbb{E}_{\mathbf{x}, \mathbf{v}}[\frac{1}{2}(\mathbf{v}^\top \mathbf{s}_\theta)^2 + \mathbf{v}^\top \nabla_{\mathbf{x}} \mathbf{s}_\theta \, \mathbf{v}]$ |
| **핵심 재주** | 허친슨 대각합 어림개: $\text{tr}(\mathbf{A}) = \mathbb{E}[\mathbf{v}^\top \mathbf{A} \mathbf{v}]$ |
| **셈** | 벡터-야코비 곱이 뒤먹임 한 번에 $\mathbf{v}^\top \nabla \mathbf{s} \, \mathbf{v}$을 준다 |
| **복잡도** | $O(D)$ 대신 $O(M)$번 뒤먹임 |
| **치우침** | 치우치지 않음(잡음 없애는 점수 맞추기와 달리) |
| **알맞은 곳** | 잡음 흔들기가 문제일 때 |

!!! tip "핵심 간추리기"

    1. **저민 점수 맞추기는 아무 쏘기를 써서** 야코비 대각합을 효율 좋게 어림한다
    2. **허친슨 어림개**가 $O(D)$번 뒤먹임을 $O(M)$번으로 바꾼다
    3. **벡터-야코비 곱 재주**가 뒤먹임 한 번에 $\mathbf{v}^\top \nabla \mathbf{s} \, \mathbf{v}$을 셈한다
    4. **라데마허 벡터**가 흔히 정규 분포보다 흩어짐이 작다
    5. **거의 모든 경우에 잡음 없애는 점수 맞추기를 쓰라.** 잡음 흔들기를 받아들일 수 없을 때 저민 점수 맞추기를 쓴다

---

## 연습문제

1. **쏘기 견주기**: 2차원 정규 분포 섞기에서 라데마허 쏘기와 정규 분포 쏘기를 견주어라. 손실 어림의 흩어짐을 재라.

2. **쏘기 횟수**: $M \in \{1, 2, 4, 8, 16\}$에 따른 마지막 손실을 그려라. 셈과 정확도의 알맞은 자리는 어디인가?

3. **저민 점수 맞추기와 잡음 없애는 점수 맞추기**: 같은 2차원 자료로 두 방법을 익혀라. 다음을 견주어라.
   - 익히기 곡선
   - 마지막 점수 마당의 품질
   - 바퀴마다 익히기 시간

4. **맞짝 뽑기**: 맞짝 뽑기를 쓸 때와 쓰지 않을 때의 흩어짐을 짜서 견주어라.

5. **차원 높은 곳에서 키우기**: $D \in \{10, 50, 100, 500\}$인 자료에서 저민 점수 맞추기를 시험하라. 어느 차원부터 쓸 수 없게 되는가?

---

## 정리하며

이 마당은 학습 목표、미리 알아야 할 것、대각합 어림 문제、허친슨 대각합 어림개을 차례로 짚었다.

**참고 문헌**

1. Song, Y., et al. (2019). "Sliced Score Matching: A Scalable Approach to Density and Score Estimation." *UAI*.
2. Hutchinson, M. F. (1989). "A Stochastic Estimator of the Trace of the Influence Matrix for Laplacian Smoothing Splines." *Communications in Statistics*.
3. Song, Y., & Ermon, S. (2019). "Generative Modeling by Estimating Gradients of the Data Distribution." *NeurIPS*.
