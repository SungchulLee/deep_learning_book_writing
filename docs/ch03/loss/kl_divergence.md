# KL 발산
쿨백-라이블러 발산은 한 확률분포가 다른 확률분포로부터 얼마나 벗어나는지를 잰다. 변분 오토인코더의 정칙화 항으로, 지식 증류의 학습 목표로, 모델의 거동을 이해하는 이론적 도구로 딥러닝 전반에 등장한다. 이 절은 정의, 핵심 성질, PyTorch 인터페이스를 소개하며, 거리 공리·가우스 계산·피셔 정보는 따로 마련한 하위 페이지에서 다룬다.

## 정의

같은 표본 공간 위의 이산 분포 $p$과 $q$에 대해 다음과 같다.

$$D_{\text{KL}}(p \| q) = \sum_i p_i \log\frac{p_i}{q_i} = \mathbb{E}_{p}\!\left[\log \frac{p_i}{q_i}\right]$$

밀도가 $f$과 $g$인 연속 분포에 대해 다음과 같다.

$$D_{\text{KL}}(f \| g) = \int f(x)\log\frac{f(x)}{g(x)}\,dx = \mathbb{E}_{f}\!\left[\log \frac{f(x)}{g(x)}\right]$$

$D_{\text{KL}}(p \| q)$은 $q$에 최적화된 부호를 써서 $p$의 표본을 부호화할 때 추가로 필요한 비트 수의 기댓값으로 해석할 수 있다. 관례 $0 \log(0/q) = 0$은 연속성에서 따라 나오며, $p_i > 0$인 어떤 $i$에서 $q_i = 0$이면 $D_{\text{KL}}$은 정의되지 않는다($p$의 받침이 $q$의 받침에 포함되어야 한다).

### 직관적인 해석

**추가 비트.** $q$에 최적인 부호를 설계해 놓고 그것으로 $p$의 표본을 부호화한다고 하자. KL 발산은 $p$에 최적인 부호에 비해 추가로 필요한 비트 수의 기댓값이다.

**기대 로그 증거비.** $D_{\text{KL}}(p \| q) = \mathbb{E}_{p}[\log(p(x)/q(x))]$은 표본이 평균적으로 $q$보다 $p$ 아래에서 얼마나 더 그럴듯한지를 잰다.

**정보 이득.** $D_{\text{KL}}(p \| q)$은 사전분포 $q$에서 사후분포 $p$으로 갱신할 때 얻는 정보량을 수치로 나타낸다.

## 음이 아님 (깁스 부등식)

KL 발산은 언제나 음이 아니다. 증명은 오목함수 $\log$에 옌센 부등식을 적용한다.

$$\begin{aligned}
D_{\text{KL}}(f \| g)
&= -\int f(x)\log\frac{g(x)}{f(x)}\,dx \\
&\geq -\log\int f(x)\frac{g(x)}{f(x)}\,dx \\
&= -\log\int g(x)\,dx \\
&= -\log 1 = 0
\end{aligned}$$

등호는 거의 모든 곳에서 $f = g$일 때에만 성립한다. **깁스 부등식**으로 알려진 이 결과는 두 분포가 같을 때 발산이 (0으로) 최소가 됨을 보장한다.

## 비대칭성

KL 발산은 대칭이 **아니다**.

$$D_{\text{KL}}(p \| q) \neq D_{\text{KL}}(q \| p) \quad\text{in general}$$

```python
import numpy as np

np.random.seed(2)

p = np.random.uniform(0., 1., 3)
q = np.random.uniform(0., 1., 3)
p, q = p / p.sum(), q / q.sum()

KL_pq = np.sum(p * np.log(p / q))
KL_qp = np.sum(q * np.log(q / p))
print(f"KL(p||q) = {KL_pq:.6f}")
print(f"KL(q||p) = {KL_qp:.6f}")
print(f"Difference: {abs(KL_pq - KL_qp):.6f}")  # non-zero
```

### 독립인 분포에 대한 가법성

$p(x, y) = p(x)p(y)$이고 $q(x, y) = q(x)q(y)$이면 다음이 성립한다.

$$D_{\text{KL}}(p(x,y) \| q(x,y)) = D_{\text{KL}}(p(x) \| q(x)) + D_{\text{KL}}(p(y) \| q(y))$$

## 교차 엔트로피와의 관계

교차 엔트로피와 KL 발산은 다음 관계로 이어진다.

$$H(P, Q) = H(P) + D_{\text{KL}}(P \| Q)$$

여기서 $H(P) = -\sum_k P(k)\log P(k)$은 $P$의 엔트로피이다. $H(P)$은 $Q$에 대해 상수이므로 교차 엔트로피 $H(P, Q)$을 최소화하는 것은 $D_{\text{KL}}(P \| Q)$을 최소화하는 것과 같다. 분류의 목표로서 교차 엔트로피 손실과 KL 최소화를 바꿔 쓸 수 있는 이유가 여기 있다.

## 순방향 KL과 역방향 KL

이 비대칭성은 실무에서 KL 발산을 어떻게 쓰는지에 결정적인 영향을 준다.

### 순방향 KL: D_KL(p || q)|KL(p || q)

$q$에 대해 순방향 KL을 최소화하면 다음을 얻는다.

$$\min_q D_{\text{KL}}(p \| q) = \min_q \mathbb{E}_p[-\log q(x)] + \text{const}$$

이는 **평균을 좇는**(**최빈값을 덮는**이라고도 한다) 성질이다. $q$이 퍼져서 $p$의 모든 최빈값을 덮으며, $p$에 질량이 있는 최빈값을 놓치면 벌점을 받는다.

**대표적인 쓰임:** 최대가능도 추정(데이터에서 $p$을 알고 $q$을 최적화한다).

### 역방향 KL: D_KL(q || p)|KL(q || p)

$q$에 대해 역방향 KL을 최소화하면 다음을 얻는다.

$$\min_q D_{\text{KL}}(q \| p) = \min_q \mathbb{E}_q[\log q(x) - \log p(x)]$$

이는 **최빈값을 좇는** 성질이다. $q$은 모든 최빈값에 퍼지는 대신 $p$의 최빈값 하나에 집중한다. $p$이 작은 영역에서는 $\log q(x) - \log p(x)$ 벌점이 커지므로 $q$은 그런 곳에 질량을 두지 않으려 한다.

**대표적인 쓰임:** 변분 추론(상수를 제외하고 $\log p$을 계산할 수는 있지만 $p$에서 표본을 뽑을 수는 없을 때).

## 거리 함수가 아니다

"발산"이라 불리기는 하지만 KL 발산은 거리 함수가 아니다. 음이 아님과 구별 불가능자의 동일성은 만족하지만 대칭성과 삼각부등식은 어긴다. 증명과 수치적 반례를 곁들인 자세한 분석은 "KL 발산과 거리 공리"에 있다.

## 국소적 거동: 피셔 정보

$D_{\text{KL}}(f_{\theta_0} \| f_\theta)$을 $\theta = \theta_0$ 주변에서 테일러 전개하면 국소적으로 KL 발산이 이차형식처럼 거동함을 알 수 있다.

$$D_{\text{KL}}(f_{\theta_0} \| f_\theta) \approx \frac{1}{2}(\theta - \theta_0)^T\, \mathbf{I}(\theta_0)\, (\theta - \theta_0)$$

여기서 $\mathbf{I}(\theta_0)$은 **피셔 정보 행렬**이다. 최솟값 근처에서 KL 발산은 마할라노비스 거리처럼 거동하며, 확률분포 공간 위의 자연스러운 리만 계량을 제공한다. 이것이 자연 경사법, 신뢰 영역 최적화(TRPO, PPO), 변분 추론의 바탕이 된다. 전체 유도는 "KL과 피셔 정보"에 있다.

## 정규분포의 KL 발산

### 단변량 공식

$p = \mathcal{N}(\mu_1, \sigma_1^2)$이고 $q = \mathcal{N}(\mu_2, \sigma_2^2)$일 때 다음과 같다.

$$D_{\text{KL}}(p \| q) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

### 일반적인 다변량 공식

$\mathbb{R}^d$에서 $p = \mathcal{N}(\mu_p, \Sigma_p)$이고 $q = \mathcal{N}(\mu_q, \Sigma_q)$일 때 다음과 같다.

$$D_{\text{KL}}(p \| q) = \frac{1}{2}\!\left[\log\frac{|\Sigma_q|}{|\Sigma_p|} - d + \operatorname{tr}\!\left(\Sigma_q^{-1}\Sigma_p\right) + (\mu_q - \mu_p)^T\Sigma_q^{-1}(\mu_q - \mu_p)\right]$$

### VAE의 특수한 경우

부호기 $q_\phi(z|x) = \mathcal{N}(\mu, \text{diag}(\sigma_1^2, \ldots, \sigma_d^2))$과 사전분포 $p(z) = \mathcal{N}(0, I)$에 대해 다음과 같다.

$$D_{\text{KL}}(q \| p) = -\frac{1}{2}\sum_{j=1}^{d}\!\left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

완전 공분산인 경우를 포함한 전체 유도는 "정규분포의 KL"에 있다.

## VAE에서의 응용

변분 오토인코더에서 손실은 복원 오차와 KL 정칙화를 결합한다. ELBO 분해에서 출발한다.

$$\log p(x) = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\text{KL}}(q_\phi(z|x) \| p(z))}_{\mathcal{L}_{\text{ELBO}}} + D_{\text{KL}}(q_\phi(z|x) \| p_\theta(z|x))$$

ELBO를 최대화하는 것은 다음을 최소화하는 것과 같다.

$$\mathcal{L}_{\text{VAE}} = \underbrace{\mathbb{E}_{z \sim q_\phi(z|x)}[-\log p_\theta(x|z)]}_{\text{reconstruction}} + \underbrace{D_{\text{KL}}(q_\phi(z|x) \| p(z))}_{\text{regularization}}$$

### KL과 복원 사이의 절충

$\beta$-VAE는 KL 항에 가중치를 도입한다. $\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \cdot D_{\text{KL}}$이다.

| KL 값 | 의미 | 효과 |
|----------|---------|--------|
| **큼** | 부호기의 출력이 $\mathcal{N}(0, I)$에서 멀다 | 더 많은 정보가 부호화되고 복원이 좋아진다 |
| **작음** | 부호기의 출력이 $\mathcal{N}(0, I)$에 가깝다 | 정보가 덜 부호화되고 잠재 공간이 매끄러워진다 |
| **0** | 모든 입력이 사전분포로 간다 | 정보가 부호화되지 않고 출력이 무작위가 된다 |

## PyTorch 구현

### VAE를 위한 정규분포 KL

부호기는 각 잠재 차원에 대해 $\mu$과 $\log\sigma^2$을 낸다.

```python
import torch

def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor,
                  reduction: str = 'sum') -> torch.Tensor:
    """q = N(mu, diag(exp(logvar)))에서 p = N(0, I)로의 KL 발산.

    식: D_KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    Args:
        mu: q의 평균, 모양 (batch_size, latent_dim).
        logvar: q의 로그 분산, 모양 (batch_size, latent_dim).
        reduction: 'sum', 'mean', 'none' 가운데 하나.

    Returns:
        밝힌 줄이기 방식을 쓴 KL 갈림.
    """
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if reduction == 'sum':
        return kl.sum()
    elif reduction == 'mean':
        return kl.mean()
    else:  # 'none' — sum over latent dims, keep batch
        return kl.sum(dim=1)
```

!!! tip "왜 로그 분산인가?"
    부호기가 $\sigma^2$ 대신 $\log\sigma^2$을 내는 것은 수치적 안정성 때문이다. `logvar`는 어떤 실수든 될 수 있고(신경망의 출력으로 알맞다), `exp(logvar)`는 언제나 양수이며(분산으로 알맞다), `log(sigma^2) = logvar`는 작은 수의 로그를 취하는 일을 피하게 해 준다.

### 전체 VAE 손실

```python
import torch.nn.functional as F

def vae_loss(recon_x: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 1.0) -> tuple:
    """완전한 VAE 손실: 복원 + beta * KL.

    Args:
        recon_x: 되살린 데이터, 모양 (batch_size, data_dim).
        x: 본디 데이터, 모양 (batch_size, data_dim).
        mu: 인코더 평균, 모양 (batch_size, latent_dim).
        logvar: 인코더 로그 분산, 모양 (batch_size, latent_dim).
        beta: KL 가중치(베타 VAE).

    Returns:
        (total_loss, recon_loss, kl_loss) 튜플.
    """
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss
```

### 실무적인 기법

**자유 비트.** 잠재 차원마다 KL의 하한을 강제하여 사후분포 붕괴를 막는다.

```python
def kl_free_bits(mu, logvar, free_bits=0.1):
    """자유 비트 제약이 있는 KL."""
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    return kl_per_dim.sum(dim=1)
```

**KL 담금질.** 학습 중에 KL 가중치를 점차 키운다.

```python
def get_beta(epoch, warmup_epochs=10, max_beta=1.0):
    """선형 KL 담금질."""
    if epoch < warmup_epochs:
        return max_beta * epoch / warmup_epochs
    return max_beta
```

### 일반적인 분포를 위한 `nn.KLDivLoss`

정규분포가 아닌 분포(예: 지식 증류)에는 PyTorch가 `nn.KLDivLoss`을 제공한다. 이 함수는 입력으로 **로그 확률**을, 목표로 **확률**을 받는다.

```python
import torch.nn as nn

kl_criterion = nn.KLDivLoss(reduction='batchmean')

# input: 학생 모델이 낸 로그 확률
log_probs = F.log_softmax(logits, dim=1)

# target: 교사 모델이 낸 확률 분포
target_probs = F.softmax(teacher_logits, dim=1)

loss = kl_criterion(log_probs, target_probs)
```

!!! warning "입력 관례"
    `nn.KLDivLoss`은 **입력**을 로그 공간으로, **목표**를 확률 공간으로 받는다. 이 관례는 많은 사용자가 예상하는 것과 반대이다. 둘을 바꿔 넣으면 아무 경고 없이 잘못된 결과가 나온다.

`reduction` 매개변수가 집계 방식을 정한다. `'batchmean'`(권장하며 표본당 참 KL을 준다), `'sum'`(날것의 합), `'mean'`(전체 원소 수로 나누므로 참 KL이 **아니다**), `'none'`(원소별)이 있다.

### 해석적 KL과 몬테카를로 KL

부호기와 사전분포가 모두 정규분포이면 해석적 공식이 정확하고 분산도 없다. 사후분포가 정규분포가 아니거나 사전분포가 복잡하면 KL을 몬테카를로로 추정해야 한다.

```python
def kl_monte_carlo(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """z ~ q에서 뽑은 표본으로 D_KL(q||p)을 추정한다.

    D_KL = E_q[log q - log p] ≈ (1/N) sum_i (log q(z_i) - log p(z_i))
    """
    return (log_q - log_p).mean()
```

## 핵심 정리

KL 발산은 두 분포 사이의 기대 로그가능도비로, $p$을 $q$으로 근사할 때 잃는 정보량을 잰다. 음이 아니고(깁스 부등식), 비대칭이며, 거리 함수가 아니다. 순방향 KL($D_{\text{KL}}(p \| q)$)은 최빈값을 덮는 성질을 가지며 최대가능도의 바탕이 되고, 역방향 KL($D_{\text{KL}}(q \| p)$)은 최빈값을 좇는 성질을 가지며 변분 추론의 바탕이 된다. 정규분포에서는 닫힌 형태의 식 덕분에 VAE에서 효율적으로 계산할 수 있다. PyTorch는 일반적인 이산 분포를 위해 (입력을 로그 공간으로 받는 관례의) `nn.KLDivLoss`을 제공하며, VAE의 KL 항은 해석적 공식으로 직접 구현한다.

## 연습문제

**연습문제 1.**
KL 발산이 음이 아님을 증명하라. 즉 $D_{\text{KL}}(p\|q) \geq 0$이며 등호는 $p = q$일 때에만 성립함을 보여라.

??? success "연습문제 1 풀이"
    볼록함수 $-\log$에 옌센 부등식을 적용하면 다음을 얻는다.

    $$
    D_{\text{KL}}(p\|q) = \mathbb{E}_p\left[-\log\frac{q}{p}\right] \geq -\log\mathbb{E}_p\left[\frac{q}{p}\right] = -\log\sum_x p(x)\frac{q(x)}{p(x)} = -\log 1 = 0
    $$

    옌센 부등식에서 등호는 $q/p$이 상수일 때, 즉 $p = q$일 때에만 성립한다. $\square$

---

**연습문제 2.**
$p = \text{Bernoulli}(0.3)$과 $q = \text{Bernoulli}(0.7)$에 대해 $D_{\text{KL}}(p\|q)$과 $D_{\text{KL}}(q\|p)$을 계산하여 KL 발산이 대칭이 아님을 보여라.

??? success "연습문제 2 풀이"
    $D_{\text{KL}}(p\|q) = 0.3\log\frac{0.3}{0.7} + 0.7\log\frac{0.7}{0.3} = 0.3(-0.847) + 0.7(0.847) = 0.339$ nats.

    $D_{\text{KL}}(q\|p) = 0.7\log\frac{0.7}{0.3} + 0.3\log\frac{0.3}{0.7} = 0.7(0.847) + 0.3(-0.847) = 0.339$ nats.

    이 특수한 경우에는 $p \leftrightarrow 1-p$을 맞바꾸는 대칭성 덕분에 두 값이 같지만, 일반적으로 그렇지는 않다.

---

**연습문제 3.**
두 단변량 정규분포 $p = \mathcal{N}(\mu_1, \sigma_1^2)$과 $q = \mathcal{N}(\mu_2, \sigma_2^2)$ 사이의 KL 발산을 유도하라.

??? success "연습문제 3 풀이"
    $$
    D_{\text{KL}}(p\|q) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}
    $$

    $q = \mathcal{N}(0, 1)$이면 $D_{\text{KL}} = -\frac{1}{2}(1 + \log\sigma_1^2 - \mu_1^2 - \sigma_1^2)$이며, 이것이 VAE 학습에서의 KL 항이다.

---

**연습문제 4.**
변분 추론에서 순방향 KL($D_{\text{KL}}(p\|q)$)과 역방향 KL($D_{\text{KL}}(q\|p)$)의 차이를 설명하라. 어느 쪽이 최빈값을 좇고 어느 쪽이 평균을 좇는가?

??? success "연습문제 4 풀이"
    순방향 KL $D_{\text{KL}}(p\|q)$은 **평균을 좇는다**(0을 피한다). $q \approx 0$일 때 $\log(p/q)$에서 오는 무한한 벌점을 피하려면 $q$이 $p > 0$인 모든 영역을 덮어야 한다. 그래서 $q$이 퍼져서 $p$의 모든 최빈값을 덮게 된다.

    역방향 KL $D_{\text{KL}}(q\|p)$은 **최빈값을 좇는다**(0을 강제한다). $p > 0$인 곳에서도 $q$이 0이어도 (벌점이 유한하므로) 안전하며, 대신 질량이 최빈값 하나에 집중된다. 표준적인 변분 추론은 역방향 KL을 최소화하는데, 그래서 불확실성을 과소평가하는 경향이 있다.
