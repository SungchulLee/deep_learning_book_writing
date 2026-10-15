# KL 발산
쿨백–라이블러(Kullback–Leibler, KL) 발산은 목표 분포와 근사 분포의 차이를 기대 로그밀도비로 나타낸다. 대칭인 거리는 아니며, 어느 분포를 앞에 두는지가 중요하다. 이 절에서는 기본 성질, 정규분포의 공식, VAE와 PyTorch에서의 사용법을 연결한다.

---

## 1. 정의

같은 표본 공간 위의 이산 분포 $p$과 $q$에 대해 다음과 같다.

$$D_{\text{KL}}(p \| q) = \sum_i p_i \log\frac{p_i}{q_i} = \mathbb{E}_{p}\!\left[\log \frac{p_i}{q_i}\right]$$

밀도가 $f$과 $g$인 연속 분포에 대해 다음과 같다.

$$D_{\text{KL}}(f \| g) = \int f(x)\log\frac{f(x)}{g(x)}\,dx = \mathbb{E}_{f}\!\left[\log \frac{f(x)}{g(x)}\right]$$

자연로그를 쓰면 단위는 나트이며, 밑이 2이면 비트이다. $p_i=0$인 항은 0으로 정의한다. $p_i>0$인데 $q_i=0$인 항이 있으면 발산은 **양의 무한대**이다. 유한한 KL을 얻으려면 적어도 $p$가 질량을 두는 곳에 $q$도 양의 질량을 두어야 하며, 연속분포에서는 절대연속성과 적분 가능성도 확인한다.

### 직관적인 해석

**추가 부호 길이.** 분포를 알고 이상적인 부호 길이를 쓸 때, 목표 분포 대신 근사 분포로 부호화하며 늘어나는 평균 길이이다. 자연로그의 값은 비트가 아니라 나트로 읽는다.

**기대 로그 증거비.** $D_{\text{KL}}(p \| q) = \mathbb{E}_{p}[\log(p(x)/q(x))]$은 표본이 평균적으로 $q$보다 $p$ 아래에서 얼마나 더 그럴듯한지를 잰다.

**정보 이득.** $D_{\text{KL}}(p \| q)$은 사전분포 $q$에서 사후분포 $p$으로 갱신할 때 얻는 정보량을 수치로 나타낸다.

---

## 2. 음이 아님 (깁스 부등식)

KL 발산은 언제나 음이 아니다. 증명은 오목함수 $\log$에 옌센 부등식을 적용한다.

$$\begin{aligned}
D_{\text{KL}}(f \| g)
&= -\int f(x)\log\frac{g(x)}{f(x)}\,dx \\
&\geq -\log\int f(x)\frac{g(x)}{f(x)}\,dx \\
&= -\log\int_{f(x)>0} g(x)\,dx \\
&\geq -\log 1 = 0
\end{aligned}$$

등호는 거의 모든 곳에서 $f = g$일 때에만 성립한다. **깁스 부등식**으로 알려진 이 결과는 두 분포가 같을 때 발산이 (0으로) 최소가 됨을 보장한다.

---

## 3. 비대칭성

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

**출력:**

```
KL(p||q) = 0.365037
KL(q||p) = 0.685929
Difference: 0.320892
```

### 독립인 분포에 대한 가법성

$p(x, y) = p(x)p(y)$이고 $q(x, y) = q(x)q(y)$이면 다음이 성립한다.

$$D_{\text{KL}}(p(x,y) \| q(x,y)) = D_{\text{KL}}(p(x) \| q(x)) + D_{\text{KL}}(p(y) \| q(y))$$

---

## 4. 교차 엔트로피와의 관계

교차 엔트로피와 KL 발산은 다음 관계로 이어진다.

$$H(P, Q) = H(P) + D_{\text{KL}}(P \| Q)$$

여기서 $H(P) = -\sum_k P(k)\log P(k)$은 $P$의 엔트로피이다. $H(P)$은 $Q$에 대해 상수이므로 교차 엔트로피 $H(P, Q)$을 최소화하는 것은 $D_{\text{KL}}(P \| Q)$을 최소화하는 것과 같다. 분류의 목표로서 교차 엔트로피 손실과 KL 최소화를 바꿔 쓸 수 있는 이유가 여기 있다.

---

## 5. 순방향 KL과 역방향 KL

이 비대칭성은 실무에서 KL 발산을 어떻게 쓰는지에 결정적인 영향을 준다.

### 순방향 KL

$q$에 대해 순방향 KL을 최소화하면 다음을 얻는다.

$$\min_q D_{\text{KL}}(p \| q) = \min_q \mathbb{E}_p[-\log q(x)] + \text{const}$$

표현력이 제한된 $q$로 여러 모드를 가진 $p$를 근사할 때, 순방향 KL은 $p$가 질량을 둔 영역을 빠뜨리는 데 큰 비용을 준다. 그래서 **모드 포괄(mode covering)** 경향이라고 부른다. 이는 특정 근사 분포족에서 나타나는 경향이며 항상 분산이 커진다는 정리는 아니다. 제한 없이 최적화할 수 있다면 최적해는 $q=p$이다.

**대표적인 쓰임:** 최대가능도 추정. 실제 데이터 분포를 정확히 아는 대신, 관측 표본의 평균 음의 로그우도로 기대값을 근사한다.

### 역방향 KL

$q$에 대해 역방향 KL을 최소화하면 다음을 얻는다.

$$\min_q D_{\text{KL}}(q \| p) = \min_q \mathbb{E}_q[\log q(x) - \log p(x)]$$

역방향 KL은 $q$가 질량을 두는 영역에서 $p$가 작을 때 큰 비용을 준다. 단봉 근사로 다봉 분포를 표현하는 경우 한 모드에 집중하는 **모드 선택(mode seeking)** 경향이 나타날 수 있다. 어느 모드를 선택하는지, 여러 모드를 덮을 수 있는지는 근사 분포족과 최적화에 달려 있다.

**대표적인 쓰임:** 변분 추론. 정규화 상수를 모르는 사후밀도를, 샘플링과 밀도 계산이 가능한 근사 분포로 표현할 때 활용한다. 목표 분포에서 직접 샘플을 전혀 얻을 수 없다는 조건이 반드시 필요한 것은 아니다.

---

## 6. 거리 함수가 아니다

"발산"이라 불리기는 하지만 KL 발산은 거리 함수가 아니다. 음이 아님과 구별 불가능자의 동일성은 만족하지만 대칭성과 삼각부등식은 어긴다. 증명과 수치적 반례를 곁들인 자세한 분석은 "KL 발산과 거리 공리"에 있다.

---

## 7. 국소적 거동: 피셔 정보

$D_{\text{KL}}(f_{\theta_0} \| f_\theta)$을 $\theta = \theta_0$ 주변에서 테일러 전개하면 국소적으로 KL 발산이 이차형식처럼 거동함을 알 수 있다.

$$D_{\text{KL}}(f_{\theta_0} \| f_\theta) \approx \frac{1}{2}(\theta - \theta_0)^T\, \mathbf{I}(\theta_0)\, (\theta - \theta_0)$$

모형이 충분히 매끄럽고 지지집합이 매개변수에 따라 바뀌지 않는 등의 정규성 조건 아래, $\mathbf{I}(\theta_0)$은 피셔 정보 행렬이다. 양의 정부호이면 이 이차형식이 국소 계량을 정의한다. 매개변수가 식별되지 않으면 행렬이 특이할 수 있으므로 역행렬을 바로 사용할 수 없다. 이 관점은 자연 경사와 KL 제약을 쓰는 최적화의 바탕이 된다.

---

## 8. 정규분포의 KL 발산

### 단변량 공식

$p = \mathcal{N}(\mu_1, \sigma_1^2)$이고 $q = \mathcal{N}(\mu_2, \sigma_2^2)$일 때 다음과 같다.

$$D_{\text{KL}}(p \| q) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

다음 정규분포 공식에서는 분산이 양수라고 가정한다.

### 일반적인 다변량 공식

$\mathbb{R}^d$에서 $p = \mathcal{N}(\mu_p, \Sigma_p)$이고 $q = \mathcal{N}(\mu_q, \Sigma_q)$일 때 다음과 같다.

$$D_{\text{KL}}(p \| q) = \frac{1}{2}\!\left[\log\frac{|\Sigma_q|}{|\Sigma_p|} - d + \operatorname{tr}\!\left(\Sigma_q^{-1}\Sigma_p\right) + (\mu_q - \mu_p)^T\Sigma_q^{-1}(\mu_q - \mu_p)\right]$$

다변량 공식의 공분산 행렬은 양의 정부호라고 가정한다. 특이한 정규분포에는 이 역행렬·행렬식 공식을 그대로 적용할 수 없다.

### VAE의 특수한 경우

인코더 $q_\phi(z|x) = \mathcal{N}(\mu, \text{diag}(\sigma_1^2, \ldots, \sigma_d^2))$과 사전분포 $p(z) = \mathcal{N}(0, I)$에 대해 다음과 같다.

$$D_{\text{KL}}(q \| p) = -\frac{1}{2}\sum_{j=1}^{d}\!\left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

완전 공분산인 경우를 포함한 전체 유도는 "정규분포의 KL"에 있다.

---

## 9. VAE에서의 응용

변분 오토인코더에서 손실은 복원 오차와 KL 정규화를 결합한다. ELBO 분해에서 출발한다.

$$\log p(x) = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\text{KL}}(q_\phi(z|x) \| p(z))}_{\mathcal{L}_{\text{ELBO}}} + D_{\text{KL}}(q_\phi(z|x) \| p_\theta(z|x))$$

ELBO를 최대화하는 것은 다음을 최소화하는 것과 같다.

$$\mathcal{L}_{\text{VAE}} = \underbrace{\mathbb{E}_{z \sim q_\phi(z|x)}[-\log p_\theta(x|z)]}_{\text{reconstruction}} + \underbrace{D_{\text{KL}}(q_\phi(z|x) \| p(z))}_{\text{regularization}}$$

### KL과 복원 사이의 절충

$\beta$-VAE는 KL 항에 가중치를 도입한다. $\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \cdot D_{\text{KL}}$이다.

| KL 값 | 의미 | 효과 |
|----------|---------|--------|
| **큼** | 근사 사후분포와 사전분포의 차이가 큼 | 입력 정보와 집계 사후분포의 불일치가 모두 기여할 수 있음 |
| **작음** | 근사 사후분포가 사전분포에 가까움 | 복원 품질이나 잠재 공간의 유용성을 단독으로 판단할 수 없음 |
| **0** | 거의 모든 입력에서 근사 사후분포가 사전분포와 같음 | 잠재변수가 입력을 구별하지 못함. 디코더 출력이 무작위라는 뜻은 아님 |

---

## 10. PyTorch 구현

### VAE를 위한 정규분포 KL

인코더는 각 잠재 차원에 대해 $\mu$과 $\log\sigma^2$을 낸다.

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
    인코더가 $\sigma^2$ 대신 $\log\sigma^2$을 내는 것은 수치적 안정성 때문이다. `logvar`는 어떤 실수든 될 수 있고(신경망의 출력으로 알맞다), `exp(logvar)`는 언제나 양수이며(분산으로 알맞다), `log(sigma^2) = logvar`는 작은 수의 로그를 취하는 일을 피하게 해 준다.

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

**Free bits.** 각 잠재 차원이나 그룹의 KL이 정한 기준 이하일 때는 패널티가 더 줄어들지 않도록 목적함수를 조정한다. 실제 KL 값에 하한을 강제하는 제약은 아니며, 사후분포 붕괴 방지를 보장하지도 않는다.

```python
def kl_free_bits(mu, logvar, free_bits=0.1):
    """자유 비트 제약이 있는 KL."""
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    return kl_per_dim.sum(dim=1)
```

**KL 가중치 스케줄링(KL annealing).** 학습 중에 KL 가중치를 점차 키운다.

```python
def get_beta(epoch, warmup_epochs=10, max_beta=1.0):
    """선형 KL 담금질."""
    if epoch < warmup_epochs:
        return max_beta * epoch / warmup_epochs
    return max_beta
```

### 일반적인 분포를 위한 `nn.KLDivLoss`

`nn.KLDivLoss`는 주로 이산 확률 텐서의 KL 항을 계산할 때 사용한다. 기본 설정에서 `input`은 근사 분포의 로그 확률이고 `target`은 목표 분포의 확률이다. `log_target=True`이면 목표도 로그 확률로 전달한다. 연속 정규분포의 밀도 값을 단순히 배열로 넣는 것만으로 연속 KL이 계산되는 것은 아니다.

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

`reduction='batchmean'`은 원소별 항의 합을 배치 크기로 나누므로, 각 행이 하나의 분포인 입력에서 표본당 KL 평균에 해당한다. `'sum'`은 합, `'mean'`은 전체 원소 수로 나눈 값, `'none'`은 원소별 항이다. 입력 텐서의 축과 집계 방식을 함께 확인한다. [공식 문서](https://docs.pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html)에 입력 관례와 집계식이 정리되어 있다.

### 해석적 KL과 몬테카를로 KL

두 분포가 정규분포이면 해석적 공식을 사용할 수 있어 몬테카를로 표본 오차가 없다. 다른 분포도 해석적 KL이 있을 수 있다. 닫힌 형태를 사용할 수 없지만 필요한 밀도와 샘플을 계산할 수 있다면, 로그밀도비의 평균으로 추정할 수 있다.

```python
def kl_monte_carlo(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """z ~ q에서 뽑은 표본으로 D_KL(q||p)을 추정한다.

    D_KL = E_q[log q - log p] ≈ (1/N) sum_i (log q(z_i) - log p(z_i))
    """
    return (log_q - log_p).mean()
```

---

## 11. 핵심 정리

KL 발산은 기대 로그밀도비이며 비음수이지만 비대칭이다. 순방향과 역방향의 근사 성향은 분포족의 제약에 따라 달라진다. 정규분포에서는 해석적 공식을, 이산 확률 텐서에서는 입력 방향과 집계 방식을 확인한 `KLDivLoss`를 사용할 수 있다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
이산 분포의 KL 발산이 음이 아님을 증명하라. $p$가 질량을 두는 곳 밖에도 $q$의 질량이 있을 수 있다는 점을 반영하라.

</div>

??? success "연습문제 1 풀이"
    $p_i>0$인 곳에서 $q_i=0$이면 KL은 무한대이므로 결론이 성립한다. 나머지 경우 옌센 부등식으로 $D_{\mathrm{KL}}(p\|q)\geq-\log\sum_{i:p_i>0}q_i\geq0$이다. 등호가 되려면 $q_i/p_i$가 $p$의 지지집합에서 상수이고 그 밖의 $q$ 질량은 0이어야 하므로 $p=q$이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
$p=\operatorname{Bernoulli}(0.3)$과 $q=\operatorname{Bernoulli}(0.7)$의 양방향 KL이 같은 이유를 설명하라. 이어서 $q=\operatorname{Bernoulli}(0.6)$으로 바꾸어 비대칭인 예를 제시하라.

</div>

??? success "연습문제 2 풀이"
    첫 쌍은 성공과 실패 확률이 서로 바뀌어 양방향 모두 $0.4\log(7/3)\approx0.339$나트이다. 같다고 해서 KL이 대칭인 것은 아니다. $q=\operatorname{Bernoulli}(0.6)$이면 $D(p\|q)=0.3\log(0.3/0.6)+0.7\log(0.7/0.4)\approx0.184$, $D(q\|p)=0.6\log(0.6/0.3)+0.4\log(0.4/0.7)\approx0.192$로 다르다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
양의 분산을 가진 두 단변량 정규분포의 KL 발산을 로그밀도비의 기대값으로 유도하라.

</div>

??? success "연습문제 3 풀이"
    로그밀도비는 $\log(\sigma_2/\sigma_1)+(x-\mu_2)^2/(2\sigma_2^2)-(x-\mu_1)^2/(2\sigma_1^2)$이다. $p$ 아래에서 두 제곱항의 기대값은 각각 $\sigma_1^2+(\mu_1-\mu_2)^2$와 $\sigma_1^2$이다. 대입하면 $D(p\|q)=\log(\sigma_2/\sigma_1)+[\sigma_1^2+(\mu_1-\mu_2)^2]/(2\sigma_2^2)-1/2$를 얻는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
순방향 KL의 모드 포괄 경향과 역방향 KL의 모드 선택 경향은 어떤 조건에서 나타나는가? 두 경향을 보편적인 정리로 읽으면 안 되는 이유를 설명하라.

</div>

??? success "연습문제 4 풀이"
    이 경향은 다봉 목표를 단봉 정규분포 등 제한된 분포족으로 근사할 때 잘 드러난다. 순방향은 목표의 질량을 빠뜨리는 비용을, 역방향은 목표 밀도가 낮은 곳에 근사 질량을 두는 비용을 강조한다. 충분히 표현력 있는 분포족에서 전역 최적화를 할 수 있다면 양쪽 모두 $q=p$가 최적이다. 따라서 KL 방향만으로 특정 분산이나 단일 모드 선택을 보장할 수 없다.

## 정리하며

이 절에서는 정의, 음이 아님 (깁스 부등식), 비대칭성, 교차 엔트로피와의 관계를 살펴보았다.
