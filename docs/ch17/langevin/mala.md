# 메트로폴리스 바로잡은 랑주뱅 알고리즘(MALA)
MALA은 랑주뱅 동역학의 기울기를 담은 제안과 메트로폴리스-헤이스팅스 받아들임-물리침 바로잡기를 합쳐, 기울기가 이끄는 살펴보기의 이점을 지키면서 ULA의 잘게 나눔 치우침을 없앤다.

---

## 1. 알고리즘

```
Algorithm: MALA
───────────────
Input: log π̃(θ), ∇log π̃(θ), step size ε
Initialize: θ₀

For t = 0, 1, ..., T-1:
    1. Propose:
       θ' = θₜ + (ε/2) ∇log π̃(θₜ) + √ε η,    η ~ N(0, I)
    
    2. Compute acceptance probability:
       α = min(1, [π̃(θ') q(θₜ|θ')] / [π̃(θₜ) q(θ'|θₜ)])
    
       where q(θ'|θ) = N(θ' | θ + (ε/2)∇log π̃(θ), εI)
    
    3. Accept/reject:
       u ~ Uniform(0,1)
       If u < α:  θₜ₊₁ = θ'    (accept)
       Else:      θₜ₊₁ = θₜ    (reject)
```

### 제안 분포

MALA의 제안은 랑주뱅이 이끈 자리를 가운데로 하는 가우스이다:

$$
q(\theta' \mid \theta) = \mathcal{N}\left(\theta' \;\Big|\; \theta + \frac{\epsilon}{2}\nabla \log \tilde{\pi}(\theta), \; \epsilon \mathbf{I}\right)
$$

이는 **비대칭**이다. 곧 기울기 항이 $\theta$과 $\theta'$에서 다르므로 $q(\theta' \mid \theta) \neq q(\theta \mid \theta')$이다. 그래서 MH 비에 제안 밀도가 들어가야 한다.

### 로그 받아들임 비

$$
\log \alpha = \log \tilde{\pi}(\theta') - \log \tilde{\pi}(\theta) + \log q(\theta \mid \theta') - \log q(\theta' \mid \theta)
$$

여기서 각 기호는 다음과 같다.

$$
\log q(\theta' \mid \theta) = -\frac{1}{2\epsilon}\left\|\theta' - \theta - \frac{\epsilon}{2}\nabla \log \tilde{\pi}(\theta)\right\|^2 + \text{const}
$$

---

## 2. 편향 보정

MH 걸음이 ULA의 $O(\epsilon)$ 잘게 나눔 치우침을 바로잡는다:

| 방법 | 멈춘 분포 | 치우침 |
|--------|------------------------|------|
| ULA | $\pi_\epsilon \neq \pi$ | $O(\epsilon)$ |
| **MALA** | $\pi$(정확) | **없음**(점근으로 정확) |
| HMC | $\pi$(정확) | 없음(점근으로 정확) |

다만 MALA의 받아들임 비율은 $\epsilon$과 차원 $d$에 기댄다:

$$
\text{Optimal } \epsilon = O(d^{-1/3})
$$

가장 좋은 받아들임 비율은 대략 **0.574**이다(Roberts & Rosenthal, 1998).

---

## 3. PyTorch 구현

```python
import torch

class MALA:
    """
    메트로폴리스로 다듬은 랑주뱅 알고리즘.
    
    매개변수
    ----------
    log_prob : callable
        log π̃(θ)을 셈한다. autograd를 받쳐야 한다.
    step_size : float
        랑주뱅 걸음 크기 ε
    """
    
    def __init__(self, log_prob, step_size=0.1):
        self.log_prob = log_prob
        self.step_size = step_size
    
    def _grad_log_prob(self, theta):
        """autograd로 ∇log π̃(θ) 셈하기."""
        theta = theta.detach().requires_grad_(True)
        lp = self.log_prob(theta)
        lp.backward()
        return theta.grad.detach()
    
    def _log_proposal(self, theta_to, theta_from):
        """log q(theta_to | theta_from) 셈하기."""
        grad = self._grad_log_prob(theta_from)
        mean = theta_from + 0.5 * self.step_size * grad
        diff = theta_to - mean
        return -0.5 / self.step_size * (diff ** 2).sum()
    
    def sample(self, theta_init, n_samples, warmup=1000):
        """MALA 사슬 돌리기."""
        d = theta_init.shape[0]
        theta = theta_init.clone().float()
        
        samples = torch.zeros(n_samples, d)
        n_accept = 0
        
        for t in range(n_samples + warmup):
            # 랑주뱅 제안
            grad = self._grad_log_prob(theta)
            mean = theta + 0.5 * self.step_size * grad
            theta_prop = mean + (self.step_size ** 0.5) * torch.randn(d)
            
            # 로그 받아들임 비
            log_alpha = (
                self.log_prob(theta_prop) - self.log_prob(theta)
                + self._log_proposal(theta, theta_prop)
                - self._log_proposal(theta_prop, theta)
            )
            
            # 받아들이거나 물리치기
            if torch.log(torch.rand(1)) < log_alpha:
                theta = theta_prop
                if t >= warmup:
                    n_accept += 1
            
            if t >= warmup:
                samples[t - warmup] = theta.detach()
        
        accept_rate = n_accept / n_samples
        return samples, accept_rate
```

---

## 4. 미리 다듬은 MALA

(HMC의 질량 행렬에 해당하는) 미리 다듬기 행렬 $\mathbf{M}$을 쓰면:

$$
\theta' = \theta + \frac{\epsilon}{2}\mathbf{M}\nabla \log \tilde{\pi}(\theta) + \sqrt{\epsilon} \, \mathbf{M}^{1/2} \boldsymbol{\eta}
$$

$\mathbf{M}$을 뒤확률 공분산의 어림값으로 두면 방향마다 걸음 크기가 고르게 되어, 방향에 따라 다른 뒤확률에서 섞임이 나아진다.

리만 다양체 판은 자리에 따라 달라지는 미리 다듬개로 **피셔 정보 행렬** $\mathbf{G}(\theta)$을 써서 그 자리의 굽음에 맞춘다.

---

## 5. 표집 층위 속의 MALA

| 방법 | 제안 | 바로잡기 | 커짐새 | 가장 알맞은 곳 |
|--------|----------|------------|---------|----------|
| 무작위 걸음 MH | $\mathcal{N}(\theta, \sigma^2 I)$ | MH | $O(d)$ | 낮은 $d$ |
| ULA/SGLD | 랑주뱅 걸음 | 없음 | — | 큰 자료 |
| **MALA** | 랑주뱅 걸음 | **MH** | $O(d^{1/3})$ | **중간 $d$** |
| HMC | 해밀턴 자취 | MH | $O(d^{1/4})$ | 높은 $d$ |

MALA은 알맞은 자리를 차지한다. 곧 HMC보다 단순하고, ULA보다 정확하며, 무작위 걸음 MH보다 커짐새가 낫다.

---

## 6. 헤이스팅스 바로잡기가 왜 필요한가

제안은 **비대칭**이다. 곧 흐름 $\epsilon s(x)$이 지금 자리에 기대므로 $q(x' | x) \neq q(x | x')$이다. 헤이스팅스 바로잡기가 없으면 자세한 균형이 깨져 사슬의 멈춘 분포가 $\pi$이 되지 않는다.

### 로그 받아들임 비를 자세히 셈하기

제안의 로그 밀도는 다음과 같다:

$$
\log q(x' | x) = -\frac{\|x' - x - \frac{\epsilon}{2} s(x)\|^2}{2\epsilon} + \text{const}
$$

$$
\log q(x | x') = -\frac{\|x - x' - \frac{\epsilon}{2} s(x')\|^2}{2\epsilon} + \text{const}
$$

그러므로 제안의 로그 비는 다음과 같다:

$$
\log q(x | x') - \log q(x' | x) = \frac{1}{2\epsilon}\left[\|x' - x - \tfrac{\epsilon}{2} s(x)\|^2 - \|x - x' - \tfrac{\epsilon}{2} s(x')\|^2\right]
$$

---

## 7. 묶음 MALA 구현

```python
import torch

def mala_batch(log_prob_fn, score_fn, x0, n_steps, epsilon):
    """나란한 사슬 여럿을 위한 묶음 MALA.

    인수:
        log_prob_fn: x [batch, dim] → log π(x) [batch]으로 잇는다.
        score_fn: x [batch, dim] → ∇ log π(x) [batch, dim]으로 잇는다.
        x0: 첫 상태 [batch, dim].
        n_steps: 되풀이 횟수.
        epsilon: 걸음 크기.

    반환값:
        samples: 마지막 표본 [batch, dim].
        accept_rate: 받아들여진 제안의 몫.
    """
    x = x0.clone()
    n_accept = 0
    sqrt_eps = epsilon ** 0.5

    for _ in range(n_steps):
        s_x = score_fn(x)
        noise = torch.randn_like(x)
        x_prop = x + 0.5 * epsilon * s_x + sqrt_eps * noise

        s_xp = score_fn(x_prop)

        # 로그 받아들임 비
        log_pi_diff = log_prob_fn(x_prop) - log_prob_fn(x)

        # 제안의 로그 비: log q(x|x') - log q(x'|x)
        fwd = x_prop - x - 0.5 * epsilon * s_x
        bwd = x - x_prop - 0.5 * epsilon * s_xp
        log_q_diff = (fwd.pow(2).sum(dim=-1) - bwd.pow(2).sum(dim=-1)) / (2 * epsilon)

        log_alpha = log_pi_diff + log_q_diff
        accept = torch.log(torch.rand(x.shape[0], device=x.device)) < log_alpha

        x = torch.where(accept.unsqueeze(-1), x_prop, x)
        n_accept += accept.float().mean().item()

    return x, n_accept / n_steps
```

---

## 8. 가장 좋게 맞추기

### 걸음 크기의 커짐새

가장 좋은 걸음 크기는 차원 $d$에 기댄다:

| 알고리즘 | 가장 좋은 $\epsilon$ | 커짐새 |
|-----------|-------------------|---------| 
| ULA | $\epsilon \propto d^{-1/3}$ | 치우침 다스리기 |
| MALA | $\epsilon \propto d^{-1/6}$ | 받아들임 비율 |

받아들임 걸음이 지나치게 거친 제안을 잡아 주므로 MALA은 더 큰 걸음을 견딘다.

### 실전에서 맞추기

**목표 받아들임 비율**: 대략 **57%**이다(Roberts & Rosenthal, 1998). 받아들임이 너무 높으면 걸음 크기가 너무 작아 섞임이 느리다. 너무 낮으면 제안 대부분이 물리쳐져 사슬이 거의 움직이지 않는다.

- 받아들임 > 70%: $\epsilon$을 키운다
- 받아들임 < 45%: $\epsilon$을 줄인다

### 보기: MALA 맞추기 확인하기

```python
# 과녁: 10차원의 N(0, 1)
dim = 10
n_samples = 1000

def log_prob(x):
    return -0.5 * x.pow(2).sum(dim=-1)

def score(x):
    return -x

x0 = torch.randn(n_samples, dim) * 3  # 지나치게 흩어진 시작

x_mala, acc = mala_batch(log_prob, score, x0.clone(), n_steps=500, epsilon=0.5)
print(f"MALA variance: {x_mala.var(dim=0).mean():.4f}, acceptance: {acc:.2%}")
```

---

## 연습문제

**연습문제 1.**
마르코프 사슬이 올바른 과녁 분포로 모이게 하는 데 받아들임 확률이 하는 몫을 설명하여라.

??? success "연습문제 1 풀이"
    받아들임 확률이 **자세한 균형** $\pi(x) T(x \to x') \alpha(x \to x') = \pi(x') T(x' \to x) \alpha(x' \to x)$을 보장한다. 여기서 $\pi$은 과녁 분포, $T$은 제안 분포, $\alpha$은 받아들임 확률이다. 자세한 균형은 $\pi$이 사슬의 멈춘 분포임을 뜻한다. 쪼갤 수 없음과 주기 없음까지 합치면 $\pi$으로의 에르고드 모임이 보장된다.

---

**연습문제 2.**
제안 분포가 너무 좁은 상황과 너무 넓은 상황을 밝혀라. 저마다 표집 효율에 어떤 영향을 주는가?

??? success "연습문제 2 풀이"
    **너무 좁을 때:** 제안이 거의 늘 받아들여지지만(받아들임 비율이 높지만) 사슬이 아주 작은 걸음을 떼어 과녁 분포를 느리게 살펴본다. 그러면 자기상관이 높고 실효 표본 크기가 작아진다. **너무 넓을 때:** 제안이 확률이 낮은 구역에 자주 떨어져 물리쳐지므로(받아들임 비율이 낮으므로) 사슬이 여러 되풀이 동안 지금 상태에 갇혀 있게 된다. 두 극단 모두 효율을 떨어뜨린다. 높은 차원에서 무작위 걸음 메트로폴리스의 가장 좋은 받아들임 비율은 대략 0.234이다(Roberts 외, 1997).

---

**연습문제 3.**
메트로폴리스-헤이스팅스 받아들임 비 $\alpha = \min\left(1, \frac{\pi(x') q(x|x')}{\pi(x) q(x'|x)}\right)$이 $\pi$에 대해 자세한 균형을 만족함을 증명하여라.

??? success "연습문제 3 풀이"
    일반성을 잃지 않고 $\pi(x') q(x|x') \leq \pi(x) q(x'|x)$이라 하자. 그러면 $\alpha(x \to x') = \frac{\pi(x') q(x|x')}{\pi(x) q(x'|x)}$이고 $\alpha(x' \to x) = 1$이다. 자세한 균형 조건은 다음을 요구한다:

    $$\pi(x) q(x'|x) \alpha(x \to x') = \pi(x) q(x'|x) \cdot \frac{\pi(x') q(x|x')}{\pi(x) q(x'|x)} = \pi(x') q(x|x')$$

    그리고 $\pi(x') q(x|x') \alpha(x' \to x) = \pi(x') q(x|x') \cdot 1 = \pi(x') q(x|x')$이다. 양변이 같다. $\square$

---

**연습문제 4.**
MCMC에서 태우기 기간이란 무엇이며, 처음 표본을 언제 버릴지 어떻게 정하는가?

??? success "연습문제 4 풀이"
    태우기 기간은 마르코프 사슬에서 아직 멈춘 분포로 모이지 않은 처음 부분이다. 치우침을 줄이려고 이 기간의 표본을 버린다. 태우기를 정하는 길은 다음과 같다. (1) 자취 그림으로 사슬이 언제 안정되는지 눈으로 살핀다. (2) 여러 사슬에서 사슬 안 흩어짐과 사슬 사이 흩어짐을 견주는 겔먼-루빈 진단($\hat{R}$)을 쓰며 $\hat{R} < 1.01$이면 모였다고 본다. (3) 실효 표본 크기(ESS) 어림값을 쓴다. (4) 흩어진 시작점에서 여러 사슬을 돌려 서로 맞는지 살핀다.

## 정리하며

| 개념 | 핵심 |
|---------|-----------|
| **MALA = ULA + MH 바로잡기** | 잘게 나눔 치우침을 없앤다 |
| **비대칭 제안** | 기울기 때문에 $q(\theta' \mid \theta) \neq q(\theta \mid \theta')$이다 |
| **가장 좋은 받아들임** | 약 57.4%(견주기: 무작위 걸음 MH 약 23.4%, HMC 약 65%) |
| **걸음 크기 커짐새** | $\epsilon = O(d^{-1/3})$ — 무작위 걸음 MH의 $O(d^{-1})$보다 낫다 |
| **미리 다듬기** | 섞임을 낫게 하려고 뒤확률의 기하에 맞춘다 |

---

**참고 문헌**

- Roberts, G. O., & Tweedie, R. L. (1996). Exponential convergence of Langevin distributions and their discrete approximations. *Bernoulli*, 2(4), 341-363.
- Roberts, G. O., & Rosenthal, J. S. (1998). Optimal scaling of discrete approximations to Langevin diffusions. *JRSS-B*, 60(1), 255-268.
- Girolami, M., & Calderhead, B. (2011). Riemann manifold Langevin and Hamiltonian Monte Carlo methods. *JRSS-B*, 73(2), 123-214.

---
