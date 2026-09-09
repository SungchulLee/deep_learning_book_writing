# 바로잡지 않은 랑주뱅 알고리즘(ULA)
바로잡지 않은 랑주뱅 알고리즘은 메트로폴리스-헤이스팅스 바로잡기 없이 랑주뱅 SDE을 잘게 나눈다. 그래서 단순하고 확률 기울기와 잘 맞지만, 걸음 크기에 기대는 치우침이 끝까지 남는다.

---

## 1. 알고리즘

```
Algorithm: Unadjusted Langevin Algorithm (ULA)
──────────────────────────────────────────────
Input: ∇log π̃(θ), step size ε, n_samples T
Initialize: θ₀

For t = 0, 1, ..., T-1:
    η ~ N(0, I)
    θₜ₊₁ = θₜ + (ε/2) ∇log π̃(θₜ) + √ε η
```

이는 그저 랑주뱅 SDE의 오일러-마루야마 잘게 나누기이다:

$$
\theta_{t+1} = \theta_t + \frac{\epsilon}{2} \nabla \log \pi(\theta_t) + \sqrt{\epsilon} \, \boldsymbol{\eta}_t
$$

---

## 2. 치우침 분석

ULA의 멈춘 분포 $\pi_\epsilon$은 참 과녁 $\pi$과 다르다. 걸음 크기 $\epsilon$에서 총 변동 거리로 잰 치우침은 다음과 같다:

$$
\text{TV}(\pi_\epsilon, \pi) = O(\epsilon)
$$

립시츠 기울기를 갖는 강한 로그 오목 과녁(매끄러움 상수 $L$, 강볼록성 $m$)에서는 다음과 같다:

$$
\text{TV}(\pi_\epsilon, \pi) \leq C \cdot \epsilon \cdot \frac{L^2 d}{m}
$$

**뜻하는 바**: 치우침 $\leq \delta$을 이루려면 걸음 크기 $\epsilon = O(\delta / (L^2 d))$이 필요하고, 모이는 데 되풀이가 $O(d/\delta^2)$번 든다. 차원에는 다항이지만 바라는 정확도에는 역제곱이다.

---

## 3. 확률 기울기 랑주뱅 동역학(SGLD)

실전에서의 핵심 넓힘은 온전한 기울기를 작은 묶음에서 얻은 **확률 기울기**로 바꾸는 것이다:

$$
\theta_{t+1} = \theta_t + \frac{\epsilon_t}{2}\left(\nabla \log p(\theta_t) + \frac{N}{n}\sum_{i \in \text{batch}} \nabla \log p(x_i \mid \theta_t)\right) + \sqrt{\epsilon_t} \, \boldsymbol{\eta}_t
$$

여기서 $N$은 자료 묶음의 크기, $n$은 작은 묶음의 크기이다.

### 걸음 크기를 줄이는 일정

다음을 만족하며 줄어드는 걸음 크기 $\epsilon_t$을 쓰면:

$$
\sum_{t=1}^{\infty} \epsilon_t = \infty, \quad \sum_{t=1}^{\infty} \epsilon_t^2 < \infty
$$

(이를테면 $\epsilon_t = a / (b + t)$) 확률 기울기 잡음과 잘게 나눔 치우침이 모두 점근으로 사라지고 표본이 참 뒤확률로 모인다.

### PyTorch 구현

```python
import torch

class SGLD:
    """
    확률 기울기 랑주뱅 움직임.
    
    작은 묶음 기울기와 랑주뱅 잡음을 어우러지게 하여
    규모를 키울 수 있는 어림 뒤확률 표집을 이룬다.
    """
    
    def __init__(self, params, lr=1e-3, weight_decay=0.0,
                 noise_scale=1.0, lr_decay=0.0):
        self.params = list(params)
        self.lr_init = lr
        self.weight_decay = weight_decay
        self.noise_scale = noise_scale
        self.lr_decay = lr_decay
        self.step_count = 0
    
    @property
    def lr(self):
        if self.lr_decay > 0:
            return self.lr_init / (1 + self.lr_decay * self.step_count)
        return self.lr_init
    
    def step(self):
        """SGLD 새로 고치기 한 번 하기."""
        eps = self.lr
        
        for p in self.params:
            if p.grad is None:
                continue
            
            # 기울기 걸음(자료 가능도와 weight_decay로 넣은 앞확률 포함)
            d_p = p.grad.data
            if self.weight_decay > 0:
                d_p = d_p + self.weight_decay * p.data
            
            # 랑주뱅 잡음
            noise = torch.randn_like(p.data) * (self.noise_scale * (2 * eps) ** 0.5)
            
            # 새로 고치기: θ ← θ - ε∇U(θ) + √(2ε)η
            p.data.add_(-eps * d_p + noise)
        
        self.step_count += 1
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

def sgld_sample(model, dataloader, n_samples=100, burnin=1000,
                thin=10, lr=1e-4):
    """
    SGLD로 신경망의 뒤확률 표본 모으기.
    """
    optimizer = SGLD(model.parameters(), lr=lr, lr_decay=1e-5)
    
    samples = []
    total_steps = burnin + n_samples * thin
    
    for step in range(total_steps):
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            break  # 걸음마다 묶음 하나
        
        if step >= burnin and (step - burnin) % thin == 0:
            samples.append({
                name: p.data.clone() 
                for name, p in model.named_parameters()
            })
    
    return samples
```

---

## 4. ULA, MALA, HMC 견주기

| 결 | ULA/SGLD | MALA | HMC |
|--------|----------|------|-----|
| MH 바로잡기 | 아니오 | 예 | 예 |
| 치우침 | $O(\epsilon)$ | $O(\epsilon^3)$ | $O(\epsilon^{L})$ |
| 확률 기울기 | 자연스럽다 | 어렵다 | 어렵다 |
| 크게 키우기 | 아주 좋다 | 좋다 | 보통이다 |
| 정확도 | 어림 | 점근으로 정확 | 점근으로 정확 |
| 맞추기 | 걸음 크기만 | 걸음 크기 | 걸음 크기, 자취, 질량 |

---

## 5. ULA/SGLD를 언제 쓰나

**다음일 때 SGLD를 써라:**

- 자료 묶음이 아주 크다(관측이 수백만 개)
- 어림 뒤확률로 넉넉하다
- SGD 기반 가르치기 흐름에 녹여 넣는다
- 단순함과 최소한의 맞추기가 필요하다

**다음일 때는 MALA/HMC가 낫다:**

- 정확한 뒤확률 표본이 필요하다
- 자료 묶음이 작거나 보통이다
- 불확실함을 재는 데 높은 정확도가 아주 중요하다

---

## 6. 자세한 치우침 보기: 1차원 가우스

$\pi(x) = \mathcal{N}(0, 1)$에서 점수는 $s(x) = -x$이다. ULA의 새로 고치기는 다음과 같다:

$$
x_{t+1} = x_t - \epsilon x_t + \sqrt{2\epsilon} \, \eta_t = (1 - \epsilon) x_t + \sqrt{2\epsilon} \, \eta_t
$$

이는 멈춘 흩어짐이 다음과 같은 AR(1) 과정이다:

$$
\sigma^2_\epsilon = \frac{2\epsilon}{1 - (1-\epsilon)^2} = \frac{1}{1 - \epsilon/2}
$$

$\epsilon = 0.1$이면 멈춘 흩어짐이 1이 아니라 $\approx 1.053$이다. 치우침은 $O(\epsilon)$이다.

---

## 7. 기본 ULA 구현

```python
import torch

def ula(score_fn, x0, n_steps, epsilon):
    """다듬지 않은 랑주뱅 알고리즘.

    인수:
        score_fn: x [batch, dim] → 점수 [batch, dim]으로 잇는다.
        x0: 첫 상태 [batch, dim].
        n_steps: 되풀이 횟수.
        epsilon: 걸음 크기.

    반환값:
        되풀이 n_steps번 뒤의 표본 [batch, dim].
    """
    x = x0.clone()
    sqrt_2eps = (2 * epsilon) ** 0.5
    for _ in range(n_steps):
        x = x + epsilon * score_fn(x) + sqrt_2eps * torch.randn_like(x)
    return x
```

---

## 8. ULA을 언제 쓰나

다음일 때 ULA이 알맞다:

- 정확한 표본이 필요하지 않을 때(이를테면 어림 추론, 최적화의 따뜻한 출발)
- 치우침이 하찮아질 만큼 걸음 크기를 작게 할 수 있을 때
- 정확함보다 빠르기가 더 중요할 때

### ULA의 모임 진단

표준 MCMC 진단을 쓰되 주의할 점이 있다:

- **자취 그림**은 멈춤과 좋은 섞임을 보여야 한다
- **ESS**은 표본이 얼마나 실효로 독립인지를 잰다
- **달리는 평균**은 안정되어야 한다. 다만 치우친 과녁으로 안정된다
- 여러 사슬이 같은 (치우친) 분포로 모여야 한다

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

이 마당은 알고리즘、치우침 분석、확률 기울기 랑주뱅 동역학(SGLD)、ULA, MALA, HMC 견주기을 차례로 짚었다.

**참고 문헌**

- Welling, M., & Teh, Y. W. (2011). Bayesian learning via stochastic gradient Langevin dynamics. *ICML*.
- Dalalyan, A. S. (2017). Theoretical guarantees for approximate sampling from a smooth and log-concave density. *JRSS-B*, 79(3), 651-676.
- Chen, T., Fox, E., & Guestrin, C. (2014). Stochastic gradient Hamiltonian Monte Carlo. *ICML*.

---
