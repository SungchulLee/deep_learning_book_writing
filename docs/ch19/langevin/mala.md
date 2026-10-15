# 메트로폴리스 보정 랑주뱅 알고리즘(MALA)

MALA는 Langevin의 한 단계 가우스 제안을 MH로 보정하는 방법이다. 비대칭 제안의 정·역밀도를 포함해 원래 목표를 불변으로 유지하며, 유한 실행의 탐색·정밀도는 별도로 확인한다.

## 1. 알고리즘

ε>0에서 스코어 s(x)=∇log γ(x)를 사용해 y=x+εs(x)/2+√εη를 제안한다. 제안밀도는 q(y|x)=N(y;x+εs(x)/2,εI)이며 일반적으로 비대칭이다.

$$r=\log\gamma(y)-\log\gamma(x)+\log q(x|y)-\log q(y|x),\qquad\log\alpha=\min(0,r).$$

절단하지 않은 r을 log α라고 쓰면 r>0일 때 의미가 맞지 않는다. log U<r 비교는 log U≤0이므로 암묵적으로 min(0,r)와 같은 수락 규칙을 구현한다. 거절하면 현재 상태를 기록하고 미지 목표 상수는 로그밀도 차이에서 소거된다. 아래 의사코드는 이 흐름이다.

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

## 2. 편향 보정

정확한 목표값·제안밀도와 올바른 MH 보정으로 구성한 MALA는 목표 π를 불변으로 갖는다. 안정적인 고정 스텝 ULA의 목표 근사 편향과 다르다. 다만 초기 영향·미탐색 모드·상관된 유한 평균의 오차·수치 구현 문제가 사라지는 것은 아니다.

대표적인 고차원 곱 목표의 점근 이론에서 이 페이지의 제안 분산 ε는 d⁻¹/³ 차수, 잡음 표준편차 √ε는 d⁻¹/⁶ 차수다. 관련 조건에서 수락률 약 0.574가 효율 기준을 최적화한다. 이는 모든 모형의 필수 수락률이 아니다.

## 3. PyTorch 구현

단일 체인 클래스는 길이 d의 상태와 스칼라 로그확률을 가정한다. grad 함수는 상태를 detach한 뒤 미분하지만 lp.backward()는 상태뿐 아니라 log_prob에 학습 가능한 매개변수가 있으면 그 경사도 누적할 수 있다. 상태 경사 계산과 모델 학습을 구분해야 한다.

sample은 입력을 float32로 바꾸고 기본 CPU 텐서에 표본·잡음을 생성한다. 임의 GPU·dtype 입력을 그대로 지원하는 구현은 아니다. 양의 ε·표본 수, 비음수 워밍업과 유한 로그비를 확인한다. warmup은 같은 고정 전이를 버리는 반복이며 스텝 크기를 적응하지 않는다. 같은 경사를 여러 번 재평가하므로 이론의 경사 비용과 실제 조각의 비용도 구분한다.

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

## 4. 상수·위치 의존 전처리

상수 대칭 양의 정부호 P를 쓰면 y=x+εPs(x)/2+√εP¹/²η를 제안한다. 제안 공분산은 εP이며 같은 고정 P의 정·역 가우스 밀도로 보정한다. HMC의 K=pᵀM⁻¹p/2 관례에서는 P=M⁻¹에 대응하며 질량 M 자체와 같지 않다.

P≈목표 공분산은 전역 스케일의 출발점이 될 수 있다. 위치 의존 P(x)는 가우스 정규화의 log det P(x)와 정·역 이차형식을 포함해야 한다. 정확한 MH는 정의한 유효 제안 자체를 보정할 수 있지만, 특정 리만 확산의 이산화라고 주장하려면 Itô의 추가 드리프트 등 해당 식을 따로 맞춰야 한다.

## 5. 방법 간 비용과 정확도 비교

MALA는 한 가우스 제안에 정·역 스코어 평가가 필요하고 HMC는 여러 적분 단계의 경사를 사용한다. 랜덤 워크·MALA의 대표 점근 반복 스케일과 HMC의 궤적당 Leapfrog 수를 하나의 혼합시간 열로 비교하지 않는다.

ULA/SGLD의 근사 편향을 허용하는 경우와 MH 보정의 비용을 지불하는 경우를 나누어 동일한 관심 함수의 오차·시간을 비교한다. 차원이 낮음·중간·높음이라는 구분만으로 방법의 우열을 결정할 수 없다.

## 6. 정·역 가우스 제안비

정방향·역방향 가우스의 공통 정규화 상수는 고정 ε에서 소거된다. 나머지 로그 제안비는 다음과 같다.

$$\log\frac{q(x|y)}{q(y|x)}=\frac{\|y-x-\epsilon s(x)/2\|^2-\|x-y-\epsilon s(y)/2\|^2}{2\epsilon}.$$

상태 의존 평균 이동으로 비대칭이 생기므로 목표밀도 비만 사용하는 수락식은 일반적으로 올바르지 않다. 위치 의존 공분산이면 공통 정규화 상수가 아니므로 행렬식 항까지 남는다. 수락 규칙의 대칭 흐름은 목표 불변성을 보이지만 초기값에서의 수렴에는 추가 조건이 필요하다.

## 7. 배치 MALA 구현

mala_batch의 입력은 배치×차원이고 log_prob_fn은 각 행의 로그확률을 길이 배치로 반환해야 한다. score_fn은 같은 배치×차원 형태를 반환한다. 행별 잡음과 수락 마스크를 사용해 여러 체인의 마지막 상태를 돌려준다.

이 배치는 n_steps개 시점의 체인 전체가 아니다. n_accept/n_steps는 매 단계 배치 수락 비율의 평균이며 행별 진단 기록을 대체하지 않는다. 양의 스텝 수·ε와 유한한 값·기기 호환성을 확인하고, 같은 경사를 사용하는 행별 독립 전이라는 전제를 유지한다.

```python
import torch

def mala_batch(log_prob_fn, score_fn, x0, n_steps, epsilon):
    """나란한 사슬 여럿을 위한 배치 MALA.

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

## 8. 스텝 단위와 조정 기준

이 페이지 전체에서 ε는 제안 분산이다. 따라서 MALA 점근 ε∝d⁻¹/³와 표준편차 √ε∝d⁻¹/⁶를 구분한다. ULA에 같은 보편적 차원 스케일을 임의로 부여하면 안 되며 허용 편향·거리·목표의 정칙성에 따라 스텝을 정한다.

0.574는 특정 점근 분석의 참고값이고 70%·45%를 모든 문제의 자동 조정 경계로 제시하지 않는다. 아래 정규 배치 예제의 분산과 수락률은 보존된 한 결과다. 수락률이 높다는 사실만으로 설정이 나쁘다거나 충분히 수렴했다고 결론내릴 수 없다. 함수별 MCSE·비용과 여러 시작점을 함께 평가한다.

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
```
MALA variance: 0.9998, acceptance: 89.18%
```

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
MALA의 수락 규칙에서 왜 정·역 제안밀도가 필요한가?

</div>

??? success "연습문제 1 풀이"
    경사에 따라 제안 평균이 바뀌므로 비대칭이다. 두 가우스 이차형식의 차이를 로그 목표비에 더해야 일반적인 MH 상세균형이 성립한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
ε∝d⁻¹/³와 d⁻¹/⁶가 왜 함께 등장하는가?

</div>

??? success "연습문제 2 풀이"
    이 페이지의 ε는 분산이고 √ε는 잡음 표준편차다. 특정 고차원 점근 이론의 같은 제안을 서로 다른 단위로 표현한 것이며 모든 목표의 보편 공식은 아니다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
로그 수락률과 절단 전 로그비를 구분하라.

</div>

??? success "연습문제 3 풀이"
    절단 전 r은 양수일 수 있지만 log α=min(0,r)는 0 이하이다. log U<r 비교는 log U≤0이므로 암묵적으로 올바른 절단을 구현한다. 비유한 r은 별도 처리해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
배치 구현의 반환값은 독립적인 목표 표본인가?

</div>

??? success "연습문제 4 풀이"
    각 행은 유한 단계 전이의 마지막 위치이며 초기 영향이 남을 수 있다. 행별 독립 난수를 사용해도 목표에 도달했다는 보장은 없고 시간 시퀀스 전체를 반환하는 것도 아니다.

## 정리하며

MALA는 비대칭 가우스 제안의 MH 보정으로 목표 불변성을 구성한다. 제안 분산·표준편차 스케일을 통일하고 전처리 관례, 배치 형태와 실제 적응·진단 범위를 구분한다.

**참고 문헌**

- Roberts, G. O., & Tweedie, R. L. (1996). Exponential convergence of Langevin distributions and their discrete approximations. *Bernoulli*, 2(4), 341-363.
- Roberts, G. O., & Rosenthal, J. S. (1998). Optimal scaling of discrete approximations to Langevin diffusions. *JRSS-B*, 60(1), 255-268.
- Girolami, M., & Calderhead, B. (2011). Riemann manifold Langevin and Hamiltonian Monte Carlo methods. *JRSS-B*, 73(2), 123-214.

---
