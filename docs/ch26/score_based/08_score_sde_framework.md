# 점수 확률 미분 방정식 틀

이 단원은 요즘 만들어 내는 모델의 핵심 부품인 점수 확률 미분 방정식 틀을 짠다. 여기서 보이는 개념과 재주를 알면 퍼짐 모델과 점수 바탕 만들어 내는 방법을 다루는 데 꼭 필요한 앎을 얻는다. 이 짜기는 또렷함과 실제 쓸모의 균형을 맞추어 배우기에도 실험하기에도 알맞다.

## 코드

```python
"""
단원 08: 점수 바탕 확률 미분 방정식
===========================

어려움: 나아간 단계
시간: 3~4시간
미리 알 것: 단원 01-07, 기본 확률 미분 방정식 앎

학습 목표:
- 이어진 때의 퍼짐 틀을 이해한다
- 흩어짐 터짐과 흩어짐 지키기 확률 미분 방정식을 짠다
- 띄엄띄엄한 DDPM을 이어진 확률 미분 방정식과 잇는다

고갱이 방정식:
dx = f(x,t)dt + g(t)dw  (forward SDE)
dx = [f(x,t) - g(t)²∇log p_t(x)]dt + g(t)dw̄  (reverse SDE)

지은이: 이성철 @ 연세대학교
"""

import torch
import numpy as np

# ========================================================================
# 메인
# ========================================================================

print("MODULE 08: Score-Based SDE Framework")
print("="*80)

print("""
이어진 때의 꼴:
--------------------------
띄엄띄엄한 걸음 t=0,1,2,...,T 대신
이어진 때 t ∈ [0, T]를 쓴다

앞으로 가는 SDE(잡음을 더한다):
dx = f(x,t)dt + g(t)dw

여기서 각 기호는 다음과 같다.
- f(x,t): 떠돎 계수
- g(t): 퍼짐 계수
- dw: 브라운 움직임

거꾸로 가는 SDE(잡음을 없앤다):
dx = [f(x,t) - g(t)²∇log p_t(x)]dt + g(t)dw̄

핵심: 점수 ∇log p_t(x)이 뒤 과정에 나타난다!
이것을 신경 그물이 배운다: s_θ(x,t) ≈ ∇log p_t(x)

두 가지 큰 꼴:
---------------------

1. VARIANCE EXPLODING (VE):
   Forward: dx = √(dσ²/dt) dw
   → 흩어짐이 커진다: σ_t² = σ_min² + t(σ_max² - σ_min²)/T
   
2. VARIANCE PRESERVING (VP):
   Forward: dx = -0.5β(t)x dt + √β(t) dw
   → 흩어짐이 가둬져 있다
   → 이것이 이어진 때의 DDPM이다!

β(t)이 잡음 짜임을 다스린다
""")

class VESDE:
    """흩어짐 터짐 확률 미분 방정식"""
    def __init__(self, sigma_min=0.01, sigma_max=50.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def sigma_t(self, t):
        """잡음 차례표"""
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    
    def forward_sde(self, x, t):
        """dx = √(dσ²/dt) dw"""
        sigma = self.sigma_t(t)
        drift = torch.zeros_like(x)
        
        # g(t) = σ(t)√(2 log(σ_max/σ_min))
        diffusion = sigma * np.sqrt(2 * np.log(self.sigma_max / self.sigma_min))
        return drift, diffusion
    
    def reverse_sde(self, score_fn, x, t):
        """
        dx = [-g²∇log p]dt + g dw̄
        
        VE에서는 f=0이므로 거꾸로 표류는 그저 -g²∇log p이다
        """
        sigma = self.sigma_t(t)
        score = score_fn(x, t)
        
        diffusion = sigma * np.sqrt(2 * np.log(self.sigma_max / self.sigma_min))
        
        # 고침: 퍼뜨리기에 맞게 퍼짐의 꼴을 바꾼다
        # 퍼짐의 꼴은 (묶음,)이지만 점수의 꼴은 (묶음, 차원)이다
        # 퍼뜨리기를 위해 퍼짐의 꼴을 (묶음, 1)으로 만들어야 한다
        if isinstance(diffusion, torch.Tensor) and diffusion.dim() > 0:
            diffusion_sq = (diffusion ** 2).unsqueeze(-1)
        else:
            diffusion_sq = diffusion ** 2
        
        drift = -diffusion_sq * score
        
        return drift, diffusion

class VPSDE:
    """흩어짐을 지키는 SDE(DDPM과 같다)"""
    def __init__(self, beta_min=0.1, beta_max=20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max
    
    def beta_t(self, t):
        """선형 잡음 차례표"""
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def forward_sde(self, x, t):
        """dx = -0.5β(t)x dt + √β(t) dw"""
        beta = self.beta_t(t)
        
        # 고침: 퍼뜨리기가 제대로 되게 한다
        if isinstance(beta, torch.Tensor) and beta.dim() > 0:
            beta = beta.unsqueeze(-1)
        
        drift = -0.5 * beta * x
        diffusion = torch.sqrt(beta) if isinstance(beta, torch.Tensor) else np.sqrt(beta)
        return drift, diffusion
    
    def reverse_sde(self, score_fn, x, t):
        """
        dx = [-0.5β(t)x - β(t)∇log p]dt + √β(t) dw̄
        """
        beta = self.beta_t(t)
        score = score_fn(x, t)
        
        # 고침: 퍼뜨리기가 제대로 되게 한다
        if isinstance(beta, torch.Tensor) and beta.dim() > 0:
            beta_reshaped = beta.unsqueeze(-1)
        else:
            beta_reshaped = beta
        
        drift = -0.5 * beta_reshaped * x - beta_reshaped * score
        diffusion = torch.sqrt(beta) if isinstance(beta, torch.Tensor) else np.sqrt(beta)
        
        return drift, diffusion

def euler_maruyama_sampler(sde, score_fn, shape, n_steps=1000):
    """
    뒤 확률 미분 방정식을 푸는 오일러-마루야마 방법
    
    Discretization:
    x_{i-1} = x_i + drift * Δt + diffusion * √Δt * z
    where z ~ N(0, I)
    """
    # 사전 분포에서 시작한다
    x = torch.randn(shape)
    
    dt = 1.0 / n_steps
    trajectory = [x.clone()]
    
    for i in range(n_steps):
        t = 1.0 - i * dt  # 1에서 0으로 거슬러 간다
        t_tensor = torch.ones(shape[0]) * t
        
        # 뒤 확률 미분 방정식 걸음
        drift, diffusion = sde.reverse_sde(score_fn, x, t_tensor)
        
        # 고침: 잡음 항의 퍼짐 퍼뜨리기를 다룬다
        if isinstance(diffusion, torch.Tensor) and diffusion.dim() > 0:
            noise = torch.randn_like(x) * diffusion.unsqueeze(-1) * np.sqrt(dt)
        else:
            noise = torch.randn_like(x) * diffusion * np.sqrt(dt)
        
        # 오일러-마루야마 고침
        x = x + drift * dt + noise
        
        if i % 100 == 0:
            trajectory.append(x.clone())
    
    return x, trajectory

print("""
확률 흐름 ODE:
--------------------
뒤 확률 미분 방정식의 대안: 정해진 뽑기!

dx = [f(x,t) - 0.5*g(t)²∇log p_t(x)]dt

SDE와 가장자리 분포는 같으나 다음이 다르다.
✓ Deterministic (no randomness)
✓ 되돌릴 수 있다(부호로 넣고 풀 수 있다)
✓ 더 빠르다(걸음을 크게 할 수 있다)
✗ 표본 품질이 떨어질 수 있다

이로써 다음이 가능해진다:
- 그림 사이 메우기
- 뜻으로 고치기
- 정확한 가능도 셈하기

DDPM과의 이음:
------------------
DDPM의 띄엄띄엄한 걸음:
x_{t-1} = √(α_t) [x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ(x_t,t)] + σ_t z

VP-SDE의 이어진 끝값:
dx = [-0.5β(t)x - β(t)ε_θ(x,t)]dt + √β(t) dw

둘은 같다! 점수 s(x,t) = -ε(x,t)/√(1-ᾱ_t)

SDE 관점의 고갱이 좋은 점:
--------------------------
✓ 하나로 된 얼개(VE, VP, sub-VP 따위)
✓ 두루 쓰는 뽑개(SDE, ODE, 예측-바로잡기)
✓ 이론으로 살피기가 쉽다
✓ 새로운 얼개와 차례표
✓ 물리와 이어진다(브라운 운동, 랑주뱅)
""")

# 단순한 보여 주기
print("\nDemonstration: VE-SDE on 2D Gaussian")
print("-" * 80)

# 참 2차원 정규 분포
mean = torch.zeros(2)
cov = torch.eye(2)

def gaussian_score(x, t):
    """정규 분포의 닫힌 꼴 점수"""
    return -x  # N(0, I)에서 점수는 -x이다

# 흩어짐 터짐 확률 미분 방정식으로 뽑는다
vesde = VESDE(sigma_min=0.01, sigma_max=10.0)
samples, _ = euler_maruyama_sampler(vesde, gaussian_score, (500, 2), n_steps=500)

print(f"Generated samples shape: {samples.shape}")
print(f"Sample mean: {samples.mean(dim=0)}")
print(f"Sample std: {samples.std(dim=0)}")
print("✓ VE-SDE sampling successful!")

print("""
뽑기 꾀:
-------------------

1. EULER-MARUYAMA (EM):
   단순한 일차 확률 미분 방정식 풀개
   x_{i-1} = x_i + drift*dt + diffusion*√dt*z

2. PREDICTOR-CORRECTOR:
   미리 헤아리개: 오일러-마루야마 걸음
   고치개: 랑주뱅 마르코프 사슬 몬테카를로 걸음
   품질이 낫고 느리다

3. ODE 푸는개:
   확률 흐름 상미분 방정식을 쓴다
   걸음 크기를 맞추어 간다(RK45, DPM-Solver)
   더 빠르고 정해져 있다

4. DDIM-STYLE:
   때 걸음을 건너뛴다(마르코프가 아니다)
   훨씬 빠르다(걸음 10~50개)
   품질의 맞바꿈

SDE 갈래 고르기:
-----------------
- 흩어짐 터짐: 조건 없는 만들어 내기에 낫다
- 흩어짐 지키기: 조건 있거나 이끈 만들어 내기에 낫다
- DDPM의 적기와 맞는다
- 조건 주기가 쉽다

실제로는 흩어짐 지키기 확률 미분 방정식(DDPM)이 가장 흔하다!
""")

print("\n✓ Module 08 complete!")
print("Next: Apply to real images (Module 09)!")


if __name__ == "__main__":
    pass```

## 논의

점수 확률 미분 방정식 틀의 짜기는 이 마당에 자리 잡은 방식을 따른다. 코드 짜임이 모델 뜻매김과 익히기 논리를 갈라 놓아 부품을 하나씩 고치기 쉽다. 얼개 고르기는 만들어 내는 모델 무리가 많은 실험에서 얻은 배움을 담고 있다.

이 짜기의 핵심에는 수치의 안정을 꼼꼼히 다루기, 고르게 맞추기 재주를 제대로 쓰기, 효율 좋은 셈 결이 든다. 익히기 절차에는 잡음 차례표, 기울기 다루기, 이따금의 따지기가 들며 모두 품질 높은 결과를 내는 데 결정적이다.

이 단원은 이론의 개념이 실제 짜기로 어떻게 옮겨지는지 보이며 만들어 내는 모델의 더 넓은 틀과 이어진다. 여기서 보이는 재주는 만들어 내는 모델이 이룰 수 있는 것의 가장자리를 넓히는 더 앞선 변형과 넓힘을 이해하는 바탕이 된다.

## 연습문제

**연습문제 1.**
구체적인 자료 묶음으로 이 단원의 으뜸 셈을 좇아라. 큰 걸음마다 텐서 꼴을 적고 모든 차원이 서로 맞는지 확인하라.

??? success "연습문제 1 풀이"
    모델에 알맞은 꼴의 들임 묶음에서 시작한다. 층이나 함수 부르기마다 셈을 따라가며 바뀜 뒤 텐서 꼴을 적는다. 겹말기 층에서는 내놓기 차원 공식을 쓴다. 눈길 얼개에서는 물음, 열쇠, 값의 차원이 맞는지 확인한다. 마지막 내놓기 꼴이 바라던 목표 차원과 맞는지 굳힌다. 이 익힘은 자료가 얼개를 어떻게 흐르는지에 대한 직관을 쌓아 준다.

---

**연습문제 2.**
이 단원에 쓰인 손실 함수를 가려내고 모델 매개변수에 대한 기울기를 이끌어 내라. 왜 이 손실 함수가 이 일에 알맞은지 설명하라.

??? success "연습문제 2 풀이"
    손실 함수는 모델이 헤아린 값과 목표 사이의 어긋남을 잰다. 잡음 헤아리기에서는 평균 제곱 어긋남 손실 $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$을 쓰는데, 이것이 로그 가능도의 변분 아래 한계에 맞물리기 때문이다. 매개변수 $\theta$에 대한 기울기는 $-2(\epsilon - \epsilon_\theta) \nabla_\theta \epsilon_\theta$이며 헤아림 어긋남을 줄이는 방향을 가리킨다. 이 손실을 가장 작게 하는 것이 퍼짐 모델에서 자료 로그 가능도의 아래 한계를 가장 크게 하는 것과 같으므로 알맞다.

---

**연습문제 3.**
다른 잡음 차례표를 받쳐 주도록 이 짜기를 고쳐라(예컨대 선형에서 코사인으로, 또는 그 반대로). 두 차례표의 익히기 움직임과 표본 품질을 견주어라.

??? success "연습문제 3 풀이"
    두 차례표를 모두 짜고 각각으로 모델을 익힌다. $\bar{\alpha}_t = \cos^2\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$으로 뜻매김한 코사인 차례표는 선형 차례표 $\beta_t = \beta_{\min} + t(\beta_{\max} - \beta_{\min})/T$에 견주어 잡음이 더 매끄럽게 늘어난다. 손실 곡선을 좇고 일정한 사이마다 표본을 만든다. 코사인 차례표는 신호 대 잡음비가 더 완만하게 줄어 때 걸음에 걸쳐 배움 신호가 더 고르므로 흔히 더 좋은 결과를 낸다.
