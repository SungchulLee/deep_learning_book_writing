# 스코어 기반 SDE 프레임워크

스코어 SDE 관점은 정방향 노이즈 과정, 시간별 주변 스코어와 생성용 역과정을 하나의 틀로 연결한다. 이 페이지는 VE·VP 계수와 오일러 샘플러를 정의한다. 학습된 이미지 생성 모델은 없으며, 아래 가우시안 시연의 시간 부호와 스코어·초기 분포를 함께 확인해야 한다.

## 1. 코드와 기존 출력

```python
"""
점수 바탕 확률 미분 방정식
===========================

어려움: 나아간 단계
시간: 3~4시간
미리 알 것: 단원 01-07, 기본 확률 미분 방정식 앎

학습 목표:
- 이어진 때의 확산 틀을 이해한다
- 흩어짐 터짐과 흩어짐 지키기 확률 미분 방정식을 짠다
- 띄엄띄엄한 DDPM을 이어진 확률 미분 방정식과 잇는다

핵심 방정식:
dx = f(x,t)dt + g(t)dw  (forward SDE)
dx = [f(x,t) - g(t)²∇log p_t(x)]dt + g(t)dw̄  (reverse SDE)

지은이: 이성철 @ 연세대학교
"""

import torch
import numpy as np
# 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
torch.manual_seed(0)

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
- g(t): 확산 계수
- dw: 브라운 액션

거꾸로 가는 SDE(잡음을 없앤다):
dx = [f(x,t) - g(t)²∇log p_t(x)]dt + g(t)dw̄

핵심: 점수 ∇log p_t(x)이 뒤 과정에 나타난다!
이것을 뉴럴 넷트웍이 배운다: s_θ(x,t) ≈ ∇log p_t(x)

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
        
        # 고침: 퍼뜨리기에 맞게 확산의 꼴을 바꾼다
        # 확산의 꼴은 (배치,)이지만 점수의 꼴은 (배치, 차원)이다
        # 퍼뜨리기를 위해 확산의 꼴을 (배치, 1)으로 만들어야 한다
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
        
        # 고침: 잡음 항의 확산 퍼뜨리기를 다룬다
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

SDE 관점의 핵심 좋은 점:
--------------------------
✓ 하나로 된 구조(VE, VP, sub-VP 따위)
✓ 두루 쓰는 뽑개(SDE, ODE, 예측-바로잡기)
✓ 이론으로 살피기가 쉽다
✓ 새로운 구조와 차례표
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
   단순한 일차 확률 미분 방정식 솔버
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

SDE 클래스 고르기:
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
    pass
```

??? note "전체 출력 (114줄)"

    ```
    MODULE 08: Score-Based SDE Framework
    ================================================================================

    이어진 때의 꼴:
    --------------------------
    띄엄띄엄한 걸음 t=0,1,2,...,T 대신
    이어진 때 t ∈ [0, T]를 쓴다

    앞으로 가는 SDE(잡음을 더한다):
    dx = f(x,t)dt + g(t)dw

    여기서 각 기호는 다음과 같다.
    - f(x,t): 떠돎 계수
    - g(t): 확산 계수
    - dw: 브라운 액션

    거꾸로 가는 SDE(잡음을 없앤다):
    dx = [f(x,t) - g(t)²∇log p_t(x)]dt + g(t)dw̄

    핵심: 점수 ∇log p_t(x)이 뒤 과정에 나타난다!
    이것을 뉴럴 넷트웍이 배운다: s_θ(x,t) ≈ ∇log p_t(x)

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

    SDE 관점의 핵심 좋은 점:
    --------------------------
    ✓ 하나로 된 구조(VE, VP, sub-VP 따위)
    ✓ 두루 쓰는 뽑개(SDE, ODE, 예측-바로잡기)
    ✓ 이론으로 살피기가 쉽다
    ✓ 새로운 구조와 차례표
    ✓ 물리와 이어진다(브라운 운동, 랑주뱅)


    Demonstration: VE-SDE on 2D Gaussian
    --------------------------------------------------------------------------------
    Generated samples shape: torch.Size([500, 2])
    Sample mean: tensor([7.6109e+26, 7.4300e+27])
    Sample std: tensor([1.1289e+29, 1.1826e+29])
    ✓ VE-SDE sampling successful!

    뽑기 꾀:
    -------------------

    1. EULER-MARUYAMA (EM):
       단순한 일차 확률 미분 방정식 솔버
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

    SDE 클래스 고르기:
    -----------------
    - 흩어짐 터짐: 조건 없는 만들어 내기에 낫다
    - 흩어짐 지키기: 조건 있거나 이끈 만들어 내기에 낫다
    - DDPM의 적기와 맞는다
    - 조건 주기가 쉽다

    실제로는 흩어짐 지키기 확률 미분 방정식(DDPM)이 가장 흔하다!


    ✓ Module 08 complete!
    Next: Apply to real images (Module 09)!
    ```

## 2. VE와 VP의 정방향 계수

정방향 과정은 $dX_t=f(X_t,t)dt+g(t)dW_t$다. VE 구현은 f=0이고 $\sigma(t)=\sigma_{min}(\sigma_{max}/\sigma_{min})^t$를 사용해 다음 계수를 반환한다.

$$g(t)=\sigma(t)\sqrt{2\log(\sigma_{max}/\sigma_{min})}$$

출력의 설명에 나오는 σ²의 선형 스케줄과 달리 **코드는 σ가 지수적으로 변하는 스케줄**이다. 깨끗한 초기 데이터에서 시작하면 조건부 추가 분산은 σ(t)²−σmin²다. 처음부터 σmin으로 평활화된 초기 관례라면 총 평활화 분산 σ(t)²로 해석할 수 있다.

VP는 β(t)=βmin+t(βmax−βmin)와 f=−β(t)x/2, g=√β(t)를 사용한다. 초기 공분산이 I일 때 이를 유지하지만 임의 데이터의 분산이 시간 내내 일정하다는 뜻은 아니다. 이 구현의 시간 구간은 0~1이다.

두 클래스는 계수만 반환한다. 학습 데이터에 정방향 노이즈를 더하는 함수나 스코어 학습 루프는 이 페이지에 없다. 계수의 반환 모양도 일치하지 않아 배치 시간과 데이터 차원에 맞는 브로드캐스팅이 필요하다. 현재 샘플러는 N×d 상태와 길이 N의 시간에 맞춰 작성되어 있으며 이미지 차원으로 자동 일반화되지 않는다.

## 3. 역시간 식과 샘플러의 부호

상태에 의존하지 않는 g에 대한 역시간 드리프트는 $b=f-g^2s$다. 이 식에서 t는 감소하므로 양의 h로 t→t−h를 적분할 때 다음처럼 이동한다.

$$x_{t-h}\approx x_t-h\,[f(x_t,t)-g(t)^2s(x_t,t)]+g(t)\sqrt h\,z$$

`reverse_sde` 메서드가 반환하는 드리프트는 위 b와 맞지만, `euler_maruyama_sampler`는 t를 줄이면서 **+drift·dt**를 사용한다. 따라서 역시간 적분의 부호가 맞지 않는다. 이름이 reverse라고 해서 호출하는 적분기의 시간 관례가 자동으로 맞는 것은 아니다.

샘플러는 언제나 CPU의 N(0,I)에서 시작한다. 이는 일반 VE 종단 주변 분포의 스케일과 맞지 않는다. 뉴럴 넷트웍 스코어를 넣으면 no_grad·모드 처리를 하지 않아 미분 그래프가 쌓일 수도 있다. trajectory는 초기와 갱신 1,101,… 시점을 저장하므로 전체 단계나 마지막 상태를 모두 포함하는 배열이 아니다.

## 4. 가우시안 시연의 불일치

데모의 `gaussian_score(x,t)`는 항상 −x다. 그러나 VE 정방향에서 초기 데이터가 N(0,I)이면 시간 t의 분산은 1+σ(t)²−σmin²이고 올바른 주변 스코어는 다음과 같다.

$$s(x,t)=-\frac{x}{1+\sigma(t)^2-\sigma_{min}^2}$$

따라서 고정 −x는 VE의 시간별 주변 스코어와 다르다. σmax=10일 때 종단 분산도 약 101이므로 초기 N(0,I)와 맞지 않는다. 여기에 역시간 부호 오류도 있으므로 출력의 'VE-SDE sampling successful'은 정확한 역과정 구현의 검증이 아니다. 정의해 둔 mean·cov 변수도 실제 시연 계산에 사용하지 않는다.

## 5. 확률 흐름 ODE의 의미

관련된 확률 흐름 ODE는 다음과 같다.

$$dX_t=\left[f(X_t,t)-\frac12g(t)^2s(X_t,t)\right]dt$$

정확한 주변 스코어와 적절한 정칙 조건, 같은 초기 분포에서 SDE와 **같은 시간별 주변 분포**를 갖는다. 개별 경로가 같다는 뜻은 아니다. 초기 잠재 샘플은 여전히 무작위이며 '결정적'이라는 말은 그 초기값이 주어진 뒤의 ODE 궤적을 뜻한다. 유일성 조건과 수치 오차를 고려해야 하고 항상 더 빠르거나 더 높은 품질을 보장하지 않는다.

연속 변화변수 공식으로 로그 밀도를 평가할 수 있지만 발산 적분, 사전분포와 수치 오차 처리가 필요하다. 이 코드에는 ODE 솔버나 우도 계산이 없다. 관련 정의는 [스코어 SDE 논문](https://arxiv.org/abs/2011.13456)을 참고한다.

## 6. DDPM 및 솔버와의 관계

출력에 적힌 DDPM 역평균의 앞 계수 √αt는 통상적인 노이즈 예측 파라미터화의 **1/√αt**와 다르다. 통상적인 식은 다음과 같다.

$$\mu_\theta(x_t,t)=\frac1{\sqrt{\alpha_t}}
\left(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)$$

주변 스코어와 노이즈 예측의 관계는 $s_\theta=-\epsilon_\theta/\sqrt{1-\bar\alpha_t}$다. 따라서 VP 역드리프트 f−βs에 대입할 때는 노이즈 예측의 부호·분모를 함께 변환해야 한다. 출력의 −βεθ 형태와 '둘은 같다'는 문장을 그대로 식의 동일성으로 읽으면 안 된다.

오일러–마루야마, 예측–보정, 확률 흐름 ODE, DDIM 계열은 서로 관련되지만 같은 수치 방법은 아니다. 오일러–마루야마의 수렴 차수는 강·약 오차와 계수 조건에 따라 구분해야 하며 무조건 '일차 솔버'라고 부르면 불명확하다. VE는 무조건 생성에, VP는 조건부 생성에 더 좋다는 고정 구분도 없다. 실제 성능과 비용은 모델·훈련·조건·솔버 설정에서 평가해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
VE에서 X₀~N(0,I)이고 σ(0)=σmin이면 시간별 주변 스코어를 구하라.

</div>

??? success "연습문제 1 풀이"

    추가 노이즈 분산은 σ(t)²−σmin²이므로 주변 공분산은 (1+σ(t)²−σmin²)I다. 따라서 스코어는 −x/(1+σ(t)²−σmin²)다. 고정 −x와 같지 않으며 종단 샘플의 분산도 이 값에 맞춰야 한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
정확한 역드리프트 f−g²s를 반환하는 메서드에 양의 dt를 곱해도 되는가?

</div>

??? success "연습문제 2 풀이"

    t가 감소하는 적분이면 그대로 +dt를 곱하면 안 된다. 음의 시간 증분을 쓰거나 감소 시간에 대한 드리프트를 별도로 반전해야 한다. 잡음의 크기는 양의 시간 폭의 제곱근으로 계산한다. 현재 샘플러는 시간 표지만 감소시키고 상태 드리프트 부호를 반전하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
확률 흐름 ODE가 결정적이라는 말과 SDE와 같은 주변 분포라는 말을 구분하라.

</div>

??? success "연습문제 3 풀이"

    초기 상태를 고정하면 ODE의 해는 정칙·유일성 조건 아래 결정된다. 생성하려면 초기 상태 자체를 분포에서 뽑으므로 출력 분포는 여전히 존재한다. 정확한 스코어와 일치하는 초기 조건에서 시간별 주변 분포가 SDE와 같다는 뜻이지 경로·수치 오차·실행시간이 같다는 뜻은 아니다. 이 페이지에는 ODE 구현이나 선형·코사인 학습 비교가 없다.



## 정리하며

SDE 프레임워크에서는 정방향 계수, 시간별 주변 스코어, 종단 초기분포와 역시간 적분 부호가 함께 맞아야 한다. 이 시연은 VE에 고정 가우시안 스코어와 잘못된 적분 부호를 사용한다. 확률 흐름 ODE와 DDPM의 연결도 계수와 파라미터화 조건을 명시해 읽는다.
