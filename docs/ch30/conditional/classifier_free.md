# Classifier-Free Guidance(CFG)

조건부 확산 모델의 예측을 강화하는 한 방법은 조건부 예측과 무조건 예측의 차이를 확대하는 것이다. [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598)는 별도 분류기의 입력 그래디언트를 계산하는 대신 두 노이즈 예측을 결합한다. 가이던스 계수의 관례와 학습 때 사용한 무조건 표현을 일치시키는 것이 중요하다.

## 1. 하나의 계수 관례로 쓰기

조건 c의 예측을 εc, 무조건 예측을 ε∅라 하면 이 페이지에서는 다음 관례를 쓴다.

$$\epsilon_{\mathrm{CFG}}=\epsilon_{\varnothing}+w(\epsilon_c-\epsilon_{\varnothing})
=w\epsilon_c+(1-w)\epsilon_{\varnothing}$$

w=0이면 무조건, w=1이면 일반 조건부 예측이다. w>1은 두 예측 사이의 보간이 아니라 외삽이다. 다른 문헌의 (1+g)εc−gε∅는 g=w−1일 때 같은 식이며 **같은 기호·같은 수치를 그대로 쓰면 다른 가이던스**가 된다.

계수를 키우면 조건 일치도가 좋아질 수 있지만 과도한 색상, 세부 왜곡이나 다양성 감소가 나타날 수도 있다. 어떤 계수가 FID나 조건 일치도를 최적화하는지는 모델·데이터·샘플러에 따라 달라진다. 7.5 같은 수치를 모든 모델의 최적값으로 사용하지 않는다.

## 2. 스코어로 해석하기

정확한 조건부·무조건 스코어를 각각 sc, s∅라 하면 베이즈 규칙으로 다음을 얻는다.

$$s_c-s_{\varnothing}=\nabla_{x_t}\log p(c\mid x_t)$$

따라서 결합한 스코어는 s∅+w∇log p(c|xₜ)다. 분산 보존 확산에서 스코어와 노이즈 예측의 관계 s=−ε/√(1−ᾱₜ)를 사용하면 위 CFG 식이 나온다. 특히 다음 차이의 **부호와 스케일**에 유의한다.

$$\nabla_{x_t}\log p(c\mid x_t)=
-\frac{\epsilon_c-\epsilon_{\varnothing}}{\sqrt{1-\bar\alpha_t}}$$

이는 정확한 스코어의 관계이며 학습된 두 예측에는 근사 오차가 있다. 분류기 가이던스는 실제 분류기에서 입력 그래디언트를 구하지만 CFG는 예측 차이를 사용한다. 별도 분류기를 없애는 대신 조건부·무조건 예측 능력을 준비해야 한다.

## 3. 조건 드롭아웃으로 학습하기

```python
def training_step(model, x_0, condition, p_uncond=0.1):
    # 조건을 마구잡이로 떨구기
    mask = torch.rand(x_0.shape[0]) < p_uncond
    condition_input = condition.clone()
    condition_input[mask] = null_condition  # 예컨대 빈 임베딩
    
    # 여느 DDPM 익히기
    t = sample_timesteps(x_0.shape[0])
    noise = torch.randn_like(x_0)
    x_t = forward_diffusion(x_0, t, noise)
    
    noise_pred = model(x_t, t, condition_input)
    loss = mse_loss(noise_pred, noise)
    
    return loss
```

배치 일부의 조건을 무조건 표현으로 바꾸고 같은 노이즈 MSE를 계산한다. 이 방식으로 하나의 모델이 두 경우를 학습한다. 조건 드롭아웃은 이를 구현하는 일반적인 방법이지만, 별도의 조건부·무조건 모델을 준비하는 등 다른 학습 구성을 원천적으로 배제하지는 않는다.

위 함수는 독립 실행 코드가 아니다. torch, null_condition, sample_timesteps, forward_diffusion과 mse_loss가 별도로 정의되어야 한다. mask는 CPU에서 생성하므로 조건 텐서의 장치와 인덱싱 조건도 맞춰야 한다. p_uncond의 유효 범위와 시간 인덱스 장치를 함수가 검사하지 않는다.

클래스 조건은 전용 null 레이블, 텍스트 조건은 학습 때 사용한 빈 프롬프트 표현 등을 사용할 수 있다. 아무 0 벡터나 임의의 클래스 0이 자동으로 무조건 조건이 되는 것은 아니다. 추론에서도 학습과 같은 표현을 사용한다.

## 4. 샘플링 알고리즘

```
알고리즘: 분류기 없는 이끌기 뽑기
───────────────────────
들임: 모델 ε_θ, 조건 c, 이끌기 잣수 w

x_T ~ N(0, I)

for t = T, T-1, ..., 1:
    # 조건 없는 헤아림
    ε_uncond = ε_θ(x_t, t, ∅)
    
    # 조건 있는 헤아림
    ε_cond = ε_θ(x_t, t, c)
    
    # 이끈 헤아림
    ε̃ = ε_uncond + w * (ε_cond - ε_uncond)
    
    # ε̃으로 하는 DDPM/DDIM 고침
    x_{t-1} = sample_step(x_t, ε̃, t)

return x_0
```

```python
import torch

class CFGSampler:
    def __init__(self, model, schedule, guidance_scale=7.5):
        self.model = model
        self.schedule = schedule
        self.w = guidance_scale
    
    @torch.no_grad()
    def sample(self, shape, condition, null_condition, device, num_steps=50):
        """분류기 없는 이끌기로 뽑는다."""
        x = torch.randn(shape, device=device)
        
        timesteps = self.get_timesteps(num_steps)
        
        for t in timesteps:
            t_batch = torch.full((shape[0],), t, device=device)
            
            # 두 헤아림을 모두 셈한다
            # 고르기 1: 피드포워드 두 번
            eps_uncond = self.model(x, t_batch, null_condition)
            eps_cond = self.model(x, t_batch, condition)
            
            # 고르기 2: 묶어 하기(더 효율이 좋다)
            # x_double = torch.cat([x, x])
            # t_double = torch.cat([t_batch, t_batch])
            # c_double = torch.cat([null_condition, condition])
            # eps_both = self.model(x_double, t_double, c_double)
            # eps_uncond, eps_cond = eps_both.chunk(2)
            
            # 이끌기를 쓴다
            eps_guided = eps_uncond + self.w * (eps_cond - eps_uncond)
            
            # DDIM 스텝
            x = self.ddim_step(x, eps_guided, t)
        
        return x
    
    def ddim_step(self, x, eps, t):
        """DDIM 고침 스텝 하나."""
        alpha_bar_t = self.schedule.alpha_bars[t]
        alpha_bar_prev = self.schedule.alpha_bars[t-1] if t > 0 else 1.0
        
        # x_0을 헤아린다
        x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
        x0_pred = torch.clamp(x0_pred, -1, 1)
        
        # x_t 쪽 방향
        dir_xt = torch.sqrt(1 - alpha_bar_prev) * eps
        
        # 다음 표본
        x_prev = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt
        
        return x_prev
```

CFGSampler는 매 단계 조건부·무조건 모델을 각각 호출한다. 하지만 get_timesteps가 정의되어 있지 않아 그대로는 sample을 완료할 수 없다. 또한 ddim_step은 다음으로 선택한 시간을 받지 않고 **항상 t−1의 계수**를 사용한다. 시간 목록을 건너뛰어도 다음 입력 상태를 선택한 시점으로 정확히 이동시키는 구현이 아니다.

t=0이면 alpha_bar_prev를 Python 실수 1.0으로 놓고 torch.sqrt에 넘기므로 Tensor를 요구하는 호출에서 실패할 수 있다. model.eval()도 호출하지 않으며 모델·스케줄·조건의 장치 이동은 호출자에게 맡긴다. 이 블록은 학습부터 빠른 생성까지 완성된 실행 예제가 아니라 핵심 결합을 보여 주는 미완성 골격으로 읽는다.

## 5. 계산 비용과 재사용 가능한 항목

별도 순방향으로 계산하면 단계마다 두 번 호출한다. B개 조건부 입력과 B개 무조건 입력을 연결해 2B 배치로 한 번에 계산할 수도 있지만, 한 번의 B 배치만큼 계산·메모리가 든다는 뜻은 아니다.

조건이 고정되어 있으면 텍스트 인코더의 조건 표현 등은 재사용할 수 있다. 그러나 무조건 **노이즈 예측 ε∅(xₜ,t)는 상태와 시간에 의존**하므로 첫 단계 결과를 전체 단계에 그대로 캐시할 수 없다. 가이던스 결과를 다른 모델에 증류하는 방법은 별도의 학습과 평가가 필요하다.

## 6. 변형과 평가

시간별 w(t)를 사용하면 생성 구간에 따라 가이던스 강도를 조절할 수 있다. 시간 인덱스가 증가할수록 노이즈가 커지는지, 실제 역과정이 어느 방향으로 진행하는지를 함께 명시해야 “초기에 강화한다”는 설명이 모호하지 않다.

네거티브 프롬프트 예측 εneg를 무조건 예측 대신 쓰는 경우 결합은 εneg+w(εpos−εneg)다. 이때 기준 예측은 더 이상 무조건 예측이 아니며, 원하지 않는 속성이 반드시 제거된다는 보장은 없다. 이 페이지의 샘플러는 텍스트 인코더나 해당 편집 절차를 구현하지 않는다.

평가에는 고정한 생성 조건·표본 수에서 조건 일치도와 분포의 다양성을 함께 사용한다. FID, 분류 정확도, 텍스트·이미지 유사도와 사람 평가가 서로 다른 정보를 줄 수 있다. 측정하지 않은 계수별 FID·CLIP 순위나 보편적인 품질 향상을 제시하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
ε∅+w(εc−ε∅)와 (1+g)εc−gε∅의 계수 관계를 구하라.

</div>

??? success "연습문제 1 풀이"

    첫 식은 wεc+(1−w)ε∅이므로 w=1+g다. 첫 관례의 w=1은 일반 조건부 예측이고 두 번째 관례의 g=0과 같다. 같은 수치 7.5를 두 관례에 그대로 넣으면 서로 다른 결합이다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
조건부·무조건 노이즈 예측 차이로 분류기 로그확률의 그래디언트를 나타내라.

</div>

??? success "연습문제 2 풀이"

    베이즈 규칙에서 ∇log p(c|xₜ)=sc−s∅다. s=−ε/√(1−ᾱₜ)를 대입하면 −(εc−ε∅)/√(1−ᾱₜ)다. 따라서 스코어 가이던스 s∅+w(sc−s∅)는 노이즈 공간에서도 같은 w의 CFG 결합이 된다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
무조건 예측을 한 번 계산해 모든 시간에 재사용할 수 없는 이유는 무엇인가?

</div>

??? success "연습문제 3 풀이"

    ε∅는 조건만 빠졌을 뿐 xₜ와 t를 입력으로 받는다. 상태와 시간이 바뀌면 예측도 바뀌므로 일반적으로 캐시한 한 텐서로 대체할 수 없다. 조건 인코더의 고정 표현을 캐시하는 것과 구분한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
CFGSampler를 완성된 50단계 샘플러로 볼 수 없는 이유를 세 가지 이상 설명하라.

</div>

??? success "연습문제 4 풀이"

    get_timesteps가 없고, 건너뛴 다음 시간 대신 t−1 계수를 사용하며, 마지막 단계에서 Python 실수를 torch.sqrt에 넘긴다. 평가 모드 설정과 모델·조건·스케줄의 장치 일치도 필요하다. 핵심 CFG 식이 맞아도 전체 생성 루프가 완성된 것은 아니다.



## 정리하며

CFG는 무조건·조건부 노이즈 예측을 결합하며 계수 관례를 일치시켜야 한다. 조건 드롭아웃과 null 표현은 두 예측의 학습을 연결한다. 시간마다 달라지는 무조건 예측을 고정 캐시하지 않으며, 결합식과 완전한 샘플러의 구현 범위를 구분한다.
