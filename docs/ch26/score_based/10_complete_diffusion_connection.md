# 온전한 퍼짐 이음

이 단원은 요즘 만들어 내는 모델의 핵심 부품인 온전한 퍼짐 이음을 짠다. 여기서 보이는 개념과 재주를 알면 퍼짐 모델과 점수 바탕 만들어 내는 방법을 다루는 데 꼭 필요한 앎을 얻는다. 이 짜기는 또렷함과 실제 쓸모의 균형을 맞추어 배우기에도 실험하기에도 알맞다.

## 1. 코드

```python
"""
단원 10: 퍼짐 모델과의 온전한 이음
=================================================

어려움: 앞섬(모아 엮기)
시간: 2~3시간
미리 알 것: 앞의 모든 단원

학습 목표:
- 점수 바탕 모델과 퍼짐 모델이 온전히 같음을 이해한다
- 모든 개념이 어떻게 하나로 묶이는지 본다
- 요즘 변형을 이해한다
- 나아간 주제로 가는 길

이 단원은 여태 배운 모든 것을 하나로 묶는다!

지은이: 이성철 @ 연세대학교
"""

import torch
import numpy as np

# ========================================================================
# 메인
# ========================================================================

print("=" * 80)
print("MODULE 10: COMPLETE DIFFUSION CONNECTION")
print("=" * 80)

print("""
온전한 그림: 베이즈 추론에서 퍼짐까지
==========================================================

개념의 온 여정을 짚어 보자.

1. 베이즈 미룸(01_Bayesian_Inference 단원):
   -----------------------------------------------------
   Problem: p(θ|D) = p(D|θ)p(θ) / p(D)
   어려움: p(D) = ∫ p(D|θ)p(θ) dθ을 셈할 수 없다
   
   필요한 풀이: 고르게 맞추지 않은 분포에서 뽑기!

2. 점수 함수(1단원):
   -----------------------------
   고갱이 눈썰미: s(x) = ∇_x log p(x)
   
   성질:
   ✓ 확률 높은 쪽을 가리킨다
   ✓ 고르게 맞추기가 필요 없다!
   ✓ 봉우리에서 0이다
   
   그런데 자료만으로 어떻게 셈하는가?

3. 점수 맞추기(2단원):
   ----------------------------
   표본에서 s_θ(x) ≈ ∇_x log p_data(x)을 배운다
   
   고갱이 재주: 잡음 지우기 점수 맞추기(DSM)
   - Add noise: x̃ = x + σε
   - 헤아리는 법을 배운다: s_θ(x̃) ≈ -ε/σ
   
   이음: 잡음 없애기가 곧 베이즈 추론이다!
   뒷분포 p(x|x̃) → 점수가 잡음 지우는 법을 알려 준다

4. 랑주뱅 움직임(3단원):
   -------------------------------
   배운 점수를 써서 표본을 뽑는다.
   x_{t+1} = x_t + ε s_θ(x_t) + √(2ε) z_t
   
   p_data(x)으로 모인다!
   모든 퍼짐 뽑기의 바탕

5. 여러 자의 점수(7단원):
   --------------------------------
   문제: σ 하나로는 어디서나 통하지 않는다
   풀이: 여러 잡음 수준의 s_θ(x, σ_i)을 배운다
   
   이것이 퍼짐에서 때 차원이 된다!

6. 이어진 꼴(8단원):
   ------------------------------------
   SDE는 모든 것을 이어지게 만든다.
   - Forward: dx = f(x,t)dt + g(t)dw
   - Reverse: dx = [f - g²∇log p_t]dt + g dw̄
   
   모든 변형을 아우르는 틀!

7. 그림 만들어 내기(9단원):
   -----------------------------
   그림을 위한 U-Net 얼개
   Training = DSM at multiple times
   Sampling = Reverse diffusion
   
   최고 수준의 만들어 내는 모델!

8. 퍼짐 모델(이 단원):
   --------------------------------
   모든 것이 하나로 묶인다!

온전한 같음 관계:
====================

점수 바탕 관점              ↔  퍼짐 관점
-----------------                ----------------
점수 함수 s(x,t)         ↔  잡음 예측 ε_θ(x,t)
                                 Relation: s(x,t) = -ε/√(1-ᾱ_t)

잡음 없애는 점수 맞추기      ↔  잡음 헤아리기 손실
                                 Both: ||ε - ε_θ||²

랑주뱅 움직임             ↔  뒤 퍼짐
                                 같은 뽑기 절차!

식힘 랑주뱅            ↔  여러 걸음 잡음 없애기
                                 차츰 잡음을 없앤다

흩어짐 터짐 확률 미분 방정식                       ↔  잡음 조건 모델
흩어짐 지키기 확률 미분 방정식                       ↔  DDPM 적기

확률 흐름 상미분 방정식         ↔  DDIM 뽑기
                                 정해진 만들어 내기

온전한 DDPM 꼴:
=========================
""")

class DDPM:
    """
    잡음 없애는 퍼짐 확률 모델
    
    이것이 여태 배운 모든 것을 하나로 만든다!
    """
    def __init__(self, n_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.n_timesteps = n_timesteps
        
        # 선형 차례표(다른 것도 쓸 수 있다)
        self.betas = torch.linspace(beta_start, beta_end, n_timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # 쓸모 있는 양을 미리 셈한다
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
    
    def q_sample(self, x_0, t, noise=None):
        """
        앞 과정: 잡음 더하기
        
        q(x_t|x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t)I)
        
        This is: x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
        
        점수 맞추기와의 이음:
        - 이것이 잡음 없애는 점수 맞추기의 "잡음 더하기"이다!
        - t가 다르면 잡음 층 σ도 다르다
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t]
        
        # 퍼뜨리기에 맞게 꼴을 바꾼다
        while len(sqrt_alpha_bar.shape) < len(x_0.shape):
            sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.unsqueeze(-1)
        
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
    
    def training_loss(self, model, x_0):
        """
        익히기 목표
        
        L = 𝔼_t 𝔼_ε ||ε - ε_θ(x_t, t)||²
        
        DSM과의 이음:
        - 이것이 바로 잡음 없애는 점수 맞추기이다!
        - DSM에서 t가 다르면 σ도 다르다
        - ε을 헤아리는 것은 점수를 헤아리는 것과 같다
        
        이끌어 내기:
        Score s(x_t,t) = ∇log p(x_t|x_0)
                       = -ε / √(1-ᾱ_t)
        
        So: ε = -√(1-ᾱ_t) * score
        """
        # 아무 때 걸음
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.n_timesteps, (batch_size,))
        
        # 잡음을 더한다(앞 과정)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        
        # 잡음을 헤아린다
        predicted_noise = model(x_t, t)
        
        # 손실
        return torch.mean((noise - predicted_noise) ** 2)
    
    def p_sample(self, model, x_t, t):
        """
        뒤 과정: 잡음 없애기 걸음 하나
        
        p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), σ_t²I)
        
        여기서 각 기호는 다음과 같다.
        μ_θ = (1/√α_t)[x_t - (β_t/√(1-ᾱ_t))ε_θ(x_t,t)]
        
        랑주뱅과의 이음:
        - 이것이 식힘 랑주뱅의 한 걸음이다!
        - ε_θ이 점수 방향을 준다
        - μ_θ이 랑주뱅 고침이다
        """
        # 잡음을 헤아린다
        predicted_noise = model(x_t, t)
        
        # 계수를 뽑는다
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]
        
        # 퍼뜨리기에 맞게 꼴을 바꾼다
        while len(alpha_t.shape) < len(x_t.shape):
            alpha_t = alpha_t.unsqueeze(-1)
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
            beta_t = beta_t.unsqueeze(-1)
        
        # 뒤 분포의 평균
        mean = (1 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
        )
        
        # 잡음을 더한다(t=0만 빼고)
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            variance = beta_t
            return mean + torch.sqrt(variance) * noise
        else:
            return mean
    
    def sample(self, model, shape):
        """
        온전한 뽑기: 잡음에서 만들어 내기
        
        This is:
        1. 식힘 랑주뱅 움직임
        2. 뒤 확률 미분 방정식과 상미분 방정식
        3. 잡음 없애는 퍼짐
        
        모두 같은 것이다!
        """
        # 순수 잡음에서 시작한다
        x = torch.randn(shape)
        
        # 뒤 퍼짐
        for t in reversed(range(self.n_timesteps)):
            t_batch = torch.ones(shape[0], dtype=torch.long) * t
            x = self.p_sample(model, x, t_batch)
        
        return x

print("DDPM class defined!")

print("""
요즘의 변형과 넓힘:
==============================

1. DDIM(잡음 지우기 퍼짐 감춤 모델):
   - 정해진 대로 뽑기(ODE)
   - 때 걸음을 건너뛴다
   - 빠른 만들어 내기(1000걸음 대신 50걸음)
   - DDPM과 같은 익히기

2. 나아진 DDPM:
   - 배운 흩어짐
   - 더 나은 잡음 짜임(코사인)
   - 나아진 얼개

3. 이끈 퍼짐:
   - 가름개 이끌기: 가름개의 기울기를 쓴다
   - 가름개 없는 이끌기: 조건 있는 것과 없는 것을 함께 익힌다
   - 최고 수준의 그림 품질

4. 숨은 자리 퍼짐(스테이블 디퓨전):
   - 숨은 자리에서 퍼짐을 돌린다(VAE)
   - 훨씬 빠르고 싸다
   - CLIP으로 글 조건 주기

5. 이어달리기 퍼짐:
   - 해상도마다 여러 모델
   - 64x64 → 256x256 → 1024x1024
   - 해상도가 높을 때 품질이 낫다

6. 동영상 퍼짐:
   - 때 차원으로 넓힌다
   - 3차원 U-Net
   - 때 눈길

7. 그 밖의 갈래:
   - 소리: WaveGrad, DiffWave
   - 글: Diffusion-LM
   - 3차원: Point-E, Shap-E
   - 분자: 분자 만들어 내기

점수 바탕 관점:
===========================
점수로 퍼짐을 이해하면 다음을 얻는다.

✓ 이론의 또렷함:
  - 왜 되는가(뽑기 이론)
  - 랑주뱅 마르코프 사슬 몬테카를로와의 이음
  - 베이즈 풀이

✓ 두루 쓰는 얼개:
  - 새 확률 미분 방정식을 짠다
  - 새로운 뽑기 절차
  - 섞은 방식

✓ 하나로 된 관점:
  - 점수 맞추기는 곧 퍼짐 익히기다
  - 랑주뱅은 곧 퍼짐 뽑기다
  - 여러 자는 곧 때 차원이다

✓ EXTENSIONS:
  - 점수 바탕은 정규가 아닌 잡음도 다룰 수 있다
  - 너그러운 차례표
  - 다른 목표

온 여정 지도:
====================

베이즈 미룸(01_Bayesian_Inference)
    ↓ (고르게 하지 않고 뽑아야 한다)
점수 함수(1단원)
    ↓ (자료에서 어떻게 배우는가?)
점수 맞추기 / DSM(2단원)
    ↓ (How to sample?)
랑주뱅 움직임(3단원)
    ↓ (여러 자가 필요하다)
여러 자의 점수(7단원)
    ↓ (Continuous formulation)
점수 바탕 SDE(8단원)
    ↓ (Apply to images)
U-Net과 익히기(9단원)
    ↓ (Equivalent formulation)
DDPM / 요즘 퍼짐(단원 10) ← 여기이다!

손에 잡히는 권함:
=========================

연구를 위해:
- 이론은 점수 바탕 관점에서 시작하라
- 짜기는 퍼짐 적기를 쓰라
- 필요에 따라 섞어 쓰라

쓰임을 위해:
- 미리 익힌 모델을 쓴다(스테이블 디퓨전 따위)
- 네 마당에 맞게 미세 조정하라
- 벌레를 잡으려면 이론을 이해하라

더 배우려면:
- 본디 논문(DDPM, 점수 바탕 SDE)
- Lilian Weng의 블로그
- Hugging Face diffusers 꾸러미
- Song Yang의 자료

우리가 이룬 것:
=======================
✓ 첫 원리에서 퍼짐 모델을 세웠다
✓ 베이즈 추론을 요즘 만들어 내는 모델과 이었다
✓ 온전한 이론 틀을 이해했다
✓ 실제 짜기의 세부를 배웠다
✓ 모든 단원에 걸친 이음을 보았다

이제 퍼짐 모델을 깊이 이해한다!

앞으로의 방향:
=================
- 한결같음 모델(한 걸음 만들어 내기)
- 흐름 맞추기(퍼짐을 대신하는 길)
- 퍼짐 변환기(DiT)
- 영상과 3차원 만들어 내기
- 다스릴 수 있는 만들어 내기
- 더 빠른 뽑기 방법
- 더 나은 얼개
- 새로운 쓰임새

이 마당은 빠르게 바뀌고 있다. 이제 이해하고 보탤 바탕이 있다!
""")

print("\n" + "=" * 80)
print("FINAL SUMMARY: THE COMPLETE UNIFIED VIEW")
print("=" * 80)

print("""
서로 같은 세 가지 꼴:
------------------------------

1. SCORE-BASED:
   - 점수 s_θ(x,t) = ∇log p_t(x)을 배운다
   - 랑주뱅 움직임으로 뽑는다
   - 잡음 수준에 걸쳐 식힌다

2. DIFFUSION-BASED:
   - 앞으로: 잡음을 차츰 더한다
   - 뒤로: 차츰 잡음을 없앤다
   - 잡음 예측 ε_θ(x,t)을 배운다

3. SDE-BASED:
   - Forward SDE: dx = f dt + g dw
   - 거꾸로 가는 SDE: dx = [f - g²∇log p_t]dt + g dw̄
   - 이어진 때로 적기

셋은 모두 같다!

고갱이 관계:
-----------------
s(x,t) = -ε_θ(x,t) / √(1-ᾱ_t)         (score ↔ noise)
DSM = Noise prediction loss            (training)
Langevin = Reverse diffusion           (sampling)
여러 자는 곧 때 조건 주기            (얼개)

한가운데 눈썰미:
---------------
Denoising = Bayesian posterior inference
잡음 지우기를 배우는 것이 곧 점수를 배우는 것이다
거듭 잡음을 지우는 것이 곧 점수로 표본을 뽑는 것이다

이것은 다음을 잇는다.
- 고전 통계(베이즈)
- 뽑기 이론(랑주뱅)
- 깊은 배움(신경 그물)
- 확률 흐름(SDE)
- 요즘 만들어 내는 모델(퍼짐)

아름다운 하나 됨! 🎯
""")

print("\n" + "=" * 80)
print("CONGRATULATIONS!")
print("=" * 80)

print("""
베이즈 추론에서 최고 수준의 퍼짐 모델에 이르는
여정을 마쳤다!

이제 다음을 이해한다.
✓ 퍼짐이 왜 되는가(점수 이론)
✓ 퍼짐 모델을 어떻게 익히는가(DSM)
✓ 어떻게 표본을 뽑는가(랑주뱅과 거꾸로 가는 SDE)
✓ 요즘 얼개(U-Net과 때)
✓ 이론 바탕(SDE, 포커-플랑크)

이 앎으로 다음을 할 수 있다.
- 퍼짐 논문을 읽고 이해한다
- 모델을 바닥부터 짠다
- 익히기 문제를 잡는다
- 새로운 방식을 짠다
- 연구에 보탠다

깊은 배움에 쏟은 정성에 고맙다! 🎓📊🚀

다음 세대의 만들어 내는 인공 지능을 세울 준비가 되었는가?
여정은 이어진다...
""")

print("=" * 80)
print("✓ MODULE 10 COMPLETE - SERIES FINALE!")
print("✓ ALL 10 MODULES COMPLETE!")
print("=" * 80)


if __name__ == "__main__":
    pass```

## 2. 논의

온전한 퍼짐 이음의 짜기는 이 마당에 자리 잡은 방식을 따른다. 코드 짜임이 모델 뜻매김과 익히기 논리를 갈라 놓아 부품을 하나씩 고치기 쉽다. 얼개 고르기는 만들어 내는 모델 무리가 많은 실험에서 얻은 배움을 담고 있다.

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

## 정리하며

**다룬 것** — 온전한 퍼짐 이음

온전한 퍼짐 이음의 짜기는 이 마당에 자리 잡은 방식을 따른다.

고갱이 갈래는 `DDPM`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
