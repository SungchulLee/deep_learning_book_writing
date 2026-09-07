# 점수 함수 바탕

이 단원은 요즘 만들어 내는 모델의 핵심 부품인 점수 함수 바탕을 짠다. 여기서 보이는 개념과 재주를 알면 퍼짐 모델과 점수 바탕 만들어 내는 방법을 다루는 데 꼭 필요한 앎을 얻는다. 이 짜기는 또렷함과 실제 쓸모의 균형을 맞추어 배우기에도 실험하기에도 알맞다.

## 코드

```python
"""
단원 01: 점수 함수 바탕
==================================

어려움: 처음
시간: 2~3시간
미리 알아 둘 것: 01_Bayesian_Inference(특히 켤레 앞분포와 최대 사후 어림)

학습 목표:
-------------------
1. 점수 함수가 무엇이며 왜 중요한지 이해한다
2. 점수 함수를 베이즈 사후 추론과 잇는다
3. 단순한 분포의 점수를 닫힌 꼴로 셈한다
4. 2차원에서 점수 마당을 그려 본다
5. 점수가 고르게 맞추는 상수를 피하는 까닭을 이해한다

수학 바탕:
-----------------------
Definition:
    점수 함수 s(x)는 로그 확률의 기울기다.
    
    s(x) = ∇_x log p(x)
    
베이즈 미룸과의 이음:
    베이즈 미룸에서 우리는 다음을 배웠다.
    p(θ|D) = p(D|θ)p(θ) / p(D)
    
    분모 p(D) = ∫ p(D|θ)p(θ) dθ은 셈할 수 없다!
    
    그러나 뒷분포의 점수는 다음과 같다.
    ∇_θ log p(θ|D) = ∇_θ log[p(D|θ)p(θ)] - ∇_θ log p(D)
                    = ∇_θ log[p(D|θ)p(θ)]    (constant w.r.t. θ!)
    
    점수는 고르개 상수가 필요 없다!

고갱이 눈썰미:
    점수를 쓰면 고르지 않은 분포를 그대로 다룰 수 있다,
    이것이 바로 베이즈 추론에 필요한 것이다!

지은이: 이성철 @ 연세대학교
날짜: 2025년 11월
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm, multivariate_normal
from matplotlib import cm

# 그리기 결결이를 정한다
plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(42)

print("=" * 80)
print("MODULE 01: SCORE FUNCTIONS BASICS")
print("=" * 80)

# ============================================================================
# 마디 1: 뜻매김과 직관
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 1: What is a Score Function?")
print("=" * 80)

print("""
정의:
-----------
확률 분포 p(x)의 점수 함수는 다음과 같다.

    s(x) = ∇_x log p(x) = (1/p(x)) * ∇_x p(x)

느낌으로 보면:
---------
- 점수는 확률이 더 높은 자리를 가리킨다
- p(x)의 봉우리에서 점수는 0이다
- 크기 ||s(x)||는 확률이 얼마나 가파르게 바뀌는지 알려 준다
- 점수에는 고르게 맞추는 상수가 필요 없다!

베이즈 미룸과의 이음:
--------------------------------
01_Bayesian_Inference 단원에서 배운 것을 떠올려라.
- Posterior: p(θ|D) ∝ p(D|θ)p(θ)
- We couldn't compute p(D) = ∫ p(D|θ)p(θ) dθ

점수를 쓰면:
- ∇_θ log p(θ|D) = ∇_θ log[p(D|θ)p(θ)]
- ∫이 필요 없다! 점수는 고르게 맞추지 않은 분포에서도 통한다!
""")

# ============================================================================
# 마디 2: 단순한 보기 - 1차원 정규 분포
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: Score of 1D Gaussian Distribution")
print("=" * 80)

print("""
Example: Gaussian N(μ, σ²)
--------------------------
PDF: p(x) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))

로그 확률밀도: log p(x) = -log(√(2πσ²)) - (x-μ)²/(2σ²)

Score: s(x) = d/dx log p(x) = -(x-μ)/σ²

눈여겨볼 점은 다음과 같다.
1. 상수 마디 -log(√(2πσ²))는 사라진다!(상수의 미분은 0)
2. 점수는 x에 대해 선형이다
3. x=μ(최빈값)에서 점수는 0이다
4. 점수는 평균 μ 쪽을 가리킨다
""")

# 셈으로 보여 준다
mu, sigma = 0.0, 1.0
x_vals = np.linspace(-4, 4, 1000)

# 확률 밀도
pdf_vals = norm.pdf(x_vals, mu, sigma)

# 로그 확률 밀도
log_pdf_vals = norm.logpdf(x_vals, mu, sigma)

# 점수(닫힌 꼴)
score_vals = -(x_vals - mu) / (sigma ** 2)

# 시각화 만들기
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# 그림 1: 확률 밀도
axes[0].plot(x_vals, pdf_vals, 'b-', linewidth=2, label='p(x)')
axes[0].fill_between(x_vals, pdf_vals, alpha=0.3)
axes[0].axvline(mu, color='r', linestyle='--', label=f'μ = {mu}')
axes[0].set_ylabel('p(x)', fontsize=12)
axes[0].set_title('Probability Density Function (PDF)', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 그림 2: 로그 확률 밀도
axes[1].plot(x_vals, log_pdf_vals, 'g-', linewidth=2, label='log p(x)')
axes[1].axvline(mu, color='r', linestyle='--', label=f'μ = {mu}')
axes[1].set_ylabel('log p(x)', fontsize=12)
axes[1].set_title('Log Probability', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 그림 3: 점수 함수
axes[2].plot(x_vals, score_vals, 'purple', linewidth=2, label=r's(x) = \nabla\log p(x)')
axes[2].axhline(0, color='k', linestyle='-', alpha=0.3)
axes[2].axvline(mu, color='r', linestyle='--', label=fr'$\mu$ = {mu} (score=0)')
axes[2].fill_between(x_vals, score_vals, alpha=0.3, color='purple')

# 방향을 보이는 화살표를 더한다
for x_point in [-2, -1, 1, 2]:
    score_at_x = -(x_point - mu) / (sigma ** 2)
    axes[2].arrow(x_point, score_at_x, 0.3 * np.sign(score_at_x), 0,
                  head_width=0.3, head_length=0.15, fc='darkred', ec='darkred', linewidth=2)

axes[2].set_xlabel('x', fontsize=12)
axes[2].set_ylabel('s(x)', fontsize=12)
axes[2].set_title(r'Score Function $s(x) = \nabla\log p(x)$\n(Points toward mean $\mu$)', fontsize=14, fontweight='bold')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(os.path.dirname(__file__), '01_score_1d_gaussian.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print("\n✓ Saved: 01_score_1d_gaussian.png")
plt.close()

print("\nKEY OBSERVATIONS:")
print("  1. Score is ZERO at the mode (x = μ)")
print("  2. Score POINTS TOWARD the mode:")
print("     - For x < μ: score is POSITIVE (points right)")
print("     - For x > μ: score is NEGATIVE (points left)")
print("  3. Magnitude increases with distance from mode")
print("  4. NO normalization constant √(2πσ²) in the score!")

# ============================================================================
# 마디 3: 2차원 정규 분포 - 점수 마당 그려 보기
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: Score Field for 2D Gaussian")
print("=" * 80)

print("""
다변량 가우스 N(μ, Σ)에서는:
---------------------------------
Score: s(x) = ∇_x log p(x) = -Σ^(-1)(x - μ)

이는 평균 μ 쪽을 가리키는 선형 함수이다!

해석:
- 점수 벡터는 확률 높은 자리를 가리킨다
- 평균 μ에서 점수는 0이다
- 함께 흩어짐 Σ이 방향과 크기를 빚는다
""")

# 2차원 정규 분포 매개변수
mu_2d = np.array([0, 0])
Sigma_2d = np.array([[1.0, 0.5], [0.5, 1.0]])  # 서로 이어짐
Sigma_inv = np.linalg.inv(Sigma_2d)

# 격자 생성
x1 = np.linspace(-3, 3, 30)
x2 = np.linspace(-3, 3, 30)
X1, X2 = np.meshgrid(x1, x2)
pos = np.dstack((X1, X2))

# 확률 밀도를 셈한다
rv = multivariate_normal(mu_2d, Sigma_2d)
pdf_2d = rv.pdf(pos)

# 점마다 점수를 셈한다
# s(x) = -Σ^(-1)(x - μ)
score_field = np.zeros((len(x2), len(x1), 2))
for i in range(len(x2)):
    for j in range(len(x1)):
        x = np.array([X1[i, j], X2[i, j]])
        score_field[i, j] = -Sigma_inv @ (x - mu_2d)

# 시각화 만들기
fig = plt.figure(figsize=(16, 6))

# 그림 1: 등고선을 갖춘 확률 밀도
ax1 = fig.add_subplot(121)
contour = ax1.contourf(X1, X2, pdf_2d, levels=20, cmap='viridis', alpha=0.8)
ax1.contour(X1, X2, pdf_2d, levels=10, colors='white', alpha=0.4, linewidths=0.5)
ax1.plot(mu_2d[0], mu_2d[1], 'r*', markersize=20, label=r'Mean $\mu$')
plt.colorbar(contour, ax=ax1, label=r'$p(x)$')
ax1.set_xlabel(r'$x_1$', fontsize=12)
ax1.set_ylabel(r'$x_2$', fontsize=12)
ax1.set_title('2D Gaussian PDF', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# 그림 2: 점수 마당
ax2 = fig.add_subplot(122)
# 바탕: 확률 밀도 등고선
ax2.contourf(X1, X2, pdf_2d, levels=20, cmap='viridis', alpha=0.3)
ax2.contour(X1, X2, pdf_2d, levels=10, colors='gray', alpha=0.3, linewidths=0.5)

# 점수 벡터
skip = 2  # 또렷하게 하려 일부만 뽑는다
ax2.quiver(X1[::skip, ::skip], X2[::skip, ::skip],
           score_field[::skip, ::skip, 0], score_field[::skip, ::skip, 1],
           color='red', alpha=0.8, scale=20, width=0.004)
ax2.plot(mu_2d[0], mu_2d[1], 'r*', markersize=20, label=r'Mean $\mu$ (score=0)')
ax2.set_xlabel(r'$x_1$', fontsize=12)
ax2.set_ylabel(r'$x_2$', fontsize=12)
ax2.set_title(r'Score Function $s(x)=\nabla \log p(x)$' r"\n(Points toward mean $\mu$)", 
                  fontsize=14, fontweight='bold')

ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

plt.tight_layout()
fig_path = os.path.join(os.path.dirname(__file__), '01_score_2d_gaussian_field.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print("\n✓ Saved: 01_score_2d_gaussian_field.png")
plt.close()

print("\nKEY OBSERVATIONS:")
print("  1. Score vectors point toward the mean (high probability)")
print("  2. Score is zero at the mean")
print("  3. The covariance structure affects score directions")
print("  4. Longer arrows = steeper probability gradient")

# ============================================================================
# 마디 4: 뽑기에서 점수가 중요한 까닭
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: Why Score Functions Enable Sampling")
print("=" * 80)

print("""
바탕이 되는 문제(베이즈 미룸에서):
-------------------------------------------------
Given: Posterior p(θ|D) ∝ p(D|θ)p(θ)
바라는 것: p(θ|D)에서 뽑은 표본
문제: 고르개 상수 p(D)를 셈할 수 없다!

점수 풀이:
------------------
고갱이 눈썰미: ∇_θ log p(θ|D) = ∇_θ log[p(D|θ)p(θ)]

점수는 p(D)가 필요 없다!

랑주뱅 움직임(3단원 미리 보기):
----------------------------------------
점수 s(x) = ∇_x log p(x)를 알면 다음으로 표본을 뽑을 수 있다.

    x_{t+1} = x_t + (ε/2) * s(x_t) + √ε * z_t
    
여기서 z_t ~ N(0, I)은 정규 잡음이다.

직관:
- 떠돎 항 (ε/2)*s(x_t)이 확률 높은 쪽으로 옮긴다
- 퍼짐 항 √ε*z_t이 마구잡이를 더한다
- 둘이 함께 분포를 살핀다!

이것이 퍼짐 모델이 도는 방식이다!
""")

# 1차원 정규 분포의 단순한 랑주뱅 뽑기를 보여 준다
print("\nDemonstration: Langevin Sampling from 1D Gaussian")
print("-" * 80)

# 랑주뱅 움직임 매개변수
n_steps = 1000
epsilon = 0.1
x_current = -3.0  # 평균에서 멀리 시작

# 저장 공간
trajectory = [x_current]

# 랑주뱅 움직임을 돌린다
for step in range(n_steps):
    # 지금 자리의 점수
    score = -(x_current - mu) / (sigma ** 2)
    
    # 랑주뱅 고침
    x_current = x_current + (epsilon / 2) * score + np.sqrt(epsilon) * np.random.randn()
    trajectory.append(x_current)

trajectory = np.array(trajectory)

# 시각화한다
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 그림 1: 자취
axes[0].plot(trajectory, 'b-', alpha=0.6, linewidth=1)
axes[0].axhline(mu, color='r', linestyle='--', linewidth=2, label=f'Target mean μ={mu}')
axes[0].set_xlabel('Step', fontsize=12)
axes[0].set_ylabel('x', fontsize=12)
axes[0].set_title('Langevin Dynamics Trajectory\n(Using score to sample from N(0,1))', 
                  fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 그림 2: 히스토그램과 참 분포
axes[1].hist(trajectory[100:], bins=50, density=True, alpha=0.6, color='blue', 
             label='Langevin samples')
axes[1].plot(x_vals, pdf_vals, 'r-', linewidth=3, label='True N(0,1)')
axes[1].set_xlabel(r'$x$', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].set_title('Samples vs True Distribution\n(Langevin correctly samples from target!)', 
                  fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(os.path.dirname(__file__), '01_langevin_preview.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print("\n✓ Saved: 01_langevin_preview.png")
plt.close()

print("\n✓ Langevin dynamics successfully sampled from target distribution!")
print("  This preview shows why scores are powerful:")
print("  - Only need score, not normalization")
print("  - Works for any distribution")
print("  - Foundation for diffusion models!")

# ============================================================================
# 마디 5: 퍼짐 모델과의 이음(미리 보기)
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 5: Preview - Connection to Diffusion Models")
print("=" * 80)

print("""
큰 그림:
---------------

1. 베이즈 미룸(01_Bayesian_Inference 단원):
   - p(θ|D) ∝ p(D|θ)p(θ)
   - 고르개 상수를 셈할 수 없다
   - 뽑기 방법이 필요하다

2. 점수 함수(이 단원):
   - s(x) = ∇_x log p(x)
   - 고르게 맞추지 않아도 통한다
   - 랑주뱅 뽑기를 가능하게 한다

3. 점수 맞추기(다음 단원):
   - 자료에서 점수를 배운다
   - 잡음 없애기와의 이음

4. 여러 자의 점수(뒤 단원):
   - 잡음 수준마다 점수를 배운다
   - σ₁ > σ₂ > ... > σ_L

5. 퍼짐 모델(마지막 단원):
   - 앞으로: 잡음을 차츰 더한다
   - 뒤로: 배운 점수로 잡음을 없앤다
   - 뽑기로 만들어 낸다!

퍼짐을 위한 고갱이 눈썰미:
-------------------------
잡음 섞인 자료의 잡음 지우기가 곧 베이즈 뒷분포 미룸이다!

주어진 것: x_noisy = x_clean + σ * noise
구할 것: p(x_clean | x_noisy)

이 사후 분포의 점수가 바로 퍼짐 모델이 배우는 것이다!

  ∇_x log p(x|x_noisy) ← 이것을 신경 그물이 배운다!
""")

# ============================================================================
# 간추리기와 익힘
# ============================================================================
print("\n" + "=" * 80)
print("MODULE SUMMARY")
print("=" * 80)

print("""
배운 것:
---------------
1. 점수 함수: s(x) = ∇_x log p(x)
2. 점수는 확률 높은 자리를 가리킨다
3. 점수는 고르개 상수가 필요 없다
4. 베이즈 사후 추론과의 이음
5. 점수는 랑주뱅 움직임으로 뽑기를 가능하게 한다
6. 퍼짐 모델의 바탕

고갱이 식:
------------
1. 점수의 뜻매김:
   s(x) = ∇_x log p(x) = (1/p(x)) * ∇_x p(x)

2. 가우스 점수:
   For N(μ, σ²): s(x) = -(x-μ)/σ²
   For N(μ, Σ): s(x) = -Σ^(-1)(x-μ)

3. 뒷분포 점수(베이즈):
   ∇_θ log p(θ|D) = ∇_θ log[p(D|θ)p(θ)]  (p(D)가 필요 없다!)

4. 랑주뱅 움직임(미리 보기):
   x_{t+1} = x_t + (ε/2)*s(x_t) + √ε*z_t

만들어진 파일:
---------------
1. 01_score_1d_gaussian.png - 1차원 정규 분포의 점수
2. 01_score_2d_gaussian_field.png - 2차원 점수 마당
3. 01_langevin_preview.png - 점수를 쓴 뽑기
""")

print("\n" + "=" * 80)
print("EXERCISES")
print("=" * 80)

print("""
익힘 1: 닫힌 꼴 점수 셈하기
---------------------------------------
다음의 점수 함수를 셈하여라.
a) 지수 분포: x ≥ 0에서 p(x) = λ exp(-λx)
b) 라플라스 분포: p(x) = (1/2b) exp(-|x-μ|/b)
c) 정규 분포 둘 섞기

익힘 2: 점수의 성질
---------------------------
다음을 밝혀라.
a) 봉우리에서 점수는 0이다
b) ∫ p(x) s(x) dx = 0 (점수의 평균은 0이다)
c) 정규 분포에서 점수는 선형이다

익힘 3: 짜기
-------------------------
다음의 점수 셈을 짜라.
a) 정규 분포 셋의 2차원 섞기
b) 점수 마당을 그려 본다
c) 랑주뱅 움직임을 돌려 뽑는다

익힘 4: 베이즈 추론과의 이음
-------------------------------------------
베타-이항 켤레 짝에서(01_Bayesian_Inference):
a) 사후 분포의 점수를 이끌어 낸다
b) 고르개 상수가 필요 없음을 보여라
c) 곧바로 셈한 사후 분포와 견준다

익힘 5: 점수 맞추기 미리 보기
---------------------------------
식 없이 p(x)에서 뽑은 표본만 있다면:
a) s(x) = ∇_x log p(x)를 곧바로 셈할 수 있는가? 왜 안 되는가?
b) 잡음 없애기를 쓴 다른 방식을 내놓는다
c) 이것이 다음 단원의 까닭이 된다!

도전 익힘: 봉우리 여럿인 분포
------------------------------------------
봉우리가 여럿인 2차원 "바둑판" 분포를 만들어라.
a) p(x)을 격자 위 정규 분포 섞기로 뜻매김한다
b) 점수 마당을 셈해 그려 본다
c) 랑주뱅 움직임을 돌린다 - 모든 봉우리를 살피는가?
d) 걸음 크기 ε을 달리하면 어떻게 되는가?
""")

print("\n" + "=" * 80)
print("NEXT MODULE: 02_score_matching_theory.py")
print("=" * 80)
print("""
다음 단원에서는 고갱이 물음을 다룬다.

  "자료만으로 점수 함수를 어떻게 배우는가?"

이것이 요즘 만들어 내는 모델의 바탕인 점수 맞추기로 이어진다!
""")

print("\n✓ Module 01 complete! Generated 3 visualizations.")
print("  Ready for Module 02: Score Matching Theory")
print("=" * 80)


if __name__ == "__main__":
    pass
```

## 논의

점수 함수 바탕의 짜기는 이 마당에 자리 잡은 방식을 따른다. 코드 짜임이 모델 뜻매김과 익히기 논리를 갈라 놓아 부품을 하나씩 고치기 쉽다. 얼개 고르기는 만들어 내는 모델 무리가 많은 실험에서 얻은 배움을 담고 있다.

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
