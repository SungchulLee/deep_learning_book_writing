# 변분 추론

변분 추론 - 연습 문제. 이 파일에는 변분 추론의 개념과 구현을 익히는 연습 문제가 담겨 있다.

변분 추론은 어림 베이즈 셈하기에 규모를 키울 수 있는 길을 준다. 이 구현은 변분 추론의 핵심 개념을 보이며, 확률 모형에서 최적화가 다룰 수 없는 적분을 어떻게 대신하는지 보여 준다.

## 코드

```python
"""
변분 추론 - 연습 문제
==================================

이 파일에는 변분 추론의 개념과 구현을 익히는 연습 문제가 담겨 있다.
풀이는 solutions/ 디렉터리에 있다.

지은이: 연세대학교 이성철 교수
전자우편: sungchulyonsei@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

sns.set_style("whitegrid")


# ============================================================================
# 연습 1: 기본 변분 추론 - 가우스 평균 어림(첫걸음)
# ============================================================================

print("=" * 80)
print("EXERCISE 1: Gaussian Mean Estimation with VI")
print("=" * 80)

exercise_1_description = """
문제:
-------
σ²을 아는 N(θ, σ²)에서 자료 점 n개를 관측한다.
θ에 대한 앞확률은 N(μ₀, σ₀²)이다.

과제:
----
1. 변분 집안 q(θ) = N(m, s²)의 ELBO 이끌어 내기
2. 가장 좋은 변분 매개변수 m*과 s* 이끌어 내기
3. ELBO를 가장 크게 하는 기울기 오르기 구현하기
4. 정확한 뒤확률과 견주기(그것도 가우스이다)
5. 변분 매개변수의 모임 그려 보기

주어진 것:
-----
- 자료: [2.3, 3.1, 2.8, 3.5, 2.9, 3.2, 2.7, 3.0]
- 아는 σ² = 0.25
- 앞확률: μ₀ = 0, σ₀² = 4.0

제출물:
------------
1. 수학으로 이끌어 내기(글로)
2. 파이썬 구현
3. 다음을 보이는 그림:
   가) 앞확률, 가능도, 뒤확률, 변분 어림
   나) ELBO의 모임
   다) 매개변수의 모임(m과 s)

힌트:
-----
- ELBO = E_q[log p(D,θ)] - E_q[log q(θ)]에서 시작하기
- 가우스 q에서는 기댓값이 닫힌 꼴이다
- 답을 켤레 뒤확률과 맞춰 보기
"""

print(exercise_1_description)

# 배우는 이를 위한 자료 만들기
np.random.seed(42)
data_ex1 = np.array([2.3, 3.1, 2.8, 3.5, 2.9, 3.2, 2.7, 3.0])
sigma_sq_ex1 = 0.25
mu_0_ex1 = 0.0
sigma_0_sq_ex1 = 4.0

print("\nData provided:")
print(f"  x = {data_ex1}")
print(f"  σ² = {sigma_sq_ex1}")
print(f"  μ₀ = {mu_0_ex1}, σ₀² = {sigma_0_sq_ex1}")

# 할 일: 배우는 이가 여기에 구현
print("\n[TODO: Implement your solution here]")
print("-" * 80)


# ============================================================================
# 연습 2: 평균장 변분 추론 - 베이즈 선형 회귀(중급)
# ============================================================================

print("\n" + "=" * 80)
print("EXERCISE 2: Bayesian Linear Regression with Mean-Field VI")
print("=" * 80)

exercise_2_description = """
문제:
-------
선형 회귀: ε ~ N(0, σ²)일 때 y = Xw + ε

베이즈 모형:
- 가능도: y | X, w, σ² ~ N(Xw, σ²I)
- 가중값의 앞확률: w ~ N(0, λ⁻¹I)
- 정밀도의 앞확률: τ = 1/σ² ~ Gamma(a₀, b₀)

과제:
----
1. 평균장 어림 이끌어 내기: q(w,τ) = q(w)q(τ)
2. q(w)과 q(τ)의 CAVI 새로 고침 이끌어 내기
3. CAVI 알고리즘 전체 구현하기
4. 정확한 뒤확률과 견주기(이 모형에는 있다!)
5. 앞확률의 세기 λ이 주는 영향 살피기

자료 만들기:
---------------
참 모형: y = 2x₁ + 3x₂ - 1 + ε, ε ~ N(0, 0.5²)
x₁, x₂ ~ U(0, 1)인 표본 n=50개 만들기

제출물:
------------
1. CAVI 새로 고침 식(수학으로 이끌어 낸 것)
2. 온전한 파이썬 구현
3. 그림:
   가) 가중값 w의 뒤확률
   나) 정밀도 τ의 뒤확률
   다) 미리봄 분포
   라) ELBO의 모임
4. 정확한 뒤확률과의 견줌
5. 앞확률에 대한 민감도 살피기

힌트:
-----
- q(w)과 q(τ)이 모두 켤레 집안에 든다
- q(w) = N(m_w, Σ_w)
- q(τ) = Gamma(a_n, b_n)
- 일반 CAVI 새로 고침 쓰기: q*_j ∝ exp{E_{-j}[log p(all)]}
"""

print(exercise_2_description)

# 배우는 이를 위한 자료 만들기
np.random.seed(42)
n_ex2 = 50
X_ex2 = np.random.rand(n_ex2, 2)
w_true_ex2 = np.array([2.0, 3.0])
y_true_ex2 = X_ex2 @ w_true_ex2 - 1.0
y_ex2 = y_true_ex2 + np.random.normal(0, 0.5, n_ex2)

print("\nData generated:")
print(f"  n = {n_ex2} samples")
print(f"  True weights: w = {w_true_ex2}")
print(f"  True intercept: b = -1.0")
print(f"  Noise std: σ = 0.5")

# 할 일: 배우는 이가 여기에 구현
print("\n[TODO: Implement your solution here]")
print("-" * 80)


# ============================================================================
# 연습 3: 섞음 모형의 변분 베이즈(상급)
# ============================================================================

print("\n" + "=" * 80)
print("EXERCISE 3: Univariate Gaussian Mixture with Full Bayesian Treatment")
print("=" * 80)

exercise_3_description = """
문제:
-------
1차원 자료에 성분 K개의 가우스 섞음 모형을 맞추되
온전한 베이즈 다루기(모든 매개변수에 앞확률).

모형:
- 자료: x_i ~ Σ_k π_k N(μ_k, τ_k⁻¹)
- 앞확률:
  * π ~ Dirichlet(α)
  * μ_k ~ N(m, (βτ_k)⁻¹)
  * τ_k ~ Gamma(a, b)

평균장 어림:
q(Z, π, μ, τ) = q(Z) q(π) ∏_k q(μ_k, τ_k)

과제:
----
1. 모든 변분 인수의 온전한 CAVI 새로 고침 이끌어 내기
2. 알고리즘 전체를 밑바닥부터 구현하기
3. μ_k과 τ_k의 얽힘 다루기
4. 모형 고르기 구현하기(ELBO로 K 고르기)
5. sklearn의 가우스 섞음 모형과 견주기

어려움:
----------
- 켤레 새로 고침에 가우스-감마의 얽힘이 들어 있다
- 빈 무리를 매끄럽게 다뤄야 한다
- 섞음 모형의 이름표 바뀜 문제
- 첫값 잡기가 결정적이다

자료:
----
가우스 3개의 섞음 만들기:
- 성분 1: μ=-5, σ=1, 무게=0.3
- 성분 2: μ=0, σ=1.5, 무게=0.5
- 성분 3: μ=5, σ=0.8, 무게=0.2
표본 n=300개

제출물:
------------
1. CAVI 새로 고침의 온전한 이끌어 냄
2. 다음을 갖춘 온전한 구현:
   - 제대로 된 첫값 잡기
   - 모임 살피기
   - 빈 무리 다루기
3. 그림:
   가) 배운 성분과 함께 그린 자료
   나) 맡음 몫(부드러운 배정)
   다) ELBO의 모임
   라) 모형 고르기 곡선(K에 따른 ELBO)
4. sklearn과의 견줌
5. 뒤확률 불확실함 살피기

덤:
-----
- K에 대한 저절로 관련성 정하기(ARD) 구현하기
- 매개변수 사이의 뒤확률 상관 그려 보기
- 대각에 낮은 계수를 더한 공분산 구현하기
"""

print(exercise_3_description)

# 복잡한 섞음 자료 만들기
np.random.seed(42)
n_ex3 = 300
mixture_params = [
    {'mean': -5.0, 'std': 1.0, 'weight': 0.3},
    {'mean': 0.0, 'std': 1.5, 'weight': 0.5},
    {'mean': 5.0, 'std': 0.8, 'weight': 0.2},
]

data_ex3 = []
labels_ex3 = []
for k, params in enumerate(mixture_params):
    n_k = int(n_ex3 * params['weight'])
    data_k = np.random.normal(params['mean'], params['std'], n_k)
    data_ex3.extend(data_k)
    labels_ex3.extend([k] * n_k)

data_ex3 = np.array(data_ex3)
labels_ex3 = np.array(labels_ex3)

print("\nMixture data generated:")
print(f"  n = {len(data_ex3)} samples")
print(f"  K = {len(mixture_params)} components")
for k, params in enumerate(mixture_params):
    print(f"  Component {k+1}: μ={params['mean']}, σ={params['std']}, π={params['weight']}")

# 자료 그려 보기
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(data_ex3, bins=50, density=True, alpha=0.7, edgecolor='black')
plt.xlabel('x', fontsize=11)
plt.ylabel('Density', fontsize=11)
plt.title('Mixture Data Distribution', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
colors = ['red', 'green', 'blue']
for k in range(len(mixture_params)):
    mask = labels_ex3 == k
    plt.hist(data_ex3[mask], bins=20, density=True, alpha=0.6, 
            color=colors[k], label=f'Component {k+1}', edgecolor='black')
plt.xlabel('x', fontsize=11)
plt.ylabel('Density', fontsize=11)
plt.title('True Component Separation', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/variational_inference/exercises/exercise_03_data.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("\n[Data visualization saved: exercise_03_data.png]")

# 할 일: 배우는 이가 여기에 구현
print("\n[TODO: Implement your solution here]")
print("-" * 80)


# ============================================================================
# 연습 4: 확률 변분 추론(상급)
# ============================================================================

print("\n" + "=" * 80)
print("EXERCISE 4: Implement Stochastic Variational Inference")
print("=" * 80)

exercise_4_description = """
문제:
-------
큰 자료의 베이즈 로지스틱 회귀에 확률 변분 추론을 구현하여라.
메모리에 한꺼번에 들어가지 않는 자료를 다룬다.

모형:
- 두 값 가르기: p(y=1|x,w) = σ(w^T x)
- 앞확률: w ~ N(0, λ⁻¹I)
- 변분 집안: q(w) = N(m, S)

과제:
----
1. 자료를 골라 뽑을 때의 ELBO 이끌어 내기
2. 확률 자연 기울기 새로 고침 구현하기
3. 작은 묶음의 흩어짐 줄이기 다루기
4. 배움률 일정 구현하기
5. 온전한 묶음 변분 추론과 견주기

확률 기울기:
-------------------
∇_θ ELBO ≈ (N/M) Σ_{i∈M} ∇_θ E_q[log p(y_i|x_i,w)] + ∇_θ E_q[log p(w)/q(w)]

여기서 M은 작은 묶음의 크기이고 N은 자료 전체의 크기이다.

자료:
----
두 값 가르기 자료 만들기:
- 표본 N = 10,000개
- 특징 d = 20개
- 참 가중값: 성김(0이 아닌 것이 5개뿐)
- 갈래 균형: 50-50

제출물:
------------
1. 다음을 갖춘 확률 변분 추론 구현:
   - 작은 묶음으로 다루기
   - 자연 기울기 새로 고침
   - 맞춰 가는 배움률
   - 모임 진단
2. 그림:
   가) 되풀이에 따른 ELBO
   나) 되풀이에 따른 시험 정확도
   다) 매개변수 어림값과 참값
   라) 배움률 일정
3. 견줌: 확률 변분 추론과 온전한 묶음 변분 추론
4. 메모리 씀씀이 살피기
5. 규모 실험(N, d, 묶음 크기를 바꿔 가며)

힌트:
-----
- 자연 기울기 쓰기: ∇_nat = S ∇_m ELBO
- 로빈스-먼로 배움률 구현하기: ρ_t = (t + τ)^{-κ}
- 흩어짐을 줄이려고 라오-블랙웰화 쓰기
- 남겨 둔 확인 자료에서 ELBO 지켜보기
"""

print(exercise_4_description)

# 큰 자료 만들기
np.random.seed(42)
n_ex4 = 10000
d_ex4 = 20
w_true_ex4 = np.zeros(d_ex4)
w_true_ex4[:5] = np.random.randn(5) * 2  # 처음 5개 특징만 관련 있다

X_ex4 = np.random.randn(n_ex4, d_ex4)
logits = X_ex4 @ w_true_ex4
probs = 1 / (1 + np.exp(-logits))
y_ex4 = (np.random.rand(n_ex4) < probs).astype(int)

print("\nLarge-scale data generated:")
print(f"  n = {n_ex4} samples")
print(f"  d = {d_ex4} features")
print(f"  Sparsity: {np.sum(w_true_ex4 != 0)} / {d_ex4} non-zero weights")
print(f"  Class balance: {np.mean(y_ex4):.2%} positive")

# 할 일: 배우는 이가 여기에 구현
print("\n[TODO: Implement your solution here]")
print("-" * 80)


# ============================================================================
# 덤 연습 문제: 깜깜이 변분 추론
# ============================================================================

print("\n" + "=" * 80)
print("BONUS EXERCISE: Black-Box Variational Inference")
print("=" * 80)

bonus_exercise_description = """
문제:
-------
켤레가 아닌 모형에 점수 함수 기울기 어림꼴로 깜깜이 변분 추론을
(REINFORCE)로 구현하여라.

모형: 베이즈 프로빗 회귀
- y_i | x_i, w ~ Bernoulli(Φ(w^T x_i))
- 여기서 Φ은 표준 정규 누적분포함수이다
- 앞확률: w ~ N(0, I)

이 모형은 켤레가 아니므로 표준 CAVI를 쓸 수 없다!

과제:
----
1. 점수 함수 기울기 어림꼴 구현하기
2. 흩어짐 줄이는 기법 구현하기:
   - 다스림 변량
   - 라오-블랙웰화
3. (될 수 있으면) 매개변수 바꾸기 재주와 견주기
4. 익히는 동안 기울기의 흩어짐 살피기

점수 함수 기울기:
-----------------------
∇_θ E_q[f(z)] = E_q[(∇_θ log q(z)) f(z)]

∇_z f을 몰라도 몬테카를로로 어림할 수 있다!

제출물:
------------
1. 깜깜이 변분 추론 구현
2. 흩어짐 줄이기 살피기
3. 모형마다 만든 변분 추론과 견주기
4. 기울기 흩어짐 그림
5. 마지막 미리봄 정확도

까다롭지만 아무 모형에나 쓰는 변분 추론을 배우게 해 준다!
"""

print(bonus_exercise_description)


# ============================================================================
# 안내
# ============================================================================

print("\n" + "=" * 80)
print("INSTRUCTIONS FOR EXERCISES")
print("=" * 80)

instructions = """
연습 문제를 푸는 법:
=========================

1. 문제를 꼼꼼히 읽어라
   - 모형 이해하기
   - 무엇을 이끌어 내고 무엇을 구현할지 가려내기
   - 힌트나 간추림을 적어 두기

2. 수학으로 이끌어 내기
   - 첫 원리에서 시작하기
   - ELBO를 드러내 적기
   - 새로 고침 식을 걸음마다 이끌어 내기
   - 단순한 경우로 수식 확인하기

3. 구현
   - 첫값 잡기부터 시작하기
   - CAVI 새로 고침을 하나씩 꼼꼼히 구현하기
   - 모임 살피기 넣기
   - 그려 보는 코드 넣기

4. 시험하기
   - 먼저 아는 단순한 경우로 시험하기
   - 정확한 풀이가 있으면 그것과 견주기
   - 기울기 셈하기를 수치로 확인하기
   - ELBO가 단조롭게 커지는지 확인하기

5. 살피기
   - 요구한 그림 만들기
   - 웃매개변수에 대한 민감도 살피기
   - 다른 방법과 견주기
   - 알아낸 것 정리해 쓰기

제출:
==========
- 코드와 이끌어 냄을 담은 파이썬 노트북
- 수학으로 이끌어 낸 내용을 담은 PDF
- 요구한 그림 모두
- 연습마다 짧은 글(1-2쪽)

자료:
=========
- 단원 노트와 코드
- 비숍 PRML 10장
- Blei 외(2017) 변분 추론 리뷰
- 상담 시간: [일정 미정]

채점:
=======
- 수학으로 이끌어 내기: 40%
- 코드 구현: 30%
- 그림과 그려 보기: 15%
- 살피기와 글쓰기: 15%

변분 추론을 즐겁게 배우자!
"""

print(instructions)

print("\n" + "=" * 80)
print("END OF EXERCISES")
print("=" * 80)


if __name__ == "__main__":
    pass
```

## 논의

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 무늬는 더 복잡한 상황으로 자연스럽게 넓혀진다. 웃매개변수와 얼개의 갈래, 여러 자료를 이것저것 시험해 보면 이해가 깊어지고 어림 추론 과제에 대한 실전 직관이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 끝까지 읽고 핵심 설계 판단을 가려내어라. 구체적인 구현의 고름 셋을 들고, 그것이 변분 추론에 왜 알맞은지 저마다 설명하여라.

??? success "연습문제 1 풀이"
    설계 결정은 구현마다 다르지만 흔히 다음이 포함된다. (1) 활성화 함수의 선택 — ReLU 계열은 포화되지 않는 경사를 주어 학습을 빠르게 한다. (2) 정규화 전략 — 배치 정규화는 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 — 있을 경우 건너뛰는 경로를 제공하여 깊은 신경망에서도 경사가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 절충을 반영한다.

---

**연습문제 2.**
입력이 기대하는 모양과 자료형을 갖는지 확인하도록 주 함수나 클래스에 입력 검증을 추가하라. 잘못된 입력에는 유익한 오류 메시지를 내라.

??? success "연습문제 2 풀이"
    `forward` 메서드(또는 해당 함수)의 첫머리에 다음과 같은 검사를 추가한다. `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`와 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. 모양을 검증할 때는 중요한 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 유익한 오류 메시지는 디버깅 속도를 크게 높이고 코드를 재사용하기에도 더 견고하게 만든다.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
변분 추론 구현이 옳은지 확인하는 두루 갖춘 시험 함수를 짜라. 빈 들임, 원소 하나짜리 들임, 아주 큰 들임, 극단 값(0, 아주 큰 수)이 든 들임 같은 모서리 경우를 시험하여라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_variational inference():
        model = Variational Inference(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (넘침을 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 경사의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    경사의 흐름을 시험하는 것은 그 구조가 처음부터 끝까지 이어지는 학습을 지원하는지 확인하는 데 특히 중요하다.
