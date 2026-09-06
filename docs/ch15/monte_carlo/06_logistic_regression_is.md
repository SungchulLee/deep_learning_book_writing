# 로지스틱 회귀 중요도 표집

06_logistic_regression_IS.py 중간 수준: 중요도 표집으로 하는 베이즈 로지스틱 회귀

확률 과정과 표집 방법은 확률적 기계 학습의 바탕 도구이다. 이 모듈은 몬테카를로 방법의 개념을 보이며 수학 이론과 셈 구현 사이의 틈을 잇는다.

## 코드

```python
"""
06_logistic_regression_IS.py

중급 단계: 중요도 표집으로 하는 베이즈 로지스틱 회귀

이 단원은 베이즈 추론을 위한 중요도 표집을 보인다
로지스틱 회귀에서 다룬다. 이는 뒤확률이 닫힌 꼴로 주어지지 않는
켤레가 아닌 모형이다.

모형:
------
가능도: yᵢ ~ Bernoulli(π(xᵢ'β))
            여기서 π(z) = 1/(1 + exp(-z))은 로지스틱 함수이다

앞확률: β ~ N(μ₀, Σ₀)

Posterior: p(β|y,X) ∝ ∏ᵢ π(xᵢ'β)^yᵢ (1-π(xᵢ'β))^(1-yᵢ) × N(β|μ₀,Σ₀)

이는 켤레가 아닌 모형이다. 곧 뒤확률이 가우스가 아니다!

제안 전략:
1. 앞확률을 제안으로(단순하지만 흔히 효율이 낮다)
2. 라플라스 어림(봉우리에서의 가우스)
3. 변분 어림
4. 맞춰 가는 제안

지은이: 베이즈 추론 교육 자료
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit  # 로지스틱 함수
import seaborn as sns

np.random.seed(42)
sns.set_style("whitegrid")


def logistic(z):
    """로지스틱 함수: π(z) = 1/(1 + exp(-z))"""
    return expit(z)


def log_likelihood_logistic(beta, X, y):
    """
    로지스틱 회귀의 로그 가능도.
    
    log p(y|X,β) = Σᵢ [yᵢ log π(xᵢ'β) + (1-yᵢ) log(1-π(xᵢ'β))]
    """
    eta = X @ beta  # 선형 미리봄 xᵢ'β
    
    # 수치로 안정된 로그 가능도
    # log π(η) = -log(1 + exp(-η))
    # log(1-π(η)) = -log(1 + exp(η))
    
    log_lik = np.sum(
        y * (-np.log1p(np.exp(-eta))) +
        (1 - y) * (-np.log1p(np.exp(eta)))
    )
    
    return log_lik


def log_prior_gaussian(beta, mu_0, Sigma_0_inv):
    """
    로그 앞확률: log p(β) = log N(β|μ₀, Σ₀)
    """
    diff = beta - mu_0
    log_prior = -0.5 * (diff.T @ Sigma_0_inv @ diff)
    return log_prior


def log_posterior_logistic(beta, X, y, mu_0, Sigma_0_inv):
    """
    고르게 하지 않은 로그 뒤확률: log p(β|y,X) ∝ log p(y|X,β) + log p(β)
    """
    return (log_likelihood_logistic(beta, X, y) +
            log_prior_gaussian(beta, mu_0, Sigma_0_inv))


def find_map_estimate(X, y, mu_0, Sigma_0_inv):
    """
    최적화로 MAP(최대 뒤확률) 어림값 찾기.
    
    MAP = argmax_β p(β|y,X)
    """
    # 로그 뒤확률의 음수(최소화를 위해)
    def neg_log_post(beta):
        return -log_posterior_logistic(beta, X, y, mu_0, Sigma_0_inv)
    
    # 경사
    def grad(beta):
        eta = X @ beta
        prob = logistic(eta)
        grad_ll = X.T @ (y - prob)
        grad_prior = -Sigma_0_inv @ (beta - mu_0)
        return -(grad_ll + grad_prior)
    
    # 최적화
    result = minimize(neg_log_post, mu_0, jac=grad, method='BFGS')
    
    return result.x


def laplace_approximation(X, y, mu_0, Sigma_0_inv):
    """
    뒤확률의 라플라스 어림 셈하기.
    
    MAP에 가운데를 맞춘 가우스로 뒤확률을 어림한다:
    p(β|y,X) ≈ N(β|β_MAP, H⁻¹)
    
    여기서 H은 MAP에서 로그 뒤확률의 음수의 헤세 행렬이다.
    
    반환값:
    --------
    beta_map : MAP 어림값
    Sigma_laplace : 라플라스 어림의 공분산 행렬
    """
    # MAP 찾기
    beta_map = find_map_estimate(X, y, mu_0, Sigma_0_inv)
    
    # MAP에서 헤세 행렬 셈하기
    eta = X @ beta_map
    prob = logistic(eta)
    
    # 로그 가능도의 헤세 행렬
    W = np.diag(prob * (1 - prob))  # 가중값 행렬
    H_ll = -X.T @ W @ X
    
    # 로그 뒤확률의 헤세 행렬
    H = H_ll - Sigma_0_inv
    
    # 공분산은 헤세 행렬의 음수의 역행렬이다
    Sigma_laplace = np.linalg.inv(-H)
    
    return beta_map, Sigma_laplace


# 보기 1: 단순한 1차원 로지스틱 회귀
# =======================================
print("=" * 70)
print("EXAMPLE 1: 1D Logistic Regression")
print("=" * 70)

# 합성 데이터 생성
np.random.seed(42)
n_obs = 100

# 참 매개변수
beta_true = np.array([0.5, 2.0])  # [절편, 기울기]

# 특징: [1, x]
x_raw = np.random.uniform(-2, 2, n_obs)
X = np.column_stack([np.ones(n_obs), x_raw])

# 두 값 결과 만들기
eta_true = X @ beta_true
prob_true = logistic(eta_true)
y = np.random.binomial(1, prob_true)

print(f"\nGenerated {n_obs} observations")
print(f"True β = {beta_true}")
print(f"Observed: {np.sum(y)} successes, {n_obs - np.sum(y)} failures")

# 앞확률: N(0, 10I) - 정보가 약함
mu_0 = np.zeros(2)
Sigma_0 = 10 * np.eye(2)
Sigma_0_inv = np.linalg.inv(Sigma_0)

print(f"\nPrior: β ~ N(0, 10I)")

# 라플라스 어림 셈하기
beta_map, Sigma_laplace = laplace_approximation(X, y, mu_0, Sigma_0_inv)

print(f"\nMAP estimate: {beta_map}")
print(f"Laplace std: {np.sqrt(np.diag(Sigma_laplace))}")


# 제안을 달리한 중요도 표집
# ------------------------------------------

# 제안 1: 앞확률(단순하지만 효율이 낮음)
print("\n" + "-" * 70)
print("PROPOSAL 1: Using Prior as Proposal")
print("-" * 70)

n_samples = 5000
prior_dist = stats.multivariate_normal(mu_0, Sigma_0)

# 앞확률에서 표집
samples_prior = prior_dist.rvs(size=n_samples)

# 중요도 무게 셈하기
log_weights_prior = np.array([
    log_posterior_logistic(beta, X, y, mu_0, Sigma_0_inv) -
    prior_dist.logpdf(beta)
    for beta in samples_prior
])

# 무게 고르게 하기
log_weights_prior_norm = log_weights_prior - np.max(log_weights_prior)
weights_prior_unnorm = np.exp(log_weights_prior_norm)
weights_prior = weights_prior_unnorm / np.sum(weights_prior_unnorm)

# ESS
ess_prior = 1.0 / np.sum(weights_prior**2)

# 어림값
beta_est_prior = np.sum(weights_prior[:, None] * samples_prior, axis=0)

print(f"ESS: {ess_prior:.1f} ({ess_prior/n_samples:.1%})")
print(f"Estimated β: {beta_est_prior}")
print(f"Error: {np.linalg.norm(beta_est_prior - beta_true):.4f}")


# 제안 2: 라플라스 어림(훨씬 나음)
print("\n" + "-" * 70)
print("PROPOSAL 2: Using Laplace Approximation as Proposal")
print("-" * 70)

laplace_dist = stats.multivariate_normal(beta_map, Sigma_laplace)

# 라플라스 어림에서 표집
samples_laplace = laplace_dist.rvs(size=n_samples)

# 중요도 무게 셈하기
log_weights_laplace = np.array([
    log_posterior_logistic(beta, X, y, mu_0, Sigma_0_inv) -
    laplace_dist.logpdf(beta)
    for beta in samples_laplace
])

# 무게 고르게 하기
log_weights_laplace_norm = log_weights_laplace - np.max(log_weights_laplace)
weights_laplace_unnorm = np.exp(log_weights_laplace_norm)
weights_laplace = weights_laplace_unnorm / np.sum(weights_laplace_unnorm)

# ESS
ess_laplace = 1.0 / np.sum(weights_laplace**2)

# 어림값
beta_est_laplace = np.sum(weights_laplace[:, None] * samples_laplace, axis=0)

print(f"ESS: {ess_laplace:.1f} ({ess_laplace/n_samples:.1%})")
print(f"Estimated β: {beta_est_laplace}")
print(f"Error: {np.linalg.norm(beta_est_laplace - beta_true):.4f}")

print(f"\nImprovement: {ess_laplace/ess_prior:.1f}x better ESS with Laplace proposal")


# 결과를 그려 본다
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 칸 1: 자료와 맞춘 곡선
ax = axes[0, 0]
sorted_idx = np.argsort(x_raw)
x_plot = x_raw[sorted_idx]

# 참 확률
X_plot = np.column_stack([np.ones(len(x_plot)), x_plot])
prob_true_plot = logistic(X_plot @ beta_true)

# MAP 어림값
prob_map = logistic(X_plot @ beta_map)

# 중요도 표집 어림값(라플라스 제안 사용)
prob_is = logistic(X_plot @ beta_est_laplace)

ax.scatter(x_raw[y==0], y[y==0], c='red', alpha=0.5, s=50, label='y=0')
ax.scatter(x_raw[y==1], y[y==1], c='blue', alpha=0.5, s=50, label='y=1')
ax.plot(x_plot, prob_true_plot, 'k-', linewidth=3, label='True', alpha=0.7)
ax.plot(x_plot, prob_map, 'g--', linewidth=2, label='MAP')
ax.plot(x_plot, prob_is, 'r:', linewidth=2, label='IS (Laplace)')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('P(y=1|x)', fontsize=12)
ax.set_title('Logistic Regression Fit', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 칸 2: 뒤확률 표본(앞확률 제안)
ax = axes[0, 1]
scatter = ax.scatter(samples_prior[:, 0], samples_prior[:, 1],
                    c=weights_prior*n_samples, cmap='viridis',
                    s=20, alpha=0.5, edgecolors='black', linewidth=0.3)
ax.plot(beta_true[0], beta_true[1], 'r*', markersize=20, label='True')
ax.plot(beta_map[0], beta_map[1], 'go', markersize=15, label='MAP')
ax.set_xlabel('β₀ (intercept)', fontsize=11)
ax.set_ylabel('β₁ (slope)', fontsize=11)
ax.set_title(f'Prior Proposal: ESS={ess_prior:.0f}', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Weight × n')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 칸 3: 뒤확률 표본(라플라스 제안)
ax = axes[1, 0]
scatter = ax.scatter(samples_laplace[:, 0], samples_laplace[:, 1],
                    c=weights_laplace*n_samples, cmap='viridis',
                    s=20, alpha=0.5, edgecolors='black', linewidth=0.3)
ax.plot(beta_true[0], beta_true[1], 'r*', markersize=20, label='True')
ax.plot(beta_map[0], beta_map[1], 'go', markersize=15, label='MAP')
ax.set_xlabel('β₀ (intercept)', fontsize=11)
ax.set_ylabel('β₁ (slope)', fontsize=11)
ax.set_title(f'Laplace Proposal: ESS={ess_laplace:.0f}', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Weight × n')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 칸 4: 무게 분포 견주기
ax = axes[1, 1]
ax.hist(weights_prior * n_samples, bins=50, alpha=0.5, density=True,
        color='blue', edgecolor='black', label='Prior proposal')
ax.hist(weights_laplace * n_samples, bins=50, alpha=0.5, density=True,
        color='green', edgecolor='black', label='Laplace proposal')
ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Uniform')
ax.set_xlabel('Weight × n', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Weight Distributions', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 5])

plt.tight_layout()
plt.savefig('/home/claude/03_Importance_Sampling/logistic_regression_1d.png',
            dpi=300, bbox_inches='tight')


# 보기 2: 차원이 더 높은 경우
# =================================
print("\n" + "=" * 70)
print("EXAMPLE 2: Higher-Dimensional Logistic Regression")
print("=" * 70)

# 특징 p=5개인 자료 만들기
n_obs_2 = 200
p = 5

# 참된 매개변수
beta_true_2 = np.array([1.0, -0.5, 0.8, -0.3, 0.6])

# 특징
X_2 = np.random.randn(n_obs_2, p)

# 결과
eta_true_2 = X_2 @ beta_true_2
prob_true_2 = logistic(eta_true_2)
y_2 = np.random.binomial(1, prob_true_2)

print(f"\nData: n={n_obs_2}, p={p} features")
print(f"True β = {beta_true_2}")

# 앞확률
mu_0_2 = np.zeros(p)
Sigma_0_2 = 5 * np.eye(p)
Sigma_0_inv_2 = np.linalg.inv(Sigma_0_2)

# 라플라스 어림
beta_map_2, Sigma_laplace_2 = laplace_approximation(X_2, y_2, mu_0_2, Sigma_0_inv_2)

print(f"\nMAP estimate: {beta_map_2}")

# 라플라스 제안을 쓴 중요도 표집
n_samples_2 = 10000
laplace_dist_2 = stats.multivariate_normal(beta_map_2, Sigma_laplace_2)
samples_2 = laplace_dist_2.rvs(size=n_samples_2)

log_weights_2 = np.array([
    log_posterior_logistic(beta, X_2, y_2, mu_0_2, Sigma_0_inv_2) -
    laplace_dist_2.logpdf(beta)
    for beta in samples_2
])

log_weights_2_norm = log_weights_2 - np.max(log_weights_2)
weights_2_unnorm = np.exp(log_weights_2_norm)
weights_2 = weights_2_unnorm / np.sum(weights_2_unnorm)

ess_2 = 1.0 / np.sum(weights_2**2)

# 뒤확률 평균
beta_post_mean = np.sum(weights_2[:, None] * samples_2, axis=0)

# 뒤확률 표준편차
beta_post_std = np.sqrt(np.sum(weights_2[:, None] * (samples_2 - beta_post_mean)**2, axis=0))

print(f"\nImportance Sampling Results (Laplace proposal):")
print(f"  n_samples: {n_samples_2}")
print(f"  ESS: {ess_2:.1f} ({ess_2/n_samples_2:.1%})")

print("\nPosterior Estimates:")
print(f"{'Feature':<10} {'True':>8} {'MAP':>8} {'IS Mean':>8} {'IS Std':>8}")
print("-" * 50)
for i in range(p):
    print(f"β{i:<9} {beta_true_2[i]:8.3f} {beta_map_2[i]:8.3f} "
          f"{beta_post_mean[i]:8.3f} {beta_post_std[i]:8.3f}")

# 믿음 구간
credible_intervals = []
for i in range(p):
    sorted_samples = samples_2[:, i][np.argsort(weights_2)[::-1]]
    sorted_weights = np.sort(weights_2)[::-1]
    cumsum = np.cumsum(sorted_weights)
    n_95 = np.searchsorted(cumsum, 0.95) + 1
    ci = np.percentile(sorted_samples[:n_95], [2.5, 97.5])
    credible_intervals.append(ci)

print("\n95% Credible Intervals:")
for i, ci in enumerate(credible_intervals):
    contains = ci[0] <= beta_true_2[i] <= ci[1]
    status = "✓" if contains else "✗"
    print(f"β{i}: [{ci[0]:6.3f}, {ci[1]:6.3f}] {status}")


# 보기 3: 미리봄
# ===================
print("\n" + "=" * 70)
print("EXAMPLE 3: Posterior Predictive Distribution")
print("=" * 70)

# 새 시험 점
x_new = np.array([0.5, -0.3, 0.2, 0.1, -0.4])

# 뒤확률 미리봄: P(y_new=1|x_new, data)
# = ∫ P(y_new=1|x_new, β) p(β|data) dβ
# ≈ Σᵢ wᵢ × logistic(x_new'βᵢ)

pred_probs = logistic(samples_2 @ x_new)
posterior_pred_prob = np.sum(weights_2 * pred_probs)

# 참 확률
true_pred_prob = logistic(x_new @ beta_true_2)

# MAP에 바탕을 둔 미리봄
map_pred_prob = logistic(x_new @ beta_map_2)

print(f"\nPredictive probability P(y=1|x_new):")
print(f"  True: {true_pred_prob:.4f}")
print(f"  MAP: {map_pred_prob:.4f}")
print(f"  Posterior mean (IS): {posterior_pred_prob:.4f}")

# 뒤확률 미리봄 분포
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(pred_probs, bins=50, weights=weights_2, density=True,
        alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(true_pred_prob, color='red', linestyle='--', linewidth=2,
           label=f'True: {true_pred_prob:.3f}')
ax.axvline(posterior_pred_prob, color='green', linestyle='-', linewidth=2,
           label=f'Posterior mean: {posterior_pred_prob:.3f}')
ax.set_xlabel('P(y=1|x_new)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Posterior Predictive Distribution', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/claude/03_Importance_Sampling/logistic_predictive.png',
            dpi=300, bbox_inches='tight')

plt.show()

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print("""
1. 로지스틱 회귀는 켤레가 아니다:
   - 뒤확률이 닫힌 꼴이 아니다
   - 손으로 푼 풀이가 없다
   - 중요도 표집을 쓸 수 있다

2. 제안 전략:
   - 앞확률: 단순하지만 흔히 효율이 낮다(ESS이 낮다)
   - 라플라스 어림: MAP에서의 가우스
     * ESS이 훨씬 낫다(흔히 10배에서 100배 나아짐)
     * 차원이 알맞을 때 좋다
   - 많은 경우 효율 30-60%을 이룰 수 있다

3. 라플라스 어림:
   - 최적화로 MAP 찾기
   - MAP에서 헤세 행렬 셈하기
   - N(β_MAP, H⁻¹)을 제안으로 쓰기
   - 뒤확률이 거의 가우스일 때 잘 듣는다

4. 차원의 효과:
   - 앞확률 제안의 ESS은 차원에 따라 지수로 나빠진다
   - 라플라스 제안이 규모를 훨씬 잘 견딘다
   - p > 10-20이면 맞춰 가는 방법이나 MCMC이 필요할 수 있다

5. 추론 과제:
   - 뒤확률의 평균과 흩어짐
   - 믿음 구간
   - 뒤확률 미리봄 분포
   - 무게 표본으로 모두 얻을 수 있다

6. 실전에서 살필 점:
   - ESS을 늘 살펴라
   - 라플라스 어림이 대개 앞확률보다 훨씬 낫다
   - 차원이 높으면(p > 20) MCMC을 생각해 보라
   - 수치 안정: 무게는 로그 공간에서 다룬다

7. 로지스틱 회귀에서 중요도 표집의 좋은 점:
   - 독립 표본(자기상관 없음)
   - 태우기 기간이 없다
   - 나란히 하기 쉽다
   - 같은 표본에서 여러 양을 셈할 수 있다

8. 중요도 표집이 잘 듣는 때:
   - 알맞은 차원(p < 20)
   - 좋은 제안을 쓸 수 있을 때(라플라스, 변분)
   - 자료가 극단으로 갈리지 않을 때
   - 넉넉한 표본 크기(n > 100)

9. MCMC과 견주기:
   - 중요도 표집: 좋은 제안이 있을 때 더 낫다
   - MCMC: 차원이 높거나 뒤확률이 복잡할 때 낫다
   - 중요도 표집: 진단하기가 더 쉽다(ESS)
   - MCMC: 제안을 고르는 데 더 너그럽다
""")


if __name__ == "__main__":
    pass
```

## 논의

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 무늬는 더 복잡한 상황으로 자연스럽게 넓어진다. 웃매개변수, 구조의 변형, 서로 다른 자료 묶음을 이리저리 시험해 보면 이해가 깊어지고 표집과 어림 일감에 대한 실전 직관이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 죽 읽고 핵심 설계 결정을 가려내어라. 구체적인 구현 고름 셋을 적고 저마다 왜 몬테카를로 방법에 알맞은지 설명하여라.

??? success "연습문제 1 풀이"
    설계 결정은 구현마다 다르지만 흔히 다음이 포함된다. (1) 활성화 함수의 선택 — ReLU 계열은 포화되지 않는 경사를 주어 학습을 빠르게 한다. (2) 정규화 전략 — 배치 정규화는 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 — 있을 경우 건너뛰는 경로를 제공하여 깊은 신경망에서도 경사가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 절충을 반영한다.

---

**연습문제 2.**
입력이 기대하는 모양과 자료형을 갖는지 확인하도록 주 함수나 클래스에 입력 검증을 추가하라. 잘못된 입력에는 유익한 오류 메시지를 내라.

??? success "연습문제 2 풀이"
    `forward` 메서드(또는 해당 함수)의 첫머리에 다음과 같은 검사를 추가한다. `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`와 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. 모양을 검증할 때는 중요한 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 유익한 오류 메시지는 디버깅 속도를 크게 높이고 코드를 재사용하기에도 더 견고하게 만든다.

---

**연습문제 3.**
이 몬테카를로 방법 구현의 자리에서 치우침과 흩어짐의 주고받음을 설명하여라. 핵심 웃매개변수가 이 주고받음에 어떻게 영향을 주는가?

??? success "연습문제 3 풀이"
    몬테카를로 방법에서 치우침과 흩어짐의 주고받음은 모형의 복잡함과 표본 크기로 드러난다. 더 복잡한 모형(이를테면 섞음 성분이 더 많거나 층이 더 깊은 모형)은 치우침을 줄이지만 흩어짐을 키우며, 자료가 적을 때 특히 그렇다. 핵심 웃매개변수가 이를 다스린다. 앞확률의 세기가 벌주기 노릇을 하고(센 앞확률은 흩어짐을 줄이지만 치우침을 키울 수 있다), 표본 크기가 어림의 정확도에 영향을 주며(표본이 많을수록 흩어짐이 줄고), 모형의 복잡함이 유연함을 정한다. 가장 좋은 균형은 쓸 수 있는 자료의 양과 바탕 분포의 참된 복잡함에 달렸다.

---

**연습문제 4.**
로지스틱 회귀 중요도 표집 구현을 확인하는 두루 갖춘 시험 함수를 적어라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단적인 값(0, 아주 큰 수)이 든 입력 같은 모서리 경우를 시험하여라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_logistic regression is():
        model = Logistic Regression IS(...)
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
