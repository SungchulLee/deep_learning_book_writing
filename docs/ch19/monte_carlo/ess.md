# 유효 표본 크기

중요도 샘플링의 웨이트 ESS와 MCMC의 자기상관 기반 ESS는 서로 다른 진단이다. 전자는 관측한 웨이트 집중도, 후자는 특정 함수의 평균에 대한 점근 분산과 연결된다. 두 값을 모두 정확한 정보량이나 같은 범위의 독립 표본 수로 해석하지 않는다.

## 1. 문제: 표본이 다 같지는 않다

중요도 표본은 제안분포에서 생성되며 밀도비 웨이트를 받는다. 소수 웨이트가 대부분을 차지하면 정규화 평균이 몇 개 값에 민감해질 수 있다. 하지만 함수별 가중 기여·누락된 모드·꼬리도 함께 살펴야 한다.

MCMC 표본에는 시점 간 자기상관이 있으며 평균의 정확도는 관심 함수 f에 따라 다르다. 양의 상관은 보통 오차를 늘리지만 음의 상관은 독립 표본보다 분산을 줄일 수 있다. MCMC에서 독립 표본이 원리적으로 불가능한 것도 아니다. 예를 들어 모든 행이 π인 유한 커널은 한 단계 후 독립 목표표본을 생성한다.

## 2. 중요도 샘플링의 웨이트 ESS

음이 아닌 유한 웨이트 w_i의 합이 양수이고 w̄_i=w_i/∑w_j이면 다음을 정의한다.

$$ESS_w=\frac1{\sum_i\bar w_i^2}=\frac{(\sum_iw_i)^2}{\sum_iw_i^2}$$

범위는 [1,N]이며 모두 같으면 N, 한 웨이트에 전부 집중하면 1이다. 유한 표본에서 웨이트가 비슷했다는 사실이 q=p를 증명하지는 않는다. 보지 못한 영역이나 재표집 중복을 이 값이 감지하지 못할 수 있다.

Var_π(f)/ESS_w를 실제 IS·SNIS 분산의 일반 공식으로 사용하지 않는다. 표준 IS에는 E_q[w²f²], SNIS에는 E_q[w²(f−I)²]가 들어가므로 f와 웨이트의 결합이 중요하다. 더 자세한 [웨이트 ESS](importance_sampling/ess.md)를 참고한다.

### 로그 웨이트 구현

함수는 비어 있지 않은 1차원 로그 웨이트를 가정한다. 모두 −∞·NaN·+∞인 입력은 별도 처리해야 한다. 아래 좋은·나쁜 예는 임의 정규 로그 웨이트의 집중도 비교이며 특정 목표·제안 밀도쌍을 검증한 결과는 아니다.

```python
import torch

def importance_sampling_ess(log_weights: torch.Tensor) -> float:
    """
    고르게 하지 않은 로그 무게로 ESS 셈하기.
    
    인수:
        log_weights: log w_i을 담은 꼴 (N,)의 텐서
    
    반환값:
        실수로 나타낸 ESS
    """
    # 수치 안정을 위해 로그 공간에서 고르게 하기
    log_w_norm = log_weights - torch.logsumexp(log_weights, dim=0)
    
    # ESS = 1 / sum(w_bar^2) = exp(-logsumexp(2 * log_w_norm))
    ess = torch.exp(-torch.logsumexp(2 * log_w_norm, dim=0))
    
    return ess.item()

# 예
N = 1000
# 경우 1: 좋은 제안(무게가 대체로 고름)
log_w_good = torch.randn(N) * 0.5
print(f"Good proposal ESS: {importance_sampling_ess(log_w_good):.0f} / {N}")

# 경우 2: 나쁜 제안(큰 무게가 몇 개뿐)
log_w_bad = torch.randn(N) * 5.0
print(f"Poor proposal ESS: {importance_sampling_ess(log_w_bad):.0f} / {N}")
```

**출력:**

```
Good proposal ESS: 804 / 1000
Poor proposal ESS: 8 / 1000
```

---

## 3. MCMC의 ESS

정상분포에서 시작한 함수열 f(X_t)의 분산이 유한하고 자기상관이 충분히 합산 가능하며 적절한 중심극한정리를 만족한다고 하자. 점근적으로 다음을 사용한다.

$$\tau_f=1+2\sum_{k=1}^\infty\rho_f(k),\qquad ESS_f\approx N/\tau_f$$

$$\operatorname{Var}(\bar f_N)\approx\operatorname{Var}_\pi(f)/ESS_f$$

이는 유한 N의 정확한 식이 아니다. 정확한 정상 분산식에는 k=1,…,N−1과 웨이트 1−k/N이 들어간다. 음의 자기상관이면 ESS_f>N일 수 있으므로 웨이트 ESS의 [1,N] 범위를 그대로 적용하지 않는다.

실제 자기상관 합은 추정 잡음 때문에 안정적인 절단·스펙트럼 방법이 필요하다. 평균·분위수·꼬리 등 관심량별 진단을 사용하고 여러 체인의 수렴과 몬테카를로 표준오차를 함께 보고한다. [MCMC 진단](../mcmc/diagnostics.md)을 참고한다.

## 4. ESS 지켜보고 낫게 하기

중요도 샘플링에서는 지지집합·관심 적분에 맞는 제안·꼬리·다봉 탐색을 검토한다. 적응이나 차원 축소가 항상 ESS와 정확도를 항상 개선하는 것은 아니다.

MCMC에서는 제안 스케일·파라메터화·알고리즘을 목표 기하와 비용에 맞춰 조정한다. 비중심화가 모든 계층 모형에서 낫거나 HMC가 모든 목표에서 랜덤 워크보다 효율적이라고 단정하지 않는다.

ESS/초는 같은 관심 함수와 동일한 진단·컴퓨팅 조건에서 비교한다. 적응·워밍업·경사 평가 등의 비용을 포함하고, 체인 길이를 늘리면 ESS가 선형으로 증가한다는 주장은 정상성·안정적인 자기상관이라는 조건에서의 근사다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
적분이 유한해도 표준 IS의 분산이 무한할 수 있는 이유는?

</div>

??? success "풀이"
    적분 가능성은 $\int|f|p<\infty$이지만 유한 분산에는 $\int f^2p^2/q<\infty$가 필요하다. q가 필요한 영역을 덮어도 꼬리의 이차 적률이 발산할 수 있다. 표본에 관측한 유한 최대 웨이트가 유한한 모집단 분산의 증거는 아니다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
웨이트 ESS 식을 정규화 웨이트에서 유도하고 해석의 한계를 설명하라.

</div>

??? success "풀이"
    w̄_i=w_i/∑w_j를 1/∑w̄_i²에 대입하면 (∑w_i)²/∑w_i²다. 이는 웨이트 집중도이며 모든 함수에서 같은 분산을 갖는 독립 표본 수라는 정확한 정의는 아니다. MCMC 자기상관 ESS와도 구분한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
표준 IS의 최적 제안과 SNIS의 점근 최적 제안을 구분하라.

</div>

??? success "풀이"
    표준 IS는 q*∝|f|p이며 f가 음이 아니고 적분이 양수이면 분산 0이다. SNIS는 |f−I|p에 비례하는 점근 최적 제안을 갖는다. 정규화 상수·미지 I와 표본 생성 때문에 실전에서는 근사 제안을 사용한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
표준정규 목표에서 아래 t 제안의 X² 추정량과 분산 조건을 설명하라.

</div>

??? success "풀이"

    코드는 원시 밀도비를 사용한 표준 IS 평균이므로 정규화된 목표·제안의 지지집합 조건 아래 불편하다. 자유도 df>0인 t 제안의 다항 꼬리는 이 정규 목표·다항 함수의 가중 이차 적률을 유한하게 한다. t 자체의 분산 존재 조건과는 다르며 모든 임의의 h에서도 유한 분산이라는 뜻은 아니다.

    ```python
    import numpy as np
    from scipy import stats

    def importance_sampling_x_squared(n_samples=10000, df=5):
        target = stats.norm(0, 1)
        proposal = stats.t(df=df)
        x = proposal.rvs(n_samples)
        weights = target.pdf(x) / proposal.pdf(x)
        f_x = x ** 2
        estimate = np.mean(weights * f_x)
        return estimate  # 1.0에 가까워야 함

    print(f"Estimate: {importance_sampling_x_squared():.4f}")
    print(f"True value: 1.0000")
    ```


## 정리하며

웨이트 ESS는 [1,N]의 집중도 지표이고, MCMC ESS는 함수별 자기상관과 점근 분산의 진단이다. 후자는 N보다 클 수 있다. 두 경우 모두 고정 임계값만으로 수렴·정확도·꼬리 탐색을 보장하지 않는다.

**참고 문헌**

1. Kong, A. (1992). A note on importance sampling using standardized weights. *Technical Report*, University of Chicago.
2. Liu, J. S. (2001). *Monte Carlo Strategies in Scientific Computing*. Springer. 2장.
3. Vehtari, A., et al. (2021). Rank-normalization, folding, and localization: An improved $\hat{R}$ for assessing convergence of MCMC. *Bayesian Analysis*, 16(2), 667-718.
