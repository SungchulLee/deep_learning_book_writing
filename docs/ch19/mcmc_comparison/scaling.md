# 차원에 따른 스케일링

스케일링은 목표군·초기조건·오차 척도와 연산 단위를 정해야 의미가 있다. 반복 수, Leapfrog 수, 총 평가비용과 함수별 ESS를 혼합한 표로 보편적 순위를 만들지 않는다.

## 1. MCMC에서의 차원의 저주

반지름 r인 균등 공에서 고정 0<δ<r의 바깥 껍질 부피 비는 1−(1−δ/r)^d여서 d→∞에 1로 간다. 이는 그 공의 균등 측도에 대한 결과다.

독립 X,Y~N(0,I_d)에서는 ‖X‖/√d→1, ‖X−Y‖/√d→√2가 확률로 성립한다. 같은 식을 임의 분포의 무작위 점에 적용하지 않는다. 높은 차원에서 모드와 전형 질량 영역이 다를 수 있지만 모든 목표가 하나의 구형 껍질을 갖지는 않는다. 차원 증가가 조건부 독립·희소 구조·직접 생성의 이점을 없애는 것도 아니다.

## 2. 방법마다의 스케일링 분석

대표적인 정칙 곱 목표의 정상상태 점근 분석을 기준으로 다음 단위를 구분한다.

| 방법 | 스텝 관례·차원 스케일 | 대표 연산 스케일 |
|---|---|---|
| RWM | 제안 표준편차 d⁻¹/² | 한 좌표의 O(d) 반복 시간 스케일 |
| MALA | 제안 표준편차 d⁻¹/⁶, 분산 d⁻¹/³ | O(d¹/³) 반복 시간 스케일 |
| HMC | 적분 스텝 d⁻¹/⁴, 적분시간 O(1) | 제안당 Leapfrog O(d¹/⁴) |

밀도·경사 평가를 O(d)로 가정하면 각각 O(d²), O(d⁴/³), O(d⁵/⁴)의 대표 비용 차수를 논할 수 있다. 이것을 임의 목표의 worst-case TV 혼합시간·정확 독립 표본 비용이라고 동일시하지 않는다.

독립 좌표의 체계적 깁스는 한 스윕에 모든 좌표를 독립 목표로 바꾸며, 스윕 비용은 조건부 생성이 O(1)씩이면 O(d)다. 한 좌표 갱신을 한 반복으로 부르는 관례와 구분한다. 강한 상관·제약에서의 깁스 시간은 보편적인 O(d)~O(d²) 범위로 묶을 수 없다.

## 3. 경험적 측정 조각의 의미

아래 measure_mixing_time은 compute_ess라는 외부 함수로 첫 좌표 ESS가 목표에 도달할 때까지 센다. worst-case TV 혼합시간을 측정하는 함수가 아니다. 빈·짧은 체인의 ESS가 정의되지 않을 수 있고 sampler.step이 위치만 반환하는 래퍼를 가정한다. HMC의 (위치,수락여부) 반환값에는 그대로 맞지 않는다.

시작점은 모두 0이고 warmup·스텝 적응·실제 경사 평가 비용을 통제하지 않는다. 반복 실행도 같은 sampler 객체의 내부 상태가 유지될 수 있다. 근거 실행이 제시되지 않은 숫자·맞춘 지수를 실험 결과로 표기하지 않는다.

plot_scaling의 기준선 d²·d^1.67·√d는 임의 배율의 참조 곡선이며 이론 검증 결과가 아니다. 총 비용과 반복의 단위를 맞추고 불확실성을 포함한 반복 실험을 설계해야 한다.

```python
import numpy as np
from scipy.stats import multivariate_normal

def measure_mixing_time(sampler, d, n_chains=10, target_ess=100):
    """목표 ESS에 이르는 섞임 시간 어림하기."""
    times = []
    
    for _ in range(n_chains):
        samples = []
        x = np.zeros(d)
        
        n_steps = 0
        while compute_ess(samples) < target_ess:
            x = sampler.step(x)
            samples.append(x[0])  # 첫 좌표 기록
            n_steps += 1
            
            if n_steps > 1e7:  # 시간 넘김
                break
        
        times.append(n_steps)
    
    return np.median(times)

def scaling_experiment(sampler_class, dimensions=[10, 20, 50, 100, 200]):
    """차원에 따른 크기 변화 재기."""
    results = []
    
    for d in dimensions:
        # 표준 가우스 타깃
        target = lambda x: -0.5 * np.sum(x**2)
        grad_target = lambda x: -x
        
        sampler = sampler_class(target, grad_target, d)
        tau = measure_mixing_time(sampler, d)
        
        results.append({'d': d, 'tau': tau})
    
    return results
```
```python
import matplotlib.pyplot as plt

def plot_scaling(results_dict, dimensions):
    """크기 변화 견줌 그리기."""
    plt.figure(figsize=(10, 6))
    
    for method, results in results_dict.items():
        taus = [r['tau'] for r in results]
        plt.loglog(dimensions, taus, 'o-', label=method)
    
    # 기준선
    d = np.array(dimensions)
    plt.loglog(d, d**2 / 10, '--', alpha=0.5, label='$O(d^2)$')
    plt.loglog(d, d**1.67 / 5, '--', alpha=0.5, label='$O(d^{5/3})$')
    plt.loglog(d, d**0.5 * 10, '--', alpha=0.5, label='$O(d^{1/2})$')
    
    plt.xlabel('Dimension $d$')
    plt.ylabel('Mixing Time $\\tau$')
    plt.legend()
    plt.title('MCMC Scaling with Dimension')
    plt.grid(True, alpha=0.3)
```

## 4. 비용·ESS와 혼합시간의 구분

관심 함수 f에서 총 비용 C, 유효한 ESS_f가 있으면 비용/ESS_f를 비교할 수 있다. 정상 CLT와 안정적인 자기상관 추정 아래 N/ESS_f는 적분 자기상관시간 τ_f와 연결된다.

τ_f를 worst-case TV 혼합시간 τ_mix와 정확히 같은 것으로 대입하면 안 된다. 함수·시작분포·거리와 단위가 다르다. HMC의 제안당 L개 경사와 스케일링의 L을 이미 포함한 비용을 다시 곱해 중복 계산하지 않는다. 경사 평가·밀도 평가·행렬 곱의 실제 비용과 워밍업을 포함한다.

## 5. 설정 오류와 탐색 병목

작거나 큰 제안 폭은 높은 수락·작은 이동 또는 낮은 수락·정체를 만들 수 있다. 아래 RWM 예제는 고정 σ=0.1과 원점 시작의 1000회 수락률을 비교하며 최적 스케일·정상상태 결과를 보이는 실험은 아니다.

MALA·HMC의 큰 스텝·비유한 경사·급변 곡률을 점검한다. 긴 HMC 궤적의 유턴은 계산 비효율일 수 있지만 그 자체가 에너지 발산의 원인은 아니다. 모드 장벽과 수치 불안정성·수락 보정을 구분한다.

```python
def rwm_acceptance_vs_dimension(sigma_fixed=0.1):
    """스텝 크기를 붙박았을 때 무작위 스텝 메트로폴리스가 무너짐을 보이기."""
    dimensions = [10, 50, 100, 500, 1000]
    
    for d in dimensions:
        target = lambda x: -0.5 * np.sum(x**2)
        
        accepts = 0
        x = np.zeros(d)
        
        for _ in range(1000):
            x_prop = x + sigma_fixed * np.random.randn(d)
            log_alpha = target(x_prop) - target(x)
            
            if np.log(np.random.rand()) < log_alpha:
                x = x_prop
                accepts += 1
        
        print(f"d={d}: acceptance rate = {accepts/1000:.3f}")

# 내임:
# d=10: 받아들임 비율 = 0.712
# d=50: 받아들임 비율 = 0.089
# d=100: 받아들임 비율 = 0.003
# d=500: 받아들임 비율 = 0.000
# d=1000: 받아들임 비율 = 0.000
```

## 6. 높은 차원을 위한 전략

정규화·whitening·희소 연산과 블록·주변화가 차원 증가의 비용을 줄일 수 있다. 아래 preconditioned_hmc는 고정 가역 선형변환 x=Lz를 사용하고 경사를 Lᵀ∇_xlog π로 바꾼다. 고정 log det L는 MH 비에서 소거되지만 정규화된 밀도나 증거 계산에는 필요하다. 이 조각은 평균을 빼지 않은 변환이며 hmc 래퍼가 외부 정의다.

PCA·무작위 사영·오토인코더로 차원을 단순히 줄이면 일반적으로 원래 사후 목표도 바뀐다. 나머지 좌표를 올바르게 주변화·조건부 처리하지 않고 같은 정확 샘플러라고 할 수 없다. 여러 체인의 속도 향상도 작업자·워밍업·통신 비용에 달리며 봉우리 수까지 선형이라는 법칙은 없다.

```python
def preconditioned_hmc(log_prob, grad_log_prob, x0, Sigma, n_samples):
    """미리 다듬기를 붙인 HMC."""
    L = np.linalg.cholesky(Sigma)
    L_inv = np.linalg.inv(L)
    
    # 하얗게 만든 공간으로 바꾸기
    def log_prob_white(z):
        x = L @ z
        return log_prob(x)
    
    def grad_log_prob_white(z):
        x = L @ z
        return L.T @ grad_log_prob(x)
    
    # 하얗게 만든 공간에서 HMC 돌리기
    z0 = L_inv @ x0
    z_samples = hmc(log_prob_white, grad_log_prob_white, z0, n_samples)
    
    # 되돌려 바꾸기
    return np.array([L @ z for z in z_samples])
```

## 7. 차원에 따른 맞추기

차원 지수는 목표군·표준화와 스텝 단위에 대한 출발점이다. MALA의 표준편차 d⁻¹/⁶를 분산 ε를 쓰는 함수에 그대로 넣지 않는다. HMC의 εL는 전형 반지름이 아니라 동역학의 물리적 시간이다.

M=I의 표준정규 정확 흐름은 x(t)=x₀cos t+p₀sin t다. t=π/2이면 x(t)=p₀여서 한 번의 운동량 생성으로 독립 목표 위치를 얻는다. 이 시간은 d와 무관하며 반지름 √d에 비례하지 않는다. 수치 HMC는 ε∝d⁻¹/⁴에서 고정 시간을 유지하려면 L∝d¹/⁴를 사용한다는 대표 분석과 연결된다. 실제 스케일·공진·비선형 목표에는 별도 조정이 필요하다.

## 8. 실전에서의 스케일링: 사례

베이즈 뉴럴 넷트웍은 파라메터 수뿐 아니라 자료 수·네트워크 계산 그래프·대칭과 모드를 고려한다. SGLD를 사용하면 목표·근사·잡음 보정을 확인해야 하며 HMC의 경사를 단순 미니배치로 바꾼 것과 같지 않다.

가우스 과정의 잠재값을 샘플링하면 자료 수가 상태 차원이 될 수 있지만 가우스 우도에서는 잠재값을 해석적으로 다룰 수도 있다. 유도점 근사는 원래 목표와 다른 근사를 도입한다. 계층 모형의 비중심화도 관측 정보가 강한 경우 항상 유리하지 않다. 모형의 구조와 연산 비용을 구체적으로 비교한다.

## 9. 스케일링을 보고하는 기준

보편적인 실전 차원 한계 20·100·1000은 없다. 큰 독립 가우스는 직접 생성하기 쉽고 낮은 차원도 분리 모드·급변 곡률에서 어려울 수 있다.

보고에는 목표군·차원·조건수·초기화·스텝 정의·워밍업·전이 단위·총 평가 수와 관심 함수의 MCSE를 포함한다. 적절한 조건의 점근 지수와 실제 유한 실험을 구분하며 기준 곡선을 실측 결과로 바꾸어 소개하지 않는다.

## 연습문제

1. MALA의 표준편차가 d⁻¹/⁶이면 제안 분산은 어떤 차수인가? d⁻¹/³이다. 구현의 ε가 어느 쪽인지 확인해야 한다.

2. 독립 좌표 깁스의 O(d) 좌표 갱신과 한 스윕을 구분하라. 한 스윕은 d번 좌표 생성이며 모든 좌표를 새로 뽑을 수 있다. 비용과 반복을 같은 단위로 비교한다.

3. 표준정규 정확 HMC의 π/2 시간이 √d인가? 아니다. x(t)=x₀cos t+p₀sin t에서 각 모드의 주파수가 1이라 π/2다.

4. ESS 도달 반복 수가 TV 혼합시간인가? 특정 함수의 정밀도 도달에 대한 경험적 수치일 뿐이다. 초기 영향·다른 함수·worst-case TV를 대표하지 않는다.

5. 알려진 고차원 가우스는 왜 좋은 비교 기준인가? 직접 생성과 정확 모멘트를 사용할 수 있어 수치 오차를 평가할 수 있다. 그 목표에서의 성능을 모든 사후로 일반화하지 않는다.

## 정리하며

스케일링에는 목표군·스텝 단위·전이와 평가비용·관심 함수의 오차를 명시한다. 점근 반복 스케일을 모든 목표의 혼합시간으로 바꾸거나 비용을 중복 곱하지 않는다.

**참고 문헌**

1. Roberts, G. O., & Rosenthal, J. S. (2001). "Optimal Scaling for Various Metropolis-Hastings Algorithms." *Statistical Science*.
2. Beskos, A., et al. (2013). "Optimal Tuning of the Hybrid Monte Carlo Algorithm." *Bernoulli*.
3. Livingstone, S., et al. (2019). "On the Geometric Ergodicity of Hamiltonian Monte Carlo." *Bernoulli*.
4. Belloni, A., & Chernozhukov, V. (2009). "On the Computational Complexity of MCMC-Based Estimators in Large Samples." *Annals of Statistics*.
5. Chopin, N., & Ridgway, J. (2017). "Leave Pima Indians Alone: Binary Regression as a Benchmark for Bayesian Computation." *Statistical Science*.
