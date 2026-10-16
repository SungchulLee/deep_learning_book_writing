# 질량 행렬

질량 행렬은 운동량의 분포와 속도 변환을 함께 정한다. 행렬·역행렬의 이름은 구현마다 다를 수 있으므로 이 절에서는 K=pᵀM⁻¹p/2와 p~N(0,M)라는 관례를 기준으로 설명한다.

## 1. 질량 행렬이 하는 일

상수 대칭 양의 정부호 M에서 운동량은 p~N(0,M), 속도는 u=ẋ=M⁻¹p다. 따라서 정상 운동량을 생성했을 때 속도의 공분산은 M⁻¹다.

같은 p를 고정하면 큰 질량은 느린 속도를 준다. 동시에 p의 분포도 M에 따라 바뀌므로 같은 운동량 비교와 평균적인 속도 스케일을 구분한다. 좁은 목표 방향에서는 더 큰 질량이 속도를 줄여 적분 주파수를 맞출 수 있다.

M=I도 올바른 커널을 구성할 수 있다. 목표의 이방성·상관에서 비효율적일 수 있다는 것이지 목표 불변성이 바로 무너지는 것은 아니다.

## 2. 기하학적 해석

운동량 p는 여접벡터, 속도 u=M⁻¹p는 접벡터다. 같은 운동에너지를 다음처럼 표현한다.

$$K=\frac12p^TM^{-1}p=\frac12u^TMu.$$

따라서 위치의 접벡터 계량은 M, 운동량의 쌍대 계량은 M⁻¹이다. 두 공간의 계량을 반대로 부르지 않는다.

U=xᵀAx/2, A>0이면 ẍ=−M⁻¹Ax다. 주파수 제곱은 대칭 양의 정부호 M⁻¹/²AM⁻¹/²의 고윳값이고 M=A를 고르면 모두 1이다. M=A⁻¹이라는 선택은 이 관례에서 주파수를 맞추지 않는다.

## 3. 가우스 목표의 주파수 맞추기

가우스 목표 N(μ,Σ)에서 A=Σ⁻¹이므로 M=Σ⁻¹는 모든 모드 주파수를 같게 하는 자연스러운 선택이다. 공통 양의 상수배는 시간 단위를 바꾼다. 비용 목적·적분시간·실제 목표에 무관한 유일한 최적 행렬이라는 정리는 아니다.

예를 들어 Σ=diag(1,0.01)이면 M=diag(1,100)에서 속도 표준편차는 각각 1·0.1이다. 좁은 방향에서 이동 속도가 작아지면서 두 방향의 상대적 동역학이 맞춰진다.

정준 표준화 z=Σ⁻¹/²(x−μ), r=Σ¹/²p를 함께 사용하면 위치·운동량이 모두 표준정규이고 H=(‖z‖²+‖r‖²)/2다. 일반 목표의 공분산 역행렬은 전역 스케일에 대한 휴리스틱이며 국소 곡률·다봉성·존재하지 않는 공분산 문제까지 해결하지 않는다.

## 4. 질량 행렬의 클래스

스칼라 M=mI는 파라메터 1개, 대각 M은 d개, 밀집 대칭 M은 d(d+1)/2개를 갖는다. 스칼라·대각 곱셈은 O(d), 밀집 곱셈은 O(d²)이며 밀집 인수분해는 갱신할 때 보통 O(d³)다.

대각 방식은 주변 스케일을 맞추지만 상관을 회전시켜 제거하지 않는다. 이 관례에서 m_i≈1/Var(x_i)를 출발점으로 사용할 수 있다. 밀집 방식은 상관을 반영할 수 있지만 더 많은 워밍업 정보·저장·계산이 필요하다. 기본값과 metric·inverse metric의 명칭은 사용한 라이브러리의 실제 운동에너지 정의를 확인한다.

## 5. 워밍업과 온라인 공분산

워밍업의 구간별 표본으로 공분산을 추정하고 정규화한 뒤 질량 또는 역질량을 갱신할 수 있다. 구간 길이 75·900·1000은 보편적인 알고리즘 명세가 아니며 전체 길이·구현에 따라 달라진다.

아래 Welford 방식은 각 입력을 한 번 처리해 평균·공분산을 누적한다. 메모리는 O(d²)다. M2/(n−1)은 독립 표본의 익숙한 표본공분산이지만 상관·비정상 워밍업 자료에서 곧바로 불편한 사후 공분산 추정이라는 뜻은 아니다. n<2에서 항등행렬을 반환하는 것은 초기화 규칙이다.

공분산과 대각을 혼합하면 비대각 잡음을 줄일 수 있지만 대각이 0인 방향까지 자동으로 양의 정부호가 되지는 않는다. 양의 ridge λI를 더하는 방식은 원래 공분산이 양의 준정부호일 때 정부호를 만들 수 있다. 표본 품질·고윳값·스텝 크기의 재조정을 함께 고려한다.

```python
class OnlineCovariance:
    def __init__(self, dim):
        self.n = 0
        self.mean = np.zeros(dim)
        self.M2 = np.zeros((dim, dim))
    
    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += np.outer(delta, delta2)
    
    def covariance(self):
        if self.n < 2:
            return np.eye(len(self.mean))
        return self.M2 / (self.n - 1)
```

## 6. 구현의 세부

M=LLᵀ인 Cholesky 인수를 저장하면 p=Lz, z~N(0,I)로 생성한다. L a=p를 풀면 K=‖a‖²/2이고 이어 Lᵀu=a를 풀면 u=M⁻¹p다. 명시적 역행렬을 만들기보다 선형계 풀이를 사용할 수 있다. 아래 조각에는 np·solve_triangular 등의 외부 정의가 필요하다.

대각 질량의 각 원소는 양수여야 하며 운동량·에너지·속도 계산에 같은 관례를 사용한다. m1,…,md는 설명용 자리표시자이지 완결된 배열 입력이 아니다.

safe_inverse는 대각값만 하한으로 제한하고 실패하면 의사역행렬을 반환한다. 이것만으로 전체 행렬의 양의 정부호를 보장할 수 없다. 큰 비대각 성분으로 부정부호일 수도 있고 특이 공분산의 의사역은 특이할 수 있어 HMC의 Cholesky 조건을 만족하지 못한다. 대칭화와 대각 하한을 충분한 안정성 증명으로 읽지 않는다.

```python
v = L @ np.random.randn(d)  # v ~ N(0, M)
```
```python
z = solve_triangular(L, v, lower=True)  # L z = v
K = 0.5 * np.dot(z, z)  # K = ½ vᵀ M⁻¹ v = ½ |z|²
```
```python
# M⁻¹ v = (L Lᵀ)⁻¹ v = L⁻ᵀ L⁻¹ v
z = solve_triangular(L, v, lower=True)
velocity = solve_triangular(L.T, z, lower=False)
```
```python
# 대각과 그 제곱근 저장
m_diag = np.array([m1, m2, ..., md])
sqrt_m = np.sqrt(m_diag)
inv_m = 1.0 / m_diag

# 운동량 표집
v = sqrt_m * np.random.randn(d)

# 운동 에너지
K = 0.5 * np.sum(v**2 * inv_m)

# 속도
velocity = v * inv_m
```
```python
def safe_inverse(Sigma, min_var=1e-6):
    """공분산 행렬을 안전하게 뒤집기."""
    # 최소 흩어짐 지키기
    Sigma = Sigma.copy()
    np.fill_diagonal(Sigma, np.maximum(np.diag(Sigma), min_var))
    
    # 거의 특이한 행렬에는 유사 역행렬 쓰기
    try:
        M = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        M = np.linalg.pinv(Sigma)
    
    # 대칭 지키기
    M = 0.5 * (M + M.T)
    
    return M
```

## 7. 전처리와 운동량 항

한 위치 갱신은 다음과 같이 쓸 수 있다.

$$x_{n+1}=x_n+\epsilon M^{-1}p_n+\frac{\epsilon^2}{2}M^{-1}\nabla\log\pi(x_n).$$

운동량 항을 빼면 확률적 이동을 경사 최적화로 잘못 해석하게 된다. 새 운동량을 생성한 한 단계에서는 첫 잡음 항의 공분산이 ε²M⁻¹이다. 다단계 HMC는 같은 운동량을 이어 사용하므로 매번 독립 잡음을 더하는 알고리즘과 다르다.

이차 목표에서 전처리된 곡률의 고윳값 비는 주파수 분리의 지표다. M=A이면 1이 되지만 낮은 조건수가 모든 비선형 목표의 빠른 혼합을 보장하지는 않는다. 국소 헤세 행렬이 부정부호일 때 역행렬을 그대로 질량으로 쓸 수도 없다.

## 8. 리만 HMC(맛보기)

위치마다 달라지는 양의 정부호 M(x)를 사용하면 다음 결합 에너지가 필요하다.

$$H=U(x)+\frac12\log\det M(x)+\frac12p^TM(x)^{-1}p.$$

로그 행렬식은 조건부 가우스 운동량을 정규화해 위치 주변분포를 유지한다. 위치 의존 계량이 변화하는 기하에 도움이 될 수 있지만 추가 미분·선형대수·일반화 적분 비용이 든다. 표준 Leapfrog에 M(x)만 대입하는 것으로 같은 정당화를 얻을 수는 없다.

## 9. 실전 권고

대각·밀집 선택은 차원 100·200이나 상관 0.8 같은 고정 경계로 결정하지 않는다. 추정 가능한 워밍업 정보, 상관 구조, 경사 계산과 행렬 연산 비용, 함수별 시간당 오차를 비교한다.

아래 estimated_cov @ M이 I에 가까운지는 공분산 맞춤의 대략적인 확인이다. 대각 원소만 1에 가깝다고 비대각까지 제거되었다고 볼 수 없다. 실제 정준 표준화는 위치와 운동량을 함께 변환한다.

cond(M)이 크다는 것은 질량 자체의 이방성을 나타낼 수 있으며 잘 맞춘 좁은 목표에서는 필요한 경우도 있다. M만의 조건수로 전처리된 동역학의 상태를 단정하지 않고 관련 곡률·주파수와 수치 풀이의 안정성을 함께 살핀다.

```python
# 달굼 뒤
estimated_cov = np.cov(warmup_samples.T)
M_inv = np.linalg.inv(M)

# 거의 항등 행렬이어야 한다
whitened = estimated_cov @ M  # 잘 맞춰졌으면 ≈ I
print("Whitening check:", np.diag(whitened))  # ≈ 1이어야 함
```
```python
cond_number = np.linalg.cond(M)
if cond_number > 1e6:
    print("Warning: Mass matrix poorly conditioned")
```

## 연습문제

1. Σ=[[1,0.9],[0.9,1]]에서 M=I와 대각 분산 기반 M이 다른가? 두 주변분산이 1이므로 여기서는 같다. 밀집 Σ⁻¹는 상관 회전을 반영해 주파수를 맞춘다.

2. 운동량과 속도의 공분산을 유도하라. p~N(0,M), u=M⁻¹p에서 Cov(u)=M⁻¹MM⁻¹=M⁻¹다. 접공간 계량 M과 쌍대 계량 M⁻¹를 구분한다.

3. 공분산 C와 질량 C⁻¹를 사용할 때 정규화가 왜 필요한가? 적은·상관된 워밍업 표본은 불안정하거나 특이한 C를 만들 수 있어 고윳값·ridge·추정 구간을 확인해야 한다.

4. safe_inverse가 양의 정부호를 보장하는가? 아니다. 대각 하한·대칭화는 부정부호를 배제하지 못하고 의사역은 특이할 수 있다. Cholesky가 가능한 유효 행렬인지 별도로 확인한다.

5. 대각 whitening check만으로 충분한가? C M의 비대각 오차도 봐야 한다. 또한 추정 C의 품질과 국소 곡률이 달라질 수 있어 이 확인은 전역 스케일 진단의 일부다.

## 정리하며

K=pᵀM⁻¹p/2 관례에서 운동량 공분산은 M, 속도 공분산은 M⁻¹이며 가우스 목표의 주파수 맞춤은 M≈Σ⁻¹다. 정규화·양의 정부호와 행렬 추정·연산 비용을 함께 확인한다.

**참고 문헌**

1. Neal, R. M. (2011). "MCMC Using Hamiltonian Dynamics." In *Handbook of Markov Chain Monte Carlo*.
2. Betancourt, M. (2017). "A Conceptual Introduction to Hamiltonian Monte Carlo." arXiv:1701.02434.
3. Girolami, M., & Calderhead, B. (2011). "Riemann Manifold Langevin and Hamiltonian Monte Carlo Methods." *JRSS-B*.
4. Stan Development Team. "Stan Reference Manual: HMC Algorithm Parameters."
