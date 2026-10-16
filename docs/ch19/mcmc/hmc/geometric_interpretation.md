# HMC의 기하학적 해석

HMC의 기하는 확장 위상 공간의 보존 구조와 위치 목표의 스케일·곡률을 연결한다. 심플렉틱 기하, 리만 계량과 통계 모형의 피셔 정보를 구분하면 적분·전처리의 역할을 더 정확히 이해할 수 있다.

## 1. 샘플링의 기하

목표밀도의 모드는 점별 밀도가 큰 곳이고 전형 집합은 확률질량이 많이 놓이는 영역이다. 고차원에서는 두 개념이 크게 다를 수 있다. 샘플러는 모드 하나를 찾는 데 그치지 않고 필요한 질량 영역을 탐색해야 한다.

심플렉틱 기하는 위치·운동량의 정준 구조, 리만 계량은 길이와 접벡터의 스케일, 피셔 정보는 파라메터화된 통계 분포족의 국소 변화를 기술한다. 이들은 관련 있지만 같은 대상은 아니다.

## 2. 위상 공간의 기하

ω=∑dx_i∧dp_i, Ω=[[0,I],[−I,0]] 관례에서 ż=Ω∇H다. 푸아송 괄호를 {f,g}=∇fᵀΩ∇g로 쓰면 자율 함수 f에 대해 df/dt={f,H}다. {f,H}=0이면 흐름을 따라 f가 보존된다.

정확한 해밀턴 흐름은 심플렉틱 형식을 보존하고 그에 따른 리우빌 체적도 보존한다. 체적을 보존한다는 사실은 유클리드 거리까지 보존하는 등거리 사상이라는 뜻이 아니다. 에너지 보존까지 함께 사용하면 exp(−H) 밀도의 불변성을 얻는다. 수치 Leapfrog는 심플렉틱이지만 원래 H를 정확히 보존하지는 않는다.

## 3. 에너지 지형 그려 보기

U(x)=−log γ(x)의 국소 최솟값은 목표밀도의 국소 모드다. 그러나 낮은 U인 작은 영역보다 조금 높은 U의 넓은 영역에 더 많은 질량이 있을 수 있다. 표준정규 d차원 목표의 위치 반지름은 큰 d에서 √d 부근에 집중한다. 비등방 가우스는 표준화한 마할라노비스 반지름으로 같은 직관을 사용한다.

아래 그림은 위치 목표의 밀도 등고선이다. HMC가 보존하는 것은 U만이 아니라 H=U+K이므로 위치 궤적이 이 등고선을 따라가야 하는 것은 아니다. 운동에너지가 변하면서 U 등고선을 가로지를 수 있다. 또한 모든 목표가 하나의 얇은 구형 전형 껍질을 갖는 것은 아니다.

```
           U increasing
               ↑
        ┌──────────────┐
        │    ╭───╮     │
        │  ╭─┤   ├─╮   │  Nested contours
        │ ╭┤ │ • │ ├╮  │  • = mode (minimum U)
        │  ╰─┤   ├─╯   │
        │    ╰───╯     │
        └──────────────┘
```

## 4. 정보 기하

관측분포족 p(y|θ)의 피셔 정보는 정칙성 조건 아래 다음과 같다.

$$G_{ij}(\theta)=E_{Y|\theta}[\partial_i\log p(Y|\theta)\partial_j\log p(Y|\theta)]=-E_{Y|\theta}[\partial_{ij}\log p(Y|\theta)].$$

기댓값의 대상은 관측 Y다. 하나의 사후밀도 π(x)의 점별 곡률 −∇²log π(x)를 언제나 이 피셔 정보와 같다고 둘 수 없다. 관측 정보·기대 정보·사전 곡률과 결합 방식은 구분해야 한다. 사후 헤세 행렬은 비로그오목 영역에서 부정부호일 수 있어 그대로 리만 계량에 쓸 수도 없다.

현재 운동에너지 관례에서 가우스 목표의 곡률 A=Σ⁻¹를 맞추는 질량은 M=A다. 따라서 M⁻¹≈곡률이라는 원래 설명은 방향이 반대다. 일반 목표에는 적절한 양의 정부호 계량의 설계·정규화가 필요하다.

리만 HMC의 H=U+log det G(x)/2+pᵀG(x)⁻¹p/2는 위치 의존 운동량 정규화를 포함한다. 운동에너지만으로 생성되는 자유 흐름은 해당 계량의 측지선과 연결되지만 퍼텐셜 힘이 있는 전체 HMC 궤적을 일반적으로 최단 측지선이라고 부를 수 없다. 측지선도 전역 최단 경로일 필요는 없다.

## 5. 조화 운동과 비선형 궤적

M=I, U=xᵀAx/2에서 선형 흐름은 다음 행렬 지수로 표현한다.

$$\binom{x(t)}{p(t)}=\exp\left[t\begin{pmatrix}0&I\\-A&0\end{pmatrix}\right]\binom{x(0)}{p(0)}.$$

A가 양의 정부호이면 적절한 모드 좌표에서 주파수 √λ_i의 조화 운동으로 나뉜다. 1차원은 타원이고 주기는 2π/√A다. 다차원은 주파수 비에 따라 주기적 또는 준주기적일 수 있으며 단일 평면 타원으로 모두 표현되지 않는다.

비선형 목표에는 복잡한 궤적이 가능하다. NUTS 유턴은 국소적인 확장 정지 신호이며 전형 집합을 충분히 탐색했다는 증명이 아니다. 공진·모드 장벽·다른 시간 스케일을 별도로 고려한다.

## 6. 그림자 해밀턴 함수

역방향 오차 분석은 대칭 심플렉틱 적분을 다음 수정 에너지의 형식적 흐름과 연결한다.

$$\widetilde H=H+\epsilon^2H_2+\epsilon^4H_4+\cdots.$$

일반적으로 이 급수가 수렴해 수치 사상이 하나의 수정 해밀턴 함수를 영원히 정확히 적분한다는 뜻은 아니다. 매끄러움·해석성·안정적인 작은 스텝과 유계 궤적 등의 조건에서 절단된 전개로 긴 시간의 작은 에너지 오차를 설명할 수 있다.

수치 점들이 원래 H 등위면에 정확히 놓이거나 어떤 목표에서도 오차가 L과 무관하게 영원히 유계라는 주장은 하지 않는다. HMC는 원래 목표의 에너지 차이를 사용한 MH 보정으로 적분의 목표 불변성을 구성한다.

## 7. 질량 행렬을 기하로 보기

p=M u인 운동량·속도 관계에서 K=pᵀM⁻¹p/2=uᵀMu/2다. 따라서 속도 접공간의 계량은 M, 운동량 쌍대 계량은 M⁻¹다.

가우스 목표에서 M=Σ⁻¹를 선택하고 z=Σ⁻¹/²(x−μ), r=Σ¹/²p로 정준변환하면 H=(‖z‖²+‖r‖²)/2가 된다. 이를 whitening 또는 표준화라고 한다. 위치만 바꾸고 운동량을 그대로 두는 변환과 구분한다.

이차 퍼텐셜 A의 전처리 동역학은 M⁻¹/²AM⁻¹/²의 고윳값으로 주파수가 정해진다. 고윳값 비가 1이면 모드 주파수가 맞지만 일반 비선형·다봉 목표의 빠른 혼합까지 보장하는 것은 아니다.

## 8. 곡률·스케일 변화와 깔때기

2차원 그래프 z=f(x₁,x₂)의 가우스 곡률은 det(∇²f)/(1+‖∇f‖²)²다. f=log π를 택하면 그 특정 유클리드 그래프의 곡률을 얻지만 이는 피셔 계량의 내재 곡률이나 HMC의 모든 난도를 하나로 나타내는 값이 아니다. 좌표·축의 스케일에도 영향을 받는다.

적분 안정성에는 퍼텐셜의 헤세 행렬과 질량으로 정해지는 국소 주파수, 그 변화와 비선형성이 중요하다. 양의 그래프 곡률이면 쉽고 음수면 발산한다는 단순 분류는 성립하지 않는다.

깔때기 예제 y~N(0,σ_y²), x|y~N(0,exp(y))는 두 번째 인자를 분산으로 정의한다. x 방향의 곡률은 exp(−y)로 y가 작을수록 커진다. x=exp(y/2)z, z~N(0,1)의 비중심 표현은 이 예제의 변하는 스케일을 분리한다. 일반 계층모형에서 어느 표현이 좋은지는 관측 정보와 함께 평가한다.

## 9. 그려 보는 방법

위상 그림은 d=1의 위치·운동량 평면에서 궤적을, 에너지 그림은 동기화된 정수시점 상태의 H를, 위치 그림은 d=2 위치로 투영한 경로를 나타낸다. 위치 경로의 교차는 위상 흐름의 유일성과 모순되지 않는다.

아래 위상·에너지 함수는 M=I를 가정하고 고정 스텝 Leapfrog로 그린다. 시작점을 사각형에서 균등하게 뽑는 위상 그림은 목표 표본의 밀도 그림이 아니다. 위치 함수의 samples가 한 궤적인지 여러 MH 전이의 표본인지에 따라 선의 의미도 달라진다. 그림 생성 조각은 샘플러 전체나 수렴 검증을 대신하지 않는다.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_phase_portrait(U, grad_U, x_range, v_range, n_traj=10):
    # 에너지 등고선
    x = np.linspace(*x_range, 100)
    v = np.linspace(*v_range, 100)
    X, V = np.meshgrid(x, v)
    H = U(X) + 0.5 * V**2
    
    plt.contour(X, V, H, levels=20, alpha=0.5)
    
    # 자취 표집
    for _ in range(n_traj):
        x0 = np.random.uniform(*x_range)
        v0 = np.random.uniform(*v_range)
        
        # 자취 흉내내기
        xs, vs = [x0], [v0]
        x_curr, v_curr = x0, v0
        
        for _ in range(200):
            # 개구리뜀 걸음
            v_curr = v_curr - 0.05 * grad_U(x_curr)
            x_curr = x_curr + 0.1 * v_curr
            v_curr = v_curr - 0.05 * grad_U(x_curr)
            xs.append(x_curr)
            vs.append(v_curr)
        
        plt.plot(xs, vs, 'b-', alpha=0.3, linewidth=0.5)
    
    plt.xlabel('x')
    plt.ylabel('v')
    plt.title('Phase Portrait')
```
```python
def plot_energy_trajectory(x0, v0, U, grad_U, n_steps=100, epsilon=0.1):
    x, v = x0.copy(), v0.copy()
    energies = [U(x) + 0.5 * np.dot(v, v)]
    
    for _ in range(n_steps):
        v = v - (epsilon/2) * grad_U(x)
        x = x + epsilon * v
        v = v - (epsilon/2) * grad_U(x)
        energies.append(U(x) + 0.5 * np.dot(v, v))
    
    plt.plot(energies)
    plt.xlabel('Leapfrog Step')
    plt.ylabel('Hamiltonian H')
    plt.title('Energy Conservation')
    plt.axhline(energies[0], color='r', linestyle='--', label='Initial')
```
```python
def plot_trajectory_2d(samples, target_log_prob=None):
    plt.plot(samples[:, 0], samples[:, 1], 'b-', alpha=0.5, linewidth=0.5)
    plt.plot(samples[0, 0], samples[0, 1], 'go', markersize=10, label='Start')
    plt.plot(samples[-1, 0], samples[-1, 1], 'ro', markersize=10, label='End')
    
    if target_log_prob is not None:
        x = np.linspace(samples[:, 0].min() - 1, samples[:, 0].max() + 1, 100)
        y = np.linspace(samples[:, 1].min() - 1, samples[:, 1].max() + 1, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[target_log_prob(np.array([xi, yi])) 
                       for xi, yi in zip(x_row, y_row)]
                      for x_row, y_row in zip(X, Y)])
        plt.contour(X, Y, np.exp(Z), levels=10, alpha=0.5, colors='gray')
    
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.legend()
```

## 10. NUTS을 기하로 보기

끝점 차이 Δx=x⁺−x⁻와 실제 속도 M⁻¹p⁺·M⁻¹p⁻의 내적을 검사하는 것이 기본 유턴 기준이다. M=I인 특수식에서 운동량을 속도처럼 쓸 수 있지만 일반 질량에서는 역질량을 포함해야 한다.

한쪽 끝의 검사만으로 전체 NUTS를 정의하지 않는다. 부분 트리와 양 끝의 유턴·수치 한계, 올바른 후보 선택을 결합해야 한다. 아래 도식은 배증 직관이며 정확한 상태 수·단계 수의 명세는 아니다. 매 확장이 항상 새로운 확률질량 영역을 추가하는 것도 아니다.

```
        Depth 0         Depth 1              Depth 2
           •       →    •───•        →    •───•───•───•
                      backward           forward extension
```

## 11. 다른 방법과의 이음

M=I에서 과감쇠 Langevin은 dX=∇log π(X)dt+√2dW, 해밀턴 흐름은 ẋ=p, ṗ=∇log π(x)다. 전자는 위치의 확산, 후자는 운동량을 포함한 결정적 흐름이며 HMC는 이 흐름에 무작위 운동량 생성과 MH 보정을 결합한다.

자연경사 방법은 통계 모형의 피셔 계량을 사용해 최적화 방향을 조정한다. HMC와 계량·전처리의 아이디어를 공유할 수 있지만 그 자체의 확률 버전이라고 동일시하지 않는다. 하나는 목적함수 최적화, 다른 하나는 지정한 목표분포를 보존하는 전이 설계이며 필요한 정규화·잡음·보정이 다르다.

## 연습문제

1. 이중 우물 U=(x²−1)²의 장벽과 분리 궤적을 설명하라. 안장 x=0,p=0의 에너지 1이 경계이고 그 아래 비평형 궤적은 한 우물에 머물 수 있다.

2. 가우스 whitening이 왜 위치·운동량을 함께 바꾸는가? z=Σ⁻¹/²(x−μ), r=Σ¹/²p로 정준 형식을 보존하며 두 이차 에너지 항을 모두 표준화한다.

3. 사후 헤세 행렬을 언제나 피셔 정보 계량으로 쓸 수 있는가? 아니다. 피셔 정보는 분포족의 관측에 대한 기대이고 점별 사후 곡률과 다르며, 사후 헤세 행렬은 부정부호일 수도 있다.

4. 그림자 해밀턴 함수의 급수는 영원히 정확한 보존량인가? 일반적으로 형식·절단 전개다. 안정적 작은 스텝과 정칙성 범위의 근사 분석으로 해석한다.

5. 깔때기에서 x=exp(y/2)z를 대입하면 어떤 장점이 있는가? 이 정의에서는 z의 조건부 스케일이 1로 고정된다. 목표를 좌표변환해 쓸 때는 야코비안과 관측 우도까지 일관되게 반영한다.

## 정리하며

HMC의 기하에서는 심플렉틱 체적 보존, 위치·운동량의 쌍대 계량, 목표 곡률과 통계적 피셔 정보를 구분한다. 궤적은 위치 밀도 등고선을 따를 필요가 없고, 수정 에너지와 전처리의 효율도 조건에 맞춰 해석해야 한다.

**참고 문헌**

1. Betancourt, M. (2017). "A Conceptual Introduction to Hamiltonian Monte Carlo." arXiv:1701.02434.
2. Girolami, M., & Calderhead, B. (2011). "Riemann Manifold Langevin and Hamiltonian Monte Carlo Methods." *JRSS-B*.
3. Amari, S. (2016). *Information Geometry and Its Applications*. Springer.
4. Leimkuhler, B., & Reich, S. (2004). *Simulating Hamiltonian Dynamics*. Cambridge University Press.
5. Neal, R. M. (2011). "MCMC Using Hamiltonian Dynamics." In *Handbook of Markov Chain Monte Carlo*.
