# 랑주뱅 동역학의 기초

랑주뱅 동역학은 로그밀도의 경사와 브라운 잡음을 결합한다. 연속시간 확산의 불변분포, 이산 근사의 바이어스와 실제 수렴 조건을 구분하는 것이 핵심이다.

## 1. 마찰·잡음과 과감쇠 극한

마찰·퍼텐셜·열 잡음을 갖는 입자의 관성 방정식은 다음처럼 표현할 수 있다.

$$m\ddot x=-\gamma\dot x-\nabla U(x)+\sqrt{2\gamma k_BT}\,\xi(t).$$

관성의 영향이 작은 과감쇠 극한에서는 적절한 시간 단위로 dx=−∇Udt+√(2k_BT)dW를 얻는다. 흰 잡음은 보통 함수의 미분이 아니라 브라운 운동의 형식적 시간 미분으로 이해한다. k_BT=1의 단위에서 exp(−U)를 목표로 하는 표준 샘플링 식이 된다. 물리적 직관이 확률과정의 존재·수렴 조건을 대신하지는 않는다.

## 2. 샘플링을 위한 랑주뱅 SDE

적절한 목표 π(x)∝exp(−U(x))에서 스코어 s=∇log π=−∇U를 쓰면 다음을 얻는다.

$$dX_t=s(X_t)dt+\sqrt2dW_t.$$

드리프트는 국소적으로 로그밀도가 증가하는 방향이고 잡음은 확산을 만든다. 이 상대 계수는 π의 정상성을 구성한다. 그러나 잡음이 있다는 사실만으로 모든 모드가 유한 시간 안에 충분히 탐색되는 것은 아니다. 상태공간·경계조건·비폭발성과 에르고드 조건을 확인한다.

## 3. 스코어 함수와 정규화 상수

스코어 함수는 상태에 대한 로그밀도 경사다. 베이즈 목표에서는 ∇log π(x|D)=∇log p(D|x)+∇log p(x)로 분해된다. 미지 정규화 상수는 x에 의존하지 않으므로 사라진다.

매끄러운 내부 모드에서는 스코어가 0이지만 스코어 0인 모든 점이 모드는 아니다. 안장점·국소 최소점·평평한 구간도 가능하다. 경계 모드나 밀도가 0인 곳에서는 같은 미분 식을 그대로 사용하지 않는다. 여기의 상태 스코어와 통계 분포족의 파라메터 스코어는 미분 대상이 다를 수 있다.

## 4. 온도를 넣은 일반 꼴

T>0에서 dX=−∇Udt+√(2T)dW는 적절한 조건 아래 π_T∝exp(−U/T)를 보존한다. 이는 T=1의 목표와 다르다. 높은 온도는 장벽의 상대적 효과를 줄이고 낮은 온도는 낮은 U 영역을 강조하지만 혼합·최적화의 성공을 자동으로 보장하지는 않는다.

잡음 계수의 T→0 극한에서 고정 시간의 결정적 방정식 ẋ=−∇U를 생각할 수 있다. 이는 일반 비볼록 U의 전역 최소점 도달을 보장하지 않는다. 긴 시간·냉각·온도 0 극한의 순서와 조건을 구분한다.

## 5. 포커-플랑크 방정식

충분히 매끄러운 밀도와 적절한 경계조건에서 확률밀도의 변화는 다음과 같다.

$$\partial_t\rho=-\nabla\cdot(\rho\nabla\log\pi)+\Delta\rho=\nabla\cdot\left(\pi\nabla\frac\rho\pi\right).$$

ρ=π를 대입하면 오른쪽은 0이다. 이는 정상성의 형식적 확인이며 적절한 확산의 존재·유일성과 수렴 증명을 모두 대신하지 않는다.

적분 부분의 경계항이 사라지면 KL 소산식 d KL(ρ_t‖π)/dt=−∫ρ_t‖∇log(ρ_t/π)‖²dx를 얻는다. 적절한 의미에서 Wasserstein 공간의 KL 경사 흐름과 연결되지만 KL이 단조 감소한다는 사실만으로 임의 속도의 수렴이나 목표의 모든 영역 탐색을 보장하지는 않는다.

## 6. 정상성과 수렴 속도의 조건

U가 충분히 매끄럽고 Hess U⪰mI, m>0이며 필요한 적분·경계 조건을 만족하면 로그소볼레프 부등식 등을 통해 KL(ρ_t‖π)≤exp(−2mt)KL(ρ₀‖π) 형태의 결과를 얻을 수 있다. 유한 초기 KL 등 조건을 확인한다.

비볼록 모형에도 비폭발성·재귀성·소산 조건 아래 수렴 정리가 있지만 단순히 경사가 Lipschitz라는 사실만으로 모든 지표의 기하 수렴을 말할 수는 없다. 연속 확산·이산 ULA·MALA의 정리를 구분한다.

가우스의 연속 OU 수렴 속도는 곡률 고윳값과 초기 오차, 거리·정밀도 기준에 달린다. O(dκ)라는 하나의 식을 모든 가우스 확산의 혼합시간으로 제시하지 않는다. 차원·조건수와 이산 스텝·반복 비용을 분리한다.

## 7. 이산화와 두 스텝 관례

Euler–Maruyama 근사는 X_{k+1}=X_k+h s(X_k)+√(2h)η_k다. 유한 h의 무보정 알고리즘이 ULA이고 MH 보정을 붙인 가우스 제안이 MALA다. εs/2+√εη 표기와는 ε=2h의 시간 단위 차이가 있다.

이차 U=xᵀAx/2, A>0에서 안정적인 정상 AR 과정의 조건은 0<h<2/λ_max(A)다. 안정해도 원래 목표와 같은 정상분포라는 뜻은 아니다. 비선형 목표에서는 Euler 근사가 불안정해질 수 있어 연속 SDE의 정상성이 이산 알고리즘으로 자동 이전되지 않는다. [ULA](ula.md)와 [MALA](mala.md)를 구분해 읽는다.

## 8. 온도 냉각과 노이즈 수준 전환

온도 냉각은 고정 U에 π_T∝exp(−U/T)를 사용한다. 반면 노이즈 조건부 스코어의 annealed Langevin은 데이터에 가우스 잡음을 합성한 q_σ의 스코어를 σ가 큰 것부터 작은 순서로 바꾼다. 두 목표군은 일반적으로 같은 것이 아니다.

아래 의사코드는 감소하는 σ와 각 q_σ에 맞는 스코어를 가정한다. 각 단계의 유한 혼합·이산화와 학습 오차가 남으며 마지막 σ>0이면 여전히 매끄럽게 된 데이터분포를 목표로 한다. [스코어와 확산 절](score_and_diffusion.md)에서 역시간 SDE와의 관계를 구분한다.

```
Initialise x ~ N(0, σ₁² I)
For l = 1, ..., L:
    ε = α σₗ²
    For k = 1, ..., K:
        x ← x + (ε/2) s(x, σₗ) + √ε η,  η ~ N(0, I)
    σₗ → σₗ₊₁
Return x
```

## 9. 최적화와의 이음

잡음을 제거하면 경사 흐름이 되지만 일반 비볼록 문제에서 반드시 MAP나 전역 모드로 수렴하는 것은 아니다. SGD의 미니배치 잡음도 일반적으로 위치 의존·비등방이며 온도에 맞춘 브라운 잡음과 같지 않다.

SGLD는 후험 목표에 맞는 확률 경사와 명시적인 가우스 잡음을 결합한다. HMC는 마찰·연속 잡음이 없는 해밀턴 흐름과 운동량 재생성을 사용한다. 저감쇠 Langevin과 연결할 수 있지만 표준 HMC 자체를 단순히 이차 Langevin SDE라고 동일시하지 않는다.

## 10. 전처리

상수 양의 정부호 P에서 다음 확산은 적절한 조건 아래 같은 π를 보존한다.

$$dX=P\nabla\log\pi(X)dt+\sqrt{2P}\,dW.$$

HMC의 질량 관례와 비교하면 P=M⁻¹이다. 가우스 목표에서 P≈공분산은 스케일을 맞추는 자연스러운 선택이다. 위치 의존 P(x)에 같은 식을 그대로 대입하면 일반적으로 목표가 달라질 수 있다. Itô 관례의 가역적 확산에는 div P 벡터를 드리프트에 더하는 등의 적절한 보정이 필요하다.

## 11. PyTorch 구현

LangevinSDE라는 클래스 이름과 달리 step은 유한 dt의 Euler 근사다. temperature=T와 기본 스코어 ∇log π를 그대로 사용하면 연속 목표는 π^(1/T)에 비례하며 이산 바이어스도 추가된다. T≥0, dt>0, 입력·스코어 형태와 기기·자료형을 확인해야 한다.

예제는 상관 가우스 목표의 정확한 스코어를 쓰지만 고정 dt=0.1은 평균이 맞더라도 정상 공분산에 이산 바이어스를 남긴다. dim은 저장만 할 뿐 입력 검증에 쓰이지 않는다. 반환한 마지막 배치와 모든 시간 궤적, 초기점 포함 n_steps+1개 기록을 구분한다. 부동소수점·자동미분 그래프 관리도 긴 시뮬레이션에서 고려해야 한다.

```python
import torch

class LangevinSDE:
    """오일러-마루야마로 푸는 이어진 시간 랑주뱅 액션."""
    
    def __init__(self, score_fn, dim, temperature=1.0):
        self.score_fn = score_fn
        self.dim = dim
        self.temperature = temperature
    
    def step(self, x, dt):
        score = self.score_fn(x)
        drift = score * dt
        diffusion = torch.sqrt(2 * self.temperature * torch.tensor(dt)) * torch.randn_like(x)
        return x + drift + diffusion
    
    def sample(self, x0, n_steps, dt, return_trajectory=False):
        x = x0.clone()
        if return_trajectory:
            trajectory = [x.clone()]
        for _ in range(n_steps):
            x = self.step(x, dt)
            if return_trajectory:
                trajectory.append(x.clone())
        if return_trajectory:
            return torch.stack(trajectory)
        return x

# 보기: 2차원 가우스
mu = torch.tensor([2.0, -1.0])
cov = torch.tensor([[1.0, 0.6], [0.6, 1.0]])
precision = torch.linalg.inv(cov)

score_fn = lambda x: -torch.matmul(x - mu, precision)
sampler = LangevinSDE(score_fn, dim=2)

x0 = torch.randn(500, 2) * 3 + torch.tensor([5.0, 5.0])
x = x0.clone()
for _ in range(1000):
    x = sampler.step(x, dt=0.1)

print(f"Sample mean: [{x[:, 0].mean():.3f}, {x[:, 1].mean():.3f}]")
print(f"True mean:   [{mu[0]:.3f}, {mu[1]:.3f}]")
```
```
Sample mean: [1.943, -1.010]
True mean:   [2.000, -1.000]
```

## 연습문제

1. 포커–플랑크 식에 ρ=π를 대입하라. −div(∇π)+Δπ=0이다. 이 계산이 비폭발성·수렴까지 증명하는 것은 아닌 이유도 설명하라.

2. 온도 π_T∝exp(−U/T)와 q_σ=p_data*N(0,σ²I)가 같은 목표군인가? 일반적으로 다르다. 하나는 밀도의 거듭제곱 정규화이고 다른 하나는 컨볼루션이다.

3. 표준정규의 ULA X_{k+1}=(1−h)X_k+√(2h)η를 분석하라. 0<h<2에서 정상분산은 1/(1−h/2)이고 목표분산 1보다 크다.

4. 스코어 0이면 모드인가? 내부 모드는 필요조건으로 0일 수 있지만 안장·국소 최소·평평한 구간도 0이다. 경계에서는 미분 조건 자체를 다시 확인해야 한다.

5. 정확한 OU의 평균·분산 수렴을 구하라. dX=−X/σ²dt+√2dW에서 평균은 초기 평균의 exp(−t/σ²)배, 분산은 σ²+(v₀−σ²)exp(−2t/σ²)다. ULA의 유한 스텝 정상 바이어스와 구분한다.

## 정리하며

Langevin의 드리프트·확산 균형은 연속 목표의 정상성을 구성한다. 이산 바이어스, 스텝 단위, 에르고드 조건과 온도·노이즈 수준의 차이를 명시하고 실제 구현의 유한 스텝 오차를 평가한다.

**참고 문헌**

1. Langevin, P. (1908). Sur la théorie du mouvement brownien. *Comptes Rendus de l'Académie des Sciences*, 146, 530-533.
2. Gardiner, C. W. (2009). *Stochastic Methods: A Handbook for the Natural and Social Sciences*. Springer.
3. Pavliotis, G. A. (2014). *Stochastic Processes and Applications*. Springer.
4. Roberts, G. O., & Tweedie, R. L. (1996). Exponential convergence of Langevin distributions and their discrete approximations. *Bernoulli*, 2(4), 341-363.
5. Welling, M., & Teh, Y. W. (2011). Bayesian learning via stochastic gradient Langevin dynamics. *ICML*.
6. Song, Y., & Ermon, S. (2019). Generative modeling by estimating gradients of the data distribution. *NeurIPS*.
