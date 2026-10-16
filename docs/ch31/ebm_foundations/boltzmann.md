# 볼츠만 분포와 온도

에너지 기반 모델은 상태마다 스칼라 에너지를 주고 낮은 에너지에 높은 확률을 부여한다. 볼츠만 분포는 이 관계를 지수함수로 표현한다. 정규화가 가능한 조건, 온도의 역할과 입력 스코어를 구분하면 이후의 분배 함수·샘플링·학습을 같은 틀에서 이해할 수 있다.

## 1. 에너지에서 확률로

상태 x, 에너지 E(x), 양의 온도 T에 대해 다음을 정의한다. 물리 상수 k_B는 1로 둔다.

$$p_T(x)=\frac{e^{-E(x)/T}}{Z(T)},\qquad
Z(T)=\int_{\mathcal X}e^{-E(x)/T}\,d\lambda(x)$$

λ는 기준 측도다. 연속 변수에서는 보통 르베그 측도, 이산 상태에서는 계수 측도를 사용해 적분을 합으로 바꾼다. 분포가 되려면 **0<Z(T)<∞**여야 한다. R 위에서 E(x)=0이면 Z=∞이므로 밀도를 정의하지 못한다. 뉴럴 넷트웍이 에너지를 출력한다는 사실만으로 정규화 가능성이 보장되지는 않는다.

에너지에 상수 c를 더하면 Z는 e^(−c/T)배가 되지만 정규화한 분포는 같다. 두 상태의 확률비는 에너지 차이만으로 정해진다.

$$\frac{p_T(x)}{p_T(y)}=e^{-(E(x)-E(y))/T}$$

## 2. 평균 에너지 제약과 최대 엔트로피

[Jaynes의 최대 엔트로피 관점](https://journals.aps.org/pr/abstract/10.1103/PhysRev.106.620)은 기준 측도와 평균 에너지를 정한 뒤 그 조건에서 엔트로피를 최대화한다. 정규화와 EₚE=U의 라그랑주 승수를 각각 a, β로 두면 다음과 같다.

$$\mathcal L[p]=-\int p\log p\,d\lambda
-a\left(\int p\,d\lambda-1\right)
-\beta\left(\int pE\,d\lambda-U\right)$$

내부 최적점에서 변분 미분을 0으로 두면 −log p−1−a−βE=0이므로 p∝exp(−βE)를 얻는다. 정규화 가능하고 제약을 만족하는 해가 존재하면 엔트로피의 오목성으로 최대임을 확인할 수 있다. 일반적인 평균 제약이 항상 β>0을 보장하는 것은 아니며, 여기서는 양의 온도 영역 β=1/T를 다룬다.

E(x)=x²이고 R 전체에서 평균 에너지가 U라면 최대 엔트로피 해는 N(0,U)이며 T=2U다. 이 예제로 수치 코드의 제약이 일치하는지 확인해 보자.

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def demonstrate_maxent_boltzmann():
    """
    볼츠만 분포가 엔트로피를 가장 크게 함을 보여 준다.
    
    볼츠만 분포의 엔트로피를 다른 분포와 견준다
    평균 에너지가 같은 것들 가운데.
    """
    # 에너지 함수를 뜻매김한다(단순한 이차식)
    def energy(x):
        return x**2
    
    # 격자를 뜻매김한다
    x = np.linspace(-5, 5, 1000)
    dx = x[1] - x[0]
    E = energy(x)
    
    # 목표 평균 에너지
    target_energy = 1.0
    
    # 목표 평균 에너지를 주는 온도를 찾는다
    def avg_energy_error(T):
        if T <= 0:
            return np.inf
        unnorm = np.exp(-E / T)
        Z = unnorm.sum() * dx
        p = unnorm / Z
        return (np.sum(p * E) * dx - target_energy)**2
    
    result = minimize_scalar(avg_energy_error, bounds=(0.1, 10), method='bounded')
    T_opt = result.x
    
    # 가장 좋은 온도의 볼츠만 분포
    boltz_unnorm = np.exp(-E / T_opt)
    Z = boltz_unnorm.sum() * dx
    p_boltz = boltz_unnorm / Z
    
    # 볼츠만 분포의 엔트로피를 셈한다
    H_boltz = -np.sum(p_boltz * np.log(p_boltz + 1e-10)) * dx
    
    # 잘린 정규 분포와 견준다(평균 에너지가 같다)
    sigma = np.sqrt(2 * target_energy)
    p_gauss = np.exp(-x**2 / (2 * sigma**2))
    p_gauss = p_gauss / (p_gauss.sum() * dx)
    H_gauss = -np.sum(p_gauss * np.log(p_gauss + 1e-10)) * dx
    
    print(f"Boltzmann entropy: {H_boltz:.4f}")
    print(f"Gaussian entropy: {H_gauss:.4f}")
    print(f"Boltzmann has {'higher' if H_boltz > H_gauss else 'lower'} entropy")
    
    return p_boltz, p_gauss, x

p_boltz, p_gauss, x = demonstrate_maxent_boltzmann()
```

**보존된 출력:**

```
Boltzmann entropy: 1.4189
Gaussian entropy: 1.7624
Boltzmann has lower entropy
```

코드는 [−5,5]의 격자에서 평균 에너지 1을 맞춘다. 하지만 비교 가우스의 sigma를 √2로 두어 분산 2를 사용한다. R에서의 목표 제약 E[x²]=1에 해당하는 가우스는 표준편차 1이다. 유한 구간에서 재정규화해도 두 분포의 평균 에너지가 같아지는 것은 아니다. 따라서 출력의 엔트로피 대소 비교는 최대 엔트로피 원리를 반박하거나 입증하는 공정한 비교가 아니다.

## 3. 온도가 바꾸는 것과 바꾸지 않는 것

낮은 T는 에너지 차이에 대한 확률비를 더 크게 만든다. 유한 이산 상태 공간에서는 T→0⁺일 때 전역 최소 에너지 상태들에 균등하게 질량이 모인다. 하지만 연속 공간의 여러 최소점에서는 일반적으로 균등 웨이트가 아니다. 고립된 비퇴화 최소점의 Hessian을 Hᵢ라 하면 라플라스 근사에서 같은 최소 에너지 우물의 질량은 (det Hᵢ)^(−1/2)에 비례한다.

유한 상태나 유한 측도 공간에서는 적절한 조건 아래 T→∞일 때 기준 측도에 대한 균등분포로 간다. Rᵈ 전체에는 정규화 가능한 균등분포가 없으므로 이 결론을 적용할 수 없다. 예를 들어 E=x²의 가우스는 T와 함께 분산이 커지며 R 위의 균등 확률분포로 수렴하지 않는다.

고정 E와 모든 유한 T>0에서 밀도의 모드는 E의 최소점이다. 온도를 올려도 위치가 이동하거나 두 우물이 자동으로 하나의 모드로 합쳐지지 않는다. 온도에 따른 상대적인 폭·질량과 모드 위치는 다른 성질이다.

```python
def analyze_temperature_sweep():
    """
    온도에 따라 볼츠만 분포가 어떻게 바뀌는지 살핀다.
    """
    x = torch.linspace(-4, 4, 1000)
    E = 0.5 * (x**2 - 4)**2  # Double-well
    dx = x[1] - x[0]
    
    temperatures = torch.tensor([0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 그림 1: 온도에 따른 분포 모양
    ax1 = axes[0, 0]
    for T in [0.5, 1.0, 2.0, 5.0]:
        log_unnorm = -E / T
        log_Z = torch.logsumexp(log_unnorm, dim=0) + torch.log(dx)
        p = torch.exp(log_unnorm - log_Z)
        ax1.plot(x.numpy(), p.numpy(), linewidth=2, label=f'T = {T}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('p(x)')
    ax1.set_title('Distribution Shape vs Temperature')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 그림 2: 온도에 따른 엔트로피
    ax2 = axes[0, 1]
    entropies = []
    for T in temperatures:
        log_unnorm = -E / T.item()
        log_Z = torch.logsumexp(log_unnorm, dim=0) + torch.log(dx)
        p = torch.exp(log_unnorm - log_Z)
        H = -torch.sum(p * torch.log(p + 1e-10)) * dx
        entropies.append(H.item())
    
    ax2.plot(temperatures.numpy(), entropies, 'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel('Temperature T')
    ax2.set_ylabel('Entropy H')
    ax2.set_title('Entropy Increases with Temperature')
    ax2.grid(True, alpha=0.3)
    
    # 그림 3: 온도에 따른 평균 에너지
    ax3 = axes[1, 0]
    avg_energies = []
    for T in temperatures:
        log_unnorm = -E / T.item()
        log_Z = torch.logsumexp(log_unnorm, dim=0) + torch.log(dx)
        p = torch.exp(log_unnorm - log_Z)
        avg_E = torch.sum(p * E) * dx
        avg_energies.append(avg_E.item())
    
    ax3.plot(temperatures.numpy(), avg_energies, 'ro-', linewidth=2, markersize=8)
    ax3.set_xlabel('Temperature T')
    ax3.set_ylabel('⟨E⟩')
    ax3.set_title('Average Energy Increases with Temperature')
    ax3.grid(True, alpha=0.3)
    
    # 그림 4: 온도에 따른 봉우리 확률
    ax4 = axes[1, 1]
    left_mode = x < 0
    right_mode = x >= 0
    
    for T in [0.5, 1.0, 2.0]:
        log_unnorm = -E / T
        log_Z = torch.logsumexp(log_unnorm, dim=0) + torch.log(dx)
        p = torch.exp(log_unnorm - log_Z)
        
        left_prob = torch.sum(p[left_mode]) * dx
        right_prob = torch.sum(p[right_mode]) * dx
        
        ax4.bar([T - 0.1, T + 0.1], [left_prob.item(), right_prob.item()], 
                width=0.15, label=f'T={T}' if T == 0.5 else None)
    
    ax4.set_xlabel('Temperature')
    ax4.set_ylabel('Mode Probability')
    ax4.set_title('Mode Occupation vs Temperature')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

analyze_temperature_sweep()
```

위 예제는 대칭 에너지 E(x)=0.5(x²−4)²를 [−4,4] 격자에서 계산한다. 모드는 모든 양의 유한 온도에서 ±2이고, 좌우 질량은 대칭성으로 각각 약 1/2이다. 그림이 평평해 보여도 임계 온도에서 가운데 모드 하나로 바뀌었다고 해석하지 않는다. 유한 격자와 범위 제한의 오차도 있다.

## 4. 입력 스코어와 에너지 그래디언트

연속 입력에 대해 E가 미분 가능하고 Z가 유한하며 x에 의존하지 않으면 다음과 같다.

$$s_T(x)=\nabla_x\log p_T(x)=-\frac1T\nabla_xE(x)$$

Z가 입력 미분에서 사라지는 성질은 [스코어 매칭](../../ch30/score_based/denoising_score_matching.md)의 출발점이다. 하지만 Z가 모델 파라미터 θ에도 독립이라는 뜻은 아니다. 파라미터를 미분하는 최대 우도 학습에서는 모델 기댓값 항이 남는다.

```python
def compute_score(energy_net, x, create_graph=True):
    """
    점수 함수 s(x) = -∇E(x)을 셈한다.
    
    파라메터
    ----------
    energy_net : nn.Module
        뉴럴 넷트웍 에너지 함수
    x : torch.Tensor
        들임 점, 꼴 (batch, dim)
    create_graph : bool
        더 높은 차수의 그래디언트를 위해 그래프를 만들지 여부
    
    반환값
    -------
    torch.Tensor
        점수 값, x과 같은 꼴
    """
    x = x.requires_grad_(True)
    energy = energy_net(x).sum()
    
    score = torch.autograd.grad(
        outputs=energy,
        inputs=x,
        create_graph=create_graph
    )[0]
    
    return -score  # Score is negative gradient of energy
```

compute_score는 음의 에너지 그래디언트만 반환하므로 **T=1**의 스코어에 해당한다. 다른 T라면 1/T가 필요하다. 배치 에너지 합을 미분하므로 각 샘플의 에너지가 다른 배치 원소에 의존하지 않는다는 가정도 필요하다. create_graph=True는 이 스코어를 사용한 로스에서 파라미터로 다시 미분할 때 활용한다.

스코어 방향으로 충분히 작은 걸음을 옮기면 국소적으로 로그 밀도를 높일 수 있지만, 큰 이산 스텝은 반대 결과를 낼 수 있다. 에너지의 정지점은 최소·최대·안장점일 수 있다. 그래디언트 하강만으로 분포에서 샘플링하는 것도 아니며 샘플링에는 적절한 확률적 동역학과 조건이 필요하다.

## 5. 지수족으로 보는 범위

고정 E에서 β만 바꾸는 계열은 충분통계 E(x), 자연모수 −β, 로그 분배 함수 log Z를 갖는 지수족이다. 에너지 자체를 임의의 뉴럴 넷트웍 파라미터로 바꾸는 전체 모델이 고정된 유한 차원 충분통계를 갖는 정규 지수족이라는 뜻은 아니다.

유한 차원 분포를 기준 측도에 대해 정의하는 식과 무한 격자의 깁스 측도를 구성하는 문제도 구분한다. 무한 계에서는 국소 조건부 분포와 극한 등 추가 정의가 필요하며 하나의 전역 유한 적분을 그대로 쓴다고 해결되지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1. 고정 이중 우물의 모드**

E(x)=0.5(x²−4)²에서 온도를 높이면 두 모드가 하나가 되는가?

</div>

??? success "연습문제 1 풀이"
    E′=2x(x²−4), E″=6x²−8이다. x=±2에서 E″=16>0이므로 모든 유한 T>0에서 밀도의 최대점이다. x=0에서는 (log p)″=8/T>0이므로 밀도의 최소점이다. 온도만으로 가운데 모드 하나로 바뀌는 임계값은 없다.

---

<div class="drillbox" markdown>

**연습문제 2. 최대 엔트로피 비교**

E=x²의 평균 에너지 1 제약에서 표준편차 √2의 가우스와 엔트로피를 비교해도 되는가?

</div>

??? success "연습문제 2 풀이"
    같은 제약의 후보가 아니다. R에서 평균 x²=1인 최대 엔트로피 해는 N(0,1)이고 T=2다. 표준편차 √2는 평균 x²=2를 갖는다. 유한 구간에서는 각 분포를 재정규화한 뒤 실제 평균 에너지까지 확인해야 한다.

---

<div class="drillbox" markdown>

**연습문제 3. 두 상태 계의 비열**

에너지 0과 ε>0이 같은 축퇴도로 하나씩 있는 계에서 비열의 최대 조건을 유도하라.

</div>

??? success "연습문제 3 풀이"
    Z=1+e^(−ε/T)이며 C=(ε/T)²e^(ε/T)/(1+e^(ε/T))²다. u=ε/(2T)로 두면 C=u²sech²u이고 도함수는 2u sech²u(1−u tanh u)다. 최대는 u tanh u=1의 양의 해 u≈1.19968에서 생겨 T≈0.41678ε다. 유한 두 상태 계의 매끄러운 봉우리이며 열역학적 특이점은 아니다.

## 정리하며

볼츠만 분포에는 양의 유한 분배 함수가 필요하다. 최대 엔트로피 유도는 같은 제약과 기준 측도 아래에서 해석하고, 온도 극한은 이산·연속 공간을 구분한다. 고정 에너지의 모드는 유한 온도에서 바뀌지 않으며 입력 스코어는 −∇E/T다.

**참고 문헌**

- Jaynes, E. T. (1957). Information theory and statistical mechanics. *Physical Review*.
- MacKay, D. J. (2003). *Information Theory, Inference and Learning Algorithms*. Cambridge University Press.
- Boltzmann, L. (1877). On the Relationship between the Second Fundamental Theorem of the Mechanical Theory of Heat and Probability Calculations.
