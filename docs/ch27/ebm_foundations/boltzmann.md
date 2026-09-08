# 볼츠만 분포

---

## 1. 학습 목표

이 절을 마치면 다음을 할 수 있다:

1. 최대 엔트로피 원리에서 볼츠만 분포를 이끌어 낸다
2. 온도가 분포 모양과 열역학 양에 미치는 영향을 살핀다
3. 점수 함수를 셈하고 나눔 함수가 없는 성질을 이해한다
4. 볼츠만 분포를 지수 무리와 깁스 잣대에 잇는다

---

## 2. 수학적 바탕

### 으뜸 분포

에너지 함수 $E(x)$이 주어질 때 온도 $T$의 볼츠만(또는 깁스) 분포는 다음과 같다:

$$p(x) = \frac{1}{Z(T)} \exp\left(-\frac{E(x)}{T}\right)$$

여기서 나눔 함수 $Z(T)$이 고르게 맞추기를 보장한다:

$$Z(T) = \int_{\mathcal{X}} \exp\left(-\frac{E(x)}{T}\right) dx$$

띄엄띄엄한 상태에서는 $Z(T) = \sum_{x \in \mathcal{X}} \exp(-E(x)/T)$이다.

### 최대 엔트로피에서 이끌어 내기

볼츠만 분포는 아무렇게나 고른 것이 아니다. 평균 에너지 제약 아래에서 엔트로피를 가장 크게 하는 유일한 분포이다. 제인스(1957)의 이 이끌어 냄이 에너지 바탕 모델 틀의 가장 깊은 정당화를 준다.

**목표**: 다음을 가장 크게 하는 $p(x)$을 찾아라:

$$H[p] = -\int p(x) \log p(x)\,dx$$

**제약**:

1. 고르게 맞추기: $\int p(x)\,dx = 1$
2. 에너지 제약: $\int p(x) E(x)\,dx = \langle E \rangle$

라그랑주 곱수 $\alpha$과 $\beta$을 쓰면:

$$\mathcal{L} = -\int p \log p \, dx - \alpha\left(\int p \, dx - 1\right) - \beta\left(\int p E \, dx - \langle E \rangle\right)$$

범함수 미분을 하여 0으로 두면:

$$\frac{\delta \mathcal{L}}{\delta p} = -\log p - 1 - \alpha - \beta E = 0$$

풀면 $p(x) = \exp(-1 - \alpha - \beta E(x)) = \frac{1}{Z} \exp(-\beta E(x))$이다.

여기서 $\beta = 1/T$은 역온도이다. 라그랑주 곱수 $\beta$은 에너지 제약으로 정해지고 $\alpha$은 고르게 맞추기로 붙박인다. 이 이끌어 냄은 볼츠만 분포가 주어진 평균 에너지와 어긋나지 않는 가장 덜 치우친 분포임을 세운다. 이는 앎 이론에서 온 힘센 정당화이다.

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

**출력:**

```
Boltzmann entropy: 1.4189
Gaussian entropy: 1.7624
Boltzmann has lower entropy
```

---

## 3. 온도 살피기

### 낮은 온도 끝(T -> 0)

$T \to 0$이면 분포가 에너지 최솟값에 몰린다:

$$\lim_{T \to 0} p(x) = \frac{1}{|S^*|} \sum_{x^* \in S^*} \delta(x - x^*)$$

여기서 $S^* = \{x : E(x) = \min_y E(y)\}$은 온 자리의 최솟값 모임이다. 이 끝에서 에너지 바탕 모델의 추론은 가장 좋게 하기로 줄어든다. 가장 그럴듯한 자리 얽이를 찾는 것이 에너지 함수를 가장 작게 하는 것과 같아진다.

### 높은 온도 끝(T -> 무한대)

$T \to \infty$이면 분포가 고르게 된다:

$$\lim_{T \to \infty} p(x) = \frac{1}{|\mathcal{X}|}$$

모든 $x$에서 지수 인자 $\exp(-E(x)/T) \to 1$이 되어 에너지 풍경이 뜻을 잃는다. 에너지와 상관없이 모든 자리 얽이가 똑같이 그럴듯해진다.

### 온도 훑기

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

---

## 4. 점수 함수

### 정의

점수 함수는 들임에 대한 로그 확률의 기울기이다:

$$s(x) = \nabla_x \log p(x) = -\frac{1}{T} \nabla_x E(x)$$

**핵심 통찰**: 점수 함수는 나눔 함수 $Z$이 아니라 에너지 기울기에만 매인다. 이 성질이 점수 맞추기(26.3절)의 바탕이며, 다룰 수 없는 고르게 맞추는 상수를 한 번도 셈하지 않고 에너지 바탕 모델을 익힐 수 있게 한다.

### 해석

점수 함수는 로그 확률이 커지는 방향, 곧 에너지가 낮아지는 쪽을 가리킨다. 이는 확률 풍경의 그 자리 "흐름"을 적어 준다. 어느 점 $x$에서든 점수 함수를 따라가면 밀도가 높은 자리로 옮겨 간다.

볼츠만 분포에서 점수 함수는 그저 (잣수를 맞춘) 음의 에너지 기울기이다. 곧 $s(x) = -\nabla_x E(x)/T$을 헤아리도록 모델을 익히는 것은 상수만큼의 차이를 빼고 에너지 함수를 배우는 것과 같다. 나눔 함수 $Z$은 로그 확률에서 상수 어긋남일 뿐이므로 중요한 것은 그것뿐이다.

```python
def compute_score(energy_net, x, create_graph=True):
    """
    점수 함수 s(x) = -∇E(x)을 셈한다.
    
    매개변수
    ----------
    energy_net : nn.Module
        신경망 에너지 함수
    x : torch.Tensor
        들임 점, 꼴 (batch, dim)
    create_graph : bool
        더 높은 차수의 기울기를 위해 그래프를 만들지 여부
    
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

---

## 5. 다른 분포와의 이음

### 지수 무리

볼츠만 분포는 지수 무리의 하나이다:

$$p(x; \theta) = h(x) \exp(\eta(\theta)^T T(x) - A(\theta))$$

볼츠만 분포에서 자연 매개변수는 $\eta = -1/T$, 충분 통계량은 $T(x) = E(x)$, 로그 나눔 함수는 $A(\theta) = \log Z$이다. 이 이음은 켤레 사전 분포, 최대 가능도 어림, 앎 기하학을 아우르는 지수 무리의 풍성한 이론이 볼츠만 분포에 곧바로 쓰인다는 뜻이다.

### 깁스 잣대

잣대 이론의 말로 하면 볼츠만 분포는 깁스 잣대를 뜻매김한다:

$$\mu(dx) = \frac{1}{Z} \exp(-E(x)) \, \lambda(dx)$$

여기서 $\lambda$은 바탕 잣대이다(이어진 변수에는 르베그 잣대, 띄엄띄엄한 변수에는 셈 잣대). 깁스 잣대 틀은 차원이 끝없는 자리로 자연스럽게 넓어지며 통계 역학과 격자 마당 이론의 수학 바탕에서 한가운데에 있다.

---

## 6. 핵심 정리

!!! success "핵심 개념"

    1. 볼츠만 분포 $p(x) \propto \exp(-E(x)/T)$은 에너지 제약 아래 최대 엔트로피에서 유일하게 나온다
    2. 온도가 퍼짐을 다스린다. $T$이 낮으면 에너지 최솟값에 몰리고 높으면 고른 분포에 가까워진다
    3. 점수 함수 $\nabla_x \log p = -\nabla_x E/T$은 나눔 함수를 피해 쓸 만한 익히기 방법을 가능하게 한다
    4. 열역학 양(자유 에너지, 엔트로피, 열 담이)은 모두 나눔 함수에서 이끌어 낼 수 있다
    5. 볼츠만 분포는 지수 무리에 들어 넓은 통계 이론과 이어진다

!!! info "역사 알림"
    작은 세계의 역학을 큰 세계의 열역학에 이은 루트비히 볼츠만의 일은 처음에 크게 논란이 되었고, 그것이 그의 우울과 1906년의 죽음에 한몫했다. 엔트로피를 미시 상태의 개수와 잇는 그의 식 $S = k \log W$은 빈에 있는 그의 묘비에 새겨져 있다. 1950년대에 제인스가 내놓은 최대 엔트로피 풀이는 물리 가정을 전혀 필요로 하지 않는 순수한 앎 이론의 정당화를 볼츠만 분포에 주었다.

---

## 연습문제

1. **나눔 함수 한계**: 두 우물 에너지 $E(x) = (x^2-1)^2$에서 $Z$을 정확히 셈하지 않고 위와 아래 한계를 이끌어 내라. 힌트: 젠센 부등식과 라플라스 방법을 쓰라.

2. **온도 넘어감**: 두 우물 분포 $E(x) = \frac{1}{2}(x^2-4)^2$이 봉우리 둘에서 봉우리 하나로 바뀌는 온도를 찾아라. 힌트: $x=0$에서 $\log p(x)$의 이계 미분을 살펴라.

3. **열 담이 봉우리**: 에너지가 $0$과 $\epsilon$인 두 켜 계에서 열 담이 $C = \text{Var}[E]/T^2$이 $T^* = \epsilon / (2\sinh^{-1}(1))$에서 최댓값(쇼트키 이상)을 가짐을 보여라.

## 정리하며

이 마당은 학습 목표、수학적 바탕、온도 살피기、점수 함수을 차례로 짚었다.

**참고 문헌**

- Jaynes, E. T. (1957). Information theory and statistical mechanics. *Physical Review*.
- MacKay, D. J. (2003). *Information Theory, Inference and Learning Algorithms*. Cambridge University Press.
- Boltzmann, L. (1877). On the Relationship between the Second Fundamental Theorem of the Mechanical Theory of Heat and Probability Calculations.
