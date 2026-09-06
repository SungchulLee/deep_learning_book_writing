# 나눔 함수



## 학습 목표

이 절을 마치면 다음을 할 수 있다:

1. 나눔 함수를 에너지 바탕 나타내기의 한가운데 양으로 이해한다
2. 열역학 양을 $\log Z$의 미분으로 이끌어 낸다
3. 나눔 함수가 왜 다룰 수 없는지와 익히기에 미치는 결과를 밝힌다
4. 차원이 낮을 때 나눔 함수를 어림하는 수치 방법을 쓴다

## 뜻매김과 몫

나눔 함수는 볼츠만 분포의 고르게 맞추는 상수이다:

$$Z(T) = \int_{\mathcal{X}} \exp\left(-\frac{E(x)}{T}\right) dx$$

띄엄띄엄한 상태 공간에서는 $Z(T) = \sum_{x \in \mathcal{X}} \exp(-E(x)/T)$이다.

"그저" 고르게 맞추는 상수인데도 $Z$은 통계 역학과 에너지 바탕 나타내기에서 가장 중요한 양이라 할 만하다. 이는 계에 대한 온전한 분포 앎을 담는다. 모든 기댓값, 적률, 열역학 양을 $Z$과 그 미분에서 뽑아낼 수 있다.

## Z에서 얻는 열역학 양

나눔 함수는 열역학 양의 만들개 함수 노릇을 한다:

| 양 | 공식 |
|----------|---------|
| 자유 에너지 | $F = -T \log Z$ |
| 평균 에너지 | $\beta = 1/T$일 때 $\langle E \rangle = -\frac{\partial \log Z}{\partial \beta}$ |
| 엔트로피 | $S = \frac{\langle E \rangle - F}{T} = \beta \langle E \rangle + \log Z$ |
| 열 담이 | $C = \frac{\partial \langle E \rangle}{\partial T} = \frac{\text{Var}[E]}{T^2}$ |
| 에너지 흩어짐 | $\text{Var}[E] = \frac{\partial^2 \log Z}{\partial \beta^2}$ |

열 담이 공식 $C = \text{Var}[E]/T^2$은 특히 배울 점이 많다. 이는 에너지의 흔들림(작은 세계의 양)이 온도에 따른 평균 에너지의 바뀜 빠르기(큰 세계의 양)를 정함을 보인다. 이 흔들림-대꾸 관계는 통계 역학의 표징이다.

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

def compute_thermodynamic_quantities(energy_fn, x_range, T):
    """
    나눔 함수에서 열역학 양을 셈한다.
    
    매개변수
    ----------
    energy_fn : callable
        Energy function E(x)
    x_range : torch.Tensor
        수치 적분을 위한 자리의 점
    T : float
        Temperature
    
    반환값
    -------
    사전
        열역학 양을 담은 사전
    """
    dx = x_range[1] - x_range[0]
    E = energy_fn(x_range)
    
    # 나눔 함수(안정을 위해 로그 합 지수)
    log_unnorm = -E / T
    log_Z = torch.logsumexp(log_unnorm, dim=0) + torch.log(dx)
    Z = torch.exp(log_Z)
    
    # 확률 분포
    p = torch.exp(log_unnorm - log_Z)
    
    # 평균 에너지
    avg_E = torch.sum(p * E) * dx
    
    # 자유 에너지
    F = -T * log_Z
    
    # 엔트로피
    S = (avg_E - F) / T
    
    # 에너지의 흩어짐(열 담이용)
    var_E = torch.sum(p * (E - avg_E)**2) * dx
    C = var_E / (T**2)  # Heat capacity
    
    return {
        'Z': Z.item(),
        'F': F.item(),
        'avg_E': avg_E.item(),
        'S': S.item(),
        'C': C.item()
    }

# 온도에 걸쳐 두 우물 퍼텐셜을 살핀다
x = torch.linspace(-4, 4, 1000)
double_well = lambda x: 0.5 * (x**2 - 4)**2

print("Thermodynamic quantities for double-well potential:")
print(f"{'T':>5} {'⟨E⟩':>10} {'F':>10} {'S':>10} {'C':>10}")
print("-" * 50)
for T in [0.5, 1.0, 2.0, 5.0]:
    q = compute_thermodynamic_quantities(double_well, x, T)
    print(f"{T:5.1f} {q['avg_E']:10.3f} {q['F']:10.3f} {q['S']:10.3f} {q['C']:10.3f}")
```

### 상 바뀜의 자취

물리 계에서 상 바뀜은 $Z$에서 이끌어 낸 열역학 양의 특이점으로 드러난다. 기계 배움의 에너지 바탕 모델은 (계의 크기가 끝없어야 하는) 참된 상 바뀜을 겪지 않지만 비슷한 현상은 나타난다:

**날카로운 에너지 담**: 에너지 풍경에 높은 담으로 잘 갈린 봉우리들이 있으면 열 담이 $C(T)$이 어떤 특징 온도에서 봉우리를 보인다. 그 온도 아래에서는 계가 봉우리 하나에 "얼어붙고" 위에서는 봉우리 사이를 자주 오간다.

**봉우리 다툼**: 임계 온도에서 분포가 봉우리 여럿에서 사실상 봉우리 하나로 바뀐다. 이 온도는 봉우리 하나에 머무는 엔트로피 비용과 에너지 이득이 균형을 이루는 곳을 가리킨다.

```python
def visualize_phase_transition_analogy():
    """
    열역학 양이 얼개 바뀜을 어떻게 알리는지 보여 준다
    볼츠만 분포에서.
    """
    x = torch.linspace(-4, 4, 1000)
    E = 0.5 * (x**2 - 4)**2  # Double-well
    
    temperatures = torch.linspace(0.1, 8.0, 100)
    quantities = {'T': [], 'avg_E': [], 'S': [], 'C': []}
    
    for T in temperatures:
        q = compute_thermodynamic_quantities(lambda x: 0.5*(x**2-4)**2, x, T.item())
        quantities['T'].append(T.item())
        quantities['avg_E'].append(q['avg_E'])
        quantities['S'].append(q['S'])
        quantities['C'].append(q['C'])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(quantities['T'], quantities['avg_E'], 'r-', linewidth=2)
    axes[0].set_xlabel('Temperature T')
    axes[0].set_ylabel('⟨E⟩')
    axes[0].set_title('Average Energy')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(quantities['T'], quantities['S'], 'b-', linewidth=2)
    axes[1].set_xlabel('Temperature T')
    axes[1].set_ylabel('Entropy S')
    axes[1].set_title('Entropy')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(quantities['T'], quantities['C'], 'g-', linewidth=2)
    axes[2].set_xlabel('Temperature T')
    axes[2].set_ylabel('Heat Capacity C')
    axes[2].set_title('Heat Capacity (Peak ≈ Transition)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_phase_transition_analogy()
```

## 다룰 수 없음 문제

차원이 높은 공간에서 나눔 함수는 다룰 수 없어진다:

$$Z = \int_{\mathbb{R}^d} \exp(-E(x)) \, dx$$

### 왜 다룰 수 없는가?

**적분 자리의 지수 자람**: 격자 바탕 수치 적분에는 $O(n^d)$번의 매김이 필요하다. 여기서 $n$은 차원마다 격자점의 개수이다. (금융 자료로는 대수롭지 않은 차원인) $d = 100$에서 $n = 10$만 되어도 $10^{100}$번을 매겨야 하며 이는 볼 수 있는 우주의 원자 수를 훌쩍 넘는다.

**일반 에너지 함수에는 닫힌 꼴이 없음**: 닫힌 꼴 나눔 함수는 특별한 경우에만 있다(정규 에너지 → 정규 적분, 선형 에너지 → 라플라스 변환). 신경망 에너지 함수에는 닫힌 꼴 $Z$이 없다.

**몬테카를로 어림의 큰 흩어짐**: $x_i$을 고르게 뽑는 순진한 어림개 $\hat{Z} = \frac{1}{N}\sum_i \exp(-E(x_i))$은 흩어짐이 아주 크다. $\exp(-E(x))$이 자리에 걸쳐 자릿수가 여러 번 달라지기 때문이다.

### 익히기에 미치는 결과

$Z$을 다룰 수 없다는 것은 깊은 결과를 낳는다:

**가능도를 곧바로 매길 수 없음**: $\log p(x) = -E(x) - \log Z$을 정확히 셈할 수 없으므로 여느 최대 가능도 익히기를 곧바로 쓸 수 없다.

**기울기에 모델 기댓값이 필요함**: 최대 가능도 기울기 $\nabla_\theta \log p(x) = -\nabla_\theta E(x) + \mathbb{E}_{p_\theta}[\nabla_\theta E]$에는 모델 분포의 표본("음의 국면")이 필요하고 그 자체가 다룰 수 없는 분포에서 마르코프 사슬 몬테카를로로 뽑기를 요구한다.

**모델 견주기가 어려움**: 가능도가 없으면 에너지 바탕 모델끼리 또는 다른 모델 무리와 견주는 데 다른 잣대(그림에는 프레셰 인셉션 거리, 에너지 바탕 따지기 등)가 필요하다.

### Z을 피하는 익히기 방법

$Z$을 다룰 수 없다는 점이 여러 아름다운 익히기 방식의 까닭이 되었으며 저마다 26.3절에서 자세히 다룬다:

**맞댐 벌어짐**: 자료에서 시작한 짧은 마르코프 사슬 몬테카를로로 모델 기댓값을 어림한다. 핵심 통찰은 자료 분포 가까이에서 시작하면 필요한 섞임 시간이 줄어든다는 것이다.

**점수 맞추기**: 모델 로그 밀도의 기울기를 자료의 것에 맞춘다. $\nabla_x \log p(x) = -\nabla_x E(x)/T$이 $Z$에 매이지 않으므로 $Z$을 아예 피한다.

**잡음 맞댐 어림**: 밀도 어림을 자료와 잡음을 가르는 가름 문제로 바꾸며, 나눔 함수가 아니라 에너지 함수만 있으면 된다.

## 나눔 함수 어림

차원이 높으면 $Z$을 정확히 셈할 수 없지만, 몇몇 어림 방법이 따지기, 모델 견주기, 차원 낮은 진단에 쓸모 있다.

### 담금질 중요도 표집(AIS)

식힘 중요도 뽑기는 다룰 만한 바탕에서 목표까지 다리를 놓는 중간 분포의 차례를 세워 $Z$을 어림한다:

$$\hat{Z} = Z_0 \cdot \prod_{k=1}^{K} \frac{p_{k}(x^{(k)})}{p_{k-1}(x^{(k)})}$$

여기서 $p_0$은 다룰 만한 분포(보기로 고른 분포나 정규 분포)이고 $p_K = p_\theta$은 목표이다. 이 방식은 익힌 에너지 바탕 모델의 나눔 함수를 어림하는 으뜸 잣대이다.

```python
def annealed_importance_sampling(energy_fn, dim, n_chains=100,
                                  n_intermediate=100, n_gibbs=10):
    """
    식힘 중요도 뽑기로 log Z을 어림한다.
    
    매개변수
    ----------
    energy_fn : callable
        Energy function E(x)
    dim : int
        Dimensionality
    n_chains : int
        나란한 사슬의 개수
    n_intermediate : int
        중간 분포의 개수
    n_gibbs : int
        중간 분포마다 깁스 걸음 수
    
    반환값
    -------
    float
        어림한 log Z
    """
    # 역온도를 0(고른 분포)에서 1(목표)까지
    betas = torch.linspace(0, 1, n_intermediate + 1)
    
    # 바탕 분포(표준 정규)에서 첫자리매김한다
    x = torch.randn(n_chains, dim)
    
    # 로그 중요도 무게
    log_weights = torch.zeros(n_chains)
    
    for k in range(1, n_intermediate + 1):
        # 지금 온도와 앞 온도의 에너지
        E = energy_fn(x)
        
        # 로그 무게를 쌓는다
        log_weights += -(betas[k] - betas[k-1]) * E
        
        # 옮김: 중간 온도의 랑주뱅 움직임
        for _ in range(n_gibbs):
            x.requires_grad_(True)
            E_current = energy_fn(x) * betas[k]
            grad = torch.autograd.grad(E_current.sum(), x)[0]
            x = x.detach() - 0.01 * grad + 0.005 * torch.randn_like(x)
    
    # 로그 평균 지수로 얻은 log Z 어림
    log_Z_ref = 0.5 * dim * np.log(2 * np.pi)  # Log Z of standard normal
    log_Z = torch.logsumexp(log_weights, dim=0) - np.log(n_chains) + log_Z_ref
    
    return log_Z.item()
```

### 다리 표집

다리 뽑기는 바탕 분포와 목표 분포 양쪽의 표본을 써서 비 $Z_\text{target}/Z_\text{reference}$을 어림한다. 두 분포 모두 뽑기 쉬울 때는 식힘 중요도 뽑기보다 효율 좋을 수 있지만, 목표에서 뽑는 것 자체가 어려운 에너지 바탕 모델에서는 덜 쓰인다.

## 근본 양으로서의 자유 에너지

실제로는 자유 에너지 $F = -T \log Z$이 $Z$ 자체보다 쓸모 있을 때가 많다. 자유 에너지는 변분 원리를 만족한다:

$$F = \min_q \left[\mathbb{E}_q[E(x)] + T \cdot H[q]\right]$$

여기서 최솟값은 모든 분포 $q$에 대해 잡고 $H[q]$은 $q$의 엔트로피이다. 가장 좋은 $q^*$은 바로 볼츠만 분포이다.

이 변분 성격 매김은 기계 배움의 변분 추론과 곧바로 이어진다. 증거 아래 한계(ELBO)는 자유 에너지 한계이며, 변분 자기 부호기는 제한된 변분 분포 무리로 자유 에너지를 가장 작게 하는 것으로 이해할 수 있다.

## 핵심 정리

!!! success "핵심 개념"

    1. 나눔 함수 $Z = \int \exp(-E(x)/T)\,dx$은 볼츠만 분포를 고르게 맞추고 그 미분으로 모든 열역학 양을 만들어 낸다
    2. 차원이 높고 두루 쓰는 에너지 함수에서 $Z$은 셈할 수 없다. 이것이 에너지 바탕 모델의 한가운데 셈 어려움이다
    3. 익히기 방법(맞댐 벌어짐, 점수 맞추기, 잡음 맞댐 어림)은 근본으로 $Z$을 셈하지 않으려는 셈속이다
    4. 자유 에너지 $F = -T\log Z$은 변분 추론과 이어지는 변분 원리를 만족한다
    5. 식힘 중요도 뽑기가 따지기 목적에서 $Z$의 가장 좋은 실제 어림을 준다

!!! warning "흔한 오해"

    - 나눔 함수는 "그저 상수"가 아니다. 모델 매개변수 $\theta$에 매이며 익히는 동안 셈에 넣어야 한다
    - $Z$을 다룰 수 없다고 해서 에너지 바탕 모델을 제대로 익힐 수 없다는 뜻은 아니다. 영리한 어림이 필요하다는 뜻이다
    - 차원이 낮을 때의 $Z$ 어림(격자 적분)은 차원의 저주 때문에 차원이 높은 곳으로 넓혀지지 않는다

## 참고 문헌

- Neal, R. M. (2001). Annealed Importance Sampling. *Statistics and Computing*.
- Meng, X.-L., & Wong, W. H. (1996). Simulating ratios of normalizing constants via a simple identity. *Statistica Sinica*.
- Salakhutdinov, R., & Murray, I. (2008). On the quantitative analysis of deep belief networks. *ICML*.

## 연습문제

1. **정확한 값과 어림값**: 마음대로 고른 2차원 에너지 함수에서 격자 적분으로 $Z$을 정확히 셈하고 식힘 중요도 뽑기 어림과 견주어라. 정확히 어림하려면 중간 분포가 몇 개 필요한가?

2. **변분 자유 에너지**: 정규 변분 분포 무리 $q$에 대해 변분 자유 에너지 $F_q = \mathbb{E}_q[E] + T \cdot H[q]$을 짜라. $q$을 가장 좋게 하고 변분 한계를 참 자유 에너지와 견주어라.

3. **열역학 적분**: 서로 다른 두 에너지 함수의 로그 나눔 함수 비를 어림하도록 열역학 적분 $\log Z_1 - \log Z_0 = \int_0^1 \langle E_1 - E_0 \rangle_\beta \, d\beta$을 짜라.
