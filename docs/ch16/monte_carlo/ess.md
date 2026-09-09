# 실효 표본 크기
실효 표본 크기(ESS)는 무게 준 표본이나 얽힌 표본 묶음의 정보량을 잰다. 중요도 표집에서 MCMC까지 몬테카를로 방법 전반에 쓰이는 근본 진단이다. 이 마당은 (중요도 표집처럼) 서로 독립인 무게 준 표본의 ESS을 다룬다. 자기상관에 기댄 MCMC의 ESS은 [진단](../mcmc/diagnostics.md)을 보아라.

---

## 1. 문제: 표본이 다 같지는 않다

중요도 표집에서는 제안 $q(x)$에서 표본 $N$개를 뽑고 무게 $w_i = p(x_i)/q(x_i)$을 준다. 무게가 몹시 고르지 않아 몇몇 표본이 좌우하면, 어림자는 사실상 $N$개보다 훨씬 적은 표본을 쓰는 셈이다.

MCMC에서는 잇달은 표본이 서로 얽혀 있다. 얽힌 표본 $N$개는 독립인 표본 $N$개보다 정보를 적게 담는다.

ESS은 이런 비효율을 숫자 하나로 잰다.

---

## 2. 중요도 표집의 ESS

### 정의

고르게 한 중요도 무게 $\bar{w}_i = w_i / \sum_j w_j$이 주어졌을 때 실효 표본 크기는 다음과 같다:

$$
\text{ESS} = \frac{1}{\sum_{i=1}^N \bar{w}_i^2}
$$

### 성질

- **범위**: $1 \leq \text{ESS} \leq N$
- **최댓값**($\text{ESS} = N$): 무게가 모두 같다($\bar{w}_i = 1/N$). 곧 $q = p$이다
- **최솟값**($\text{ESS} = 1$): 무게 하나가 1이고 나머지는 0이다(무게가 온통 주저앉음)

### 유도

ESS은 무게 준 어림자와 같은 흩어짐을 낼 $p$의 독립 표본 개수로 정한다:

$$
\text{Var}\left[\hat{I}_{\text{IS}}\right] \approx \frac{\text{Var}_p[f(X)]}{\text{ESS}}
$$

스스로 고르게 하는 어림자에서는 다음이 나온다:

$$
\text{ESS} \approx \frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2} = \frac{1}{\sum_i \bar{w}_i^2}
$$

### 해석

| ESS / N | 질 | 풀이 |
|---------|---------|----------------|
| > 0.5 | 아주 좋음 | 제안이 과녁과 잘 맞는다 |
| 0.1 - 0.5 | 좋음 | 대부분의 쓰임새에 넉넉하다 |
| 0.01 - 0.1 | 나쁨 | 결과가 미덥지 않을 수 있다 |
| < 0.01 | 아주 나쁨 | 사실상 무게가 주저앉음 |

### PyTorch 구현

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

---

## 3. MCMC의 ESS

### 정의

얽힌 MCMC 표본에서 ESS은 자기상관을 헤아린다:

$$
\text{ESS} = \frac{N}{1 + 2\sum_{k=1}^{K} \rho_k}
$$

여기서 $\rho_k$은 뒤짐 $k$에서의 자기상관이며, 자기상관이 시끄러워지면 합을 끊는다.

### 해석

- **높은 상관**: $\text{ESS} \ll N$(실효 독립 표본 하나마다 표본이 많이 필요하다)
- **낮은 상관**: $\text{ESS} \approx N$(효율적인 섞임)
- **독립 표본**: $\text{ESS} = N$(가장 좋지만 MCMC에서는 불가능하다)

### 몬테카를로 오차와의 이음

MCMC 어림자의 흩어짐은 다음처럼 커진다:

$$
\text{Var}\left[\frac{1}{N}\sum_{t=1}^N f(X^{(t)})\right] = \frac{\sigma_f^2}{N_{\text{eff}}}
$$

곧 ESS이 뒤확률 어림값의 정밀도를 곧바로 다스린다는 뜻이다.

---

## 4. ESS 지켜보고 낫게 하기

### 중요도 표집의 경우

- **더 좋은 제안 고르기**: 과녁의 모양에 맞춘다
- **알아서 맞추는 방법 쓰기**: 앞선 되풀이에 기대어 제안을 새로 고친다
- **차원 줄이기**: 차원이 낮은 문제일수록 ESS이 좋다

### MCMC의 경우

- **걸음 크기 맞추기**: 가장 좋은 받아들임 비율이 ESS을 낫게 한다
- **HMC/NUTS 쓰기**: 무작위 걸음보다 자기상관이 낮다
- **매개변수 바꾸기**: 가운데를 벗긴 매개변수화가 상관을 줄인다
- **더 오래 돌리기**: ESS은 사슬 길이에 선형으로 자란다

### 초당 ESS

가장 쓸모 있는 효율 잣대는 ESS과 셈 값을 함께 본다:

$$
\text{Efficiency} = \frac{\text{ESS}}{\text{wall-clock time}}
$$

되풀이마다의 ESS은 낮아도 되풀이가 빠른 방법이 통틀어 더 효율적일 수 있다.

---

## 연습문제

**연습문제 1.**
과녁 적분이 끝이 있는데도 중요도 표집의 흩어짐이 왜 끝없을 수 있는지 설명하여라.

??? success "연습문제 1 풀이"
    중요도 표집 어림자의 흩어짐은 $\text{Var}_q[w(x) f(x)]$에 비례하며, 여기서 $w(x) = p(x)/q(x)$은 중요도 무게이다. $q(x)$의 꼬리가 $p(x) f(x)$보다 가벼우면, $q$은 확률을 거의 주지 않는데 $p$은 주는 구역에서 비 $p(x)/q(x)$이 한없이 커질 수 있다. 그러면 이따금 어림값을 좌우하는 몹시 큰 무게가 생겨, 적분 $\mathbb{E}_p[f(X)]$이 끝이 있는데도 흩어짐이 끝없어진다(또는 사실상 끝없어진다).

---

**연습문제 2.**
중요도 무게 $w_1, \ldots, w_N$으로 나타낸 실효 표본 크기(ESS)의 공식을 이끌어 내어라.

??? success "연습문제 2 풀이"
    ESS은 무게 준 표본이 과녁 분포의 독립 표본 몇 개에 맞먹는지를 잰다:

    $$\text{ESS} = \frac{\left(\sum_{i=1}^N w_i\right)^2}{\sum_{i=1}^N w_i^2}$$

    무게가 모두 같으면($w_i = c$) ESS $= N$이다. 무게 하나가 좌우하면 ESS $\approx 1$이다. 이는 스스로 고르게 하는 중요도 표집 어림자의 흩어짐을 과녁에서 뽑은 독립 동일 분포 표본의 흩어짐에 견주어 뜯어보면 나온다.

---

**연습문제 3.**
중요도 표집으로 $\mathbb{E}_p[f(X)]$을 어림할 때 가장 좋은 제안 분포가 $q^*(x) \propto |f(x)| p(x)$임을 보여라.

??? success "연습문제 3 풀이"
    중요도 표집 어림자의 흩어짐은 $\text{Var}_q\left[\frac{f(X)p(X)}{q(X)}\right] / N$이다. 제약 $\int q(x) dx = 1$ 아래 라그랑주 곱수로 이를 $q$에 대해 가장 작게 하면 $q^*(x) = |f(x)| p(x) / \int |f(x')| p(x') dx'$이 나온다. $f \geq 0$일 때 이것이 흩어짐 0인 제안이다(어림자가 표본 하나로 정확한 답을 되돌린다). 실전에서 $q^*$은 우리가 셈하려는 바로 그 적분을 필요로 하므로 쓸 수 없다.

---

**연습문제 4.**
$X \sim \mathcal{N}(0,1)$일 때 $t$분포를 제안으로 써서 $\mathbb{E}[X^2]$의 단순한 중요도 표집 어림자를 구현하여라.

??? success "연습문제 4 풀이"
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
    $t$분포는 가우스보다 꼬리가 무거워 중요도 무게의 흩어짐이 끝이 있음을 보장한다.

## 정리하며

| 자리 | ESS 공식 | 범위 |
|---------|-------------|-------|
| **중요도 표집** | $1 / \sum \bar{w}_i^2$ | $[1, N]$ |
| **MCMC** | $N / (1 + 2\sum \rho_k)$ | $[1, N]$ |
| **받아들일 만한 최솟값** | — | > 100(점 어림값), > 400(구간) |

---

**참고 문헌**

1. Kong, A. (1992). A note on importance sampling using standardized weights. *Technical Report*, University of Chicago.
2. Liu, J. S. (2001). *Monte Carlo Strategies in Scientific Computing*. Springer. 2장.
3. Vehtari, A., et al. (2021). Rank-normalization, folding, and localization: An improved $\hat{R}$ for assessing convergence of MCMC. *Bayesian Analysis*, 16(2), 667-718.
