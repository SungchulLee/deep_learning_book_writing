# 몬테카를로 드롭아웃의 표본 수렴

---

## 1. 두루 보기

MC 드롭아웃 불확실성 근사의 성능은 앞으로 걸음 횟수 $T$에 크게 매인다. 이 글은 엄밀한 수렴 살피기, 참으로 쓸 테두리, 표본 수 고르기 길잡이를 준다.

---

## 2. 이론으로 본 수렴 틀

### 몬테카를로 근사

MC 드롭아웃은 샘플링로 예측하는 분포를 추정한다.

$$
\mathbb{E}_{q_\theta(\omega)}[f(\mathbf{x}; \omega)] \approx \hat{\mu}_T = \frac{1}{T} \sum_{t=1}^{T} f(\mathbf{x}; \hat{\omega}_t)
$$

여기서 $\hat{\omega}_t \sim q_\theta(\omega)$은 드롭아웃 분포에서 뽑은 서로 독립이고 고르게 흩어진 표본이다.

### 치우치지 않음

MC 추정량은 치우치지 않는다.

$$
\mathbb{E}[\hat{\mu}_T] = \mathbb{E}\left[\frac{1}{T} \sum_{t=1}^{T} f(\mathbf{x}; \hat{\omega}_t)\right] = \frac{1}{T} \sum_{t=1}^{T} \mathbb{E}[f(\mathbf{x}; \hat{\omega}_t)] = \mathbb{E}_{q_\theta}[f(\mathbf{x}; \omega)]
$$

### 평균 추정량의 분산

$\sigma^2_f = \text{Var}_{q_\theta}[f(\mathbf{x}; \omega)]$을 드롭아웃 분포 아래 네트워크 출력의 분산이라 하자. MC 평균 추정량의 분산은

$$
\text{Var}[\hat{\mu}_T] = \frac{\sigma^2_f}{T}
$$

**표준편차**:

$$
\text{SE}[\hat{\mu}_T] = \frac{\sigma_f}{\sqrt{T}}
$$

이는 여느 몬테카를로 빠르기인 $O(1/\sqrt{T})$으로 준다.

---

## 3. 수렴 테두리

### 가운데 끝 정리

$T$이 클 때 가운데 끝 정리에 따라

$$
\sqrt{T}(\hat{\mu}_T - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2_f)
$$

여기서 $\mu = \mathbb{E}_{q_\theta}[f(\mathbf{x}; \omega)]$이다.

이는 끝으로 가는 믿음 구간을 준다.

$$
P\left( \left| \hat{\mu}_T - \mu \right| \leq z_{\alpha/2} \frac{\sigma_f}{\sqrt{T}} \right) \approx 1 - \alpha
$$

95% 믿음에서는 $z_{0.025} \approx 1.96$이다.

### 마디 있는 표본 테두리(회프딩)

네트워크 출력이 $f(\mathbf{x}; \omega) \in [a, b]$으로 갇혀 있으면 회프딩 부등식이 준다.

$$
P\left( \left| \hat{\mu}_T - \mu \right| \geq \epsilon \right) \leq 2 \exp\left( -\frac{2T\epsilon^2}{(b-a)^2} \right)
$$

**있어야 할 표본 수로 뒤집기**:

확률 $1 - \delta$ 이상으로 $|\hat{\mu}_T - \mu| \leq \epsilon$을 이루려면

$$
T \geq \frac{(b-a)^2 \ln(2/\delta)}{2\epsilon^2}
$$

### 마디 있는 표본 테두리(체비쇼프)

갇혀 있다는 가정이 없으면 체비쇼프 부등식이 준다.

$$
P\left( \left| \hat{\mu}_T - \mu \right| \geq k \frac{\sigma_f}{\sqrt{T}} \right) \leq \frac{1}{k^2}
$$

95% 믿음($\delta = 0.05$)이면 $k = \sqrt{20} \approx 4.47$이 있어야 한다.

---

## 4. 분산 근사의 수렴

### 표본 분산 추정량

인식적 불확실성은 표본 분산으로 추정한다.

$$
\hat{\sigma}^2_T = \frac{1}{T-1} \sum_{t=1}^{T} \left( f(\mathbf{x}; \hat{\omega}_t) - \hat{\mu}_T \right)^2
$$

### 표본 분산의 분포

$f(\mathbf{x}; \omega)$이 거의 가우시안이면(깊은 네트워크에서는 가운데 끝 정리로 흔히 그럴듯하다)

$$
\frac{(T-1)\hat{\sigma}^2_T}{\sigma^2_f} \sim \chi^2_{T-1}
$$

**평균과 분산**:

$$
\mathbb{E}[\hat{\sigma}^2_T] = \sigma^2_f, \quad \text{Var}[\hat{\sigma}^2_T] = \frac{2\sigma^4_f}{T-1}
$$

### 분산 근사의 견준 어긋남

분산 추정량의 바뀜 값은

$$
\text{CV}[\hat{\sigma}^2_T] = \frac{\sqrt{\text{Var}[\hat{\sigma}^2_T]}}{\mathbb{E}[\hat{\sigma}^2_T]} = \sqrt{\frac{2}{T-1}}
$$

| $T$ | 분산의 견준 어긋남 |
|-----|---------------------------|
| 10 | 47% |
| 30 | 26% |
| 50 | 20% |
| 100 | 14% |
| 500 | 6% |

**뜻하는 바:** 분산 근사는 같은 맞음을 이루려면 평균 근사보다 표본이 훨씬 많이 있어야 한다.

### 분산의 믿음 구간

카이제곱 분포를 쓰면

$$
P\left( \frac{(T-1)\hat{\sigma}^2_T}{\chi^2_{T-1, \alpha/2}} \leq \sigma^2_f \leq \frac{(T-1)\hat{\sigma}^2_T}{\chi^2_{T-1, 1-\alpha/2}} \right) = 1 - \alpha
$$

---

## 5. 엔트로피와 서로 나눈 소식의 수렴

### 예측한 엔트로피 근사

분류에서 예측한 엔트로피는

$$
\mathbb{H}[\mathbf{y} | \mathbf{x}, \mathcal{D}] = -\sum_{c=1}^{C} p_c \log p_c
$$

여기서 $p_c = \mathbb{E}_{q_\theta}[\text{softmax}(f(\mathbf{x}; \omega))_c]$이다.

MC 근사는 $\hat{p}_c = \frac{1}{T} \sum_{t=1}^T \text{softmax}(f(\mathbf{x}; \hat{\omega}_t))_c$을 쓴다.

### 엔트로피 근사의 바이어스

겪은 분포의 엔트로피는 치우친 추정량이다.

$$
\mathbb{E}[\hat{\mathbb{H}}_T] = \mathbb{H}[p] - \frac{C - 1}{2T} + O(T^{-2})
$$

여기서 $C$은 클래스 수다. 바이어스는 음수이므로 엔트로피를 낮게 본다.

**밀러-매도 바로잡기**:

$$
\hat{\mathbb{H}}^{\text{MM}}_T = \hat{\mathbb{H}}_T + \frac{C - 1}{2T}
$$

### 서로 나눈 소식의 수렴

서로 나눈 소식 $\mathbb{I}[\mathbf{y}, \omega | \mathbf{x}, \mathcal{D}] = \mathbb{H}[\mathbf{y}] - \mathbb{E}[\mathbb{H}[\mathbf{y} | \omega]]$에는 다음이 든다.

1. 평균 낸 분포의 엔트로피(바이어스: $-\frac{C-1}{2T}$)
2. 낱낱 분포의 평균 엔트로피(치우치지 않음)

두 바이어스가 얼마쯤 서로 지우지만, $T$이 작으면 서로 나눈 소식 근사는 여전히 흔들릴 수 있다.

---

## 6. 겪어 본 수렴 살피기

### 수렴 살펴보기

```python
import torch
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

def compute_convergence_diagnostics(
    model: torch.nn.Module,
    x: torch.Tensor,
    max_samples: int = 500,
    checkpoints: List[int] = None
) -> dict:
    """
    MC 드롭아웃 어림의 모임을 살핀다.
    
    표본 수에 따른 평균, 흩어짐, 그리고 그 표준편차의
    달리는 어림을 돌려준다.
    """
    if checkpoints is None:
        checkpoints = [5, 10, 20, 30, 50, 100, 200, 300, 500]
    checkpoints = [c for c in checkpoints if c <= max_samples]
    
    model.eval()
    model.enable_mc_dropout()
    
    # 표본을 모두 모은다
    samples = []
    with torch.no_grad():
        for _ in range(max_samples):
            output = model(x)
            samples.append(output.cpu())
    
    samples = torch.stack(samples, dim=0)  # (T, B, D)
    
    results = {
        'checkpoints': checkpoints,
        'running_mean': [],
        'running_var': [],
        'mean_se': [],
        'var_se': []
    }
    
    for T in checkpoints:
        subset = samples[:T]
        
        # 달리는 평균과 흩어짐
        mean_T = subset.mean(dim=0)
        var_T = subset.var(dim=0, unbiased=True)
        
        results['running_mean'].append(mean_T)
        results['running_var'].append(var_T)
        
        # 표준편차(마지막 어림을 참값 대신 쓴다)
        # 평균의 SE = std / sqrt(T)
        results['mean_se'].append(subset.std(dim=0) / np.sqrt(T))
        
        # 흩어짐의 SE ≈ var * sqrt(2/(T-1))
        results['var_se'].append(var_T * np.sqrt(2 / (T - 1)))
    
    return results

def plot_convergence(results: dict, output_idx: int = 0, batch_idx: int = 0):
    """평균과 흩어짐 어림의 모임을 그린다."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    checkpoints = results['checkpoints']
    
    # 고른 날임의 값을 뽑는다
    means = [m[batch_idx, output_idx].item() for m in results['running_mean']]
    vars_ = [v[batch_idx, output_idx].item() for v in results['running_var']]
    mean_ses = [se[batch_idx, output_idx].item() for se in results['mean_se']]
    var_ses = [se[batch_idx, output_idx].item() for se in results['var_se']]
    
    # 평균의 모임
    axes[0].errorbar(checkpoints, means, yerr=[1.96*se for se in mean_ses],
                     marker='o', capsize=3)
    axes[0].axhline(means[-1], color='r', linestyle='--', alpha=0.5,
                    label=f'마지막 어림: {means[-1]:.4f}')
    axes[0].set_xlabel('MC 표본 수')
    axes[0].set_ylabel('어림한 평균')
    axes[0].set_title('평균의 모임')
    axes[0].set_xscale('log')
    axes[0].legend()
    
    # 흩어짐의 모임
    axes[1].errorbar(checkpoints, vars_, yerr=[1.96*se for se in var_ses],
                     marker='o', capsize=3)
    axes[1].axhline(vars_[-1], color='r', linestyle='--', alpha=0.5,
                    label=f'마지막 어림: {vars_[-1]:.4f}')
    axes[1].set_xlabel('MC 표본 수')
    axes[1].set_ylabel('어림한 흩어짐')
    axes[1].set_title('흩어짐의 모임')
    axes[1].set_xscale('log')
    axes[1].legend()
    
    plt.tight_layout()
    return fig
```

### 쓸모 있는 표본 크기

MC 표본이 참으로 남남이 아니면(이를테면 켜에 걸쳐 드롭아웃 마스크가 서로 얽히면) 쓸모 있는 표본 크기는 $T$보다 작을 수 있다.

$$
T_{\text{eff}} = \frac{T}{1 + 2\sum_{k=1}^{\infty} \rho_k}
$$

여기서 $\rho_k$은 뒤짐 $k$에서의 스스로 얽힘이다.

```python
def compute_effective_sample_size(samples: torch.Tensor) -> float:
    """
    스스로 얽힘을 헤아린 쓸모 있는 표본 크기를 셈한다.
    
    Args:
        samples: MC 표본의 (T, ...) 텐서
        
    Returns:
        쓸모 있는 표본 크기
    """
    T = samples.shape[0]
    samples_flat = samples.reshape(T, -1).mean(dim=1)  # 차수에 걸쳐 평균
    
    # 스스로 얽힘을 셈한다
    samples_centered = samples_flat - samples_flat.mean()
    var = (samples_centered ** 2).mean()
    
    if var < 1e-10:
        return float(T)
    
    # 스스로 얽힘 함수
    max_lag = min(T // 2, 100)
    rho_sum = 0
    
    for k in range(1, max_lag):
        rho_k = (samples_centered[:-k] * samples_centered[k:]).mean() / var
        if rho_k < 0.05:  # 셈이 든든하도록 끊는 값
            break
        rho_sum += rho_k
    
    T_eff = T / (1 + 2 * rho_sum)
    return max(1.0, T_eff)
```

---

## 7. 참으로 표본 크기 고르기

### 일에 따른 길잡이

| 일 | 가장 적은 $T$ | 즐겨 쓰는 $T$ | 붙임말 |
|------|-------------|-----------------|-------|
| 점 추론 | 10~20 | 30 | 평균은 빨리 모인다 |
| 믿음 구간 | 30~50 | 100 | 분산이 든든해야 한다 |
| 캘리브레이션 굽이 | 100 | 200~500 | 통 자에는 촘촘함이 있어야 한다 |
| 살아 있는 학습 | 20~50 | 50~100 | 줄 세우기는 잡음에 든든하다 |
| 분포 외 알아내기 | 50~100 | 100~200 | 꼬리 특성이 걸린다 |
| 목숨이 걸린 일 | 200 이상 | 500 이상 | 조심스러운 길 |

### 맞추어 가는 샘플링 꾀

```python
def adaptive_mc_sampling(
    model: torch.nn.Module,
    x: torch.Tensor,
    initial_samples: int = 20,
    max_samples: int = 500,
    tolerance: float = 0.01,
    patience: int = 3
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    어림이 모이면 멈추는 맞추어 가는 MC 표본 뽑기.
    
    평균과 흩어짐 어림의 견준 바뀜을 지켜보다가
    둘 다 든든해지면 멈춘다.
    
    Args:
        model: MC 드롭아웃 모형
        x: 들임 텐서
        initial_samples: 처음 표본 수
        max_samples: 뽑을 수 있는 가장 많은 표본 수
        tolerance: 모였다고 볼 견준 바뀜 임계
        patience: 지나야 할 살핌 횟수
        
    Returns:
        mean: 모인 평균 어림
        std: 모인 표준편차 어림  
        n_samples: 참으로 쓴 표본 수
    """
    model.eval()
    model.enable_mc_dropout()
    
    samples = []
    consecutive_stable = 0
    prev_mean, prev_var = None, None
    
    batch_size = 10  # 되돌 때마다의 표본 수
    
    with torch.no_grad():
        # 처음 표본
        for _ in range(initial_samples):
            samples.append(model(x))
        
        while len(samples) < max_samples:
            # 표본을 더 더한다
            for _ in range(batch_size):
                if len(samples) >= max_samples:
                    break
                samples.append(model(x))
            
            # 이제까지의 어림을 셈한다
            stacked = torch.stack(samples, dim=0)
            curr_mean = stacked.mean(dim=0)
            curr_var = stacked.var(dim=0)
            
            # 모였는지 살핀다
            if prev_mean is not None:
                mean_change = (curr_mean - prev_mean).abs() / (curr_mean.abs() + 1e-8)
                var_change = (curr_var - prev_var).abs() / (curr_var.abs() + 1e-8)
                
                mean_stable = (mean_change < tolerance).all()
                var_stable = (var_change < tolerance).all()
                
                if mean_stable and var_stable:
                    consecutive_stable += 1
                    if consecutive_stable >= patience:
                        break
                else:
                    consecutive_stable = 0
            
            prev_mean, prev_var = curr_mean, curr_var
    
    final_samples = torch.stack(samples, dim=0)
    mean = final_samples.mean(dim=0)
    std = final_samples.std(dim=0)
    
    return mean, std, len(samples)
```

### 배치의 잘 듦 헤아리기

```python
def estimate_optimal_batch_samples(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_time_ms: float = 100.0,
    warmup_runs: int = 5
) -> int:
    """
    때 예산이 주어졌을 때 가장 알맞은 MC 표본 수를 어림한다.
    
    Args:
        model: MC 드롭아웃 모형
        x: 본보기가 되는 들임
        target_time_ms: 밀리초 단위 때 예산
        warmup_runs: 몸풀기 앞으로 걸음 횟수
        
    Returns:
        즐겨 쓸 MC 표본 수
    """
    import time
    
    model.eval()
    model.enable_mc_dropout()
    
    # 몸풀기
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(x)
    
    # 앞으로 걸음 하나의 때를 잰다
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    n_timing_runs = 20
    
    with torch.no_grad():
        for _ in range(n_timing_runs):
            _ = model(x)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed_ms = (time.perf_counter() - start) * 1000
    time_per_sample = elapsed_ms / n_timing_runs
    
    optimal_samples = int(target_time_ms / time_per_sample)
    
    # 이치에 닿는 테두리로 자른다
    return max(10, min(500, optimal_samples))
```

---

## 8. 이론이 주는 아래끝

### MC 근사의 크라메르-라오 테두리

표본 $T$개로 평균 $\mu$을 추정할 때 분산은 아래로 갇힌다.

$$
\text{Var}[\hat{\mu}_T] \geq \frac{\sigma^2_f}{T}
$$

MC 근사는 서로 독립인 표본에서 이 테두리에 다다른다(잘 든다).

### 소식 이론으로 본 눈

MC 근사와 참 파라메터 사이의 서로 나눈 소식은

$$
I(\hat{\mu}_T; \mu) = \frac{1}{2} \log\left(1 + \frac{T \sigma^2_\mu}{\sigma^2_f}\right)
$$

여기서 $\sigma^2_\mu$은 $\mu$의 앞선 분산이다. 소식은 $T$에 따라 로그로 는다.

### 줄어드는 보람

표본 $T$에서 $T+1$으로 갈 때 늘어나는 소식은

$$
\Delta I_T = I(\hat{\mu}_{T+1}; \mu) - I(\hat{\mu}_T; \mu) \approx \frac{\sigma^2_\mu}{2T\sigma^2_f} \quad \text{for large } T
$$

이 $O(1/T)$ 줄어듦이 표본을 더 뽑을 때의 보람이 줄어드는 정도를 잰다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
ReLU 활성화와 가우시안 웨이트 사전분포를 지닌 두 켜 뉴럴 넷트웍에서, 이 마디에서 밝힌 방법에 따른 근사 사후분포의 꼴을 이끌어 내어라.

</div>

??? success "연습문제 1 풀이"
    웨이트가 $W_1, W_2$이고 가우시안 사전분포가 $p(W) = \mathcal{N}(0, \sigma_p^2 I)$일 때 사후분포 $p(W | D) \propto p(D | W) p(W)$은 다룰 수 없다. 이 마디의 근사 방법은 다룰 수 있는 꼴을 낸다. 변분 추론이면 웨이트마다 서로 독립인 가우시안 사후분포 $q(w_{ij}) = \mathcal{N}(\mu_{ij}, \sigma_{ij}^2)$을 지니고, 라플라스 근사이면 사후분포가 MAP 추정을 가운데로 삼고 공분산이 헤세 행렬의 역인 가우시안 하나이며, MC 드롭아웃이면 사후분포가 드롭아웃 마스크 분포로 넌지시 세워진다. 근사마다 참 사후분포 모습의 서로 다른 특성을 담는다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 방법에서 얻은 불확실성 근사의 캘리브레이션을 MC 드롭아웃, 깊은 앙상블과 비교하는 시험을 꾸며라. 쓸 자와 그림을 밝혀라.

</div>

??? success "연습문제 2 풀이"
    자: (1) 통 15개의 바라는 캘리브레이션 오차(ECE), (2) 브라이어 점수, (3) 음수 로그 우도(NLL), (4) 분포 외 알아내기의 AUROC. 그림: 방법마다 본 잦기를 예측한 신뢰도에 대고 그린 미더움 그림. 절차: 모든 방법을 CIFAR-10(분포 안)에서 익히고, CIFAR-10 시험 데이터에서 캘리브레이션을, SVHN에서 분포 외 알아내기를 따진다. 온도 잣대 잡기를 일 끝난 뒤 베이스라인으로 쓴다. 아무렇게나 하는 시드 5개에 걸친 평균과 표준편차를 알린다. 눈금이 잘 맞은 방법은 미더움 그림에서 점이 대각선에 가깝고 ECE가 낮다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
베이즈 뉴럴 넷트웍의 추론 분산이 인식적 불확실성과 우연적 불확실성으로 쪼개짐을 증명하여라. 학습 데이터 크기 $N \to \infty$일 때 두 몫이 어떻게 되는지 보여라.

</div>

??? success "연습문제 3 풀이"
    추론 분산은 온 분산 법칙으로 쪼개진다. $\text{Var}[y | x, D] = \underbrace{\mathbb{E}_{p(\theta|D)}[\text{Var}[y | x, \theta]]}_{\text{타고난}} + \underbrace{\text{Var}_{p(\theta|D)}[\mathbb{E}[y | x, \theta]]}_{\text{앎의}}$. 우연적 몫은 데이터를 낳는 흐름의 줄일 수 없는 잡음을 담으며 $N \to \infty$이어도 그대로다. 인식적 몫은 파라메터의 불확실성을 드러내며 사후분포가 참 파라메터 언저리로 모이므로 $O(1/N)$으로 준다. 끝에 가면 우연적 불확실성만 남는다. 이 쪼갬은 데이터를 더 모아야 할 때(인식적 불확실성이 클 때)와 우연적 잡음을 받아들여야 할 때(우연적 불확실성이 클 때)를 가르는 데 중요하다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
이 마디의 불확실성 재기 방법을 거래 구조의 자리 크기 잡기에 어떻게 쓸 수 있는지 다루어라. 손에 잡히는 판단 규칙을 내놓아라.

</div>

??? success "연습문제 4 풀이"
    판단 규칙: 자리 크기를 인식적 불확실성에 반비례하게 잡는다. $\hat{y}$을 예측한 돌아옴, $\sigma_e^2$을 인식적 분산이라 하자. 자리는 $w = \frac{\hat{y}}{\lambda \sigma_e^2}$이고 $\lambda$은 위험 꺼림 값이다. 인식적 불확실성이 크면(낯선 시장 상황) 자리를 줄이고, 작으면(익숙한 판) 더 굳게 거래한다. 여기에 더해 그 위로는 거래하지 않는 인식적 불확실성 위끝을 두어 삼갈 수 있다. 이 틀은 모델의 신뢰도로 잣대를 잡은 켈리 잣대 특성의 크기 잡기를 절로 이룬다. 앞으로 걸어가며 살피기로 되짚어 시험해 $\lambda$의 눈금을 맞춘다. $\square$

## 정리하며

**핵심으로 챙길 것**:

1. **평균 근사**는 $O(1/\sqrt{T})$으로 모인다 — 표본 100개면 $\sigma_f$ 대비 표준편차가 약 10%다

2. **분산 근사**는 더 더디게 모여, 약 14%의 견준 어긋남에 $T \approx 100$이 있어야 한다

3. **웬만한 쓰임에서는** $T = 50$~$100$이 맞음과 잘 듦 사이의 좋은 자리다

4. **목숨이 걸린 쓰임에서는** 수렴을 살피며 $T \geq 200$을 써야 한다

5. **맞추어 가는 샘플링**은 근사가 일찍 든든해지면 계산을 줄일 수 있다

**살펴볼 거리**

1. Gal, Y. (2016). Uncertainty in Deep Learning. *PhD Thesis*.

2. Robert, C. P., & Casella, G. (2004). Monte Carlo Statistical Methods. *Springer*.

3. Geyer, C. J. (1992). Practical Markov Chain Monte Carlo. *Statistical Science*.
