# 몬테카를로 드롭아웃의 표본 수렴

---

## 1. 개관

MC 드롭아웃 불확실성 근사의 성능은 순전파 횟수 $T$에 크게 달려 있다. 이 글은 엄밀한 수렴 분석, 실용적인 바운드, 표본 수 선택 지침을 준다.

---

## 2. 이론으로 본 수렴 틀

### 몬테카를로 근사

MC 드롭아웃은 샘플링으로 예측 분포를 근사한다.

$$
\mathbb{E}_{q_\theta(\omega)}[f(\mathbf{x}; \omega)] \approx \hat{\mu}_T = \frac{1}{T} \sum_{t=1}^{T} f(\mathbf{x}; \hat{\omega}_t)
$$

여기서 $\hat{\omega}_t \sim q_\theta(\omega)$는 드롭아웃 분포에서 뽑은 독립 동일 분포(i.i.d.) 표본이다.

### 비편향성

MC 추정량은 비편향(unbiased) 추정량이다 — 기댓값이 참값과 같다.

$$
\mathbb{E}[\hat{\mu}_T] = \mathbb{E}\left[\frac{1}{T} \sum_{t=1}^{T} f(\mathbf{x}; \hat{\omega}_t)\right] = \frac{1}{T} \sum_{t=1}^{T} \mathbb{E}[f(\mathbf{x}; \hat{\omega}_t)] = \mathbb{E}_{q_\theta}[f(\mathbf{x}; \omega)]
$$

### 평균 추정량의 분산

$\sigma^2_f = \text{Var}_{q_\theta}[f(\mathbf{x}; \omega)]$를 드롭아웃 분포 아래 네트워크 출력의 분산이라 하자. MC 평균 추정량의 분산은

$$
\text{Var}[\hat{\mu}_T] = \frac{\sigma^2_f}{T}
$$

**표준오차**:

$$
\text{SE}[\hat{\mu}_T] = \frac{\sigma_f}{\sqrt{T}}
$$

즉 몬테카를로의 전형적인 속도 $O(1/\sqrt{T})$로 줄어든다.

---

## 3. 수렴 바운드

### 중심극한정리

$T$가 클 때 중심극한정리에 따라

$$
\sqrt{T}(\hat{\mu}_T - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2_f)
$$

여기서 $\mu = \mathbb{E}_{q_\theta}[f(\mathbf{x}; \omega)]$이다.

이로부터 점근적 신뢰구간을 얻는다.

$$
P\left( \left| \hat{\mu}_T - \mu \right| \leq z_{\alpha/2} \frac{\sigma_f}{\sqrt{T}} \right) \approx 1 - \alpha
$$

95% 신뢰수준에서는 $z_{0.025} \approx 1.96$이다.

### 유한 표본 바운드(회프딩)

네트워크 출력이 $f(\mathbf{x}; \omega) \in [a, b]$로 유계이면 회프딩 부등식에서 다음을 얻는다.

$$
P\left( \left| \hat{\mu}_T - \mu \right| \geq \epsilon \right) \leq 2 \exp\left( -\frac{2T\epsilon^2}{(b-a)^2} \right)
$$

**필요한 표본 수로 바꾸기**:

확률 $1 - \delta$ 이상으로 $|\hat{\mu}_T - \mu| \leq \epsilon$을 이루려면

$$
T \geq \frac{(b-a)^2 \ln(2/\delta)}{2\epsilon^2}
$$

### 유한 표본 바운드(체비쇼프)

유계 가정이 없으면 체비쇼프 부등식에서 다음을 얻는다.

$$
P\left( \left| \hat{\mu}_T - \mu \right| \geq k \frac{\sigma_f}{\sqrt{T}} \right) \leq \frac{1}{k^2}
$$

95% 신뢰수준($\delta = 0.05$)이면 $k = \sqrt{20} \approx 4.47$이 필요하다.

---

## 4. 분산 근사의 수렴

### 표본 분산 추정량

인식적 불확실성은 표본 분산으로 추정한다.

$$
\hat{\sigma}^2_T = \frac{1}{T-1} \sum_{t=1}^{T} \left( f(\mathbf{x}; \hat{\omega}_t) - \hat{\mu}_T \right)^2
$$

### 표본 분산의 분포

$f(\mathbf{x}; \omega)$가 거의 가우시안이면(깊은 네트워크에서는 중심극한정리 덕에 흔히 그럴듯하다)

$$
\frac{(T-1)\hat{\sigma}^2_T}{\sigma^2_f} \sim \chi^2_{T-1}
$$

**평균과 분산**:

$$
\mathbb{E}[\hat{\sigma}^2_T] = \sigma^2_f, \quad \text{Var}[\hat{\sigma}^2_T] = \frac{2\sigma^4_f}{T-1}
$$

### 분산 근사의 상대 오차

분산 추정량의 변동계수(표준편차를 평균으로 나눈 값)는

$$
\text{CV}[\hat{\sigma}^2_T] = \frac{\sqrt{\text{Var}[\hat{\sigma}^2_T]}}{\mathbb{E}[\hat{\sigma}^2_T]} = \sqrt{\frac{2}{T-1}}
$$

| $T$ | 분산의 상대 오차 |
|-----|---------------------------|
| 10 | 47% |
| 30 | 26% |
| 50 | 20% |
| 100 | 14% |
| 500 | 6% |

**뜻하는 바:** 같은 정확도를 얻으려면 분산 근사는 평균 근사보다 표본이 훨씬 많이 필요하다.

### 분산의 신뢰구간

카이제곱 분포를 쓰면

$$
P\left( \frac{(T-1)\hat{\sigma}^2_T}{\chi^2_{T-1, \alpha/2}} \leq \sigma^2_f \leq \frac{(T-1)\hat{\sigma}^2_T}{\chi^2_{T-1, 1-\alpha/2}} \right) = 1 - \alpha
$$

---

## 5. 엔트로피와 상호 정보량의 수렴

### 예측 엔트로피 근사

분류에서 예측 엔트로피는

$$
\mathbb{H}[\mathbf{y} | \mathbf{x}, \mathcal{D}] = -\sum_{c=1}^{C} p_c \log p_c
$$

여기서 $p_c = \mathbb{E}_{q_\theta}[\text{softmax}(f(\mathbf{x}; \omega))_c]$이다.

MC 근사는 $\hat{p}_c = \frac{1}{T} \sum_{t=1}^T \text{softmax}(f(\mathbf{x}; \hat{\omega}_t))_c$를 쓴다.

### 엔트로피 근사의 바이어스

경험 분포의 엔트로피는 편향 추정량이다.

$$
\mathbb{E}[\hat{\mathbb{H}}_T] = \mathbb{H}[p] - \frac{C - 1}{2T} + O(T^{-2})
$$

여기서 $C$는 클래스 수다. 바이어스가 음수이므로 엔트로피를 과소추정한다.

**밀러-매도 보정**:

$$
\hat{\mathbb{H}}^{\text{MM}}_T = \hat{\mathbb{H}}_T + \frac{C - 1}{2T}
$$

### 상호 정보량의 수렴

상호 정보량 $\mathbb{I}[\mathbf{y}, \omega | \mathbf{x}, \mathcal{D}] = \mathbb{H}[\mathbf{y}] - \mathbb{E}[\mathbb{H}[\mathbf{y} | \omega]]$에는 다음이 든다.

1. 평균 낸 분포의 엔트로피(바이어스: $-\frac{C-1}{2T}$)
2. 개별 분포의 평균 엔트로피(비편향)

둘째 항에는 바이어스가 없어 첫째 항의 음의 바이어스를 지워 주지 못한다. 그래서 상호 정보량 근사는 아래로 치우치며, $T$가 작으면 크게 흔들릴 수 있다.

---

## 6. 경험적 수렴 살피기

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

### 유효 표본 크기

MC 표본이 실제로 서로 독립이 아니면(표본 사이에 상관이 있으면) 유효 표본 크기는 $T$보다 작을 수 있다.

$$
T_{\text{eff}} = \frac{T}{1 + 2\sum_{k=1}^{\infty} \rho_k}
$$

여기서 $\rho_k$는 시차(lag) $k$에서의 자기상관이다.

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

## 7. 실제로 표본 크기 고르기

### 작업별 지침

| 작업 | 최소 $T$ | 권장 $T$ | 비고 |
|------|-------------|-----------------|-------|
| 점 예측 | 10~20 | 30 | 평균은 빨리 수렴한다 |
| 신뢰구간 | 30~50 | 100 | 분산이 안정적이어야 한다 |
| 캘리브레이션 곡선 | 100 | 200~500 | 버킷 수에 비해 충분한 표본이 있어야 한다 |
| 능동 학습 | 20~50 | 50~100 | 순위 매기기는 잡음에 강건하다 |
| 분포 외 탐지 | 50~100 | 100~200 | 꼬리 특성이 중요하다 |
| 안전이 중요한 작업 | 200 이상 | 500 이상 | 보수적 선택 |

### 적응적 샘플링 기법

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

### 배치 효율 추정

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
        warmup_runs: 몸풀기 앞으로 스텝 횟수
        
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
    
    # 앞으로 스텝 하나의 때를 잰다
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

## 8. 이론이 주는 하한

### MC 근사의 크라메르-라오 하한

표본 $T$개로 평균 $\mu$를 추정할 때 분산에는 다음 하한이 있다.

$$
\text{Var}[\hat{\mu}_T] \geq \frac{\sigma^2_f}{T}
$$

MC 근사는 서로 독립인 표본에서 이 바운드를 달성한다(효율적이다).

### 정보 이론의 관점

MC 근사와 참 파라메터 사이의 상호 정보량은

$$
I(\hat{\mu}_T; \mu) = \frac{1}{2} \log\left(1 + \frac{T \sigma^2_\mu}{\sigma^2_f}\right)
$$

여기서 $\sigma^2_\mu$는 $\mu$의 사전 분산이다. 정보는 $T$에 따라 로그로 늘어난다.

### 수확 체감

표본 $T$에서 $T+1$로 갈 때 늘어나는 정보량은

$$
\Delta I_T = I(\hat{\mu}_{T+1}; \mu) - I(\hat{\mu}_T; \mu) \approx \frac{\sigma^2_\mu}{2T\sigma^2_f} \quad \text{for large } T
$$

이 $O(1/T)$ 감소가 표본을 더 뽑을 때 얻는 이득이 얼마나 줄어드는지(수확 체감)를 보여 준다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
ReLU 활성화와 가우시안 웨이트 사전분포를 지닌 2-레이어 뉴럴 네트워크에서, 평균장 변분 추론 · 라플라스 근사 · MC 드롭아웃이 저마다 내놓는 근사 사후분포의 형태를 견주어라.

</div>

??? success "풀이"
    웨이트가 $W_1, W_2$이고 가우시안 사전분포가 $p(W) = \mathcal{N}(0, \sigma_p^2 I)$일 때 사후분포 $p(W | D) \propto p(D | W) p(W)$는 다룰 수 없다. 세 근사 방법은 저마다 다룰 수 있는 형태를 낸다. 변분 추론이면 웨이트마다 서로 독립인 가우시안 사후분포 $q(w_{ij}) = \mathcal{N}(\mu_{ij}, \sigma_{ij}^2)$을 지니고, 라플라스 근사이면 사후분포가 MAP 추정을 중심으로 하고 공분산이 헤세 행렬의 역인 가우시안 하나이며, MC 드롭아웃이면 사후분포가 드롭아웃 마스크 분포로 암묵적으로 정의된다. 근사마다 참 사후분포 형태의 서로 다른 특성을 담는다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
MC 드롭아웃에서 얻은 불확실성 추정의 캘리브레이션을 딥 앙상블과 비교하는 실험을 설계하여라. 쓸 지표와 그림을 밝혀라.

</div>

??? success "풀이"
    지표: (1) 버킷 15개의 기대 캘리브레이션 오차(ECE), (2) 브라이어 점수, (3) 음의 로그 우도(NLL), (4) 분포 외 탐지의 AUROC. 그림: 방법마다 관측 빈도를 예측 신뢰도에 대고 그린 신뢰도 다이어그램(reliability diagram). 절차: 모든 방법을 CIFAR-10(분포 내)에서 학습하고, CIFAR-10 테스트 데이터에서 캘리브레이션을, SVHN에서 분포 외 탐지를 평가한다. 사후(post-hoc) 온도 스케일링을 베이스라인으로 쓴다. 무작위 시드 5개에 걸친 평균과 표준편차를 보고한다. 캘리브레이션이 잘 된 방법은 신뢰도 다이어그램에서 점이 대각선에 가깝고 ECE가 낮다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
베이즈 뉴럴 네트워크의 예측 분산이 인식적 불확실성과 우연적 불확실성으로 분해됨을 증명하여라. 학습 데이터 크기 $N \to \infty$일 때 두 항이 어떻게 되는지 보여라.

</div>

??? success "풀이"
    예측 분산은 전분산 법칙으로 분해된다. $\text{Var}[y | x, D] = \underbrace{\mathbb{E}_{p(\theta|D)}[\text{Var}[y | x, \theta]]}_{\text{타고난}} + \underbrace{\text{Var}_{p(\theta|D)}[\mathbb{E}[y | x, \theta]]}_{\text{앎의}}$. 우연적 항은 데이터 생성 과정의 줄일 수 없는 잡음을 담으며 $N \to \infty$이어도 그대로다. 인식적 항은 파라메터의 불확실성을 드러내며 사후분포가 참 파라메터 부근으로 수렴하므로 $O(1/N)$으로 줄어든다. 극한에서는 우연적 불확실성만 남는다. 이 분해는 데이터를 더 모아야 할 때(인식적 불확실성이 클 때)와 우연적 잡음을 받아들여야 할 때(우연적 불확실성이 클 때)를 가르는 데 중요하다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
이 절의 불확실성 추정 방법을 트레이딩 시스템의 포지션 크기 결정에 어떻게 쓸 수 있는지 논하여라. 실용적인 판단 규칙을 제시하여라.

</div>

??? success "풀이"
    판단 규칙: 포지션 크기를 인식적 불확실성에 반비례하게 잡는다. $\hat{y}$을 예측 수익률, $\sigma_e^2$을 인식적 분산이라 하자. 포지션은 $w = \frac{\hat{y}}{\lambda \sigma_e^2}$이고 $\lambda$는 위험 회피 계수다. 인식적 불확실성이 크면(낯선 시장 상황) 포지션을 줄이고, 작으면(익숙한 국면) 더 과감하게 거래한다. 여기에 더해 인식적 불확실성의 상한을 두고 그 위로는 거래하지 않게 해 기권(abstain)할 수 있다. 이 틀은 모델의 신뢰도로 스케일한 켈리 기준 방식의 포지션 크기 결정을 자연스럽게 구현한다. 워크포워드(walk-forward) 분석으로 백테스트해 $\lambda$를 보정한다. $\square$

## 정리하며

**핵심으로 챙길 것**:

1. **평균 근사**는 $O(1/\sqrt{T})$로 수렴한다 — 표본 100개면 표준오차가 $\sigma_f$의 10%다

2. **분산 근사**는 더 느리게 수렴해, 상대 오차 약 14%에 $T \approx 100$이 필요하다

3. **대부분의 응용에서는** $T = 50$~$100$이 정확도와 효율 사이의 좋은 절충점이다

4. **안전이 중요한 응용에서는** 수렴을 살피며 $T \geq 200$을 써야 한다

5. **적응적 샘플링**은 근사가 일찍 안정되면 계산을 줄일 수 있다

**살펴볼 거리**

1. Gal, Y. (2016). Uncertainty in Deep Learning. *PhD Thesis*.

2. Robert, C. P., & Casella, G. (2004). Monte Carlo Statistical Methods. *Springer*.

3. Geyer, C. J. (1992). Practical Markov Chain Monte Carlo. *Statistical Science*.
