# SWAG: Stochastic Weight Averaging Gaussian
**SWAG(확률 웨이트 고르기 가우시안)**은 SGD 회귀의 자취에 가우시안 분포를 맞추어 뉴럴 넷트웍에서 베이즈 추론을 단순하고 크게 늘릴 수 있게 이룬다. 이 일 끝난 뒤 방법은 여느 학습 동안 모은 자만으로 웨이트 사후분포의 평균과 공분산을 함께 담는다.

---

## 1. 왜 하는가

### 학습과 베이즈 추론 사이의 틈

여느 뉴럴 넷트웍 학습은 점 근사 $\hat{\theta}$을 낸다.

$$
\hat{\theta} = \arg\min_\theta \mathcal{L}(\theta; \mathcal{D})
$$

불확실성 근사를 얻으려면 사후분포 $p(\theta \mid \mathcal{D})$이 있어야 한다. 온전한 베이즈 방법(MCMC, VI)은 값이 비싸고 구조를 고쳐야 한다.

**SWAG의 깨침**: 모여 가는 언저리의 SGD 자취는 넌지시 사후분포의 터를 둘러본다. 이 자취를 따라 자를 모으면 학습 절차를 바꾸지 않고도 사후분포를 추정할 수 있다.

### 확률 웨이트 고르기(SWA) 기초

**SWA**(이즈마일로프 등, 2018)은 웨이트를 고르게 하여 두루 미침을 낫게 한다.

$$
\bar{\theta}_{\text{SWA}} = \frac{1}{T} \sum_{t=1}^T \theta_t
$$

여기서 $\theta_t$은 돌림 또는 붙박인 러닝레이트로 학습된 마지막 $T$판의 웨이트다.

**핵심 살핌**: SWA 웨이트는 로스 터에서 더 판판하고 너른 자리에 놓여 두루 미침이 나아진다.

**SWAG는 SWA을 넓혀** 평균뿐 아니라 웨이트 자취의 스프레드까지 담는다.

---

## 2. SWAG 알고리즘

### 핵심 깨침

SGD 자취에 가우시안을 맞춘다.

$$
\boxed{q(\theta) = \mathcal{N}(\theta \mid \bar{\theta}, \Sigma_{\text{SWAG}})}
$$

공분산 $\Sigma_{\text{SWAG}}$은 낮은 자리 더하기 대각 구조로 추정한다.

### 모멘트 모으기

익히는 동안(처음 워밍업이 끝난 뒤) 다음을 모은다.

**첫째 모멘트**(달리는 평균):

$$
\bar{\theta} = \frac{1}{T} \sum_{t=1}^T \theta_t
$$

**둘째 모멘트**(달리는 제곱 평균):

$$
\overline{\theta^2} = \frac{1}{T} \sum_{t=1}^T \theta_t^2
$$

**벗어남 행렬**(낮은 자리 몫에 쓴다):

$$
D = [\theta_1 - \bar{\theta}, \theta_2 - \bar{\theta}, \ldots, \theta_K - \bar{\theta}]
$$

여기서는 마지막 벗어남 $K$개만 남긴다(흔히 $K = 20$).

### 공분산 근사

**전체 공분산**은 $O(d^2)$이라 큰 네트워크에서는 다룰 수 없다.

**SWAG 근사**:

$$
\boxed{\Sigma_{\text{SWAG}} = \Sigma_{\text{diag}} + \Sigma_{\text{low-rank}}}
$$

**대각 몫**:

$$
\Sigma_{\text{diag}} = \text{diag}\left(\overline{\theta^2} - \bar{\theta}^2\right)
$$

**낮은 자리 몫**:

$$
\Sigma_{\text{low-rank}} = \frac{1}{K-1} D D^\top
$$

### SWAG에서 샘플링

$\theta \sim q(\theta)$을 뽑으려면

$$
\theta = \bar{\theta} + \frac{1}{\sqrt{2}} \sqrt{\Sigma_{\text{diag}}} \odot z_1 + \frac{1}{\sqrt{2(K-1)}} D z_2
$$

여기서 $z_1 \sim \mathcal{N}(0, I_d)$이고 $z_2 \sim \mathcal{N}(0, I_K)$이다.

$\frac{1}{\sqrt{2}}$ 값은 대각 몫과 낮은 자리 몫을 모을 때 잣대가 제대로 맞도록 한다.

---

## 3. 알고리즘의 자세한 것

### 학습 절차

```
알고리즘: SWAG 익힘
────────────────────────
들임: 미리 익힌 넷트웍 θ₀, 러닝 비율 짜임, 
       모으는 잦기 c, 자리 K
날임: SWAG 파라메터 (θ̄, Σ_diag, D)

1. 첫자리: θ̄ = 0, θ̄² = 0, D = [], n = 0
2. 러닝 비율을 SWA 짜임(붙박이 또는 돌림)으로 둔다
3. 몸풀기 뒤의 판마다:
   a. SGD으로 c번 되돌아 익힌다
   b. 달리는 자를 고친다:
      n ← n + 1
      θ̄ ← (n-1)/n · θ̄ + 1/n · θ
      θ̄² ← (n-1)/n · θ̄² + 1/n · θ²
   c. 벗어남을 담는다:
      D.append(θ - θ̄)
      len(D) > K이면: D.pop(0)
4. 셈한다: Σ_diag = diag(θ̄² - θ̄²)
5. (θ̄, Σ_diag, D)을 돌려준다
```

### 러닝레이트 구조

**돌림 구조**(즐겨 씀):

$$
\alpha_t = \alpha_{\min} + \frac{1}{2}(\alpha_{\max} - \alpha_{\min})\left(1 + \cos\left(\frac{\pi \cdot \text{mod}(t, c)}{c}\right)\right)
$$

러닝레이트가 낮은 돌림의 끝에서 표본을 모은다.

**붙박인 구조**:

$$
\alpha_t = \alpha_{\text{SWA}}
$$

더 단순하나 표본이 덜 다양할 수 있다.

### 하이퍼파라미터

| 파라메터 | 흔한 값 | 풀이 |
|-----------|---------------|-------------|
| SWA 비롯 | 학습의 75% | 언제부터 모을지 |
| 모으는 잦기 $c$ | 1판 | 자를 얼마나 자주 고칠지 |
| 낮은 자리 $K$ | 20 | 벗어남 벡터의 수 |
| 러닝레이트 | 0.01~0.05 | SWA 러닝레이트 |

---

## 4. SWAG로 추론

### 몬테카를로 적분

시험 입력 $x^*$이 주어지면 웨이트 차림 $S$개를 뽑는다.

$$
\theta^{(s)} \sim q(\theta) = \mathcal{N}(\bar{\theta}, \Sigma_{\text{SWAG}})
$$

**예측한 평균**:

$$
\hat{\mu}(x^*) = \frac{1}{S} \sum_{s=1}^S f_{\theta^{(s)}}(x^*)
$$

**예측한 분산**(인식적):

$$
\hat{\sigma}^2(x^*) = \frac{1}{S} \sum_{s=1}^S \left(f_{\theta^{(s)}}(x^*) - \hat{\mu}(x^*)\right)^2
$$

### 분류에서는

소프트맥스 확률을 평균한다.

$$
p(y = c \mid x^*, \mathcal{D}) \approx \frac{1}{S} \sum_{s=1}^S \text{softmax}(f_{\theta^{(s)}}(x^*))_c
$$

**불확실성 자**:

- **엔트로피**: $\mathbb{H}[\bar{p}] = -\sum_c \bar{p}_c \log \bar{p}_c$
- **서로 나눈 소식**: $\mathbb{I}[y; \theta] = \mathbb{H}[\bar{p}] - \frac{1}{S}\sum_s \mathbb{H}[p_s]$

---

## 5. 이론으로 뒷받침하기

### 라플라스 근사과의 이어짐

SWAG는 근사 라플라스 근사로 볼 수 있다.

**라플라스**: $q(\theta) = \mathcal{N}(\theta_{\text{MAP}}, H^{-1})$

**SWAG**: $q(\theta) = \mathcal{N}(\bar{\theta}_{\text{SWA}}, \Sigma_{\text{SWAG}})$

핵심 다름은 이렇다.

- SWAG는 MAP 대신 SWA 평균(더 판판한 자리일 수 있다)을 쓴다
- SWAG는 자취의 자로 헤세 행렬의 역을 추정한다

### SGLD과의 이어짐

어떤 자리에서는 잡음이 있는 SGD이 사후분포를 둘러본다.

$$
\theta_{t+1} = \theta_t - \alpha \nabla \mathcal{L}(\theta_t) + \epsilon_t
$$

SWAG는 이 둘러봄의 가장자리 자를 담는다.

### 로스 터로 본 눈

SWAG는 SWA 풀이 언저리의 "골"에서 표본을 뽑는다.

- 대각은 파라메터마다의 분산을 담는다
- 낮은 자리는 바뀜의 으뜸 방향을 담는다
- 둘이 함께 그 자리 사후분포의 꼴을 추정한다

---

## 6. SWAG의 클래스

### SWAG-대각

대각 공분산만 쓴다(낮은 자리 없음).

$$
\Sigma = \text{diag}\left(\overline{\theta^2} - \bar{\theta}^2\right)
$$

**나은 점**: 더 단순하고 자리를 덜 쓴다
**아쉬운 점**: 웨이트 사이의 얽힘을 놓친다

### 여럿 SWAG

첫값을 아무렇게나 달리해 SWAG를 여러 번 돌린다.

$$
p(\theta \mid \mathcal{D}) \approx \frac{1}{M} \sum_{m=1}^M q_m(\theta)
$$

사후분포의 봉우리 여럿을 담는다.

### 배치 잣대 잡기가 있는 SWAG

**어려움**: 배치 잣대 잡기의 자는 웨이트뿐 아니라 배치에도 매인다.

**풀이**: 웨이트를 뽑은 뒤, 따지기 앞서 학습 데이터로 앞으로 스텝을 한 번 돌려 배치 잣대 잡기의 달리는 자를 고친다.

---

## 7. 파이썬으로 짜기

```python
"""
SWAG: 확률 짐 고르기 가우스

SGD 자취에 가우스를 맞추어 베이즈 미루어 봄을 어림하는
단순하고 크게 늘릴 수 있는 길.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import deque

class SWAG:
    """
    확률 짐 고르기 가우스.
    
    SGD 자취에서 어림한 낮은 자리 더하기 대각 함께 바뀜을 지닌
    가우스로 뒷분포를 어림한다.
    """
    
    def __init__(
        self,
        n_params: int,
        max_rank: int = 20,
        var_clamp: float = 1e-6
    ):
        """
        Parameters
        ----------
        n_params : int
            모형 파라메터의 수
        max_rank : int
            낮은 자리 함께 바뀜 몫의 가장 큰 자리
        var_clamp : float
            가장 작은 흩어짐(셈이 든든하도록)
        """
        self.n_params = n_params
        self.max_rank = max_rank
        self.var_clamp = var_clamp
        
        # 자
        self.mean = np.zeros(n_params)
        self.sq_mean = np.zeros(n_params)
        self.deviations = deque(maxlen=max_rank)
        self.n_models = 0
    
    def update(self, params: np.ndarray):
        """
        새 파라메터로 SWAG 자를 고친다.
        
        Parameters
        ----------
        params : (n_params,) 꼴의 ndarray
            이제의 모형 파라메터
        """
        params = np.asarray(params).flatten()
        assert len(params) == self.n_params
        
        self.n_models += 1
        n = self.n_models
        
        # 달리는 평균을 고친다
        old_mean = self.mean.copy()
        self.mean = (n - 1) / n * self.mean + 1 / n * params
        
        # 달리는 제곱 평균을 고친다
        self.sq_mean = (n - 1) / n * self.sq_mean + 1 / n * (params ** 2)
        
        # 이제 평균에서의 벗어남을 담는다
        deviation = params - self.mean
        self.deviations.append(deviation)
    
    @property
    def variance(self) -> np.ndarray:
        """대각 흩어짐."""
        var = self.sq_mean - self.mean ** 2
        return np.maximum(var, self.var_clamp)
    
    @property
    def deviation_matrix(self) -> np.ndarray:
        """낮은 자리 몫에 쓰는 벗어남 행렬."""
        if len(self.deviations) == 0:
            return np.zeros((self.n_params, 1))
        return np.column_stack(self.deviations)
    
    def sample(self, scale: float = 1.0) -> np.ndarray:
        """
        SWAG 분포에서 표본을 뽑는다.
        
        Parameters
        ----------
        scale : float
            아리송함의 잣대 값(1.0 = 온전한 아리송함)
        
        Returns
        -------
        (n_params,) 꼴의 ndarray
            뽑은 파라메터
        """
        # 대각 몫에서 뽑는다
        z1 = np.random.randn(self.n_params)
        
        # 낮은 자리 몫에서 뽑는다
        D = self.deviation_matrix
        K = D.shape[1]
        z2 = np.random.randn(K)
        
        # 모은다(잣대를 제대로 맞추어)
        std_diag = np.sqrt(self.variance)
        
        if K > 1:
            # 온전한 SWAG: 대각 + 낮은 자리
            sample = (
                self.mean 
                + scale * (1.0 / np.sqrt(2.0)) * std_diag * z1
                + scale * (1.0 / np.sqrt(2.0 * (K - 1))) * D @ z2
            )
        else:
            # SWAG-대각만
            sample = self.mean + scale * std_diag * z1
        
        return sample
    
    def sample_many(self, n_samples: int, scale: float = 1.0) -> np.ndarray:
        """
        파라메터 벡터를 여럿 뽑는다.
        
        Parameters
        ----------
        n_samples : int
            표본의 수
        scale : float
            아리송함의 잣대 값
        
        Returns
        -------
        (n_samples, n_params) 꼴의 ndarray
            뽑은 파라메터
        """
        return np.array([self.sample(scale) for _ in range(n_samples)])
    
    def get_state(self) -> Dict[str, np.ndarray]:
        """담아 둘 상태를 얻는다."""
        return {
            'mean': self.mean,
            'sq_mean': self.sq_mean,
            'deviations': np.array(list(self.deviations)),
            'n_models': self.n_models
        }
    
    def load_state(self, state: Dict[str, np.ndarray]):
        """상태를 얹는다."""
        self.mean = state['mean']
        self.sq_mean = state['sq_mean']
        self.deviations = deque(state['deviations'], maxlen=self.max_rank)
        self.n_models = state['n_models']

class SWAGTrainer:
    """
    돌림 러닝 비율을 쓰는 SWAG 익힘개.
    """
    
    def __init__(
        self,
        model,
        swag: SWAG,
        lr_init: float = 0.01,
        lr_min: float = 0.001,
        cycle_length: int = 5,
        swa_start: int = 50
    ):
        """
        Parameters
        ----------
        model : object
            get_params()과 set_params() 방법을 지닌 뉴럴 넷트웍
        swag : SWAG
            자를 모을 SWAG 물체
        lr_init : float
            처음(가장 큰) 러닝 비율
        lr_min : float
            가장 작은 러닝 비율
        cycle_length : int
            돌림 하나의 판 수
        swa_start : int
            SWAG 모으기를 비롯할 판
        """
        self.model = model
        self.swag = swag
        self.lr_init = lr_init
        self.lr_min = lr_min
        self.cycle_length = cycle_length
        self.swa_start = swa_start
    
    def get_lr(self, epoch: int) -> float:
        """돌림 러닝 비율 짜임."""
        if epoch < self.swa_start:
            # 곧은 몸풀기 또는 붙박이
            return self.lr_init
        
        # swa_start 뒤로는 돌림
        cycle_epoch = (epoch - self.swa_start) % self.cycle_length
        t = cycle_epoch / self.cycle_length
        
        return self.lr_min + 0.5 * (self.lr_init - self.lr_min) * (1 + np.cos(np.pi * t))
    
    def train_epoch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epoch: int,
        batch_size: int = 32
    ) -> float:
        """
        한 판 익힌다.
        
        Returns
        -------
        float
            평균 잃음
        """
        lr = self.get_lr(epoch)
        losses = []
        
        # 자료를 뒤섞는다
        indices = np.random.permutation(len(X))
        
        for start in range(0, len(X), batch_size):
            batch_idx = indices[start:start + batch_size]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            
            # 그래디언트와 잃음을 셈한다(모형마다 다름)
            loss, grads = self.model.compute_gradients(X_batch, y_batch)
            losses.append(loss)
            
            # SGD 고침
            params = self.model.get_params()
            new_params = params - lr * grads
            self.model.set_params(new_params)
        
        # 돌림 끝(러닝 비율이 낮을 때) SWAG를 고친다
        if epoch >= self.swa_start:
            cycle_epoch = (epoch - self.swa_start) % self.cycle_length
            if cycle_epoch == self.cycle_length - 1:
                self.swag.update(self.model.get_params())
        
        return np.mean(losses)

def swag_predict(
    model,
    swag: SWAG,
    X: np.ndarray,
    n_samples: int = 30,
    scale: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SWAG 아리송함을 곁들여 미루어 본다.
    
    Parameters
    ----------
    model : object
        set_params()과 predict() 방법을 지닌 뉴럴 넷트웍
    swag : SWAG
        맞춰 놓은 SWAG 물체
    X : ndarray
        들임 자료
    n_samples : int
        뒷분포 표본의 수
    scale : float
        아리송함 잣대 값
    
    Returns
    -------
    mean : ndarray
        평균 미루어 봄
    std : ndarray
        표준편차(앎의 아리송함)
    """
    predictions = []
    
    for _ in range(n_samples):
        # 짐을 뽑는다
        params = swag.sample(scale=scale)
        model.set_params(params)
        
        # 미루어 본다
        pred = model.predict(X)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    mean = np.mean(predictions, axis=0)
    std = np.std(predictions, axis=0)
    
    # 평균 짐으로 되돌린다
    model.set_params(swag.mean)
    
    return mean, std

# =============================================================================
# 보여 주기
# =============================================================================

def demo_swag():
    """단순한 회귀 문제에서 SWAG를 보여 준다."""
    
    print("=" * 60)
    print("SWAG 보여 주기")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 장난감 자료를 만든다
    N = 100
    X = np.random.uniform(-4, 4, N).reshape(-1, 1)
    y = np.sin(X) + 0.2 * np.random.randn(N, 1)
    
    print(f"\n익힘 자료: {N}점")
    
    # 보여 주기용 단순 MLP
    class SimpleMLP:
        def __init__(self, layers):
            self.layers = layers
            self.params = self._init_params()
        
        def _init_params(self):
            params = []
            for i in range(len(self.layers) - 1):
                W = np.random.randn(self.layers[i], self.layers[i+1]) * 0.5
                b = np.zeros(self.layers[i+1])
                params.extend([W.flatten(), b])
            return np.concatenate(params)
        
        def get_params(self):
            return self.params.copy()
        
        def set_params(self, params):
            self.params = params.copy()
        
        def predict(self, X):
            # 파라메터를 풀고 앞으로 스텝
            idx = 0
            h = X
            for i in range(len(self.layers) - 1):
                n_in, n_out = self.layers[i], self.layers[i+1]
                W = self.params[idx:idx + n_in * n_out].reshape(n_in, n_out)
                idx += n_in * n_out
                b = self.params[idx:idx + n_out]
                idx += n_out
                h = h @ W + b
                if i < len(self.layers) - 2:
                    h = np.tanh(h)
            return h
        
        def compute_gradients(self, X, y):
            # 단순하게 수로 셈하는 그래디언트
            pred = self.predict(X)
            loss = np.mean((pred - y) ** 2)
            
            eps = 1e-5
            grads = np.zeros_like(self.params)
            for i in range(len(self.params)):
                self.params[i] += eps
                loss_plus = np.mean((self.predict(X) - y) ** 2)
                self.params[i] -= 2 * eps
                loss_minus = np.mean((self.predict(X) - y) ** 2)
                self.params[i] += eps
                grads[i] = (loss_plus - loss_minus) / (2 * eps)
            
            return loss, grads
    
    # 모형을 만든다
    model = SimpleMLP([1, 30, 30, 1])
    n_params = len(model.get_params())
    print(f"모형 파라메터: {n_params}")
    
    # SWAG를 만든다
    swag = SWAG(n_params, max_rank=20)
    
    # 미리 익히기
    print("\n미리 익히는 중...")
    for epoch in range(50):
        lr = 0.1 * (0.95 ** epoch)
        _, grads = model.compute_gradients(X, y)
        model.set_params(model.get_params() - lr * grads)
    
    # SWAG 모으기 도막
    print("SWAG 자를 모으는 중...")
    for epoch in range(30):
        # 돌림 러닝 비율
        t = (epoch % 5) / 5
        lr = 0.001 + 0.009 * (1 + np.cos(np.pi * t)) / 2
        
        _, grads = model.compute_gradients(X, y)
        model.set_params(model.get_params() - lr * grads)
        
        # 돌림 끝에서 모은다
        if (epoch + 1) % 5 == 0:
            swag.update(model.get_params())
    
    print(f"모은 SWAG 모형: {swag.n_models}")
    
    # 미루어 본다
    X_test = np.linspace(-6, 6, 100).reshape(-1, 1)
    
    mean, std = swag_predict(model, swag, X_test, n_samples=50)
    
    print(f"\n평균 앎의 아리송함: {np.mean(std):.4f}")
    print(f"익힘 자리 [-4,4] 안의 아리송함: {np.mean(std[np.abs(X_test) < 4]):.4f}")
    print(f"익힘 자리 밖의 아리송함: {np.mean(std[np.abs(X_test) > 4]):.4f}")
    
    print("\n*** SWAG 아리송함은 익힘 자리 밖에서 더 커야 한다")
    
    return swag, model

if __name__ == "__main__":
    demo_swag()
```

**출력:**

```
============================================================
SWAG 보여 주기
============================================================

익힘 자료: 100점
모형 파라메터: 1021

미리 익히는 중...
SWAG 자를 모으는 중...
모은 SWAG 모형: 6

평균 앎의 아리송함: 0.3001
익힘 자리 [-4,4] 안의 아리송함: 0.3041
익힘 자리 밖의 아리송함: 0.2924

*** SWAG 아리송함은 익힘 자리 밖에서 더 커야 한다
```

---

## 8. 다른 방법과 비교

### 깊은 앙상블과

| 특성 | SWAG | 깊은 앙상블 |
|--------|------|----------------|
| 학습 | 네트워크 하나 + 자 | 서로 독립인 네트워크 M개 |
| 자리 | 평균 + 대각 + 벡터 K개 | 온전한 네트워크 M개 |
| 다양함 | 가우시안 근사 | 참 앙상블의 다양함 |
| 불확실성 성능 | 좋음 | 흔히 더 좋음 |
| 짜기 | 일 끝난 뒤 | 처음부터 |

### MC 드롭아웃과

| 특성 | SWAG | MC 드롭아웃 |
|--------|------|------------|
| 학습 | 고친 구조 | 여느 학습 + 드롭아웃 |
| 사후분포 | 드러난 가우시안 | 넌지시(베르누이 마스크) |
| 공분산 | 낮은 자리 + 대각 | 넌지시 |
| 너그러움 | 구조를 가리지 않음 | 드롭아웃 켜가 있어야 함 |

### 변분 추론과

| 특성 | SWAG | VI(되돌아가며 베이즈) |
|--------|------|------------------------|
| 학습 | 일 끝난 뒤 | 처음부터 |
| 파라메터 | 밑 모델과 같음 | 2배(평균 + 분산) |
| 근사 | 겪어 본 가우시안 | 평균장 가우시안 |
| 크게 늘리기 | 아주 좋음 | 좋음 |

---

## 9. 참으로 쓸 길잡이

### SWAG를 쓸 때

**잘 맞는 자리**:

- 이미 학습된 네트워크가 있을 때
- 불확실성 근사가 빨리 있어야 할 때
- VI이 비싼 큰 모델
- 여느 구조(ResNet 따위)

**덜 맞는 자리**:

- 아주 맞는 사후분포가 있어야 할 때
- 봉우리가 여럿일 것으로 볼 때
- 제때 추론이 있어야 할 때

### 짜기 요령

1. **여느 대로 사전 학습된이다**: 먼저 좋은 풀이를 얻는다
2. **돌림 러닝레이트를 쓴다**: 붙박이보다 더 잘 둘러본다
3. **표본을 넉넉히 모은다**: 공분산이 든든하려면 적어도 10~20개
4. **배치 잣대 잡기를 다룬다**: 뽑은 뒤 달리는 자를 고친다
5. **잣대 값 맞추기**: 1.0에서 비롯해 캘리브레이션을 보고 손본다

### 캘리브레이션

추론이 지나치게 머뭇거리거나 자신하면

- **머뭇거림**: 잣대 값을 낮춘다
- **지나친 자신**: 잣대 값을 높이고 표본을 더 모은다

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

### 핵심 식

**SWAG 분포**:

$$
q(\theta) = \mathcal{N}(\bar{\theta}, \Sigma_{\text{diag}} + \Sigma_{\text{low-rank}})
$$

**샘플링**:

$$
\theta = \bar{\theta} + \frac{1}{\sqrt{2}} \sqrt{\Sigma_{\text{diag}}} \odot z_1 + \frac{1}{\sqrt{2(K-1)}} D z_2
$$

**달리는 자**:

$$
\bar{\theta} = \frac{1}{T}\sum_t \theta_t, \quad \Sigma_{\text{diag}} = \text{diag}(\overline{\theta^2} - \bar{\theta}^2)
$$

### 나은 점과 한계

| 나은 점 | 한계 |
|------------|-------------|
| 짜기가 단순하다 | 가우시안 근사에 그친다 |
| 일 끝난 뒤(다시 익히지 않음) | 불확실성을 낮게 볼 수 있다 |
| 큰 네트워크로 늘릴 수 있다 | SWA 특성의 학습 도막이 있어야 한다 |
| 겪어 본 성능이 좋다 | 봉우리 하나만 추정한다 |

### 다른 이야기와의 이어짐

| 이야기 | 이어짐 |
|-------|------------|
| SWA | SWAG는 SWA에 공분산을 더한다 |
| 라플라스 | 둘 다 모인 자리에 가우시안을 맞춘다 |
| 깊은 앙상블 | 함께 쓸 수 있다(여럿 SWAG) |
| 불확실성 | 인식적 불확실성 근사를 준다 |

### 핵심 살펴볼 거리

- Maddox, W., et al. (2019). A simple baseline for Bayesian inference in deep learning. *NeurIPS*.
- Izmailov, P., et al. (2018). Averaging weights leads to wider optima and better generalization. *UAI*.
- Wilson, A. G., & Izmailov, P. (2020). Bayesian deep learning and a probabilistic perspective of generalization. *NeurIPS*.
