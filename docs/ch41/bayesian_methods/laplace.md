# 신경 그물의 라플라스 어림
**라플라스 어림**은 MAP(가장 큰 뒷분포) 어림을 가운데로 삼는 가우스 분포를 맞추어 베이즈 미루어 봄을 단순하게 일 끝난 뒤 이룬다. 이 오래된 재주를 신경 그물에 쓰면 그물을 다시 익히지 않고도 셈으로 다룰 수 있는 아리송함 어림을 얻는다.

---

## 1. 왜 하는가: 일 끝난 뒤의 아리송함

### 일 끝난 뒤 방법이 끌리는 까닭

여느 방법(SGD, Adam)으로 신경 그물을 익혔다. 이제 아리송함 어림이 있어야 한다. 고를 수 있는 길은 이렇다.

1. **베이즈 방법으로 다시 익히기**(VI, MCMC) — 비싸고 새로 익혀야 한다
2. **시험할 때 드롭아웃 쓰기**(MC 드롭아웃) — 싸지만 얼개에 드롭아웃이 있어야 한다
3. **모둠 익히기** — 그물 $M$개를 익혀야 한다
4. **라플라스 어림** — 익힌 그물을 그대로 쓰고 그 언저리에 가우스를 맞춘다

라플라스 어림은 **일 끝난 뒤** 방법이다. 이미 익힌 그물을 받아 아리송함 재기를 덧댄다.

### 고갱이 깨침

뒷분포를 봉우리(MAP 어림)를 가운데로 삼는 가우스로 어림한다.

$$
\boxed{p(\theta \mid \mathcal{D}) \approx q(\theta) = \mathcal{N}(\theta \mid \hat{\theta}_{\text{MAP}}, \Sigma)}
$$

여기서

- $\hat{\theta}_{\text{MAP}}$은 익힌 그물의 짐
- $\Sigma = H^{-1}$은 MAP에서의 잃음의 헤세 행렬의 거꿀

---

## 2. 수학 밑바탕

### 이끌어 내기

잣대를 맞추지 않은 로그 뒷분포에서 비롯한다.

$$
\log p(\theta \mid \mathcal{D}) = \log p(\mathcal{D} \mid \theta) + \log p(\theta) - \log p(\mathcal{D})
$$

MAP 어림 $\hat{\theta}$ 언저리에서 테일러로 펼치면

$$
\log p(\theta \mid \mathcal{D}) \approx \log p(\hat{\theta} \mid \mathcal{D}) + \underbrace{\nabla \log p(\hat{\theta} \mid \mathcal{D})}_{= 0 \text{ at MAP}} (\theta - \hat{\theta}) - \frac{1}{2}(\theta - \hat{\theta})^\top H (\theta - \hat{\theta})
$$

여기서

$$
H = -\nabla^2_\theta \log p(\theta \mid \mathcal{D})\big|_{\hat{\theta}}
$$

은 음수 로그 뒷분포의 헤세 행렬이다(그 자리 가장 낮은 곳에서 양으로 굳는다).

지수를 취하면

$$
p(\theta \mid \mathcal{D}) \propto \exp\left(-\frac{1}{2}(\theta - \hat{\theta})^\top H (\theta - \hat{\theta})\right)
$$

이는 함께 바뀜이 $\Sigma = H^{-1}$인 가우스다.

### 헤세 행렬

음수 로그 그럴듯함 잃음 $\mathcal{L}(\theta) = -\log p(\mathcal{D} \mid \theta)$과 가우스 앞선 분포를 지닌 신경 그물에서

$$
H = \nabla^2 \mathcal{L}(\theta)\big|_{\hat{\theta}} + \frac{1}{\sigma_0^2} I
$$

**몫**:

- **그럴듯함 헤세 행렬**: 잃음 낯의 굽음
- **앞선 분포 몫**: 정칙화(짐 줄이기 $\lambda = 1/\sigma_0^2$에 맞물린다)

### 넓힌 가우스-뉴턴(GGN)

정확한 헤세 행렬은 비싸고 양으로 굳지 않을 수 있다. 그래서 **GGN 어림**을 흔히 쓴다.

$$
H_{\text{GGN}} = J^\top \nabla^2 \mathcal{L}_{\text{out}} J + \frac{1}{\sigma_0^2} I
$$

여기서

- $J = \nabla_\theta f_\theta(X)$은 매개변수에 대한 그물 날임의 야코비 행렬
- $\nabla^2 \mathcal{L}_{\text{out}}$은 그물 날임에 대한 잃음의 헤세 행렬

**MSE 잃음에서는**: GGN이 피셔 소식 행렬과 같다.

**나은 점**: 언제나 양으로 반쯤 굳고, 흔히 어림이 좋다.

---

## 3. 헤세 행렬 어림

### 크게 늘리기의 어려움

매개변수가 $d$개인 그물에서

- **온전한 헤세 행렬**: 자리 $O(d^2)$, 거꿀 셈 $O(d^3)$
- 요즘 그물: $d = 10^6$에서 $10^9$

**풀이**: 얼개를 지닌 어림을 쓴다.

### 대각 어림

매개변수끼리 남남이라고 본다.

$$
\Sigma = \text{diag}(\sigma_1^2, \ldots, \sigma_d^2) \quad \text{where} \quad \sigma_i^2 = 1/H_{ii}
$$

**셈**: 헤세 행렬의 대각만 있으면 되므로 잘 어림할 수 있다.

**한계**: 매개변수 사이의 얽힘을 놓친다.

### 크로네커로 쪼갠 굽음 어림(KFAC)

짐 행렬이 $W^{(l)} \in \mathbb{R}^{n_{l-1} \times n_l}$인 켜 $l$에서

$$
\boxed{H^{(l)} \approx A^{(l)} \otimes G^{(l)}}
$$

여기서

- $A^{(l)} = \frac{1}{N}\sum_{n=1}^N a_n^{(l-1)} (a_n^{(l-1)})^\top$ — 들임 살림의 함께 바뀜
- $G^{(l)} = \frac{1}{N}\sum_{n=1}^N g_n^{(l)} (g_n^{(l)})^\top$ — 날임 기울기의 함께 바뀜

**고갱이 결**: 크로네커 곱의 거꿀은 잘 셈된다.

$$
(A \otimes G)^{-1} = A^{-1} \otimes G^{-1}
$$

**번거로움**: 켜마다 $O((n_l \cdot n_{l-1})^3)$ 대신 $O(n_l^3 + n_{l-1}^3)$이다.

### 낮은 자리 어림

헤세 행렬의 으뜸 고유 몫만 남긴다.

$$
H \approx V \Lambda V^\top
$$

여기서 $V \in \mathbb{R}^{d \times r}$은 으뜸 고유 벡터 $r$개를 담는다.

**거꿀**:

$$
\Sigma = V \Lambda^{-1} V^\top + \frac{1}{\lambda_{\min}} (I - VV^\top)
$$

### 어림끼리 견주기

| 어림 | 자리 | 거꿀 | 얽힘 담기 |
|---------------|---------|-----------|----------------------|
| 온전 | $O(d^2)$ | $O(d^3)$ | 그렇다(모두) |
| 대각 | $O(d)$ | $O(d)$ | 아니다 |
| KFAC | $O(\sum_l n_l^2)$ | $O(\sum_l n_l^3)$ | 켜 안에서 |
| 낮은 자리 | $O(dr)$ | $O(dr^2)$ | 으뜸 방향 $r$개 |

---

## 4. 마지막 켜 라플라스

### 왜 하는가

참으로 쓸 만한 단순화가 있다. 라플라스를 마지막 켜에만 쓰는 것이다.

$$
p(\theta_L \mid \mathcal{D}, \theta_{1:L-1}) \approx \mathcal{N}(\theta_L \mid \hat{\theta}_L, \Sigma_L)
$$

**까닭**:

- 앞 켜는 결(드러냄)을 배운다
- 마지막 켜가 끝의 가름/되돌이를 한다
- 미루어 봄의 아리송함은 거의 마지막 켜에서 온다

### 짜기

1. 그물을 여느 대로 익힌다
2. 마지막 켜만 빼고 모두 얼린다
3. 마지막 켜 짐에 대해서만 헤세 행렬을 셈한다
4. 훨씬 작다: 매개변수가 $d_L = n_{L-1} \times n_L + n_L$개

### 미루어 보는 분포

$\phi(x)$을 결 뽑개라 할 때 마지막 켜 $f_\theta(x) = W^\top \phi(x) + b$에서

**곧게 펴기**:

$$
f_\theta(x) \approx f_{\hat{\theta}}(x) + \phi(x)^\top (\theta - \hat{\theta})
$$

**미루어 본 평균**:

$$
\mathbb{E}[f(x^*)] = f_{\hat{\theta}}(x^*)
$$

**미루어 본 흩어짐**:

$$
\text{Var}[f(x^*)] = \phi(x^*)^\top \Sigma_L \phi(x^*)
$$

---

## 5. 미루어 보기

### 되돌이에서는

가우스 그럴듯함 $p(y \mid f, \sigma^2)$에서

**미루어 보는 분포**:

$$
p(y^* \mid x^*, \mathcal{D}) = \mathcal{N}(y^* \mid \mu(x^*), \sigma^2(x^*))
$$

**평균**:

$$
\mu(x^*) = f_{\hat{\theta}}(x^*)
$$

**흩어짐**(곧게 편 뒤):

$$
\sigma^2(x^*) = \underbrace{J(x^*)^\top \Sigma \, J(x^*)}_{\text{앎의}} + \underbrace{\sigma^2_{\text{noise}}}_{\text{타고난}}
$$

여기서 $J(x^*) = \nabla_\theta f_\theta(x^*)|_{\hat{\theta}}$은 야코비 행렬이다.

### 가름에서는

**길 1: 프로빗 어림**

시그모이드 날임을 쓰는 둘 가름에서

$$
p(y=1 \mid x^*) \approx \sigma\left(\frac{\mu(x^*)}{\sqrt{1 + \pi \sigma^2(x^*)/8}}\right)
$$

**길 2: 몬테카를로 표본 뽑기**

$\theta^{(s)} \sim \mathcal{N}(\hat{\theta}, \Sigma)$을 뽑아 평균한다.

$$
p(y = c \mid x^*) \approx \frac{1}{S} \sum_{s=1}^S \text{softmax}(f_{\theta^{(s)}}(x^*))_c
$$

---

## 6. 파이썬으로 짜기

```python
"""
신경 그물의 라플라스 어림

MAP 어림 자리에 가우스를 맞추어 일 끝난 뒤에
아리송함을 재기.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Callable
from scipy.linalg import cho_factor, cho_solve

class LaplaceMLP:
    """
    아리송함을 위해 라플라스 어림을 쓰는 MLP.
    """
    
    def __init__(
        self,
        layer_sizes: list,
        activation: str = 'tanh',
        prior_precision: float = 1.0
    ):
        """
        Parameters
        ----------
        layer_sizes : list
            켜마다의 크기 [들임, 숨은 켜..., 날임]
        activation : str
            살림 함수('tanh' 또는 'relu')
        prior_precision : float
            앞선 분포의 촘촘함(1/sigma_0^2), 짐 줄이기와 같음
        """
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.prior_precision = prior_precision
        
        # 짐의 첫자리를 잡는다
        self.params = self._init_params()
        
        # 라플라스 값(맞춘 뒤에 셈한다)
        self.hessian = None
        self.covariance = None
    
    def _init_params(self) -> np.ndarray:
        """자비에 첫자리 잡기로 짐을 마련한다."""
        params = []
        for i in range(len(self.layer_sizes) - 1):
            n_in, n_out = self.layer_sizes[i], self.layer_sizes[i+1]
            W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / (n_in + n_out))
            b = np.zeros(n_out)
            params.extend([W.flatten(), b])
        return np.concatenate(params)
    
    def _unpack_params(self, params: np.ndarray):
        """편 매개변수 벡터를 짐 행렬로 푼다."""
        weights = []
        idx = 0
        for i in range(len(self.layer_sizes) - 1):
            n_in, n_out = self.layer_sizes[i], self.layer_sizes[i+1]
            W = params[idx:idx + n_in * n_out].reshape(n_in, n_out)
            idx += n_in * n_out
            b = params[idx:idx + n_out]
            idx += n_out
            weights.append((W, b))
        return weights
    
    def _activate(self, x: np.ndarray) -> np.ndarray:
        """살림 함수를 건다."""
        if self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        else:
            raise ValueError(f"모르는 살림 함수: {self.activation}")
    
    def _activate_grad(self, x: np.ndarray) -> np.ndarray:
        """살림 함수의 미분."""
        if self.activation == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation == 'relu':
            return (x > 0).astype(float)
        else:
            raise ValueError(f"모르는 살림 함수: {self.activation}")
    
    def forward(self, X: np.ndarray, params: Optional[np.ndarray] = None) -> np.ndarray:
        """
        앞으로 걸음.
        
        Parameters
        ----------
        X : (N, input_dim) 꼴의 ndarray
            들임 자료
        params : ndarray, 골라 씀
            쓸 매개변수(기본값: self.params)
        
        Returns
        -------
        (N, output_dim) 꼴의 ndarray
            그물 날임
        """
        if params is None:
            params = self.params
        
        weights = self._unpack_params(params)
        h = X
        
        for i, (W, b) in enumerate(weights):
            h = h @ W + b
            if i < len(weights) - 1:
                h = self._activate(h)
        
        return h
    
    def loss(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: Optional[np.ndarray] = None
    ) -> float:
        """
        음수 로그 뒷분포를 셈한다(붙박이 항은 뺀다).
        
        잃음 = MSE + (prior_precision/2) * ||params||^2
        """
        if params is None:
            params = self.params
        
        pred = self.forward(X, params)
        mse = np.mean((pred - y) ** 2)
        reg = 0.5 * self.prior_precision * np.sum(params ** 2)
        
        return mse + reg / len(X)
    
    def _compute_jacobian(self, X: np.ndarray) -> np.ndarray:
        """
        매개변수에 대한 그물 날임의 야코비 행렬을 셈한다.
        
        Returns
        -------
        (N * output_dim, n_params) 꼴의 ndarray
        """
        N = len(X)
        n_params = len(self.params)
        output_dim = self.layer_sizes[-1]
        
        # 단순하게 수로 셈하는 야코비 행렬
        eps = 1e-5
        J = np.zeros((N * output_dim, n_params))
        
        f0 = self.forward(X).flatten()
        
        for i in range(n_params):
            params_plus = self.params.copy()
            params_plus[i] += eps
            f_plus = self.forward(X, params_plus).flatten()
            J[:, i] = (f_plus - f0) / eps
        
        return J
    
    def compute_hessian_diagonal(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        헤세 행렬의 대각을 셈한다.
        
        단순하게 수로 미분해 셈한다.
        """
        n_params = len(self.params)
        diag_H = np.zeros(n_params)
        
        eps = 1e-4
        
        for i in range(n_params):
            params_plus = self.params.copy()
            params_plus[i] += eps
            
            params_minus = self.params.copy()
            params_minus[i] -= eps
            
            loss_plus = self.loss(X, y, params_plus)
            loss_minus = self.loss(X, y, params_minus)
            loss_center = self.loss(X, y)
            
            diag_H[i] = (loss_plus - 2 * loss_center + loss_minus) / (eps ** 2)
        
        # 앞선 분포 몫을 더한다
        diag_H += self.prior_precision / len(X)
        
        return np.maximum(diag_H, 1e-6)  # 양이 되게 한다
    
    def compute_ggn(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        헤세 행렬의 넓힌 가우스-뉴턴 어림을 셈한다.
        
        H_GGN = J^T J / (N * sigma^2) + prior_precision * I
        """
        N = len(X)
        J = self._compute_jacobian(X)
        
        # 남은 값으로 잡음 흩어짐을 어림한다
        residuals = self.forward(X) - y
        sigma2 = np.var(residuals) + 1e-6
        
        # GGN = J^T J / (N * sigma^2)
        H = J.T @ J / (N * sigma2)
        
        # 앞선 분포를 더한다
        H += self.prior_precision * np.eye(len(self.params))
        
        return H
    
    def fit_laplace(
        self,
        X: np.ndarray,
        y: np.ndarray,
        hessian_type: str = 'diagonal'
    ):
        """
        익힘이 끝난 뒤 라플라스 어림을 맞춘다.
        
        Parameters
        ----------
        X : ndarray
            익힘 들임
        y : ndarray
            익힘 과녁
        hessian_type : str
            'diagonal' 또는 'full'(GGN)
        """
        if hessian_type == 'diagonal':
            diag_H = self.compute_hessian_diagonal(X, y)
            self.hessian = np.diag(diag_H)
            self.covariance = np.diag(1.0 / diag_H)
        
        elif hessian_type == 'full':
            self.hessian = self.compute_ggn(X, y)
            # 다독인 뒤 거꿀을 셈한다
            self.hessian += 1e-4 * np.eye(len(self.params))
            self.covariance = np.linalg.inv(self.hessian)
        
        else:
            raise ValueError(f"모르는 hessian_type: {hessian_type}")
    
    def predict(
        self,
        X: np.ndarray,
        n_samples: int = 0,
        return_std: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        아리송함을 곁들여 미루어 본다.
        
        Parameters
        ----------
        X : ndarray
            시험 들임
        n_samples : int
            0보다 크면 MC 표본 뽑기를, 아니면 곧게 펴기를 쓴다
        return_std : bool
            잣대 어긋남을 돌려줄지
        
        Returns
        -------
        mean : ndarray
            미루어 본 평균
        std : ndarray (골라 씀)
            미루어 본 잣대 어긋남
        """
        mean = self.forward(X)
        
        if not return_std:
            return mean, None
        
        if self.covariance is None:
            raise ValueError("먼저 fit_laplace()을 불러야 한다")
        
        if n_samples > 0:
            # 몬테카를로 표본 뽑기
            predictions = []
            for _ in range(n_samples):
                # 매개변수를 뽑는다
                params = np.random.multivariate_normal(
                    self.params, self.covariance
                )
                pred = self.forward(X, params)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            mean = np.mean(predictions, axis=0)
            std = np.std(predictions, axis=0)
        
        else:
            # 곧게 펴기
            J = self._compute_jacobian(X)
            
            # 흩어짐 = J @ Sigma @ J^T
            # 날임마다 J_i @ Sigma @ J_i^T을 셈한다
            var = np.sum(J @ self.covariance * J, axis=1)
            var = var.reshape(X.shape[0], -1)
            std = np.sqrt(np.maximum(var, 1e-10))
        
        return mean, std

class LastLayerLaplace:
    """
    마지막 켜에만 건 라플라스 어림.
    """
    
    def __init__(
        self,
        feature_extractor: Callable,
        n_features: int,
        n_outputs: int,
        prior_precision: float = 1.0
    ):
        """
        Parameters
        ----------
        feature_extractor : Callable
            X을 결로 옮기는 함수
        n_features : int
            결 밭의 차수
        n_outputs : int
            날임의 수
        prior_precision : float
            앞선 분포의 촘촘함
        """
        self.feature_extractor = feature_extractor
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.prior_precision = prior_precision
        
        # 마지막 켜 매개변수: W (n_features x n_outputs) + b (n_outputs)
        self.W = np.zeros((n_features, n_outputs))
        self.b = np.zeros(n_outputs)
        
        # 라플라스 함께 바뀜
        self.precision = None
        self.covariance = None
    
    @property
    def n_params(self) -> int:
        return self.n_features * self.n_outputs + self.n_outputs
    
    def _pack_params(self) -> np.ndarray:
        return np.concatenate([self.W.flatten(), self.b])
    
    def _unpack_params(self, params: np.ndarray):
        W = params[:self.n_features * self.n_outputs].reshape(
            self.n_features, self.n_outputs
        )
        b = params[self.n_features * self.n_outputs:]
        return W, b
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """결을 셈하고 마지막 켜를 건다."""
        phi = self.feature_extractor(X)
        return phi @ self.W + self.b
    
    def fit(self, X: np.ndarray, y: np.ndarray, n_iter: int = 100, lr: float = 0.01):
        """마지막 켜의 짐을 맞춘다."""
        phi = self.feature_extractor(X)
        
        for _ in range(n_iter):
            # MSE의 기울기
            pred = phi @ self.W + self.b
            error = pred - y
            
            grad_W = phi.T @ error / len(X) + self.prior_precision * self.W / len(X)
            grad_b = np.mean(error, axis=0)
            
            self.W -= lr * grad_W
            self.b -= lr * grad_b
    
    def fit_laplace(self, X: np.ndarray, y: np.ndarray):
        """마지막 켜의 라플라스 어림을 셈한다."""
        N = len(X)
        phi = self.feature_extractor(X)  # (N, n_features)
        
        # 잡음 흩어짐을 어림한다
        pred = phi @ self.W + self.b
        sigma2 = np.var(y - pred) + 1e-6
        
        # 마지막 켜에서는 헤세 행렬의 꼴이 깔끔하다
        # H_W = Phi^T Phi / (N * sigma^2) + prior_precision * I
        # 여기서는 단순하게 온전한 헤세 행렬을 셈한다
        
        # 치우침을 위해 결에 1을 덧댄다
        phi_aug = np.column_stack([phi, np.ones(N)])  # (N, n_features + 1)
        
        # 날임마다의 헤세 행렬(날임끼리 남남이라고 본다)
        H_single = phi_aug.T @ phi_aug / (N * sigma2)
        H_single += self.prior_precision * np.eye(self.n_features + 1)
        
        # 담아 둔다(이 단순한 자리에서는 날임마다 같다)
        self.precision = H_single
        self.covariance = np.linalg.inv(H_single)
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """아리송함을 곁들여 미루어 본다."""
        N = len(X)
        phi = self.feature_extractor(X)
        
        # 평균 미루어 봄
        mean = phi @ self.W + self.b
        
        # 곧게 펴서 얻는 흩어짐
        phi_aug = np.column_stack([phi, np.ones(N)])
        
        # var[i] = phi_aug[i] @ Sigma @ phi_aug[i]
        var = np.sum(phi_aug @ self.covariance * phi_aug, axis=1)
        var = np.maximum(var, 1e-10)
        
        # 날임마다 펼친다
        std = np.sqrt(var)[:, np.newaxis] * np.ones((1, self.n_outputs))
        
        return mean, std

# =============================================================================
# 보여 주기
# =============================================================================

def demo_laplace():
    """라플라스 어림을 보여 준다."""
    
    print("=" * 60)
    print("라플라스 어림 보여 주기")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 장난감 자료를 만든다
    N = 50
    X_train = np.random.uniform(-4, 4, N).reshape(-1, 1)
    y_train = np.sin(X_train) + 0.2 * np.random.randn(N, 1)
    
    X_test = np.linspace(-6, 6, 100).reshape(-1, 1)
    
    print(f"\n익힘 자료: {N}점")
    
    # MLP을 만들고 익힌다
    model = LaplaceMLP([1, 30, 30, 1], activation='tanh', prior_precision=0.1)
    
    # 기울기 내림으로 익힌다
    print("그물을 익히는 중...")
    lr = 0.1
    for epoch in range(200):
        # 기울기를 수로 셈한다
        grads = np.zeros_like(model.params)
        eps = 1e-5
        for i in range(len(model.params)):
            model.params[i] += eps
            loss_plus = model.loss(X_train, y_train)
            model.params[i] -= 2 * eps
            loss_minus = model.loss(X_train, y_train)
            model.params[i] += eps
            grads[i] = (loss_plus - loss_minus) / (2 * eps)
        
        model.params -= lr * grads
        lr *= 0.995  # 줄이기
    
    final_loss = model.loss(X_train, y_train)
    print(f"마지막 익힘 잃음: {final_loss:.4f}")
    
    # 라플라스 어림을 맞춘다
    print("\n라플라스 어림을 맞추는 중(대각)...")
    model.fit_laplace(X_train, y_train, hessian_type='diagonal')
    
    # 미루어 본다
    mean, std = model.predict(X_test, n_samples=0)
    
    print(f"\n아리송함 자:")
    print(f"  익힘 자리 [-4,4] 안의 평균 잣대 어긋남: {np.mean(std[np.abs(X_test) < 4]):.4f}")
    print(f"  익힘 자리 밖의 평균 잣대 어긋남: {np.mean(std[np.abs(X_test) > 4]):.4f}")
    
    # MC 표본 뽑기와 견준다
    print("\n곧게 펴기와 MC 표본 뽑기를 견주는 중...")
    mean_mc, std_mc = model.predict(X_test, n_samples=100)
    
    print(f"  평균끼리의 얽힘: {np.corrcoef(mean.flatten(), mean_mc.flatten())[0,1]:.4f}")
    print(f"  잣대 어긋남끼리의 얽힘: {np.corrcoef(std.flatten(), std_mc.flatten())[0,1]:.4f}")
    
    print("\n*** 아리송함은 익힘 자리 밖에서 더 커야 한다")
    
    return model

def demo_last_layer_laplace():
    """마지막 켜 라플라스를 보여 준다."""
    
    print("\n" + "=" * 60)
    print("마지막 켜 라플라스 보여 주기")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 자료를 만든다
    N = 100
    X_train = np.random.uniform(-4, 4, N).reshape(-1, 1)
    y_train = np.sin(X_train) + 0.2 * np.random.randn(N, 1)
    
    # 단순한 결 뽑개(붙박인 아무 결)
    n_features = 50
    W_feat = np.random.randn(1, n_features) * 0.5
    b_feat = np.random.randn(n_features)
    
    def feature_extractor(X):
        return np.tanh(X @ W_feat + b_feat)
    
    # 만들고 맞춘다
    model = LastLayerLaplace(
        feature_extractor=feature_extractor,
        n_features=n_features,
        n_outputs=1,
        prior_precision=1.0
    )
    
    model.fit(X_train, y_train, n_iter=100, lr=0.1)
    model.fit_laplace(X_train, y_train)
    
    # 미루어 본다
    X_test = np.linspace(-6, 6, 100).reshape(-1, 1)
    mean, std = model.predict(X_test)
    
    print(f"\n마지막 켜 라플라스(매개변수 {model.n_params}개뿐)")
    print(f"  익힘 자리 안의 평균 잣대 어긋남: {np.mean(std[np.abs(X_test) < 4]):.4f}")
    print(f"  밖의 평균 잣대 어긋남: {np.mean(std[np.abs(X_test) > 4]):.4f}")
    
    return model

if __name__ == "__main__":
    demo_laplace()
    demo_last_layer_laplace()
```

---

## 7. 참으로 헤아릴 것

### 헤세 행렬 어림 고르기

| 방법 | 쓸 자리 |
|--------|-------------|
| **대각** | 큰 그물, 빠른 어림, 거친 아리송함 |
| **KFAC** | 가운데 그물, 더 나은 아리송함, 얼개를 지님 |
| **마지막 켜** | 미리 익힌 그물, 풀이할 수 있는 결 |
| **온전한 GGN** | 작은 그물, 가장 좋은 어림 |

### 하이퍼파라미터 맞추기

**앞선 분포의 촘촘함**($\lambda = 1/\sigma_0^2$):

- 익힐 때 쓴 짐 줄이기에 맞물린다
- 클수록 → 정칙화가 세고 뒷분포가 좁아진다
- 남겨 둔 자료로 엇갈려 따진다

**온도 잣대 잡기**:
아리송함의 눈금이 어긋나면 함께 바뀜의 잣대를 맞춘다.

$$
\Sigma_{\text{scaled}} = T \cdot \Sigma
$$

$T > 1$이면 아리송함이 커지고 $T < 1$이면 작아진다.

### 셈 요령

1. **야코비 행렬을 미리 셈한다**: 시험 점이 붙박이면 $J(x^*)$을 한 번만 셈한다
2. **성기거나 얼개를 지닌 헤세 행렬을 쓴다**: KFAC이나 덩이 대각
3. **GPU으로 빠르게**: 행렬 셈은 나란히 하기 좋다
4. **조금씩 고치기**: 살아 있는 배움에서는 헤세 행렬을 조금씩 고친다

---

## 연습문제

**연습문제 1.**
ReLU 살림과 가우스 짐 앞선 분포를 지닌 두 켜 신경 그물에서, 이 마디에서 밝힌 방법에 따른 어림 뒷분포의 꼴을 이끌어 내어라.

??? success "연습문제 1 풀이"
    짐이 $W_1, W_2$이고 가우스 앞선 분포가 $p(W) = \mathcal{N}(0, \sigma_p^2 I)$일 때 뒷분포 $p(W | D) \propto p(D | W) p(W)$은 다룰 수 없다. 이 마디의 어림 방법은 다룰 수 있는 꼴을 낸다. 변이 미루어 봄이면 짐마다 서로 남남인 가우스 뒷분포 $q(w_{ij}) = \mathcal{N}(\mu_{ij}, \sigma_{ij}^2)$을 지니고, 라플라스 어림이면 뒷분포가 MAP 어림을 가운데로 삼고 함께 바뀜이 헤세 행렬의 거꿀인 가우스 하나이며, MC 드롭아웃이면 뒷분포가 드롭아웃 가리개 분포로 넌지시 세워진다. 어림마다 참 뒷분포 모습의 서로 다른 결을 담는다. $\square$

---

**연습문제 2.**
이 방법에서 얻은 아리송함 어림의 눈금 맞음을 MC 드롭아웃, 깊은 모둠과 견주는 시험을 꾸며라. 쓸 자와 그림을 밝혀라.

??? success "연습문제 2 풀이"
    자: (1) 통 15개의 바라는 눈금 맞음 어긋남(ECE), (2) 브라이어 점수, (3) 음수 로그 그럴듯함(NLL), (4) 밖 분포 알아내기의 AUROC. 그림: 방법마다 본 잦기를 미루어 본 자신함에 대고 그린 미더움 그림. 절차: 모든 방법을 CIFAR-10(분포 안)에서 익히고, CIFAR-10 시험 자료에서 눈금 맞음을, SVHN에서 밖 분포 알아내기를 따진다. 온도 잣대 잡기를 일 끝난 뒤 밑금으로 쓴다. 아무렇게나 하는 씨앗 5개에 걸친 평균과 잣대 어긋남을 알린다. 눈금이 잘 맞은 방법은 미더움 그림에서 점이 대각선에 가깝고 ECE가 낮다. $\square$

---

**연습문제 3.**
베이즈 신경 그물의 미루어 봄 흩어짐이 앎의 아리송함과 타고난 아리송함으로 쪼개짐을 증명하여라. 익힘 자료 크기 $N \to \infty$일 때 두 몫이 어떻게 되는지 보여라.

??? success "연습문제 3 풀이"
    미루어 봄 흩어짐은 온 흩어짐 법칙으로 쪼개진다. $\text{Var}[y | x, D] = \underbrace{\mathbb{E}_{p(\theta|D)}[\text{Var}[y | x, \theta]]}_{\text{타고난}} + \underbrace{\text{Var}_{p(\theta|D)}[\mathbb{E}[y | x, \theta]]}_{\text{앎의}}$. 타고난 몫은 자료를 낳는 흐름의 줄일 수 없는 잡음을 담으며 $N \to \infty$이어도 그대로다. 앎의 몫은 매개변수의 아리송함을 드러내며 뒷분포가 참 매개변수 언저리로 모이므로 $O(1/N)$으로 준다. 끝에 가면 타고난 아리송함만 남는다. 이 쪼갬은 자료를 더 모아야 할 때(앎의 아리송함이 클 때)와 타고난 잡음을 받아들여야 할 때(타고난 아리송함이 클 때)를 가르는 데 종요롭다. $\square$

---

**연습문제 4.**
이 마디의 아리송함 재기 방법을 거래 얼개의 자리 크기 잡기에 어떻게 쓸 수 있는지 다루어라. 손에 잡히는 판단 규칙을 내놓아라.

??? success "연습문제 4 풀이"
    판단 규칙: 자리 크기를 앎의 아리송함에 반비례하게 잡는다. $\hat{y}$을 미루어 본 돌아옴, $\sigma_e^2$을 앎의 흩어짐이라 하자. 자리는 $w = \frac{\hat{y}}{\lambda \sigma_e^2}$이고 $\lambda$은 무릅씀 꺼림 값이다. 앎의 아리송함이 크면(낯선 저자 형편) 자리를 줄이고, 작으면(익숙한 판) 더 굳게 거래한다. 여기에 더해 그 위로는 거래하지 않는 앎의 아리송함 위끝을 두어 삼갈 수 있다. 이 틀은 모형의 자신함으로 잣대를 잡은 켈리 잣대 결의 크기 잡기를 절로 이룬다. 앞으로 걸어가며 살피기로 되짚어 시험해 $\lambda$의 눈금을 맞춘다. $\square$

## 정리하며

### 고갱이 식

**라플라스 뒷분포**:

$$
q(\theta) = \mathcal{N}(\hat{\theta}_{\text{MAP}}, H^{-1})
$$

**미루어 본 흩어짐**(곧게 편 뒤):

$$
\text{Var}[f(x^*)] = J(x^*)^\top \Sigma J(x^*)
$$

**GGN 어림**:

$$
H_{\text{GGN}} = J^\top \nabla^2 \mathcal{L}_{\text{out}} J + \lambda I
$$

### 나은 점과 한계

| 나은 점 | 한계 |
|------------|-------------|
| 일 끝난 뒤(다시 익히지 않음) | 가우스라는 가정 |
| 이치에 닿는 밑바탕 | 그 자리만의 어림(봉우리 하나) |
| 어떤 얼개에도 듣는다 | 헤세 행렬 셈이 비쌀 수 있다 |
| 오래된 통계와 이어진다 | 아리송함을 낮게 볼 수 있다 |

### 다른 방법과의 이어짐

| 방법 | 사이 |
|--------|--------------|
| **SWAG** | 가우스를 맞추되 자취에서 얻는다 |
| **변이** | 라플라스는 VI의 남다른 한 자리다 |
| **모둠** | 라플라스와 함께 쓸 수 있다 |
| **피셔 소식** | 어떤 잃음에서는 GGN ≈ 피셔다 |

### 고갱이 살펴볼 거리

- MacKay, D. J. (1992). A practical Bayesian framework for backpropagation networks. *Neural Computation*.
- Ritter, H., et al. (2018). A scalable Laplace approximation for neural networks. *ICLR*.
- Daxberger, E., et al. (2021). Laplace redux — Effortless Bayesian deep learning. *NeurIPS*.
- Martens, J., & Grosse, R. (2015). Optimizing neural networks with Kronecker-factored approximate curvature. *ICML*.
