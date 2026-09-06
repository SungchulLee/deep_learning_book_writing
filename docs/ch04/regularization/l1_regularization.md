# L1 정칙화 (라쏘)
## 개요

라쏘(Least Absolute Shrinkage and Selection Operator)라고도 하는 L1 정칙화는 모델 가중치의 절댓값에 비례하는 벌점을 손실 함수에 더한다. 이 기법은 학습된 매개변수의 희소성을 북돋우며, 관련 없는 특징의 가중치를 정확히 0으로 몰아 사실상 자동 특징 선택을 수행한다.

## 수학적 정식화

### L1 벌점을 더한 표준 손실

매개변수가 $\theta = \{w_1, w_2, \ldots, w_n\}$인 손실 함수 $\mathcal{L}(\theta)$에 대해 L1 정칙화는 목적 함수를 다음과 같이 바꾼다.

$$
\mathcal{L}_{\text{L1}}(\theta) = \mathcal{L}(\theta) + \lambda \sum_{i=1}^{n} |w_i|
$$

여기서 각 기호는 다음과 같다.

- $\mathcal{L}(\theta)$은 원래의 손실 함수이다(예: MSE, 교차 엔트로피)
- $\lambda \geq 0$은 정칙화의 강도이다(초매개변수)
- $\sum_{i=1}^{n} |w_i|$은 가중치 벡터의 L1 노름이다

### L1을 쓰는 선형 회귀 (라쏘 회귀)

설계 행렬 $X \in \mathbb{R}^{m \times n}$, 목표 $y \in \mathbb{R}^m$, 가중치 $w \in \mathbb{R}^n$인 선형 회귀에 대해 다음이 성립한다.

$$
\mathcal{L}_{\text{Lasso}}(w) = \frac{1}{2m} \|Xw - y\|_2^2 + \lambda \|w\|_1
$$

항을 펼치면 다음과 같다.

$$
\mathcal{L}_{\text{Lasso}}(w) = \frac{1}{2m} \sum_{j=1}^{m} \left( \sum_{i=1}^{n} x_{ji} w_i - y_j \right)^2 + \lambda \sum_{i=1}^{n} |w_i|
$$

### 기울기와 부분기울기

L1 노름은 $w_i = 0$에서 미분할 수 없다. 부분기울기를 쓴다.

$$
\frac{\partial}{\partial w_i} |w_i| = 
\begin{cases}
+1 & \text{if } w_i > 0 \\
-1 & \text{if } w_i < 0 \\
[-1, +1] & \text{if } w_i = 0
\end{cases}
$$

이는 부호 함수를 써서 다음과 같이 쓸 수 있다.

$$
\frac{\partial}{\partial w_i} |w_i| = \text{sign}(w_i) = 
\begin{cases}
+1 & \text{if } w_i > 0 \\
-1 & \text{if } w_i < 0 \\
0 & \text{if } w_i = 0
\end{cases}
$$

L1 정칙화된 손실의 전체 기울기는 다음과 같다.

$$
\nabla_{w} \mathcal{L}_{\text{L1}} = \nabla_{w} \mathcal{L} + \lambda \cdot \text{sign}(w)
$$

## 기하학적 해석

### 제약 영역

L1 정칙화는 L1 공 제약을 갖는 제약 최적화와 동등하다.

$$
\min_w \mathcal{L}(w) \quad \text{subject to} \quad \|w\|_1 \leq t
$$

2차원에서 L1 공은 축 위에 모서리가 있는 마름모(회전한 정사각형) 모양을 이룬다.

$$
\|w\|_1 = |w_1| + |w_2| \leq t
$$

### 기하로부터 나오는 희소성

L1 제약 영역의 마름모 모양은 좌표축 위에 **모서리**를 갖는다. 손실 함수의 등고선이 제약 영역과 만날 때 이 모서리에서 닿을 가능성이 더 크며, 그 결과 어떤 가중치는 정확히 0인 해가 된다.

**L2(원형 제약)와의 비교:**

| 성질 | L1 (마름모) | L2 (원) |
|----------|--------------|-------------|
| 모서리 점 | 있음 (축 위) | 없음 |
| 희소한 해 | 그렇다 | 드물다 |
| 미분 가능 | 아니다 (모서리에서) | 그렇다 |

## 희소성과 특징 선택

### L1이 희소한 해를 만드는 이유

가중치 하나 $w_i$에 대한 최적화 지형을 생각하자.

1. **영에서 떨어져 있을 때**: 기울기에 $\lambda \cdot \text{sign}(w_i)$이 들어가 $w_i$을 영 쪽으로 민다
2. **영에서**: 부분기울기가 구간 $[-\lambda, +\lambda]$을 포함한다
3. **영에 머무를 조건**: $w_i = 0$에서 원래 손실의 기울기가 $[-\lambda, +\lambda]$ 안에 있으면 $w_i = 0$이 최적이다

즉 예측력이 약한 특징의 가중치는 정확히 0으로 몰린다.

### 연성 문턱화 연산자

L1 정칙화된 최소제곱에서 (좌표 하강법의) 각 좌표에 대한 닫힌 형태의 해는 다음과 같다.

$$
w_i^* = S_{\lambda}(z_i) = \text{sign}(z_i) \cdot \max(|z_i| - \lambda, 0)
$$

여기서 $z_i$은 다른 가중치를 고정했을 때 $w_i$에 대한 보통최소제곱의 해이다. 이것이 **연성 문턱화** 또는 **수축** 연산자이다.

## PyTorch 구현

### L1 정칙화 직접 구현하기

```python
import torch
import torch.nn as nn

def l1_regularization(model: nn.Module, lambda_l1: float) -> torch.Tensor:
    """
    모델의 모든 매개변수에 대한 L1 정칙화 벌점을 계산한다.
    
    인수:
        model: 신경망 모델
        lambda_l1: 정칙화의 강도
        
    반환값:
        L1 벌점 항
    """
    l1_penalty = torch.tensor(0., requires_grad=True)
    for param in model.parameters():
        l1_penalty = l1_penalty + torch.sum(torch.abs(param))
    return lambda_l1 * l1_penalty

class L1RegularizedTrainer:
    """L1 정칙화를 지원하는 학습기."""
    
    def __init__(self, model, criterion, optimizer, lambda_l1=0.01):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lambda_l1 = lambda_l1
    
    def train_step(self, X, y):
        self.optimizer.zero_grad()
        
        # 순전파
        predictions = self.model(X)
        loss = self.criterion(predictions, y)
        
        # L1 벌점 더하기
        l1_penalty = l1_regularization(self.model, self.lambda_l1)
        total_loss = loss + l1_penalty
        
        # 역전파
        total_loss.backward()
        self.optimizer.step()
        
        return loss.item(), l1_penalty.item()
```

### PyTorch의 내장 정칙화 장치 쓰기

```python
import torch.nn as nn
from torch.nn.utils import parametrize

class L1Regularizer(nn.Module):
    """매개변수화로 구현한 L1 가중치 정칙화 장치."""
    
    def __init__(self, lambda_l1: float):
        super().__init__()
        self.lambda_l1 = lambda_l1
    
    def forward(self, weight):
        return weight
    
    def right_inverse(self, weight):
        return weight

def add_l1_regularization_to_loss(model, base_loss, lambda_l1):
    """아무 손실 함수에나 L1 정칙화를 더한다."""
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return base_loss + lambda_l1 * l1_norm
```

### L1 정칙화를 쓰는 신경망

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class SparseNN(nn.Module):
    """희소 특징 학습을 위해 설계한 신경망."""
    
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def get_l1_norm(self):
        """모든 가중치의 L1 노름을 계산한다."""
        l1_norm = 0
        for param in self.parameters():
            l1_norm += torch.sum(torch.abs(param))
        return l1_norm
    
    def count_zero_weights(self, threshold=1e-6):
        """사실상 영인 가중치의 수를 센다."""
        total = 0
        zeros = 0
        for param in self.parameters():
            total += param.numel()
            zeros += (param.abs() < threshold).sum().item()
        return zeros, total

def train_with_l1(model, train_loader, val_loader, 
                  lambda_l1=0.01, epochs=100, lr=0.001):
    """
    L1 정칙화로 모델을 학습시킨다.
    
    인수:
        model: 신경망
        train_loader: 학습 데이터 로더
        val_loader: 검증 데이터 로더
        lambda_l1: L1 정칙화의 강도
        epochs: 학습 에포크 수
        lr: 학습률
        
    반환값:
        학습 이력 사전
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'l1_norm': [], 'sparsity': []
    }
    
    for epoch in range(epochs):
        # 학습
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            predictions = model(X_batch)
            mse_loss = criterion(predictions, y_batch)
            l1_penalty = lambda_l1 * model.get_l1_norm()
            loss = mse_loss + l1_penalty
            
            loss.backward()
            optimizer.step()
            train_loss += mse_loss.item()
        
        # 검증
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                val_loss += criterion(predictions, y_batch).item()
        
        # 지표를 추적한다
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        zeros, total = model.count_zero_weights()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['l1_norm'].append(model.get_l1_norm().item())
        history['sparsity'].append(zeros / total)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, Sparsity={zeros/total:.2%}")
    
    return history
```

### L1을 위한 근접 경사 하강법

L1로 더 효율적으로 최적화하려면 근접 경사 하강법을 쓴다.

```python
def proximal_l1(weights: torch.Tensor, lambda_l1: float, 
                lr: float) -> torch.Tensor:
    """
    L1 정칙화의 근접 연산자 (연성 문턱화).
    
    인수:
        weights: 매개변수 텐서
        lambda_l1: 정칙화의 강도
        lr: 학습률
        
    반환값:
        연성 문턱화를 거친 가중치
    """
    threshold = lambda_l1 * lr
    return torch.sign(weights) * torch.clamp(torch.abs(weights) - threshold, min=0)

class ProximalL1Optimizer:
    """L1을 위한 근접 경사 하강법을 구현한 최적화기."""
    
    def __init__(self, model, lr=0.01, lambda_l1=0.01):
        self.model = model
        self.lr = lr
        self.lambda_l1 = lambda_l1
    
    def step(self):
        """근접 갱신과 함께 최적화 단계를 한 번 수행한다."""
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    # 기울기 단계
                    param.data -= self.lr * param.grad
                    # 근접 단계 (연성 문턱화)
                    param.data = proximal_l1(param.data, self.lambda_l1, self.lr)
    
    def zero_grad(self):
        """모든 매개변수의 기울기를 0으로 만든다."""
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()
```

## scikit-learn 구현

```python
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
import numpy as np

def lasso_feature_selection(X, y, alphas=None):
    """
    교차 검증과 함께 라쏘로 특징을 선택한다.
    
    인수:
        X: 특징 행렬 (n_samples, n_features)
        y: 목표 벡터
        alphas: 시도해 볼 정칙화 값
        
    반환값:
        선택된 특징의 인덱스와 라쏘 모델
    """
    # 특징 표준화 (L1에서 중요하다)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 교차 검증으로 최적 alpha 찾기
    if alphas is None:
        alphas = np.logspace(-4, 1, 50)
    
    lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42)
    lasso_cv.fit(X_scaled, y)
    
    print(f"Optimal alpha: {lasso_cv.alpha_:.6f}")
    print(f"Non-zero coefficients: {np.sum(lasso_cv.coef_ != 0)}/{len(lasso_cv.coef_)}")
    
    # 선택된 특징 얻기
    selected_features = np.where(lasso_cv.coef_ != 0)[0]
    
    return selected_features, lasso_cv, scaler

def compare_regularization_strengths(X, y, alphas):
    """alpha 값에 따른 희소성의 양상을 비교한다."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = []
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, random_state=42)
        lasso.fit(X_scaled, y)
        
        n_nonzero = np.sum(lasso.coef_ != 0)
        results.append({
            'alpha': alpha,
            'n_nonzero': n_nonzero,
            'coefficients': lasso.coef_.copy()
        })
    
    return results
```

## 정칙화 경로

**정칙화 경로**는 $\lambda$이 변할 때 계수가 어떻게 바뀌는지 보여 준다.

```python
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path

def plot_lasso_path(X, y, eps=1e-3, n_alphas=100):
    """
    라쏘의 정칙화 경로를 그린다.
    
    인수:
        X: 특징 행렬
        y: 목표 벡터
        eps: 경로의 길이 (alpha_min / alpha_max)
        n_alphas: alpha 값의 개수
    """
    # 표준화
    X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
    y_centered = y - y.mean()
    
    # 라쏘 경로 계산
    alphas, coefs, _ = lasso_path(X_scaled, y_centered, eps=eps, n_alphas=n_alphas)
    
    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    for i in range(coefs.shape[0]):
        plt.plot(alphas, coefs[i], label=f'Feature {i}')
    
    plt.xscale('log')
    plt.xlabel('Regularization strength (α)')
    plt.ylabel('Coefficient value')
    plt.title('Lasso Regularization Path')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.gca().invert_xaxis()  # alpha를 줄여 가며
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    
    return alphas, coefs
```

## 비교: L1과 L2

| 항목 | L1 (라쏘) | L2 (능선) |
|--------|------------|------------|
| 벌점 | $\lambda \sum \|w_i\|$ | $\lambda \sum w_i^2$ |
| 제약의 모양 | 마름모 | 원 |
| 희소한 해 | 그렇다 | 아니다 |
| 특징 선택 | 내장 | 없음 |
| 미분 가능 | 아니다 (0에서) | 그렇다 |
| 상관된 특징 | 하나를 고른다 | 가중치를 나눈다 |
| 닫힌 형태의 해 | 없음 | 있음 |

## 초매개변수 선택

### lambda를 위한 교차 검증

```python
from sklearn.model_selection import cross_val_score
import numpy as np

def select_lambda_cv(X, y, lambdas, cv=5):
    """
    교차 검증으로 최적의 lambda를 고른다.
    
    인수:
        X: 특징
        y: 목푯값
        lambdas: 평가할 lambda 값
        cv: 교차 검증 겹의 수
        
    반환값:
        최적의 lambda와 교차 검증 점수
    """
    from sklearn.linear_model import Lasso
    
    cv_scores = []
    for lam in lambdas:
        model = Lasso(alpha=lam)
        scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
        cv_scores.append(-scores.mean())
    
    optimal_idx = np.argmin(cv_scores)
    return lambdas[optimal_idx], cv_scores
```

### 정보 기준

대안으로 모델 선택에 AIC나 BIC를 쓴다.

$$
\text{AIC} = 2k - 2\ln(\hat{L})
$$

$$
\text{BIC} = k\ln(n) - 2\ln(\hat{L})
$$

여기서 $k$은 영이 아닌 매개변수의 개수이다.

## 딥러닝에서의 응용

### 희소한 입력층

첫 층에 L1 정칙화를 걸면 입력 특징의 선택이 촉진된다.

```python
class SparseInputNetwork(nn.Module):
    """특징 선택을 위해 입력층에 L1 정칙화를 건 신경망."""
    
    def __init__(self, input_dim, hidden_dims, output_dim, input_l1=0.01):
        super().__init__()
        self.input_l1 = input_l1
        
        # 첫 층 (강하게 정칙화)
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        
        # 은닉층
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.ReLU(),
            ])
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.hidden = nn.Sequential(*layers)
    
    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        return self.hidden(x)
    
    def get_input_l1_penalty(self):
        """입력층에만 적용하는 L1 벌점."""
        return self.input_l1 * torch.sum(torch.abs(self.input_layer.weight))
```

## 실무 지침

### L1 정칙화를 쓸 때

1. **특징 선택이 필요할 때**: 특징이 많고 그중 가장 중요한 것을 가려내고 싶을 때
2. **해석 가능성이 중요할 때**: 희소한 모델이 해석하기 더 쉽다
3. **고차원 데이터**: $p > n$일 때(특징이 표본보다 많을 때)
4. **관련 없는 특징이 의심될 때**: 많은 특징이 잡음일 가능성이 클 때

### lambda 고르기

- **너무 작으면**: 정칙화가 약하고 과적합하며 영이 아닌 가중치가 많다
- **너무 크면**: 과소적합하고 고른 특징이 너무 적다
- **교차 검증을 쓰라**: 데이터가 선택을 이끌게 하라
- **척도를 고려하라**: L1을 적용하기 전에 특징을 표준화하라

### 흔한 함정

1. **특징의 척도 조정**: L1은 크기에 따라 벌점을 준다. 척도를 맞추지 않은 특징은 편향된 선택으로 이어진다
2. **상관된 특징**: L1은 상관된 특징 중 하나를 임의로 고른다
3. **미분 불가능성**: 표준 경사 하강법은 어려움을 겪을 수 있으므로 근접 방법을 쓴다
4. **편향-분산 절충**: $\lambda$이 크면 편향은 늘고 분산은 준다

## 참고 문헌

1. Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267-288.
2. Hastie, T., Tibshirani, R., & Wainwright, M. (2015). *Statistical Learning with Sparsity: The Lasso and Generalizations*. CRC Press.
3. Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.

## 연습문제

**연습문제 1.**
영에서의 부분기울기를 써서 L1 정칙화가 희소한 가중치를 만드는 이유를 설명하라.

??? success "연습문제 1 풀이"
    L1 벌점 $\lambda|w|$은 $w \neq 0$에서 부분기울기 $\lambda\text{sign}(w)$을, $w = 0$에서 부분미분 $[-\lambda, \lambda]$을 갖는다. 데이터 기울기의 크기가 $\lambda$보다 작으면 가중치는 영에 머무르며, 이 죽은 구간이 정확한 영을 만든다.

---

**연습문제 2.**
고차원 회귀 문제에서 L1과 L2가 만드는 희소성의 양상을 비교하라.

??? success "연습문제 2 풀이"
    L1은 관련 없는 특징을 0으로 만들어(강한 희소성) 일부만 남긴다. L2는 모든 특징을 영 쪽으로 수축시키되(약한 희소성) 전부 남긴다. 특징 선택에는 L1이, 상관된 특징에서 안정적인 예측에는 L2가 낫다.

---

**연습문제 3.**
L1 정칙화 학습을 위한 근접 경사 하강법을 구현하라.

??? success "연습문제 3 풀이"
    ```python
    def proximal_step(w, grad, lr, lam):
        w_tmp = w - lr * grad  # 기울기 단계
        # 연성 문턱화 (L1의 근접 연산자)
        return torch.sign(w_tmp) * torch.clamp(w_tmp.abs() - lr * lam, min=0)
    ```

---

**연습문제 4.**
L1 정칙화를 라플라스 사전분포로 보는 베이즈적 해석을 유도하라.

??? success "연습문제 4 풀이"
    라플라스 사전분포는 $p(w) \propto e^{-\lambda|w|}$이다. MAP는 $\log p(w|D) \propto \log p(D|w) - \lambda\sum|w_i| = -L(w) - \lambda\|w\|_1$이다. 그 음수를 최소화하는 것이 L1 정칙화된 손실이다.
