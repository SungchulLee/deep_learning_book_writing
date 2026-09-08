# 엘라스틱 넷 정칙화

엘라스틱 넷은 L1(라쏘)과 L2(능선) 정칙화를 결합하여 두 방식의 장점을 모두 물려받는다. 라쏘처럼 희소한 모델을 이끌면서도 능선 회귀의 안정성을 유지하므로 상관된 특징을 다룰 때 특히 효과적이다.

---

## 1. 수학적 정식화

### 결합된 벌점

엘라스틱 넷의 벌점은 L1 노름과 L2 노름의 가중 결합이다.

$$
\Omega(w) = \alpha \|w\|_1 + \frac{1 - \alpha}{2} \|w\|_2^2
$$

여기서 각 기호는 다음과 같다.

- $\alpha \in [0, 1]$은 혼합 매개변수이다
- $\alpha = 1$이면 순수한 L1(라쏘)이 된다
- $\alpha = 0$이면 순수한 L2(능선)가 된다
- $0 < \alpha < 1$이면 엘라스틱 넷이 된다

### 전체 목적 함수

선형 회귀에 대해 다음과 같다.

$$
\mathcal{L}_{\text{ElasticNet}}(w) = \frac{1}{2m} \|Xw - y\|_2^2 + \lambda \left( \alpha \|w\|_1 + \frac{1 - \alpha}{2} \|w\|_2^2 \right)
$$

이는 별도의 정칙화 매개변수로 다시 쓸 수 있다.

$$
\mathcal{L}(w) = \frac{1}{2m} \|Xw - y\|_2^2 + \lambda_1 \|w\|_1 + \lambda_2 \|w\|_2^2
$$

여기서 $\lambda_1 = \lambda \alpha$이고 $\lambda_2 = \lambda (1 - \alpha) / 2$이다.

### 기울기

기울기는 L1의 부분기울기와 L2의 기울기를 결합한다.

$$
\nabla_w \mathcal{L} = \frac{1}{m} X^T(Xw - y) + \lambda \alpha \cdot \text{sign}(w) + \lambda(1 - \alpha) w
$$

L1 항에서 오는 영에서의 미분 불가능성은 그대로 남으므로 부분기울기 방법이나 근접 최적화가 필요하다.

---

## 2. 왜 엘라스틱 넷인가?

### 순수한 L1(라쏘)의 한계

1. **상관된 특징**: 라쏘는 상관된 특징 중 하나를 임의로 고른다
2. **포화**: $n < p$이면 라쏘는 많아야 $n$개의 특징만 고른다
3. **불안정성**: 데이터가 조금만 바뀌어도 특징 선택이 뒤집힐 수 있다

### 순수한 L2(능선)의 한계

1. **희소성 없음**: 모든 계수가 영이 아닌 채로 남는다
2. **특징 선택 없음**: 관련 없는 특징을 가려내지 못한다
3. **해석 가능성**: 조밀한 모델은 해석하기 더 어렵다

### 엘라스틱 넷의 해결책

엘라스틱 넷은 다음으로 이 문제들을 다룬다.

1. **묶음 선택**: 상관된 특징의 무리를 함께 고르거나 함께 버리는 경향이 있다
2. **포화 없음**: $n$개보다 많은 특징을 고를 수 있다
3. **안정성**: L2 성분이 상관된 특징 사이의 선택을 안정시킨다

---

## 3. 기하학적 해석

### 제약 영역의 모양

엘라스틱 넷의 제약 영역은 L1의 마름모와 L2의 원 사이를 보간한다.

$$
\alpha \|w\|_1 + \frac{1 - \alpha}{2} \|w\|_2^2 \leq t
$$

이는 다음을 갖는 "둥근 마름모" 모양을 만든다.

- 희소성을 북돋우는 (L1에서 오는) 모서리
- 극단적인 모서리 해를 막는 (L2에서 오는) 굽은 변

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_elastic_net_constraint(alphas=[0.0, 0.3, 0.7, 1.0]):
    """혼합 매개변수에 따른 엘라스틱 넷의 제약 영역을 시각화한다."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    
    theta = np.linspace(0, 2*np.pi, 1000)
    
    for ax, alpha in zip(axes, alphas):
        # 각도마다 경계점 찾기
        w1_vals = []
        w2_vals = []
        
        for t in theta:
            # 방향
            d1, d2 = np.cos(t), np.sin(t)
            
            # alpha*||w||_1 + (1-alpha)/2*||w||_2^2 = 1 을 만족하는 r 찾기
            # 수치 탐색 이용
            for r in np.linspace(0.01, 3, 1000):
                w1, w2 = r * d1, r * d2
                penalty = alpha * (abs(w1) + abs(w2)) + (1-alpha)/2 * (w1**2 + w2**2)
                if penalty >= 1:
                    w1_vals.append(w1)
                    w2_vals.append(w2)
                    break
        
        ax.plot(w1_vals, w2_vals, 'b-', linewidth=2)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title(f'α = {alpha}' + (' (Ridge)' if alpha == 0 else ' (Lasso)' if alpha == 1 else ''))
        ax.set_xlabel('$w_1$')
        ax.set_ylabel('$w_2$')
    
    plt.tight_layout()
    return fig
```

---

## 4. PyTorch 구현

### 엘라스틱 넷 정칙화 직접 구현하기

```python
import torch
import torch.nn as nn
import torch.optim as optim

def elastic_net_penalty(model: nn.Module, lambda_reg: float, 
                        alpha: float) -> torch.Tensor:
    """
    엘라스틱 넷 정칙화의 벌점을 계산한다.
    
    인수:
        model: 신경망 모델
        lambda_reg: 전체 정칙화의 강도
        alpha: 혼합 매개변수 (0이면 능선, 1이면 라쏘)
        
    반환값:
        엘라스틱 넷의 벌점 항
    """
    l1_penalty = torch.tensor(0., device=next(model.parameters()).device)
    l2_penalty = torch.tensor(0., device=next(model.parameters()).device)
    
    for param in model.parameters():
        l1_penalty = l1_penalty + torch.sum(torch.abs(param))
        l2_penalty = l2_penalty + torch.sum(param ** 2)
    
    return lambda_reg * (alpha * l1_penalty + (1 - alpha) / 2 * l2_penalty)

class ElasticNetRegularizedModel(nn.Module):
    """엘라스틱 넷 정칙화를 쓰는 신경망."""
    
    def __init__(self, input_dim, hidden_dims, output_dim, 
                 lambda_reg=0.01, alpha=0.5):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.alpha = alpha
        
        # 신경망 만들기
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def get_elastic_net_penalty(self):
        """엘라스틱 넷의 벌점을 계산한다."""
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        l2_norm = sum((p ** 2).sum() for p in self.parameters())
        return self.lambda_reg * (self.alpha * l1_norm + 
                                   (1 - self.alpha) / 2 * l2_norm)
    
    def get_sparsity(self, threshold=1e-6):
        """영에 가까운 가중치의 비율을 계산한다."""
        total = 0
        zeros = 0
        for param in self.parameters():
            total += param.numel()
            zeros += (param.abs() < threshold).sum().item()
        return zeros / total
```

### 엘라스틱 넷으로 학습하기

```python
def train_elastic_net_model(
    model: ElasticNetRegularizedModel,
    train_loader,
    val_loader,
    epochs: int = 100,
    lr: float = 0.001
) -> dict:
    """
    엘라스틱 넷 정칙화로 모델을 학습시킨다.
    
    인수:
        model: 엘라스틱 넷이 내장된 모델
        train_loader: 학습 데이터
        val_loader: 검증 데이터
        epochs: 에포크 수
        lr: 학습률
        
    반환값:
        학습 이력
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [], 'val_loss': [], 
        'penalty': [], 'sparsity': []
    }
    
    for epoch in range(epochs):
        # 학습
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            predictions = model(X_batch)
            mse_loss = criterion(predictions, y_batch)
            penalty = model.get_elastic_net_penalty()
            total_loss = mse_loss + penalty
            
            total_loss.backward()
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
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['penalty'].append(model.get_elastic_net_penalty().item())
        history['sparsity'].append(model.get_sparsity())
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Train={train_loss/len(train_loader):.4f}, "
                  f"Val={val_loss/len(val_loader):.4f}, "
                  f"Sparsity={model.get_sparsity():.2%}")
    
    return history
```

### 엘라스틱 넷을 위한 근접 경사 하강법

```python
def proximal_elastic_net(w: torch.Tensor, lambda_reg: float, 
                          alpha: float, lr: float) -> torch.Tensor:
    """
    엘라스틱 넷의 근접 연산자.
    
    엘라스틱 넷의 근접 연산자는 연성 문턱화 뒤에
    (L2 성분 때문에) 배율을 조정하는 것이다.
    
    인수:
        w: 가중치 텐서
        lambda_reg: 정칙화의 강도
        alpha: 혼합 매개변수
        lr: 학습률
        
    반환값:
        근접 단계를 거친 뒤의 가중치
    """
    # L1 문턱값
    l1_threshold = lambda_reg * alpha * lr
    
    # L2 배율 인수
    l2_scale = 1.0 / (1.0 + lambda_reg * (1 - alpha) * lr)
    
    # 연성 문턱화 뒤 배율 조정
    soft_thresh = torch.sign(w) * torch.clamp(torch.abs(w) - l1_threshold, min=0)
    return l2_scale * soft_thresh

class ProximalElasticNetOptimizer:
    """엘라스틱 넷을 위한 근접 경사 하강 최적화기."""
    
    def __init__(self, model, lr=0.01, lambda_reg=0.01, alpha=0.5):
        self.model = model
        self.lr = lr
        self.lambda_reg = lambda_reg
        self.alpha = alpha
    
    def step(self):
        """근접 경사 단계를 한 번 수행한다."""
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    # 기울기 단계 (매끄러운 손실 부분에 대해)
                    param.data -= self.lr * param.grad
                    # 근접 단계 (엘라스틱 넷 벌점에 대해)
                    param.data = proximal_elastic_net(
                        param.data, self.lambda_reg, self.alpha, self.lr
                    )
    
    def zero_grad(self):
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()
```

---

## 5. scikit-learn 구현

```python
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import numpy as np

def elastic_net_analysis(X, y, l1_ratios=None, alphas=None):
    """
    교차 검증을 곁들인 엘라스틱 넷의 종합 분석.
    
    인수:
        X: 특징 행렬
        y: 목표 벡터
        l1_ratios: 시도해 볼 L1 비율 값 (이 책의 표기로는 alpha)
        alphas: 시도해 볼 정칙화의 강도
        
    반환값:
        가장 좋은 모델과 분석 결과
    """
    # 특징을 표준화한다
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if l1_ratios is None:
        l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
    if alphas is None:
        alphas = np.logspace(-4, 1, 50)
    
    # 두 매개변수 모두에 대한 교차 검증
    elastic_cv = ElasticNetCV(
        l1_ratio=l1_ratios,
        alphas=alphas,
        cv=5,
        random_state=42,
        max_iter=10000
    )
    elastic_cv.fit(X_scaled, y)
    
    # 결과
    n_nonzero = np.sum(elastic_cv.coef_ != 0)
    
    print("Elastic Net CV Results:")
    print(f"  Optimal alpha (λ): {elastic_cv.alpha_:.6f}")
    print(f"  Optimal l1_ratio: {elastic_cv.l1_ratio_:.2f}")
    print(f"  Non-zero coefficients: {n_nonzero}/{len(elastic_cv.coef_)}")
    print(f"  R² score: {elastic_cv.score(X_scaled, y):.4f}")
    
    return elastic_cv, scaler

def compare_regularization_methods(X, y, alpha_values):
    """라쏘, 능선, 엘라스틱 넷을 비교한다."""
    from sklearn.linear_model import Lasso, Ridge
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    
    for alpha in alpha_values:
        # 라쏘
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X_scaled, y)
        lasso_nonzero = np.sum(lasso.coef_ != 0)
        
        # 능선
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_scaled, y)
        ridge_nonzero = np.sum(np.abs(ridge.coef_) > 1e-6)
        
        # 엘라스틱 넷 (50% 혼합)
        elastic = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000)
        elastic.fit(X_scaled, y)
        elastic_nonzero = np.sum(elastic.coef_ != 0)
        
        results[alpha] = {
            'lasso_nonzero': lasso_nonzero,
            'ridge_nonzero': ridge_nonzero,
            'elastic_nonzero': elastic_nonzero,
            'lasso_score': lasso.score(X_scaled, y),
            'ridge_score': ridge.score(X_scaled, y),
            'elastic_score': elastic.score(X_scaled, y)
        }
    
    return results
```

---

## 6. 초매개변수 선택

### 이차원 탐색

엘라스틱 넷에는 초매개변수가 둘 있다. $\lambda$(전체 강도)와 $\alpha$(L1/L2 혼합)이다.

```python
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

def grid_search_elastic_net(X, y):
    """
    엘라스틱 넷의 두 초매개변수에 대한 격자 탐색.
    """
    param_grid = {
        'alpha': np.logspace(-4, 1, 20),  # 전체 정칙화 강도
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]  # 혼합
    }
    
    grid_search = GridSearchCV(
        ElasticNet(max_iter=10000),
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_, grid_search.cv_results_
```

### 정칙화 경로

```python
from sklearn.linear_model import enet_path
import matplotlib.pyplot as plt

def plot_elastic_net_path(X, y, l1_ratio=0.5, eps=1e-3):
    """
    엘라스틱 넷의 정칙화 경로를 그린다.
    
    인수:
        X: 특징 행렬
        y: 목표 벡터
        l1_ratio: 고정된 혼합 매개변수
        eps: 경로의 길이
    """
    # 표준화
    X_centered = X - X.mean(axis=0)
    y_centered = y - y.mean()
    
    # 경로 계산
    alphas, coefs, _ = enet_path(
        X_centered, y_centered, 
        l1_ratio=l1_ratio,
        eps=eps
    )
    
    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    for i in range(coefs.shape[0]):
        plt.plot(alphas, coefs[i], label=f'Feature {i}')
    
    plt.xscale('log')
    plt.xlabel('Regularization strength (α)')
    plt.ylabel('Coefficient value')
    plt.title(f'Elastic Net Path (l1_ratio={l1_ratio})')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.gca().invert_xaxis()
    plt.legend(loc='best', fontsize=8)
    
    return alphas, coefs
```

---

## 7. 이론적 성질

### 묶음 효과

상관계수가 $\rho$인 강하게 상관된 특징 $x_i$과 $x_j$에 대해 엘라스틱 넷의 계수는 다음을 만족한다.

$$
|w_i - w_j| \leq \frac{1}{\lambda(1-\alpha)} \|y\|_1 \sqrt{2(1 - \rho)}
$$

즉 강하게 상관된 특징은 비슷한 계수를 갖는다. 하나를 임의로 고르는 라쏘와 다른 점이다.

### 유일한 해

해가 여럿일 수 있는 라쏘와 달리, $\alpha < 1$인 엘라스틱 넷은 순볼록한 L2 항 덕분에 해가 유일하다.

### 오라클 성질

어떤 조건 아래에서 엘라스틱 넷은 오라클 성질을 얻는다. 즉 표본 크기가 커질수록 확률이 1에 가까워지도록 올바른 특징을 고른다.

---

## 8. 실무 지침

### 혼합 매개변수 α 고르기

| 상황 | 권장 α |
|----------|---------------|
| 강한 특징 선택이 필요할 때 | 0.9 - 0.99 |
| 적당한 희소성 | 0.5 - 0.7 |
| 약간의 희소성과 함께 안정성 | 0.1 - 0.3 |
| 강하게 상관된 특징 | 0.1 - 0.5 |

### 엘라스틱 넷을 쓸 때

1. **상관된 특징**: 특징의 무리가 서로 상관되어 있고 묶음 선택을 원할 때
2. **높은 차원**: $p >> n$이어서 라쏘가 포화할 때
3. **안정성이 필요할 때**: 순수한 희소성보다 일관된 특징 선택이 더 중요할 때
4. **L1과 L2 중 무엇이 나을지 모를 때**: 어느 정칙화가 더 좋은지 확신이 없을 때

### 비교 요약

| 방법 | 희소성 | 안정성 | 상관된 특징 | 유일한 해 |
|--------|----------|-----------|---------------------|-----------------|
| L1 (라쏘) | 높음 | 낮음 | 임의 선택 | 언제나 그렇지는 않음 |
| L2 (능선) | 없음 | 높음 | 동등한 가중 | 그렇다 |
| 엘라스틱 넷 | 보통 | 보통~높음 | 묶음 선택 | 그렇다 |

---

## 9. 응용

### 묶음을 이용한 특징 선택

```python
def grouped_feature_selection(X, y, feature_groups, alpha=0.5):
    """
    특징의 묶음을 존중하는 특징 선택.
    
    인수:
        X: 특징 행렬
        y: 목푯값
        feature_groups: 묶음 이름을 특징 인덱스에 대응시키는 사전
        alpha: 엘라스틱 넷의 l1_ratio
        
    반환값:
        선택된 특징 묶음과 모델
    """
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler
    
    # 벌점이 모든 계수에 같은 세기로 걸리므로, 특성의 자가 제각각이면
    # 정칙화의 세기도 제각각이 된다. 표준화가 앞서야 하는 까닭이다
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 엘라스틱 넷 적합.
    # l1_ratio가 L1과 L2의 배합을 정한다. 1이면 순수 라쏘, 0이면 순수 릿지다.
    # 0.5쯤이면 라쏘의 성긴 해를 얻으면서도, 서로 상관된 특성 가운데
    # 하나만 남기고 버리는 라쏘의 변덕을 L2가 눌러 준다.
    # CV가 붙은 클래스이므로 벌점의 세기(alpha)는 교차 검증으로 알아서 고른다
    model = ElasticNetCV(l1_ratio=alpha, cv=5)
    model.fit(X_scaled, y)

    # ── 묶음 선택 분석 ──────────────────────────────────────────────
    # 엘라스틱 넷 자체는 묶음을 모른다. 계수를 하나씩 따로 볼 뿐이다.
    # 여기서는 적합이 끝난 뒤 계수를 묶음별로 갈라 "이 묶음이 통째로
    # 살아남았는지, 흩어져 몇 개만 남았는지"를 사람이 읽는다.
    # 묶음을 통째로 넣거나 빼려면 group lasso 같은 다른 벌점이 필요하다
    selected_groups = {}
    for group_name, indices in feature_groups.items():
        group_coefs = model.coef_[indices]
        # 라쏘 성분 덕분에 계수가 정확히 0이 되므로 이 비교가 뜻을 갖는다.
        # 릿지만 썼다면 0에 가깝기만 할 뿐 0은 아니어서 이렇게 셀 수 없다
        n_selected = np.sum(group_coefs != 0)
        n_total = len(indices)
        selected_groups[group_name] = {
            'selected': n_selected,
            'total': n_total,
            'ratio': n_selected / n_total,   # 묶음이 얼마나 살아남았나
            # 살아남은 계수의 크기. ratio가 낮아도 이 값이 크면
            # 그 묶음에 소수의 강한 특성이 있다는 뜻이다
            'mean_coef': np.mean(np.abs(group_coefs))
        }

    return selected_groups, model
```

---

## 연습문제

**연습문제 1.**
엘라스틱 넷의 벌점을 쓰고, 혼합 매개변수 $\alpha$이 L1과 L2의 균형을 어떻게 조절하는지 설명하라.

??? success "연습문제 1 풀이"
    엘라스틱 넷은 $\alpha \in [0,1]$에 대해 $\Omega(w) = \alpha\|w\|_1 + \frac{1-\alpha}{2}\|w\|_2^2$이다. $\alpha = 1$이면 순수한 라쏘(L1), $\alpha = 0$이면 순수한 능선(L2)이다. 중간 값은 희소성(L1)과 묶음 선택(L2)을 결합한다.

---

**연습문제 2.**
엘라스틱 넷이 순수한 라쏘보다 상관된 특징을 더 잘 다루는 이유를 설명하라.

??? success "연습문제 2 풀이"
    상관된 특징이 있으면 라쏘는 하나를 임의로 고르고 나머지를 0으로 만든다(불안정한 선택). 엘라스틱 넷의 L2 항은 상관된 특징이 비슷한 계수를 갖도록 이끈다(묶음 효과). $x_i \approx x_j$이면 $|w_i - w_j|$이 벌점을 받아 둘 다 남는다.

---

**연습문제 3.**
PyTorch 학습 루프에서 엘라스틱 넷 정칙화를 구현하라.

??? success "연습문제 3 풀이"
    ```python
    alpha = 0.5  # 혼합 비율
    lam = 0.01   # 전체 강도
    l1 = sum(p.abs().sum() for p in model.parameters())
    l2 = sum(p.pow(2).sum() for p in model.parameters())
    reg = lam * (alpha * l1 + (1-alpha) * 0.5 * l2)
    loss = criterion(model(x), y) + reg
    ```

---

**연습문제 4.**
엘라스틱 넷 벌점에 대한 근접 연산자를 유도하라.

??? success "연습문제 4 풀이"
    근접 연산자는 분해된다. 먼저 L2 배율 조정 $\tilde{v} = v/(1 + \lambda(1-\alpha))$을 적용하고, 이어서 L1 연성 문턱화 $\text{prox}(v) = \text{sign}(\tilde{v})\max(|\tilde{v}| - \lambda\alpha, 0)$을 적용한다.

## 정리하며

이 마당은 수학적 정식화、왜 엘라스틱 넷인가?、기하학적 해석、PyTorch 구현을 차례로 짚었다.

**참고 문헌**

1. Zou, H., & Hastie, T. (2005). Regularization and Variable Selection via the Elastic Net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301-320.
2. Hastie, T., Tibshirani, R., & Wainwright, M. (2015). *Statistical Learning with Sparsity*. CRC Press.
3. Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization Paths for Generalized Linear Models via Coordinate Descent. *Journal of Statistical Software*, 33(1), 1-22.
