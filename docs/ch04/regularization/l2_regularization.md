# L2 정칙화 (능선)

능선 회귀 또는 가중치 감쇠라고도 하는 L2 정칙화는 모델 가중치의 크기의 제곱에 비례하는 벌점을 손실 함수에 더한다. L1 정칙화와 달리 L2는 작지만 영이 아닌 가중치를 이끌어 내며, 그 결과 어느 한 가중치도 지나치게 커지지 않는 매끄러운 가중치 분포를 얻는다.

---

## 1. 수학적 정식화

### L2 벌점을 더한 표준 손실

매개변수가 $\theta = \{w_1, w_2, \ldots, w_n\}$인 손실 함수 $\mathcal{L}(\theta)$에 대해 L2 정칙화는 목적 함수를 다음과 같이 바꾼다.

$$
\mathcal{L}_{\text{L2}}(\theta) = \mathcal{L}(\theta) + \lambda \sum_{i=1}^{n} w_i^2
$$

동등하게, L2 노름의 제곱을 쓰는 벡터 표기로 다음과 같다.

$$
\mathcal{L}_{\text{L2}}(\theta) = \mathcal{L}(\theta) + \lambda \|w\|_2^2
$$

여기서 각 기호는 다음과 같다.

- $\mathcal{L}(\theta)$은 원래의 손실 함수이다
- $\lambda \geq 0$은 정칙화의 강도이다
- $\|w\|_2^2 = w^T w = \sum_{i=1}^{n} w_i^2$은 L2 노름의 제곱이다

### L2를 쓰는 선형 회귀 (능선 회귀)

설계 행렬 $X \in \mathbb{R}^{m \times n}$, 목표 $y \in \mathbb{R}^m$, 가중치 $w \in \mathbb{R}^n$인 선형 회귀에 대해 다음이 성립한다.

$$
\mathcal{L}_{\text{Ridge}}(w) = \frac{1}{2m} \|Xw - y\|_2^2 + \lambda \|w\|_2^2
$$

펼치면 다음과 같다.

$$
\mathcal{L}_{\text{Ridge}}(w) = \frac{1}{2m} (Xw - y)^T(Xw - y) + \lambda w^T w
$$

### 기울기의 유도

L2 벌점은 어디서나 미분 가능하다.

$$
\frac{\partial}{\partial w_i} \left( \lambda \sum_{j=1}^{n} w_j^2 \right) = 2\lambda w_i
$$

벡터 형태로는 다음과 같다.

$$
\nabla_w \left( \lambda \|w\|_2^2 \right) = 2\lambda w
$$

L2 정칙화된 손실의 전체 기울기는 다음과 같다.

$$
\nabla_w \mathcal{L}_{\text{L2}} = \nabla_w \mathcal{L} + 2\lambda w
$$

### 능선 회귀의 닫힌 형태 해

기울기를 영으로 놓으면 다음과 같다.

$$
\nabla_w \mathcal{L}_{\text{Ridge}} = \frac{1}{m} X^T(Xw - y) + 2\lambda w = 0
$$

$w$에 대해 풀면 다음과 같다.

$$
\frac{1}{m} X^T X w + 2\lambda w = \frac{1}{m} X^T y
$$

$$
\left( \frac{1}{m} X^T X + 2\lambda I \right) w = \frac{1}{m} X^T y
$$

$$
w^* = \left( X^T X + 2m\lambda I \right)^{-1} X^T y
$$

**핵심**: 항 $2m\lambda I$은 $X^T X$이 특이행렬일 때에도(예: $n > m$일 때) 행렬이 언제나 가역이 되도록 보장한다.

---

## 2. 기하학적 해석

### 제약 영역

L2 정칙화는 L2 공 제약을 갖는 제약 최적화와 동등하다.

$$
\min_w \mathcal{L}(w) \quad \text{subject to} \quad \|w\|_2^2 \leq t
$$

2차원에서 L2 공은 원이다.

$$
\|w\|_2^2 = w_1^2 + w_2^2 \leq t
$$

### L2가 수축시키되 희소하게 만들지는 않는 이유

원형의 제약 영역에는 모서리가 없다. 손실 함수의 등고선은 보통 두 좌표가 모두 영이 아닌 점에서 원과 만난다. 그 결과는 다음과 같다.

- 모든 가중치가 영 쪽으로 수축한다
- 그러나 정확히 영이 되는 일은 드물다
- 매끄럽고 연속적인 가중치 분포를 얻는다

### 베이즈적 해석

L2 정칙화는 가중치에 대한 **정규 사전분포**에 대응한다.

$$
p(w) = \mathcal{N}(0, \sigma^2 I)
$$

정칙화의 강도는 사전분포의 분산과 다음 관계에 있다.

$$
\lambda = \frac{1}{2\sigma^2}
$$

이 사전분포를 쓰는 **MAP 추정**은 능선 회귀의 해를 준다.

$$
w_{\text{MAP}} = \arg\max_w \left[ \log p(y|X, w) + \log p(w) \right]
$$

---

## 3. 가중치 감쇠로서의 해석

### 경사 하강법과의 관계

경사 하강법에서 L2 벌점은 매 단계 가중치를 수축시키는 항을 더한다.

$$
w_{t+1} = w_t - \eta \nabla_w \mathcal{L} - 2\eta\lambda w_t
$$

정리하면 다음과 같다.

$$
w_{t+1} = (1 - 2\eta\lambda) w_t - \eta \nabla_w \mathcal{L}
$$

인수 $(1 - 2\eta\lambda)$이 현재 가중치에 곱해져 매 단계 가중치를 영 쪽으로 **감쇠**시킨다. 이것이 L2 정칙화를 흔히 **가중치 감쇠**라 부르는 까닭이다.

### AdamW: 분리된 가중치 감쇠

L2 정칙화를 쓰는 표준 Adam은 적응형 학습률과 가중치 감쇠를 완전히 분리하지 못한다. AdamW가 이를 바로잡는다.

```python
# L2를 쓰는 표준 Adam (이상적이지 않다)
gradient = grad + 2 * lambda * w
# 정칙화 항이 적응형 학습률로 배율 조정된다

# AdamW (제대로 된 가중치 감쇠)
w = w - lr * adam_step(grad) - lr * lambda * w
# 기울기를 거치지 않고 가중치 감쇠를 직접 적용한다
```

---

## 4. PyTorch 구현

### L2 정칙화 직접 구현하기

```python
import torch
import torch.nn as nn
import torch.optim as optim

def l2_regularization(model: nn.Module, lambda_l2: float) -> torch.Tensor:
    """
    L2 정칙화의 벌점을 계산한다.
    
    인수:
        model: 신경망 모델
        lambda_l2: 정칙화의 강도
        
    반환값:
        L2 벌점 항 (가중치의 L2 노름의 제곱)
    """
    # next(model.parameters())로 첫 매개변수를 꺼내 그 device를 따라간다.
    # 이렇게 해 두면 모델이 GPU에 있든 CPU에 있든 같은 곳에서 셈한다.
    # device를 맞추지 않으면 아래 덧셈에서 오류가 난다
    l2_penalty = torch.tensor(0., device=next(model.parameters()).device)

    for param in model.parameters():
        # += 가 아니라 = ... + ... 을 쓴다. += 는 제자리 연산이라
        # autograd가 되돌아갈 값을 덮어써 역전파가 깨질 수 있다
        l2_penalty = l2_penalty + torch.sum(param ** 2)

    # 주의: model.parameters()는 편향과 정규화 층의 감마·베타까지 모두 준다.
    # 보통은 가중치에만 벌을 주므로, 엄밀히 하려면 이름으로 걸러야 한다.
    # 편향에 벌을 주면 모델이 출력을 원점 쪽으로 끌어당기게 되어
    # 목표의 평균이 0에서 멀 때 손해다
    return lambda_l2 * l2_penalty

class L2RegularizedModel(nn.Module):
    """L2 정칙화 계산이 내장된 모델."""
    
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        
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
    
    def l2_penalty(self, lambda_l2=0.01):
        """모든 매개변수에 대한 L2 벌점을 계산한다."""
        # 파이썬 정수 0에서 시작한다. 첫 덧셈에서 텐서가 되므로
        # device를 따로 맞출 필요가 없다는 것이 앞의 함수와 다른 점이다.
        # 다만 매개변수가 하나도 없는 모델이면 텐서가 아니라 0이 나온다
        penalty = 0
        for param in self.parameters():
            penalty += torch.sum(param ** 2)

        # 쓸 때는 손실에 더한다: loss = criterion(pred, y) + model.l2_penalty()
        # 옵티마이저의 weight_decay로 거는 것과 수학적으로는 같지만,
        # 이렇게 손실에 명시하면 벌점 값을 따로 찍어 볼 수 있어
        # lambda를 고르거나 문제를 찾을 때 편하다
        return lambda_l2 * penalty
```

### 최적화기의 weight_decay 매개변수 쓰기

PyTorch의 최적화기에는 가중치 감쇠가 내장되어 있다.

```python
# 가중치 감쇠를 쓰는 SGD (L2 정칙화를 구현한다)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)

# 가중치 감쇠를 쓰는 Adam
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# AdamW - 제대로 분리된 가중치 감쇠 (권장)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

**참고**: `weight_decay` 매개변수는 가중치 감쇠를 다음과 같이 구현한다.

$$
w_{t+1} = w_t - \eta \nabla \mathcal{L} - \eta \cdot \text{weight\_decay} \cdot w_t
$$

이는 손실 식에서 $\lambda = \text{weight\_decay} / 2$인 L2와 동등하다.

### L2를 쓰는 완전한 학습 루프

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def train_with_l2_regularization(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lambda_l2: float = 0.01,
    epochs: int = 100,
    lr: float = 0.001,
    use_adamw: bool = True
) -> dict:
    """
    L2 정칙화로 모델을 학습시킨다.
    
    인수:
        model: 신경망
        train_loader: 학습 데이터
        val_loader: 검증 데이터
        lambda_l2: L2 정칙화의 강도
        epochs: 에포크 수
        lr: 학습률
        use_adamw: AdamW를 쓸지(True) 직접 L2를 더할지(False)
        
    반환값:
        학습 이력
    """
    criterion = nn.MSELoss()
    
    if use_adamw:
        # 최적화기에 내장된 가중치 감쇠 쓰기.
        # 2를 곱하는 것은 오타가 아니다. 위의 l2_regularization은
        # lambda * sum(w^2)을 돌려주므로 기울기가 2*lambda*w인데,
        # weight_decay는 갱신에 wd*w를 곧바로 더한다. 그래서 눈금을
        # 맞추려면 wd = 2*lambda여야 한다.
        # 다만 상수를 맞추어도 두 갈래가 똑같아지지는 않는다. 손실에
        # 더한 벌점은 Adam의 적응 눈금(sqrt(v)로 나누기)을 함께 거치지만
        # AdamW의 감쇠는 그 눈금을 거치지 않고 가중치에 직접 걸린다.
        # AdamW가 따로 있는 이유가 바로 이 분리다
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=2*lambda_l2)
        manual_l2 = False
    else:
        # L2 정칙화 직접 구현.
        # weight_decay를 주지 않은 맨 Adam이어야 한다. 여기에 weight_decay를
        # 또 주면 벌점이 두 번 걸린다
        optimizer = optim.Adam(model.parameters(), lr=lr)
        manual_l2 = True
    
    history = {'train_loss': [], 'val_loss': [], 'weight_norm': []}
    
    for epoch in range(epochs):
        # 학습 단계
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            
            if manual_l2:
                l2_penalty = l2_regularization(model, lambda_l2)
                total_loss = loss + l2_penalty
            else:
                total_loss = loss
            
            total_loss.backward()
            optimizer.step()
            # 역전파는 total_loss로 하되 기록은 loss로 한다. 벌점을 뺀
            # 값이라야 lambda_l2가 다른 실험끼리, 또 use_adamw의 두
            # 갈래끼리 손실 곡선을 견줄 수 있다
            train_loss += loss.item()
        
        # 검증 단계
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                val_loss += criterion(predictions, y_batch).item()
        
        # 가중치 통계 계산
        # 층별 노름을 제곱해 더한 뒤 제곱근을 취해야 전체 노름이 된다.
        # 노름을 그냥 더하면 다른 값이 나온다.
        # 이 값이 정칙화가 듣고 있는지 보는 눈금이다. lambda_l2를 키우면
        # 이 곡선이 더 낮은 자리에서 평평해져야 한다
        total_norm = sum(p.norm().item() ** 2 for p in model.parameters()) ** 0.5
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['weight_norm'].append(total_norm)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Train={train_loss/len(train_loader):.4f}, "
                  f"Val={val_loss/len(val_loader):.4f}, ||w||={total_norm:.4f}")
    
    return history
```

### 선택적 L2 정칙화

층마다 서로 다른 정칙화 강도를 적용한다.

```python
def create_param_groups_with_l2(model, base_lr=0.001, 
                                 layer_decay_rates=None):
    """
    층마다 다른 L2 정칙화를 갖는 매개변수 묶음을 만든다.
    
    인수:
        model: 신경망
        base_lr: 기본 학습률
        layer_decay_rates: 층 이름을 가중치 감쇠 값에 대응시키는 사전
        
    반환값:
        최적화기를 위한 매개변수 묶음의 목록
    """
    if layer_decay_rates is None:
        layer_decay_rates = {}
    
    param_groups = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # 층에 맞는 감쇠율 찾기
        weight_decay = 0.01  # 기본값
        for layer_name, decay in layer_decay_rates.items():
            if layer_name in name:
                weight_decay = decay
                break
        
        # 편향은 정칙화하지 않는다 (흔한 관행)
        if 'bias' in name:
            weight_decay = 0.0
        
        param_groups.append({
            'params': param,
            'lr': base_lr,
            'weight_decay': weight_decay
        })
    
    return param_groups

# 사용 예
model = L2RegularizedModel(input_dim=20, hidden_dims=[128, 64], output_dim=1)

# 층마다 다른 정칙화
layer_decays = {
    'network.0': 0.001,  # 첫 층: 가벼운 정칙화
    'network.2': 0.01,   # 둘째 층: 중간 정칙화
    'network.4': 0.1,    # 출력층: 강한 정칙화
}

param_groups = create_param_groups_with_l2(model, layer_decay_rates=layer_decays)
optimizer = optim.AdamW(param_groups)
```

---

## 5. scikit-learn 구현

```python
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
import numpy as np

def ridge_regression_analysis(X, y, alphas=None):
    """
    정칙화의 강도를 달리하며 능선 회귀를 분석한다.
    
    인수:
        X: 특징 행렬
        y: 목표 벡터
        alphas: 시도해 볼 정칙화 값
        
    반환값:
        최적 모델과 분석 결과
    """
    # 특징을 표준화한다.
    # 능선 회귀에서는 이것이 선택이 아니라 필수다. 벌점이 계수의
    # 크기를 보고 매겨지는데, 눈금이 큰 특징일수록 계수가 작아지므로
    # 표준화하지 않으면 단위를 무엇으로 쟀느냐가 정칙화의 세기를
    # 좌우해 버린다
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if alphas is None:
        alphas = np.logspace(-4, 4, 50)
    
    # 최적 alpha를 찾기 위한 교차 검증.
    # RidgeCV는 alpha마다 따로 적합시키는 것이 아니라 능선 해의
    # 닫힌 형태를 이용해 한꺼번에 푼다. 그래서 GridSearchCV로 같은 일을
    # 하는 것보다 훨씬 빠르다
    ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_squared_error')
    ridge_cv.fit(X_scaled, y)
    
    print(f"Optimal alpha: {ridge_cv.alpha_:.6f}")
    print(f"R² score: {ridge_cv.score(X_scaled, y):.4f}")
    print(f"Coefficient range: [{ridge_cv.coef_.min():.4f}, {ridge_cv.coef_.max():.4f}]")
    print(f"Coefficient L2 norm: {np.linalg.norm(ridge_cv.coef_):.4f}")
    
    return ridge_cv, scaler

def plot_ridge_coefficients(X, y, alphas):
    """정칙화의 강도에 따라 계수가 어떻게 바뀌는지 시각화한다."""
    import matplotlib.pyplot as plt
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    coefs = []
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_scaled, y)
        coefs.append(ridge.coef_)
    
    coefs = np.array(coefs)
    
    plt.figure(figsize=(10, 6))
    for i in range(coefs.shape[1]):
        plt.plot(alphas, coefs[:, i], label=f'Feature {i}')
    
    plt.xscale('log')
    plt.xlabel('Regularization strength (α)')
    plt.ylabel('Coefficient value')
    plt.title('Ridge Coefficient Shrinkage')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.legend(loc='best', fontsize=8)
    
    return coefs
```

---

## 6. 특잇값 분해의 관점

능선 회귀는 SVD를 통해 우아하게 해석된다. $X = U \Sigma V^T$이면 다음과 같다.

$$
w_{\text{Ridge}} = V D_\lambda \Sigma^{-1} U^T y
$$

여기서 $D_\lambda$은 성분이 다음과 같은 대각행렬이다.

$$
d_i = \frac{\sigma_i^2}{\sigma_i^2 + \lambda}
$$

**해석**: 능선 회귀는 특잇값이 작은 방향의 계수를 더 세게 수축시킨다. 이는 강한 방향의 신호는 지키면서 잡음이 많은 방향에서의 과적합을 막는다.

```python
def ridge_via_svd(X, y, lambda_reg):
    """
    SVD로 능선 회귀의 해를 계산한다.
    
    이는 능선 회귀가 특잇값 방향마다 어떻게
    수축시키는지 드러낸다.
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T
    
    # 수축 인수
    d = s ** 2 / (s ** 2 + lambda_reg)
    
    # 능선 회귀의 해
    w_ridge = V @ np.diag(d / s) @ U.T @ y
    
    return w_ridge, d, s
```

---

## 7. 실효 자유도

능선 회귀의 실효 자유도는 다음과 같다.

$$
\text{df}_\lambda = \sum_{i=1}^{n} \frac{\sigma_i^2}{\sigma_i^2 + \lambda} = \text{tr}(X(X^TX + \lambda I)^{-1}X^T)
$$

이는 모델의 복잡도를 잰다. $\lambda \to 0$이면 $\text{df} \to n$(보통최소제곱), $\lambda \to \infty$이면 $\text{df} \to 0$이다.

---

## 8. 비교: L2와 L1

| 항목 | L2 (능선) | L1 (라쏘) |
|--------|------------|------------|
| 벌점 | $\lambda \sum w_i^2$ | $\lambda \sum \|w_i\|$ |
| 제약의 모양 | 원/구 | 마름모/교차다포체 |
| 희소한 해 | 아니다 | 그렇다 |
| 닫힌 형태 | 있음 | 없음 |
| 미분 가능 | 그렇다 | 아니다 (0에서) |
| 상관된 특징 | 가중치를 고르게 나눈다 | 하나를 고른다 |
| 베이즈 사전분포 | 정규분포 | 라플라스분포 |

---

## 9. L2 정칙화를 쓸 때

### 알맞은 쓰임새

1. **큰 가중치를 막을 때**: 극단적인 가중치가 불안정을 일으킬 때
2. **상관된 특징**: L2는 다중공선성을 매끄럽게 다룬다
3. **모든 특징이 유의미할 때**: 모든 특징이 기여한다고 믿을 때
4. **수치적 안정성**: $\lambda I$을 더하면 가역성이 보장된다
5. **딥러닝**: 신경망의 표준 정칙화 방법이다

### 초매개변수 선택

```python
from sklearn.model_selection import GridSearchCV

def select_optimal_l2(model_class, X, y, param_grid, cv=5):
    """
    격자 탐색으로 최적의 L2 정칙화 강도를 고른다.
    
    인수:
        model_class: 모델 클래스 (예: Ridge)
        X: 특징
        y: 목푯값
        param_grid: 매개변수 격자 (예: {'alpha': [0.01, 0.1, 1.0]})
        cv: 교차 검증의 겹 수
        
    반환값:
        가장 좋은 모델과 탐색 결과
    """
    grid_search = GridSearchCV(
        model_class(),   # 아직 적합하지 않은 빈 모델. 격자의 값마다 새로 만든다
        param_grid,
        cv=cv,
        # sklearn의 scoring은 "클수록 좋다"는 규약이므로 MSE에 음수를 붙인 이름을 쓴다
        scoring='neg_mean_squared_error',
        # 훈련 점수도 함께 남긴다. 검증 점수만 보면 과적합인지 과소적합인지
        # 가릴 수 없다. 훈련은 좋은데 검증이 나쁘면 alpha를 키워야 하고,
        # 둘 다 나쁘면 alpha를 줄이거나 모델을 바꿔야 한다
        return_train_score=True
    )

    # fit 안에서 격자의 값마다 cv겹 교차검증을 돌린 뒤,
    # 가장 좋은 값으로 전체 자료에 다시 적합한다(refit이 기본값 True)
    grid_search.fit(X, y)

    print(f"Best parameters: {grid_search.best_params_}")
    # 음수를 씌워 원래의 MSE로 되돌려 찍는다
    print(f"Best CV score: {-grid_search.best_score_:.4f}")

    # 곡선 전체(cv_results_)를 함께 돌려주는 까닭은, 최적값 하나보다
    # alpha에 따라 점수가 어떻게 움직이는지가 더 많은 것을 말해 주기 때문이다
    return grid_search.best_estimator_, grid_search.cv_results_
```

---

## 10. 실무 지침

### 정칙화 강도의 선택

- **너무 작으면($\lambda \to 0$)**: 정칙화하지 않은 모델처럼 행동하여 과적합할 수 있다
- **너무 크면($\lambda \to \infty$)**: 모든 가중치가 영으로 수축하여 과소적합한다
- **최적**: 편향-분산 절충의 균형을 맞춘다

### 특징의 척도 조정

**L2 정칙화를 적용하기 전에 언제나 특징을 표준화하라.** 벌점은 모든 가중치를 똑같이 다루므로, 척도가 다른 특징은 불공평하게 벌점을 받게 된다.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

# 옳은 방법: 척도를 맞춘 뒤 정칙화한다.
#
# 왜 순서가 중요한가.
# L2 벌점은 모든 계수를 같은 세기로 누른다. 그런데 특성의 자가 제각각이면
# 자가 큰 특성일수록 계수가 작아지므로 벌을 덜 받는다. 곧 정칙화의 세기가
# 특성의 단위에 따라 달라져 버린다. 표준화를 먼저 해 두면 모든 계수가
# 같은 자 위에 놓여 alpha 하나가 모두에 똑같이 걸린다.
#
# Pipeline으로 묶는 까닭.
# 교차 검증을 할 때 겹(fold)마다 훈련 자료로만 평균과 표준편차를 다시 잰다.
# 손으로 먼저 표준화해 두면 검증 겹의 정보가 훈련에 새어 들어
# 성능이 실제보다 좋게 나온다.
pipeline = Pipeline([
    ('scaler', StandardScaler()),   # 1단계: 각 특성을 평균 0, 표준편차 1로
    ('ridge', Ridge(alpha=1.0))     # 2단계: 그 위에서 L2 벌점을 건다
])
```

### 편향 항

**편향 항은 정칙화하지 마라.** 편향(절편)은 예측을 옮길 뿐이므로 벌점을 주지 않아야 한다.

```python
# PyTorch에서는 가중치와 편향을 나눈다
def l2_regularization_weights_only(model, lambda_l2):
    """편향은 빼고 가중치에만 적용하는 L2 벌점."""
    penalty = 0
    for name, param in model.named_parameters():
        if 'weight' in name:  # 가중치 행렬만
            penalty += torch.sum(param ** 2)
    return lambda_l2 * penalty
```

---

## 연습문제

**연습문제 1.**
L2 정칙화가 SGD에서는 가중치 감쇠와 동등하지만 Adam에서는 그렇지 않음을 보여라.

??? success "연습문제 1 풀이"
    SGD에서는 $w \leftarrow w - \eta(\nabla L + \lambda w) = (1-\eta\lambda)w - \eta\nabla L$이며, 인수 $(1-\eta\lambda)$이 가중치 감쇠이다. Adam에서는 적응형 학습률 때문에 $\lambda w$도 이차 모멘트 추정값으로 나뉘므로 L2 정칙화와 가중치 감쇠가 동등하지 않게 된다. AdamW가 이를 바로잡는다.

---

**연습문제 2.**
L2 정칙화를 가중치에 대한 정규 사전분포로 보는 베이즈적 해석을 유도하라.

??? success "연습문제 2 풀이"
    사전분포 $w \sim \mathcal{N}(0, \sigma_w^2)$을 쓰는 MAP는 $\log p(w|D) \propto \log p(D|w) + \log p(w) = -L(w) - \frac{\|w\|^2}{2\sigma_w^2}$이다. $\lambda = 1/(2\sigma_w^2)$으로 두면 L2 정칙화를 얻는다.

---

**연습문제 3.**
L2 정칙화가 가중치를 영 쪽으로 수축시키되 결코 정확히 영으로 만들지 않는 이유를 기하적으로 설명하라.

??? success "연습문제 3 풀이"
    L2의 제약 영역은 구이다. $\|w\|^2$의 기울기는 $2w$으로 언제나 반지름 방향 바깥을 가리킨다. 벌점은 $w$에 비례하여 영 쪽으로 향하는 연속적인 힘을 만들지만, 이 힘은 $w \to 0$일 때 사라지므로 가중치는 영에 점근적으로 다가갈 뿐 결코 닿지 않는다.

---

**연습문제 4.**
PyTorch에서 `weight_decay` 매개변수를 쓰는 방법과 벌점을 직접 더하는 방법 모두로 L2 정칙화를 구현하라.

??? success "연습문제 4 풀이"
    ```python
    # 방법 1: weight_decay
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
    # 방법 2: 직접 구현
    loss = criterion(model(x), y)
    l2_penalty = sum(p.pow(2).sum() for p in model.parameters())
    loss = loss + 0.01 * l2_penalty
    loss.backward()
    ```

## 정리하며

이 마당은 수학적 정식화、기하학적 해석、가중치 감쇠로서의 해석、PyTorch 구현을 차례로 짚었다.

**참고 문헌**

1. Hoerl, A. E., & Kennard, R. W. (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems. *Technometrics*, 12(1), 55-67.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
3. Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR 2019*.
