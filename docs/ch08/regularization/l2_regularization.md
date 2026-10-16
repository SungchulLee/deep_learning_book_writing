# L2 정규화 (릿지)

L2 정규화는 웨이트 제곱 노름의 패널티를 로스에 더한다. 선형 최소제곱에서는 릿지 회귀가 되며, 적절한 조건에서 계수의 노름과 추정 변동을 줄일 수 있다. 웨이트 감쇠와의 관계는 옵티마이저와 패널티의 배율에 따라 달라진다.

## 1. 수학적 정식화

### L2 패널티를 더한 표준 로스

파라메터가 $\theta = \{w_1, w_2, \ldots, w_n\}$인 로스 함수 $\mathcal{L}(\theta)$에 대해 L2 정규화는 목적 함수를 다음과 같이 바꾼다.

$$
\mathcal{L}_{\text{L2}}(\theta) = \mathcal{L}(\theta) + \lambda \sum_{i=1}^{n} w_i^2
$$

동등하게, L2 노름의 제곱을 쓰는 벡터 표기로 다음과 같다.

$$
\mathcal{L}_{\text{L2}}(\theta) = \mathcal{L}(\theta) + \lambda \|w\|_2^2
$$

여기서 각 기호는 다음과 같다.

- $\mathcal{L}(\theta)$은 원래의 로스 함수이다
- $\lambda \geq 0$은 정규화의 강도이다
- $\|w\|_2^2 = w^T w = \sum_{i=1}^{n} w_i^2$은 L2 노름의 제곱이다

### L2를 쓰는 선형 회귀 (릿지 회귀)

설계 행렬 $X \in \mathbb{R}^{m \times n}$, 목표 $y \in \mathbb{R}^m$, 웨이트 $w \in \mathbb{R}^n$인 선형 회귀에 대해 다음이 성립한다.

$$
\mathcal{L}_{\text{Ridge}}(w) = \frac{1}{2m} \|Xw - y\|_2^2 + \lambda \|w\|_2^2
$$

펼치면 다음과 같다.

$$
\mathcal{L}_{\text{Ridge}}(w) = \frac{1}{2m} (Xw - y)^T(Xw - y) + \lambda w^T w
$$

### 그래디언트의 유도

L2 패널티는 어디서나 미분 가능하다.

$$
\frac{\partial}{\partial w_i} \left( \lambda \sum_{j=1}^{n} w_j^2 \right) = 2\lambda w_i
$$

벡터 형태로는 다음과 같다.

$$
\nabla_w \left( \lambda \|w\|_2^2 \right) = 2\lambda w
$$

L2 정규화된 로스의 전체 그래디언트는 다음과 같다.

$$
\nabla_w \mathcal{L}_{\text{L2}} = \nabla_w \mathcal{L} + 2\lambda w
$$

### 릿지 회귀의 닫힌 형태 해

그래디언트를 영으로 놓으면 다음과 같다.

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

**핵심**: 모든 계수를 벌점에 포함하고 $\lambda>0$이면 항 $2m\lambda I$은 $X^T X$이 특이행렬일 때에도(예: $n > m$일 때) 행렬이 언제나 가역이 되도록 보장한다.

---

## 2. 기하학적 해석

### 제약 영역

적절한 볼록성·정규성 조건에서 L2 패널티와 노름 제약의 최적해를 대응시킬 수 있다.

$$
\min_w \mathcal{L}(w) \quad \text{subject to} \quad \|w\|_2^2 \leq t
$$

2차원에서 L2 공은 원이다.

$$
\|w\|_2^2 = w_1^2 + w_2^2 \leq t
$$

### L2가 수축시키되 희소하게 만들지는 않는 이유

원형의 제약 영역에는 모서리가 없다. 로스 함수의 등고선은 보통 두 좌표가 모두 영이 아닌 점에서 원과 만난다. 그 결과는 다음과 같다.

- 전체 노름은 줄어드는 경향이 있지만 상관된 피처에서 모든 개별 계수의 절댓값이 반드시 줄지는 않는다
- 그러나 정확히 영이 되는 일은 드물다
- 계수의 정확한 확률분포를 정해 주는 것은 아니다

### 베이즈적 해석

L2 정규화는 웨이트에 대한 **정규 사전분포**에 대응한다.

$$
p(w) = \mathcal{N}(0, \sigma^2 I)
$$

데이터 로스를 **합 NLL**로 정의하면 사전 분산과 아래 관계를 갖는다. 평균 NLL이라면 표본 수 $m$으로 나눈 $\lambda=1/(2m\sigma^2)$가 같은 사전에 대응한다. 앞의 제곱 오차를 정규 우도로 읽으려면 관측 잡음 분산과 로스 배율도 맞춰야 한다.

$$
\lambda = \frac{1}{2\sigma^2}
$$

이 사전분포를 쓰는 **MAP 추정**은 릿지 회귀의 해를 준다.

$$
w_{\text{MAP}} = \arg\max_w \left[ \log p(y|X, w) + \log p(w) \right]
$$

---

## 3. 웨이트 감쇠로서의 해석

### 경사 하강법과의 관계

경사 하강법에서 L2 패널티는 매 단계 웨이트를 수축시키는 항을 더한다.

$$
w_{t+1} = w_t - \eta \nabla_w \mathcal{L} - 2\eta\lambda w_t
$$

정리하면 다음과 같다.

$$
w_{t+1} = (1 - 2\eta\lambda) w_t - \eta \nabla_w \mathcal{L}
$$

인수 $(1 - 2\eta\lambda)$이 현재 웨이트에 곱해져 매 단계 웨이트를 영 쪽으로 **감쇠**시킨다. 이것이 L2 정규화를 흔히 **웨이트 감쇠**라 부르는 까닭이다.

### AdamW: 분리된 웨이트 감쇠

L2 정규화를 쓰는 표준 Adam은 적응형 러닝레이트와 웨이트 감쇠를 완전히 분리하지 못한다. AdamW가 이를 바로잡는다.

```python
# lambda는 파이썬의 예약어라 변수 이름으로 쓸 수 없다. lam으로 적는다.

# L2를 쓰는 표준 Adam (이상적이지 않다)
gradient = grad + 2 * lam * w
# 정칙화 항이 적응형 러닝레이트로 배율 조정된다

# AdamW (제대로 된 웨이트 감쇠)
w = w - lr * adam_step(grad) - lr * lam * w
# 그래디언트를 거치지 않고 웨이트 감쇠를 직접 적용한다
```

---

## 4. PyTorch 구현

### L2 정규화 직접 구현하기

```python
import torch
import torch.nn as nn
import torch.optim as optim

def l2_regularization(model: nn.Module, lambda_l2: float) -> torch.Tensor:
    """
    L2 정칙화의 벌점을 계산한다.
    
    인수:
        model: 뉴럴 넷트웍 모델
        lambda_l2: 정칙화의 강도
        
    반환값:
        L2 벌점 항 (웨이트의 L2 노름의 제곱)
    """
    # next(model.parameters())로 첫 파라메터를 꺼내 그 device를 따라간다.
    # 이렇게 해 두면 모델이 GPU에 있든 CPU에 있든 같은 곳에서 셈한다.
    # device를 맞추지 않으면 아래 덧셈에서 오류가 난다
    l2_penalty = torch.tensor(0., device=next(model.parameters()).device)

    for param in model.parameters():
        # += 가 아니라 = ... + ... 을 쓴다. += 는 제자리 연산이라
        # autograd가 되돌아갈 값을 덮어써 역전파가 깨질 수 있다
        l2_penalty = l2_penalty + torch.sum(param ** 2)

    # 주의: model.parameters()는 바이어스와 정규화 레이어의 감마·베타까지 모두 준다.
    # 보통은 웨이트에만 벌을 주므로, 엄밀히 하려면 이름으로 걸러야 한다.
    # 바이어스에 벌을 주면 모델이 출력을 원점 쪽으로 끌어당기게 되어
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
        """모든 파라메터에 대한 L2 벌점을 계산한다."""
        # 파이썬 정수 0에서 시작한다. 첫 덧셈에서 텐서가 되므로
        # device를 따로 맞출 필요가 없다는 것이 앞의 함수와 다른 점이다.
        # 다만 파라메터가 하나도 없는 모델이면 텐서가 아니라 0이 나온다
        penalty = 0
        for param in self.parameters():
            penalty += torch.sum(param ** 2)

        # 쓸 때는 로스에 더한다: loss = criterion(pred, y) + model.l2_penalty()
        # 옵티마이저의 weight_decay로 거는 것과 수학적으로는 같지만,
        # 이렇게 로스에 명시하면 벌점 값을 따로 찍어 볼 수 있어
        # lambda를 고르거나 문제를 찾을 때 편하다
        return lambda_l2 * penalty
```

### 옵티마이저의 weight_decay 파라메터 쓰기

PyTorch의 옵티마이저에는 웨이트 감쇠가 내장되어 있다.

```python
# 웨이트 감쇠를 쓰는 SGD (L2 정칙화를 구현한다)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)

# 웨이트 감쇠를 쓰는 Adam
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# AdamW - 제대로 분리된 웨이트 감쇠 (권장)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

아래 식은 모멘텀 없는 일반 SGD에서의 감쇠를 나타낸다. 옵티마이저마다 weight_decay가 적응형 경사 안에 결합되는지, 별도로 적용되는지가 다르므로 이름만으로 같은 갱신이라고 가정하지 않는다.

$$
w_{t+1} = w_t - \eta \nabla \mathcal{L} - \eta \cdot \text{weight\_decay} \cdot w_t
$$

이는 모멘텀이 없는 일반 SGD에서, 같은 데이터 로스에 `weight_decay/2`를 계수로 한 제곱 노름 패널티를 더하는 것과 동등하다. AdamW 같은 분리형 감쇠의 일반적인 동등식은 아니다.

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
        model: 뉴럴 넷트웍
        train_loader: 학습 데이터
        val_loader: 검증 데이터
        lambda_l2: L2 정칙화의 강도
        epochs: 에포크 수
        lr: 러닝레이트
        use_adamw: AdamW를 쓸지(True) 직접 L2를 더할지(False)
        
    반환값:
        학습 이력
    """
    criterion = nn.MSELoss()
    
    if use_adamw:
        # 최적화기에 내장된 웨이트 감쇠 쓰기.
        # 2를 곱하는 것은 오타가 아니다. 위의 l2_regularization은
        # lambda * sum(w^2)을 돌려주므로 기울기가 2*lambda*w인데,
        # weight_decay는 갱신에 wd*w를 곧바로 더한다. 그래서 눈금을
        # 맞추려면 wd = 2*lambda여야 한다.
        # 다만 상수를 맞추어도 두 클래스가 똑같아지지는 않는다. 로스에
        # 더한 벌점은 Adam의 적응 눈금(sqrt(v)로 나누기)을 함께 거치지만
        # AdamW의 감쇠는 그 눈금을 거치지 않고 웨이트에 직접 걸린다.
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
            # 클래스끼리 로스 곡선을 견줄 수 있다
            train_loss += loss.item()
        
        # 검증 단계
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                val_loss += criterion(predictions, y_batch).item()
        
        # 웨이트 통계 계산
        # 레이어별 노름을 제곱해 더한 뒤 제곱근을 취해야 전체 노름이 된다.
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

### 선택적 L2 정규화

레이어마다 서로 다른 정규화 강도를 적용한다.

```python
def create_param_groups_with_l2(model, base_lr=0.001, 
                                 layer_decay_rates=None):
    """
    레이어마다 다른 L2 정칙화를 갖는 파라메터 배치를 만든다.
    
    인수:
        model: 뉴럴 넷트웍
        base_lr: 기본 러닝레이트
        layer_decay_rates: 레이어 이름을 웨이트 감쇠 값에 대응시키는 사전
        
    반환값:
        최적화기를 위한 파라메터 배치의 목록
    """
    if layer_decay_rates is None:
        layer_decay_rates = {}
    
    param_groups = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # 레이어에 맞는 감쇠율 찾기.
        # name은 'network.0.weight' 같은 꼴이라, 사전의 열쇠가 그 안에
        # 들어 있기만 하면 걸린다. 부분 문자열로 맞추는 방식이라
        # 'fc1'이 'fc10'에도 걸린다는 점을 조심하라.
        # break가 있어 먼저 걸리는 것 하나만 쓰는데, 사전의 차례가
        # 곧 우선순위가 되므로 좁은 이름을 앞에 두어야 한다
        weight_decay = 0.01  # 기본값
        for layer_name, decay in layer_decay_rates.items():
            if layer_name in name:
                weight_decay = decay
                break
        
        # 바이어스는 정칙화하지 않는다 (흔한 관행).
        # 위의 레이어별 설정보다 나중에 오므로 무엇을 지정했든 이 규칙이
        # 이긴다. 바이어스는 과적합의 원인이 아니라 출력의 중심을 옮기는
        # 자유도라 눌러 둘 까닭이 없다.
        # 이 페이지 앞머리의 l2_regularization 함수가 model.parameters()로
        # 바이어스까지 훑던 문제를 여기서는 이름을 보고 갈라 해결한다
        if 'bias' in name:
            weight_decay = 0.0
        
        # 파라미터 하나에 배치 하나를 만든다. 최적화기는 배치마다 다른
        # 설정을 지닐 수 있으므로, 이렇게 잘게 나누면 레이어는 물론 웨이트와
        # 바이어스까지 따로 다룰 수 있다. 배치 수가 늘어도 셈이 느려지지는
        # 않는다. 어차피 파라미터마다 갱신하기 때문이다.
        # 이 목록을 optim.AdamW(param_groups)처럼 넘겨 쓴다
        param_groups.append({
            'params': param,
            'lr': base_lr,
            'weight_decay': weight_decay
        })
    
    return param_groups

# 사용 예
model = L2RegularizedModel(input_dim=20, hidden_dims=[128, 64], output_dim=1)

# 레이어마다 다른 정칙화
layer_decays = {
    'network.0': 0.001,  # 첫 레이어: 가벼운 정칙화
    'network.2': 0.01,   # 둘째 레이어: 중간 정칙화
    'network.4': 0.1,    # 출력 레이어: 강한 정칙화
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
        X: 피처 행렬
        y: 목표 벡터
        alphas: 시도해 볼 정칙화 값
        
    반환값:
        최적 모델과 분석 결과
    """
    # 피처를 표준화한다.
    # 능선 회귀에서는 이것이 선택이 아니라 필수다. 벌점이 계수의
    # 크기를 보고 매겨지는데, 눈금이 큰 피처일수록 계수가 작아지므로
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

릿지 회귀는 SVD를 통해 우아하게 해석된다. $X = U \Sigma V^T$이면 다음과 같다.

$$
w_{\text{Ridge}}=V\operatorname{diag}\!\left(\frac{\sigma_i}{\sigma_i^2+a}\right)U^\top y,\qquad a=2m\lambda
$$

여기서 $D_\lambda$은 성분이 다음과 같은 대각행렬이다.

$$
d_i=\frac{\sigma_i^2}{\sigma_i^2+a}
$$

**해석**: 릿지 회귀는 특잇값이 작은 방향의 계수를 더 세게 수축시킨다. 이는 강한 방향의 신호는 지키면서 잡음이 많은 방향에서의 과적합을 막는다.

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

릿지 회귀의 실효 자유도는 다음과 같다.

$$
\operatorname{df}=\sum_i\frac{\sigma_i^2}{\sigma_i^2+a}=\operatorname{tr}\!\left[X(X^\top X+aI)^{-1}X^\top\right],\qquad a=2m\lambda
$$

이 값은 전체 계수를 벌점에 포함한 선형 모형의 실효 자유도이다. $\lambda\to0^+$이면 $\operatorname{rank}(X)$로, $\lambda\to\infty$이면 0으로 수렴한다. 절편을 제외한 모형은 절편의 자유도를 별도로 처리한다.

---

## 8. 비교: L2와 L1

| 항목 | L2 (릿지) | L1 (라쏘) |
|--------|------------|------------|
| 패널티 | $\lambda \sum w_i^2$ | $\lambda \sum \|w_i\|$ |
| 제약의 모양 | 원/구 | 마름모/교차다포체 |
| 희소한 해 | 직접 유도하지 않음 | 선호함 |
| 선형 최소제곱의 해 | 양의 강도에서 닫힌 해 | 일반적으로 반복 계산. 특수한 설계에서는 닫힌 해 |
| 미분 가능 | 그렇다 | 아니다 (0에서) |
| 상관된 피처 | 계수를 분산시키는 경향 | 선택이 불안정할 수 있음 |
| 베이즈 사전분포 | 정규분포 | 라플라스분포 |

---

## 9. L2 정규화를 쓸 때

### 알맞은 쓰임새

1. **큰 웨이트를 막을 때**: 극단적인 웨이트가 불안정을 일으킬 때
2. **상관된 피처**: L2는 다중공선성을 매끄럽게 다룬다
3. **모든 피처가 유의미할 때**: 모든 피처가 기여한다고 믿을 때
4. **수치적 안정성**: $\lambda I$을 더하면 가역성이 보장된다
5. **딥러닝**: 뉴럴 넷트웍의 표준 정규화 방법이다

### 하이퍼파라미터 선택

```python
from sklearn.model_selection import GridSearchCV

def select_optimal_l2(model_class, X, y, param_grid, cv=5):
    """
    격자 탐색으로 최적의 L2 정칙화 강도를 고른다.
    
    인수:
        model_class: 모델 클래스 (예: Ridge)
        X: 피처
        y: 목푯값
        param_grid: 파라메터 격자 (예: {'alpha': [0.01, 0.1, 1.0]})
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

### 정규화 강도의 선택

- **너무 작으면($\lambda \to 0$)**: 정규화하지 않은 모델처럼 액션하여 과적합할 수 있다
- **너무 크면($\lambda \to \infty$)**: 모든 웨이트가 영으로 수축하여 과소적합한다
- **최적**: 편향-분산 절충의 균형을 맞춘다

### 피처의 척도 조정

같은 계수 크기에 같은 패널티를 주면 피처 단위가 다른 경우 효과도 달라진다. 표준화는 흔한 선택이지만, 단위에 맞는 사전 지식이나 피처별 패널티가 있다면 그 설계를 사용할 수 있다. 전처리 통계량은 학습 분할에서만 구한다.

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

### 바이어스 항

선형 회귀에서는 흔히 절편을 패널티에서 제외한다. 뉴럴 넷트웍의 바이어스와 정규화 레이어 계수도 감쇠에서 제외하는 경우가 있지만, 과제와 구조에 따른 선택이다. 어떤 파라메터가 패널티에 포함되는지 명시하고 비교한다.

```python
# PyTorch에서는 웨이트와 바이어스를 나눈다
def l2_regularization_weights_only(model, lambda_l2):
    """바이어스는 빼고 웨이트에만 적용하는 L2 벌점."""
    penalty = 0
    for name, param in model.named_parameters():
        if 'weight' in name:  # 웨이트 행렬만
            penalty += torch.sum(param ** 2)
    return lambda_l2 * penalty
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
패널티가 $\lambda\|w\|^2$일 때 모멘텀 없는 SGD의 갱신을 구하고 AdamW와 구분하라.

</div>

??? success "연습문제 1 풀이"
    일반 SGD는 $w^+=(1-2\eta\lambda)w-\eta\nabla L$이므로 감쇠 계수 $2\lambda$에 대응한다. Adam에 L2를 결합하면 패널티 경사도 적응형 적률 추정에 참여한다. AdamW는 감쇠를 그 변환과 분리하므로 같은 L2 목적의 Adam 갱신과 일반적으로 다르다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
L2 패널티의 가우시안 사전 해석에서 데이터 로스의 집계 방식이 중요한 이유를 설명하라.

</div>

??? success "연습문제 2 풀이"
    분산 $s^2$의 독립 가우시안 사전은 $\|w\|^2/(2s^2)$의 음의 로그 사전을 준다. 합 NLL에서는 계수가 $1/(2s^2)$이며, 평균 NLL로 전체 MAP 목적을 나누면 $1/(2ms^2)$이다. 관측 분산이나 MSE의 상수배도 대응 계수에 영향을 준다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
L2가 희소성을 직접 유도하지 않는 이유와 계수가 정확히 0일 수 있는 예를 들어라.

</div>

??? success "연습문제 3 풀이"
    L2는 0에서 매끄럽고 경사가 $2\lambda w$이므로 L1처럼 0을 선호하는 부분미분 구간이 없다. 그러나 단일 직교 피처가 목표와 직교하면 릿지 계수 $x^\top y/(x^\top x+a)$가 정확히 0이다. 따라서 결코 0이 되지 않는다는 주장은 틀리다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
PyTorch에서 `weight_decay` 파라메터를 쓰는 방법과 패널티를 직접 더하는 방법 모두로 L2 정규화를 구현하라.

</div>

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

L2는 제곱 노름 패널티이며 릿지의 축소 효과를 특잇값 방향에서 분석할 수 있다. 평균 제곱 오차의 배율을 바꾸면 행렬에 더하는 계수도 달라진다. 웨이트 감쇠와의 대응은 옵티마이저별로 확인하고, 사전분포 해석에는 합·평균 NLL의 구분을 반영한다.

**참고 문헌**

1. Hoerl, A. E., & Kennard, R. W. (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems. *Technometrics*, 12(1), 55-67.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
3. Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR 2019*.
