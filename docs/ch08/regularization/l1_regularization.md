# L1 정규화 (라쏘)

라쏘(Least Absolute Shrinkage and Selection Operator)라고도 하는 L1 정규화는 모델 웨이트의 절댓값에 비례하는 패널티를 로스 함수에 더한다. 이 기법은 학습된 파라메터의 희소성을 북돋우며, 관련 없는 피처의 웨이트를 정확히 0으로 몰아 사실상 자동 피처 선택을 수행한다.

---

## 1. 수학적 정식화

### L1 패널티를 더한 표준 로스

파라메터가 $\theta = \{w_1, w_2, \ldots, w_n\}$인 로스 함수 $\mathcal{L}(\theta)$에 대해 L1 정규화는 목적 함수를 다음과 같이 바꾼다.

$$
\mathcal{L}_{\text{L1}}(\theta) = \mathcal{L}(\theta) + \lambda \sum_{i=1}^{n} |w_i|
$$

여기서 각 기호는 다음과 같다.

- $\mathcal{L}(\theta)$은 원래의 로스 함수이다(예: MSE, 교차 엔트로피)
- $\lambda \geq 0$은 정규화의 강도이다(하이퍼파라미터)
- $\sum_{i=1}^{n} |w_i|$은 웨이트 벡터의 L1 노름이다

### L1을 쓰는 선형 회귀 (라쏘 회귀)

설계 행렬 $X \in \mathbb{R}^{m \times n}$, 목표 $y \in \mathbb{R}^m$, 웨이트 $w \in \mathbb{R}^n$인 선형 회귀에 대해 다음이 성립한다.

$$
\mathcal{L}_{\text{Lasso}}(w) = \frac{1}{2m} \|Xw - y\|_2^2 + \lambda \|w\|_1
$$

항을 펼치면 다음과 같다.

$$
\mathcal{L}_{\text{Lasso}}(w) = \frac{1}{2m} \sum_{j=1}^{m} \left( \sum_{i=1}^{n} x_{ji} w_i - y_j \right)^2 + \lambda \sum_{i=1}^{n} |w_i|
$$

### 그래디언트와 부분그래디언트

L1 노름은 $w_i = 0$에서 미분할 수 없다. 부분그래디언트를 쓴다.

$$
\frac{\partial}{\partial w_i} |w_i| = 
\begin{cases}
+1 & \text{if } w_i > 0 \\
-1 & \text{if } w_i < 0 \\
[-1, +1] & \text{if } w_i = 0
\end{cases}
$$

0에서 허용되는 여러 부분그래디언트 중 0을 선택하면 다음 부호 함수가 된다. 이는 전체 부분미분 집합을 나타내는 식은 아니다.

$$
\frac{\partial}{\partial w_i} |w_i| = \text{sign}(w_i) = 
\begin{cases}
+1 & \text{if } w_i > 0 \\
-1 & \text{if } w_i < 0 \\
0 & \text{if } w_i = 0
\end{cases}
$$

선택한 부분그래디언트에 따른 갱신용 벡터는 다음과 같다. 0에서는 일반적인 도함수가 존재하지 않는다.

$$
\nabla_{w} \mathcal{L}_{\text{L1}} = \nabla_{w} \mathcal{L} + \lambda \cdot \text{sign}(w)
$$

---

## 2. 기하학적 해석

### 제약 영역

적절한 볼록성·정규성 조건에서 패널티 강도에 대응하는 L1 노름 제약 문제를 구성할 수 있다.

$$
\min_w \mathcal{L}(w) \quad \text{subject to} \quad \|w\|_1 \leq t
$$

2차원에서 L1 공은 축 위에 모서리가 있는 마름모(회전한 정사각형) 모양을 이룬다.

$$
\|w\|_1 = |w_1| + |w_2| \leq t
$$

### 기하로부터 나오는 희소성

L1 제약 영역의 마름모 모양은 좌표축 위에 **모서리**를 갖는다. 로스 함수의 등고선이 제약 영역과 만날 때 이 모서리에서 닿을 가능성이 더 크며, 그 결과 어떤 웨이트는 정확히 0인 해가 된다.

**L2(원형 제약)와의 비교:**

| 성질 | L1 (마름모) | L2 (원) |
|----------|--------------|-------------|
| 모서리 점 | 있음 (축 위) | 없음 |
| 희소한 해 | 그렇다 | 드물다 |
| 미분 가능 | 아니다 (모서리에서) | 그렇다 |

---

## 3. 희소성과 피처 선택

### L1이 희소한 해를 만드는 이유

웨이트 하나 $w_i$에 대한 최적화 지형을 생각하자.

1. **영에서 떨어져 있을 때**: 그래디언트에 $\lambda \cdot \text{sign}(w_i)$이 들어가 $w_i$을 영 쪽으로 민다
2. **영에서**: 부분그래디언트가 구간 $[-\lambda, +\lambda]$을 포함한다
3. **영에 머무를 조건**: $w_i = 0$에서 원래 로스의 기울기가 $[-\lambda, +\lambda]$ 안에 있으면 $w_i = 0$이 최적이다

이는 최적점의 조건이다. 일반 SGD나 Adam에 L1 항을 더하는 것만으로 유한한 반복에서 계수가 정확히 0이 되지는 않을 수 있다. 정확한 희소성이 필요하면 근접법이나 좌표 하강법과 같은 알고리즘을 검토한다.

### 소프트 스레숄딩 연산자

L1 정규화된 최소제곱에서 (좌표 하강법의) 각 좌표에 대한 해석적 해는 다음과 같다.

$$
w_i^*=S_{\lambda/a_i}(z_i),\qquad a_i=\|X_{:,i}\|_2^2/m>0
$$

$z_i$는 다른 계수를 고정한 조건부 최소제곱 계수이고 $S_t(z)=\operatorname{sign}(z)\max(|z|-t,0)$이다. 임계가 단순히 $\lambda$가 되려면 열의 제곱 노름을 $m$으로 맞춰 $a_i=1$이어야 한다. 영벡터 열은 별도로 처리한다.

---

## 4. PyTorch 구현

### L1 정규화 직접 구현하기

```python
import torch
import torch.nn as nn

def l1_regularization(model: nn.Module, lambda_l1: float) -> torch.Tensor:
    """
    모델의 모든 파라메터에 대한 L1 정칙화 벌점을 계산한다.
    
    인수:
        model: 뉴럴 넷트웍 모델
        lambda_l1: 정칙화의 강도
        
    반환값:
        L1 벌점 항
    """
    # requires_grad=True는 사실 필요 없다. 0은 상수이고, 아래에서 더하는
    # param 쪽이 이미 그래디언트를 나르기 때문이다.
    # 주의: device를 맞추지 않아 모델이 GPU에 있으면 덧셈에서 오류가 난다.
    # 뒤의 l2_regularization은 next(model.parameters()).device로 이를 피한다
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

### PyTorch의 내장 정규화 장치 쓰기

```python
import torch.nn as nn
from torch.nn.utils import parametrize

class L1Regularizer(nn.Module):
    """파라메터화로 구현한 L1 웨이트 정칙화 장치.

    L2와 달리 L1은 옵티마이저의 weight_decay로 걸 수 없다. weight_decay는
    웨이트에 비례하는 항을 그래디언트에 더하는데(= L2), L1의 그래디언트는 크기와
    무관한 부호 sign(w)이기 때문이다. 그래서 로스 쪽에서 직접 더해야 한다.
    """

    def __init__(self, lambda_l1: float):
        super().__init__()
        self.lambda_l1 = lambda_l1   # 벌점의 세기

    def forward(self, weight):
        # parametrize가 요구하는 자리. 웨이트를 바꾸지 않고 그대로 흘려보낸다.
        # 실제 벌점은 아래 함수에서 로스에 더한다
        return weight

    def right_inverse(self, weight):
        # forward의 역함수. 여기서는 항등이므로 그대로 돌려준다.
        # 이것이 있어야 웨이트를 직접 대입할 때 parametrize가 되돌릴 수 있다
        return weight

def add_l1_regularization_to_loss(model, base_loss, lambda_l1):
    """아무 로스 함수에나 L1 정칙화를 더한다."""
    # 모든 파라메터의 절댓값을 더한다. 이것이 ||w||_1 이다.
    # 절댓값은 원점에서 뾰족해 미분이 +-lambda로 뛰므로, 데이터가 주는
    # 그래디언트가 lambda보다 작으면 웨이트가 정확히 0에 머문다.
    # L2가 0에 가깝게만 만드는 것과 달리 L1이 성긴 해를 주는 까닭이다
    l1_norm = sum(p.abs().sum() for p in model.parameters())

    # 주의: 바이어스까지 포함해 더하고 있다. 엄밀하게는 바이어스에 벌을 주지
    # 않는 것이 보통이며, 그러려면 model.parameters() 대신
    # 이름이 'bias'가 아닌 파라메터만 골라야 한다
    return base_loss + lambda_l1 * l1_norm
```

### L1 정규화를 쓰는 뉴럴 넷트웍

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class SparseNN(nn.Module):
    """희소 피처 학습을 위해 설계한 뉴럴 넷트웍."""
    
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
        """모든 웨이트의 L1 노름을 계산한다."""
        # 0은 파이썬 정수라 첫 +=에서 텐서로 바뀐다. 그 뒤로는 제자리
        # 덧셈이지만 덧셈의 역전파는 덮어쓴 값을 필요로 하지 않아 안전하다
        l1_norm = 0
        for param in self.parameters():
            # 바이어스까지 함께 센다. 보통 바이어스에는 벌점을 주지 않는데,
            # 바이어스는 과적합의 원인이 아니라 출력의 중심을 옮기는 자유도라서다.
            # 웨이트만 고르려면 named_parameters()로 걸러야 한다
            l1_norm += torch.sum(torch.abs(param))
        return l1_norm
    
    def count_zero_weights(self, threshold=1e-6):
        """사실상 영인 웨이트의 수를 센다."""
        # 이름 그대로 "사실상" 영이다. L1 벌점을 로스에 더해 Adam으로
        # 미분해 내려가는 방식은 웨이트를 0 근처로 밀 뿐 0에 못박지
        # 못한다. 정확히 0을 얻으려면 아래의 근접 경사 하강법이 필요하다.
        # 그래서 여기서 나온 희소도는 임계값을 얼마로 잡느냐에 달려 있다
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
        model: 뉴럴 넷트웍
        train_loader: 학습 데이터 로더
        val_loader: 검증 데이터 로더
        lambda_l1: L1 정칙화의 강도
        epochs: 학습 에포크 수
        lr: 러닝레이트
        
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
            # 벌점은 배치와 무관하다. 데이터를 보지 않고 웨이트만 보므로
            # 한 에포크 안에서도 값이 거의 같지만, 배치마다 다시 계산해야
            # 한다. 갱신 뒤에는 웨이트가 달라져 있기 때문이다
            l1_penalty = lambda_l1 * model.get_l1_norm()
            loss = mse_loss + l1_penalty
            
            loss.backward()
            optimizer.step()
            # 기록하는 것은 loss가 아니라 mse_loss다. 벌점을 빼고 재야
            # lambda_l1이 다른 실험끼리 로스 곡선을 견줄 수 있다.
            # 벌점까지 넣으면 강도가 셀수록 로스가 커 보여 뜻이 흐려진다
            train_loss += mse_loss.item()
        
        # 검증
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                val_loss += criterion(predictions, y_batch).item()
        
        # 지표를 추적한다
        # len(loader)는 배치의 개수다. 즉 표본이 아니라 묶음으로 나눈
        # 평균이라, 마지막 배치가 짧으면 그 배치가 실제보다 큰 무게를
        # 얻는다. 표본 단위로 재려면 로스에 배치 크기를 곱해 두었다가
        # 표본 총수로 나누어야 한다
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
    L1 정칙화의 근접 연산자 (연성 임계화).
    
    인수:
        weights: 파라메터 텐서
        lambda_l1: 정칙화의 강도
        lr: 러닝레이트
        
    반환값:
        연성 임계화를 거친 웨이트
    """
    # 이것이 앞에서 말한 "0에 못박는" 연산이다. 로스에 |w|를 더해
    # 미분해 내려가는 방식은 웨이트를 0 근처로 밀 뿐이지만, 연성
    # 임계화는 크기가 임계값에 못 미치는 웨이트를 정확히 0으로 만든다.
    # 그래서 count_zero_weights의 임계값 타령이 여기서는 필요 없다.
    #
    # 식을 뜯어보면, 부호는 그대로 두고 크기에서 threshold만큼을 빼되
    # 음수가 되면 0에서 멈춘다. |w| <= threshold인 웨이트는 모두 0으로
    # 떨어지고, 살아남은 웨이트도 threshold만큼 0 쪽으로 당겨진다.
    # 임계가 lambda * lr인 까닭은 한 걸음에 벌점이 미치는 크기가
    # 그만큼이기 때문이다
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
                    # 두 걸음을 나누어 밟는 것이 근접 경사 하강법의 요체다.
                    # 먼저 벌점을 뺀 로스만으로 보통의 경사 하강을 하고,
                    # 그다음 벌점을 임계화라는 별도의 연산으로 적용한다.
                    # |w|는 0에서 미분할 수 없으므로 벌점을 그래디언트에
                    # 섞지 않고 이렇게 떼어 놓는 것이다.
                    # 주의: param.grad에는 벌점이 들어 있으면 안 된다.
                    # 로스에 L1을 더해 backward한 뒤 이 최적화기를 쓰면
                    # 벌점이 두 번 걸린다
                    # 그래디언트 단계
                    param.data -= self.lr * param.grad
                    # 근접 단계 (연성 임계화)
                    param.data = proximal_l1(param.data, self.lambda_l1, self.lr)
    
    def zero_grad(self):
        """모든 파라메터의 그래디언트를 0으로 만든다."""
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()
```

---

## 5. scikit-learn 구현

```python
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
import numpy as np

def lasso_feature_selection(X, y, alphas=None):
    """
    교차 검증과 함께 라쏘로 피처를 선택한다.
    
    인수:
        X: 피처 행렬 (n_samples, n_features)
        y: 목표 벡터
        alphas: 시도해 볼 정칙화 값
        
    반환값:
        선택된 피처의 인덱스와 라쏘 모델
    """
    # 피처 표준화 (L1에서 중요하다).
    # 벌점이 모든 계수에 같은 세기로 걸리므로, 자가 큰 특성일수록
    # 계수가 작아져 벌을 덜 받는다. 그러면 "어떤 특성이 살아남는가"가
    # 실제 중요도가 아니라 단위에 따라 정해져 버린다
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 교차 검증으로 최적 alpha 찾기.
    # logspace를 쓰는 까닭은 alpha가 자릿수 단위로 움직이기 때문이다.
    # 1e-4에서 10까지를 고르게 나누면 큰 값 쪽만 촘촘해져 쓸모가 없다
    if alphas is None:
        alphas = np.logspace(-4, 1, 50)

    lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42)
    lasso_cv.fit(X_scaled, y)

    print(f"Optimal alpha: {lasso_cv.alpha_:.6f}")
    print(f"Non-zero coefficients: {np.sum(lasso_cv.coef_ != 0)}/{len(lasso_cv.coef_)}")

    # 선택된 피처 얻기.
    # != 0 으로 세는 것이 뜻을 갖는 까닭은 L1이 계수를 정확히 0으로
    # 몰기 때문이다. 릿지라면 0에 가깝기만 할 뿐 0은 아니라 이렇게 셀 수 없다
    selected_features = np.where(lasso_cv.coef_ != 0)[0]

    # scaler를 함께 돌려주는 것이 중요하다. 새 데이터에 이 모델을
    # 쓰려면 학습 때와 똑같은 평균과 표준편차로 맞춰야 한다
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
            # .copy()가 반드시 필요하다. lasso.coef_는 다음 회귀에서
            # 새 Lasso 객체가 만들어지므로 지금은 괜찮아 보이지만,
            # 같은 객체를 다시 fit하는 꼴로 바꾸면 모든 항목이 마지막
            # 계수를 함께 가리키게 된다. 사본을 뜨는 버릇이 안전하다
            'coefficients': lasso.coef_.copy()
        })

    # alpha를 키워 가며 n_nonzero가 줄어드는 자취가 곧 정칙화 경로다.
    # alpha가 충분히 크면 모든 계수가 0이 되고 모델은 절편만 남는다
    return results
```

---

## 6. 정규화 경로

**정규화 경로**는 $\lambda$이 변할 때 계수가 어떻게 바뀌는지 보여 준다.

```python
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path

def plot_lasso_path(X, y, eps=1e-3, n_alphas=100):
    """
    라쏘의 정칙화 경로를 그린다.
    
    인수:
        X: 피처 행렬
        y: 목표 벡터
        eps: 경로의 길이 (alpha_min / alpha_max)
        n_alphas: alpha 값의 개수
    """
    # 표준화
    # 벌점이 계수의 크기를 보고 매겨지므로, 눈금을 맞추지 않으면
    # 단위를 무엇으로 쟀느냐가 어느 피처가 먼저 0이 되는지를 좌우한다.
    # y를 중심화하면 절편을 따로 다루지 않아도 된다
    X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
    y_centered = y - y.mean()
    
    # 라쏘 경로 계산
    # alpha마다 따로 적합시키는 것이 아니라, 라쏘의 해가 alpha에 대해
    # 조각별 선형이라는 성질을 써서 경로 전체를 한 번에 얻는다.
    # 그림에서 계수들이 하나씩 0에서 떨어져 나오는 모습이 보이는데,
    # 그 순서가 곧 피처의 중요도 순위다
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

---

## 7. 비교: L1과 L2

| 항목 | L1 (라쏘) | L2 (릿지) |
|--------|------------|------------|
| 패널티 | $\lambda \sum \|w_i\|$ | $\lambda \sum w_i^2$ |
| 제약의 모양 | 마름모 | 원 |
| 희소한 해 | 선호함 | 직접 유도하지 않음 |
| 피처 선택 | 내장 | 없음 |
| 미분 가능 | 아니다 (0에서) | 그렇다 |
| 상관된 피처 | 선택이 불안정할 수 있음 | 계수를 분산시키는 경향 |
| 선형 최소제곱의 해 | 일반적으로 반복 계산. 직교 설계 등 특수한 경우 닫힌 해 | 양의 L2 강도에서 닫힌 해 |

---

## 8. 하이퍼파라미터 선택

### lambda를 위한 교차 검증

```python
from sklearn.model_selection import cross_val_score
import numpy as np

def select_lambda_cv(X, y, lambdas, cv=5):
    """
    교차 검증으로 최적의 lambda를 고른다.
    
    인수:
        X: 피처
        y: 목푯값
        lambdas: 평가할 lambda 값
        cv: 교차 검증 겹의 수
        
    반환값:
        최적의 lambda와 교차 검증 점수
    """
    from sklearn.linear_model import Lasso

    # lambda는 학습으로 정해지지 않는 초파라메터다. 훈련 자료에서 고르면
    # 늘 lambda=0(벌점 없음)이 이기므로, 반드시 검증으로 골라야 한다
    cv_scores = []
    for lam in lambdas:
        model = Lasso(alpha=lam)   # sklearn에서는 lambda를 alpha라 부른다

        # sklearn의 scoring은 "클수록 좋다"는 규약이므로 MSE에 음수를 붙여 준다.
        # 그래서 여기서 다시 음수를 씌워 원래의 MSE로 되돌린다
        scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
        cv_scores.append(-scores.mean())   # 겹 5개의 평균 MSE

    # 이제는 보통의 MSE이므로 가장 작은 것을 고른다
    optimal_idx = np.argmin(cv_scores)

    # 곡선 전체를 함께 돌려주는 까닭은, 최솟값 하나보다 곡선의 모양이
    # 더 많은 것을 말해 주기 때문이다. 바닥이 평평하면 lambda를 조금
    # 크게 잡아 더 성긴 모델을 고르는 편이 낫다(1-표준오차 규칙)
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

AIC·BIC의 $k$는 해당 통계 모형의 자유도이고 $\hat L$은 그 모형의 최대우도이다. 라쏘의 비영 계수 수를 자유도의 추정값으로 쓰는 데에는 가정이 필요하며, 분산·절편의 추정도 고려한다. 비볼록 뉴럴 넷트웍의 임의 L1 학습 결과에 이 식을 그대로 적용하지 않는다.

---

## 9. 딥러닝에서의 응용

### 희소한 입력 레이어

첫 레이어의 개별 계수에 L1을 적용하면 연결 수준의 희소성을 유도할 수 있다. 입력 피처 하나를 완전히 제거하려면 그 피처의 모든 출력 연결이 0이어야 하므로, 개별 계수의 희소성과 피처 선택은 다르다. 피처 단위로 없애려면 그룹 패널티도 고려한다.

```python
class SparseInputNetwork(nn.Module):
    """피처 선택을 위해 입력 레이어에 L1 정칙화를 건 뉴럴 넷트웍."""
    
    def __init__(self, input_dim, hidden_dims, output_dim, input_l1=0.01):
        super().__init__()
        self.input_l1 = input_l1
        
        # 첫 레이어 (강하게 정칙화).
        # 이 레이어만 따로 떼어 둔 까닭은 여기에만 벌점을 걸기 위해서다.
        # 입력 피처 하나에 해당하는 것은 weight의 한 "열"이라, 그 열이
        # 통째로 0이 되면 그 피처는 뉴럴 넷트웍에 아예 들어오지 못한다.
        # 곧 피처 선택이다. 은닉 레이어까지 눌러 버리면 표현력만 잃고
        # 어느 피처가 쓸모없는지는 알 수 없다
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        
        # 은닉 레이어
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
        """입력 레이어에만 적용하는 L1 벌점."""
        # .weight만 쓰고 .bias는 빼는 점을 눈여겨보라. 바이어스는 특정 입력
        # 피처에 딸린 값이 아니라 뉴런마다 하나씩인 상수라, 피처를
        # 고르는 일과 관계가 없다.
        # 다만 원소마다 벌점을 주는 이 형태는 열을 통째로 밀어내지는
        # 못한다. 한 피처를 확실히 끄려면 열의 노름에 벌점을 주는
        # 배치 라쏘가 필요하다
        return self.input_l1 * torch.sum(torch.abs(self.input_layer.weight))
```

---

## 10. 실무 지침

### L1 정규화를 쓸 때

1. **피처 선택이 필요할 때**: 피처가 많고 그중 가장 중요한 것을 가려내고 싶을 때
2. **해석 가능성이 중요할 때**: 희소한 모델이 해석하기 더 쉽다
3. **고차원 데이터**: $p > n$일 때(피처가 표본보다 많을 때)
4. **관련 없는 피처가 의심될 때**: 많은 피처가 잡음일 가능성이 클 때

### lambda 고르기

- **너무 작으면**: 정규화가 약하고 과적합하며 영이 아닌 웨이트가 많다
- **너무 크면**: 과소적합하고 고른 피처가 너무 적다
- **교차 검증을 쓰라**: 데이터가 선택을 이끌게 하라
- **척도를 고려하라**: L1을 적용하기 전에 피처를 표준화하라

### 흔한 함정

1. **피처의 척도 조정**: L1은 크기에 따라 패널티를 준다. 척도를 맞추지 않은 피처는 바이어스된 선택으로 이어진다
2. **상관된 피처**: 비슷한 피처 사이의 선택이 불안정할 수 있고 해가 유일하지 않을 수도 있다
3. **미분 불가능성**: 표준 경사 하강법은 어려움을 겪을 수 있으므로 근접 방법을 쓴다
4. **편향-분산 절충**: $\lambda$이 크면 바이어스는 늘고 분산은 준다

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
부분그래디언트의 최적 조건과 단순 경사 갱신에서의 정확한 0을 구분하라.

</div>

??? success "연습문제 1 풀이"
    미분 가능한 데이터 로스에서 계수 0이 최적이려면 $|\partial_iL|\leq\lambda$가 필요하며, 볼록 문제에서는 전체 부분그래디언트 조건이 전역 최적성을 보장한다. 그러나 sign(0)=0을 고른 SGD 갱신이 이 조건을 자동으로 만족시키지는 않는다. 근접 갱신은 임계 이하의 값을 정확히 0으로 만든다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
L1과 L2의 희소성 경향과 그 한계를 비교하라.

</div>

??? success "연습문제 2 풀이"
    L1은 0에서의 부분미분 구간 때문에 정확한 영 계수를 선호한다. L2는 노름을 줄이는 연속 패널티로 희소성을 직접 목표로 하지 않는다. 두 방식 모두 데이터의 대칭이나 구조에 따라 영 계수를 가질 수 있고, 선택된 피처가 인과적·본질적으로 중요하다는 보장은 없다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
L1 정규화 학습을 위한 근접 경사 하강법을 구현하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    def proximal_step(w, grad, lr, lam):
        w_tmp = w - lr * grad  # 그래디언트 단계
        # 연성 임계화 (L1의 근접 연산자)
        return torch.sign(w_tmp) * torch.clamp(w_tmp.abs() - lr * lam, min=0)
    ```

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
라플라스 사전분포와 L1 패널티의 관계를 합 NLL과 평균 NLL에서 각각 설명하라.

</div>

??? success "연습문제 4 풀이"
    독립 라플라스 사전의 스케일이 $b$이면 음의 로그 사전은 $\|w\|_1/b$에 상수를 더한 값이다. 합 NLL에 더할 계수는 $1/b$, 표본 수 m으로 전체 목적을 나눠 평균 NLL로 쓰면 $1/(mb)$이다. 데이터 로스의 집계 방식을 바꾸면 같은 사전에 대응하는 패널티 계수도 바뀐다.

## 정리하며

L1은 절댓값 패널티로 희소해를 선호한다. 최적 조건과 실제 알고리즘의 갱신을 구분하며, 정확한 0이 필요하면 근접법 등을 사용한다. 피처 스케일, 상관관계와 패널티 대상에 따라 선택 결과가 달라지며, 뉴럴 넷트웍의 연결 희소성이 곧 입력 피처 제거를 뜻하지는 않는다.

**참고 문헌**

1. Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267-288.
2. Hastie, T., Tibshirani, R., & Wainwright, M. (2015). *Statistical Learning with Sparsity: The Lasso and Generalizations*. CRC Press.
3. Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
