# 다항 회귀

입력과 출력의 관계가 비선형이면 선형 모델은 과소적합한다. 다항 회귀는 입력 특징을 $[1, x, x^2, x^3]$처럼 원래 변수의 거듭제곱을 포함하도록 확장하여 이 문제를 해결한다. 특징은 비선형이지만 모델은 여전히 매개변수에 대해 선형이므로 같은 경사 하강 장치로 학습할 수 있다. 이 튜토리얼은 삼차 데이터셋에 차수가 다른 다항식들을 적합시켜 편향-분산 절충을 보여준다.

## 코드

```python
"""
==============================================================================
07_polynomial_regression.py
==============================================================================
DIFFICULTY: ⭐⭐⭐ (Intermediate-Advanced)

DESCRIPTION:
    Polynomial regression to fit non-linear relationships.
    Demonstrates feature engineering and overfitting.

다루는 것:
    - Polynomial feature expansion
    - Overfitting vs underfitting
    - Model complexity trade-offs
    - Feature engineering

PREREQUISITES:
    - Tutorial 06 (Multivariate regression)

TIME: ~25 minutes
==============================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("POLYNOMIAL REGRESSION")
print("=" * 70)

# ============================================================================
# 1부: 비선형 데이터 생성
# ============================================================================
print("\n" + "=" * 70)
print("PART 1: GENERATE NON-LINEAR DATA")
print("=" * 70)

torch.manual_seed(42)
np.random.seed(42)

# 비선형 함수로부터 데이터를 생성한다
n_samples = 100
X = np.linspace(-3, 3, n_samples)
y_true = 0.5 * X**3 - 2*X**2 + X + 3  # True cubic function
y = y_true + np.random.normal(0, 2, n_samples)  # Add noise

print(f"Generated {n_samples} samples from cubic function")
print(f"True function: y = 0.5x³ - 2x² + x + 3")

# ============================================================================
# 2부: 다항 특징 확장
# ============================================================================
print("\n" + "=" * 70)
print("PART 2: POLYNOMIAL FEATURE EXPANSION")
print("=" * 70)

def create_polynomial_features(X, degree):
    """
    Create polynomial features up to specified degree
    
    For X and degree=3:
        Returns: [1, X, X², X³]
    """
    X = X.reshape(-1, 1)
    features = []
    for d in range(degree + 1):
        features.append(X ** d)
    return np.concatenate(features, axis=1)

# 텐서로 바꾼다
X_tensor = torch.FloatTensor(X).reshape(-1, 1)
y_tensor = torch.FloatTensor(y).reshape(-1, 1)

print("""
Polynomial Features:
- Degree 1 (Linear): [1, X]
- Degree 2 (Quadratic): [1, X, X²]
- Degree 3 (Cubic): [1, X, X², X³]
- Higher degrees: More complex curves
""")

# ============================================================================
# 3부: 여러 다항식 차수로 모델 학습하기
# ============================================================================
print("\n" + "=" * 70)
print("PART 3: TRAIN MODELS WITH DIFFERENT DEGREES")
print("=" * 70)

degrees = [1, 2, 3, 5, 10]
models = {}
results = {}

for degree in degrees:
    print(f"\nTraining polynomial degree {degree}...")
    
    # 다항 특징을 만든다
    X_poly = create_polynomial_features(X, degree)
    X_poly_tensor = torch.FloatTensor(X_poly)
    
    # 모델
    model = nn.Linear(degree + 1, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 학습
    n_epochs = 1000
    losses = []
    
    for epoch in range(n_epochs):
        y_pred = model(X_poly_tensor)
        loss = criterion(y_pred, y_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    # 결과를 저장한다
    models[degree] = model
    results[degree] = {
        'losses': losses,
        'final_loss': losses[-1],
        'X_poly': X_poly_tensor
    }
    
    print(f"  Final loss: {losses[-1]:.4f}")

# ============================================================================
# 4부: 여러 모델 시각화
# ============================================================================
print("\n" + "=" * 70)
print("PART 4: VISUALIZE MODEL COMPLEXITY")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, degree in enumerate(degrees):
    ax = axes[idx]
    
    # 원래 데이터
    ax.scatter(X, y, alpha=0.5, s=20, label='Data')
    ax.plot(X, y_true, 'g--', linewidth=2, label='True Function', alpha=0.7)
    
    # 모델의 예측
    model = models[degree]
    with torch.no_grad():
        X_poly = results[degree]['X_poly']
        y_pred = model(X_poly).numpy()
    
    ax.plot(X, y_pred, 'r-', linewidth=2, label=f'Degree {degree}')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title(f'Polynomial Degree {degree} (Loss: {results[degree]["final_loss"]:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-10, 15)

# 손실 비교
ax = axes[5]
for degree in degrees:
    ax.plot(results[degree]['losses'], label=f'Degree {degree}', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss Comparison')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('/home/claude/pytorch_linear_regression_tutorial/07_polynomial_comparison.png', dpi=100)
print("Saved visualization")
plt.show()

# ============================================================================
# 5부: 과적합과 과소적합 이해하기
# ============================================================================
print("\n" + "=" * 70)
print("PART 5: ANALYSIS")
print("=" * 70)

print("""
MODEL COMPLEXITY ANALYSIS:

Degree 1 (Linear):
  ❌ UNDERFITTING
  - Too simple to capture the cubic relationship
  - High training error
  - High test error (if we had test data)

Degree 2 (Quadratic):
  ⚠️ STILL UNDERFITTING
  - Better than linear but not enough
  - Can't capture cubic term
  
Degree 3 (Cubic):
  ✅ JUST RIGHT
  - Matches the true function degree
  - Good fit to data
  - Generalizes well
  
Degree 5:
  ⚠️ STARTING TO OVERFIT
  - More flexible than needed
  - Fits noise in training data
  - May not generalize well

Degree 10:
  ❌ SEVERE OVERFITTING
  - Extremely flexible
  - Fits training data too closely
  - Wiggly, unrealistic predictions
  - Poor generalization
  
KEY INSIGHT: Choose model complexity to match problem complexity!
""")

# ============================================================================
# 6부: 사용자 정의 다항 모델
# ============================================================================
print("\n" + "=" * 70)
print("PART 6: CUSTOM POLYNOMIAL MODEL CLASS")
print("=" * 70)

class PolynomialRegression(nn.Module):
    """사용자 정의 다항 회귀 모델"""
    
    def __init__(self, degree):
        super(PolynomialRegression, self).__init__()
        self.degree = degree
        self.linear = nn.Linear(degree + 1, 1)
        
    def create_features(self, x):
        """다항 특징을 만든다"""
        features = []
        for d in range(self.degree + 1):
            features.append(x ** d)
        return torch.cat(features, dim=1)
    
    def forward(self, x):
        x_poly = self.create_features(x)
        return self.linear(x_poly)

# 사용 예
model = PolynomialRegression(degree=3)
print(f"Created PolynomialRegression model with degree 3")
print(model)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Polynomial Regression Key Points:

1. Feature Engineering:
   - Transform X into [1, X, X², X³, ...]
   - Still linear in parameters (it's still linear regression!)
   - Can fit non-linear relationships

2. Model Complexity:
   - Higher degree = more flexible
   - Too simple = underfitting (high bias)
   - Too complex = overfitting (high variance)

3. Choosing Degree:
   - Use domain knowledge
   - Use validation set
   - Try different degrees and compare

4. Overfitting Prevention (next tutorial):
   - Regularization (L1, L2)
   - More training data
   - Cross-validation
   - Early stopping

Next: Tutorial 08 - Regularization!
""")


if __name__ == "__main__":
    pass
```

## 논의

다항 특징 확장은 선택한 차수 $d$에 대해 입력 $x$를 벡터 $[1, x, x^2, \ldots, x^d]$로 바꾼다. 그 결과인 모델 $\hat{y} = \sum_{k=0}^d w_k x^k$는 $x$에 대해서는 다항식이지만 매개변수 $w_0, \ldots, w_d$에 대해서는 여전히 선형 함수이다. 따라서 표준 `nn.Linear(d+1, 1)` 층과 MSE 손실을 수정 없이 쓸 수 있다. 비선형성은 모델 구조가 아니라 전적으로 특징 공학에서 온다.

같은 데이터에서 차수가 다른 모델들을 비교하면 편향-분산 절충이 뚜렷이 보인다. 1차(선형) 모델은 삼차 추세를 잡아내지 못해 과소적합하며 편향이 크다. 3차 모델은 참된 데이터 생성 과정과 일치하여 잘 적합한다. 10차 모델은 학습 데이터의 잡음까지 외울 만큼 유연해서 구불구불한 곡선을 만들며 과적합한다. 알맞은 모델 복잡도를 고르는 것은 기계 학습의 핵심 과제이다.

실무에서 다항식의 차수는 눈으로 살펴보는 대신 검증 집합이나 교차 검증으로 고른다. 검증 손실은 모델 복잡도가 커짐에 따라 처음에는 (편향이 줄면서) 감소하다가, 모델이 과적합하기 시작하면 (분산이 커지면서) 증가한다. 검증 손실을 최소로 만드는 차수가 가장 좋은 절충이다. 정칙화(다음 튜토리얼에서 다룬다)는 큰 가중치에 벌점을 주어 적합 곡선을 매끄럽게 만드는 대안적 접근을 제공한다.

## 연습문제

**익힘 1.**
$[-\pi, \pi]$ 위에서 $y = \sin(x) + \epsilon$으로부터 데이터를 생성하고 차수 3, 5, 9의 다항식을 적합시켜라. 과적합하지 않으면서 사인 함수를 가장 잘 근사하는 차수는 무엇인가?

??? success "익힘 1 풀이"
    ```python
    import torch, numpy as np
    import torch.nn as nn
    
    np.random.seed(42)
    X = np.linspace(-np.pi, np.pi, 100)
    y = np.sin(X) + np.random.normal(0, 0.2, 100)
    
    for degree in [3, 5, 9]:
        features = np.column_stack([X**d for d in range(degree + 1)])
        X_t = torch.FloatTensor(features)
        y_t = torch.FloatTensor(y).reshape(-1, 1)
        model = nn.Linear(degree + 1, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(1000):
            loss = nn.MSELoss()(model(X_t), y_t)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        print(f'Degree {degree}: loss={loss.item():.4f}')
    # sin(x)를 근사할 때는 대체로 차수 5가 가장 좋은 균형을 준다.
    ```

---

**익힘 2.**
10차 다항식이 학습 데이터는 거의 완벽하게 적합하면서도 보지 않은 시험 점에서는 성능이 나쁠 수 있는 이유를 설명하라.

??? success "익힘 2 풀이"
    10차 다항식은 자유 매개변수가 11개여서, 적당한 크기의 데이터셋이라면 모든 학습 점을 지나거나 그 근처를 지나기에 충분하다. 모델은 신호뿐 아니라 잡음까지 적합하여 데이터 점 사이에서 급격한 진동을 만든다(룽게 현상). 학습 점 사이에 놓인 시험 점에서는 이 진동이 큰 예측 오차를 낳는다. 이것이 과적합의 전형적 특징이다. 학습 오차는 낮지만 일반화 오차는 높다.

---

**익힘 3.**
학습/시험 분할(80/20)을 구현하고 1부터 15까지의 다항식 차수에 대한 학습 손실과 시험 손실을 함께 그려라. 시험 손실이 최소가 되는 차수를 찾아라.

??? success "익힘 3 풀이"
    ```python
    import torch, numpy as np
    import torch.nn as nn
    
    np.random.seed(42)
    X = np.linspace(-3, 3, 100)
    y_true = 0.5 * X**3 - 2*X**2 + X + 3
    y = y_true + np.random.normal(0, 2, 100)
    
    split = 80
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    for degree in range(1, 16):
        feat_train = np.column_stack([X_train**d for d in range(degree+1)])
        feat_test = np.column_stack([X_test**d for d in range(degree+1)])
        Xt = torch.FloatTensor(feat_train)
        yt = torch.FloatTensor(y_train).reshape(-1,1)
        Xte = torch.FloatTensor(feat_test)
        yte = torch.FloatTensor(y_test).reshape(-1,1)
        model = nn.Linear(degree+1, 1)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(2000):
            loss = nn.MSELoss()(model(Xt), yt)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            test_loss = nn.MSELoss()(model(Xte), yte)
        print(f'Degree {degree:2d}: train_loss={loss.item():.4f}, test_loss={test_loss.item():.4f}')
    # 참 함수가 삼차식이므로 차수 3이 시험 손실을 최소로 만들어야 한다.
    ```
