# 로지스틱 회귀 MLE

이차원 합성 데이터를 선형 점수와 시그모이드로 분류한다. BCE 최소화가 베르누이 조건부 우도를 최대화하는 과정임을 살펴본다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
================================================================================
로지스틱 회귀 최대가능도 — 둘 분류
================================================================================

어려움: ⭐⭐⭐ 앞선(3단계)

학습 목표:
- 로지스틱 회귀를 최대가능도로 이해한다
- 둘 교차 엔트로피 손실과의 이음을 본다
- PyTorch로 분류를 짠다
- 시그모이드 함수와 로그 승산을 배운다

문제: 둘 분류 — 특징 x에서 y ∈ {0, 1}을 예측한다

모델: P(y=1 | x) = σ(w^T x + b) = 1 / (1 + exp(-(w^T x + b)))

여기서 σ은 시그모이드(로지스틱) 함수다

최대가능도로 세우기:
Likelihood: L(w, b) = ∏ P(y_i | x_i, w, b)
           = ∏ σ(w^T x_i + b)^y_i * (1 - σ(w^T x_i + b))^(1-y_i)

로그 가능도: ℓ(w, b) = Σ [y_i log(σ(w^T x_i + b)) + (1-y_i) log(1 - σ(w^T x_i + b))]

이는 둘 교차 엔트로피 손실을 가장 작게 하는 것과 같다!

핵심 통찰: 교차 엔트로피 = 음의 로그 가능도

지은이: PyTorch 최대가능도 학습
DATE: 2025
================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# ========================================================================
# 메인
# ========================================================================


def generate_classification_data(n_samples: int = 200, seed: int = 42):
    """합성 이진 분류 데이터를 생성한다"""
    torch.manual_seed(seed)
    
    # 두 개의 군집을 생성한다
    n_per_class = n_samples // 2
    
    # 클래스 0: (-2, -2)를 중심으로
    X0 = torch.randn(n_per_class, 2) * 0.8 + torch.tensor([-2.0, -2.0])
    y0 = torch.zeros(n_per_class, 1)
    
    # 클래스 1: (2, 2)를 중심으로
    X1 = torch.randn(n_per_class, 2) * 0.8 + torch.tensor([2.0, 2.0])
    y1 = torch.ones(n_per_class, 1)
    
    # 합친다
    X = torch.cat([X0, X1], dim=0)
    y = torch.cat([y0, y1], dim=0)
    
    # 뒤섞는다
    perm = torch.randperm(n_samples)
    X, y = X[perm], y[perm]
    
    return X, y


def compute_log_likelihood(X, y, w, b):
    """
    로지스틱 회귀의 로그 가능도를 계산한다.
    
    ℓ(w,b) = Σ [y_i log(σ(z_i)) + (1-y_i) log(1 - σ(z_i))]
    where z_i = w^T x_i + b
    """
    # 로짓을 계산한다
    logits = X @ w + b
    
    # 시그모이드를 적용한다 (수치적으로 안정한 버전)
    probs = torch.sigmoid(logits)
    
    # 로그가능도
    epsilon = 1e-8  # For numerical stability
    log_lik = torch.sum(
        y * torch.log(probs + epsilon) + (1 - y) * torch.log(1 - probs + epsilon)
    )
    
    return log_lik


class LogisticRegressionModel(nn.Module):
    """PyTorch 모듈로 구현한 로지스틱 회귀"""
    
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """순전파"""
        logits = self.linear(x)
        probs = self.sigmoid(logits)
        return probs


def train_logistic_regression(X, y, learning_rate=0.1, n_epochs=1000):
    """MLE(교차 엔트로피 손실)으로 로지스틱 회귀를 학습시킨다"""
    
    input_dim = X.shape[1]
    model = LogisticRegressionModel(input_dim)
    
    # 손실 함수: 이진 교차 엔트로피 (음의 로그가능도와 같다)
    criterion = nn.BCELoss()
    
    # 최적화기
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 학습 기록
    history = {'loss': [], 'accuracy': []}
    
    for epoch in range(n_epochs):
        # 순전파
        probs = model(X)
        loss = criterion(probs, y)
        
        # 정확도를 계산한다
        predictions = (probs > 0.5).float()
        accuracy = (predictions == y).float().mean()
        
        # 이력 저장
        history['loss'].append(loss.item())
        history['accuracy'].append(accuracy.item())
        
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            print(f"   Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}, Acc: {accuracy.item():.2%}")
    
    return model, history


def visualize_results(X, y, model, history):
    """종합적인 시각화를 만든다"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # ================================================================
    # 그림 1: 결정 경계
    # ================================================================
    ax1 = plt.subplot(2, 3, 1)
    
    # 격자를 만든다
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # 격자에서 예측한다
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        Z = model(grid).numpy().reshape(xx.shape)
    
    # 결정 경계와 영역을 그린다
    ax1.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
    ax1.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    
    # 데이터 점을 그린다
    X_np, y_np = X.numpy(), y.numpy().flatten()
    scatter = ax1.scatter(X_np[y_np == 0, 0], X_np[y_np == 0, 1], 
                         c='blue', marker='o', s=50, edgecolors='black', label='Class 0')
    scatter = ax1.scatter(X_np[y_np == 1, 0], X_np[y_np == 1, 1], 
                         c='red', marker='^', s=50, edgecolors='black', label='Class 1')
    
    ax1.set_xlabel('Feature 1', fontsize=12)
    ax1.set_ylabel('Feature 2', fontsize=12)
    ax1.set_title('Decision Boundary', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ================================================================
    # 그림 2: 손실의 수렴 (로그가능도)
    # ================================================================
    ax2 = plt.subplot(2, 3, 2)
    
    # 손실을 로그가능도로 바꾼다 (loss = -log_lik / n)
    log_likelihood = [-loss * len(X) for loss in history['loss']]
    
    ax2.plot(log_likelihood, 'b-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Log-Likelihood', fontsize=12)
    ax2.set_title('Log-Likelihood Convergence', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # ================================================================
    # 그림 3: 정확도의 수렴
    # ================================================================
    ax3 = plt.subplot(2, 3, 3)
    
    ax3.plot(history['accuracy'], 'g-', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('Accuracy Convergence', fontsize=14, fontweight='bold')
    ax3.axhline(0.5, color='gray', linestyle='--', label='Random chance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.05])
    
    # ================================================================
    # 그림 4: 확률 예측
    # ================================================================
    ax4 = plt.subplot(2, 3, 4)
    
    with torch.no_grad():
        probs = model(X).numpy().flatten()
    
    # 참 클래스별로 나눈다
    probs_class0 = probs[y.numpy().flatten() == 0]
    probs_class1 = probs[y.numpy().flatten() == 1]
    
    ax4.hist(probs_class0, bins=20, alpha=0.6, color='blue', label='True Class 0', edgecolor='black')
    ax4.hist(probs_class1, bins=20, alpha=0.6, color='red', label='True Class 1', edgecolor='black')
    ax4.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Decision threshold')
    ax4.set_xlabel('Predicted Probability', fontsize=12)
    ax4.set_ylabel('Count', fontsize=12)
    ax4.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # ================================================================
    # 그림 5: 시그모이드 함수
    # ================================================================
    ax5 = plt.subplot(2, 3, 5)
    
    z = np.linspace(-6, 6, 100)
    sigmoid = 1 / (1 + np.exp(-z))
    
    ax5.plot(z, sigmoid, 'b-', linewidth=3)
    ax5.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax5.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Logit (w^T x + b)', fontsize=12)
    ax5.set_ylabel('P(y=1)', fontsize=12)
    ax5.set_title('Sigmoid/Logistic Function', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.text(3, 0.2, 'σ(z) = 1/(1 + e^(-z))', fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='wheat'))
    
    # ================================================================
    # 그림 6: 혼동 행렬
    # ================================================================
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # 혼동 행렬을 계산한다
    with torch.no_grad():
        predictions = (model(X) > 0.5).float()
        
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y.numpy(), predictions.numpy())
    
    # 혼동 행렬을 표시한다
    im = ax6.imshow(cm, cmap='Blues', alpha=0.7)
    
    # 글자 주석을 추가한다
    for i in range(2):
        for j in range(2):
            text = ax6.text(j, i, str(cm[i, j]), ha="center", va="center",
                          color="black", fontsize=20, fontweight='bold')
    
    ax6.set_xticks([0, 1])
    ax6.set_yticks([0, 1])
    ax6.set_xticklabels(['Pred 0', 'Pred 1'])
    ax6.set_yticklabels(['True 0', 'True 1'])
    ax6.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    
    # 지표를 추가한다
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_text = f"""
정확도:  {accuracy:.3f}
정밀도: {precision:.3f}
재현율:    {recall:.3f}
F1-Score:  {f1:.3f}
"""
    ax6.text(0.5, -0.25, metrics_text, ha='center', va='top',
            fontsize=10, family='monospace', transform=ax6.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('logistic_regression_mle_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 Figure saved as 'logistic_regression_mle_results.png'")
    plt.show()


def main():
    print("=" * 80)
    print("LOGISTIC REGRESSION MLE - Binary Classification")
    print("=" * 80)
    
    # 데이터를 생성한다
    print("\n🎲 Generating classification data...")
    X, y = generate_classification_data(n_samples=200)
    
    print(f"   • Dataset size: {len(X)} samples")
    print(f"   • Features: {X.shape[1]} dimensions")
    print(f"   • Class 0: {(y == 0).sum().item()} samples")
    print(f"   • Class 1: {(y == 1).sum().item()} samples")
    
    # 모델을 학습시킨다
    print("\n🔥 Training Logistic Regression via MLE...")
    print("-" * 80)
    model, history = train_logistic_regression(X, y, learning_rate=0.1, n_epochs=1000)
    
    # 최종 평가
    print("\n📊 Final Evaluation:")
    print("-" * 80)
    with torch.no_grad():
        probs = model(X)
        predictions = (probs > 0.5).float()
        accuracy = (predictions == y).float().mean()
        
        # 로그가능도를 계산한다
        w = model.linear.weight.data
        b = model.linear.bias.data
        log_lik = compute_log_likelihood(X, y, w.T, b)
        
    print(f"   • Final Accuracy: {accuracy.item():.2%}")
    print(f"   • Final Log-Likelihood: {log_lik.item():.2f}")
    print(f"   • Final Loss (BCE): {history['loss'][-1]:.4f}")
    
    # 모델 매개변수를 보여준다
    print(f"\n   Model Parameters:")
    print(f"   • Weight w: {model.linear.weight.data.numpy().flatten()}")
    print(f"   • Bias b: {model.linear.bias.data.item():.4f}")
    
    # 시각화한다
    print("\n📊 Creating visualizations...")
    visualize_results(X, y, model, history)
    
    print("\n" + "=" * 80)
    print("✅ COMPLETE!")
    print("=" * 80)
    print("\n💡 KEY TAKEAWAYS:")
    print("   1. Logistic regression IS maximum likelihood estimation")
    print("   2. Binary cross-entropy = Negative log-likelihood")
    print("   3. Sigmoid maps real numbers to probabilities [0, 1]")
    print("   4. Gradient descent optimizes the MLE")
    print("   5. This is the foundation of neural networks!")
    print("\n" + "=" * 80)


"""
🎓 EXERCISES:

1. 보통: 여러 클래스 로지스틱 회귀(소프트맥스)
   - 클래스를 3개 넘게로 넓힌다
   - 클래스 교차 엔트로피를 쓴다
   - 분류 테두리를 그림으로 본다

2. 보통: 정칙화한 로지스틱 회귀
   - L2 정칙화를 더한다: 손실 + λ||w||²
   - 이는 정규 앞확률을 쓴 최대사후와 같다!
   - 정칙화한 것과 안 한 것을 견준다

3. 어려움: 특징 만들기
   - 다항 특징을 더한다(x₁², x₁x₂, x₂²)
   - 비선형 분류 테두리를 만든다
   - 선형 특징과 견준다

4. 어려움: 치우친 클래스
   - 클래스 비가 10:1인 데이터를 만든다
   - 가중치 실은 손실 함수를 써 본다
   - 정밀도, 재현율, F1으로 평가한다

5. 어려움: 확률로 읽기
   - 눈금 굽이를 그린다(예측 확률과 참 확률)
   - 눈금 맞추기에 플랫 잣대를 짠다
   - 눈금이 잘 맞는 모델과 안 맞는 모델을 견준다
"""


if __name__ == "__main__":
    main()
```

**출력:**

```
================================================================================
LOGISTIC REGRESSION MLE - Binary Classification
================================================================================

🎲 Generating classification data...
   • Dataset size: 200 samples
   • Features: 2 dimensions
   • Class 0: 100 samples
   • Class 1: 100 samples

🔥 Training Logistic Regression via MLE...
--------------------------------------------------------------------------------
   Epoch 200/1000, Loss: 0.0025, Acc: 100.00%
   Epoch 400/1000, Loss: 0.0013, Acc: 100.00%
   Epoch 600/1000, Loss: 0.0008, Acc: 100.00%
   Epoch 800/1000, Loss: 0.0006, Acc: 100.00%
   Epoch 1000/1000, Loss: 0.0004, Acc: 100.00%

📊 Final Evaluation:
--------------------------------------------------------------------------------
   • Final Accuracy: 100.00%
   • Final Log-Likelihood: -0.09
   • Final Loss (BCE): 0.0004

   Model Parameters:
   • Weight w: [2.645255  3.4574885]
   • Bias b: -0.1702

📊 Creating visualizations...

📊 Figure saved as 'logistic_regression_mle_results.png'

================================================================================
✅ COMPLETE!
================================================================================

💡 KEY TAKEAWAYS:
   1. Logistic regression IS maximum likelihood estimation
   2. Binary cross-entropy = Negative log-likelihood
   3. Sigmoid maps real numbers to probabilities [0, 1]
   4. Gradient descent optimizes the MLE
   5. This is the foundation of neural networks!

================================================================================
```

## 2. 논의

### 로짓에서 확률로

모델은 $z=x^\top w+b$, $p=\sigma(z)$를 계산한다. 입력 특징이 두 개이므로 가중치 두 개와 편향 하나, 총 세 개의 학습 파라미터가 있다. 출력 확률을 0.5와 비교하면 선형 결정 경계를 얻는다.

### 손실과 우도 비교

코드는 확률 출력에 `BCELoss`를 적용한다. 같은 확률과 감소 방식에서 BCE의 합은 베르누이 음의 로그우도다. 별도 로그우도 함수가 로그 안에 작은 상수를 더하면 원래 우도와 수치적으로 완전히 같은 함수는 아니다. 큰 로짓을 다룰 때는 로짓을 직접 받는 `BCEWithLogitsLoss`가 안정적인 대안이다.

### 최적해와 평가의 해석

선형 점수의 로지스틱 손실은 파라미터에 대해 볼록하지만, 완전히 선형 분리되는 표본에서는 유한한 비정규화 MLE가 존재하지 않을 수 있다. 정확도가 높아진 뒤에도 가중치 크기가 계속 커질 수 있으므로 분류 정확도만으로 파라미터 수렴을 판단하지 않는다. 예제의 혼동 행렬과 지표는 사용한 학습 데이터의 성능이며 독립 테스트 성능이 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
기본 예제의 로지스틱 회귀 모델에는 학습 파라미터가 몇 개 있는가?

</div>

??? success "연습문제 1 풀이"
    입력 특징 두 개에서 출력 하나를 만드는 선형 계층이므로 가중치 2개와 편향 1개, 총 3개다. 시그모이드에는 학습 파라미터가 없다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
같은 로지스틱 회귀 모델에서 Adam과 SGD를 공정하게 비교하려면 무엇을 맞춰야 하는가?

</div>

??? success "연습문제 2 풀이"
    같은 초기 가중치와 데이터 분할, 배치 순서, 평가 기준을 사용한다. 각 옵티마이저에 적합한 학습률을 탐색하고 여러 시드의 변동도 비교한다. Adam이 항상 빠르거나 일반화가 더 좋다고 미리 결론내리지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
선형 분리되는 표본과 불균형 표본에서 정확도만으로 학습을 평가할 때의 한계를 설명하라.

</div>

??? success "연습문제 3 풀이"
    완전 분리에서는 정확도가 같아도 유한한 MLE 없이 가중치가 계속 커질 수 있다. 불균형에서는 다수 클래스만 예측해도 정확도가 높을 수 있다. 가중치 노름과 손실, 정밀도·재현율과 별도 검증 성능을 함께 확인한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
층이나 블록의 개수를 설정할 수 있도록 `LogisticRegressionModel`를 확장하라. `__init__`에 `num_layers` 매개변수를 추가하고 `nn.ModuleList`로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

</div>

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 순회한다. (평범한 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 PyTorch가 모든 매개변수를 최적화 대상으로 등록한다. 시험은 다음과 같이 한다. `for n in [2, 4, 8]: model = LogisticRegressionModel(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.

## 정리하며

로지스틱 회귀는 확률 추정과 임곗값 결정을 연결한다. BCE와 로그우도를 비교할 때는 합·평균과 수치 안정화의 차이를 확인하고, 분리 가능성과 검증 성능도 점검한다.
