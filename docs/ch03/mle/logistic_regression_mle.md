# 로지스틱 회귀 MLE

로지스틱 회귀 MLE - 이진 분류. 문제: 이진 분류 - 특징 x로부터 y ∈ {0, 1}을 예측한다

이 튜토리얼은 PyTorch에서 최대가능도 추정에 대한 기초적인 이해를 쌓는다. 코드를 따라가 보면 모델을 세우고, 손실 함수를 정의하고, 경사 하강법으로 학습시키고, 분류 과제에서 성능을 평가하는 법을 알 수 있다.

## 코드

```python
#!/usr/bin/env python3
"""
================================================================================
LOGISTIC REGRESSION MLE - Binary Classification
================================================================================

DIFFICULTY: ⭐⭐⭐ Advanced (Level 3)

LEARNING OBJECTIVES:
- Understand logistic regression as MLE
- See connection to binary cross-entropy loss
- Implement classification in PyTorch
- Learn about sigmoid function and log-odds

PROBLEM: Binary classification - predict y ∈ {0, 1} from features x

MODEL: P(y=1 | x) = σ(w^T x + b) = 1 / (1 + exp(-(w^T x + b)))

where σ is the sigmoid/logistic function

MLE FORMULATION:
Likelihood: L(w, b) = ∏ P(y_i | x_i, w, b)
           = ∏ σ(w^T x_i + b)^y_i * (1 - σ(w^T x_i + b))^(1-y_i)

Log-likelihood: ℓ(w, b) = Σ [y_i log(σ(w^T x_i + b)) + (1-y_i) log(1 - σ(w^T x_i + b))]

This is equivalent to MINIMIZING binary cross-entropy loss!

KEY INSIGHT: Cross-entropy = Negative log-likelihood

AUTHOR: PyTorch MLE Tutorial  
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
    Compute log-likelihood for logistic regression.
    
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
Accuracy:  {accuracy:.3f}
Precision: {precision:.3f}
Recall:    {recall:.3f}
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

1. MEDIUM: Multi-class logistic regression (softmax)
   - Extend to 3+ classes
   - Use categorical cross-entropy
   - Visualize decision boundaries

2. MEDIUM: Regularized logistic regression
   - Add L2 regularization: loss + λ||w||²
   - This is equivalent to MAP with Gaussian prior!
   - Compare regularized vs unregularized

3. CHALLENGING: Feature engineering
   - Add polynomial features (x₁², x₁x₂, x₂²)
   - Create non-linear decision boundaries
   - Compare with linear features

4. CHALLENGING: Imbalanced classes
   - Generate data with 10:1 class ratio
   - Try weighted loss functions
   - Evaluate with precision, recall, F1

5. CHALLENGING: Probabilistic interpretation
   - Plot calibration curves (predicted vs actual probability)
   - Implement Platt scaling for calibration
   - Compare well-calibrated vs poorly-calibrated models
"""


if __name__ == "__main__":
    main()```

## 논의

`LogisticRegressionModel` 클래스는 PyTorch의 `nn.Module` 인터페이스를 사용하여 모델 구조를 감싼다. `forward` 메서드가 계산 그래프를 정의하므로, 학습 중에 PyTorch의 autograd 체계가 경사 계산을 자동으로 처리한다. 이런 모듈식 설계 덕분에 개별 구성 요소를 고치거나 모델을 더 큰 파이프라인에 넣기가 쉬워진다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 통계적 추론 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
기본 초기화 상태의 `LogisticRegressionModel`에서 학습 가능한 매개변수의 총 개수를 계산하라. 가중치와 편향을 모두 포함하여 층별로 나누어 세어라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)` 각각에는 `in_features * out_features`개의 가중치 매개변수와 (`bias=False`가 아닌 한) `out_features`개의 편향 매개변수가 있다. `nn.Conv2d(in_c, out_c, k)`에는 `in_c * out_c * k * k`개의 가중치와 `out_c`개의 편향이 있다. `nn.Embedding(num, dim)`에는 `num * dim`개의 매개변수가 있다. 모든 층에 대해 더하면 된다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

---

**연습문제 2.**
최적화기를 Adam으로 바꾸고(`torch.optim.Adam`에 `lr=0.001`을 쓴다) 원래 최적화기와 학습 수렴을 비교하라. 두 손실 곡선을 같은 그래프에 그려라.

??? success "연습문제 2 풀이"
    최적화기를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 바꾼다. Adam은 매개변수마다 적응적인 학습률과 운동량 추정값을 유지하므로 초반 에폭에서 대체로 더 빠르게 수렴한다. Adam의 손실 곡선은 보통 처음 몇 에폭에서 더 가파르게 떨어지지만, 최적점 근처에서는 운동량을 쓴 SGD보다 조금 더 흔들릴 수 있다. 공정한 비교를 위해 둘을 같은 난수 씨앗과 같은 에폭 수로 실행하라.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
층이나 블록의 개수를 설정할 수 있도록 `LogisticRegressionModel`를 확장하라. `__init__`에 `num_layers` 매개변수를 추가하고 `nn.ModuleList`로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 순회한다. (평범한 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 PyTorch가 모든 매개변수를 최적화 대상으로 등록한다. 시험은 다음과 같이 한다. `for n in [2, 4, 8]: model = LogisticRegressionModel(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
