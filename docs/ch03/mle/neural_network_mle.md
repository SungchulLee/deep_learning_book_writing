# 신경망 MLE

신경망 MLE - 최대가능도로 하는 딥러닝. 핵심 통찰: 신경망 학습이 곧 최대가능도 추정이다!

이 튜토리얼은 PyTorch에서 최대가능도 추정에 대한 기초적인 이해를 쌓는다. 코드를 따라가 보면 모델을 세우고, 손실 함수를 정의하고, 경사 하강법으로 학습시키고, 분류 과제에서 성능을 평가하는 법을 알 수 있다.

## 코드

```python
#!/usr/bin/env python3
"""
================================================================================
NEURAL NETWORK MLE - Deep Learning with Maximum Likelihood
================================================================================

DIFFICULTY: ⭐⭐⭐ Advanced (Level 3)

LEARNING OBJECTIVES:
- Understand how neural networks use MLE
- See the connection between loss functions and likelihood
- Implement custom MLE-based losses
- Learn about heteroscedastic regression (predicting uncertainty)

KEY INSIGHT: Neural network training IS maximum likelihood estimation!

STANDARD REGRESSION:
- Network predicts: ŷ = f(x; θ)
- Assume: y ~ N(ŷ, σ²) with fixed σ
- MLE objective: minimize Σ(y - ŷ)² (MSE loss)

HETEROSCEDASTIC REGRESSION:
- Network predicts BOTH mean AND variance: (μ̂, σ̂²) = f(x; θ)
- Model: y ~ N(μ̂, σ̂²) with varying σ
- MLE objective: maximize Σ log N(y | μ̂, σ̂²)
             = minimize Σ [log(σ̂²) + (y - μ̂)²/σ̂²]

This allows the network to express UNCERTAINTY in its predictions!

APPLICATIONS:
- Regression with uncertainty quantification
- Robust regression (outlier handling)
- Active learning (query high-uncertainty points)
- Risk-sensitive decision making

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


def generate_heteroscedastic_data(n_samples: int = 300, seed: int = 42):
    """
    Generate data where noise varies with x (heteroscedastic).
    
    y = sin(x) + ε, where ε ~ N(0, σ(x)²) and σ(x) increases with |x|
    """
    torch.manual_seed(seed)
    
    # x 값을 생성한다
    x = torch.rand(n_samples, 1) * 10 - 5  # Range: [-5, 5]
    
    # 참 함수: 사인파
    y_true = torch.sin(x)
    
    # 이분산 잡음: σ(x) = 0.1 + 0.1 * |x|
    sigma_x = 0.1 + 0.1 * torch.abs(x)
    noise = torch.randn_like(x) * sigma_x
    
    y = y_true + noise
    
    return x, y, sigma_x


class StandardNN(nn.Module):
    """평균만 예측하는 표준 신경망"""
    
    def __init__(self, hidden_size=50):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.network(x)


class HeteroscedasticNN(nn.Module):
    """
    Neural network predicting BOTH mean and variance.
    
    This is the MLE approach for heteroscedastic regression!
    """
    
    def __init__(self, hidden_size=50):
        super().__init__()
        
        # 공유되는 은닉층
        self.shared = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # 평균 갈래
        self.mean_head = nn.Linear(hidden_size, 1)
        
        # 로그 분산 갈래 (σ² > 0을 보장하려고 log(σ²)을 예측한다)
        self.logvar_head = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        """
        Returns:
            mean: Predicted mean
            logvar: Predicted log-variance (log(σ²))
        """
        features = self.shared(x)
        mean = self.mean_head(features)
        logvar = self.logvar_head(features)
        return mean, logvar


def gaussian_nll_loss(y_true, y_pred_mean, y_pred_logvar):
    """
    Gaussian Negative Log-Likelihood loss.
    
    This is the MLE objective for heteroscedastic regression!
    
    NLL = -log N(y | μ, σ²)
        = 0.5 * [log(2π) + log(σ²) + (y - μ)² / σ²]
    
    Ignoring constants:
    NLL = 0.5 * [log(σ²) + (y - μ)² / σ²]
        = 0.5 * [log_var + (y - μ)² / exp(log_var)]
    """
    # 음의 로그가능도를 계산한다
    variance = torch.exp(y_pred_logvar)
    loss = 0.5 * (y_pred_logvar + (y_true - y_pred_mean) ** 2 / variance)
    
    return loss.mean()


def train_standard_nn(x_train, y_train, epochs=1000, lr=0.01):
    """MSE 손실로 표준 신경망을 학습시킨다"""
    
    model = StandardNN(hidden_size=50)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    history = []
    
    for epoch in range(epochs):
        # 순전파
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        history.append(loss.item())
        
        if (epoch + 1) % 200 == 0:
            print(f"   Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return model, history


def train_heteroscedastic_nn(x_train, y_train, epochs=1000, lr=0.01):
    """사용자 정의 MLE 손실로 이분산 신경망을 학습시킨다"""
    
    model = HeteroscedasticNN(hidden_size=50)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = []
    
    for epoch in range(epochs):
        # 순전파
        y_pred_mean, y_pred_logvar = model(x_train)
        
        # 음의 로그가능도를 계산한다 (우리의 MLE 목표이다!)
        loss = gaussian_nll_loss(y_train, y_pred_mean, y_pred_logvar)
        
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        history.append(loss.item())
        
        if (epoch + 1) % 200 == 0:
            print(f"   Epoch {epoch+1}/{epochs}, NLL Loss: {loss.item():.4f}")
    
    return model, history


def visualize_results(x, y, x_test, true_sigma, model_standard, model_hetero):
    """종합적인 시각화를 만든다"""
    
    fig = plt.figure(figsize=(18, 12))
    
    # 예측을 얻는다
    with torch.no_grad():
        # 표준 모델
        y_pred_standard = model_standard(x_test)
        
        # 이분산 모델
        y_pred_mean, y_pred_logvar = model_hetero(x_test)
        y_pred_std = torch.sqrt(torch.exp(y_pred_logvar))
    
    x_np = x.numpy().flatten()
    y_np = y.numpy().flatten()
    x_test_np = x_test.numpy().flatten()
    
    # ================================================================
    # 그림 1: 표준 신경망의 예측
    # ================================================================
    ax1 = plt.subplot(2, 3, 1)
    
    # 그림을 그리기 위해 정렬한다
    sort_idx = torch.argsort(x_test.flatten())
    x_sorted = x_test_np[sort_idx]
    y_pred_sorted = y_pred_standard.numpy().flatten()[sort_idx]
    
    ax1.scatter(x_np, y_np, alpha=0.5, s=20, label='Data', color='blue')
    ax1.plot(x_sorted, y_pred_sorted, 'r-', linewidth=2, label='Standard NN')
    ax1.plot(x_sorted, np.sin(x_sorted), 'g--', linewidth=2, label='True function')
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Standard NN (MSE Loss)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ================================================================
    # 그림 2: 불확실성을 함께 낸 이분산 신경망
    # ================================================================
    ax2 = plt.subplot(2, 3, 2)
    
    y_mean_sorted = y_pred_mean.numpy().flatten()[sort_idx]
    y_std_sorted = y_pred_std.numpy().flatten()[sort_idx]
    
    ax2.scatter(x_np, y_np, alpha=0.5, s=20, label='Data', color='blue')
    ax2.plot(x_sorted, y_mean_sorted, 'r-', linewidth=2, label='Predicted mean')
    ax2.plot(x_sorted, np.sin(x_sorted), 'g--', linewidth=2, label='True function')
    
    # 불확실성 띠를 그린다 (±1σ, ±2σ)
    ax2.fill_between(x_sorted, 
                     y_mean_sorted - 2*y_std_sorted,
                     y_mean_sorted + 2*y_std_sorted,
                     alpha=0.2, color='red', label='±2σ (95% CI)')
    ax2.fill_between(x_sorted,
                     y_mean_sorted - y_std_sorted,
                     y_mean_sorted + y_std_sorted,
                     alpha=0.3, color='red', label='±1σ (68% CI)')
    
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_title('Heteroscedastic NN (MLE Loss)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ================================================================
    # 그림 3: 예측 불확실성과 참 불확실성
    # ================================================================
    ax3 = plt.subplot(2, 3, 3)
    
    true_sigma_sorted = true_sigma.numpy().flatten()[sort_idx]
    
    ax3.plot(x_sorted, true_sigma_sorted, 'g-', linewidth=3, label='True σ(x)')
    ax3.plot(x_sorted, y_std_sorted, 'r-', linewidth=3, label='Predicted σ(x)')
    ax3.fill_between(x_sorted, 0, true_sigma_sorted, alpha=0.2, color='green')
    
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('σ (Standard Deviation)', fontsize=12)
    ax3.set_title('Uncertainty Estimation', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # ================================================================
    # 그림 4: 잔차 비교
    # ================================================================
    ax4 = plt.subplot(2, 3, 4)
    
    residuals_standard = (y - model_standard(x)).numpy().flatten()
    residuals_hetero = (y - y_pred_mean).numpy().flatten()
    
    ax4.scatter(x_np, residuals_standard, alpha=0.5, s=20, label='Standard NN', color='blue')
    ax4.scatter(x_np, residuals_hetero, alpha=0.5, s=20, label='Heteroscedastic NN', color='red')
    ax4.axhline(0, color='black', linestyle='--', linewidth=2)
    
    ax4.set_xlabel('x', fontsize=12)
    ax4.set_ylabel('Residuals', fontsize=12)
    ax4.set_title('Residual Analysis', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # ================================================================
    # 그림 5: 보정 그림
    # ================================================================
    ax5 = plt.subplot(2, 3, 5)
    
    # 이분산 모델의 정규화된 잔차를 계산한다
    residuals = (y - y_pred_mean).numpy().flatten()
    predicted_stds = y_pred_std.numpy().flatten()
    normalized_residuals = residuals / predicted_stds
    
    # 정규화된 잔차의 히스토그램 (잘 보정되었다면 N(0,1)이어야 한다)
    ax5.hist(normalized_residuals, bins=30, density=True, alpha=0.7, 
            edgecolor='black', label='Normalized Residuals')
    
    # N(0,1) 분포를 겹쳐 그린다
    x_range = np.linspace(-4, 4, 100)
    from scipy.stats import norm
    ax5.plot(x_range, norm.pdf(x_range), 'r-', linewidth=2, label='N(0,1)')
    
    ax5.set_xlabel('Normalized Residuals', fontsize=12)
    ax5.set_ylabel('Density', fontsize=12)
    ax5.set_title('Calibration Check', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # ================================================================
    # 그림 6: 로그가능도 비교
    # ================================================================
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # 로그가능도들을 계산한다
    with torch.no_grad():
        # 표준 모델 (σ = 1로 고정되어 있다고 가정)
        residuals_std = y - model_standard(x)
        nll_standard = 0.5 * (np.log(2 * np.pi) + torch.mean(residuals_std ** 2)).item()
        
        # 이분산 모델
        mean_het, logvar_het = model_hetero(x)
        nll_hetero = gaussian_nll_loss(y, mean_het, logvar_het).item()
    
    # 비교 표
    table_data = [
        ['Model', 'Negative Log-Likelihood'],
        ['Standard NN', f'{nll_standard:.4f}'],
        ['Heteroscedastic NN', f'{nll_hetero:.4f}'],
        ['Improvement', f'{nll_standard - nll_hetero:.4f}'],
    ]
    
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 3)
    
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 개선된 부분을 강조한다
    table[(3, 0)].set_facecolor('#FFF9C4')
    table[(3, 1)].set_facecolor('#FFF9C4')
    
    ax6.set_title('Model Comparison (Lower is Better)', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('neural_network_mle_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 Figure saved as 'neural_network_mle_results.png'")
    plt.show()


def main():
    print("=" * 80)
    print("NEURAL NETWORK MLE - Deep Learning with Uncertainty")
    print("=" * 80)
    
    # 데이터를 생성한다
    print("\n🎲 Generating heteroscedastic data...")
    x_train, y_train, true_sigma = generate_heteroscedastic_data(n_samples=300)
    
    # 매끄러운 예측을 위한 시험 데이터
    x_test = torch.linspace(-5, 5, 200).unsqueeze(1)
    
    print(f"   • Training samples: {len(x_train)}")
    print(f"   • Noise varies with x (heteroscedastic)")
    
    # 표준 신경망을 학습시킨다
    print("\n🔵 Training Standard NN (MSE Loss)...")
    print("-" * 80)
    model_standard, history_standard = train_standard_nn(x_train, y_train, epochs=1000, lr=0.01)
    
    # 이분산 신경망을 학습시킨다
    print("\n🔴 Training Heteroscedastic NN (MLE Loss)...")
    print("-" * 80)
    model_hetero, history_hetero = train_heteroscedastic_nn(x_train, y_train, epochs=1000, lr=0.01)
    
    # 평가
    print("\n📊 Evaluation:")
    print("-" * 80)
    
    with torch.no_grad():
        # 표준 모델
        y_pred_std = model_standard(x_train)
        mse_std = torch.mean((y_train - y_pred_std) ** 2).item()
        
        # 이분산 모델
        y_pred_mean, y_pred_logvar = model_hetero(x_train)
        mse_het = torch.mean((y_train - y_pred_mean) ** 2).item()
        nll_het = gaussian_nll_loss(y_train, y_pred_mean, y_pred_logvar).item()
    
    print(f"   Standard NN:")
    print(f"      MSE: {mse_std:.4f}")
    
    print(f"\n   Heteroscedastic NN:")
    print(f"      MSE: {mse_het:.4f}")
    print(f"      NLL: {nll_het:.4f}")
    
    # 시각화한다
    print("\n📊 Creating visualizations...")
    visualize_results(x_train, y_train, x_test, true_sigma, model_standard, model_hetero)
    
    print("\n" + "=" * 80)
    print("✅ COMPLETE!")
    print("=" * 80)
    print("\n💡 KEY TAKEAWAYS:")
    print("   1. Neural networks ARE MLE when trained with appropriate losses")
    print("   2. MSE = MLE with Gaussian assumption and fixed variance")
    print("   3. Heteroscedastic networks predict uncertainty!")
    print("   4. Custom loss functions = Custom probabilistic assumptions")
    print("   5. This enables uncertainty-aware deep learning")
    print("\n   🎯 Applications:")
    print("      • Medical diagnosis (quantify confidence)")
    print("      • Autonomous vehicles (safety-critical decisions)")
    print("      • Financial modeling (risk assessment)")
    print("      • Active learning (query uncertain points)")
    print("\n" + "=" * 80)


"""
🎓 EXERCISES:

1. MEDIUM: Classification with uncertainty
   - Extend to classification task
   - Predict class probabilities (softmax)
   - Use negative log-likelihood (cross-entropy)
   - Visualize prediction confidence

2. MEDIUM: Different noise models
   - Laplace noise: use absolute error instead of squared
   - Student-t noise: robust to outliers
   - Compare likelihood functions

3. CHALLENGING: Bayesian Neural Networks
   - Add dropout for uncertainty estimation
   - Monte Carlo dropout: multiple forward passes
   - Compare epistemic vs aleatoric uncertainty

4. CHALLENGING: Multi-output regression
   - Predict vector outputs with covariance
   - Full covariance matrix vs diagonal
   - Multivariate Gaussian likelihood

5. CHALLENGING: Active learning
   - Use uncertainty to select informative samples
   - Train on small dataset
   - Iteratively query high-uncertainty points
   - Show learning curve improves faster
"""


if __name__ == "__main__":
    main()```

## 논의

이 구현은 2개의 클래스(`StandardNN`, `HeteroscedasticNN`)를 정의하며, 이들이 함께 작동하여 완전한 최대가능도 추정 구조를 이룬다. 각 클래스가 서로 다른 구성 요소를 감싸므로 코드가 모듈식이 되고 확장하기 쉬워진다. `forward` 메서드들이 PyTorch가 자동 미분에 사용하는 계산 그래프를 정의한다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 통계적 추론 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
기본 초기화 상태의 `StandardNN`에서 학습 가능한 매개변수의 총 개수를 계산하라. 가중치와 편향을 모두 포함하여 층별로 나누어 세어라.

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
층이나 블록의 개수를 설정할 수 있도록 `StandardNN`를 확장하라. `__init__`에 `num_layers` 매개변수를 추가하고 `nn.ModuleList`로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 순회한다. (평범한 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 PyTorch가 모든 매개변수를 최적화 대상으로 등록한다. 시험은 다음과 같이 한다. `for n in [2, 4, 8]: model = StandardNN(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
