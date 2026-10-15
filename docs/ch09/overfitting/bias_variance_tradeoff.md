# 편향-분산 절충

제곱 오차는 학습 데이터에 대한 평균 예측의 편향 제곱, 예측의 분산, 관측 잡음으로 나누어 해석할 수 있다. 이 예제는 결정 트리 깊이와 부트스트랩 재표집으로 그 관계를 근사적으로 살펴본다.

## 1. 코드

```python
"""
편향-분산 절충 시연
====================================
이 스크립트는 기대 예측 오차를 분해하여 편향-분산 절충의 개념을
보인다.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# ========================================================================
# 메인
# ========================================================================

# 난수 씨앗 고정
np.random.seed(42)

def generate_data(n_samples=200, noise_std=0.3):
    """알려진 함수로 합성 데이터를 만든다"""
    X = np.linspace(0, 10, n_samples)
    # 참 함수: sin(x)
    y_true = np.sin(X)
    # 잡음 더하기
    y = y_true + np.random.normal(0, noise_std, n_samples)
    return X.reshape(-1, 1), y, y_true

def compute_bias_variance(X_train, y_train, X_test, y_test_true, 
                          max_depth, n_iterations=100):
    """
    부트스트랩 표본으로 여러 번 학습시켜 모델의 편향과 분산을 계산한다.
    
    
    기대 오차 = 편향² + 분산 + 줄일 수 없는 오차
    """
    predictions = []
    
    for i in range(n_iterations):
        # 부트스트랩 표본
        indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_boot = X_train[indices]
        y_boot = y_train[indices]
        
        # 모델을 학습시킨다
        model = DecisionTreeRegressor(max_depth=max_depth, random_state=i)
        model.fit(X_boot, y_boot)
        
        # 시험 집합에 대해 예측
        y_pred = model.predict(X_test)
        predictions.append(y_pred)
    
    predictions = np.array(predictions)
    
    # 편향 계산
    mean_prediction = np.mean(predictions, axis=0)
    bias = np.mean((mean_prediction - y_test_true) ** 2)
    
    # 분산 계산
    variance = np.mean(np.var(predictions, axis=0))
    
    # 전체 기대 오차 (편향² + 분산)
    expected_error = bias + variance
    
    return bias, variance, expected_error, predictions, mean_prediction

def analyze_complexity_range(X_train, y_train, X_test, y_test_true):
    """모델의 복잡도를 달리하며 편향-분산 절충을 분석한다"""
    max_depths = range(1, 16)
    biases = []
    variances = []
    total_errors = []
    
    print("Analyzing bias-variance tradeoff...")
    for depth in max_depths:
        print(f"Processing max_depth={depth}...")
        bias, variance, total_error, _, _ = compute_bias_variance(
            X_train, y_train, X_test, y_test_true, 
            max_depth=depth, n_iterations=50
        )
        biases.append(bias)
        variances.append(variance)
        total_errors.append(total_error)
    
    return max_depths, biases, variances, total_errors

def plot_bias_variance_tradeoff(max_depths, biases, variances, total_errors):
    """편향-분산 절충을 그린다"""
    plt.figure(figsize=(12, 5))
    
    # 그림 1: 편향-분산 절충
    plt.subplot(1, 2, 1)
    plt.plot(max_depths, biases, 'b-o', label='Bias²', linewidth=2)
    plt.plot(max_depths, variances, 'r-s', label='Variance', linewidth=2)
    plt.plot(max_depths, total_errors, 'g-^', label='Total Error (Bias² + Variance)', 
             linewidth=2, markersize=8)
    
    # 최적의 복잡도 찾기
    optimal_idx = np.argmin(total_errors)
    optimal_depth = max_depths[optimal_idx]
    plt.axvline(x=optimal_depth, color='purple', linestyle='--', 
                label=f'Optimal Complexity (depth={optimal_depth})')
    
    plt.xlabel('Model Complexity (Max Depth)', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.title('Bias-Variance Tradeoff', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 그림 2: 성분 분석
    plt.subplot(1, 2, 2)
    width = 0.35
    x = np.arange(len(max_depths))
    
    plt.bar(x, biases, width, label='Bias²', alpha=0.8)
    plt.bar(x, variances, width, bottom=biases, label='Variance', alpha=0.8)
    
    plt.xlabel('Model Complexity (Max Depth)', fontsize=12)
    plt.ylabel('Error Components', fontsize=12)
    plt.title('Stacked Error Components', fontsize=14, fontweight='bold')
    plt.xticks(x[::2], [max_depths[i] for i in range(0, len(max_depths), 2)])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return plt.gcf()

def visualize_predictions_by_complexity(X_train, y_train, X_test, y_test_true):
    """복잡도 수준마다의 예측을 보인다"""
    complexities = [1, 3, 7, 15]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, depth in enumerate(complexities):
        ax = axes[idx]
        
        bias, variance, total_error, predictions, mean_pred = compute_bias_variance(
            X_train, y_train, X_test, y_test_true, 
            max_depth=depth, n_iterations=30
        )
        
        # 그림을 그리기 위해 정렬한다
        sort_idx = np.argsort(X_test.ravel())
        
        # 개별 예측 그리기 (모델의 변동을 보인다)
        for pred in predictions[:10]:  # 알아보기 쉽게 처음 10개만 보인다
            ax.plot(X_test[sort_idx], pred[sort_idx], 'gray', alpha=0.2, linewidth=0.5)
        
        # 평균 예측 그리기
        ax.plot(X_test[sort_idx], mean_pred[sort_idx], 'b-', 
                label='Mean Prediction', linewidth=2)
        
        # 참 함수 그리기
        ax.plot(X_test[sort_idx], y_test_true[sort_idx], 'g-', 
                label='True Function', linewidth=2)
        
        # 학습 데이터 그리기
        ax.scatter(X_train, y_train, alpha=0.3, s=20, c='red', label='Training Data')
        
        ax.set_title(f'Max Depth = {depth}\n'
                    f'Bias² = {bias:.4f}, Variance = {variance:.4f}\n'
                    f'Total Error = {total_error:.4f}',
                    fontsize=11)
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 해석 덧붙이기
        if bias > variance:
            interpretation = "HIGH BIAS\n(Underfitting)"
            color = 'orange'
        elif variance > bias * 2:
            interpretation = "HIGH VARIANCE\n(Overfitting)"
            color = 'red'
        else:
            interpretation = "BALANCED"
            color = 'green'
        
        ax.text(0.05, 0.95, interpretation, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    plt.tight_layout()
    return fig

# 메인 실행
if __name__ == "__main__":
    print("="*70)
    print("Bias-Variance Tradeoff Analysis")
    print("="*70)
    
    # 데이터를 생성한다
    X, y, y_true = generate_data(n_samples=200, noise_std=0.3)
    
    # 데이터 나누기
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 시험 집합의 참값 얻기
    y_test_true = np.sin(X_test.ravel())
    
    # 복잡도를 달리하며 편향-분산 분석
    max_depths, biases, variances, total_errors = analyze_complexity_range(
        X_train, y_train, X_test, y_test_true
    )
    
    # 요약 출력
    print("\n" + "="*70)
    print("Bias-Variance Analysis Summary")
    print("="*70)
    print(f"{'Max Depth':<12} {'Bias²':<15} {'Variance':<15} {'Total Error':<15}")
    print("-"*70)
    
    for depth, bias, var, total in zip(max_depths, biases, variances, total_errors):
        print(f"{depth:<12} {bias:<15.4f} {var:<15.4f} {total:<15.4f}")
    
    optimal_idx = np.argmin(total_errors)
    print("="*70)
    print(f"Optimal Complexity: Max Depth = {max_depths[optimal_idx]}")
    print(f"Minimum Total Error: {total_errors[optimal_idx]:.4f}")
    print("="*70)
    
    # 시각화 만들기
    fig1 = plot_bias_variance_tradeoff(max_depths, biases, variances, total_errors)
    plt.savefig('bias_variance_tradeoff.png', dpi=150, bbox_inches='tight')
    print("\nBias-variance tradeoff plot saved as 'bias_variance_tradeoff.png'")
    
    fig2 = visualize_predictions_by_complexity(X_train, y_train, X_test, y_test_true)
    plt.savefig('bias_variance_predictions.png', dpi=150, bbox_inches='tight')
    print("Predictions visualization saved as 'bias_variance_predictions.png'")
    
    plt.show()
    
    # 핵심 개념
    print("\n" + "="*70)
    print("Key Concepts:")
    print("="*70)
    print("• BIAS: Error from incorrect assumptions (underfitting)")
    print("  - High bias → model too simple → systematic errors")
    print("  - Low complexity models have high bias")
    print("\n• VARIANCE: Error from sensitivity to training data (overfitting)")
    print("  - High variance → model too complex → unstable predictions")
    print("  - High complexity models have high variance")
    print("\n• TRADEOFF: As complexity increases:")
    print("  - Bias decreases (model captures more patterns)")
    print("  - Variance increases (model becomes more sensitive)")
    print("\n• OPTIMAL MODEL: Minimizes (Bias² + Variance)")
    print("="*70)
```

**출력:**

```
======================================================================
Bias-Variance Tradeoff Analysis
======================================================================
Analyzing bias-variance tradeoff...
Processing max_depth=1...
Processing max_depth=2...
Processing max_depth=3...
Processing max_depth=4...
Processing max_depth=5...
Processing max_depth=6...
Processing max_depth=7...
Processing max_depth=8...
Processing max_depth=9...
Processing max_depth=10...
Processing max_depth=11...
Processing max_depth=12...
Processing max_depth=13...
Processing max_depth=14...
Processing max_depth=15...

======================================================================
Bias-Variance Analysis Summary
======================================================================
Max Depth    Bias²           Variance        Total Error    
----------------------------------------------------------------------
1            0.3250          0.0241          0.3491         
2            0.0741          0.0750          0.1491         
3            0.0195          0.0791          0.0987         
4            0.0185          0.0395          0.0579         
5            0.0227          0.0388          0.0614         
6            0.0279          0.0431          0.0710         
7            0.0337          0.0473          0.0810         
8            0.0380          0.0438          0.0818         
9            0.0398          0.0473          0.0871         
10           0.0425          0.0496          0.0921         
11           0.0417          0.0492          0.0908         
12           0.0379          0.0500          0.0878         
13           0.0445          0.0443          0.0888         
14           0.0351          0.0511          0.0862         
15           0.0397          0.0461          0.0858         
======================================================================
Optimal Complexity: Max Depth = 4
Minimum Total Error: 0.0579
======================================================================

Bias-variance tradeoff plot saved as 'bias_variance_tradeoff.png'
Predictions visualization saved as 'bias_variance_predictions.png'

======================================================================
Key Concepts:
======================================================================
• BIAS: Error from incorrect assumptions (underfitting)
  - High bias → model too simple → systematic errors
  - Low complexity models have high bias

• VARIANCE: Error from sensitivity to training data (overfitting)
  - High variance → model too complex → unstable predictions
  - High complexity models have high variance

• TRADEOFF: As complexity increases:
  - Bias decreases (model captures more patterns)
  - Variance increases (model becomes more sensitive)

• OPTIMAL MODEL: Minimizes (Bias² + Variance)
======================================================================
```

## 2. 논의

고정된 입력 x에서 $Y=f(x)+\varepsilon$, $E[\varepsilon\mid x]=0$이고 새 관측 잡음이 학습 과정과 독립이면 기대 제곱 오차는 $(E_D[\hat f_D(x)]-f(x))^2+\operatorname{Var}_D(\hat f_D(x))+\operatorname{Var}(\varepsilon\mid x)$로 분해된다. 이는 제곱 오차에 대한 분해이며 모든 손실에 그대로 적용되는 식은 아니다.

코드는 하나의 학습셋에서 복원 추출한 표본들로 트리를 학습한다. 변수 bias에는 편향 자체가 아니라 평균 예측과 참 함수 sin(x)의 차이를 제곱해 입력들에 걸쳐 평균한 값이 담긴다. variance는 부트스트랩 예측의 변동이다. expected_error에는 이 둘만 더하므로 새 관측값에 대한 전체 MSE의 잡음 항은 포함하지 않는다.

부트스트랩 변동은 데이터 생성 과정에서 독립적인 학습셋을 반복 추출한 변동의 근사이다. 깊이를 늘릴 때 편향이 반드시 단조 감소하거나 분산이 반드시 단조 증가하지는 않는다. 그림의 최적 깊이는 시도한 깊이 중 이 근사 기준이 가장 작은 값이다. 같은 평가 입력으로 깊이를 고르면 이후의 독립적인 최종 평가가 필요하다. 앙상블 트리 수를 늘리는 경우처럼 복잡도를 늘려 분산을 줄일 수 있는 상황도 있어, 모든 복잡도 변수에 같은 두 곡선 모양을 가정하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
코드의 bias와 expected_error가 각각 무엇을 뜻하는가?

</div>

??? success "연습문제 1 풀이"
    bias는 부트스트랩 평균 예측과 참 함수 사이의 평균 제곱 차이이다. expected_error는 그 값과 부트스트랩 예측 분산의 합이며 관측 잡음의 분산은 제외되어 있다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
생성 잡음의 표준편차가 0.3일 때 새 관측값에 대한 기대 MSE에 추가되는 항은 얼마인가?

</div>

??? success "연습문제 2 풀이"
    독립적인 평균 0 잡음의 분산인 0.3²=0.09이다. 유한한 표본에서 계산한 관측 MSE가 근사 오차에 정확히 0.09를 더한 값과 같아야 하는 것은 아니다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
부트스트랩 대신 데이터 생성 과정을 반복하면 무엇을 직접 추정할 수 있는가?

</div>

??? success "연습문제 3 풀이"
    각 반복에서 독립적인 학습셋을 새로 만들고 같은 입력 격자에서 예측하면, 학습 데이터 생성 과정에 따른 편향과 분산을 직접 근사한다. 하나의 표본에 조건을 둔 부트스트랩 근사와 구분된다.

## 정리하며

이 예제는 참 함수를 아는 합성 데이터에서 편향 제곱과 분산을 근사한다. 잡음 항의 포함 여부와 재표집의 의미를 구분해야 전체 예측 오차와 연결할 수 있다.
