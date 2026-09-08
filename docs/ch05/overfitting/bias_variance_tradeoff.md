# 편향-분산 절충

편향-분산 절충은 예측 오차가 편향의 제곱, 분산, 줄일 수 없는 잡음으로 분해된다는 근본 개념이다. 편향이 크면(과소적합) 조직적인 오차가 생기고, 분산이 크면(과적합) 예측이 불안정해진다. 최적의 모델 복잡도는 이 둘의 합을 최소로 한다.

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
    print("="*70)```

## 2. 논의

이 실험은 `max_depth`를 달리한 결정 트리를 부트스트랩 표본으로 학습시키고, 따로 떼어 둔 시험 집합에서 편향(평균 예측과 참 함수의 차이의 제곱)과 분산(부트스트랩 표본에 걸친 예측의 변동)을 잰다.

얕은 트리(`max_depth`가 작은 경우)는 사인 모양을 나타내지 못하므로 편향이 크고, 참 함수에서 조직적으로 벗어난 예측을 낸다. 깊은 트리는 편향은 작지만 개별 부트스트랩 표본의 잡음까지 맞추므로 분산이 커서 매번 크게 다른 예측을 낸다.

최적의 깊이는 편향의 제곱과 분산의 합을 최소로 한다. 그림은 이 절충을 서로 엇갈리는 두 곡선으로 보여 주며, 전체 오차는 그 사이의 알맞은 지점에서 최소가 된다. 이 틀은 다항식의 차수, 층의 수, 정칙화의 강도, 앙상블 트리의 개수 등 어떤 복잡도 매개변수에도 적용된다.

## 연습문제

**연습문제 1.**
코드를 따라가며 쓰인 주요 자료 구조를 찾아라. 각각에 대해 자료형, (해당한다면) 모양, 파이프라인에서의 구실을 적어라.

??? success "연습문제 1 풀이"
    코드를 꼼꼼히 읽으며 변수 대입마다 살펴본다. 텐서는 `.shape`과 `.dtype`을 확인하고, 클래스는 `__init__`의 매개변수와 `forward`/`__call__`의 서명을 확인한다. 이름, 자료형, 모양, 구실을 열로 하는 표에 정리한다.

---


**연습문제 2.**
오류 처리와 입력 검증을 넣도록 코드를 고쳐라. 이 코드를 실전에 쓸 수 있게 하려면 어떤 검사를 더하겠는가?

??? success "연습문제 2 풀이"
    입력에 자료형 검사(`isinstance`), 모양 검증(`assert tensor.dim() == expected`), 값 범위 검사(예: 확률이 [0,1] 안인지)를 넣고, 입출력 연산은 try-except로 감싼다. 빈 배치나 NaN 같은 경계 상황에는 경고를 남긴다. 매개변수와 반환값의 자료형을 적은 독스트링을 붙인다.

---


**연습문제 3.**
직접 고른 새로운 쓰임새를 지원하도록 코드를 확장하라. 무엇을 왜 바꿀지 설명하라.

??? success "연습문제 3 풀이"
    알맞은 확장을 하나 고른다(예: 다른 데이터셋, 지표 추가, 새 모델 변형). 필요한 변경을 설명한다. 새 임포트, 클래스 정의 수정, 초매개변수 갱신, 새로운 시각화나 기록 등이다. 핵심 변경을 구현하고 간단한 시험으로 올바름을 확인한다.

## 정리하며

**다룬 것** — 편향-분산 절충

이 실험은 `max_depth`를 달리한 결정 트리를 부트스트랩 표본으로 학습시키고, 따로 떼어 둔 시험 집합에서 편향(평균 예측과 참 함수의 차이의 제곱)과 분산(부트스트랩 표본에 걸친 예측의 변동)을 잰다.

앞의 연습문제 3개로 직접 확인할 수 있다.
