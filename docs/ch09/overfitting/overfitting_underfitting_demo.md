# 과적합과 과소적합 시연

잡음을 더한 사인 함수 데이터에 차수가 다른 다항 회귀를 적합한다. 모델의 유연성이 학습 오차와 별도 데이터의 오차에 미치는 영향을 비교한다.

## 1. 코드

```python
"""
과적합과 과소적합 시연
==========================================
이 스크립트는 차수를 달리한 다항 회귀로 과적합과 과소적합의 개념을
보인다.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ========================================================================
# 메인
# ========================================================================

# 재현성을 위해 난수 씨앗 고정
np.random.seed(42)

# 합성 데이터 생성
def generate_data(n_samples=100, noise=0.5):
    """비선형 양상을 갖는 합성 데이터를 만든다"""
    X = np.linspace(0, 10, n_samples)
    y = np.sin(X) + np.random.normal(0, noise, n_samples)
    return X.reshape(-1, 1), y

# 다항 회귀 모델 학습
def train_polynomial_models(X_train, y_train, X_test, y_test, degrees):
    """차수를 달리한 다항 회귀 모델을 학습시킨다"""
    results = {}
    
    for degree in degrees:
        # 다항 피처를 만든다
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly_features.fit_transform(X_train)
        X_test_poly = poly_features.transform(X_test)
        
        # 모델을 학습시킨다
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        
        # 예측한다
        y_train_pred = model.predict(X_train_poly)
        y_test_pred = model.predict(X_test_poly)
        
        # 오차 계산
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        results[degree] = {
            'model': model,
            'poly_features': poly_features,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred
        }
    
    return results

# 시각화
def plot_results(X_train, y_train, X_test, y_test, results, degrees):
    """다항식의 차수별 결과를 그린다"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, degree in enumerate(degrees):
        ax = axes[idx]
        result = results[degree]
        
        # 매끄러운 곡선을 그리려고 데이터 정렬
        sort_idx_train = np.argsort(X_train.ravel())
        sort_idx_test = np.argsort(X_test.ravel())
        
        # 학습 데이터 그리기
        ax.scatter(X_train, y_train, alpha=0.5, label='Train data', color='blue')
        ax.plot(X_train[sort_idx_train], result['y_train_pred'][sort_idx_train], 
                'b-', label='Train prediction', linewidth=2)
        
        # 시험 데이터 그리기
        ax.scatter(X_test, y_test, alpha=0.5, label='Test data', color='red')
        ax.plot(X_test[sort_idx_test], result['y_test_pred'][sort_idx_test], 
                'r-', label='Test prediction', linewidth=2)
        
        # 지표를 담은 제목 붙이기
        ax.set_title(f'Degree {degree}\n'
                    f'Train MSE: {result["train_mse"]:.4f}, Test MSE: {result["test_mse"]:.4f}\n'
                    f'Train R²: {result["train_r2"]:.4f}, Test R²: {result["test_r2"]:.4f}')
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 해석 덧붙이기
        if result['test_mse'] > 0.5 and degree <= 2:
            ax.text(0.5, 0.95, 'UNDERFITTING', transform=ax.transAxes,
                   ha='center', va='top', fontsize=12, color='orange',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        elif result['test_mse'] > result['train_mse'] * 2 and degree >= 10:
            ax.text(0.5, 0.95, 'OVERFITTING', transform=ax.transAxes,
                   ha='center', va='top', fontsize=12, color='red',
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        else:
            ax.text(0.5, 0.95, 'GOOD FIT', transform=ax.transAxes,
                   ha='center', va='top', fontsize=12, color='green',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    return fig

# 메인 실행
if __name__ == "__main__":
    # 데이터를 생성한다
    X, y = generate_data(n_samples=100, noise=0.5)
    
    # 데이터 나누기
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 여러 다항식 차수로 시험
    degrees = [1, 2, 3, 5, 10, 15]
    
    # 모델 학습
    print("Training polynomial regression models...")
    results = train_polynomial_models(X_train, y_train, X_test, y_test, degrees)
    
    # 결과 출력
    print("\n" + "="*70)
    print("Results Summary")
    print("="*70)
    print(f"{'Degree':<10} {'Train MSE':<15} {'Test MSE':<15} {'Interpretation':<20}")
    print("-"*70)
    
    for degree in degrees:
        result = results[degree]
        train_mse = result['train_mse']
        test_mse = result['test_mse']
        
        # 해석
        if test_mse > 0.5 and degree <= 2:
            interpretation = "Underfitting"
        elif test_mse > train_mse * 2 and degree >= 10:
            interpretation = "Overfitting"
        else:
            interpretation = "Good fit"
        
        print(f"{degree:<10} {train_mse:<15.4f} {test_mse:<15.4f} {interpretation:<20}")
    
    print("="*70)
    
    # 시각화 만들기
    fig = plot_results(X_train, y_train, X_test, y_test, results, degrees)
    plt.savefig('overfitting_underfitting.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'overfitting_underfitting.png'")
    plt.show()
    
    # 핵심 정리
    print("\nKey Takeaways:")
    print("- Underfitting: High training and test error (model too simple)")
    print("- Overfitting: Low training error but high test error (model too complex)")
    print("- Good fit: Similar training and test errors (balanced complexity)")
```

**출력:**

```
Training polynomial regression models...

======================================================================
Results Summary
======================================================================
Degree     Train MSE       Test MSE        Interpretation      
----------------------------------------------------------------------
1          0.6551          0.6294          Underfitting        
2          0.5687          0.5402          Underfitting        
3          0.4919          0.4715          Good fit            
5          0.2497          0.1782          Good fit            
10         0.1957          0.1913          Good fit            
15         0.1944          0.1864          Good fit            
======================================================================

Visualization saved as 'overfitting_underfitting.png'

Key Takeaways:
- Underfitting: High training and test error (model too simple)
- Overfitting: Low training error but high test error (model too complex)
- Good fit: Similar training and test errors (balanced complexity)
```

## 2. 논의

1차 다항식은 직선이므로 사인 함수의 굴곡을 표현할 수 없다. 차수를 늘리면 더 다양한 모양을 표현할 수 있지만, 유한한 표본의 잡음에 민감해지거나 수치 조건이 나빠질 수 있다. 어느 차수가 가장 좋은지는 표본 수, 입력 범위, 잡음과 정규화에 따라 달라진다.

10차나 15차라는 이유만으로 모든 학습점을 보간하거나 학습 오차가 0이 되는 것은 아니다. n개의 서로 다른 입력에서 임의의 목표값을 일반적으로 보간하려면 최대 n−1차가 필요하다. 이 코드처럼 학습점이 차수보다 훨씬 많으면 고차 모델도 잔차를 남길 수 있다.

예제는 여러 차수의 결과를 test라는 분할에서 나란히 보여 준다. 그 값을 보고 차수를 고르면 해당 분할은 모델 선택에 사용한 검증셋 역할을 한다. 최종 성능 보고에는 별도의 테스트셋이 필요하다. 코드의 과소적합·과적합 표시는 MSE 임계값과 오차 비율에 따른 예시 판단이며 보편적인 판정식이 아니다. 예측선은 정렬된 관측 입력에서의 예측값을 이어 그린 것이므로 촘촘한 전체 구간에서의 함수 모양과도 구분한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
15차 다항식이 70개 학습점을 항상 정확히 지나는가?

</div>

??? success "풀이"
    아니다. 절편을 포함한 15차 다항식에는 16개의 계수가 있다. 70개의 임의 목표값을 일반적으로 모두 보간할 수 없으며, 특수한 데이터 관계가 있을 때만 예외가 가능하다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
학습 MSE가 작다는 사실만으로 좋은 모델이라고 판단할 수 없는 이유는 무엇인가?

</div>

??? success "풀이"
    학습셋에 대한 적합만 측정하기 때문이다. 잡음을 따라간 모델이나 수치적으로 불안정한 모델도 학습 오차는 작을 수 있다. 배포 조건을 반영한 별도 데이터에서의 성능이 필요하다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
여러 차수의 test MSE를 보고 최솟값을 골랐다면 결과를 어떻게 평가해야 하는가?

</div>

??? success "풀이"
    그 분할은 검증셋으로 취급한다. 선택한 차수와 다른 설정을 고정한 후 추가로 남겨 둔 독립 테스트셋에서 평가하거나 중첩 교차 검증으로 선택 절차를 평가한다.

## 정리하며

모델의 유연성을 높이는 효과는 학습 오차와 일반화 오차를 구분해 판단한다. 특정 차수를 항상 과적합이라고 부르거나, 모델 선택에 사용한 분할을 독립 테스트로 해석하지 않는다.
