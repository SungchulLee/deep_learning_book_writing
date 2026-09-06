# 과적합과 과소적합 시연

과적합은 모델이 학습 데이터의 잡음까지 외워 학습 오차는 낮지만 시험 오차는 높아지는 현상이다. 과소적합은 모델이 너무 단순하여 밑에 깔린 양상을 잡아내지 못하는 현상이다. 이 스크립트는 차수를 달리한 다항 회귀로 두 현상을 보이며 복잡도가 일반화에 어떤 영향을 주는지 드러낸다.

## 코드

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
        # 다항 특징을 만든다
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
    print("- Good fit: Similar training and test errors (balanced complexity)")```

## 논의

차수가 1인 다항 회귀(직선)는 곡률을 나타낼 수 없으므로 사인 모양의 데이터에 과소적합한다. 학습 MSE와 시험 MSE가 모두 높고 서로 비슷하다. 차수가 3~5이면 모델이 밑에 깔린 양상을 잡아내어 학습 MSE와 시험 MSE가 모두 낮고 비슷해진다. 알맞은 적합이다.

차수가 10~15이면 모델이 모든 학습 점을 지날 만큼 유연해져 학습 MSE가 거의 0이 된다. 그러나 다항식이 학습 점 사이에서 심하게 진동하여 시험 MSE는 아주 높아진다. 학습 오차와 시험 오차가 이렇게 크게 벌어지는 것이 과적합의 특징이다.

그림의 격자는 다항식의 차수가 커질 때 예측이 어떻게 바뀌는지 보여 주어, 과소적합에서 알맞은 적합을 거쳐 과적합으로 가는 흐름을 눈으로 짚게 해 준다. 핵심은 모델의 복잡도를 데이터를 만들어 낸 과정의 참 복잡도에 맞추어야 한다는 것이다.

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

