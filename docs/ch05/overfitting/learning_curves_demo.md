# 학습 곡선 시연

학습 곡선은 학습 집합의 크기에 따른 학습 오차와 검증 오차를 그려 과적합과 과소적합을 진단하는 도구가 된다. 두 곡선의 간격이 크면 분산이 크다는 뜻이고, 두 곡선이 모두 높은 오차에서 평평해지면 편향이 크다는 뜻이다. 이 곡선은 모델의 복잡도와 데이터 수집에 관한 판단을 이끈다.

## 코드

```python
"""
학습 곡선 시연
=============================
이 스크립트는 학습 곡선으로 과적합과 과소적합을 진단하는 법을
보인다.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# ========================================================================
# 메인
# ========================================================================

# 난수 씨앗 고정
np.random.seed(42)

def generate_data(n_samples=1000, noise=0.3):
    """비선형 양상을 갖는 합성 데이터를 만든다"""
    X = np.linspace(0, 10, n_samples)
    y = np.sin(X) + 0.5 * X + np.random.normal(0, noise, n_samples)
    return X.reshape(-1, 1), y

def plot_learning_curve(estimator, title, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)):
    """
    주어진 추정기의 학습 곡선을 그린다
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes,
        scoring='neg_mean_squared_error', n_jobs=-1
    )
    
    # 양수 MSE로 바꾸기
    train_scores = -train_scores
    val_scores = -val_scores
    
    # 평균과 표준편차 계산
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # 그림 만들기
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 학습 곡선 그리기
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training error', linewidth=2)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.2, color='blue')
    
    ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation error', linewidth=2)
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.2, color='red')
    
    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title(f'Learning Curve - {title}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 해석 덧붙이기
    final_train_error = train_mean[-1]
    final_val_error = val_mean[-1]
    gap = final_val_error - final_train_error
    
    if final_val_error > 0.5 and gap < 0.2:
        interpretation = "HIGH BIAS (Underfitting)\n• Both errors are high\n• Small gap between curves"
        color = 'orange'
    elif gap > 0.5:
        interpretation = "HIGH VARIANCE (Overfitting)\n• Large gap between curves\n• Low training error"
        color = 'red'
    else:
        interpretation = "GOOD FIT\n• Low errors\n• Small gap"
        color = 'green'
    
    ax.text(0.02, 0.98, interpretation, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    plt.tight_layout()
    return fig, train_mean, val_mean

def compare_learning_curves(X, y):
    """모델마다의 학습 곡선을 견준다"""
    models = [
        ('Linear Regression (Underfitting)', 
         LinearRegression()),
        
        ('Polynomial Regression (Degree 3)', 
         Pipeline([
             ('poly', PolynomialFeatures(degree=3)),
             ('linear', LinearRegression())
         ])),
        
        ('Decision Tree (max_depth=2, Underfitting)',
         DecisionTreeRegressor(max_depth=2, random_state=42)),
        
        ('Decision Tree (max_depth=20, Overfitting)',
         DecisionTreeRegressor(max_depth=20, random_state=42)),
        
        ('Random Forest (Good Fit)',
         RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)),
        
        ('Ridge Regression (Degree 10, Regularized)',
         Pipeline([
             ('poly', PolynomialFeatures(degree=10)),
             ('ridge', Ridge(alpha=1.0))
         ]))
    ]
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    axes = axes.ravel()
    
    results = []
    
    for idx, (name, model) in enumerate(models):
        print(f"\nProcessing: {name}...")
        
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y, cv=5, train_sizes=train_sizes,
            scoring='neg_mean_squared_error', n_jobs=-1
        )
        
        # 양수 MSE로 바꾸기
        train_scores = -train_scores
        val_scores = -val_scores
        
        # 통계 계산
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # 그래프 그리기
        ax = axes[idx]
        ax.plot(train_sizes_abs, train_mean, 'o-', color='blue', 
                label='Training error', linewidth=2)
        ax.fill_between(train_sizes_abs, train_mean - train_std, 
                        train_mean + train_std, alpha=0.2, color='blue')
        
        ax.plot(train_sizes_abs, val_mean, 'o-', color='red', 
                label='Validation error', linewidth=2)
        ax.fill_between(train_sizes_abs, val_mean - val_std, 
                        val_mean + val_std, alpha=0.2, color='red')
        
        ax.set_xlabel('Training Set Size', fontsize=10)
        ax.set_ylabel('Mean Squared Error', fontsize=10)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 결과를 저장한다
        final_train = train_mean[-1]
        final_val = val_mean[-1]
        gap = final_val - final_train
        
        results.append({
            'name': name,
            'train_error': final_train,
            'val_error': final_val,
            'gap': gap
        })
        
        # 진단 덧붙이기
        if final_val > 0.5 and gap < 0.2:
            diagnosis = "Underfitting"
            color = 'orange'
        elif gap > 0.5:
            diagnosis = "Overfitting"
            color = 'red'
        else:
            diagnosis = "Good Fit"
            color = 'green'
        
        ax.text(0.95, 0.95, diagnosis, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))
    
    plt.tight_layout()
    return fig, results

def diagnose_from_learning_curve(train_error, val_error):
    """학습 곡선의 지표로 모델의 문제를 진단한다"""
    gap = val_error - train_error
    
    print("\nDiagnosis:")
    print(f"Training Error: {train_error:.4f}")
    print(f"Validation Error: {val_error:.4f}")
    print(f"Gap: {gap:.4f}")
    print()
    
    if val_error > 0.5 and gap < 0.2:
        print("→ HIGH BIAS (Underfitting)")
        print("  Symptoms:")
        print("    • Both training and validation errors are high")
        print("    • Small gap between training and validation curves")
        print("    • Curves plateau at high error")
        print("  Solutions:")
        print("    • Use more complex model")
        print("    • Add more features")
        print("    • Reduce regularization")
        print("    • Train longer")
    elif gap > 0.5:
        print("→ HIGH VARIANCE (Overfitting)")
        print("  Symptoms:")
        print("    • Large gap between training and validation errors")
        print("    • Low training error but high validation error")
        print("    • Validation error doesn't improve with more data")
        print("  Solutions:")
        print("    • Get more training data")
        print("    • Use simpler model")
        print("    • Add regularization")
        print("    • Use ensemble methods")
        print("    • Feature selection")
    else:
        print("→ GOOD FIT")
        print("  • Both errors are reasonably low")
        print("  • Small gap between curves")
        print("  • Model generalizes well")

# 메인 실행
if __name__ == "__main__":
    print("="*70)
    print("Learning Curves Analysis")
    print("="*70)
    
    # 데이터를 생성한다
    X, y = generate_data(n_samples=1000, noise=0.3)
    
    # 예제 1: 과소적합하는 모델
    print("\n1. UNDERFITTING EXAMPLE - Linear Regression")
    print("-"*70)
    model_underfit = LinearRegression()
    fig1, train_mean, val_mean = plot_learning_curve(
        model_underfit, 'Linear Regression (Underfitting)', X, y
    )
    plt.savefig('learning_curve_underfitting.png', dpi=150, bbox_inches='tight')
    print(f"Final Training Error: {train_mean[-1]:.4f}")
    print(f"Final Validation Error: {val_mean[-1]:.4f}")
    diagnose_from_learning_curve(train_mean[-1], val_mean[-1])
    
    # 예제 2: 과적합하는 모델
    print("\n2. OVERFITTING EXAMPLE - Deep Decision Tree")
    print("-"*70)
    model_overfit = DecisionTreeRegressor(max_depth=20, random_state=42)
    fig2, train_mean, val_mean = plot_learning_curve(
        model_overfit, 'Decision Tree (Overfitting)', X, y
    )
    plt.savefig('learning_curve_overfitting.png', dpi=150, bbox_inches='tight')
    print(f"Final Training Error: {train_mean[-1]:.4f}")
    print(f"Final Validation Error: {val_mean[-1]:.4f}")
    diagnose_from_learning_curve(train_mean[-1], val_mean[-1])
    
    # 예제 3: 알맞게 적합한 모델
    print("\n3. GOOD FIT EXAMPLE - Random Forest")
    print("-"*70)
    model_good = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    fig3, train_mean, val_mean = plot_learning_curve(
        model_good, 'Random Forest (Good Fit)', X, y
    )
    plt.savefig('learning_curve_good_fit.png', dpi=150, bbox_inches='tight')
    print(f"Final Training Error: {train_mean[-1]:.4f}")
    print(f"Final Validation Error: {val_mean[-1]:.4f}")
    diagnose_from_learning_curve(train_mean[-1], val_mean[-1])
    
    # 비교 그림
    print("\n4. COMPARING MULTIPLE MODELS")
    print("-"*70)
    fig4, results = compare_learning_curves(X, y)
    plt.savefig('learning_curves_comparison.png', dpi=150, bbox_inches='tight')
    
    print("\n" + "="*70)
    print("Comparison Summary")
    print("="*70)
    print(f"{'Model':<50} {'Train Error':<15} {'Val Error':<15} {'Gap':<10}")
    print("-"*70)
    for result in results:
        print(f"{result['name']:<50} {result['train_error']:<15.4f} "
              f"{result['val_error']:<15.4f} {result['gap']:<10.4f}")
    print("="*70)
    
    plt.show()
    
    print("\n" + "="*70)
    print("Key Insights from Learning Curves:")
    print("="*70)
    print("""
1. HIGH BIAS (Underfitting):
   • 두 곡선이 모두 높은 오차에서 평평해진다
   • 학습과 검증 사이의 간격이 작다
   • 데이터를 더 모아도 큰 도움이 안 된다
   → 해법: 모델 복잡도를 높인다

2. HIGH VARIANCE (Overfitting):
   • 두 곡선 사이의 간격이 크다
   • 학습 오차는 낮고 검증 오차는 높다
   • 데이터를 더 모으면 도움이 된다
   → 해법: 데이터를 더 모으거나 복잡도를 낮춘다

3. 알맞은 적합:
   • 두 오차가 모두 낮다
   • 두 곡선 사이의 간격이 작다
   • 곡선이 한곳으로 모인다
   → 모델이 잘 돌고 있다!
    """)
    print("="*70)```

## 논의

학습 곡선은 곧바로 손쓸 수 있는 진단을 준다. 학습 오차와 검증 오차가 모두 높은 값에서 간격이 좁은 채 평평해지면 모델이 과소적합하는 것이다. 이때는 모델의 복잡도를 높이거나 특징을 더하거나 정칙화를 줄인다. 학습 오차는 낮은데 검증 오차가 높아 간격이 크면 과적합하는 것이다. 이때는 데이터를 더 모으거나 모델을 단순하게 하거나 정칙화를 세게 한다.

같은 데이터에 대해 여섯 모델(선형 회귀, 다항 회귀, 얕은 결정 트리와 깊은 결정 트리, 랜덤 포리스트, 정칙화된 다항식)을 비교하면 과소적합에서 알맞은 적합을 거쳐 과적합에 이르는 전 범위를 볼 수 있다. 랜덤 포리스트와 정칙화된 다항식이 가장 좋은 균형을 이룬다.

학습 곡선은 데이터를 더 모으는 것이 도움이 될지도 알려 준다. 학습 집합이 커질 때 검증 곡선이 아직 내려가고 있다면 데이터를 더 모으는 편이 성능을 높일 가능성이 크다. 두 곡선이 이미 수렴했다면 데이터를 더해도 효과가 거의 없으므로 특징 공학이나 모델 구조에 힘을 쏟는 편이 낫다.

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

