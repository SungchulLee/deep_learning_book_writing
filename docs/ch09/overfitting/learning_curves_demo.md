# 학습 곡선 시연

학습 데이터의 크기를 늘리며 학습 MSE와 교차 검증 MSE가 어떻게 달라지는지 비교한다. 이 페이지의 학습 곡선은 에포크에 따른 로스 곡선과 구분한다.

## 1. 코드

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
1. 높은 바이어스(과소적합):
   • 두 곡선이 모두 높은 오차에서 평평해진다
   • 학습과 검증 사이의 간격이 작다
   • 데이터를 더 모아도 큰 도움이 안 된다
   → 해법: 모델 복잡도를 높인다

2. 높은 분산(과적합):
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
    print("="*70)
```

??? note "전체 출력 (107줄)"

    ```
    ======================================================================
    Learning Curves Analysis
    ======================================================================

    1. UNDERFITTING EXAMPLE - Linear Regression
    ----------------------------------------------------------------------
    Final Training Error: 0.5126
    Final Validation Error: 0.9155

    Diagnosis:
    Training Error: 0.5126
    Validation Error: 0.9155
    Gap: 0.4029

    → GOOD FIT
      • Both errors are reasonably low
      • Small gap between curves
      • Model generalizes well

    2. OVERFITTING EXAMPLE - Deep Decision Tree
    ----------------------------------------------------------------------
    Final Training Error: 0.0006
    Final Validation Error: 0.5762

    Diagnosis:
    Training Error: 0.0006
    Validation Error: 0.5762
    Gap: 0.5756

    → HIGH VARIANCE (Overfitting)
      Symptoms:
        • Large gap between training and validation errors
        • Low training error but high validation error
        • Validation error doesn't improve with more data
      Solutions:
        • Get more training data
        • Use simpler model
        • Add regularization
        • Use ensemble methods
        • Feature selection

    3. GOOD FIT EXAMPLE - Random Forest
    ----------------------------------------------------------------------
    Final Training Error: 0.0745
    Final Validation Error: 0.4776

    Diagnosis:
    Training Error: 0.0745
    Validation Error: 0.4776
    Gap: 0.4031

    → GOOD FIT
      • Both errors are reasonably low
      • Small gap between curves
      • Model generalizes well

    4. COMPARING MULTIPLE MODELS
    ----------------------------------------------------------------------

    Processing: Linear Regression (Underfitting)...

    Processing: Polynomial Regression (Degree 3)...

    Processing: Decision Tree (max_depth=2, Underfitting)...

    Processing: Decision Tree (max_depth=20, Overfitting)...

    Processing: Random Forest (Good Fit)...

    Processing: Ridge Regression (Degree 10, Regularized)...

    ======================================================================
    Comparison Summary
    ======================================================================
    Model                                              Train Error     Val Error       Gap       
    ----------------------------------------------------------------------
    Linear Regression (Underfitting)                   0.5126          0.9155          0.4029    
    Polynomial Regression (Degree 3)                   0.2887          13.2502         12.9615   
    Decision Tree (max_depth=2, Underfitting)          0.1832          0.6468          0.4635    
    Decision Tree (max_depth=20, Overfitting)          0.0006          0.5762          0.5756    
    Random Forest (Good Fit)                           0.0745          0.4776          0.4031    
    Ridge Regression (Degree 10, Regularized)          0.0853          159.3212        159.2360  
    ======================================================================

    ======================================================================
    Key Insights from Learning Curves:
    ======================================================================

    1. 높은 바이어스(과소적합):
       • 두 곡선이 모두 높은 오차에서 평평해진다
       • 학습과 검증 사이의 간격이 작다
       • 데이터를 더 모아도 큰 도움이 안 된다
       → 해법: 모델 복잡도를 높인다

    2. 높은 분산(과적합):
       • 두 곡선 사이의 간격이 크다
       • 학습 오차는 낮고 검증 오차는 높다
       • 데이터를 더 모으면 도움이 된다
       → 해법: 데이터를 더 모으거나 복잡도를 낮춘다

    3. 알맞은 적합:
       • 두 오차가 모두 낮다
       • 두 곡선 사이의 간격이 작다
       • 곡선이 한곳으로 모인다
       → 모델이 잘 돌고 있다!
        
    ======================================================================
    ```


## 2. 논의

낮은 학습 오차와 높은 검증 오차의 격차는 과적합을 의심할 단서이다. 두 오차가 모두 높으면 모델의 표현력 부족뿐 아니라 피처 누락, 강한 정규화나 최적화 실패도 점검해야 한다. 격차만으로 편향과 분산을 수치적으로 분리하거나 원인을 확정할 수는 없다.

코드는 여섯 모델의 교차 검증 결과를 비교한다. 회귀에서 cv=5는 기본적으로 섞지 않은 KFold를 사용하고, 입력 X는 순서대로 생성되어 있다. learning_curve의 훈련 부분집합도 기본적으로 섞지 않아 작은 표본에서는 입력 구간이 치우칠 수 있다. 따라서 곡선에는 표본 수뿐 아니라 입력 범위와 분할 방식의 영향도 포함된다. 독립·동일 분포를 가정한 비교에서는 셔플을 포함한 분할 설계를, 시간 예측에서는 시간 순서를 지키는 설계를 따로 검토한다.

scoring의 음수 MSE를 다시 부호 반전하므로 그림에서는 작을수록 좋다. 음영은 폴드별 점수의 표준편차이며 신뢰구간이 아니다. 검증 오차가 감소하는 추세는 데이터 추가의 효과를 추측하는 근거이지만, 현재 범위 밖에서도 같은 속도로 줄어든다는 보장은 없다. 코드의 고정 임계값 진단도 목표값 단위와 잡음 수준에 따라 달라지는 예시 규칙이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
neg_mean_squared_error의 부호를 바꾸는 이유는 무엇인가?

</div>

??? success "연습문제 1 풀이"
    scikit-learn의 점수는 큰 값이 좋다는 규약을 맞추기 위해 MSE에 음수를 붙인다. 이를 다시 음수로 곱하면 원래 MSE가 되어 작을수록 좋다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
그림의 평균±표준편차 음영을 95% 신뢰구간이라고 할 수 있는가?

</div>

??? success "연습문제 2 풀이"
    아니다. 폴드 점수의 산포를 표시한 것이며 폴드 간 학습 데이터도 겹친다. 표준편차 자체가 평균 추정의 표준오차나 95% 구간은 아니다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
입력이 정렬되어 있는 상태에서 작은 훈련 부분집합을 앞에서 선택하면 어떤 문제가 있는가?

</div>

??? success "연습문제 3 풀이"
    표본 수 증가와 함께 학습 입력의 범위도 변할 수 있다. 검증 성능 변화가 단순히 표본 수의 효과인지 구분하기 어려워진다. 문제의 배포 조건에 맞는 분할·셔플 방식과 반복 실험을 설계해야 한다.

## 정리하며

학습 곡선은 데이터 양과 일반화 성능의 관계를 진단하는 자료이다. 오차의 단위, 분할 방식과 불확실성을 함께 읽고, 곡선 모양만으로 특정 원인이나 최적 모델을 단정하지 않는다.
