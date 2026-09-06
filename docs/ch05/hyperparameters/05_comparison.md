# 비교

초매개변수 조율 방법마다 철저함, 속도, 쓰기 쉬움 사이의 절충이 다르다. 이 스크립트는 같은 데이터셋에서 격자 탐색, 무작위 탐색, 베이즈 최적화, 간단한 AutoML을 맞대어 비교하며 최고 점수, 계산 시간, 효율을 잰다.

## 코드

```python
"""
초매개변수 조율 방법의 종합 비교

이 스크립트는 초매개변수 조율 방법을 모두 견준다:
1. 격자 탐색
2. 무작위 탐색
3. 베이즈 최적화 (Optuna)
4. 간단한 AutoML

다음 기준으로 평가한다:
- 얻은 최고 점수
- 걸린 시간
- 모델 평가의 횟수
- 쓰기 쉬움
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
import time

# ========================================================================
# 메인
# ========================================================================

# 베이즈 최적화를 위한 Optuna
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available. Install with: pip install optuna")

from utils import load_sample_dataset, plot_search_results


def run_grid_search(X_train, y_train):
    """격자 탐색 실행"""
    print("\n--- Running Grid Search ---")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='accuracy', 
        n_jobs=-1, verbose=0
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    return {
        'method': 'Grid Search',
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'time': elapsed_time,
        'n_iterations': len(grid_search.cv_results_['params']),
        'estimator': grid_search.best_estimator_
    }


def run_random_search(X_train, y_train, n_iter=50):
    """무작위 탐색 실행"""
    print("\n--- Running Random Search ---")
    
    param_distributions = {
        'n_estimators': randint(50, 300),
        'max_depth': [10, 20, 30, None],
        'min_samples_split': randint(2, 15),
    }
    
    rf = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        rf, param_distributions, n_iter=n_iter, 
        cv=5, scoring='accuracy', n_jobs=-1, 
        random_state=42, verbose=0
    )
    
    start_time = time.time()
    random_search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    return {
        'method': 'Random Search',
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_,
        'time': elapsed_time,
        'n_iterations': n_iter,
        'estimator': random_search.best_estimator_
    }


def run_bayesian_optimization(X_train, y_train, n_trials=50):
    """Optuna로 베이즈 최적화 실행"""
    if not OPTUNA_AVAILABLE:
        return None
    
    print("\n--- Running Bayesian Optimization ---")
    
    from sklearn.model_selection import cross_val_score
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
        }
        
        rf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        score = cross_val_score(rf, X_train, y_train, cv=5, 
                                scoring='accuracy', n_jobs=-1).mean()
        return score
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    start_time = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed_time = time.time() - start_time
    
    # 최종 모델 학습
    best_rf = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1)
    best_rf.fit(X_train, y_train)
    
    return {
        'method': 'Bayesian Opt',
        'best_params': study.best_params,
        'best_score': study.best_value,
        'time': elapsed_time,
        'n_iterations': n_trials,
        'estimator': best_rf
    }


def run_simple_automl(X_train, y_train):
    """간단한 AutoML 실행 (모델 선택)"""
    print("\n--- Running Simple AutoML ---")
    
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
    }
    
    start_time = time.time()
    best_score = 0
    best_model = None
    best_name = None
    
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, 
                                scoring='accuracy', n_jobs=-1)
        mean_score = scores.mean()
        
        if mean_score > best_score:
            best_score = mean_score
            best_model = model
            best_name = name
    
    # 가장 좋은 모델 적합
    best_model.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    return {
        'method': 'Simple AutoML',
        'best_params': {'model': best_name},
        'best_score': best_score,
        'time': elapsed_time,
        'n_iterations': len(models) * 5,  # 모델마다 × 교차 검증 겹 수
        'estimator': best_model
    }


def comprehensive_comparison():
    """
    모든 방법을 종합적으로 견주어 본다
    """
    print("="*70)
    print("COMPREHENSIVE HYPERPARAMETER TUNING COMPARISON")
    print("="*70)
    
    # 데이터를 불러온다
    print("\nLoading dataset...")
    X_train, X_test, y_train, y_test = load_sample_dataset('wine')
    
    # 모든 방법 실행
    results = []
    
    # 격자 탐색
    result = run_grid_search(X_train, y_train)
    if result:
        result['test_score'] = result['estimator'].score(X_test, y_test)
        results.append(result)
        print(f"Grid Search - CV: {result['best_score']:.4f}, "
              f"Test: {result['test_score']:.4f}, Time: {result['time']:.2f}s")
    
    # 무작위 탐색
    result = run_random_search(X_train, y_train, n_iter=50)
    if result:
        result['test_score'] = result['estimator'].score(X_test, y_test)
        results.append(result)
        print(f"Random Search - CV: {result['best_score']:.4f}, "
              f"Test: {result['test_score']:.4f}, Time: {result['time']:.2f}s")
    
    # 베이즈 최적화
    if OPTUNA_AVAILABLE:
        result = run_bayesian_optimization(X_train, y_train, n_trials=50)
        if result:
            result['test_score'] = result['estimator'].score(X_test, y_test)
            results.append(result)
            print(f"Bayesian Opt - CV: {result['best_score']:.4f}, "
                  f"Test: {result['test_score']:.4f}, Time: {result['time']:.2f}s")
    
    # 간단한 AutoML
    result = run_simple_automl(X_train, y_train)
    if result:
        result['test_score'] = result['estimator'].score(X_test, y_test)
        results.append(result)
        print(f"Simple AutoML - CV: {result['best_score']:.4f}, "
              f"Test: {result['test_score']:.4f}, Time: {result['time']:.2f}s")
    
    return results


def visualize_comparison(results):
    """
    모든 방법을 견주는 시각화 만들기
    """
    print("\n\nCreating comparison visualizations...")
    
    # DataFrame 만들기
    df = pd.DataFrame([
        {
            'Method': r['method'],
            'CV Score': r['best_score'],
            'Test Score': r['test_score'],
            'Time (s)': r['time'],
            'Iterations': r['n_iterations']
        }
        for r in results
    ])
    
    # 부분 그림을 갖는 도형 만들기
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Hyperparameter Tuning Methods Comparison', 
                 fontsize=16, fontweight='bold')
    
    # 1. 교차 검증 점수
    ax1 = axes[0, 0]
    bars = ax1.bar(df['Method'], df['CV Score'], color='steelblue', alpha=0.8)
    ax1.set_ylabel('Cross-Validation Score', fontsize=11)
    ax1.set_title('Best CV Score by Method', fontsize=12, fontweight='bold')
    ax1.set_ylim([df['CV Score'].min() * 0.95, df['CV Score'].max() * 1.02])
    ax1.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 2. 시험 점수
    ax2 = axes[0, 1]
    bars = ax2.bar(df['Method'], df['Test Score'], color='coral', alpha=0.8)
    ax2.set_ylabel('Test Score', fontsize=11)
    ax2.set_title('Test Set Score by Method', fontsize=12, fontweight='bold')
    ax2.set_ylim([df['Test Score'].min() * 0.95, df['Test Score'].max() * 1.02])
    ax2.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 3. 시간 비교
    ax3 = axes[1, 0]
    bars = ax3.bar(df['Method'], df['Time (s)'], color='lightgreen', alpha=0.8)
    ax3.set_ylabel('Time (seconds)', fontsize=11)
    ax3.set_title('Computation Time by Method', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s', ha='center', va='bottom', fontsize=9)
    
    # 4. 효율 (초당 점수)
    ax4 = axes[1, 1]
    efficiency = df['CV Score'] / df['Time (s)']
    bars = ax4.bar(df['Method'], efficiency, color='plum', alpha=0.8)
    ax4.set_ylabel('CV Score / Time', fontsize=11)
    ax4.set_title('Efficiency (Score per Second)', fontsize=12, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/home/claude/hyperparameter_tuning/comparison_results.png', 
                dpi=300, bbox_inches='tight')
    print("Saved visualization to 'comparison_results.png'")
    plt.show()
    
    return df


def print_summary_table(df):
    """
    보기 좋은 요약표 출력
    """
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)
    
    # 우승자 찾기
    best_cv = df.loc[df['CV Score'].idxmax()]
    best_test = df.loc[df['Test Score'].idxmax()]
    fastest = df.loc[df['Time (s)'].idxmin()]
    most_efficient = df.loc[(df['CV Score'] / df['Time (s)']).idxmax()]
    
    print("\n📊 WINNERS:")
    print(f"  🏆 Best CV Score:    {best_cv['Method']} ({best_cv['CV Score']:.4f})")
    print(f"  🎯 Best Test Score:  {best_test['Method']} ({best_test['Test Score']:.4f})")
    print(f"  ⚡ Fastest:          {fastest['Method']} ({fastest['Time (s)']:.2f}s)")
    print(f"  💡 Most Efficient:   {most_efficient['Method']} "
          f"({(most_efficient['CV Score']/most_efficient['Time (s)']):.4f})")


def recommendations():
    """
    각 방법을 언제 쓸지 권고 출력
    """
    print("\n" + "="*70)
    print("RECOMMENDATIONS - WHEN TO USE EACH METHOD")
    print("="*70)
    
    recommendations_text = """
    
📋 격자 탐색
   ✅ 쓸 때:
      - 작은 매개변수 공간 (조합 100개 미만)
      - 모든 조합을 시도해야 할 때
      - 해석 가능성이 중요할 때
      - 계산 자원이 넉넉할 때
   ❌ 쓰지 말아야 할 때:
      - 큰 매개변수 공간
      - 초매개변수가 많을 때
      - 시간이나 자원이 부족할 때

🎲 무작위 탐색
   ✅ 쓸 때:
      - 큰 매개변수 공간
      - 연속적인 매개변수 분포
      - 계산 예산이 적을 때
      - 처음에 빠르게 살펴볼 때
   ❌ 쓰지 말아야 할 때:
      - 매개변수 공간이 아주 작을 때
      - 남김없이 훑어야 할 때

🧠 베이즈 최적화
   ✅ 쓸 때:
      - 모델 평가가 비쌀 때
      - 표본 효율이 중요할 때
      - 매개변수 공간이 중간 크기일 때
      - 준비의 복잡함을 감당할 수 있을 때
   ❌ 쓰지 말아야 할 때:
      - 모델 학습이 아주 빠를 때
      - 공간이 극도로 클 때
      - 단순함이 필요할 때

🤖 AutoML
   ✅ 쓸 때:
      - 새 과제를 시작할 때
      - 기준선이 빨리 필요할 때
      - 기계학습 지식이 적을 때
      - 여러 모델을 시도하고 싶을 때
   ❌ 쓰지 말아야 할 때:
      - 온전한 제어가 필요할 때
      - 요구가 아주 구체적일 때
      - 검증 없이 쓰는 중요한 실전 시스템

💡 일반적인 조언:
   1. 빠르게 살펴보려면 무작위 탐색으로 시작하라
   2. 비싼 모델에는 베이즈 최적화를 쓰라
   3. 좁은 범위의 마지막 미세 조율에는 격자 탐색을 쓰라
   4. 기준선과 모델 선택에는 AutoML을 쓰라
   5. 언제나 따로 둔 시험 집합에서 검증하라
   6. 튼튼한 추정에는 중첩 교차 검증을 고려하라
"""
    print(recommendations_text)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("STARTING COMPREHENSIVE COMPARISON")
    print("="*70)
    print("\nThis will run all hyperparameter tuning methods and compare them.")
    print("This may take a few minutes...\n")
    
    # 비교 실행
    results = comprehensive_comparison()
    
    # 시각화한다
    df = visualize_comparison(results)
    
    # 요약 출력
    print_summary_table(df)
    
    # 권고 출력
    recommendations()
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)
    print("\nKey Insights:")
    print("- Different methods have different trade-offs")
    print("- No single 'best' method for all situations")
    print("- Consider your constraints: time, resources, accuracy needs")
    print("- Start simple, increase complexity as needed")
    print("\nVisualization saved to 'comparison_results.png'")```

## 논의

비교해 보면 모든 기준에서 앞서는 조율 방법은 없다. 격자 탐색은 격자 안에서 가장 좋은 조합을 찾는 것은 보장하지만 규모를 키우기 어렵다. 무작위 탐색은 빠르고 뜻밖에 효과적이다. 베이즈 최적화는 알맞은 계산량으로 가장 좋은 점수를 낸다. 간단한 AutoML은 준비가 거의 필요 없으면서도 쓸 만한 결과를 준다.

계산 1초당 점수로 재는 효율은 단순한 방법에 유리할 때가 많다. 개발 초기에 빠르게 살펴볼 때에는 무작위 탐색이 가장 값어치 있다. 정확도가 가장 중요한 마지막 모델 선택에서는 베이즈 최적화가 복잡함을 감수할 만하다.

방법별로 교차 검증 점수, 시험 점수, 시간, 효율을 견주어 그린 그림은 판단의 틀을 뚜렷이 보여 준다. 실무에서는 대개 빠른 시제품 단계에서 무작위 탐색으로 시작하고, 실전 모델 조율에서는 베이즈 최적화로 옮겨 간다.

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

