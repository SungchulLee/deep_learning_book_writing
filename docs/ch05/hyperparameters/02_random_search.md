# 무작위 탐색

무작위 탐색은 지정한 분포에서 매개변수 값을 정해진 횟수만큼 뽑는다. 특히 어떤 매개변수가 다른 것보다 훨씬 중요할 때, 무작위 탐색이 격자 탐색보다 좋은 초매개변수를 더 빨리 찾는다는 연구 결과가 있다. 연속 분포도 지원하므로 더 세밀하게 탐색할 수 있다.

## 1. 코드

```python
"""
초매개변수 조율을 위한 무작위 탐색

무작위 탐색은 분포에서 매개변수 값을 무작위로 뽑는다.
모든 조합을 시도하지 않고 정해진 수만큼 설정을 뽑는다.

장점:
- 매개변수 공간이 크면 격자 탐색보다 효율적이다
- 더 다양한 매개변수 조합을 살펴볼 수 있다
- 격자 탐색보다 좋은 매개변수를 더 빨리 찾는 일이 많다
- 연속 분포를 쓸 수 있다

단점:
- 절대적으로 가장 좋은 조합을 찾는다는 보장이 없다
- 실행할 때마다 결과가 달라질 수 있다 (random_state를 지정하지 않으면)
- 공간이 복잡하면 반복을 더 많이 해야 할 수 있다
"""

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import randint, uniform
import time
from utils import (load_sample_dataset, print_results, 

# ========================================================================
# 메인
# ========================================================================
                   plot_parameter_importance)


def random_search_random_forest():
    """
    랜덤 포리스트 분류기로 무작위 탐색 시연
    """
    print("\n" + "="*60)
    print("RANDOM SEARCH - RANDOM FOREST CLASSIFIER")
    print("="*60)
    
    # 데이터를 불러온다
    X_train, X_test, y_train, y_test = load_sample_dataset('wine')
    
    # 매개변수 분포 정의
    # 참고: 연속 매개변수에는 scipy.stats의 분포를 쓴다
    param_distributions = {
        'n_estimators': randint(50, 500),  # 50부터 499까지의 무작위 정수
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    
    print("\nParameter Distributions:")
    print("-" * 40)
    for param, dist in param_distributions.items():
        print(f"{param}: {dist}")
    print("-" * 40)
    
    # 모델을 만든다
    rf = RandomForestClassifier(random_state=42)
    
    # 무작위 탐색 객체 만들기
    n_iter = 100  # 시도할 무작위 조합의 수
    print(f"\nWill try {n_iter} random combinations...")
    
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        random_state=42,
        return_train_score=True
    )
    
    # 탐색 수행
    start_time = time.time()
    random_search.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    # 가장 좋은 모델을 얻어 평가
    best_model = random_search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    
    # 결과 출력
    print_results(
        method_name="Random Search (Random Forest)",
        best_params=random_search.best_params_,
        best_score=random_search.best_score_,
        search_time=search_time,
        test_score=test_score
    )
    
    # 예측한다
    y_pred = best_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 상위 5개 매개변수 조합 보이기
    import pandas as pd
    results_df = pd.DataFrame(random_search.cv_results_)
    results_df = results_df.sort_values('rank_test_score')
    
    print("\nTop 5 Parameter Combinations:")
    print("-" * 60)
    top_5_params = results_df[['params', 'mean_test_score']].head()
    for idx, row in top_5_params.iterrows():
        print(f"\nRank {row.name + 1}: Score = {row['mean_test_score']:.4f}")
        for param, value in row['params'].items():
            print(f"  {param}: {value}")
    
    return random_search


def random_search_gradient_boosting():
    """
    그래디언트 부스팅 분류기로 무작위 탐색 시연
    """
    print("\n" + "="*60)
    print("RANDOM SEARCH - GRADIENT BOOSTING CLASSIFIER")
    print("="*60)
    
    # 데이터를 불러온다
    X_train, X_test, y_train, y_test = load_sample_dataset('synthetic')
    
    # 매개변수 분포 정의
    param_distributions = {
        'n_estimators': randint(50, 300),
        'learning_rate': uniform(0.01, 0.3),  # 0.01부터 0.31까지 균등분포
        'max_depth': randint(3, 10),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'subsample': uniform(0.6, 0.4),  # 0.6부터 1.0까지 균등분포
        'max_features': ['sqrt', 'log2', None]
    }
    
    print("\nParameter Distributions:")
    print("-" * 40)
    for param, dist in param_distributions.items():
        print(f"{param}: {dist}")
    print("-" * 40)
    
    # 모델을 만든다
    gb = GradientBoostingClassifier(random_state=42)
    
    # 무작위 탐색 객체 만들기
    n_iter = 50
    print(f"\nWill try {n_iter} random combinations...")
    
    random_search = RandomizedSearchCV(
        estimator=gb,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    # 탐색 수행
    start_time = time.time()
    random_search.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    # 가장 좋은 모델을 얻어 평가
    best_model = random_search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    
    # 결과 출력
    print_results(
        method_name="Random Search (Gradient Boosting)",
        best_params=random_search.best_params_,
        best_score=random_search.best_score_,
        search_time=search_time,
        test_score=test_score
    )
    
    # 예측한다
    y_pred = best_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return random_search


def compare_n_iter():
    """
    무작위 탐색에서 반복 횟수를 달리하여 견주기
    """
    print("\n" + "="*60)
    print("COMPARING DIFFERENT NUMBER OF ITERATIONS")
    print("="*60)
    
    # 데이터를 불러온다
    X_train, X_test, y_train, y_test = load_sample_dataset('iris')
    
    # 매개변수 분포 정의
    param_distributions = {
        'n_estimators': randint(50, 300),
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': randint(2, 15),
        'max_features': ['sqrt', 'log2']
    }
    
    # 반복 횟수를 달리하여 시도
    n_iters = [10, 25, 50, 100]
    results = []
    
    print("\nTrying different numbers of random combinations...\n")
    
    for n_iter in n_iters:
        rf = RandomForestClassifier(random_state=42)
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        start_time = time.time()
        random_search.fit(X_train, y_train)
        search_time = time.time() - start_time
        
        test_score = random_search.score(X_test, y_test)
        
        results.append({
            'n_iter': n_iter,
            'cv_score': random_search.best_score_,
            'test_score': test_score,
            'time': search_time
        })
        
        print(f"n_iter={n_iter:3d}: CV Score={random_search.best_score_:.4f}, "
              f"Test Score={test_score:.4f}, Time={search_time:.2f}s")
    
    print("\n" + "="*60)
    print("Observations:")
    print("- More iterations generally lead to better scores")
    print("- But returns diminish after a certain point")
    print("- Balance between computational cost and performance")
    print("="*60)
    
    return results


def random_vs_grid_comparison():
    """
    무작위 탐색과 격자 탐색의 직접 비교
    """
    print("\n" + "="*60)
    print("RANDOM SEARCH VS GRID SEARCH")
    print("="*60)
    
    from sklearn.model_selection import GridSearchCV
    
    # 데이터를 불러온다
    X_train, X_test, y_train, y_test = load_sample_dataset('wine')
    
    # 적당한 크기의 매개변수 공간 정의
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2']
    }
    
    # 격자 탐색의 전체 조합 수 계산
    total_combinations = 4 * 4 * 3 * 2  # 조합 96개
    
    print(f"\nGrid Search will try all {total_combinations} combinations")
    print(f"Random Search will try 50 random combinations")
    
    # 격자 탐색
    print("\n--- Running Grid Search ---")
    rf_grid = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf_grid,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    grid_time = time.time() - start_time
    grid_score = grid_search.best_score_
    grid_test = grid_search.score(X_test, y_test)
    
    # 무작위 탐색
    print("\n--- Running Random Search ---")
    param_distributions = {
        'n_estimators': randint(50, 350),
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': randint(2, 15),
        'max_features': ['sqrt', 'log2']
    }
    
    rf_random = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        estimator=rf_random,
        param_distributions=param_distributions,
        n_iter=50,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    start_time = time.time()
    random_search.fit(X_train, y_train)
    random_time = time.time() - start_time
    random_score = random_search.best_score_
    random_test = random_search.score(X_test, y_test)
    
    # 결과 비교
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"\nGrid Search:")
    print(f"  Best CV Score: {grid_score:.4f}")
    print(f"  Test Score: {grid_test:.4f}")
    print(f"  Time: {grid_time:.2f} seconds")
    print(f"  Combinations tried: {total_combinations}")
    
    print(f"\nRandom Search:")
    print(f"  Best CV Score: {random_score:.4f}")
    print(f"  Test Score: {random_test:.4f}")
    print(f"  Time: {random_time:.2f} seconds")
    print(f"  Combinations tried: 50")
    
    print(f"\nTime Savings: {((grid_time - random_time) / grid_time * 100):.1f}%")
    print(f"Score Difference: {abs(grid_score - random_score):.4f}")
    
    return grid_search, random_search


if __name__ == "__main__":
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING: RANDOM SEARCH")
    print("="*60)
    
    print("\nRandom Search samples parameter settings from specified")
    print("distributions for a fixed number of iterations. It's often")
    print("more efficient than Grid Search, especially for large parameter")
    print("spaces.")
    
    # 예제 실행
    print("\n\n### Example 1: Random Forest with Random Search ###")
    rs_rf = random_search_random_forest()
    
    print("\n\n### Example 2: Gradient Boosting with Random Search ###")
    rs_gb = random_search_gradient_boosting()
    
    print("\n\n### Example 3: Effect of Number of Iterations ###")
    iter_results = compare_n_iter()
    
    print("\n\n### Example 4: Random vs Grid Search ###")
    grid, random = random_vs_grid_comparison()
    
    print("\n\nRandom Search completed! Check the results above.")
    print("\nKey Takeaways:")
    print("- Random Search is more efficient for large parameter spaces")
    print("- Can use continuous distributions (not just discrete grids)")
    print("- Often finds good parameters with fewer iterations")
    print("- Trade-off between number of iterations and computation time")```

## 2. 논의

`RandomizedSearchCV`를 쓰는 무작위 탐색은 격자를 모두 열거하는 대신 매개변수 분포에서 값을 뽑는다. Bergstra와 Bengio(2012)는 어떤 초매개변수가 다른 것보다 훨씬 중요할 때, 무작위 탐색이 중요한 매개변수의 서로 다른 값을 더 많이 살펴보므로 좋은 설정을 더 빨리 찾는다는 것을 보였다.

`scipy.stats.uniform`이나 `scipy.stats.randint` 같은 연속 분포를 쓰면 이산적인 격자점보다 세밀하게 탐색할 수 있다. 학습률의 경우 0.001에서 0.01로 바꾸는 것과 0.01에서 0.1로 바꾸는 것의 효과가 비슷하므로 로그 균등분포가 더 알맞을 때가 많다.

격자 탐색과 무작위 탐색을 견주어 보면 무작위 탐색이 훨씬 짧은 시간에 비슷한 정확도에 이른다. 매개변수 공간이 커질수록 이 이점은 더 뚜렷해진다. 격자 탐색의 비용은 지수적으로 늘지만 무작위 탐색의 비용은 정해 둔 반복 횟수에 고정되기 때문이다.

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

**다룬 것** — 무작위 탐색

`RandomizedSearchCV`를 쓰는 무작위 탐색은 격자를 모두 열거하는 대신 매개변수 분포에서 값을 뽑는다.

앞의 연습문제 3개로 직접 확인할 수 있다.
