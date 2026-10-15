# 격자 탐색

격자 탐색은 지정한 유한 후보 조합을 같은 교차 검증 기준으로 평가한다. 관측된 후보 중 최고 점수를 찾는 절차이며 참 일반화 성능이나 격자 밖의 최적값을 보장하지는 않는다.

## 1. 코드

```python
"""
초매개변수 조율을 위한 격자 탐색

격자 탐색은 지정한 매개변수 격자를 남김없이 훑는다.
가능한 모든 매개변수 조합을 시도한다.

장점:
- 격자 안에서 가장 좋은 조합을 찾는 것을 보장한다
- 이해하고 구현하기 쉽다
- 작은 매개변수 공간에 알맞다

단점:
- 격자가 크면 계산 비용이 크다
- 차원의 저주를 겪는다
- 격자점 사이의 최적값을 놓칠 수 있다
"""

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import time
from utils import (load_sample_dataset, print_results,
                   create_param_grid_summary, plot_parameter_importance)

# ========================================================================
# 메인
# ========================================================================


def grid_search_random_forest():
    """
    랜덤 포리스트 분류기로 격자 탐색 시연
    """
    print("\n" + "="*60)
    print("GRID SEARCH - RANDOM FOREST CLASSIFIER")
    print("="*60)
    
    # 데이터를 불러온다
    X_train, X_test, y_train, y_test = load_sample_dataset('wine')
    
    # 매개변수 격자 정의
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
    }
    
    # 격자 요약 보이기
    total_combinations = create_param_grid_summary(param_grid)
    
    # 모델을 만든다
    rf = RandomForestClassifier(random_state=42)
    
    # 격자 탐색 객체 만들기
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,  # 5겹 교차 검증
        scoring='accuracy',
        n_jobs=-1,  # 쓸 수 있는 모든 코어 사용
        verbose=1,
        return_train_score=True
    )
    
    # 탐색 수행
    print(f"\nSearching through {total_combinations} combinations...")
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    # 가장 좋은 모델을 얻어 평가
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    
    # 결과 출력
    print_results(
        method_name="Grid Search (Random Forest)",
        best_params=grid_search.best_params_,
        best_score=grid_search.best_score_,
        search_time=search_time,
        test_score=test_score
    )
    
    # 예측한다
    y_pred = best_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 매개변수 중요도 그리기
    plot_parameter_importance(grid_search.cv_results_, 'n_estimators')
    plot_parameter_importance(grid_search.cv_results_, 'max_depth')
    
    return grid_search


def grid_search_svm():
    """
    서포트 벡터 머신으로 격자 탐색 시연
    """
    print("\n" + "="*60)
    print("GRID SEARCH - SUPPORT VECTOR MACHINE")
    print("="*60)
    
    # 데이터를 불러온다
    X_train, X_test, y_train, y_test = load_sample_dataset('iris')
    
    # SVM을 위한 매개변수 격자 정의
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    
    # 격자 요약 보이기
    total_combinations = create_param_grid_summary(param_grid)
    
    # 모델을 만든다
    svm = SVC(random_state=42)
    
    # 격자 탐색 객체 만들기
    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # 탐색 수행
    print(f"\nSearching through {total_combinations} combinations...")
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    # 가장 좋은 모델을 얻어 평가
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    
    # 결과 출력
    print_results(
        method_name="Grid Search (SVM)",
        best_params=grid_search.best_params_,
        best_score=grid_search.best_score_,
        search_time=search_time,
        test_score=test_score
    )
    
    # 예측한다
    y_pred = best_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 매개변수 중요도 그리기
    plot_parameter_importance(grid_search.cv_results_, 'C')
    plot_parameter_importance(grid_search.cv_results_, 'gamma')
    
    return grid_search


def nested_grid_search():
    """
    격자 탐색과 함께 중첩 교차 검증 시연
    이래야 모델의 성능을 더 튼튼하게 추정할 수 있다
    """
    print("\n" + "="*60)
    print("NESTED GRID SEARCH (More Robust Evaluation)")
    print("="*60)
    
    from sklearn.model_selection import cross_val_score
    
    # 데이터를 불러온다
    X_train, X_test, y_train, y_test = load_sample_dataset('synthetic')
    
    # 매개변수 격자 정의
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    
    # 초매개변수 조율을 위한 안쪽 교차 검증
    inner_cv = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )
    
    # 모델 평가를 위한 바깥 교차 검증
    print("\nPerforming nested cross-validation...")
    start_time = time.time()
    outer_scores = cross_val_score(
        inner_cv, X_train, y_train, 
        cv=5, scoring='accuracy', n_jobs=-1
    )
    search_time = time.time() - start_time
    
    print(f"\nOuter CV Scores: {outer_scores}")
    print(f"Mean Score: {outer_scores.mean():.4f} (+/- {outer_scores.std() * 2:.4f})")
    print(f"Search Time: {search_time:.2f} seconds")
    
    # 가장 좋은 모델을 얻으려고 학습 집합 전체에 적합
    inner_cv.fit(X_train, y_train)
    test_score = inner_cv.score(X_test, y_test)
    
    print(f"\nBest Parameters: {inner_cv.best_params_}")
    print(f"Test Set Score: {test_score:.4f}")
    
    return inner_cv


if __name__ == "__main__":
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING: GRID SEARCH")
    print("="*60)
    
    print("\nGrid Search systematically works through multiple combinations")
    print("of parameter values, cross-validating as it goes to determine")
    print("which combination gives the best performance.")
    
    # 예제 실행
    print("\n\n### Example 1: Random Forest ###")
    gs_rf = grid_search_random_forest()
    
    print("\n\n### Example 2: Support Vector Machine ###")
    gs_svm = grid_search_svm()
    
    print("\n\n### Example 3: Nested Cross-Validation ###")
    gs_nested = nested_grid_search()
    
    print("\n\nGrid Search completed! Check the results above.")
    print("\nKey Takeaways:")
    print("- Grid Search is exhaustive and guaranteed to find the best")
    print("  combination within your specified grid")
    print("- Computational cost grows exponentially with parameters")
    print("- Use nested CV for unbiased performance estimates")
    print("- Start with coarse grid, then refine around best values")
```

??? note "전체 출력 (116줄)"

    ```

    ============================================================
    HYPERPARAMETER TUNING: GRID SEARCH
    ============================================================

    Grid Search systematically works through multiple combinations
    of parameter values, cross-validating as it goes to determine
    which combination gives the best performance.


    ### Example 1: Random Forest ###

    ============================================================
    GRID SEARCH - RANDOM FOREST CLASSIFIER
    ============================================================

    격자 살펴보기
      n_estimators             3가지  [50, 100, 200]
      max_depth                4가지  [None, 10, 20, 30]
      min_samples_split        3가지  [2, 5, 10]
      min_samples_leaf         3가지  [1, 2, 4]
      max_features             2가지  ['sqrt', 'log2']
      합쳐서 216가지 짝

    Searching through 216 combinations...
    Fitting 5 folds for each of 216 candidates, totalling 1080 fits

    ============================================================
    Grid Search (Random Forest)
    ============================================================
      가장 좋은 조절 값:
        max_depth                None
        max_features             sqrt
        min_samples_leaf         2
        min_samples_split        2
        n_estimators             50
      맞대 보기 점수 : 0.9757
      시험 점수      : 1.0000
      걸린 값        : 121.53초

    Classification Report:
                  precision    recall  f1-score   support

               0       1.00      1.00      1.00        18
               1       1.00      1.00      1.00        21
               2       1.00      1.00      1.00        15

        accuracy                           1.00        54
       macro avg       1.00      1.00      1.00        54
    weighted avg       1.00      1.00      1.00        54



    ### Example 2: Support Vector Machine ###

    ============================================================
    GRID SEARCH - SUPPORT VECTOR MACHINE
    ============================================================

    격자 살펴보기
      C                        4가지  [0.1, 1, 10, 100]
      gamma                    5가지  ['scale', 'auto', 0.001, 0.01, 0.1]
      kernel                   3가지  ['rbf', 'poly', 'sigmoid']
      합쳐서 60가지 짝

    Searching through 60 combinations...
    Fitting 5 folds for each of 60 candidates, totalling 300 fits

    ============================================================
    Grid Search (SVM)
    ============================================================
      가장 좋은 조절 값:
        C                        1
        gamma                    0.1
        kernel                   rbf
      맞대 보기 점수 : 0.9810
      시험 점수      : 0.9111
      걸린 값        : 0.16초

    Classification Report:
                  precision    recall  f1-score   support

               0       1.00      1.00      1.00        15
               1       0.82      0.93      0.88        15
               2       0.92      0.80      0.86        15

        accuracy                           0.91        45
       macro avg       0.92      0.91      0.91        45
    weighted avg       0.92      0.91      0.91        45



    ### Example 3: Nested Cross-Validation ###

    ============================================================
    NESTED GRID SEARCH (More Robust Evaluation)
    ============================================================

    Performing nested cross-validation...

    Outer CV Scores: [0.78571429 0.71428571 0.74285714 0.77142857 0.62857143]
    Mean Score: 0.7286 (+/- 0.1114)
    Search Time: 16.57 seconds

    Best Parameters: {'max_depth': 20, 'min_samples_split': 5, 'n_estimators': 200}
    Test Set Score: 0.7467


    Grid Search completed! Check the results above.

    Key Takeaways:
    - Grid Search is exhaustive and guaranteed to find the best
      combination within your specified grid
    - Computational cost grows exponentially with parameters
    - Use nested CV for unbiased performance estimates
    - Start with coarse grid, then refine around best values
    ```


## 2. 논의

랜덤 포리스트 격자는 3×4×3×3×2=216개 조합이다. 5폴드이면 후보 평가 적합이 1080회이고, 기본 refit=True에서는 최선 설정을 전체 학습 데이터에 다시 적합하는 한 번이 추가된다. 후보에 따라 적합 시간은 다를 수 있다.

중첩 검증은 안쪽에서 설정을 고르고 바깥에서 선택 절차를 평가한다. 바깥 점수는 안쪽 최고 점수의 선택 효과를 줄이는 데 유용하지만 모든 데이터 조건에서 완전히 불편한 추정이라고 단정하지 않는다. 전처리도 안쪽 학습 부분에서만 맞춰야 한다.

코드는 utils의 데이터 로딩과 결과 출력 도우미를 가져오므로 별도 파일과 환경이 필요하다. 큰 병렬 탐색은 메모리와 내부 모델의 병렬화를 함께 고려한다. 성긴 격자를 좁혀 가는 방식도 추가 선택 절차이므로 검증 비용을 기록한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
216개 후보와 5폴드의 적합 횟수를 refit까지 구분해 구하라.

</div>

??? success "연습문제 1 풀이"
    후보 평가에 1080회, 기본 최선 모델 재적합에 한 번이 더해져 1081회이다. 중첩 바깥 검증이나 후속 탐색은 이 수에 추가된다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
best_score_가 독립 테스트 점수와 다른 이유를 설명하라.

</div>

??? success "연습문제 2 풀이"
    같은 검증 점수를 높이도록 후보를 골랐기 때문이다. 선택한 최고 점수에는 선택 효과가 있고 독립 테스트는 선택을 마친 모델을 평가한다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
격자 밖의 좋은 설정을 찾기 위해 무엇을 바꿀 수 있는가?

</div>

??? success "연습문제 3 풀이"
    검증과 예산을 유지하며 후보 범위를 넓히거나 무작위 탐색 등 다른 방법을 사용한다. 범위를 추가한 사실과 전체 탐색 횟수를 기록하고 테스트셋을 탐색에 쓰지 않는다.

## 정리하며

격자 탐색은 명시한 후보 집합을 평가한다. 후보·폴드·재적합의 비용을 구분하고 최고 교차 검증 점수와 선택 후 평가를 분리한다.
