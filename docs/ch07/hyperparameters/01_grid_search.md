# 격자 탐색

격자 탐색은 지정한 매개변수 격자의 모든 조합을 교차 검증으로 남김없이 평가한다. 격자 안에서 가장 좋은 조합을 찾는 것은 보장되지만 계산 비용이 매개변수의 수에 따라 지수적으로 커진다. 속도보다 철저함이 중요한 작은 매개변수 공간에 알맞다.

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

# ========================================================================
# 메인
# ========================================================================
                   create_param_grid_summary, plot_parameter_importance)


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

## 2. 논의

`GridSearchCV`를 쓰는 격자 탐색은 모든 매개변수 조합을 교차 검증으로 평가한다. `n_estimators` 3개, `max_depth` 4개, `min_samples_split` 3개, `min_samples_leaf` 3개, `max_features` 2개로 이루어진 격자라면 조합이 $3 \times 4 \times 3 \times 3 \times 2 = 216$개이고, 각각을 5겹 교차 검증으로 평가하므로 모두 1,080번 모델을 적합시킨다.

중첩 교차 검증은 선택된 모델의 성능을 치우침 없이 추정하게 해 준다. 안쪽 고리는 초매개변수를 조율하고, 바깥 고리는 조율 절차 전체를 따로 떼어 둔 데이터로 평가한다. 중첩하지 않으면 모델을 바로 그 점수를 최대로 하도록 골랐으므로 보고된 교차 검증 점수가 낙관적으로 치우친다.

실무에서는 성긴 격자로 가망 있는 영역을 찾은 뒤, 가장 좋은 값 둘레에 더 촘촘한 격자를 놓아 다듬는다. 이 두 단계 방식은 계산 비용을 줄이면서도 최적에 가까운 초매개변수를 찾아 준다.

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

**다룬 것** — 격자 탐색

`GridSearchCV`를 쓰는 격자 탐색은 모든 매개변수 조합을 교차 검증으로 평가한다.

앞의 연습문제 3개로 직접 확인할 수 있다.
