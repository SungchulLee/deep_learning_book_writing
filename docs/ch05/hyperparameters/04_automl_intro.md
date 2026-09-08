# AutoML 소개

AutoML은 특징 전처리, 모델 선택, 초매개변수 조율, 앙상블 구성에 이르는 기계학습 파이프라인을 자동화한다. 여러 모델과 설정을 자동으로 시도하므로 쓸 만한 기준선을 세우는 데 드는 전문 지식과 시간을 줄여 준다. 다만 실전에 배포하려면 그 결과를 이해하는 일이 여전히 중요하다.

## 1. 코드

```python
"""
AutoML - 자동화된 기계학습

AutoML은 다음을 포함한 기계학습 파이프라인 전체를 자동화한다:
- 특징 전처리
- 모델 선택
- 초매개변수 조율
- 앙상블 구성

널리 쓰이는 AutoML 라이브러리:
- TPOT: 유전 프로그래밍을 쓴다
- Auto-sklearn: scikit-learn에 자동 모델 선택을 더한다
- H2O AutoML: 기업용에 초점을 둔 AutoML
- PyCaret: 코드를 적게 쓰는 기계학습 라이브러리
- AutoKeras: 딥러닝을 위한 AutoML

이 예제는 (설치가 더 간단한) TPOT으로 AutoML의 기본 개념을 보이고,
AutoML 비슷한 작업 흐름을 직접 구현하는 법도 보인다.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import time

# ========================================================================
# 메인
# ========================================================================

# TPOT은 선택 사항이다. 없으면 직접 만든 AutoML을 보인다
try:
    from tpot import TPOTClassifier
    TPOT_AVAILABLE = True
except ImportError:
    print("TPOT not installed. Will demonstrate manual AutoML approach.")
    TPOT_AVAILABLE = False

from utils import load_sample_dataset, print_results


def simple_automl():
    """
    여러 모델을 시도하여 가장 좋은 것을 고르는 간단한 AutoML 구현
    """
    print("\n" + "="*60)
    print("SIMPLE AUTOML - MODEL SELECTION")
    print("="*60)
    
    print("\nThis example automatically tries multiple models and")
    print("selects the best one based on cross-validation scores.")
    
    # 데이터를 불러온다
    X_train, X_test, y_train, y_test = load_sample_dataset('wine')
    
    # 매개변수 선택지와 함께 후보 모델 정의
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42),
        'SVM (Linear)': SVC(kernel='linear', random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    }
    
    print(f"\nEvaluating {len(models)} different models...")
    
    results = []
    start_time = time.time()
    
    for name, model in models.items():
        print(f"\nTrying {name}...")
        
        # 척도 조정을 포함한 파이프라인 만들기
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        # 교차 검증으로 평가
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, 
            cv=5, scoring='accuracy', n_jobs=-1
        )
        
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        
        results.append({
            'model': name,
            'mean_cv_score': mean_score,
            'std_cv_score': std_score,
            'pipeline': pipeline
        })
        
        print(f"  CV Score: {mean_score:.4f} (+/- {std_score:.4f})")
    
    total_time = time.time() - start_time
    
    # 점수로 정렬
    results.sort(key=lambda x: x['mean_cv_score'], reverse=True)
    
    # 가장 좋은 모델 고르기
    best_result = results[0]
    best_pipeline = best_result['pipeline']
    
    # 학습 집합 전체로 학습한 뒤 평가
    best_pipeline.fit(X_train, y_train)
    test_score = best_pipeline.score(X_test, y_test)
    
    # 결과 출력
    print("\n" + "="*60)
    print("MODEL SELECTION RESULTS")
    print("="*60)
    
    print("\nAll Models (sorted by performance):")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['model']}: {result['mean_cv_score']:.4f} "
              f"(+/- {result['std_cv_score']:.4f})")
    
    print(f"\n{'Best Model:':<20} {best_result['model']}")
    print(f"{'Best CV Score:':<20} {best_result['mean_cv_score']:.4f}")
    print(f"{'Test Score:':<20} {test_score:.4f}")
    print(f"{'Total Time:':<20} {total_time:.2f} seconds")
    
    # 예측
    y_pred = best_pipeline.predict(X_test)
    print("\nClassification Report (Best Model):")
    print(classification_report(y_test, y_pred))
    
    return best_pipeline, results


def automl_with_hyperparameter_tuning():
    """
    모델마다 초매개변수 조율까지 포함하는 더 발전된 AutoML
    """
    print("\n" + "="*60)
    print("AUTOML WITH HYPERPARAMETER TUNING")
    print("="*60)
    
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint, uniform
    
    # 데이터를 불러온다
    X_train, X_test, y_train, y_test = load_sample_dataset('synthetic')
    
    # 매개변수 분포와 함께 모델 정의
    model_configs = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'model__n_estimators': randint(50, 300),
                'model__max_depth': [10, 20, 30, None],
                'model__min_samples_split': randint(2, 10),
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'model__n_estimators': randint(50, 200),
                'model__learning_rate': uniform(0.01, 0.2),
                'model__max_depth': randint(3, 10),
            }
        },
        'SVM': {
            'model': SVC(random_state=42),
            'params': {
                'model__C': uniform(0.1, 100),
                'model__kernel': ['rbf', 'linear'],
                'model__gamma': ['scale', 'auto'],
            }
        }
    }
    
    print(f"\nTuning {len(model_configs)} models with hyperparameter search...")
    
    results = []
    start_time = time.time()
    
    for name, config in model_configs.items():
        print(f"\n--- Tuning {name} ---")
        
        # 파이프라인 만들기
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', config['model'])
        ])
        
        # 초매개변수에 대한 무작위 탐색
        random_search = RandomizedSearchCV(
            pipeline,
            config['params'],
            n_iter=20,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        random_search.fit(X_train, y_train)
        
        results.append({
            'model': name,
            'best_score': random_search.best_score_,
            'best_params': random_search.best_params_,
            'estimator': random_search.best_estimator_
        })
        
        print(f"Best CV Score: {random_search.best_score_:.4f}")
    
    total_time = time.time() - start_time
    
    # 가장 좋은 모델 고르기
    results.sort(key=lambda x: x['best_score'], reverse=True)
    best_result = results[0]
    
    # 시험 집합에서 평가
    test_score = best_result['estimator'].score(X_test, y_test)
    
    # 결과 출력
    print("\n" + "="*60)
    print("AUTOML RESULTS")
    print("="*60)
    
    print("\nAll Models (sorted by performance):")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['model']}")
        print(f"   CV Score: {result['best_score']:.4f}")
        print(f"   Best Parameters:")
        for param, value in result['best_params'].items():
            print(f"     {param}: {value}")
    
    print(f"\n{'='*60}")
    print(f"{'Best Model:':<25} {best_result['model']}")
    print(f"{'Best CV Score:':<25} {best_result['best_score']:.4f}")
    print(f"{'Test Score:':<25} {test_score:.4f}")
    print(f"{'Total Time:':<25} {total_time:.2f} seconds")
    print(f"{'='*60}")
    
    return best_result['estimator'], results


def tpot_automl_example():
    """
    진짜 AutoML 라이브러리인 TPOT을 쓰는 예제
    """
    if not TPOT_AVAILABLE:
        print("\n" + "="*60)
        print("TPOT NOT AVAILABLE")
        print("="*60)
        print("\nTPOT is not installed. To use TPOT AutoML:")
        print("  pip install tpot")
        print("\nTPOT uses genetic programming to automatically design")
        print("and optimize machine learning pipelines.")
        return None
    
    print("\n" + "="*60)
    print("TPOT AUTOML EXAMPLE")
    print("="*60)
    
    print("\nTPOT uses genetic programming to automatically discover")
    print("the best machine learning pipeline for your data.")
    
    # 데이터를 불러온다
    X_train, X_test, y_train, y_test = load_sample_dataset('iris')
    
    print("\nInitializing TPOT...")
    print("This will take a few minutes as it evolves pipelines...")
    
    # TPOT 분류기 만들기
    tpot = TPOTClassifier(
        generations=5,  # 실행할 반복의 수
        population_size=20,  # 세대마다의 파이프라인 수
        cv=5,
        random_state=42,
        verbosity=2,
        n_jobs=-1,
        max_time_mins=3,  # 최대 시간(분)
        max_eval_time_mins=0.5,  # 파이프라인 하나의 최대 시간
    )
    
    # TPOT 실행
    start_time = time.time()
    tpot.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    # 평가한다
    train_score = tpot.score(X_train, y_train)
    test_score = tpot.score(X_test, y_test)
    
    print("\n" + "="*60)
    print("TPOT RESULTS")
    print("="*60)
    print(f"\nBest Pipeline Score (CV): {train_score:.4f}")
    print(f"Test Score: {test_score:.4f}")
    print(f"Search Time: {search_time:.2f} seconds")
    
    # 가장 좋은 파이프라인 내보내기
    print("\nExporting best pipeline to 'best_pipeline.py'...")
    tpot.export('/home/claude/hyperparameter_tuning/tpot_best_pipeline.py')
    
    # 가장 좋은 파이프라인 보이기
    print("\nBest Pipeline:")
    print(tpot.fitted_pipeline_)
    
    # 예측한다
    y_pred = tpot.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return tpot


def ensemble_automl():
    """
    AutoML이 찾은 가장 좋은 모델들로 앙상블 만들기
    """
    print("\n" + "="*60)
    print("ENSEMBLE AUTOML")
    print("="*60)
    
    from sklearn.ensemble import VotingClassifier
    
    print("\nCombining multiple good models into an ensemble...")
    
    # 데이터를 불러온다
    X_train, X_test, y_train, y_test = load_sample_dataset('wine')
    
    # 모델 정의
    models = {
        'rf': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42),
        'gb': GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42),
        'svm': SVC(kernel='rbf', C=10, probability=True, random_state=42),
    }
    
    # 개별 모델 평가
    print("\nIndividual model performance:")
    individual_results = []
    
    for name, model in models.items():
        pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        mean_score = scores.mean()
        individual_results.append((name, mean_score))
        print(f"  {name}: {mean_score:.4f}")
    
    # 투표 앙상블 만들기
    voting_clf = VotingClassifier(
        estimators=[(name, Pipeline([('scaler', StandardScaler()), ('model', model)])) 
                    for name, model in models.items()],
        voting='soft'  # 확률 예측 사용
    )
    
    # 앙상블 평가
    print("\nTraining ensemble...")
    start_time = time.time()
    ensemble_scores = cross_val_score(voting_clf, X_train, y_train, cv=5, scoring='accuracy')
    ensemble_time = time.time() - start_time
    
    # 전체 데이터로 학습한 뒤 시험
    voting_clf.fit(X_train, y_train)
    test_score = voting_clf.score(X_test, y_test)
    
    print("\n" + "="*60)
    print("ENSEMBLE RESULTS")
    print("="*60)
    print(f"\nEnsemble CV Score: {ensemble_scores.mean():.4f} (+/- {ensemble_scores.std():.4f})")
    print(f"Ensemble Test Score: {test_score:.4f}")
    print(f"Training Time: {ensemble_time:.2f} seconds")
    
    # 가장 좋은 개별 모델과 비교
    best_individual = max(individual_results, key=lambda x: x[1])
    print(f"\nBest Individual Model: {best_individual[0]} ({best_individual[1]:.4f})")
    print(f"Ensemble Improvement: {(ensemble_scores.mean() - best_individual[1]):.4f}")
    
    return voting_clf


if __name__ == "__main__":
    print("\n" + "="*60)
    print("AUTOMATED MACHINE LEARNING (AutoML)")
    print("="*60)
    
    print("\nAutoML automates the machine learning pipeline including")
    print("feature engineering, model selection, and hyperparameter tuning.")
    print("It makes ML accessible and efficient by automating repetitive tasks.")
    
    # 예제 실행
    print("\n\n### Example 1: Simple Model Selection ###")
    best_model, all_results = simple_automl()
    
    print("\n\n### Example 2: AutoML with Hyperparameter Tuning ###")
    tuned_model, tuning_results = automl_with_hyperparameter_tuning()
    
    print("\n\n### Example 3: TPOT AutoML ###")
    tpot_model = tpot_automl_example()
    
    print("\n\n### Example 4: Ensemble AutoML ###")
    ensemble = ensemble_automl()
    
    print("\n\nAutoML examples completed!")
    print("\nKey Takeaways:")
    print("- AutoML automates model selection and tuning")
    print("- Can save significant time in model development")
    print("- TPOT and Auto-sklearn are powerful AutoML tools")
    print("- Ensembles often improve over individual models")
    print("- Great for baseline models and non-experts")
    print("\nAutoML libraries to explore:")
    print("  - TPOT: pip install tpot")
    print("  - Auto-sklearn: pip install auto-sklearn")
    print("  - PyCaret: pip install pycaret")
    print("  - H2O AutoML: pip install h2o")```

## 2. 논의

간단한 AutoML은 여러 모델 계열(랜덤 포리스트, 그래디언트 부스팅, SVM, KNN, 로지스틱 회귀)을 교차 검증으로 평가하여 가장 좋은 것을 고른다. 전문 지식이 거의 없어도 몇 분 만에 튼튼한 기준선을 얻을 수 있다. 다만 초매개변수를 조율하지 않으면 고른 모델이 최적이 아닐 수 있다.

고급 AutoML 변형은 표준화를 포함하는 파이프라인 안에서 `RandomizedSearchCV`로 모델마다 초매개변수를 조율하며 모델 선택을 함께 수행한다. 더 철저하지만 모델의 수와 초매개변수 분포가 늘어나는 만큼 시간도 더 걸린다.

앙상블 AutoML은 성능이 좋은 모델들을 부드러운 투표(확률 평균)를 쓰는 `VotingClassifier`로 묶는다. 모델 계열마다 저지르는 오차의 종류가 다르고 평균이 이러한 약점을 상쇄하므로 앙상블은 대체로 개별 모델보다 낫다.

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

**다룬 것** — AutoML 소개

간단한 AutoML은 여러 모델 계열(랜덤 포리스트, 그래디언트 부스팅, SVM, KNN, 로지스틱 회귀)을 교차 검증으로 평가하여 가장 좋은 것을 고른다.

앞의 연습문제 3개로 직접 확인할 수 있다.
