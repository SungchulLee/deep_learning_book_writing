# AutoML 소개

AutoML은 전처리와 모델·하이퍼파라미터 선택의 일부를 자동화한다. 이 예제는 여러 계열의 교차 검증, 랜덤 탐색과 투표 앙상블을 보여 준다.

## 1. 코드

??? note "코드 (420줄)"

    ```python
    """
    AutoML - 자동화된 기계학습

    AutoML은 다음을 포함한 기계학습 파이프라인 전체를 자동화한다:
    - 피처 전처리
    - 모델 선택
    - 초파라메터 조율
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

        # 파라메터 선택지와 함께 후보 모델 정의
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
        모델마다 초파라메터 조율까지 포함하는 더 발전된 AutoML
        """
        print("\n" + "="*60)
        print("AUTOML WITH HYPERPARAMETER TUNING")
        print("="*60)

        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import randint, uniform

        # 데이터를 불러온다
        X_train, X_test, y_train, y_test = load_sample_dataset('synthetic')

        # 파라메터 분포와 함께 모델 정의
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

            # 초파라메터에 대한 무작위 탐색
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
        tpot.export('tpot_best_pipeline.py')

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
        print("  - H2O AutoML: pip install h2o")
    ```


??? note "전체 출력 (176줄)"

    ```
    TPOT not installed. Will demonstrate manual AutoML approach.

    ============================================================
    AUTOMATED MACHINE LEARNING (AutoML)
    ============================================================

    AutoML automates the machine learning pipeline including
    feature engineering, model selection, and hyperparameter tuning.
    It makes ML accessible and efficient by automating repetitive tasks.


    ### Example 1: Simple Model Selection ###

    ============================================================
    SIMPLE AUTOML - MODEL SELECTION
    ============================================================

    This example automatically tries multiple models and
    selects the best one based on cross-validation scores.

    Evaluating 6 different models...

    Trying Random Forest...
      CV Score: 0.9673 (+/- 0.0310)

    Trying Gradient Boosting...
      CV Score: 0.8863 (+/- 0.0613)

    Trying SVM (RBF)...
      CV Score: 0.9920 (+/- 0.0160)

    Trying SVM (Linear)...
      CV Score: 0.9757 (+/- 0.0199)

    Trying K-Nearest Neighbors...
      CV Score: 0.9593 (+/- 0.0365)

    Trying Logistic Regression...
      CV Score: 0.9837 (+/- 0.0200)

    ============================================================
    MODEL SELECTION RESULTS
    ============================================================

    All Models (sorted by performance):
    1. SVM (RBF): 0.9920 (+/- 0.0160)
    2. Logistic Regression: 0.9837 (+/- 0.0200)
    3. SVM (Linear): 0.9757 (+/- 0.0199)
    4. Random Forest: 0.9673 (+/- 0.0310)
    5. K-Nearest Neighbors: 0.9593 (+/- 0.0365)
    6. Gradient Boosting: 0.8863 (+/- 0.0613)

    Best Model:          SVM (RBF)
    Best CV Score:       0.9920
    Test Score:          0.9815
    Total Time:          21.82 seconds

    Classification Report (Best Model):
                  precision    recall  f1-score   support

               0       1.00      1.00      1.00        18
               1       0.95      1.00      0.98        21
               2       1.00      0.93      0.97        15

        accuracy                           0.98        54
       macro avg       0.98      0.98      0.98        54
    weighted avg       0.98      0.98      0.98        54



    ### Example 2: AutoML with Hyperparameter Tuning ###

    ============================================================
    AUTOML WITH HYPERPARAMETER TUNING
    ============================================================

    Tuning 3 models with hyperparameter search...

    --- Tuning Random Forest ---
    Best CV Score: 0.7315

    --- Tuning Gradient Boosting ---
    Best CV Score: 0.7258

    --- Tuning SVM ---
    Best CV Score: 0.7714

    ============================================================
    AUTOML RESULTS
    ============================================================

    All Models (sorted by performance):

    1. SVM
       CV Score: 0.7714
       Best Parameters:
         model__C: 5.908361216819946
         model__gamma: auto
         model__kernel: rbf

    2. Random Forest
       CV Score: 0.7315
       Best Parameters:
         model__max_depth: 10
         model__min_samples_split: 2
         model__n_estimators: 253

    3. Gradient Boosting
       CV Score: 0.7258
       Best Parameters:
         model__learning_rate: 0.08649239825343255
         model__max_depth: 6
         model__n_estimators: 113

    ============================================================
    Best Model:               SVM
    Best CV Score:            0.7714
    Test Score:               0.8000
    Total Time:               64.41 seconds
    ============================================================


    ### Example 3: TPOT AutoML ###

    ============================================================
    TPOT NOT AVAILABLE
    ============================================================

    TPOT is not installed. To use TPOT AutoML:
      pip install tpot

    TPOT uses genetic programming to automatically design
    and optimize machine learning pipelines.


    ### Example 4: Ensemble AutoML ###

    ============================================================
    ENSEMBLE AUTOML
    ============================================================

    Combining multiple good models into an ensemble...

    Individual model performance:
      rf: 0.9753
      gb: 0.9027
      svm: 0.9920

    Training ensemble...

    ============================================================
    ENSEMBLE RESULTS
    ============================================================

    Ensemble CV Score: 0.9593 (+/- 0.0511)
    Ensemble Test Score: 1.0000
    Training Time: 7.81 seconds

    Best Individual Model: svm (0.9920)
    Ensemble Improvement: -0.0327


    AutoML examples completed!

    Key Takeaways:
    - AutoML automates model selection and tuning
    - Can save significant time in model development
    - TPOT and Auto-sklearn are powerful AutoML tools
    - Ensembles often improve over individual models
    - Great for baseline models and non-experts

    AutoML libraries to explore:
      - TPOT: pip install tpot
      - Auto-sklearn: pip install auto-sklearn
      - PyCaret: pip install pycaret
      - H2O AutoML: pip install h2o
    ```


## 2. 논의

간단한 AutoML은 후보 모델을 같은 교차 검증 기준에서 비교한다. 파이프라인에 scaler를 포함하면 각 폴드 학습 부분에서 전처리를 맞출 수 있다. 모든 후보에 표준화가 반드시 필요한 것은 아니지만 데이터 누출을 막는 구조는 유지해야 한다.

모델 선택과 하이퍼파라미터 탐색을 함께 하면 후보 수와 비용이 커진다. 자동화가 데이터 의미, 분할·지표·누출 검증까지 대신하는 것은 아니다. 최선 점수의 선택 바이어스도 남는다.

soft voting은 클래스별 확률을 평균낸다. 클래스 순서와 확률 추정의 품질이 맞아야 하며, 오차가 충분히 다르지 않거나 나쁜 모델이 포함되면 개별 모델보다 나쁠 수 있다. TPOT 예제는 해당 버전의 API와 별도 의존성이 필요하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
scaler를 Pipeline 안에 넣는 이유를 설명하라.

</div>

??? success "연습문제 1 풀이"
    교차 검증의 각 학습 부분에서만 scaler를 fit하고 해당 검증 부분에 변환을 적용하기 위해서이다. 전체 데이터를 미리 표준화하는 것과 다르다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
soft voting의 입력과 결과를 해석할 때 필요한 조건을 제시하라.

</div>

??? success "연습문제 2 풀이"
    각 모델의 클래스 순서와 predict_proba 출력을 맞춘다. 확률 보정과 웨이트 선택도 검증하고 평균이 자동으로 좋은 확률이나 정확도를 보장한다고 보지 않는다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
AutoML의 최고 CV 점수가 최종 성능의 불편추정인가?

</div>

??? success "연습문제 3 풀이"
    많은 후보 중 고른 최고값이므로 일반적으로 선택 효과가 있다. 전처리와 탐색을 포함한 절차를 바깥 평가나 독립 테스트에서 확인해야 한다.

## 정리하며

AutoML도 평가 설계와 탐색 예산이 필요하다. 전처리를 폴드 안에 두고 후보·앙상블 선택에 사용한 지표와 최종 평가를 구분한다.
