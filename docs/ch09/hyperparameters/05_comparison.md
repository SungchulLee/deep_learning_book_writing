# 비교

격자·무작위·TPE 탐색과 간단한 AutoML의 관측 점수와 시간을 비교한다. 각 방법이 보는 후보 집합과 예산이 다르므로 이 실험의 결과를 보편적 순위로 읽지 않는다.

## 1. 코드

??? note "코드 (435줄)"

    ```python
    """
    초파라메터 조율 방법의 종합 비교

    이 스크립트는 초파라메터 조율 방법을 모두 견준다:
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
        plt.savefig('comparison_results.png', 
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
          - 작은 파라메터 공간 (조합 100개 미만)
          - 모든 조합을 시도해야 할 때
          - 해석 가능성이 중요할 때
          - 계산 자원이 넉넉할 때
       ❌ 쓰지 말아야 할 때:
          - 큰 파라메터 공간
          - 초파라메터가 많을 때
          - 시간이나 자원이 부족할 때

    🎲 무작위 탐색
       ✅ 쓸 때:
          - 큰 파라메터 공간
          - 연속적인 파라메터 분포
          - 계산 예산이 적을 때
          - 처음에 빠르게 살펴볼 때
       ❌ 쓰지 말아야 할 때:
          - 파라메터 공간이 아주 작을 때
          - 남김없이 훑어야 할 때

    🧠 베이즈 최적화
       ✅ 쓸 때:
          - 모델 평가가 비쌀 때
          - 표본 효율이 중요할 때
          - 파라메터 공간이 중간 크기일 때
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
        print("\nVisualization saved to 'comparison_results.png'")
    ```


??? note "전체 출력 (111줄)"

    ```
    Optuna not available. Install with: pip install optuna

    ======================================================================
    STARTING COMPREHENSIVE COMPARISON
    ======================================================================

    This will run all hyperparameter tuning methods and compare them.
    This may take a few minutes...

    ======================================================================
    COMPREHENSIVE HYPERPARAMETER TUNING COMPARISON
    ======================================================================

    Loading dataset...

    --- Running Grid Search ---
    Grid Search - CV: 0.9753, Test: 1.0000, Time: 22.41s

    --- Running Random Search ---
    Random Search - CV: 0.9753, Test: 1.0000, Time: 31.27s

    --- Running Simple AutoML ---
    Simple AutoML - CV: 0.9920, Test: 0.9815, Time: 2.37s


    Creating comparison visualizations...
    Saved visualization to 'comparison_results.png'

    ======================================================================
    SUMMARY TABLE
    ======================================================================
           Method  CV Score  Test Score  Time (s)  Iterations
      Grid Search  0.975333    1.000000 22.406776          27
    Random Search  0.975333    1.000000 31.272469          50
    Simple AutoML  0.992000    0.981481  2.365698          15
    ======================================================================

    📊 WINNERS:
      🏆 Best CV Score:    Simple AutoML (0.9920)
      🎯 Best Test Score:  Grid Search (1.0000)
      ⚡ Fastest:          Simple AutoML (2.37s)
      💡 Most Efficient:   Simple AutoML (0.4193)

    ======================================================================
    RECOMMENDATIONS - WHEN TO USE EACH METHOD
    ======================================================================

        
    📋 격자 탐색
       ✅ 쓸 때:
          - 작은 파라메터 공간 (조합 100개 미만)
          - 모든 조합을 시도해야 할 때
          - 해석 가능성이 중요할 때
          - 계산 자원이 넉넉할 때
       ❌ 쓰지 말아야 할 때:
          - 큰 파라메터 공간
          - 초파라메터가 많을 때
          - 시간이나 자원이 부족할 때

    🎲 무작위 탐색
       ✅ 쓸 때:
          - 큰 파라메터 공간
          - 연속적인 파라메터 분포
          - 계산 예산이 적을 때
          - 처음에 빠르게 살펴볼 때
       ❌ 쓰지 말아야 할 때:
          - 파라메터 공간이 아주 작을 때
          - 남김없이 훑어야 할 때

    🧠 베이즈 최적화
       ✅ 쓸 때:
          - 모델 평가가 비쌀 때
          - 표본 효율이 중요할 때
          - 파라메터 공간이 중간 크기일 때
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


    ======================================================================
    COMPARISON COMPLETE!
    ======================================================================

    Key Insights:
    - Different methods have different trade-offs
    - No single 'best' method for all situations
    - Consider your constraints: time, resources, accuracy needs
    - Start simple, increase complexity as needed

    Visualization saved to 'comparison_results.png'
    ```


## 2. 논의

격자는 유한 후보 조합 전체를, 무작위 탐색과 TPE는 정한 수의 후보를 평가한다. AutoML은 모델 계열까지 바꾼다. 따라서 같은 설정을 찾는 알고리즘의 순수한 비교라기보다 서로 다른 탐색 절차의 시연이다.

최고 점수를 시간으로 나눈 효율은 지표의 원점과 시간 측정 범위에 민감하다. 여러 비용·정확도 조건에서의 곡선과 원하는 품질에 도달하는 시간을 보는 편이 실제 선택에 도움이 될 수 있다.

테스트 점수를 보며 방법을 고르면 선택 바이어스가 생긴다. 같은 데이터 분할과 적합·탐색 예산, 반복 실행의 변동을 명시하고 선택 후 별도 평가를 유지한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
격자 후보 27개, 무작위 후보 50개를 같은 예산이라고 볼 수 있는가?

</div>

??? success "연습문제 1 풀이"
    후보 수가 다르고 각 모델의 학습 비용도 다를 수 있다. 총 시간·적합 횟수·자원 중 무엇을 맞출지 정해야 한다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
정확도/초 지표만으로 방법을 고를 때 생기는 문제를 설명하라.

</div>

??? success "연습문제 2 풀이"
    매우 빠르지만 품질이 낮은 방법이 유리해질 수 있고, 측정 시간의 준비·병렬화 비용에 따라 비율도 바뀐다. 목표 정확도와 허용 시간·메모리 조건을 함께 고려한다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
TPE가 이번 표에서 최고라고 다른 데이터에서도 최고인가?

</div>

??? success "연습문제 3 풀이"
    아니다. 탐색 공간, 평가 잡음과 예산, 모델 계열이 바뀌면 결과가 달라진다. 해당 실험 조건의 결과로만 해석한다.

## 정리하며

탐색 방법 비교는 공간과 예산의 정의에서 시작한다. 관측된 점수·시간은 조건부 결과이며 테스트를 선택에 반복 사용하지 않아야 한다.
