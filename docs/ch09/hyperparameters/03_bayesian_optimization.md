# 베이즈 최적화

베이즈 최적화 계열의 탐색은 과거 후보와 점수로 다음 평가를 제안한다. 이 페이지는 Optuna의 TPE와 여러 샘플러, 가지치기 인터페이스를 사용한다.

## 1. 코드

```python
"""
초파라메터 조율을 위한 베이즈 최적화

베이즈 최적화는 확률 모형으로 최적의 초파라메터를 찾는 탐색을 이끈다.
목적 함수의 대리 모델을 세우고 그것으로 다음에 어디를 뽑을지 정한다.


장점:
- 평가가 비쌀 때 무작위 탐색보다 효율적이다
- 앞선 반복에서 배운다
- 더 적은 평가로 더 좋은 파라메터를 찾을 수 있다
- 잡음이 있는 목적 함수를 잘 다룬다

단점:
- 구현이 더 복잡하다
- 반복 하나가 더 느릴 수 있다
- 국소 최적에 갇힐 수 있다
- 추가 라이브러리가 필요하다 (예: Optuna, scikit-optimize)

이 예제는 요즘의 초파라메터 최적화 틀인 Optuna를 쓴다.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import time

# ========================================================================
# 메인
# ========================================================================

# 참고: pip install optuna 으로 설치한다
try:
    import optuna
    from optuna.visualization import (plot_optimization_history, 
                                       plot_param_importances,
                                       plot_parallel_coordinate)
    OPTUNA_AVAILABLE = True
except ImportError:
    print("Optuna not installed. Install with: pip install optuna")
    OPTUNA_AVAILABLE = False

from utils import load_sample_dataset, print_results


def bayesian_optimization_rf():
    """
    Optuna로 랜덤 포리스트에 베이즈 최적화 적용
    """
    if not OPTUNA_AVAILABLE:
        print("Please install optuna: pip install optuna")
        return None
    
    print("\n" + "="*60)
    print("BAYESIAN OPTIMIZATION - RANDOM FOREST")
    print("="*60)
    
    # 데이터를 불러온다
    X_train, X_test, y_train, y_test = load_sample_dataset('wine')
    
    # 목적 함수 정의
    def objective(trial):
        """
        Optuna가 최적화할 목적 함수
        
        파라메터:
        -----------
        trial : optuna.Trial
            파라메터 값을 제안하는 시도 객체
            
        반환값:
        --------
        float : 최대로 하려는 교차 검증 점수
        """
        # 초파라메터 제안
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        }
        
        # 모델을 만들어 평가
        rf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        score = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1).mean()
        
        return score
    
    # 스터디 만들기
    print("\nCreating optimization study...")
    study = optuna.create_study(
        direction='maximize',  # 정확도를 최대로 하려 한다
        sampler=optuna.samplers.TPESampler(seed=42)  # 트리 구조 파젠 추정기
    )
    
    # 최적화
    n_trials = 100
    print(f"Running {n_trials} optimization trials...")
    start_time = time.time()
    
    # 진행 상황을 보이려고 콜백 쓰기
    def callback(study, trial):
        if trial.number % 10 == 0:
            print(f"Trial {trial.number}: Best score = {study.best_value:.4f}")
    
    study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=False)
    search_time = time.time() - start_time
    
    # 가장 좋은 파라메터 얻기
    best_params = study.best_params
    best_score = study.best_value
    
    # 가장 좋은 파라메터로 최종 모델 학습
    best_rf = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    best_rf.fit(X_train, y_train)
    test_score = best_rf.score(X_test, y_test)
    
    # 결과 출력
    print_results(
        method_name="Bayesian Optimization (Random Forest)",
        best_params=best_params,
        best_score=best_score,
        search_time=search_time,
        test_score=test_score
    )
    
    # 예측한다
    y_pred = best_rf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 최적화 이력 보이기
    print("\nOptimization Progress:")
    print(f"  Best trial: {study.best_trial.number}")
    print(f"  Total trials: {len(study.trials)}")
    print(f"  Trials that completed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    
    # 시각화 (선택 사항 - plotly가 필요하다)
    try:
        import matplotlib.pyplot as plt
        
        # 최적화 이력 그리기
        fig = plot_optimization_history(study)
        fig.show()
        
        # 파라메터 중요도 그리기
        fig = plot_param_importances(study)
        fig.show()
        
    except ImportError:
        print("\nInstall plotly for visualizations: pip install plotly")
    
    return study


def bayesian_optimization_gb():
    """
    그래디언트 부스팅에 베이즈 최적화 적용
    """
    if not OPTUNA_AVAILABLE:
        print("Please install optuna: pip install optuna")
        return None
    
    print("\n" + "="*60)
    print("BAYESIAN OPTIMIZATION - GRADIENT BOOSTING")
    print("="*60)
    
    # 데이터를 불러온다
    X_train, X_test, y_train, y_test = load_sample_dataset('synthetic')
    
    # 목적 함수 정의
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        }
        
        gb = GradientBoostingClassifier(**params, random_state=42)
        score = cross_val_score(gb, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1).mean()
        
        return score
    
    # 스터디를 만들어 실행
    print("\nRunning optimization...")
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    start_time = time.time()
    study.optimize(objective, n_trials=50, show_progress_bar=False)
    search_time = time.time() - start_time
    
    # 최종 모델 학습
    best_gb = GradientBoostingClassifier(**study.best_params, random_state=42)
    best_gb.fit(X_train, y_train)
    test_score = best_gb.score(X_test, y_test)
    
    # 결과 출력
    print_results(
        method_name="Bayesian Optimization (Gradient Boosting)",
        best_params=study.best_params,
        best_score=study.best_value,
        search_time=search_time,
        test_score=test_score
    )
    
    return study


def pruning_example():
    """
    가망 없는 시도를 일찍 끝내는 가지치기 시연
    최적화를 크게 빠르게 할 수 있다
    """
    if not OPTUNA_AVAILABLE:
        print("Please install optuna: pip install optuna")
        return None
    
    print("\n" + "="*60)
    print("BAYESIAN OPTIMIZATION WITH PRUNING")
    print("="*60)
    
    print("\nPruning stops unpromising trials early to save computation time.")
    
    # 데이터를 불러온다
    X_train, X_test, y_train, y_test = load_sample_dataset('iris')
    
    from sklearn.model_selection import cross_validate
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
        }
        
        rf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        
        # 중간 보고와 함께 교차 검증 수행
        cv_results = cross_validate(
            rf, X_train, y_train, cv=5, scoring='accuracy', 
            return_train_score=False, n_jobs=-1
        )
        
        # 가지치기를 위해 중간값 보고
        for i, score in enumerate(cv_results['test_score']):
            trial.report(score, i)
            # 이 시도를 가지쳐야 하는지 확인
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return cv_results['test_score'].mean()
    
    # 가지치기를 갖춘 스터디 만들기
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )
    
    print("\nRunning optimization with pruning...")
    start_time = time.time()
    study.optimize(objective, n_trials=50, show_progress_bar=False)
    search_time = time.time() - start_time
    
    # 통계
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    print(f"\nOptimization Statistics:")
    print(f"  Total trials: {len(study.trials)}")
    print(f"  Completed trials: {len(complete_trials)}")
    print(f"  Pruned trials: {len(pruned_trials)}")
    print(f"  Time saved: ~{len(pruned_trials) / len(study.trials) * 100:.1f}%")
    print(f"  Total time: {search_time:.2f} seconds")
    print(f"\nBest score: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    
    return study


def compare_samplers():
    """
    베이즈 최적화에서 표집 전략을 달리하여 견주기
    """
    if not OPTUNA_AVAILABLE:
        print("Please install optuna: pip install optuna")
        return None
    
    print("\n" + "="*60)
    print("COMPARING DIFFERENT SAMPLERS")
    print("="*60)
    
    # 데이터를 불러온다
    X_train, X_test, y_train, y_test = load_sample_dataset('wine')
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
        }
        
        rf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        score = cross_val_score(rf, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1).mean()
        return score
    
    # 여러 표집기 시험
    samplers = {
        'TPE': optuna.samplers.TPESampler(seed=42),
        'Random': optuna.samplers.RandomSampler(seed=42),
        'CMA-ES': optuna.samplers.CmaEsSampler(seed=42),
    }
    
    results = {}
    
    print("\nTesting different sampling strategies...")
    for name, sampler in samplers.items():
        print(f"\n--- {name} Sampler ---")
        study = optuna.create_study(direction='maximize', sampler=sampler)
        
        start_time = time.time()
        study.optimize(objective, n_trials=30, show_progress_bar=False)
        elapsed_time = time.time() - start_time
        
        results[name] = {
            'best_score': study.best_value,
            'time': elapsed_time
        }
        
        print(f"Best score: {study.best_value:.4f}")
        print(f"Time: {elapsed_time:.2f} seconds")
    
    # 요약
    print("\n" + "="*60)
    print("SAMPLER COMPARISON SUMMARY")
    print("="*60)
    for name, result in results.items():
        print(f"{name:10s}: Score={result['best_score']:.4f}, Time={result['time']:.2f}s")
    
    return results


if __name__ == "__main__":
    if not OPTUNA_AVAILABLE:
        print("\n" + "="*60)
        print("ERROR: Optuna not installed")
        print("="*60)
        print("\nPlease install Optuna to run this example:")
        print("  pip install optuna")
        print("\nOptuna is a powerful hyperparameter optimization framework")
        print("that implements Bayesian optimization and other advanced techniques.")
        exit(1)
    
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING: BAYESIAN OPTIMIZATION")
    print("="*60)
    
    print("\nBayesian Optimization uses past evaluation results to build")
    print("a probabilistic model and intelligently choose the next set of")
    print("hyperparameters to evaluate. It's more efficient than random")
    print("search for expensive evaluations.")
    
    # 예제 실행
    print("\n\n### Example 1: Random Forest with Bayesian Optimization ###")
    study_rf = bayesian_optimization_rf()
    
    print("\n\n### Example 2: Gradient Boosting with Bayesian Optimization ###")
    study_gb = bayesian_optimization_gb()
    
    print("\n\n### Example 3: Optimization with Pruning ###")
    study_pruned = pruning_example()
    
    print("\n\n### Example 4: Comparing Different Samplers ###")
    sampler_results = compare_samplers()
    
    print("\n\nBayesian Optimization completed! Check the results above.")
    print("\nKey Takeaways:")
    print("- Bayesian Optimization is sample-efficient")
    print("- TPE sampler works well for most problems")
    print("- Pruning can significantly reduce computation time")
    print("- More sophisticated than random/grid search")
    print("- Great for expensive model evaluations")
```

## 2. 논의

TPE는 성능이 좋은 후보와 나머지 후보의 분포를 구분해 다음 후보를 제안한다. 평가를 활용한다는 장점이 있지만 탐색 공간·초기 탐색·관측 잡음에 따라 무작위 탐색보다 낫지 않을 수도 있다.

가지치기 예제는 cross_validate를 먼저 끝내 모든 폴드 점수를 얻은 뒤 report와 should_prune을 호출한다. 따라서 이미 지출한 교차 검증 계산을 줄이지 못한다. 실제 절약을 얻으려면 학습이나 폴드 계산 중간에 순차 보고하고 남은 계산 전에 중단해야 한다. [Optuna 안내](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html)의 중간값 보고 흐름을 구분한다.

TPE·CMA-ES·무작위 샘플러는 지원하는 변수와 병렬화 조건이 다르다. 후보 수뿐 아니라 총 시간과 자원, 중단된 후보를 포함한 탐색 기록을 함께 비교한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
현재 가지치기 예제에서 교차 검증 계산 시간이 절약되지 않는 이유를 설명하라.

</div>

??? success "풀이"
    중간 점수 보고 전에 cross_validate가 모든 폴드의 학습·평가를 끝낸다. TrialPruned로 기록해도 이미 수행한 계산은 돌아오지 않는다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
가지치기가 느리게 개선되는 좋은 후보를 버릴 수 있는 이유를 설명하라.

</div>

??? success "풀이"
    초기 지표가 최종 순위를 반영하지 않을 수 있다. 워밍업과 최소 자원, 같은 step의 지표 의미를 맞추고 가지치기 없는 기준과 비교한다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
베이즈 탐색의 최고 점수도 독립 평가가 필요한가?

</div>

??? success "풀이"
    그렇다. 앞선 점수를 이용해 후보를 적응적으로 고르므로 관측된 최고 점수에는 선택 효과가 있다. 탐색이 끝난 뒤 독립 테스트나 바깥 검증으로 절차를 평가한다.

## 정리하며

적응형 탐색과 가지치기의 원리를 실제 코드의 계산 순서와 맞춰 읽는다. 가지치기는 남은 계산을 수행하기 전에 판정해야 자원을 줄일 수 있으며 최고 탐색 점수는 최종 독립 성능이 아니다.
