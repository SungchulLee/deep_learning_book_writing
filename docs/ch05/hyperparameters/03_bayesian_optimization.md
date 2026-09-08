# 베이즈 최적화

베이즈 최적화는 목적 함수의 확률적 대리 모델을 세우고, 이를 써서 다음에 평가할 초매개변수 설정을 똑똑하게 고른다. 무작위 탐색보다 표본 효율이 높으므로 한 번의 시도에 시간이 많이 드는 값비싼 모델 평가에 알맞다.

## 1. 코드

```python
"""
초매개변수 조율을 위한 베이즈 최적화

베이즈 최적화는 확률 모형으로 최적의 초매개변수를 찾는 탐색을 이끈다.
목적 함수의 대리 모델을 세우고 그것으로 다음에 어디를 뽑을지 정한다.


장점:
- 평가가 비쌀 때 무작위 탐색보다 효율적이다
- 앞선 반복에서 배운다
- 더 적은 평가로 더 좋은 매개변수를 찾을 수 있다
- 잡음이 있는 목적 함수를 잘 다룬다

단점:
- 구현이 더 복잡하다
- 반복 하나가 더 느릴 수 있다
- 국소 최적에 갇힐 수 있다
- 추가 라이브러리가 필요하다 (예: Optuna, scikit-optimize)

이 예제는 요즘의 초매개변수 최적화 틀인 Optuna를 쓴다.
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
        
        매개변수:
        -----------
        trial : optuna.Trial
            매개변수 값을 제안하는 시도 객체
            
        반환값:
        --------
        float : 최대로 하려는 교차 검증 점수
        """
        # 초매개변수 제안
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
    
    # 가장 좋은 매개변수 얻기
    best_params = study.best_params
    best_score = study.best_value
    
    # 가장 좋은 매개변수로 최종 모델 학습
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
        
        # 매개변수 중요도 그리기
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
    print("- Great for expensive model evaluations")```

## 2. 논의

Optuna의 트리 구조 파젠 추정기(TPE)는 "좋은" 시도와 "나쁜" 시도에 대해 밀도 모델을 따로 세우고, 좋은 밀도와 나쁜 밀도의 비가 가장 큰 영역에서 새 설정을 뽑는다. 앞선 평가에서 배우므로 이러한 방향성 있는 탐색이 무작위 표집보다 빨리 수렴한다.

가지치기(가망 없는 시도를 일찍 끝내는 것)로 계산 시간을 30~50% 아낄 수 있다. 중간 교차 검증 점수를 알려 주면 Optuna는 이미 끝난 시도들의 중앙값보다 못한 시도를 중단하고 자원을 더 가망 있는 설정으로 돌린다.

표집기마다 탐색과 활용의 절충이 다르다. TPE가 기본값이며 대부분의 문제에서 잘 통한다. CMA-ES는 서로 상관된 연속 매개변수에 효과적이다. 무작위 표집은 기준선이 되며 공간 전체를 살펴보게 해 준다.

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

**다룬 것** — 베이즈 최적화

Optuna의 트리 구조 파젠 추정기(TPE)는 "좋은" 시도와 "나쁜" 시도에 대해 밀도 모델을 따로 세우고, 좋은 밀도와 나쁜 밀도의 비가 가장 큰 영역에서 새 설정을 뽑는다.

앞의 연습문제 3개로 직접 확인할 수 있다.
