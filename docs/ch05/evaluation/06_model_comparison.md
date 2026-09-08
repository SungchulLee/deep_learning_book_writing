# 모델 비교

여러 모델을 견줄 때에는 평균 점수만 보아서는 안 된다. 대응 t 검정 같은 통계 검정은 성능 차이가 유의한지 가려 주고, 학습 곡선과 검증 곡선은 편향과 분산, 그리고 초매개변수가 일반화에 미치는 영향을 진단한다.

## 1. 코드

```python
"""
모델의 비교와 선택
===============================

가장 좋은 모델을 견주고 고르는 기법.

다루는 주제:
- 여러 모델 견주기
- 통계적 유의성 검정
- 학습 곡선
- 검증 곡선
- 모델 선택 전략
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (

# ========================================================================
# 메인
# ========================================================================
    cross_val_score, learning_curve, validation_curve
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy import stats


class ModelComparison:
    """
    여러 모델을 견주는 도구
    """
    
    @staticmethod
    def compare_models_cv(models_dict, X, y, cv=5, scoring='accuracy'):
        """
        교차 검증으로 여러 모델 견주기
        
        인수:
            models_dict: {모델 이름: 모델 객체} 사전
            X: 특징
            y: 목푯값
            cv: 교차 검증 전략
            scoring: 쓸 지표
        
        반환값:
            모델별 결과를 담은 사전
        """
        results = {}
        
        print("=" * 70)
        print("MODEL COMPARISON USING CROSS-VALIDATION")
        print("=" * 70)
        print(f"\nCross-Validation: {cv}-fold")
        print(f"Scoring Metric: {scoring}")
        print("\n" + "-" * 70)
        
        for name, model in models_dict.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            
            results[name] = {
                'scores': scores,
                'mean': scores.mean(),
                'std': scores.std(),
                'min': scores.min(),
                'max': scores.max()
            }
            
            print(f"\n{name}:")
            print(f"  Mean Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            print(f"  Score Range: [{scores.min():.4f}, {scores.max():.4f}]")
            print(f"  Individual Scores: {[f'{s:.4f}' for s in scores]}")
        
        # 가장 좋은 모델 찾기
        best_model = max(results.items(), key=lambda x: x[1]['mean'])
        print("\n" + "=" * 70)
        print(f"BEST MODEL: {best_model[0]}")
        print(f"Mean Score: {best_model[1]['mean']:.4f}")
        print("=" * 70)
        
        return results
    
    @staticmethod
    def paired_ttest(scores1, scores2, model1_name="Model 1", model2_name="Model 2"):
        """
        두 모델을 통계적으로 견주는 대응 t 검정
        
        쓸 때: 두 모델의 짝지어진 교차 검증 점수가 있을 때
        
        H0 (귀무가설): 두 모델의 성능이 같다
        H1 (대립가설): 두 모델의 성능이 다르다
        
        인수:
            scores1: 모델 1의 교차 검증 점수
            scores2: 모델 2의 교차 검증 점수
            model1_name: 모델 1의 이름
            model2_name: 모델 2의 이름
        
        반환값:
            검정 결과를 담은 사전
        """
        t_stat, p_value = stats.ttest_rel(scores1, scores2)
        
        print("\n" + "=" * 70)
        print("PAIRED T-TEST FOR MODEL COMPARISON")
        print("=" * 70)
        
        print(f"\nComparing: {model1_name} vs {model2_name}")
        print(f"\n{model1_name} scores: {scores1}")
        print(f"{model2_name} scores: {scores2}")
        
        print(f"\nMean difference: {np.mean(scores1 - scores2):.4f}")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        
        alpha = 0.05
        if p_value < alpha:
            print(f"\nConclusion (α={alpha}):")
            print(f"  ✓ Statistically significant difference (p < {alpha})")
            if np.mean(scores1) > np.mean(scores2):
                print(f"  → {model1_name} performs significantly better")
            else:
                print(f"  → {model2_name} performs significantly better")
        else:
            print(f"\nConclusion (α={alpha}):")
            print(f"  ✗ No statistically significant difference (p >= {alpha})")
            print(f"  → Cannot conclude that one model is better")
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'mean_difference': np.mean(scores1 - scores2)
        }
    
    @staticmethod
    def plot_learning_curve(model, X, y, cv=5, scoring='accuracy',
                           train_sizes=np.linspace(0.1, 1.0, 10),
                           title="Learning Curve"):
        """
        편향과 분산을 진단하려고 학습 곡선 그리기
        
        학습 곡선이 보여 주는 것:
        - 학습 집합의 크기에 따라 학습 점수와 검증 점수가 어떻게 변하는지
        - 모델의 편향이 큰지 분산이 큰지
        
        해석:
        - 편향이 클 때 (과소적합):
            → 학습 점수와 검증 점수가 모두 낮다
            → 점수가 일찍 평평해진다
            → 곡선 사이의 간격이 작다
            → 해결: 더 복잡한 모델, 더 많은 특징
        
        - 분산이 클 때 (과적합):
            → 학습 점수는 높고 검증 점수는 낮다
            → 곡선 사이의 간격이 크다
            → 데이터를 더 모으면 검증 점수가 오를 수 있다
            → 해결: 더 많은 데이터, 정칙화, 더 단순한 모델
        
        - 알맞게 적합했을 때:
            → 두 점수가 모두 높다
            → 곡선 사이의 간격이 작다
            → 점수가 수렴한다
        
        인수:
            model: scikit-learn 모델
            X: 특징
            y: 목푯값
            cv: 교차 검증 전략
            scoring: 쓸 지표
            train_sizes: 쓸 학습 집합 크기의 배열
            title: 그림의 제목
        """
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, scoring=scoring,
            train_sizes=train_sizes, n_jobs=-1
        )
        
        # 평균과 표준편차 계산
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # 그래프 그리기
        plt.figure(figsize=(10, 6))
        
        plt.plot(train_sizes, train_mean, label='Training score', 
                color='blue', marker='o')
        plt.fill_between(train_sizes, train_mean - train_std,
                        train_mean + train_std, alpha=0.15, color='blue')
        
        plt.plot(train_sizes, val_mean, label='Validation score',
                color='red', marker='s')
        plt.fill_between(train_sizes, val_mean - val_std,
                        val_mean + val_std, alpha=0.15, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel(f'{scoring.replace("_", " ").title()}')
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # 진단
        final_gap = train_mean[-1] - val_mean[-1]
        
        print("\n" + "=" * 70)
        print("LEARNING CURVE DIAGNOSIS")
        print("=" * 70)
        
        print(f"\nFinal training score: {train_mean[-1]:.4f}")
        print(f"Final validation score: {val_mean[-1]:.4f}")
        print(f"Gap: {final_gap:.4f}")
        
        if final_gap > 0.1:
            print("\n⚠ HIGH VARIANCE (Overfitting)")
            print("  → Large gap between training and validation scores")
            print("  → Consider: More data, regularization, simpler model")
        elif val_mean[-1] < 0.6:
            print("\n⚠ HIGH BIAS (Underfitting)")
            print("  → Both scores are low")
            print("  → Consider: More complex model, more features")
        else:
            print("\n✓ GOOD FIT")
            print("  → Scores are high and close together")
        
        return plt.gcf()
    
    @staticmethod
    def plot_validation_curve(model, X, y, param_name, param_range,
                             cv=5, scoring='accuracy'):
        """
        초매개변수 조율을 위한 검증 곡선 그리기
        
        초매개변수의 값에 따라 모델의 성능이 어떻게 변하는지 보인다
        
        해석:
        - 최적의 초매개변수 값을 찾는 데 도움이 된다
        - 모델이 어디서 과적합하거나 과소적합하기 시작하는지 보인다
        
        인수:
            model: scikit-learn 모델
            X: 특징
            y: 목푯값
            param_name: 바꿀 매개변수의 이름
            param_range: 시도할 매개변수 값의 배열
            cv: 교차 검증 전략
            scoring: 쓸 지표
        """
        train_scores, val_scores = validation_curve(
            model, X, y, param_name=param_name, param_range=param_range,
            cv=cv, scoring=scoring, n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(param_range, train_mean, label='Training score',
                color='blue', marker='o')
        plt.fill_between(param_range, train_mean - train_std,
                        train_mean + train_std, alpha=0.15, color='blue')
        
        plt.plot(param_range, val_mean, label='Validation score',
                color='red', marker='s')
        plt.fill_between(param_range, val_mean - val_std,
                        val_mean + val_std, alpha=0.15, color='red')
        
        plt.xlabel(param_name)
        plt.ylabel(f'{scoring.replace("_", " ").title()}')
        plt.title(f'Validation Curve - {param_name}')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # 최적값 찾기
        optimal_idx = np.argmax(val_mean)
        optimal_value = param_range[optimal_idx]
        optimal_score = val_mean[optimal_idx]
        
        plt.axvline(optimal_value, color='green', linestyle='--', alpha=0.7,
                   label=f'Optimal: {optimal_value}')
        plt.legend(loc='best')
        
        print("\n" + "=" * 70)
        print("VALIDATION CURVE ANALYSIS")
        print("=" * 70)
        
        print(f"\nParameter: {param_name}")
        print(f"Optimal value: {optimal_value}")
        print(f"Validation score at optimal: {optimal_score:.4f}")
        
        return plt.gcf()


def model_selection_strategy_guide():
    """
    모델 선택을 위한 종합 안내
    """
    guide = """
    모델 선택 전략 안내
    ==============================
    
    1단계: 문제 정의하기
    --------------------------
    □ 분류인가 회귀인가?
    □ 어떤 지표가 가장 중요한가? (정확도, 정밀도, 재현율 등)
    □ 오류 종류마다의 비용은 얼마인가?
    □ 해석 가능성이 요구되는가?
    □ 속도나 자원에 제약이 있는가?
    
    2단계: 기준선 모델
    ---------------------
    언제나 간단한 기준선에서 시작하라:
    → 분류: 로지스틱 회귀, 결정 트리
    → 회귀: 선형 회귀, 능선 회귀
    
    왜? 받아들일 만한 최소 성능을 정해 준다
    
    3단계: 여러 모델 시도하기
    ---------------------------
    여러 종류의 모델을 두루 시도하라:
    → 선형: 로지스틱/선형 회귀, SVM
    → 트리 기반: 결정 트리, 랜덤 포리스트, XGBoost
    → 사례 기반: KNN
    → 신경망: 신경망 (데이터 크기가 감당하면)
    
    4단계: 교차 검증
    ------------------------
    교차 검증으로 모델마다 평가하라:
    → 대부분의 경우 K=5나 10
    → 분류에는 층화 K겹
    → 시계열에는 시계열 분할
    
    5단계: 통계적 비교
    ------------------------------
    상위 모델들을 대응 t 검정으로 견주라:
    → 차이가 통계적으로 유의한가?
    → 평균 점수가 가장 높다고 그냥 고르지 마라
    
    6단계: 학습 곡선
    -----------------------
    상위 2~3개 모델의 학습 곡선을 그려 보라:
    → 과적합과 과소적합을 확인한다
    → 데이터를 더 모으면 도움이 될지 가늠한다
    
    7단계: 초매개변수 조율
    -----------------------------
    가장 좋은 모델의 초매개변수를 조율하라:
    → 격자 탐색이나 무작위 탐색을 쓴다
    → 검증 곡선으로 탐색을 이끈다
    → 검증 집합에 과적합하지 않도록 조심하라!
    
    8단계: 최종 평가
    ------------------------
    따로 떼어 둔 시험 집합에서 최종 모델을 평가하라:
    → 시험 집합은 모델 선택 중에 절대 쓰면 안 된다
    → 이래야 성능을 치우침 없이 추정할 수 있다
    
    피해야 할 흔한 함정:
    ========================
    
    ✗ 한 번의 학습-시험 분할로 모델을 고르기
       → 대신 교차 검증을 쓰라
    
    ✗ 모델 선택에 시험 집합을 쓰기
       → 시험 집합은 최종 평가에만 쓴다
    
    ✗ 데이터 유출을 확인하지 않기
       → 특징에 미래의 정보가 들어 있지 않게 하라
    
    ✗ 계산 제약을 무시하기
       → 학습 시간과 예측 시간을 고려하라
    
    ✗ 전체 정확도만 보기
       → 혼동 행렬과 클래스별 성능을 확인하라
    
    ✗ 가정을 확인하지 않기
       → 예를 들어 선형 모델은 관계가 선형이라고 가정한다
    
    ✗ 지나친 조율로 검증 집합에 과적합하기
       → 중첩 교차 검증이 도움이 된다
    
    판단 기준:
    =================
    
    다음이면 더 단순한 모델을 고른다:
    → 성능 차이가 통계적으로 유의하지 않다
    → 해석 가능성이 중요하다
    → 더 빠른 예측이 필요하다
    → 학습 데이터가 적다
    
    다음이면 더 복잡한 모델을 고른다:
    → 성능이 뚜렷이 나아진다
    → 속을 알 수 없어도 괜찮다
    → 계산 자원이 넉넉하다
    → 큰 데이터셋이 있다
    """
    print(guide)


# 사용 예
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    print("=" * 70)
    print("MODEL COMPARISON AND SELECTION DEMONSTRATION")
    print("=" * 70)
    
    # 예시 데이터 생성
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_redundant=5, random_state=42
    )
    
    # 견줄 모델 정의
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42)
    }
    
    # 모델 비교
    comparison = ModelComparison()
    results = comparison.compare_models_cv(models, X, y, cv=5)
    
    # 상위 두 모델의 통계적 비교
    print("\n\n" + "=" * 70)
    print("STATISTICAL COMPARISON OF TOP TWO MODELS")
    print("=" * 70)
    
    sorted_models = sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True)
    top_two = sorted_models[:2]
    
    comparison.paired_ttest(
        top_two[0][1]['scores'],
        top_two[1][1]['scores'],
        top_two[0][0],
        top_two[1][0]
    )
    
    # 전략 안내
    print("\n\n" + "=" * 70)
    print("MODEL SELECTION STRATEGY GUIDE")
    print("=" * 70)
    model_selection_strategy_guide()
    
    print("\n" + "=" * 70)
    print("Note: Run with matplotlib backend to see learning/validation curves")
    print("=" * 70)```

## 2. 논의

서로 다른 모델의 교차 검증 점수는 겹마다 같은 데이터 분할을 쓰므로 대응 표본이다. 따라서 대응 t 검정이 알맞은 통계 검정이다. 이 검정은 겹에 걸친 변동을 고려하여 평균 성능 차이가 0과 유의하게 다른지 가린다.

학습 곡선은 학습 집합의 크기에 따른 성능을 그려 편향과 분산을 진단한다. 학습 점수와 검증 점수가 모두 낮고 서로 가까우면 편향이 큰 것이다(과소적합). 학습 점수는 높은데 검증 점수가 훨씬 낮으면 분산이 큰 것이다(과적합). 데이터를 더 모으는 것은 분산이 큰 경우에만 도움이 된다.

검증 곡선은 초매개변수 하나가 학습 성능과 검증 성능에 어떤 영향을 주는지 보여 준다. 최적의 값은 보통 검증 점수가 가장 높은 곳이며, 학습 점수와 검증 점수의 간격이 그 설정에서의 과적합 정도를 나타낸다.

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

**다룬 것** — 모델 비교

서로 다른 모델의 교차 검증 점수는 겹마다 같은 데이터 분할을 쓰므로 대응 표본이다.

핵심 클래스는 `ModelComparison`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
