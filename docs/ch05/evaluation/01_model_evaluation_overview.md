# 모델 평가 개관

모델 평가는 기계학습 모델이 보지 못한 데이터에 얼마나 잘 일반화되는지를 가늠하는 일이다. 핵심 개념으로는 학습-시험-검증 분할, 편향-분산 절충, 과적합과 과소적합의 구별이 있다. 이 바탕을 이해해야 실전에서 믿을 만한 모델을 만들 수 있다.

## 1. 코드

```python
"""
모델 평가와 지표 - 개관
========================================

이 모듈은 기계학습 모델의 평가와 성능 지표에서 꼭 알아야 할 개념을
훑어본다.

핵심 개념:
- 학습, 시험, 검증 데이터
- 편향-분산 절충
- 과적합과 과소적합
- 교차 검증 전략
"""

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ========================================================================
# 메인
# ========================================================================


class ModelEvaluationOverview:
    """
    모델 평가의 근본 개념을 보인다
    """
    
    @staticmethod
    def train_test_split_example(X, y, test_size=0.2, random_state=42):
        """
        데이터를 학습 집합과 시험 집합으로 나눈다
        
        인수:
            X: 특징
            y: 목표 변수
            test_size: 시험에 쓸 데이터의 비율 (기본값 0.2)
            random_state: 재현성을 위한 난수 씨앗
            
        반환값:
            X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Testing set size: {len(X_test)}")
        print(f"Training/Test ratio: {len(X_train)/len(X_test):.2f}")
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def demonstrate_overfitting():
        """
        간단한 예로 과적합의 개념을 보인다
        """
        # 잡음이 섞인 합성 데이터 생성
        np.random.seed(42)
        X = np.linspace(0, 10, 50)
        y_true = 2 * X + 1
        y_noisy = y_true + np.random.normal(0, 2, 50)
        
        # 단순한 모델 (알맞은 적합)
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        
        # 선형 모델
        linear_model = LinearRegression()
        linear_model.fit(X.reshape(-1, 1), y_noisy)
        
        # 과적합 모델 (차수가 높은 다항식)
        poly_features = PolynomialFeatures(degree=15)
        X_poly = poly_features.fit_transform(X.reshape(-1, 1))
        overfit_model = LinearRegression()
        overfit_model.fit(X_poly, y_noisy)
        
        print("Linear Model (Good Fit) - Coefficients:", linear_model.coef_)
        print("Number of features in overfit model:", X_poly.shape[1])
        
        return {
            'X': X,
            'y_true': y_true,
            'y_noisy': y_noisy,
            'linear_model': linear_model,
            'overfit_model': overfit_model,
            'poly_features': poly_features
        }


def bias_variance_tradeoff_explanation():
    """
    편향-분산 절충을 설명한다
    """
    explanation = """
    편향-분산 절충
    ======================
    
    편향:
    - 학습 알고리즘의 지나치게 단순한 가정에서 오는 오차
    - 편향이 크면 → 과소적합
    - 모델이 특징과 목푯값 사이의 중요한 관계를 놓친다
    
    분산:
    - 학습 집합의 작은 요동에 민감하여 생기는 오차
    - 분산이 크면 → 과적합
    - 모델이 밑에 깔린 양상과 함께 잡음까지 잡아낸다
    
    목표: 편향과 분산을 모두 최소로 하는 알맞은 지점 찾기
    
    전체 오차 = 편향² + 분산 + 줄일 수 없는 오차
    """
    print(explanation)


def holdout_vs_cross_validation():
    """
    홀드아웃 검증과 교차 검증을 견준다
    """
    comparison = """
    홀드아웃 검증과 교차 검증
    =======================================
    
    홀드아웃 검증:
    - 데이터를 한 번 나눈다: 학습/시험 (또는 학습/검증/시험)
    - 장점: 빠르고 간단하다
    - 단점: 성능 추정의 분산이 크고 데이터를 낭비한다
    
    교차 검증:
    - 데이터를 K개 겹으로 나누어 K-1개로 학습하고 1개로 시험하기를 K번 되풀이한다
    - 장점: 데이터를 더 잘 쓰고 성능 추정이 더 믿을 만하다
    - 단점: 계산 비용이 크다 (K배 느리다)
    
    좋은 관행:
    - 데이터셋이 크거나 시간이 부족하면 홀드아웃을 쓴다
    - 데이터셋이 작거나 최종 모델을 평가할 때에는 교차 검증을 쓴다
    """
    print(comparison)


if __name__ == "__main__":
    print("=" * 60)
    print("MODEL EVALUATION AND METRICS - OVERVIEW")
    print("=" * 60)
    
    # 학습-시험 분할 시연
    print("\n1. TRAIN-TEST SPLIT EXAMPLE")
    print("-" * 40)
    X_sample = np.random.randn(1000, 10)
    y_sample = np.random.randint(0, 2, 1000)
    ModelEvaluationOverview.train_test_split_example(X_sample, y_sample)
    
    # 과적합 시연
    print("\n2. OVERFITTING DEMONSTRATION")
    print("-" * 40)
    ModelEvaluationOverview.demonstrate_overfitting()
    
    # 편향-분산 절충 설명
    print("\n3. BIAS-VARIANCE TRADEOFF")
    print("-" * 40)
    bias_variance_tradeoff_explanation()
    
    # 검증 전략 비교
    print("\n4. VALIDATION STRATEGIES")
    print("-" * 40)
    holdout_vs_cross_validation()
```

## 2. 논의

학습-시험 분할은 가장 기본적인 평가 전략이지만, 잡음이 섞였을 수 있는 추정값 하나만 준다. 교차 검증은 여러 분할에 걸쳐 평균을 내어 이를 다루되 계산 비용이 더 든다. 홀드아웃과 교차 검증 중 어느 것을 고를지는 데이터셋의 크기와 쓸 수 있는 계산 자원에 달렸다.

과적합을 알아내려면 학습 오차와 시험 오차를 견주어야 한다. 학습 오차가 시험 오차보다 훨씬 낮다면 모델이 학습 데이터의 잡음까지 외운 것이다. 과적합 시연은 다항 회귀를 쓴다. 15차 다항식은 학습 데이터에 완벽히 들어맞지만 시험 데이터에서 심하게 진동하는 반면, 선형 모델은 참 추세를 잡아낸다.

편향-분산 절충은 일반화를 이해하는 이론적 틀을 준다. 전체 오차는 $\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$으로 분해되며, 모델의 복잡도를 높이면 편향은 줄지만 분산은 는다. 최적의 모델은 두 성분의 균형을 맞춘다.

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

**다룬 것** — 모델 평가 개관

학습-시험 분할은 가장 기본적인 평가 전략이지만, 잡음이 섞였을 수 있는 추정값 하나만 준다.

핵심 클래스는 `ModelEvaluationOverview`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
