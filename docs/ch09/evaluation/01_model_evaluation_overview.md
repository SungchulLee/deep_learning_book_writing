# 모델 평가 개관

모델 평가는 학습에 사용하지 않은 데이터에서 예측 오차를 추정하는 과정이다. 분할의 역할, 지표와 선택 절차를 먼저 정하고 그 추정의 한계를 함께 읽는다.

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
            X: 피처
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
    
    바이어스:
    - 학습 알고리즘의 지나치게 단순한 가정에서 오는 오차
    - 바이어스가 크면 → 과소적합
    - 모델이 피처와 목푯값 사이의 중요한 관계를 놓친다
    
    분산:
    - 학습 집합의 작은 요동에 민감하여 생기는 오차
    - 분산이 크면 → 과적합
    - 모델이 밑에 깔린 양상과 함께 잡음까지 잡아낸다
    
    목표: 편향과 분산을 모두 최소로 하는 알맞은 지점 찾기
    
    전체 오차 = 바이어스² + 분산 + 줄일 수 없는 오차
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

**출력:**

```
============================================================
MODEL EVALUATION AND METRICS - OVERVIEW
============================================================

1. TRAIN-TEST SPLIT EXAMPLE
----------------------------------------
Training set size: 800
Testing set size: 200
Training/Test ratio: 4.00

2. OVERFITTING DEMONSTRATION
----------------------------------------
Linear Model (Good Fit) - Coefficients: [1.8840332]
Number of features in overfit model: 16

3. BIAS-VARIANCE TRADEOFF
----------------------------------------

    편향-분산 절충
    ======================
    
    바이어스:
    - 학습 알고리즘의 지나치게 단순한 가정에서 오는 오차
    - 바이어스가 크면 → 과소적합
    - 모델이 피처와 목푯값 사이의 중요한 관계를 놓친다
    
    분산:
    - 학습 집합의 작은 요동에 민감하여 생기는 오차
    - 분산이 크면 → 과적합
    - 모델이 밑에 깔린 양상과 함께 잡음까지 잡아낸다
    
    목표: 편향과 분산을 모두 최소로 하는 알맞은 지점 찾기
    
    전체 오차 = 바이어스² + 분산 + 줄일 수 없는 오차
    

4. VALIDATION STRATEGIES
----------------------------------------

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
    
```

## 2. 논의

학습셋은 계수를 적합하고 검증셋은 구조·하이퍼파라미터를 선택하며 테스트셋은 선택 후 최종 평가에 사용한다. 같은 테스트 결과로 반복해 설정을 바꾸면 테스트셋도 선택에 사용된다.

학습·검증 성능의 차이는 과적합의 단서이지만 분포 차이, 모드 차이와 표본 수의 영향도 있다. 높은 차수의 다항식이 항상 학습 데이터를 완벽히 맞추거나 선형 모델이 항상 참 관계를 찾는 것은 아니다.

제곱 오차의 편향–분산 분해는 기대값에 관한 결과이다. 복잡도를 늘릴 때 학습된 추정량의 편향과 분산이 반드시 단조롭게 움직이는 것은 아니므로 단일 곡선을 일반 법칙으로 해석하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
학습·검증·테스트의 역할을 구분하라.

</div>

??? success "풀이"
    학습은 계수 적합, 검증은 설정 선택, 테스트는 선택 후 평가에 쓴다. 모든 전처리도 학습 분할 안에서 맞추고 선택 과정에 테스트 정보를 사용하지 않는다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
학습 정확도와 검증 정확도의 차이가 크면 잡음을 암기한 것이 증명되는가?

</div>

??? success "풀이"
    그 차이는 과적합의 단서지만 분포·표본 난이도·모드 차이도 원인일 수 있다. 같은 평가 조건과 분할의 독립성을 먼저 확인한다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
교차 검증에서도 독립 최종 평가가 필요한 경우를 설명하라.

</div>

??? success "풀이"
    교차 검증 점수로 많은 후보를 선택하면 최선 점수에 선택 바이어스가 들어갈 수 있다. 중첩 검증이나 별도 테스트셋으로 선택 절차 전체의 성능을 평가한다.

## 정리하며

평가 설계는 데이터를 나누는 역할에서 시작한다. 학습 곡선은 진단 도구이고, 최종 성능 추정은 선택에 쓰지 않은 데이터에서 수행해야 한다.
